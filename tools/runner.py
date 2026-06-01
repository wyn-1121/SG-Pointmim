import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import numpy as np
from torchvision import transforms
from datasets import data_transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

# === 新增：本地保存 PLY 文件的函数 (用于 test_net) ===
def local_save_ply(filename, points):
    points = points.reshape(-1, 3)
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt='%f %f %f')

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    start_epoch = 0
    best_metrics = Acc_Metric(-float('inf')) 
    metrics = Acc_Metric(-float('inf'))

    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter(['Loss', 'L_pixel', 'L_latent'])

        num_iter = 0

        base_model.train()
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)
            
            out = base_model(points, epoch=epoch)
            
            total_loss = out['loss']
            L_pixel = out['L_pixel']
            L_latent = out['L_latent']
            
            if total_loss.dim() > 0:
                total_loss = total_loss.mean()
                L_pixel = L_pixel.mean()
                L_latent = L_latent.mean()
            
            total_loss.backward()

            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                total_loss = dist_utils.reduce_tensor(total_loss, args)
                L_pixel = dist_utils.reduce_tensor(L_pixel, args)
                L_latent = dist_utils.reduce_tensor(L_latent, args)

            losses.update([total_loss.item()*1000, L_pixel.item()*1000, L_latent.item()*1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', total_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/L_pixel', L_pixel.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/L_latent', L_latent.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Total_Loss', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/L_pixel', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/L_latent', losses.avg(2), epoch)
        
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        
        if epoch % 25 == 0 and epoch > 0:
             builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()
    test_losses = AverageMeter(['Loss', 'L_pixel', 'L_latent'])
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if config.dataset.val._base_.NAME == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, config.dataset.val.others.npoints)
            else:
                points = data.cuda()
            assert points.size(1) == config.dataset.val.others.npoints
            out = base_model(points, epoch=0)
            total_loss = out['loss']
            L_pixel = out['L_pixel']
            L_latent = out['L_latent']
            if args.distributed:
                total_loss = dist_utils.reduce_tensor(total_loss, args)
                L_pixel = dist_utils.reduce_tensor(L_pixel, args)
                L_latent = dist_utils.reduce_tensor(L_latent, args)
            test_losses.update([total_loss.item() * 1000, L_pixel.item() * 1000, L_latent.item() * 1000])
        print_log('[Validation] EPOCH: %d  Val Loss = %.4f (Pixel: %.4f, Latent: %.4f)' % 
                  (epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2)), logger=logger)
        if args.distributed:
            torch.cuda.synchronize()
    if val_writer is not None:
        val_writer.add_scalar('Metric/Val_Loss', test_losses.avg(0), epoch)
        val_writer.add_scalar('Metric/Val_L_pixel', test_losses.avg(1), epoch)
        val_writer.add_scalar('Metric/Val_L_latent', test_losses.avg(2), epoch)
    return Acc_Metric(-test_losses.avg(0))

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('=== DEBUG: Tester start ===', logger=logger)
    
    # 1. Dataset
    print_log('=== DEBUG: Building dataset... ===', logger=logger)
    dataset_ret = builder.dataset_builder(args, config.dataset.test)
    if isinstance(dataset_ret, tuple):
        test_dataloader = dataset_ret[1]
    else:
        test_dataloader = dataset_ret
    
    print_log(f'=== DEBUG: Dataloader length: {len(test_dataloader)} ===', logger=logger)

    # 2. Model
    print_log('=== DEBUG: Building model... ===', logger=logger)
    base_model = builder.model_builder(config.model)
    
    # 3. Load Weights
    print_log(f'=== DEBUG: Loading weights from {args.ckpts}... ===', logger=logger)
    builder.load_model(base_model, args.ckpts, logger=logger) 
    print_log('=== DEBUG: Weights loaded successfully. ===', logger=logger)
    
    # 4. GPU
    if args.use_gpu:
        print_log(f'=== DEBUG: Moving model to GPU (Rank {args.local_rank})... ===', logger=logger)
        base_model.to(args.local_rank)
    
    base_model.eval()
    
    # 5. Path
    vis_path = os.path.join(args.experiment_path, 'vis_result')
    os.makedirs(vis_path, exist_ok=True)
    print_log(f'=== DEBUG: Save path is {vis_path} ===', logger=logger)

    # 6. Loop
    print_log('=== DEBUG: Starting inference loop... ===', logger=logger)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_dataloader):
            print_log(f'=== DEBUG: Processing batch {idx} ===', logger=logger)
            
            if isinstance(batch_data, (list, tuple)):
                points = batch_data[-1].cuda()
            else:
                points = batch_data.cuda()
            
            # 7. Forward
            try:
                ret = base_model(points, vis=True)
            except TypeError as e:
                print_log(f'=== DEBUG: vis=True failed ({e}), trying default... ===', logger=logger)
                ret = base_model(points)
            
            # 8. Get Pred
            pred = None
            if isinstance(ret, tuple):
                pred = ret[1]
            elif isinstance(ret, dict):
                pred = ret.get('recon', ret.get('pred', None))
            else:
                pred = ret
            
            if pred is None:
                print_log('=== DEBUG ERROR: No pred found ===', logger=logger)
                break

            # 9. Save
            input_pc = points[0].detach().cpu().numpy()
            pred_pc = pred[0].detach().cpu().numpy()
            
            input_pc = input_pc.reshape(-1, 3)
            pred_pc = pred_pc.reshape(-1, 3)
            
            local_save_ply(os.path.join(vis_path, f'{idx}_input.ply'), input_pc)
            local_save_ply(os.path.join(vis_path, f'{idx}_pred.ply'), pred_pc)
            
            print_log(f'=== DEBUG: Saved batch {idx} ===', logger=logger)
                
    print_log('=== DEBUG: Test finished. ===', logger=logger)
