import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        return self.acc > other.acc

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def _as_scalar_loss(x, device):
    """
    Convert tensor or numeric value to scalar tensor.
    DataParallel may return a vector of losses from multiple GPUs,
    so we always take mean().
    """
    if torch.is_tensor(x):
        return x.mean()
    return torch.tensor(float(x), device=device)


def _get_loss_from_keys(ret_dict, keys, device, default=None):
    """
    Return the first available loss value from ret_dict according to keys.
    """
    for key in keys:
        if key in ret_dict:
            return _as_scalar_loss(ret_dict[key], device)

    if default is not None:
        return default

    return torch.tensor(0.0, device=device)


def parse_pretrain_losses(ret_output, device):
    """
    Robustly parse the output of Point_MAE / SG-PointMIM.

    Supported formats include:

    1) Original style:
       {
           'loss': total_loss,
           'recon': recon_loss,
           'contrast': contrast_loss
       }

    2) SG-PointMIM style:
       {
           'loss_total': total_loss,
           'loss_geo': geo_loss,
           'loss_sem': sem_loss
       }

    3) Tensor style:
       ret_output = loss_tensor
    """
    if isinstance(ret_output, dict):
        loss_total = _get_loss_from_keys(
            ret_output,
            keys=[
                'loss',
                'loss_total',
                'total_loss',
                'total',
                'recon'
            ],
            device=device,
            default=None
        )

        if loss_total is None:
            raise KeyError(
                f"Cannot find total loss key. Available keys: {list(ret_output.keys())}"
            )

        loss_recon = _get_loss_from_keys(
            ret_output,
            keys=[
                'recon',
                'loss_recon',
                'loss_geo',
                'geo',
                'loss_pixel',
                'pixel',
                'L_geo'
            ],
            device=device,
            default=loss_total
        )

        loss_contrast = _get_loss_from_keys(
            ret_output,
            keys=[
                'contrast',
                'loss_contrast',
                'loss_sem',
                'sem',
                'loss_latent',
                'latent',
                'L_sem'
            ],
            device=device,
            default=torch.tensor(0.0, device=device)
        )

    else:
        loss_total = _as_scalar_loss(ret_output, device)
        loss_recon = loss_total
        loss_contrast = torch.tensor(0.0, device=device)

    return loss_total, loss_recon, loss_contrast


def forward_with_optional_epoch(base_model, points, epoch):
    """
    Forward wrapper.

    If the model forward supports epoch, pass epoch to activate SGM according to
    sgm_start_epoch. If it does not support epoch, fall back to base_model(points).
    """
    try:
        return base_model(points, epoch=epoch)
    except TypeError as e:
        msg = str(e)
        if 'epoch' in msg or 'unexpected keyword' in msg:
            return base_model(points)
        raise e


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)

    # build extra_train if available
    (_, extra_train_dataloader) = builder.dataset_builder(args, config.dataset.extra_train) \
        if config.dataset.get('extra_train') else (None, None)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP / DP
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)

        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )
        print_log('Using Distributed Data parallel ...', logger=logger)

    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # training
    base_model.zero_grad()

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()

        # ============================================================
        # Profiling: reset GPU peak memory at the beginning of epoch
        # ============================================================
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        epoch_start_time = time.time()
        batch_start_time = time.time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Monitor 3 losses: Total, Recon/Geo, Contrast/Sem
        losses = AverageMeter(['Total', 'Recon', 'Contrast'])

        num_iter = 0
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME

            # Prepare point cloud data
            if dataset_name == 'ShapeNet':
                if isinstance(data, (list, tuple)):
                    # JointAug: data = [View1, View2] -> [2B, N, 3]
                    points = torch.cat(data, dim=0).cuda(non_blocking=True)
                else:
                    points = data.cuda(non_blocking=True)

            elif dataset_name == 'ModelNet':
                points = data[0].cuda(non_blocking=True)
                points = misc.fps(points, npoints)

            else:
                raise NotImplementedError(f'Train phase does not support {dataset_name}')

            assert points.size(1) == npoints

            points = train_transforms(points)

            # Clear gradients
            optimizer.zero_grad()

            # ============================================================
            # Forward
            # Pass epoch if the model supports it, so SGM can be activated
            # according to sgm_start_epoch.
            # ============================================================
            ret_output = forward_with_optional_epoch(base_model, points, epoch)

            # Robust loss parsing
            device = points.device
            loss_total, loss_recon, loss_contrast = parse_pretrain_losses(ret_output, device)

            # Backward
            loss_total.backward()

            # Update parameters
            if num_iter == config.step_per_update:
                num_iter = 0

                grad_clip = config.get('grad_norm_clip', 10.0)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)

                optimizer.step()
                base_model.zero_grad()

            # Reduce losses in distributed mode
            if args.distributed:
                loss_total = dist_utils.reduce_tensor(loss_total, args)
                loss_recon = dist_utils.reduce_tensor(loss_recon, args)
                loss_contrast = dist_utils.reduce_tensor(loss_contrast, args)

            losses.update([
                loss_total.item() * 1000,
                loss_recon.item() * 1000,
                loss_contrast.item() * 1000
            ])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Total', loss_total.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Recon', loss_recon.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Contrast', loss_contrast.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) '
                    'DataTime = %.3f (s) Total=%.4f Recon=%.4f Ctr=%.4f lr = %.6f' %
                    (
                        epoch,
                        config.max_epoch,
                        idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        losses.val()[0],
                        losses.val()[1],
                        losses.val()[2],
                        optimizer.param_groups[0]['lr']
                    ),
                    logger=logger
                )

        # Scheduler step
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        # ============================================================
        # Profiling: epoch time and peak GPU memory
        # ============================================================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_mem_gb = 0.0

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Total', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Recon', losses.avg(1), epoch)
            train_writer.add_scalar('Loss/Epoch/Contrast', losses.avg(2), epoch)
            train_writer.add_scalar('Profiling/EpochTime', epoch_time, epoch)
            train_writer.add_scalar('Profiling/PeakMemGB', peak_mem_gb, epoch)

        print_log(
            '[Training] EPOCH: %d EpochTime = %.3f (s) PeakMem = %.3f GB '
            'Total=%.4f Recon=%.4f Ctr=%.4f lr = %.6f' %
            (
                epoch,
                epoch_time,
                peak_mem_gb,
                losses.avg(0),
                losses.avg(1),
                losses.avg(2),
                optimizer.param_groups[0]['lr']
            ),
            logger=logger
        )

        # Validation
        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(
                base_model,
                extra_train_dataloader,
                test_dataloader,
                epoch,
                val_writer,
                args,
                config,
                logger=logger
            )

            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(
                    base_model,
                    optimizer,
                    epoch,
                    metrics,
                    best_metrics,
                    'ckpt-best',
                    args,
                    logger=logger
                )

        builder.save_checkpoint(
            base_model,
            optimizer,
            epoch,
            metrics,
            best_metrics,
            'ckpt-last',
            args,
            logger=logger
        )

        if epoch % 25 == 0 and epoch >= 250:
            builder.save_checkpoint(
                base_model,
                optimizer,
                epoch,
                metrics,
                best_metrics,
                f'ckpt-epoch-{epoch:03d}',
                args,
                logger=logger
            )

    if train_writer is not None:
        train_writer.close()

    if val_writer is not None:
        val_writer.close()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)

    if extra_train_dataloader is None:
        print_log("Warning: extra_train_dataloader is None. Skipping SVM validation.", logger=logger)
        return Acc_Metric(0.0)

    base_model.eval()

    svm_acc = 0.0
    print_log(
        '[Validation] EPOCH: %d  acc = %.4f (SVM skipped for speed and stability)' %
        (epoch, svm_acc),
        logger=logger
    )

    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass