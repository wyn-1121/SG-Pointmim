import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2



@MMODELS.register_module()
class sg_pointmim(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        
        self.encoder_dims = config.transformer_config.encoder_dims
        self.MAE_target_encoder = Encoder(encoder_channel=self.encoder_dims)
        self.momentum = 0.996 


        self.loss_alpha = config.get('loss_alpha', 1.0) 
        self.loss_beta = config.get('loss_beta', 20.0)
        self.sgm_hint_ratio = config.transformer_config.get('sgm_hint_ratio', 0.05)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        

        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.MAE_latent_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        self.latent_increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.encoder_dims, 1) 
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)
        
        self.latent_loss_func = nn.MSELoss()
        self.last_logged_epoch = -1

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            
    def update_target_encoder(self):
        with torch.no_grad():
            for param_q, param_k in zip(self.MAE_encoder.encoder.parameters(), self.MAE_target_encoder.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @staticmethod
    def simple_ncut_proxy(x_full):
        B, G, C = x_full.shape
        device = x_full.device
        
        # 1. Normalize Features
        x = x_full.float() 
        x_norm = F.normalize(x, p=2, dim=-1) # B G C
        
        if torch.isnan(x_norm).any():
            return torch.zeros(B, G, dtype=torch.bool, device=device)

        # 2. Similarity Matrix
        similarity_matrix = x_norm @ x_norm.transpose(1, 2)
        
        # 3. Laplacian
        D = torch.sum(similarity_matrix, dim=1).clamp(min=1e-8)
        D_inv_sqrt_vec = 1.0 / D.sqrt()
        term = D_inv_sqrt_vec.unsqueeze(2) * similarity_matrix * D_inv_sqrt_vec.unsqueeze(1)
        normalized_laplacian = torch.eye(G, device=device).unsqueeze(0) - term

        # 4. Eigen Decomposition
        try:
            _, vecs = torch.linalg.eigh(normalized_laplacian)
            fiedler_vec = vecs[:, :, 1] 
            
            # 5. Partition
            medians = torch.median(fiedler_vec, dim=1, keepdim=True).values
            clusters = fiedler_vec > medians
            return clusters

        except RuntimeError:
            return torch.zeros(B, G, dtype=torch.bool, device=device)

    def generate_self_guided_mask(self, x_full, mask_ratio, num_group):
        B, G, C = x_full.shape
        device = x_full.device
        
        # 1. Clustering
        clusters = self.simple_ncut_proxy(x_full)
        
        # Fallback to Random
        if clusters.sum() == 0: 
            rand_mask = torch.zeros(B, G, dtype=torch.bool, device=device)
            num_mask = int(mask_ratio * num_group)
            for i in range(B):
                rand_indices = torch.randperm(G, device=device)[:num_mask]
                rand_mask[i, rand_indices] = True
            return rand_mask
        
        # 2. Identify Target Cluster
        target_mask = clusters
        count = clusters.sum(dim=1, keepdim=True) 
        flip_indices = count > (G // 2)
        target_mask = torch.where(flip_indices, ~clusters, clusters)

        # 3. Relevance Scoring
        x_norm = F.normalize(x_full, p=2, dim=-1)
        num_mask = int(mask_ratio * num_group)
        
        target_features = x_full * target_mask.unsqueeze(-1)
        sum_features = target_features.sum(dim=1)
        count_features = target_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_target = sum_features / count_features
        mean_target_norm = F.normalize(mean_target, p=2, dim=-1)
        
        relevance_scores = torch.bmm(x_norm, mean_target_norm.unsqueeze(-1)).squeeze(-1)
        
        # 4. Masking Top-K Relevant
        _, topk_indices = torch.topk(relevance_scores, k=num_mask, dim=1, largest=True)
        bool_masked_pos = torch.zeros(B, G, dtype=torch.bool, device=device)
        bool_masked_pos.scatter_(1, topk_indices, True)
        
        # 5. Hint Tokens (Configurable)
        hint_ratio = self.sgm_hint_ratio 
        num_hints = int(num_mask * hint_ratio)
        if num_hints > 0:
            rand_hint = torch.rand(B, num_mask, device=device)
            _, hint_indices_local = torch.topk(rand_hint, k=num_hints, dim=1)
            hint_indices_global = torch.gather(topk_indices, 1, hint_indices_local)
            bool_masked_pos.scatter_(1, hint_indices_global, False)

        return bool_masked_pos

    def forward(self, pts, vis = False, epoch = 0, **kwargs):
        neighborhood, center = self.group_divider(pts)

        # 1. Get Teacher Features
        with torch.no_grad():
            T_full = self.MAE_target_encoder(neighborhood) # B G C

        # 2. SGM Logic
        use_sgm_config = self.config.transformer_config.get('use_sgm', False)
        sgm_start_epoch = self.config.transformer_config.get('sgm_start_epoch', 100)
        
        use_sg_mask = self.training and use_sgm_config and (epoch >= sgm_start_epoch)
        sg_mask = None
        
        if use_sg_mask:
            if self.last_logged_epoch != epoch:
                if epoch == sgm_start_epoch:
                    print_log(f'\n[SGM] Warmup finished! SGM ACTIVATED at Epoch {epoch}', logger='Point_MAE')
                self.last_logged_epoch = epoch

            sg_mask = self.generate_self_guided_mask(
                T_full, 
                self.config.transformer_config.mask_ratio,
                self.num_group
            )

        # 3. Mask & Encode
        x_vis, mask = self.MAE_encoder(neighborhood, center, external_mask=sg_mask) 
        
        B, C = x_vis.shape[0], x_vis.shape[-1]
        M = mask.sum(dim=1).max().item()
        
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        mask_token = self.mask_token.expand(B, M, -1)
        
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        
        # 4. Latent Target with LayerNorm
        T_mask_raw = T_full[mask].reshape(B, -1, self.encoder_dims)
        T_mask = F.layer_norm(T_mask_raw, (self.encoder_dims,))

        # 5. Geometric Reconstruction & Loss
        x_rec_pixel = self.MAE_decoder(x_full, pos_full, M)
        rebuild_points = self.increase_dim(x_rec_pixel.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        L_pixel = self.loss_func(rebuild_points, gt_points)

        # 6. Semantic Alignment & Loss
        x_rec_latent = self.MAE_latent_decoder(x_full, pos_full, M)
        pred_latent_mask = self.latent_increase_dim(x_rec_latent.transpose(1, 2)).transpose(1, 2)
        L_latent = self.latent_loss_func(pred_latent_mask, T_mask)

        # 7. Weighted Total Loss
        total_loss = self.loss_alpha * L_latent + self.loss_beta * L_pixel

        if self.training:
            self.update_target_encoder()

        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            
            rebuild_points_reshaped = rebuild_points.reshape(B, M, self.group_size, 3)
            masked_centers = center[mask].reshape(B, M, 3).unsqueeze(2)
            full_rebuild = rebuild_points_reshaped + masked_centers
            
            full = torch.cat([full_vis.reshape(B, -1, 3), full_rebuild.reshape(B, -1, 3)], dim=1)
            full_center = torch.cat([center[mask].reshape(B, -1, 3), center[~mask].reshape(B, -1, 3)], dim=1)
            
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center
        else:
            return {
                'loss': total_loss,
                'L_pixel': L_pixel,
                'L_latent': L_latent,
            }

@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        self.build_loss_func()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(get_missing_parameters_message(incompatible.missing_keys), logger='Transformer')
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='Transformer')
            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
