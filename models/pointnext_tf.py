import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F

from utils.logging_utils import render_pc_compare_o3d
from models.base_model import BaseModule, ModelRegistry
from models.pointnext import PointNextEncoder, PointNextDecoder, SegHead
from models.temporal_fusion import TemporalDecoder


@ModelRegistry.register("pointnet_tf")
class PointNetTF(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        # Single-frame backbone (reused from pointnet)
        self.encoder = PointNextEncoder()
        self.decoder = PointNextDecoder()
        self.head = SegHead()

        # Temporal fusion on the encoder bottleneck
        tcfg = config.get("temporal", {})
        in_ch = tcfg.get("in_ch", self.encoder.out_channels)
        dim = tcfg.get("dim", in_ch)
        nhead = tcfg.get("nhead", 8)
        depth = tcfg.get("depth", (1, 1, 1, 1))
        G = tcfg.get("G", 4)
        N = tcfg.get("N", 8)
        pool_stride = tcfg.get("pool_stride", 2)
        mlp_ratio = tcfg.get("mlp_ratio", 4)
        dropout = tcfg.get("dropout", 0.0)

        self.temporal = TemporalDecoder(in_ch=in_ch, dim=dim, nhead=nhead, depth=depth, G=G, N=N, pool_stride=pool_stride, mlp_ratio=mlp_ratio, dropout=dropout)

        self.temporal_lambda = tcfg.get("lambda", 0.1)

        pretrained_ckpt = config.get("pretrained_pointnet_ckpt", None)
        if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
            ckpt = torch.load(pretrained_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state, strict=False)
            print("[ALNetTF] loaded pretrained ALNet from", pretrained_ckpt)
            print("  missing:", missing)
            print("  unexpected:", unexpected)

        self.freeze_backbone_first = bool(config.get("freeze_backbone_first", False))
        if self.freeze_backbone_first:
            for name, p in self.named_parameters():
                if "temporal" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.img_log_dir = config.get("img_log_dir", "./logs")
        os.makedirs(f"{self.img_log_dir}/train", exist_ok=True)
        os.makedirs(f"{self.img_log_dir}/val", exist_ok=True)
        self.shuffle = False
        self.pin_memory = False

        self.save_hyperparameters()

    def forward(self, batch):
        pos = batch["pos"]  # [B, T, N, 3] or [T, N, 3]
        x_in = batch["x"]  # [B, T, 4, N] or [T, 4, N]

        B, T, N, _ = pos.shape

        self.temporal.reset_state(device=pos.device)

        preds = []

        for t in range(T):
            pos_t = pos[:, t, :, :].contiguous()  # [B, N, 3]
            x_t = x_in[:, t, :, :].contiguous()  # [B, 4, N]

            data_t = {"pos": pos_t, "x": x_t}
            p_list, f_list = self.encoder.forward_seg_feat(data_t)

            bottleneck = f_list[-1]

            cf_in = bottleneck.unsqueeze(2)  # [B, Cb, 1, N_enc]
            cf_out = self.temporal(cf_in, add_to_recent=True)  # [B, Cb, 1, N_enc]
            cf_out = cf_out.squeeze(2)  # [B, Cb, N_enc]

            f_list[-1] = cf_out
            f_dec = self.decoder(p_list, f_list).squeeze(-1)  # [B, C_dec, N]
            out_t = self.head(f_dec)  # [B, 3, N]
            out_t = torch.tanh(out_t)  # [-1, 1]
            preds.append(out_t.unsqueeze(1))  # [B, 1, 3, N]

        pred = torch.cat(preds, dim=1)  # [B, T, 3, N]
        return pred

    @staticmethod
    def masked_L1(pred, gt, mask=None):
        if mask is not None:
            diff = (pred - gt).abs()  # [B,3,N]
            diff = diff * mask.unsqueeze(1)
            return diff.sum() / (mask.sum() * pred.size(1) + 1e-8)
        else:
            return F.l1_loss(pred, gt)

    @staticmethod
    def grad_loss_proj_1d(pred, gt, pos, mask, n_dir = 3):
        B, C, N = pred.shape
        device = pred.device
        base_dirs = torch.tensor([[1.0, 0.7, 0.3], [0.3, 1.0, 0.7], [0.7, 0.3, 1.0]], device=device, dtype=pos.dtype)
        base_dirs = base_dirs[:n_dir]
        base_dirs = base_dirs / (base_dirs.norm(dim=-1, keepdim=True) + 1e-8)
        loss = 0.0
        for d in base_dirs:  # d: [3]
            s = (pos * d).sum(-1)  # [B,N]
            idx = s.argsort(dim=-1)  # [B,N]
            idx_exp = idx.unsqueeze(1).expand(-1, C, -1)
            pred_s = torch.gather(pred, 2, idx_exp)  # [B,3,N]
            gt_s = torch.gather(gt, 2, idx_exp)  # [B,3,N]
            mask_s = torch.gather(mask, 1, idx)  # [B,N]
            dp = pred_s[..., 1:] - pred_s[..., :-1]  # [B,3,N-1]
            dg = gt_s[..., 1:] - gt_s[..., :-1]  # [B,3,N-1]
            valid = mask_s[..., 1:] & mask_s[..., :-1]  # [B,N-1]
            diff = (dp - dg).pow(2).sum(1)  # [B,N-1]
            diff = diff * valid
            loss += diff.sum() / (valid.sum() + 1e-8)
        return loss / float(len(base_dirs))

    @staticmethod
    def temporal_stats_loss(pred_seq, mask_seq):
        w = mask_seq.float()  # [B, T, N]
        denom = w.sum(-1, keepdim=True).clamp_min(1.0)  # [B, T, 1]

        mean = (pred_seq * w.unsqueeze(2)).sum(-1) / denom  # [B, T, 3]
        sq = (pred_seq ** 2) * w.unsqueeze(2)
        var = (sq.sum(-1) / denom) - mean ** 2  # [B, T, 3]

        mean_diff = (mean[:, 1:] - mean[:, :-1]).abs().mean()
        var_diff = (var[:, 1:] - var[:, :-1]).abs().mean()

        loss = mean_diff + var_diff
        return {"loss": loss, "mean_diff": mean_diff, "var_diff": var_diff}

    def log_pcl_seq(self, epoch, seq, seq_pred, stage, batch_idx):
        pcl_grids = []
        poses = seq['pos'][0].detach().cpu().numpy()
        xs = seq['x'][0, :, :-1].permute(0, 2, 1).detach().cpu().numpy()
        ys = seq['y'][0].permute(0, 2, 1).detach().cpu().numpy()
        preds = seq_pred[0].permute(0, 2, 1).detach().cpu().numpy()
        xs = np.clip(xs * 127.5 + 127.5, 0, 255).astype(np.uint8)
        ys = np.clip(ys * 127.5 + 127.5, 0, 255).astype(np.uint8)
        preds = np.clip(preds * 127.5 + 127.5, 0, 255).astype(np.uint8)
        for pos, color_input, color_gt, color_pred in zip(poses, xs, ys, preds):
            grid = render_pc_compare_o3d(save_path=None, pos=pos, input_color=color_input, pred_color=color_pred, gt_color=color_gt, is_save=False)
            pcl_grids.append(grid)
        scene_name = seq['names'][0][0].split('_')[1]
        start_frame = seq['names'][0][0].split('_')[2]
        end_frame = seq['names'][-1][0].split('_')[2]
        pcl_name = f"{scene_name}_{start_frame}_{end_frame}"
        save_path = f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{pcl_name}.jpg'
        grid_save = np.concatenate(pcl_grids, axis=0)
        cv.imwrite(save_path, grid_save[..., ::-1])
