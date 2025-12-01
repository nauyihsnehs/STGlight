import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn

from models.base_model import ModelRegistry
from models.vlsnet import VLSNet
from models.temporal_fusion import TemporalDecoder

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]


@ModelRegistry.register("vlsnet_tf")
class VLSNetTF(VLSNet):
    def __init__(self, config):
        super().__init__(config)

        td_cfg = config.get("temporal_decoder", {})
        td_dim = td_cfg.get("dim", getattr(self, "feat_channels", 512))

        self.temporal_decoder = TemporalDecoder(
            in_ch=getattr(self, "feat_channels", 512),
            dim=td_dim,
            nhead=td_cfg.get("nhead", 8),
            depth=td_cfg.get("depth", (1, 1, 1, 1)),
            G=td_cfg.get("G", 4),
            N=td_cfg.get("N", 8),
            pool_stride=td_cfg.get("pool_stride", 2),
            mlp_ratio=td_cfg.get("mlp_ratio", 4),
            dropout=td_cfg.get("dropout", 0.0),
        )

        if td_dim != getattr(self, "feat_channels", 512):
            modules = list(self.spatial_encoder.children())
            if isinstance(modules[-1], nn.Linear):
                out_features = modules[-1].out_features
                modules[-1] = nn.Linear(td_dim, out_features)
                self.spatial_encoder = nn.Sequential(*modules)

        self.temporal_loss_weight = config.get("temporal_loss_weight", 0.0)

        pretrained_ckpt = config.get("pretrained_ckpt", None)
        if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
            ckpt = torch.load(pretrained_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state, strict=False)
            print("[VLSNetTF] loaded pretrained VLSNet from", pretrained_ckpt)
            print("  missing:", missing)
            print("  unexpected:", unexpected)
            for m in self.temporal_decoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                if isinstance(m, nn.MultiheadAttention):
                    if m.in_proj_weight is not None:
                        nn.init.zeros_(m.in_proj_weight)
                    if m.in_proj_bias is not None:
                        nn.init.zeros_(m.in_proj_bias)
                    nn.init.zeros_(m.out_proj.weight)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)

        self.freeze_backbone_first = bool(config.get("freeze_backbone_first", False))
        if self.freeze_backbone_first:
            for name, p in self.named_parameters():
                if "temporal_decoder" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    @torch.no_grad()
    def reset_temporal_state(self, device=None, G=None):
        dev = device if device is not None else getattr(self, "device", None)
        self.temporal_decoder.reset_state(device=dev, G=G)

    def forward(self, x, init_pos=None, init_color=None, add_to_recent=False):
        if init_pos is None or init_color is None:
            raise ValueError("VLSNetTF.forward requires init_pos and init_color.")

        query_points = self.get_query_gaussian(init_pos, init_color)  # [B, N, gs_dim]
        features = self.encoder(x)  # [B, C, H', W']

        fused_features = self.temporal_decoder(features, add_to_recent=add_to_recent)

        features_tf = features + fused_features
        global_features = self.spatial_encoder(features_tf)  # [B, feature_dim]

        batch_size, num_points, _ = query_points.size()
        expanded_features = global_features.unsqueeze(1).expand(-1, num_points, -1)
        combined = torch.cat([expanded_features, query_points], dim=-1)
        output = self.mlp_head(combined)  # [B, N, gs_dim]
        return output

    def log_images(self, epoch, env_pre, env_gt, ldr_input, img_name, stage, batch_idx):
        env_save = env_pre.permute(2, 0, 3, 1).reshape(env_pre.size(2), -1, env_pre.size(1)).float().detach().cpu().numpy()
        env_gt_save = torch.expm1(env_gt).permute(2, 0, 3, 1).reshape(env_gt.size(2), -1, env_gt.size(1)).float().detach().cpu().numpy()
        ldr_input = ldr_input.permute(2, 0, 3, 1).reshape(ldr_input.size(2), -1, ldr_input.size(1)).float().detach().cpu().numpy() / 2 + 0.5
        grid = np.concatenate((env_save, env_gt_save, ldr_input), axis=0)[..., ::-1]
        scene_name = img_name[0].split('_')[0]
        start_name = img_name[0].split('_')[-1]
        end_name = img_name[-1].split('_')[-1]
        save_name = f'{scene_name}-{start_name}-{end_name}'
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b_{batch_idx}_{save_name}_grid.exr', grid, exr_save_params)
