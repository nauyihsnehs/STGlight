import os
import random
import sys

from torchvision.models import ResNet18_Weights

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2 as cv

from models.base_model import BaseModule, ModelRegistry

TINY_NUMBER = 1e-8
exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION,
                   cv.IMWRITE_EXR_COMPRESSION_PIZ]


class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU(inplace=True)
        self.fc = nn.Linear(dim, dim)
        self.do = nn.Dropout(p)

    def forward(self, x):
        y = self.fc(self.do(self.act(self.norm(x))))
        return x + y


class RegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, sg_num, use_lambda=True, p=0.1, depth=4):
        super().__init__()
        self.sg_num = sg_num
        self.use_lambda = use_lambda

        self.proj = nn.Sequential(
            nn.Flatten(),  # [B, C, 1, 1] -> [B, C]
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(hidden_dim, p=p) for _ in range(depth)])

        self.head_color = nn.Linear(hidden_dim, sg_num * 3)
        self.head_scale = nn.Linear(hidden_dim, sg_num * 1)
        if use_lambda:
            self.head_la = nn.Linear(hidden_dim, sg_num * 1)

        for m in [self.head_color, self.head_scale] + ([self.head_la] if use_lambda else []):
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, feats):  # [B, C, 1, 1]
        x = self.trunk(self.proj(feats))  # [B, H]
        color = self.head_color(x).view(-1, self.sg_num, 3)
        scale = self.head_scale(x).view(-1, self.sg_num, 1)
        if self.use_lambda:
            la = self.head_la(x).view(-1, self.sg_num, 1)
            params = torch.cat([color, scale, la], dim=-1)  # [B, sg_num, 5]
        else:
            params = torch.cat([color, scale], dim=-1)  # [B, sg_num, 4]
        return params


@ModelRegistry.register("alnet")
class ALNet(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.encoder_type = config.get("encoder", "resnet18")
        self.input_channels = config.get("input_channels", 3)
        self.sg_num = sg_num = config.get("sg_num", 16)
        self.sg_param_num = config.get("sg_param_num", 5)
        # 4 (rgb+scale) 5 (rgb+scale+lambda)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.img_log_dir = config.get('img_log_dir', './logs')
        os.makedirs(f'{self.img_log_dir}/train', exist_ok=True)
        os.makedirs(f'{self.img_log_dir}/val', exist_ok=True)

        self.la_num = la_num = config.get("la_num", 20)
        W = config.get('W', 256)
        H = W // 2
        self.H, self.W = H, W
        self.dots_sh = [H, W]

        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(0., 2 * np.pi, W)], indexing="ij")
        view_dirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                                dim=-1).unsqueeze(-2).unsqueeze(0)

        lgtSGLobes = torch.from_numpy(np.load(f'utils/asg_fib_lobes_{sg_num}.npy') + TINY_NUMBER)  # .cuda()
        sg_lobe = lgtSGLobes.view(1, 1, 1, sg_num, 3).repeat(1, H, W, 1, 1)
        lgtSGLambdas = torch.full((sg_num, 1), la_num)
        sg_la = lgtSGLambdas.view(1, 1, 1, sg_num, 1).repeat(1, H, W, 1, 1)

        self.register_buffer('view_dirs', view_dirs)
        self.register_buffer('sg_lobe', sg_lobe)
        self.register_buffer('sg_la', sg_la)

        is_pretrained = config.get("pretrained", True)
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT) if is_pretrained else models.resnet18(
            weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_channels = 512

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.regression_head = RegressionHead(
            in_dim=self.feat_channels,
            hidden_dim=self.hidden_dim,
            sg_num=self.sg_num,
            use_lambda=(self.sg_param_num == 5),
            p=0.1,
            depth=4
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def sg_rendering(self, sg_params, la=None, sg_lobe=None):
        color = torch.sigmoid(sg_params[..., :3])
        scale = nn.Softplus()(sg_params[..., 3:4])
        la = torch.sigmoid(sg_params[..., -1:]) * 100 if self.sg_param_num == 5 else None
        color = color[:, None, None, ...].repeat(1, *self.dots_sh, 1, 1)
        scale = scale[:, None, None, ...].repeat(1, *self.dots_sh, 1, 1)
        la = la[:, None, None, ...].repeat(1, *self.dots_sh, 1, 1) if self.sg_param_num == 5 else 1 / self.sg_la
        sg_lobe = sg_lobe if sg_lobe is not None else self.sg_lobe
        rgb = scale * color * torch.exp(la * (torch.sum(self.view_dirs * sg_lobe, dim=-1, keepdim=True) - 1.))
        envmap = torch.sum(rgb, dim=-2).permute(0, 3, 1, 2).float()
        return envmap

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)  # [b, feat_channels, h', w']

        pooled_features = self.pooling(features)  # [b, feat_channels, 1, 1]

        output = self.regression_head(pooled_features)  # [b, sg_num * sg_param_num]

        batch_size = x.size(0)
        pred_sg = output.view(batch_size, self.sg_num, self.sg_param_num)
        scale_raw = pred_sg[..., 3:4]
        scale_raw = scale_raw * 1.8 - scale_raw.detach() * 0.8
        pred_sg[..., 3:4] = scale_raw
        pred_pano = self.sg_rendering(pred_sg)
        return pred_sg, pred_pano

    def log_image(self, epoch, idx_max, env_pre, env_gt, ldr_input, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        env_save = torch.expm1(env_pre[save_idx]).permute(1, 2, 0).float().detach().cpu().numpy()
        env_gt_save = torch.expm1(env_gt[save_idx]).permute(1, 2, 0).float().detach().cpu().numpy()
        ldr_input_save = ldr_input[save_idx].permute(1, 2, 0).float().detach().cpu().numpy()
        grid = np.concatenate((env_save, env_gt_save, ldr_input_save), axis=0)
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b_{batch_idx}_{img_name[save_idx]}_grid.exr',
                   grid, exr_save_params)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        hdr_pano, ldr_pano, partial_mask, img_name = batch
        B, C, H, W = hdr_pano.shape
        s = random.gauss(1.0, 0.25)
        s = max(0.5, min(2.0, s))
        hdr_pano = hdr_pano * s
        ldr_pano = ldr_pano * s

        k_img = random.randint(0, 9)  # for hdr/ldr
        k_mask = random.randint(0, 9)  # for mask
        shift_img = int(round(W * k_img / 10.0))
        shift_mask = int(round(W * k_mask / 10.0))

        if shift_img != 0:
            hdr_pano = torch.roll(hdr_pano, shifts=shift_img, dims=-1)
            ldr_pano = torch.roll(ldr_pano, shifts=shift_img, dims=-1)
        if shift_mask != 0:
            partial_mask = torch.roll(partial_mask, shifts=shift_mask, dims=-1)

        if random.random() < 0.5:
            hdr_pano = torch.flip(hdr_pano, dims=[3])
            ldr_pano = torch.flip(ldr_pano, dims=[3])

        if random.random() < 0.5:
            partial_mask = torch.flip(partial_mask, dims=[3])

        gt_pano = torch.log1p(hdr_pano.clamp_min(0.0)).clamp_max(5.0)
        input_pano = ldr_pano * partial_mask
        return gt_pano, input_pano, img_name
