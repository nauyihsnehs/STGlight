import math
import os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np

from datasets.vlsnet_dataset import visualize_image_coords, max_pooling

import torch
import torch.nn as nn
import cv2 as cv
from torchvision.models import resnet18

from models.base_model import BaseModule, ModelRegistry
from tdgs.gaussian_splatting import VLSRender

TINY_NUMBER = 1e-8
exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION,
                   cv.IMWRITE_EXR_COMPRESSION_PIZ]


def get_cameras(cx, cy, fx=853.33 / 4, fy=853.33 / 4, device=None):
    intrins = torch.tensor([[fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]]).float()
    viewmat = torch.tensor([[1, 0, 0, 0.0],
                            [0, 1, 0, 0.0],
                            [0, 0, 1, 0.0],
                            [0., 0., 0., 1]]).float()
    return (intrins, viewmat) if device is None else (intrins.to(device), viewmat.to(device))


@ModelRegistry.register("vlsnet")
class VLSNet(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.encoder_type = config.get("encoder", "resnet18")
        self.gs_dim = config.get("gs_dim", 13)
        self.feature_dim = config.get("feature_dim", 512)
        self.input_channels = config.get("input_channels", 3)
        self.img_log_dir = config.get('img_log_dir', './logs')
        os.makedirs(f'{self.img_log_dir}/train', exist_ok=True)
        os.makedirs(f'{self.img_log_dir}/val', exist_ok=True)

        self.H, self.W = config.get("h", 240), config.get("w", 320)
        intrins, viewmat = get_cameras(self.W / 2, self.H / 2)
        fx, cx, cy = intrins[0, 0], intrins[0, 2], intrins[1, 2]
        pixel_scale_x = cx / fx
        pixel_scale_y = cy / fx

        self.register_buffer('pixel_scale_x', pixel_scale_x)
        self.register_buffer('pixel_scale_y', pixel_scale_y)
        self.register_buffer('intrins', intrins)
        self.register_buffer('viewmat', viewmat)

        self.vls_render = VLSRender(intrins, viewmat, device=self.device)

        backbone = resnet18(pretrained=config.get("pretrained", False))
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_channels = 512

        self.spatial_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_channels, self.feature_dim)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(self.feature_dim + self.gs_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.gs_dim)
        )

    def update_intrinsics(self, fx, fy, cx, cy):
        self.pixel_scale_x = self.pixel_scale_x * 0 + cx / fx
        self.pixel_scale_y = self.pixel_scale_x * 0 + cy / fy
        intrins, viewmat = get_cameras(cx, cy, fx, fy, self.device)
        self.intrins = intrins
        self.viewmat = viewmat

    def get_query_gaussian(self, init_pos, init_color):
        device = init_pos.device
        bs, num_gaussians, _ = init_pos.shape
        scales = torch.ones(bs, num_gaussians, 1, device=device) * 0.01
        hdr_scales = torch.ones(bs, num_gaussians, 1, device=device) * 10
        query_gaussian = torch.cat([init_pos, init_color, hdr_scales, scales], dim=-1)
        return query_gaussian

    def forward(self, x, init_pos=None, init_color=None) -> torch.Tensor:
        query_points = self.get_query_gaussian(init_pos, init_color)
        features = self.encoder(x)
        global_features = self.spatial_encoder(features)
        batch_size, num_points, _ = query_points.size()
        expanded_features = global_features.unsqueeze(1).expand(-1, num_points, -1)
        combined = torch.cat([expanded_features, query_points], dim=-1)
        output = self.mlp_head(combined)
        return output

    @staticmethod
    def get_loss(pred_lin, gt_log, pred_pos, init_pos, w_lin=0.25, w_log=1.0):
        gt_lin = torch.expm1(gt_log)
        pred_log = torch.log1p(torch.clamp_min(pred_lin, 0))
        lum = gt_lin.mean(dim=-1, keepdim=True)
        mask = (lum > 1e-6).float()
        pp_loss = w_log * nn.MSELoss()(pred_log, gt_log) + w_lin * nn.MSELoss()(pred_lin * mask, gt_lin * mask)

        if pred_pos is not None:
            pos_loss = nn.MSELoss()(pred_pos, init_pos)
            return pp_loss + pos_loss
        else:
            return pp_loss

    def get_position(self, pred_means3D, depth, init_pos=None):
        bs, dh, dw = depth.shape
        if init_pos is not None:
            means3D = init_pos
            pred_pos = None
        else:
            means3D = torch.tanh(pred_means3D)
            pred_pos = means3D
        pixel_idx = (means3D.clone() + 1) / 2
        pixel_idx[..., 0] = pixel_idx[..., 0] * (dw - 1)
        pixel_idx[..., 1] = (1 - pixel_idx[..., 1]) * (dh - 1)
        means_depth = depth[torch.arange(bs, device=depth.device).unsqueeze(1), pixel_idx[..., 1].long(), pixel_idx[..., 0].long()].unsqueeze(-1)
        means_x = means3D[..., :1] * means_depth * self.pixel_scale_x
        means_y = means3D[..., 1:] * means_depth * self.pixel_scale_y
        means3D_fixed = torch.cat([means_x, means_y, -means_depth], dim=-1)
        return pred_pos, means3D_fixed

    def gs_activate(self, pred_gs, depth, init_pos=None):
        position = pred_gs[..., :2]
        pred_pos, means3D = self.get_position(position, depth, init_pos=init_pos)

        colors = torch.sigmoid(pred_gs[..., 2:5])
        colors = colors / (colors.sum(dim=-1, keepdim=True) + 1e-6)
        hdr_colors = nn.Softplus()(pred_gs[..., 5:-1]) * 10
        colors = colors * hdr_colors

        scales = torch.sigmoid(pred_gs[..., -1:])
        log_min, log_max = math.log(0.005), math.log(0.05)
        scales = torch.exp(log_min + (log_max - log_min) * scales)
        scales = scales.repeat(1, 1, 3)

        quats = torch.zeros_like(hdr_colors.repeat(1, 1, 4))
        quats[..., -1] = 1.0
        opacities = torch.ones_like(means3D[..., :1])

        return pred_pos, means3D, colors, scales, opacities, quats

    def log_image(self, epoch, env_pre, env_gt, ldr_input, init_pos, init_color, img_name, stage, batch_idx):
        env_save = env_pre[0].permute(1, 2, 0).float().detach().cpu().numpy()
        env_gt_save = torch.expm1(env_gt[0].permute(1, 2, 0)).float().detach().cpu().numpy()
        ldr_input_save = ldr_input[0].permute(1, 2, 0).float().detach().cpu().numpy() / 2 + 0.5
        pos_vis = visualize_image_coords(init_pos[0].float().detach().cpu().numpy(), init_color[0].float().detach().cpu().numpy(), self.H, self.W, 10)
        grid1 = np.concatenate((env_save, env_gt_save), axis=0)
        grid2 = np.concatenate((pos_vis, ldr_input_save), axis=0)
        grid = np.concatenate((grid1, grid2), axis=1)[..., ::-1]
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b_{batch_idx}_{img_name[0]}_grid.exr', grid, exr_save_params)
