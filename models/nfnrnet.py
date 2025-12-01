import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.base_model import BaseModule, ModelRegistry

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_FLOAT, cv.IMWRITE_EXR_COMPRESSION,
                   cv.IMWRITE_EXR_COMPRESSION_ZIP]


class VGGLoss(nn.Module):
    def __init__(self, layers=(2, 7, 12, 21), weight: float = 1.0):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, l)]))
            prev = l

        for p in self.parameters():
            p.requires_grad = False

        self.weight = weight

    def _norm(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=t.device)[None, :, None, None]
        return (t - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_n, y_n = self._norm(x), self._norm(y)
        total = 0.0
        for s in self.slices:
            x_n = s(x_n)
            y_n = s(y_n)
            total = total + F.l1_loss(x_n, y_n)
        return total * self.weight


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv_up = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.block1 = ConvBlock(out_ch + skip_ch, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv_up(x)
        assert x.shape[2:] == skip.shape[2:], \
            f"spatial mismatch: {x.shape[2:]} vs {skip.shape[2:]}"
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


def tv_l1(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return dx.mean() + dy.mean()


def hdr_to_vgg_input(x: torch.Tensor, clip_val: float = 8.0) -> torch.Tensor:
    x = torch.clamp(x, 0.0, clip_val) / clip_val
    x = torch.clamp(x, 0.0, 1.0)
    x = x ** (1.0 / 2.2)
    return x


class NFNetBackbone(nn.Module):
    def __init__(
            self,
            env_channels: int = 3,
            light_channels: int = 3,
            base_ch: int = 48,
    ):
        super().__init__()
        self.env_channels = env_channels
        self.light_channels = light_channels

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.env_stem = ConvBlock(env_channels + 1, c1)
        self.env_down1 = Down(c1, c2)
        self.env_down2 = Down(c2, c3)

        self.light_stem = ConvBlock(light_channels, c1)
        self.light_down1 = Down(c1, c2)
        self.light_down2 = Down(c2, c3)

        self.mid = ConvBlock(c3, c3)
        self.up2 = Up(in_ch=c3, skip_ch=c3, out_ch=c2)  # H/4 -> H/2
        self.up1 = Up(in_ch=c2, skip_ch=c2, out_ch=c1)  # H/2 -> H
        self.head = ConvBlock(c1, c1)
        self.w_head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _make_mask(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # 1 where valid, 0 where (approximately) black
        return (x.abs().sum(dim=1, keepdim=True) > eps).float()

    def forward(self, env, light, env_mask=None):
        env_in = torch.cat([env, env_mask], dim=1)

        e = self.env_stem(env_in)  # B, c1, H,   W
        e, e_s1 = self.env_down1(e)  # B, c2, H/2, W/2
        e, e_s2 = self.env_down2(e)  # B, c3, H/4, W/4

        l = self.light_stem(light)  # B, c1, H,   W
        l, l_s1 = self.light_down1(l)  # B, c2, H/2, W/2
        l, l_s2 = self.light_down2(l)  # B, c3, H/4, W/4

        x = e + l
        s1 = e_s1 + l_s1
        s2 = e_s2 + l_s2

        x = self.mid(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.head(x)
        w = torch.sigmoid(self.w_head(x))
        return w


class NFNet(nn.Module):
    def __init__(self, env_channels=3, light_channels=3, base_ch=48):
        super().__init__()
        self.backbone = NFNetBackbone(env_channels=env_channels,
                                      light_channels=light_channels,
                                      base_ch=base_ch)

    def forward(self, env, light, env_mask=None):
        W = self.backbone(env, light, env_mask)
        fused = W * env + (1.0 - W) * light
        return fused, W


def nfnet_loss(fused_hdr, gt_hdr, W, lambda_tv):
    l2 = F.mse_loss(fused_hdr, gt_hdr)

    l_tv = torch.tensor(0.0, device=fused_hdr.device)
    if lambda_tv > 0.0:
        l_tv = tv_l1(W) * lambda_tv

    loss = l2 + l_tv
    return {"loss": loss, "l2": l2, "l_tv": l_tv}


def normalize_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + eps
    return (x - mean) / std


def normalize_depth(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    d_min = d.amin(dim=(2, 3), keepdim=True)
    d_max = d.amax(dim=(2, 3), keepdim=True)
    scale = (d_max - d_min).clamp(min=eps)
    return (d - d_min) / scale


class NRNetBackbone(nn.Module):
    def __init__(self, env_channels=3, albedo_channels=3, depth_channels=1, base_ch=32):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8

        self.app_stem = ConvBlock(env_channels + 1, c1)
        self.app_down1 = Down(c1, c2)
        self.app_down2 = Down(c2, c3)
        self.app_down3 = Down(c3, c4)
        self.app_down4 = Down(c4, c4)

        self.geo_stem = ConvBlock(albedo_channels + depth_channels + 1, c1)
        self.geo_down1 = Down(c1, c2)
        self.geo_down2 = Down(c2, c3)
        self.geo_down3 = Down(c3, c4)
        self.geo_down4 = Down(c4, c4)

        self.mid = nn.Sequential(
            ConvBlock(c4, c4),
            nn.Conv2d(c4, c4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )

        self.up4 = Up(in_ch=c4, skip_ch=c4, out_ch=c4)  # H/16 -> H/8
        self.up3 = Up(in_ch=c4, skip_ch=c4, out_ch=c3)  # H/8  -> H/4
        self.up2 = Up(in_ch=c3, skip_ch=c3, out_ch=c2)  # H/4  -> H/2
        self.up1 = Up(in_ch=c2, skip_ch=c2, out_ch=c1)  # H/2  -> H

        self.head = ConvBlock(c1, c1)
        self.res_head = nn.Conv2d(c1, env_channels, kernel_size=1)

    @staticmethod
    def _make_mask(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (x.abs().sum(dim=1, keepdim=True) > eps).float()

    def forward(self, current_env, albedo_env, depth_env, mask_current=None, mask_albedo=None, ):
        app_in = torch.cat([current_env, mask_current], dim=1)
        geo_in = torch.cat([albedo_env, depth_env, mask_albedo], dim=1)

        a = self.app_stem(app_in)  # H
        a, a_s1 = self.app_down1(a)  # H/2
        a, a_s2 = self.app_down2(a)  # H/4
        a, a_s3 = self.app_down3(a)  # H/8
        a, a_s4 = self.app_down4(a)  # H/16

        g = self.geo_stem(geo_in)
        g, g_s1 = self.geo_down1(g)
        g, g_s2 = self.geo_down2(g)
        g, g_s3 = self.geo_down3(g)
        g, g_s4 = self.geo_down4(g)

        x = a + g
        s4 = a_s4 + g_s4
        s3 = a_s3 + g_s3
        s2 = a_s2 + g_s2
        s1 = a_s1 + g_s1

        x = self.mid(x)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.head(x)
        residual = self.res_head(x)
        return residual


class NRNet(nn.Module):
    def __init__(self, env_channels=3, albedo_channels=3, depth_channels=1, base_ch=32, normalize_inputs=False):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.backbone = NRNetBackbone(
            env_channels=env_channels,
            albedo_channels=albedo_channels,
            depth_channels=depth_channels,
            base_ch=base_ch,
        )

    def forward(self, current_env, albedo_env, depth_env, current_mask, albedo_mask):
        if self.normalize_inputs:
            current_norm = normalize_image(current_env)
            albedo_norm = normalize_image(albedo_env)
            depth_norm = normalize_depth(depth_env)
        else:
            current_norm = current_env
            albedo_norm = albedo_env
            depth_norm = depth_env

        residual = self.backbone(current_norm, albedo_norm, depth_norm, current_mask, albedo_mask)
        pred_env = current_env + residual
        return pred_env, residual


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps):
    assert mask.ndim == 4 and mask.shape[1] == 1
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum() * pred.shape[1]
    denom = denom.clamp(min=eps)
    return diff2.sum() / denom


def nrnet_loss(pred_env, gt_env, mask_albedo, lambda_global=0.1, vgg=None, lambda_vgg=0.0):
    l_albedo = masked_mse(pred_env, gt_env, mask_albedo)
    l_global = F.mse_loss(pred_env, gt_env)

    l_vgg = torch.tensor(0.0, device=pred_env.device)
    if vgg is not None and lambda_vgg > 0.0:
        pred_tm = torch.clamp(pred_env, 0.0, 1.0)
        gt_tm = torch.clamp(gt_env, 0.0, 1.0)
        l_vgg = vgg(pred_tm, gt_tm) * lambda_vgg

    loss = l_albedo + lambda_global * l_global + l_vgg
    return {"loss": loss, "l_albedo": l_albedo, "l_global": l_global, "l_vgg": l_vgg}


@ModelRegistry.register("nfnet")
class NFNetModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = NFNet(
            env_channels=config.get("env_channels", 3),
            light_channels=config.get("light_channels", 3),
            base_ch=config.get("base_ch", 48),
        )
        self.img_log_dir = config.get("img_log_dir")

        self.lambda_vgg = config.get("lambda_vgg", 0.0)
        self.lambda_tv = config.get("lambda_tv", 0.01)

    def on_train_epoch_start(self):
        n_batches = int(self.trainer.num_training_batches)
        sample_num = min(10, max(1, n_batches - 1))
        interval = n_batches / sample_num
        self._save_idx = [int(i * interval) for i in range(sample_num)]

    def on_validation_start(self):
        n_batches = int(self.trainer.num_val_batches[0])
        sample_num = min(3, max(1, n_batches - 1))
        interval = n_batches / sample_num
        self._val_save_idx = [int(i * interval) for i in range(sample_num)]

    def log_image(self, epoch, idx_max, gt, env, light, mask, pred, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        to_np = lambda x: x[save_idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
        tone = lambda x: np.expm1(x[save_idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5)
        gt_save = tone(gt)
        pred_save = tone(pred)
        light_save = tone(light)
        env_save = to_np(env)
        mask_save = mask[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        env_save = env_save * mask_save + (1 - mask_save) * light_save
        grid1 = np.concatenate((gt_save, pred_save), axis=0)
        grid2 = np.concatenate((env_save, light_save), axis=0)
        grid_hdr = np.concatenate((grid1, grid2), axis=1)
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{epoch:03d}_b{batch_idx}_{img_name[save_idx]}.exr', grid_hdr, exr_save_params)


@ModelRegistry.register("nrnet")
class NRNetModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = NRNet(
            env_channels=config.get("env_channels", 3),
            albedo_channels=config.get("albedo_channels", 3),
            depth_channels=config.get("depth_channels", 1),
            base_ch=config.get("base_ch", 32),
            normalize_inputs=config.get("normalize_inputs", False),
        )

        self.lambda_global = config.get("lambda_global", 0.1)
        self.lambda_vgg = config.get("lambda_vgg", 0.05)
        self.vgg = VGGLoss().eval() if self.lambda_vgg > 0 else None
        self.img_log_dir = config.get("img_log_dir")

    def forward(self, batch):
        gt, cur, alb, dep, img_name = batch
        pred, res = self.model(cur, alb, dep)
        return {"pred": pred, "res": res}

    def on_train_epoch_start(self):
        n_batches = int(self.trainer.num_training_batches)
        sample_num = min(10, max(1, n_batches - 1))
        interval = n_batches / sample_num
        self._save_idx = [int(i * interval) for i in range(sample_num)]

    def on_validation_start(self):
        n_batches = int(self.trainer.num_val_batches[0])
        sample_num = min(3, max(1, n_batches - 1))
        interval = n_batches / sample_num
        self._val_save_idx = [int(i * interval) for i in range(sample_num)]

    def log_image(self, epoch, idx_max, gt, cur, alb, dep, pred, img_name, stage, batch_idx):
        save_idx = np.random.randint(0, idx_max)
        gt_save = gt[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        cur_save = cur[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        alb_save = alb[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        dep_save = dep[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        dep_save = np.concatenate([dep_save] * 3, axis=2)
        pred_save = pred[save_idx].permute(1, 2, 0).detach().cpu().numpy()
        place_holder = np.ones_like(pred_save)

        grid1 = np.concatenate((gt_save, pred_save), axis=0)
        grid2 = np.concatenate((cur_save, alb_save), axis=0)
        grid3 = np.concatenate((dep_save, place_holder), axis=0)
        grid = np.concatenate((grid1, grid2, grid3), axis=1)
        grid_ldr = np.clip(grid * 0.5 + 0.5, 0, 1) * 255
        cv.imwrite(f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{img_name[save_idx]}.jpg', grid_ldr)
