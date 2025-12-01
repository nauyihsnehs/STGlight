import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn

from models.base_model import BaseModule, ModelRegistry
from models.nfnrnet import NFNetBackbone, NRNetBackbone, nfnet_loss, nrnet_loss, VGGLoss, normalize_image, normalize_depth
from models.temporal_fusion import TemporalDecoder

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_FLOAT, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_ZIP]


class NFNetBackboneTF(NFNetBackbone):
    def __init__(self, env_channels=3, light_channels=3, base_ch=48, tf_dim=512, tf_nhead=8, tf_depth=(1, 1, 1, 1), tf_G=4, tf_N=8, tf_pool_stride=2, tf_mlp_ratio=4.0, tf_dropout=0.0):
        super().__init__(env_channels=env_channels, light_channels=light_channels, base_ch=base_ch)

        # Encoder bottleneck channels (c3 = base_ch * 4)
        latent_ch = self.env_down2.block2.bn.num_features

        self.temporal = TemporalDecoder(in_ch=latent_ch, dim=tf_dim, nhead=tf_nhead, depth=tf_depth, G=tf_G, N=tf_N, pool_stride=tf_pool_stride, mlp_ratio=tf_mlp_ratio, dropout=tf_dropout)

    @torch.no_grad()
    def reset_state(self, device = None, G = None):
        self.temporal.reset_state(device=device, G=G)

    def forward(self, env, light, env_mask=None, add_to_recent=True):
        if env_mask is None:
            env_mask = self._make_mask(env)

        env_in = torch.cat([env, env_mask], dim=1)
        e = self.env_stem(env_in)
        e, e_s1 = self.env_down1(e)
        e, e_s2 = self.env_down2(e)

        l = self.light_stem(light)
        l, l_s1 = self.light_down1(l)
        l, l_s2 = self.light_down2(l)

        x = e + l
        s1 = e_s1 + l_s1
        s2 = e_s2 + l_s2

        x = self.temporal(x, add_to_recent=add_to_recent)

        x = self.mid(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.head(x)
        w = torch.sigmoid(self.w_head(x))
        return w


class NFNetTF(nn.Module):
    def __init__(self, env_channels=3, light_channels=3, base_ch=48, tf_dim=512, tf_nhead=8, tf_depth=(1, 1, 1, 1), tf_G=4, tf_N=8, tf_pool_stride=2, tf_mlp_ratio=4.0, tf_dropout=0.0):
        super().__init__()
        self.backbone = NFNetBackboneTF(env_channels=env_channels, light_channels=light_channels, base_ch=base_ch, tf_dim=tf_dim, tf_nhead=tf_nhead, tf_depth=tf_depth, tf_G=tf_G, tf_N=tf_N, tf_pool_stride=tf_pool_stride, tf_mlp_ratio=tf_mlp_ratio, tf_dropout=tf_dropout)

    @torch.no_grad()
    def reset_state(self, device = None, G = None):
        self.backbone.reset_state(device=device, G=G)

    def forward(self, env, light, env_mask=None, add_to_recent=True):
        W = self.backbone(env, light, env_mask, add_to_recent=add_to_recent)
        fused = W * env + (1.0 - W) * light
        return fused, W


class NRNetBackboneTF(NRNetBackbone):
    def __init__(self, env_channels=3, albedo_channels=3, depth_channels=1, base_ch=32, tf_dim=512, tf_nhead=8, tf_depth=(1, 1, 1, 1), tf_G=4, tf_N=8, tf_pool_stride=2, tf_mlp_ratio=4.0, tf_dropout=0.0):
        super().__init__(env_channels=env_channels, albedo_channels=albedo_channels, depth_channels=depth_channels, base_ch=base_ch)

        latent_ch = self.app_down4.block2.bn.num_features

        self.temporal = TemporalDecoder(in_ch=latent_ch, dim=tf_dim, nhead=tf_nhead, depth=tf_depth, G=tf_G, N=tf_N, pool_stride=tf_pool_stride, mlp_ratio=tf_mlp_ratio, dropout=tf_dropout)

    @torch.no_grad()
    def reset_state(self, device = None, G = None):
        self.temporal.reset_state(device=device, G=G)

    def forward(self, current_env, albedo_env, depth_env, mask_current=None, mask_albedo=None, add_to_recent=True):
        if mask_current is None:
            mask_current = self._make_mask(current_env)
        if mask_albedo is None:
            mask_albedo = self._make_mask(albedo_env)

        app_in = torch.cat([current_env, mask_current], dim=1)
        geo_in = torch.cat([albedo_env, depth_env, mask_albedo], dim=1)

        a = self.app_stem(app_in)
        a, a_s1 = self.app_down1(a)
        a, a_s2 = self.app_down2(a)
        a, a_s3 = self.app_down3(a)
        a, a_s4 = self.app_down4(a)

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

        x = self.temporal(x, add_to_recent=add_to_recent)

        x = self.mid(x)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.head(x)
        residual = self.res_head(x)
        return residual


class NRNetTF(nn.Module):
    def __init__(self, env_channels=3, albedo_channels=3, depth_channels=1, base_ch=32, normalize_inputs=False, tf_dim=512, tf_nhead=8, tf_depth=(1, 1, 1, 1), tf_G=4, tf_N=8, tf_pool_stride=2, tf_mlp_ratio=4.0, tf_dropout=0.0):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.backbone = NRNetBackboneTF(env_channels=env_channels, albedo_channels=albedo_channels, depth_channels=depth_channels, base_ch=base_ch, tf_dim=tf_dim, tf_nhead=tf_nhead, tf_depth=tf_depth, tf_G=tf_G, tf_N=tf_N, tf_pool_stride=tf_pool_stride, tf_mlp_ratio=tf_mlp_ratio, tf_dropout=tf_dropout)

    @torch.no_grad()
    def reset_state(self, device = None, G = None):
        self.backbone.reset_state(device=device, G=G)

    def forward(self, current_env, albedo_env, depth_env, current_mask, albedo_mask, add_to_recent=True):
        if self.normalize_inputs:
            current_norm = normalize_image(current_env)
            albedo_norm = normalize_image(albedo_env)
            depth_norm = normalize_depth(depth_env)
        else:
            current_norm = current_env
            albedo_norm = albedo_env
            depth_norm = depth_env

        residual = self.backbone(current_norm, albedo_norm, depth_norm, current_mask, albedo_mask, add_to_recent=add_to_recent)
        pred_env = current_env + residual
        return pred_env, residual


@ModelRegistry.register("nfnet_tf")
class NFNetTFModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = NFNetTF(
            env_channels=config.get("env_channels", 3),
            light_channels=config.get("light_channels", 3),
            base_ch=config.get("base_ch", 48),
            tf_dim=config.get("tf_dim", 512),
            tf_nhead=config.get("tf_nhead", 8),
            tf_depth=config.get("tf_depth", (1, 1, 1, 1)),
            tf_G=config.get("tf_G", 4),
            tf_N=config.get("tf_N", 8),
            tf_pool_stride=config.get("tf_pool_stride", 2),
            tf_mlp_ratio=config.get("tf_mlp_ratio", 4.0),
            tf_dropout=config.get("tf_dropout", 0.0),
        )
        pretrained_ckpt = config.get("pretrained_nfnet_ckpt", None)
        if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
            ckpt = torch.load(pretrained_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)

            model_state = self.model.state_dict()
            adapted = {}
            for k, v in state.items():
                if not k.startswith("model."):
                    continue
                subk = k[len("model."):]
                if subk in model_state:
                    adapted[subk] = v

            missing, unexpected = self.model.load_state_dict(adapted, strict=False)
            print("[NFNetTF] loaded from", pretrained_ckpt)
            print("  missing:", missing)
            print("  unexpected:", unexpected)

        self.freeze_backbone_first = bool(config.get("freeze_backbone_first", False))
        if self.freeze_backbone_first:
            for name, p in self.model.named_parameters():
                if "backbone.temporal" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        self.img_log_dir = config.get("img_log_dir")
        self.lambda_tv = config.get("lambda_tv", 0.05)
        self.lambda_temporal = config.get("lambda_temporal", 0.1)

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

    def log_image(self, epoch, gt, env, light, mask, pred, img_name, stage: str, batch_idx: int):
        T, C, H, W = gt.shape
        columns = []
        for save_idx in range(T):
            to_np = lambda x: x[save_idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
            tone = lambda x: np.expm1(x[save_idx].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5)

            gt_save = tone(gt)
            pred_save = tone(pred)
            light_save = tone(light)
            env_save = to_np(env)
            mask_save = mask[save_idx].permute(1, 2, 0).detach().cpu().numpy()
            env_save = env_save * mask_save + (1 - mask_save) * light_save

            col = np.concatenate((gt_save, pred_save, env_save), axis=0)
            columns.append(col)

        grid_hdr = np.concatenate(columns, axis=1)

        seq_name = img_name[0] if isinstance(img_name, (list, tuple)) and len(img_name) > 0 else "seq"
        out_path = f"{self.img_log_dir}/{stage}/e{epoch:03d}_b{batch_idx}_{seq_name}.exr"
        cv.imwrite(out_path, grid_hdr, exr_save_params)


@ModelRegistry.register("nrnet_tf")
class NRNetTFModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = NRNetTF(
            env_channels=config.get("env_channels", 3),
            albedo_channels=config.get("albedo_channels", 3),
            depth_channels=config.get("depth_channels", 1),
            base_ch=config.get("base_ch", 32),
            normalize_inputs=config.get("normalize_inputs", False),
            tf_dim=config.get("tf_dim", 512),
            tf_nhead=config.get("tf_nhead", 8),
            tf_depth=config.get("tf_depth", [1, 1, 1, 1]),
            tf_G=config.get("tf_G", 4),
            tf_N=config.get("tf_N", 8),
            tf_pool_stride=config.get("tf_pool_stride", 2),
            tf_mlp_ratio=config.get("tf_mlp_ratio", 4.0),
            tf_dropout=config.get("tf_dropout", 0.0),
        )

        pretrained_ckpt = config.get("pretrained_nrnet_ckpt", None)
        if pretrained_ckpt is not None and os.path.isfile(pretrained_ckpt):
            ckpt = torch.load(pretrained_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)

            model_state = self.model.state_dict()
            adapted = {}
            for k, v in state.items():
                if not k.startswith("model."):
                    continue
                subk = k[len("model."):]
                if subk in model_state:
                    adapted[subk] = v

            missing, unexpected = self.model.load_state_dict(adapted, strict=False)
            print("[NRNetTF] loaded from", pretrained_ckpt)
            print("  missing:", missing)
            print("  unexpected:", unexpected)

        self.freeze_backbone_first = bool(config.get("freeze_backbone_first", False))
        if self.freeze_backbone_first:
            for name, p in self.model.named_parameters():
                if "backbone.temporal" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        self.lambda_global = config.get("lambda_global", 0.1)
        self.lambda_vgg = config.get("lambda_vgg", 0.05)
        self.lambda_temporal = config.get("lambda_temporal", 0.1)
        self.vgg = VGGLoss().eval() if self.lambda_vgg > 0 else None
        self.img_log_dir = config.get("img_log_dir")

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

    def log_image(self, epoch: int, gt: torch.Tensor, cur: torch.Tensor, alb: torch.Tensor, pred: torch.Tensor,
                  img_name, stage: str, batch_idx: int):
        T, C, H, W = gt.shape
        columns = []
        for t in range(T):
            def to_np_frame(x: torch.Tensor) -> np.ndarray:
                return x[t].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5

            gt_save = to_np_frame(gt)
            cur_save = to_np_frame(cur)
            alb_save = to_np_frame(alb)
            pred_save = to_np_frame(pred)
            cur_save[cur_save < 0.004] = alb_save[cur_save < 0.004]
            col = np.concatenate((gt_save, cur_save, pred_save), axis=0)
            columns.append(col)

        grid = np.concatenate(columns, axis=1)
        grid_ldr = np.clip(grid, 0.0, 1.0) * 255.0
        grid_ldr = grid_ldr.astype(np.uint8)
        seq_name = img_name[0] if isinstance(img_name, (list, tuple)) and len(img_name) > 0 else "seq"
        out_path = f"{self.img_log_dir}/{stage}/e{epoch:03d}_b{batch_idx}_{seq_name}.jpg"
        cv.imwrite(out_path, grid_ldr)
