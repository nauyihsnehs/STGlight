import os, sys
from typing import List

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logging_utils import render_pc_compare_o3d
from models.base_model import BaseModule, ModelRegistry
from models.pointnext_layers import create_convblock1d, create_convblock2d, create_act, create_grouper, \
    furthest_point_sample, three_interpolation

CHANNEL_MAP = {'dp_fj': lambda x: 3 + x}


def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


class LocalAggregation(nn.Module):
    def __init__(self,
                 channels,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 ):
        super().__init__()
        channels[0] = CHANNEL_MAP[feature_type](channels[0])
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_convblock2d(channels[i], channels[i + 1], norm_args=norm_args, act_args=None if i == (len(channels) - 2) and not last_act else act_args, **conv_args))
        self.convs = nn.Sequential(*convs)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf):
        p, f = pf
        dp, fj = self.grouper(p, p, f)
        fj = torch.cat([dp, fj], 1)
        f = self.pool(self.convs(fj))
        return f


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, stride=1, group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'}, act_args={'act': 'relu'}, conv_args=None, feature_type='dp_fj', use_res=False, is_head=False):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels if is_head else CHANNEL_MAP[feature_type](channels[0])

        if self.use_res:
            self.skipconv = create_convblock1d(in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[-1] else nn.Identity()
            self.act = create_act(act_args)

        create_conv = create_convblock1d if is_head else create_convblock2d
        convs = []
        for i in range(len(channels) - 1):
            convs.append(create_conv(channels[i], channels[i + 1], norm_args=norm_args if not is_head else None, act_args=None if i == len(channels) - 2 and (self.use_res or is_head) else act_args, **conv_args))
        self.convs = nn.Sequential(*convs)
        if not is_head:
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            self.sample_fn = furthest_point_sample

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            dp, fj = self.grouper(new_p, p, f)
            fj = torch.cat([dp, fj], 1)
            f = self.pool(self.convs(fj))
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    def __init__(self, mlp, upsample=True, norm_args={'norm': 'bn1d'}, act_args={'act': 'relu'}):
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1], norm_args=norm_args, act_args=act_args))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1], norm_args=norm_args, act_args=act_args))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        p1, f1 = pf1
        p2, f2 = pf2
        f = self.convs(torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
        return f


class InvResMLP(nn.Module):
    def __init__(self, in_channels, norm_args=None, act_args=None, aggr_args={'feature_type': 'dp_fj', "reduction": 'max'}, group_args={'NAME': 'ballquery'}, conv_args=None, expansion=1, use_res=True, num_posconvs=2, less_act=False, **kwargs):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels], norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None, group_args=group_args, conv_args=conv_args, **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1], norm_args=norm_args, act_args=act_args if (i != len(channels) - 2) and not less_act else None, **conv_args))
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class PointNextEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, width: int = 32,
                 blocks=[1, 2, 3, 2, 2], strides=[1, 4, 4, 4, 4],
                 block='InvResMLP', nsample=32, radius=0.1,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery', 'normalize_dp': True},
                 sa_layers=2, sa_use_res=True, **kwargs):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'})
        self.act_args = kwargs.get('act_args', {'act': 'relu'})
        self.conv_args = kwargs.get('conv_args', {})
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)

        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args['radius'] = self.radii[i]
            group_args['nsample'] = self.nsample[i]
            encoder.append(self._make_enc(block, channels[i], blocks[i], stride=strides[i], group_args=group_args, is_head=i == 0 and strides[i] == 1))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        param_list = []
        if isinstance(param, List):
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append([param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args['radius']
        nsample = group_args['nsample']
        group_args['radius'] = radii[0]
        group_args['nsample'] = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels, self.sa_layers if not is_head else 1, stride, group_args=group_args, sampler=self.sampler, norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args, is_head=is_head, use_res=self.sa_use_res, **self.aggr_args))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args['radius'] = radii[i]
            group_args['nsample'] = nsample[i]
            layers.append(block(self.in_channels, aggr_args=self.aggr_args, norm_args=self.norm_args, act_args=self.act_args, group_args=group_args, conv_args=self.conv_args, expansion=self.expansion, use_res=self.use_res))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)


class PointNextDecoder(nn.Module):
    def __init__(self, encoder_channel_list=[32, 64, 128, 256, 512], decoder_layers=2, decoder_stages=4, **kwargs):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 4))
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:]([p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
        return f[-len(self.decoder) - 1]


class SegHead(nn.Module):
    def __init__(self, num_classes=3, in_channels=32, mlps=None, norm_args={'norm': 'bn'}, dropout=0.5):
        super().__init__()

        self.global_feat = None
        multiplier = 1
        in_channels *= multiplier

        mlps = [in_channels, in_channels] + [num_classes]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1], norm_args=norm_args, act_args='relu'))
            heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1]))
        self.head = nn.Sequential(*heads)

    def forward(self, end_points):
        return self.head(end_points)


@ModelRegistry.register("pointnet")
class PointNet(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PointNextEncoder()
        self.decoder = PointNextDecoder()
        self.head = SegHead()
        self.img_log_dir = config.get('img_log_dir', './logs')
        os.makedirs(f'{self.img_log_dir}/train', exist_ok=True)
        os.makedirs(f'{self.img_log_dir}/val', exist_ok=True)
        self.shuffle = False
        self.pin_memory = False
        self.save_hyperparameters()

    def forward(self, data):
        p, f = self.encoder.forward_seg_feat(data)
        f = self.decoder(p, f).squeeze(-1)
        f = self.head(f)
        f = torch.tanh(f)
        return f

    @staticmethod
    def masked_L1(pred: torch.Tensor, gt: torch.Tensor, mask=None):
        if mask is not None:
            diff = (pred - gt).abs()
            diff = diff * mask.unsqueeze(1)
            return diff.sum() / (mask.sum() * pred.size(1) + 1e-8)
        else:
            return F.l1_loss(pred, gt)

    @staticmethod
    def grad_loss_proj_1d(pred, gt, pos, mask, n_dir: int = 3):
        B, C, N = pred.shape
        device = pred.device

        base_dirs = torch.tensor([
            [1.0, 0.7, 0.3],
            [0.3, 1.0, 0.7],
            [0.7, 0.3, 1.0],
        ], device=device, dtype=pos.dtype)
        base_dirs = base_dirs[:n_dir]
        base_dirs = base_dirs / (base_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        loss = 0.0
        for d in base_dirs:
            s = (pos * d).sum(-1)
            idx = s.argsort(dim=-1)

            idx_exp = idx.unsqueeze(1).expand(-1, C, -1)
            pred_s = torch.gather(pred, 2, idx_exp)
            gt_s = torch.gather(gt, 2, idx_exp)
            mask_s = torch.gather(mask, 1, idx)

            dp = pred_s[..., 1:] - pred_s[..., :-1]
            dg = gt_s[..., 1:] - gt_s[..., :-1]

            valid = mask_s[..., 1:] & mask_s[..., :-1]

            diff = (dp - dg).pow(2).sum(1)
            diff = diff * valid

            loss += diff.sum() / (valid.sum() + 1e-8)

        return loss / float(len(base_dirs))

    def log_pcl(self, epoch, pos, color_input, color_pred, color_gt, pcl_name, stage, batch_idx):
        pos = pos[0].detach().cpu().numpy()
        color_input = color_input[0].permute(1, 0).detach().cpu().numpy()
        color_pred = color_pred[0].permute(1, 0).detach().cpu().numpy()
        color_gt = color_gt[0].permute(1, 0).detach().cpu().numpy()
        color_input = np.clip(color_input * 127.5 + 127.5, 0, 255).astype(np.uint8)
        color_pred = np.clip(color_pred * 127.5 + 127.5, 0, 255).astype(np.uint8)
        color_gt = np.clip(color_gt * 127.5 + 127.5, 0, 255).astype(np.uint8)

        grid_save_path = f'{self.img_log_dir}/{stage}/e{str(epoch).zfill(3)}_b{batch_idx}_{pcl_name[0]}_grid.jpg'
        render_pc_compare_o3d(grid_save_path, pos, color_input, color_pred, color_gt)
