import torch
import torch.nn as nn
import copy
from torch.autograd import Function
from models.pointnet2_batch import pointnet2_cuda


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv2d, self).__init__(*args, (1, 1), **kwargs)
        else:
            super(Conv2d, self).__init__(*args, **kwargs)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv1d, self).__init__(*args, 1, **kwargs)
        else:
            super(Conv1d, self).__init__(*args, **kwargs)


def create_norm(norm_args, channels, dim=1):
    if norm_args is None:
        return None
    else:
        if dim == 1:
            return nn.BatchNorm1d(channels)
        elif dim == 2:
            return nn.BatchNorm2d(channels)


def create_act(act_args):
    if act_args is None:
        return None
    else:
        return nn.ReLU()


def create_convblock1d(*args, norm_args=None, act_args=None, **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)

    norm_layer = create_norm(norm_args, out_channels, dim=1)
    bias = False if norm_layer is not None else bias
    conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
    if norm_layer is not None:
        conv_layer.append(norm_layer)
    act_layer = create_act(act_args)
    if act_args is not None:
        conv_layer.append(act_layer)

    return nn.Sequential(*conv_layer)


def create_convblock2d(*args, norm_args=None, act_args=None, **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)

    norm_layer = create_norm(norm_args, out_channels, dim=2)
    bias = False if norm_layer is not None else bias
    conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
    if norm_layer is not None:
        conv_layer.append(norm_layer)
    act_layer = create_act(act_args)
    if act_args is not None:
        conv_layer.append(act_layer)

    return nn.Sequential(*conv_layer)


class GroupingOperation(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features, idx):
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, relative_xyz=True, normalize_dp=False, normalize_by_std=False, normalize_by_allstd=False, normalize_by_allstd2=False, return_only_idx=False):
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz, support_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)
    grouper = QueryAndGroup(radius, nsample, **group_args_copy)
    return grouper


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2_cuda.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown, known):
        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2_cuda.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features, idx, weight):
        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2_cuda.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.zeros([B, c, m], device='cuda', requires_grad=True)
        grad_out_data = grad_out.data.contiguous()

        pointnet2_cuda.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


def three_interpolation(unknown_xyz, known_xyz, know_feat):
    dist, idx = three_nn(unknown_xyz, known_xyz)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = three_interpolate(know_feat, idx, weight)
    return interpolated_feats


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    fj = torch.cat([dp, fj], 1)
    return fj
