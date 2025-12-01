import math
from collections import deque

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, mlp_ratio=4):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AttnBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mlp_ratio)

    def forward(self, x, ctx=None, attn_mask=None, key_padding_mask=None):
        q = self.norm1(x)
        if ctx is None:
            k = v = q
        else:
            k = v = self.norm1(ctx)
        attn_out, _ = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TFBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, depth=(1, 1, 1, 1), G=4, N=8,
                 pool_stride=2, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.d_model, self.G, self.N = d_model, G, N
        if isinstance(depth, int):
            d1 = d2 = d3 = d4 = depth
        else:
            d1, d2, d3, d4 = depth

        self.register_buffer("global_mem", torch.zeros(1, G, d_model))
        nn.init.normal_(self.global_mem, std=0.02)
        self.recent_queue = deque(maxlen=N)

        self.td1 = nn.ModuleList([AttnBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(d1)])
        self.td2 = nn.ModuleList([AttnBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(d2)])
        self.td3 = nn.ModuleList([AttnBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(d3)])
        self.td4 = nn.ModuleList([AttnBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(d4)])

        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride) if pool_stride > 1 else None

    def _downsample_tokens(self, x):
        if self.pool is None: return x
        return self.pool(x.transpose(1, 2)).transpose(1, 2).contiguous()

    @torch.no_grad()
    def reset_state(self, device=None, G=None):
        device = device or self.global_mem.device
        if G is None: G = self.G
        self.G = G
        self.global_mem = torch.zeros(1, G, self.d_model, device=device)
        nn.init.normal_(self.global_mem, std=0.02)
        self.recent_queue.clear()

    def forward(self, Fc, add_to_recent=True):
        B, T, D = Fc.shape
        device = Fc.device

        if len(self.recent_queue) == 0:
            recent = Fc
        else:
            recent = torch.cat(list(self.recent_queue) + [Fc], dim=1)
        Ft = self._downsample_tokens(recent)
        for blk in self.td1:
            Ft = blk(Ft, ctx=None)

        Fg = self.global_mem.expand(B, -1, -1).to(device)  # [B, G, D]
        Fc2 = Fc
        for blk in self.td2:
            Fc2 = blk(Fc2, ctx=Fg)

        Fc3 = Fc2
        for blk in self.td3:
            Fc3 = blk(Fc3, ctx=Ft)

        cur_summary = Fc3.mean(dim=1, keepdim=True)  # [B, 1, D]
        Fg_in = torch.cat([Fg, cur_summary.expand(B, self.G, D)], dim=1)  # [B, G*2, D]
        Fg4 = Fg
        for blk in self.td4:
            Fg4 = blk(Fg4, ctx=Fg_in)

        gate = self.gate(cur_summary)  # [B, 1, D]
        Fg_new = Fg * (1 - gate) + Fg4 * gate  # [B, G, D]

        if add_to_recent:
            self.recent_queue.append(self._downsample_tokens(Fc).detach())
        self.global_mem = Fg_new.mean(dim=0, keepdim=True).detach()  # [1, G, D]

        return Fc3, Fg_new


def build_2d_sincos_pos_embed(H, W, dim, device):
    def get_1d_sincos(L, D_half):
        pos = torch.arange(L, device=device).float()
        div = torch.exp(torch.arange(0, D_half, 2, device=device).float() * (-math.log(10000.0) / D_half))
        sin = torch.sin(pos[:, None] * div[None, :])
        cos = torch.cos(pos[:, None] * div[None, :])
        pe = torch.zeros(L, D_half, device=device)
        pe[:, 0::2] = sin
        pe[:, 1::2] = cos
        return pe

    D_half = dim // 2
    pe_y = get_1d_sincos(H, D_half)  # [H, D/2]
    pe_x = get_1d_sincos(W, dim - D_half)  # [W, D - D/2]
    pe_y = pe_y[:, None, :]  # [H, 1, D/2]
    pe_x = pe_x[None, :, :]  # [1, W, D/2]
    pe = torch.cat([pe_y.expand(H, W, -1), pe_x.expand(H, W, -1)], dim=-1)  # [H, W, D]
    return pe.view(1, H * W, dim)


class TemporalDecoder(nn.Module):
    def __init__(self, in_ch=256, dim=512, nhead=8, depth=(1, 1, 1, 1), G=4, N=8,
                 pool_stride=2, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.in_ch, self.dim = in_ch, dim
        self.proj_in = nn.Linear(in_ch, dim) if dim != in_ch else None
        self.tf = TFBlock(d_model=dim, nhead=nhead, depth=depth, G=G, N=N,
                          pool_stride=pool_stride, mlp_ratio=mlp_ratio, dropout=dropout)
        self.proj_out = nn.Linear(dim, in_ch) if dim != in_ch else None
        self._cached_shape = None
        self.register_buffer("pos", None, persistent=False)

    @torch.no_grad()
    def reset_state(self, device=None, G=None):
        self.tf.reset_state(device=device, G=G)

    def _maybe_build_pos(self, H, W, device):
        need_new = (self.pos is None) or (self._cached_shape != (H, W, self.dim)) or (self.pos.device != device)
        if need_new:
            self.pos = build_2d_sincos_pos_embed(H, W, self.dim, device)
            self._cached_shape = (H, W, self.dim)

    def forward(self, CF_feat, add_to_recent=True):
        B, C, H, W = CF_feat.shape
        x = CF_feat.flatten(2).transpose(1, 2)
        if self.proj_in is not None:
            x = self.proj_in(x)
        self._maybe_build_pos(H, W, x.device)
        x = x + self.pos
        x_fused, _ = self.tf(x, add_to_recent=add_to_recent)
        x_out = x_fused.transpose(1, 2).view(B, self.dim, H, W)  # [B, C, H, W]
        if self.proj_out is not None:
            x_out = self.proj_out(x_out.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        return x_out
