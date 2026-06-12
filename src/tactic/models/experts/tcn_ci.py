"""f3 CI-TCN encoder (sec10.1): per-asset shared-weight dilated TCN, no cross-sectional layer.

6 residual blocks, kernel 5, dilations [1,2,4,8,16,32], 64 channels, GELU, dropout 0.1, causal
(left-padded) convolutions -> mean-pool over time -> MLP -> quantile heads. This is the
ablation twin of f4 (which adds one attention layer over the date's tokens).
"""
from __future__ import annotations

import torch
import torch.nn as nn

DILATIONS = [1, 2, 4, 8, 16, 32]


class _CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):                      # x: (B, C, L)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))  # left-pad only (causal)


class _TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.c1 = _CausalConv1d(c_in, c_out, kernel, dilation)
        self.c2 = _CausalConv1d(c_out, c_out, kernel, dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(self.act(self.c1(x)))
        h = self.drop(self.act(self.c2(h)))
        return self.act(h + self.down(x))


class TCNEncoder(nn.Module):
    def __init__(self, c_in=18, channels=64, kernel=5, dropout=0.1):
        super().__init__()
        blocks, c = [], c_in
        for d in DILATIONS:
            blocks.append(_TemporalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.net = nn.Sequential(*blocks)
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.GELU())
        self.out_dim = channels

    def forward(self, x):                       # x: (N, 18, L)
        h = self.net(x)                         # (N, channels, L)
        h = h.mean(dim=-1)                      # mean-pool over time
        return self.mlp(h)                      # (N, channels)


class QuantileHead(nn.Module):
    """Median + softplus increments -> non-crossing quantiles (sec10.2)."""
    def __init__(self, d_in, n_quantiles):
        super().__init__()
        self.median = nn.Linear(d_in, 1)
        self.inc = nn.Linear(d_in, n_quantiles - 1)
        self.n = n_quantiles

    def forward(self, h):
        med = self.median(h)                    # (N,1)
        inc = nn.functional.softplus(self.inc(h))  # (N, n-1) >=0
        mid = self.n // 2
        cols = []
        for j in range(self.n):
            if j == mid:
                cols.append(med)
            elif j > mid:
                cols.append(med + inc[:, mid:j].sum(dim=1, keepdim=True))
            else:
                cols.append(med - inc[:, j:mid].sum(dim=1, keepdim=True))
        return torch.cat(cols, dim=1)           # (N, n) ascending


class F3Model(nn.Module):
    """f3: encoder -> quantile head (no cross-sectional interaction)."""
    def __init__(self, n_quantiles=7, channels=64, dropout=0.1, static_dim=12):
        super().__init__()
        self.enc = TCNEncoder(channels=channels, dropout=dropout)
        self.head = QuantileHead(self.enc.out_dim + static_dim, n_quantiles)

    def forward(self, x_seq, u, m=None):
        h = torch.cat([self.enc(x_seq), u], dim=1)
        return self.head(h)
