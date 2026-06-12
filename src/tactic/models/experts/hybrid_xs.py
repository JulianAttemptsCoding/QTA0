"""f4 Hybrid (sec10.1): f3 encoder -> token [enc_i, u_i] -> ONE MHSA layer over the date's
N_t tokens (4 heads, d_model=64, pre-LN) + residual -> quantile heads + gate logits.

Param-count target 0.15-0.25M (asserted in tests). Setting the attention to identity recovers
f3 (architecture-wiring ablation, sec10 QA).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .tcn_ci import QuantileHead, TCNEncoder


class F4Model(nn.Module):
    def __init__(self, n_quantiles=7, channels=64, dropout=0.1, static_dim=12,
                 d_model=64, n_heads=4, use_attention=True):
        super().__init__()
        self.enc = TCNEncoder(channels=channels, dropout=dropout)
        self.token_dim = self.enc.out_dim + static_dim
        self.proj = nn.Linear(self.token_dim, d_model)
        self.use_attention = use_attention
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = QuantileHead(d_model, n_quantiles)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, x_seq, u, m=None):
        # x_seq: (N, 18, L), u: (N, 12). One date = one batch -> tokens are the N entities.
        tok = self.proj(torch.cat([self.enc(x_seq), u], dim=1))   # (N, d_model)
        if self.use_attention:
            h = self.ln(tok).unsqueeze(0)                          # (1, N, d_model)
            a, _ = self.attn(h, h, h)
            tok = tok + a.squeeze(0)                               # residual
        return self.head(tok)                                     # (N, n_quantiles)

    def gate_logits(self, x_seq, u):
        tok = self.proj(torch.cat([self.enc(x_seq), u], dim=1))
        return self.gate(tok).squeeze(-1)


def build_f4(cfg=None):
    return F4Model()
