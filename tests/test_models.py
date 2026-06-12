import numpy as np
import torch

from tactic.common.config import load_config
from tactic.models.experts.hybrid_xs import F4Model
from tactic.models.experts.tcn_ci import F3Model
from tactic.models.train import TAUS, pinball_torch


def test_f4_param_count_under_cap():
    n = sum(p.numel() for p in F4Model().parameters())
    assert n <= load_config()["experts"]["param_cap"]  # 300k config cap
    assert n > 100_000  # non-trivial capacity


def test_forward_shapes_and_noncrossing():
    N, L = 12, 64
    x = torch.randn(N, 18, L)
    u = torch.randn(N, 12)
    for model in (F3Model(), F4Model()):
        out = model(x, u)
        assert out.shape == (N, len(TAUS))
        assert bool((out[:, 1:] >= out[:, :-1] - 1e-5).all())  # non-crossing


def test_f4_attention_identity_recovers_f3_shape():
    # f4 with attention off must still run and match f3 output shape (wiring ablation, sec10 QA)
    x = torch.randn(8, 18, 64); u = torch.randn(8, 12)
    out = F4Model(use_attention=False)(x, u)
    assert out.shape == (8, len(TAUS))


def test_pinball_torch_nonnegative():
    preds = torch.randn(10, 7)
    y = torch.randn(10)
    taus = torch.tensor(TAUS, dtype=torch.float32)
    assert float(pinball_torch(preds, y, taus)) >= 0
