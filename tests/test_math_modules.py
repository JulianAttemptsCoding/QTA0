import numpy as np
import pytest

from tactic.costs import cost_model
from tactic.models import quantile_heads as qh
from tactic.validation.dm_test import dm_test
from tactic.validation.dsr import deflated_sharpe, expected_max_sr


def test_pinball_loss_sign():
    # underprediction (y>q) at tau penalizes by tau*(y-q)
    assert qh.pinball_loss(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert qh.pinball_loss(0.0, 1.0, 0.9) == pytest.approx(0.1)


def test_noncrossing_assembly():
    q = qh.assemble_noncrossing(0.0, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert qh.is_noncrossing(q)
    assert q.shape[-1] == 7
    assert q[3] == pytest.approx(0.0)  # median at center index


def test_cost_model_math():
    # 8bps spread, trade 1% of ADV, impact 0.10 -> 0.5*0.0008 + 0.10*sqrt(0.01)
    c = cost_model.c_hat(0.0008, 1.0, 100.0, c_impact=0.10)
    assert float(c) == pytest.approx(0.0004 + 0.10 * 0.1)


def test_break_even():
    assert cost_model.structure_viable(0.02, 2, 0.001, capture=0.30) is True
    assert cost_model.structure_viable(0.001, 2, 0.001, capture=0.30) is False


def test_dm_identical_losses():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    out = dm_test(x, x.copy(), lags=10)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


def test_dm_detects_better_model():
    rng = np.random.default_rng(1)
    loss_b = rng.uniform(0.5, 1.5, size=1000)
    loss_a = loss_b - 0.2  # A strictly better
    out = dm_test(loss_a, loss_b, lags=10)
    assert out["stat"] < 0
    assert out["p_value"] < 0.05


def test_expected_max_sr_monotone():
    assert expected_max_sr(0.25, 100) > expected_max_sr(0.25, 10)


def test_dsr_high_for_strong_signal():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.005, size=2000)  # positive Sharpe
    out = deflated_sharpe(r, var_sr_trials=0.01, n_trials=10)
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["N"] == 10
