"""CPCV leakage/coverage + S1 gate + PBO + deep-panel isolation (revision_plan §8.1-8.2)."""
import numpy as np
import pytest

from tactic.validation.cpcv import cpcv_paths, combined_oos_coverage
from tactic.validation.pbo import pbo
from tactic.models.baselines import baseline_pinball, s1_gate


# ---------------- CPCV ----------------
def test_cpcv_fold_count_and_coverage():
    plan = cpcv_paths(2625, n_groups=8, k_test=2, purge=2, embargo=5)
    assert plan.n_folds == 28                      # C(8,2)
    assert plan.oos_count_per_group == 7           # C(7,1)
    cov = combined_oos_coverage(plan)
    assert cov.min() == cov.max() == 7             # every session OOS in exactly 7 folds


def test_cpcv_no_test_row_in_train():
    plan = cpcv_paths(800, n_groups=8, k_test=2, purge=2, embargo=5)
    for f in plan.folds:
        assert set(f.train_pos).isdisjoint(set(f.test_pos))


def test_cpcv_purge_embargo_gaps():
    purge, embargo = 2, 5
    plan = cpcv_paths(800, n_groups=8, k_test=2, purge=purge, embargo=embargo)
    for f in plan.folds:
        test = set(f.test_pos.tolist())
        train = set(f.train_pos.tolist())
        for p in train:
            # no train decision whose forward label (p+1..p+purge) overlaps test
            assert all((p + h) not in test for h in range(1, purge + 1))
            # no train within embargo positions after a test position
            assert all((p - h) not in test for h in range(1, embargo + 1))


def test_cpcv_run2_shape():
    plan = cpcv_paths(6500, n_groups=10, k_test=2, purge=2, embargo=5)
    assert plan.n_folds == 45                       # C(10,2)
    assert plan.oos_count_per_group == 9            # C(9,1)
    assert combined_oos_coverage(plan).min() == 9


# ---------------- S1 gate ----------------
def test_s1_constant_predictor_fails():
    rng = np.random.default_rng(0)
    y_tr = rng.normal(size=5000)
    y_va = rng.normal(size=2000)
    taus = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    base = baseline_pinball(y_tr, y_va, taus)
    # a constant predictor IS the baseline -> margin ~0 -> FAIL
    gate = s1_gate(base, base)
    assert not gate["pass"]
    assert abs(gate["margin"]) < 1e-9


def test_s1_better_model_passes():
    base = 1.0
    assert s1_gate(0.95, base)["pass"]              # 5% better -> pass
    assert not s1_gate(0.999, base)["pass"]         # 0.1% better -> fail


# ---------------- PBO ----------------
def test_pbo_single_config_nan():
    M = np.random.default_rng(0).normal(size=(40, 1))
    out = pbo(M)
    assert np.isnan(out["pbo"])


def test_pbo_range_and_overfit_signal():
    rng = np.random.default_rng(1)
    # pure-noise configs -> PBO should be near 0.5 (no real skill, IS-best is random OOS)
    M = rng.normal(size=(64, 8))
    out = pbo(M, S=8)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_configs"] == 8


# ---------------- Run-2 deep-panel isolation (§1B.3) ----------------
def test_deep_ingest_writes_raw_to_quarantine_only():
    from pathlib import Path
    import tactic.ingest.yfinance_deep as yd
    src = Path(yd.__file__).read_text(encoding="utf-8")
    # raw pulls go to the quarantine, panel to the (env-overridable) curated sink — never a
    # hard-coded data/curated path.
    assert "QUARANTINE / \"yfinance\"" in src
    assert "data/curated" not in src.replace("\\", "/")

