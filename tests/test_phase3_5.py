import numpy as np
import pandas as pd
import pytest

from tactic.costs.spreads import abdi_ranaldo, corwin_schultz
from tactic.diagnostics.ic_decay import nw_tstat
from tactic.diagnostics.tugofwar_table import tugofwar_table
from tactic.diagnostics.variance_ratio import variance_ratio
from tactic.labels.build_labels import _entity_labels


def _rw_ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = 0.002
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)) + spread / 2)
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)) - spread / 2)
    return pd.Series(high), pd.Series(low), pd.Series(close)


def test_corwin_schultz_nonnegative():
    h, l, c = _rw_ohlc()
    s = corwin_schultz(h, l)
    assert (s.dropna() >= 0).all()
    assert s.dropna().mean() > 0


def test_abdi_ranaldo_finite():
    h, l, c = _rw_ohlc()
    s = abdi_ranaldo(h, l, c)
    assert np.isfinite(s.dropna()).all()


def test_vr_random_walk_near_one():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 4000)
    out = variance_ratio(r, 5)
    assert out["vr"] == pytest.approx(1.0, abs=0.12)
    assert out["p_value"] > 0.05  # do not reject RW


def test_vr_positive_autocorr_above_one():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.01, 4000)
    r = np.zeros_like(e)
    for t in range(1, len(e)):
        r[t] = 0.4 * r[t - 1] + e[t]  # positive autocorrelation -> VR>1
    out = variance_ratio(r, 5)
    assert out["vr"] > 1.1
    assert out["p_value"] < 0.05


def test_labels_component_identity():
    n = 50
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    g = pd.DataFrame({
        "entity": "AAA", "date": pd.bdate_range("2020-01-01", periods=n).date,
        "a_open": px * (1 + rng.normal(0, 0.001, n)),
        "a_close": px,
        "label_source": "auction",
    })
    lab = _entity_labels(g, lam=0.94, floor=0.002)
    # r_cc == r_on + r_in
    np.testing.assert_allclose(lab["r_cc"], lab["r_on"] + lab["r_in"], atol=1e-12, equal_nan=True)
    # r_oo[d] == r_in[d-1] + r_on[d]
    recon = lab["r_in"].shift(1) + lab["r_on"]
    np.testing.assert_allclose(lab["r_oo"].iloc[2:], recon.iloc[2:], atol=1e-12)
    assert (lab["sigma"] >= 0.002).all()


def test_nw_tstat_zero_for_zero_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 500)
    assert abs(nw_tstat(x)) < 3


def test_tugofwar_runs():
    rng = np.random.default_rng(5)
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=120).date
    for ent in [f"E{i}" for i in range(40)]:
        r_on = rng.normal(0, 0.01, len(dates))
        r_in = rng.normal(0, 0.01, len(dates))
        rows.append(pd.DataFrame({"entity": ent, "date": dates, "r_on": r_on,
                                  "r_in": r_in, "r_cc": r_on + r_in}))
    labels = pd.concat(rows, ignore_index=True)
    tab = tugofwar_table(labels)
    assert {"sort_leg", "decile", "next_on", "next_in"}.issubset(tab.columns)
    assert len(tab) > 0
