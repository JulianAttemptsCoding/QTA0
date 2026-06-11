"""Diebold-Mariano test on daily loss differentials (sec8), Newey-West HAC, default lag 10.

For forecasts use pinball loss differentials; for strategies use net-P&L differentials.
Harvey-Leybourne-Newbold small-sample correction applied.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _nw_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    gamma0 = float(d @ d) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        cov = float(d[k:] @ d[:-k]) / n
        var += 2.0 * w * cov
    return var


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 10,
            hln_correction: bool = True) -> dict:
    """Test H0: E[loss_a - loss_b] = 0. Negative stat => model A has lower loss (better).

    Returns dict(stat, p_value, mean_diff). Two-sided p by Student-t (HLN) or normal.
    """
    la = np.asarray(loss_a, float)
    lb = np.asarray(loss_b, float)
    if la.shape != lb.shape:
        raise ValueError("loss series must be same length")
    d = la - lb
    d = d[np.isfinite(d)]
    n = len(d)
    if n < lags + 2:
        raise ValueError(f"need > {lags+1} obs")
    var = _nw_var(d, lags)
    if var <= 0:
        return {"stat": 0.0, "p_value": 1.0, "mean_diff": float(d.mean())}
    stat = d.mean() / np.sqrt(var / n)
    if hln_correction:
        h = lags + 1
        corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= corr
        p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"stat": float(stat), "p_value": float(p), "mean_diff": float(d.mean())}
