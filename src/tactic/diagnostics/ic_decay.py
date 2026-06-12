"""IC and IC-decay diagnostic (sec7.2).

Daily cross-sectional Spearman IC of each feature vs each component at horizons h=1..20;
Newey-West t-stat of the mean IC; exponential decay fit rho(h)=rho1*exp(-phi*h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def nw_tstat(x: np.ndarray, lags: int = 10) -> float:
    """Newey-West HAC t-stat for the mean of a (serially correlated) series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    d = x - x.mean()
    var = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        var += 2.0 * w * float(d[k:] @ d[:-k]) / n
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else np.nan


def _daily_ic(feat: pd.DataFrame, lab: pd.DataFrame, fcol: str, ccol: str, h: int) -> np.ndarray:
    """Spearman IC per date between feat[fcol](t) and lab[ccol] shifted by +h within entity."""
    lab = lab.sort_values(["entity", "date"]).copy()
    lab["fwd"] = lab.groupby("entity")[ccol].shift(-h)
    m = feat[["entity", "date", fcol]].merge(
        lab[["entity", "date", "fwd"]], on=["entity", "date"], how="inner").dropna()
    ics = []
    for _, g in m.groupby("date"):
        if len(g) >= 5:
            ics.append(g[fcol].corr(g["fwd"], method="spearman"))
    return np.array([x for x in ics if np.isfinite(x)], float)


def ic_decay(features: pd.DataFrame, labels: pd.DataFrame,
             feature_cols: list[str] | None = None,
             components=("r_on", "r_in", "r_cc"),
             horizons=range(1, 21), nw_lags: int = 10) -> pd.DataFrame:
    feature_cols = feature_cols or [c for c in features.columns if c not in ("entity", "date")]
    rows = []
    for fcol in feature_cols:
        for ccol in components:
            ic_by_h = {}
            for h in horizons:
                ics = _daily_ic(features, labels, fcol, ccol, h)
                mean_ic = float(np.mean(ics)) if len(ics) else np.nan
                ic_by_h[h] = mean_ic
                rows.append({"feature": fcol, "component": ccol, "h": h,
                             "ic_mean": mean_ic, "t_nw": nw_tstat(ics, nw_lags),
                             "n_days": len(ics)})
            # decay fit on |ic| over h
            hs = np.array(list(ic_by_h.keys()), float)
            ys = np.array([ic_by_h[h] for h in ic_by_h], float)
            ok = np.isfinite(ys)
            if ok.sum() >= 3:
                try:
                    (rho1, phi), _ = curve_fit(lambda h, a, b: a * np.exp(-b * h),
                                               hs[ok], ys[ok], p0=[ys[ok][0] or 0.01, 0.1],
                                               maxfev=5000)
                    for r in rows:
                        if r["feature"] == fcol and r["component"] == ccol:
                            r["rho1"], r["phi"] = float(rho1), float(phi)
                except Exception:
                    pass
    return pd.DataFrame(rows)
