"""Lo-MacKinlay variance-ratio test (sec7.1), heteroskedasticity-robust statistic z*(q).

VR(q) = sigma_b^2(q) / sigma_a^2 with overlapping estimator; under the random walk null
VR(q)=1. z*(q) ~ N(0,1) using the heteroskedasticity-consistent variance theta*(q).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def variance_ratio(returns, q: int) -> dict:
    """VR(q) and the robust z-statistic / two-sided p-value for one q."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < q + 1:
        return {"q": q, "vr": np.nan, "z": np.nan, "p_value": np.nan}
    mu = r.mean()
    # sigma_a^2: 1-period variance (unbiased)
    sig_a = np.sum((r - mu) ** 2) / (n - 1)
    # sigma_b^2: q-period overlapping variance
    cum = np.cumsum(r)
    q_ret = cum[q - 1:] - np.concatenate([[0.0], cum[:-q]])  # rolling q-sums, length n-q+1
    m = q * (n - q + 1) * (1 - q / n)
    sig_b = np.sum((q_ret - q * mu) ** 2) / m
    vr = sig_b / sig_a if sig_a > 0 else np.nan

    # heteroskedasticity-robust theta*(q)
    denom = np.sum((r - mu) ** 2) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum(((r[j:] - mu) ** 2) * ((r[:-j] - mu) ** 2))
        delta = num / denom if denom > 0 else 0.0  # Lo-MacKinlay delta_j (no n factor)
        theta += ((2 * (q - j) / q) ** 2) * delta
    z = (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return {"q": q, "vr": float(vr), "z": float(z), "p_value": float(p)}


def variance_ratios(returns, qs=(2, 5, 10, 20)) -> list[dict]:
    return [variance_ratio(returns, q) for q in qs]
