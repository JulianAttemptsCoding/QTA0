"""Non-crossing quantile heads + pinball loss (sec10.2, Appendix A.2).

Network predicts the median + softplus increments so quantiles are non-crossing by
construction. The single training objective is summed pinball loss on the quantile grid
(no auxiliary terms, sec0.3.5).
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y, q, tau):
    """L_tau(y,q) = max(tau*(y-q), (tau-1)*(y-q)). Elementwise; returns array."""
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    tau = np.asarray(tau, float)
    e = y - q
    return np.maximum(tau * e, (tau - 1.0) * e)


def total_pinball(y, q_grid, taus) -> float:
    """Sum over the quantile grid (Appendix A.2). q_grid shape (..., n_tau)."""
    q_grid = np.asarray(q_grid, float)
    taus = np.asarray(taus, float)
    y = np.asarray(y, float)[..., None]
    return float(pinball_loss(y, q_grid, taus).sum())


def assemble_noncrossing(median, increments):
    """Build a non-crossing quantile vector from a median and nonneg softplus increments.

    `increments` are the positive gaps between adjacent quantiles, split half below /
    half above the median index. Returns an ascending quantile array.
    """
    median = np.asarray(median, float)
    inc = np.abs(np.asarray(increments, float))  # softplus output is already >=0
    n = inc.shape[-1] + 1
    mid = n // 2
    out = np.empty(inc.shape[:-1] + (n,), float)
    out[..., mid] = median
    for j in range(mid + 1, n):
        out[..., j] = out[..., j - 1] + inc[..., j - 1]
    for j in range(mid - 1, -1, -1):
        out[..., j] = out[..., j + 1] - inc[..., j]
    return out


def is_noncrossing(q_grid) -> bool:
    q = np.asarray(q_grid, float)
    return bool(np.all(np.diff(q, axis=-1) >= -1e-9))
