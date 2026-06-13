"""Probability of Backtest Overfitting via CSCV (de Prado sec15.1); G8 requires PBO < 0.20.

Combinatorially-Symmetric Cross-Validation: take a performance matrix M (n_observations rows x
n_configs columns; each column is one candidate strategy/config's per-period performance, e.g.
per-fold net returns). Partition the rows into S contiguous blocks; for every way of choosing S/2
blocks as the in-sample (IS) set and the complement as out-of-sample (OOS), pick the config that
is best IS and record its OOS rank. PBO = P(best-IS config lands in the bottom half OOS), estimated
from the distribution of logits w = ln(r/(1-r)) where r is the OOS relative rank.

Needs >= 2 configs to be meaningful; with a single config PBO is undefined (returns nan + note).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _eval(block_rows, M):
    # mean performance per config over the given rows
    return M[block_rows].mean(axis=0)


def pbo(perf_matrix, S: int = 16) -> dict:
    """Compute PBO via CSCV.

    perf_matrix: array-like (n_obs, n_configs). Returns dict(pbo, n_configs, S, n_splits, logits...).
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (n_obs, n_configs)")
    n_obs, n_cfg = M.shape
    if n_cfg < 2:
        return {"pbo": float("nan"), "n_configs": n_cfg, "S": 0, "n_splits": 0,
                "note": "PBO undefined for < 2 configs (need a config ensemble)"}

    # largest even S that does not exceed n_obs
    S = min(S, n_obs)
    if S % 2 == 1:
        S -= 1
    if S < 2:
        return {"pbo": float("nan"), "n_configs": n_cfg, "S": S, "n_splits": 0,
                "note": "too few observations for CSCV"}

    edges = np.linspace(0, n_obs, S + 1).round().astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]

    logits = []
    n_below = 0
    n_tot = 0
    for is_blocks in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_blocks])
        if len(is_rows) == 0 or len(oos_rows) == 0:
            continue
        is_perf = _eval(is_rows, M)
        oos_perf = _eval(oos_rows, M)
        n_star = int(np.argmax(is_perf))                  # best config in-sample
        # OOS relative rank of the IS-best config (1 = worst .. n_cfg = best)
        order = np.argsort(oos_perf)                      # ascending
        rank = int(np.flatnonzero(order == n_star)[0]) + 1
        r = rank / (n_cfg + 1)                            # relative rank in (0,1)
        w = np.log(r / (1.0 - r))
        logits.append(w)
        n_tot += 1
        if r < 0.5:
            n_below += 1

    logits = np.asarray(logits, dtype=float)
    pbo_val = float(n_below / n_tot) if n_tot else float("nan")
    return {"pbo": pbo_val, "n_configs": n_cfg, "S": S, "n_splits": n_tot,
            "logit_mean": float(logits.mean()) if n_tot else float("nan"),
            "logit_std": float(logits.std()) if n_tot else float("nan")}
