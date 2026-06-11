"""Deflated Sharpe Ratio (Appendix A.9). N is machine-read from the trial ledger.

DSR = Phi( ((SR - SR0)*sqrt(T-1)) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2) )
SR0 = sqrt(Var[SR_trials]) * ((1-gE)*Phi^-1(1-1/N) + gE*Phi^-1(1-1/(N*e)))
gE = 0.5772 (Euler-Mascheroni); g3,g4 = skew/kurtosis of daily net returns.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..common.registry import current_N

_GAMMA_E = 0.5772156649


def expected_max_sr(var_sr_trials: float, n_trials: int) -> float:
    """SR0: expected maximum Sharpe across `n_trials` under the null (de Prado)."""
    if n_trials < 2:
        return 0.0
    e = np.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(max(var_sr_trials, 0.0)) * ((1 - _GAMMA_E) * z1 + _GAMMA_E * z2))


def deflated_sharpe(
    returns: np.ndarray,
    var_sr_trials: float,
    n_trials: int | None = None,
    sr_benchmark: float | None = None,
) -> dict:
    """Return DSR and its inputs. `returns` are periodic (e.g. daily) net returns.

    If `n_trials` is None it is read from the ledger via registry.current_N().
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3:
        raise ValueError("need >=3 returns for DSR")
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd if sd > 0 else 0.0
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))  # non-excess kurtosis
    if n_trials is None:
        n_trials = max(current_N(), 1)
    sr0 = sr_benchmark if sr_benchmark is not None else expected_max_sr(var_sr_trials, n_trials)
    denom = np.sqrt(max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    dsr = float(stats.norm.cdf(z))
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "N": n_trials, "T": T,
            "skew": g3, "kurtosis": g4, "z": float(z)}
