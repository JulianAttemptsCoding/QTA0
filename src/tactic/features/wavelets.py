"""MODWT energy features (sec6.2 channels 16-18, Appendix QA sec6.4).

Computed on the trailing window only with LEFT-only reflect padding so there is no future
contamination: shifting the input by one day shifts the output by one day. Energy at MODWT
detail levels 2,3,4 of the return series.
"""
from __future__ import annotations

import numpy as np
import pywt


def modwt_energy(series, levels=(2, 3, 4), wavelet: str = "db4"):
    """Return per-level detail energy (sum of squared MODWT detail coefficients).

    `series` is the trailing return window ending at the decision day (no future data).
    Uses `pywt.swt` (stationary/undecimated wavelet transform = MODWT) with left-reflect
    handling via manual padding to avoid right-edge (future) leakage.
    """
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    max_level = max(levels)
    n = len(x)
    if n < 2 ** max_level:
        return {lvl: 0.0 for lvl in levels}

    # pywt.swt requires length divisible by 2**level; left-pad by reflecting the EARLIEST
    # samples (past), never the future tail.
    need = (2 ** max_level) - (n % (2 ** max_level)) if n % (2 ** max_level) else 0
    if need:
        pad = x[:need][::-1]
        xp = np.concatenate([pad, x])
    else:
        xp = x

    coeffs = pywt.swt(xp, wavelet, level=max_level, trim_approx=False, norm=True)
    # coeffs: list of (cA, cD) from coarsest..finest depending on pywt version; index by level
    # pywt.swt returns level results coarsest-first; map to 1..max_level
    energies = {}
    L = len(coeffs)
    for lvl in levels:
        cD = coeffs[L - lvl][1]  # finest detail is last
        # drop the padded prefix region from energy so padding can't inflate it
        cD_valid = cD[need:] if need else cD
        energies[lvl] = float(np.sum(cD_valid ** 2))
    return energies
