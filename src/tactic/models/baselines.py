"""Unconditional-quantile baseline + S1 sanity gate (revision_plan §4.1).

`baseline_pinball` scores the *train empirical quantiles* (one constant per tau) on an eval set.
A trained model that cannot beat this constant predictor by a relative margin has collapsed to the
unconditional marginal (the failure mode of the prior run). Gate S1 enforces that margin on the
validation set, catching the collapse BEFORE any OOS spend.
"""
from __future__ import annotations

import numpy as np

TAUS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def _pinball_const(q: float, y: np.ndarray, tau: float) -> float:
    e = y - q
    return float(np.maximum(tau * e, (tau - 1) * e).mean())


def baseline_pinball(y_train: np.ndarray, y_eval: np.ndarray, taus=TAUS) -> float:
    """Summed pinball of the constant train-empirical-quantile predictor on `y_eval`."""
    y_train = np.asarray(y_train, float)
    y_train = y_train[np.isfinite(y_train)]
    y_eval = np.asarray(y_eval, float)
    y_eval = y_eval[np.isfinite(y_eval)]
    if len(y_train) == 0 or len(y_eval) == 0:
        return float("nan")
    qs = np.quantile(y_train, taus)
    return float(sum(_pinball_const(q, y_eval, t) for q, t in zip(qs, taus)))


def s1_gate(model_val_pinball: float, baseline_val_pinball: float,
            min_rel: float = 0.01) -> dict:
    """Gate S1: model val pinball must beat the unconditional baseline by >= `min_rel` (relative).

    margin = (baseline - model) / baseline. PASS iff margin >= min_rel.
    """
    b = float(baseline_val_pinball)
    m = float(model_val_pinball)
    margin = (b - m) / b if b > 0 else float("nan")
    return {"pass": bool(np.isfinite(margin) and margin >= min_rel),
            "margin": float(margin), "min_rel": float(min_rel),
            "model_pinball": m, "baseline_pinball": b}
