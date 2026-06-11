"""Risk-constrained Kelly sizing (sec12.6, cvxpy): max E[log wealth] s.t. P(DD>30pct)<=5pct via lambda_K."""
from __future__ import annotations


def rc_kelly_weights(scenarios, probs, cfg=None):
    raise NotImplementedError("PLAN.md scaffold stub - implement this phase. See module docstring.")
