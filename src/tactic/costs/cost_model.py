"""Per-crossing cost model (sec5.1): c_hat = 0.5*s_blend + c_impact*sqrt(trade$/ADV21$).

c_impact = 0.10 (config; sqrt-units of participation = 10 bps at 1% of ADV). The backtest
engine charges per executed trade; holding structures (OO/ON/ID) imply crossings/day.
"""
from __future__ import annotations

import numpy as np

CROSSINGS_PER_DAY = {"OO": 2, "ON": 2, "ID": 2}  # OO round-trip amortized by holding period


def c_hat(s_blend, trade_dollars, adv_dollars, c_impact: float = 0.10):
    """Cost as a fraction of traded notional. Scalars or arrays.

    s_blend is a fractional spread (0.0008 = 8 bps). participation = trade$/ADV21$.
    """
    s_blend = np.asarray(s_blend, float)
    participation = np.asarray(trade_dollars, float) / np.maximum(np.asarray(adv_dollars, float), 1.0)
    return 0.5 * s_blend + c_impact * np.sqrt(np.maximum(participation, 0.0))


def required_gross_daily_alpha(crossings_per_day: int, mean_c_hat_topk: float) -> float:
    """Break-even arithmetic (Appendix A.10)."""
    return crossings_per_day * mean_c_hat_topk


def structure_viable(component_decile_spread: float, crossings_per_day: int,
                     mean_c_hat_topk: float, capture: float = 0.30) -> bool:
    """Viable iff decile spread * capture (<=30%) > required gross alpha (Appendix A.10)."""
    return component_decile_spread * capture > required_gross_daily_alpha(
        crossings_per_day, mean_c_hat_topk
    )
