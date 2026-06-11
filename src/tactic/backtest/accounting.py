"""Backtest accounting (sec13): explicit NAV identity with a conservation test.

ret_net = ret_gross - costs (to 1e-10). Accounting is in adjusted-return space (dividends
embedded); costs are subtracted explicitly; cash earns the RF (French series). Delisting
days realize terminal_return.
"""
from __future__ import annotations

import numpy as np


def gross_return(weights: np.ndarray, asset_returns: np.ndarray) -> float:
    """Portfolio gross return = w . r (weights as of decision, returns over the period)."""
    w = np.asarray(weights, float)
    r = np.asarray(asset_returns, float)
    return float(np.nansum(w * r))


def cost_drag(prev_w: np.ndarray, target_w: np.ndarray, c_hat_vec: np.ndarray) -> float:
    """Cost = sum_i |Δw_i| * c_hat_i (cost charged per executed dollar of turnover)."""
    dw = np.abs(np.asarray(target_w, float) - np.asarray(prev_w, float))
    return float(np.nansum(dw * np.asarray(c_hat_vec, float)))


def net_return(weights, asset_returns, prev_w, target_w, c_hat_vec,
               cash_weight: float = 0.0, rf: float = 0.0) -> dict:
    g = gross_return(weights, asset_returns)
    c = cost_drag(prev_w, target_w, c_hat_vec)
    cash = cash_weight * rf
    net = g + cash - c
    return {"gross": g, "cost": c, "cash": cash, "net": net}


def assert_conservation(gross: float, cost: float, net: float, cash: float = 0.0,
                        tol: float = 1e-10) -> None:
    """NAV identity check (sec13 QA): |net - (gross + cash - cost)| <= tol."""
    resid = abs(net - (gross + cash - cost))
    if resid > tol:
        raise AssertionError(f"NAV conservation violated: residual={resid:.3e} > {tol:.1e}")
