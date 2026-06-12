"""Spread surface (sec5.1) on RAW prices: rolling 21d Corwin-Schultz (A.4), Abdi-Ranaldo,
EDGE (bidask). Blend = median of the three, floored at 2*fee_floor, capped, winsorized.

All estimators use only trailing data (rolling windows), so the surface is PIT-safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from bidask import edge_rolling

from ..common.config import CURATED, load_config
from ..common.io import write_parquet

_K = 3 - 2 * np.sqrt(2.0)  # Corwin-Schultz constant 3 - 2*sqrt(2)


def corwin_schultz(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Corwin-Schultz high-low spread (Appendix A.4). Negatives -> 0, then 21d mean."""
    h, l = np.log(high.to_numpy()), np.log(low.to_numpy())
    hl = (h - l) ** 2  # [ln(H/L)]^2 per day
    h2 = np.maximum(high.to_numpy()[1:], high.to_numpy()[:-1])
    l2 = np.minimum(low.to_numpy()[1:], low.to_numpy()[:-1])
    gamma = (np.log(h2) - np.log(l2)) ** 2          # 2-day high/low, length n-1
    beta = hl[1:] + hl[:-1]                          # sum over j=0,1, length n-1
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.where(s < 0, 0.0, s)
    s = np.concatenate([[np.nan], s])               # align to day t (pair t-1,t)
    return pd.Series(s, index=high.index).rolling(window, min_periods=window // 2).mean()


def abdi_ranaldo(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """Abdi-Ranaldo (2017): s = 2*sqrt(max(0, E[(c_t-m_t)(c_t-m_{t+1})])), 21d mean version."""
    c = np.log(close.to_numpy())
    m = (np.log(high.to_numpy()) + np.log(low.to_numpy())) / 2.0
    prod = (c[:-1] - m[:-1]) * (c[:-1] - m[1:])      # (c_t-m_t)(c_t-m_{t+1}), length n-1
    prod = np.concatenate([prod, [np.nan]])          # align to day t
    e = pd.Series(prod, index=close.index).rolling(window, min_periods=window // 2).mean()
    return 2 * np.sqrt(e.clip(lower=0))


def _entity_spreads(g: pd.DataFrame, window: int) -> pd.DataFrame:
    g = g.sort_values("date")
    out = pd.DataFrame({"entity": g["entity"].to_numpy(), "date": g["date"].to_numpy()})
    out["s_cs"] = corwin_schultz(g["h"], g["l"], window).to_numpy()
    out["s_ar"] = abdi_ranaldo(g["h"], g["l"], g["c"], window).to_numpy()
    edge_df = g[["o", "h", "l", "c"]].rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})
    try:
        out["s_edge"] = edge_rolling(edge_df, window=window).to_numpy()
    except Exception:
        out["s_edge"] = np.nan
    return out


def build_spread_surface(prices_raw: pd.DataFrame | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Build the blended spread surface. Reads curated raw prices if `prices_raw` is None.

    Output columns: entity, date, s_cs, s_ar, s_edge, s_blend (all fractional spreads).
    """
    cfg = cfg or load_config()
    win = 21
    fee_floor = cfg["costs"]["fee_floor_bps"] / 1e4
    cap = cfg["costs"]["spread_cap_bps"] / 1e4

    if prices_raw is None:
        df = pd.read_parquet(CURATED / "prices_daily.parquet")
        df = df[df["adjustment"] == "raw"].copy()
    else:
        df = prices_raw.copy()
    if "entity" not in df.columns:
        df["entity"] = df["symbol"]
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[(df["h"] > 0) & (df["l"] > 0) & (df["c"] > 0)]

    parts = [_entity_spreads(g, win) for _, g in df.groupby("entity", sort=False)]
    surf = pd.concat(parts, ignore_index=True)

    surf["s_blend"] = surf[["s_cs", "s_ar", "s_edge"]].median(axis=1, skipna=True)
    surf["s_blend"] = surf["s_blend"].clip(lower=2 * fee_floor, upper=cap)
    # per-day 99.5pct winsorize
    surf["date"] = pd.to_datetime(surf["date"]).dt.date
    caps = surf.groupby("date")["s_blend"].transform(lambda s: s.quantile(0.995))
    surf["s_blend"] = np.minimum(surf["s_blend"], caps)

    write_parquet(surf, CURATED / "spread_surface.parquet")
    return surf
