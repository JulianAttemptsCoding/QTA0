"""RUN 2 deep-history ingest (revision_plan §1B / B0). EXPLORATORY, SURVIVORSHIP-BIASED.

Pulls free yfinance daily history (2000-2026) for a fixed survivor pool, maps it to the repo bar
schema, and writes a SEPARATE panel under data/deep_panel/curated (env override TACTIC_DATA_DIR) so
the clean Run-1 curated panel is never touched. Raw pulls are quarantined under
data/diagnostic_quarantine/yfinance/.

Pool rule (deterministic, §1B.1): keep a name iff yfinance returns continuous daily data from
POOL_START through END with < 2% missing sessions and a first bar within 5 sessions of POOL_START.
This yields a *survivor* pool by construction — which is exactly why Run 2's benchmark is the pool
itself (bias cancels in the relative comparison), never SPY, never deployable.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import CURATED, QUARANTINE
from ..common.io import write_parquet
from ..ingest.candidate_symbols import LIQUID_BIG

POOL_START = "2000-01-01"
POOL_END = "2026-06-11"          # exclusive-ish end for yfinance; data through 2026-06-10
MAX_MISSING = 0.02


def _yf_symbol(sym: str) -> str:
    return sym.replace(".", "-")   # yfinance uses BRK-B, not BRK.B


def _flatten(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Normalize a yf.download frame (single ticker) to columns o,h,l,c,adj,v,date."""
    if isinstance(df.columns, pd.MultiIndex):
        # columns like (field, ticker) -> take the ticker level off
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in need):
        return pd.DataFrame()
    adj = cols.get("adj close", cols.get("adjclose", cols["close"]))
    out = pd.DataFrame({
        "date": pd.to_datetime(df.index).date,
        "o": df[cols["open"]].to_numpy(float),
        "h": df[cols["high"]].to_numpy(float),
        "l": df[cols["low"]].to_numpy(float),
        "c": df[cols["close"]].to_numpy(float),
        "adj": df[adj].to_numpy(float),
        "v": df[cols["volume"]].to_numpy(float),
    })
    out["symbol"] = sym
    return out.dropna(subset=["o", "h", "l", "c"]).reset_index(drop=True)


def fetch_one(sym: str, start=POOL_START, end=POOL_END) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(_yf_symbol(sym), start=start, end=end, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    return _flatten(raw, sym)


def _to_bar_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Build both 'all' (Adj-Close total-return adjusted OHLC) and 'raw' (unadjusted) bar sets."""
    df = df.sort_values("date").reset_index(drop=True)
    ratio = (df["adj"] / df["c"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # yfinance has no trade-count; use volume as a finite proxy so feature ch05 (trade-count z)
    # is not all-NaN (which would void every L=64 window). Exploratory Run 2 degradation, documented.
    base = {"symbol": df["symbol"], "date": df["date"], "v": df["v"],
            "n": df["v"], "t": pd.NaT, "feed": "yfinance"}
    allset = pd.DataFrame({**base,
                           "o": df["o"] * ratio, "h": df["h"] * ratio, "l": df["l"] * ratio,
                           "c": df["adj"], "vw": df["adj"], "adjustment": "all"})
    rawset = pd.DataFrame({**base,
                           "o": df["o"], "h": df["h"], "l": df["l"], "c": df["c"],
                           "vw": df["c"], "adjustment": "raw"})
    return pd.concat([allset, rawset], ignore_index=True)


def ingest_deep(candidates=None, start=POOL_START, end=POOL_END, max_missing=MAX_MISSING,
                pool_csv: str | Path | None = None) -> dict:
    """Fetch, filter to the survivor pool, write deep_panel prices + raw quarantine + pool.csv.

    Returns dict(pool, dropped, n_sessions, prices_path).
    """
    candidates = candidates or LIQUID_BIG
    qdir = QUARANTINE / "yfinance"
    qdir.mkdir(parents=True, exist_ok=True)
    start_d = dt.date.fromisoformat(start)

    raws: dict[str, pd.DataFrame] = {}
    for sym in candidates:
        df = fetch_one(sym, start, end)
        if df.empty:
            continue
        df.to_parquet(qdir / f"{sym.replace('.', '_')}.parquet", index=False)
        raws[sym] = df
        print(f"[yf] {sym:6s} rows={len(df):5d} first={df['date'].min()} last={df['date'].max()}")

    if not raws:
        raise RuntimeError("yfinance returned no data for any candidate")

    # session calendar = the candidate with the most rows (a long-lived survivor / SPY)
    ref_sym = max(raws, key=lambda s: len(raws[s]))
    ref_dates = set(d for d in raws[ref_sym]["date"] if d >= start_d)
    n_sessions = len(ref_dates)
    print(f"[yf] session calendar from {ref_sym}: {n_sessions} sessions since {start}")

    pool, dropped = [], []
    for sym, df in raws.items():
        d = df[df["date"] >= start_d]
        first = d["date"].min()
        present = len(set(d["date"]) & ref_dates)
        frac = present / max(n_sessions, 1)
        # within 5 sessions of POOL_START?
        starts_early = first <= sorted(ref_dates)[5] if n_sessions > 5 else True
        if frac >= (1 - max_missing) and starts_early:
            pool.append({"symbol": sym, "first_date": str(first), "rows": int(len(d)),
                         "coverage": round(frac, 4)})
        else:
            dropped.append({"symbol": sym, "first_date": str(first), "coverage": round(frac, 4),
                            "reason": "late_start" if not starts_early else "too_sparse"})

    pool_syms = [p["symbol"] for p in pool]
    print(f"[yf] pool size={len(pool_syms)} dropped={len(dropped)}")

    # write deep panel prices (both adjustments) for pool only
    parts = [_to_bar_schema(raws[s]) for s in pool_syms]
    prices = pd.concat(parts, ignore_index=True)
    prices = prices[prices["date"] >= start_d].reset_index(drop=True)
    write_parquet(prices, CURATED / "prices_daily.parquet")

    if pool_csv:
        Path(pool_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pool).to_csv(pool_csv, index=False)

    return {"pool": pool, "dropped": dropped, "n_sessions": n_sessions,
            "prices_path": str(CURATED / "prices_daily.parquet"), "ref_symbol": ref_sym}


if __name__ == "__main__":
    import os
    assert os.environ.get("TACTIC_DATA_DIR"), "set TACTIC_DATA_DIR=data/deep_panel first"
    r = ingest_deep(pool_csv=Path("results/run2_pool_tmp.csv"))
    print("pool", len(r["pool"]), "dropped", len(r["dropped"]))
