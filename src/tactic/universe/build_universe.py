"""PIT tradable universe (sec4.2). tradable iff: (PIT S&P-500 member OR config ETF) AND
price>$5 AND ADV21>$5M AND blended spread<40bps AND listed>=126d AND status active.

Output universe_membership(entity, date, tradable, reason_codes). Target width 400-700.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import CURATED, load_config
from ..common.io import write_parquet
from ..ingest.candidate_symbols import ETFS


def build_universe(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    p = cfg["panel"]
    price_min, adv_min = p["price_min"], p["adv_min_usd"]
    spread_max = p["spread_max_bps"] / 1e4
    min_age = p["min_age_days"]

    prices = pd.read_parquet(CURATED / "prices_daily.parquet")
    prices = prices[prices["adjustment"] == "all"].copy()
    prices["entity"] = prices.get("entity", prices["symbol"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["entity", "date"])

    # ADV21 ($) and listing age (sessions since first bar)
    prices["dollar_vol"] = prices["c"] * prices["v"]
    prices["adv21"] = prices.groupby("entity")["dollar_vol"].transform(
        lambda s: s.rolling(21, min_periods=10).mean())
    prices["age"] = prices.groupby("entity").cumcount()

    df = prices[["entity", "date", "c", "adv21", "age"]].copy()

    # membership: PIT S&P 500 OR ETF
    member = pd.Series(False, index=df.index)
    mpath = CURATED / "sp500_membership.parquet"
    if mpath.exists():
        memb = pd.read_parquet(mpath)
        memb["date"] = pd.to_datetime(memb["date"]).dt.date
        memb_set = set(map(tuple, memb[["symbol", "date"]].to_numpy()))
        member = df.apply(lambda r: (r["entity"], r["date"]) in memb_set, axis=1)
    is_etf = df["entity"].isin(ETFS)
    eligible_membership = member | is_etf

    # spread
    spath = CURATED / "spread_surface.parquet"
    if spath.exists():
        surf = pd.read_parquet(spath)
        surf["date"] = pd.to_datetime(surf["date"]).dt.date
        df = df.merge(surf[["entity", "date", "s_blend"]], on=["entity", "date"], how="left")
    else:
        df["s_blend"] = 0.0

    cond_price = df["c"] > price_min
    cond_adv = df["adv21"] > adv_min
    cond_spread = df["s_blend"].fillna(1.0) < spread_max
    cond_age = df["age"] >= min_age
    df["tradable"] = (eligible_membership.to_numpy() & cond_price & cond_adv
                      & cond_spread & cond_age)

    def reasons(row):
        r = []
        if not (row["c"] > price_min): r.append("price")
        if not (row["adv21"] > adv_min): r.append("adv")
        if not (row["s_blend"] < spread_max if pd.notna(row["s_blend"]) else False): r.append("spread")
        if not (row["age"] >= min_age): r.append("age")
        return ",".join(r) if r else "ok"

    df["reason_codes"] = df.apply(reasons, axis=1)
    out = df[["entity", "date", "tradable", "reason_codes", "adv21", "s_blend"]]
    write_parquet(out, CURATED / "universe_membership.parquet")
    return out
