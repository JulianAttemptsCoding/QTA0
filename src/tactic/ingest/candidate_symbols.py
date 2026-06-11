"""Candidate symbol list (§3.3): union of PIT S&P 500 + liquid ETFs + delisted-recovery set."""
from __future__ import annotations

import pandas as pd

from ..common.config import CURATED

# §3.3b fixed liquid-ETF list; leveraged/inverse explicitly excluded.
ETFS = ["SPY", "QQQ", "IWM", "DIA",
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


def build(include_inactive: bool = True) -> list[str]:
    syms: set[str] = set(ETFS)
    memb = CURATED / "sp500_membership.parquet"
    if memb.exists():
        syms |= set(pd.read_parquet(memb)["symbol"].unique())
    if include_inactive:
        snap = CURATED / "assets_snapshots.parquet"
        if snap.exists():
            df = pd.read_parquet(snap)
            inactive = df[df["status_query"] == "inactive"]["symbol"].dropna().unique()
            # §3.3c: inactive that ever matched (a) — intersect with seen members if available
            members = set(pd.read_parquet(memb)["symbol"].unique()) if memb.exists() else set()
            syms |= ({s for s in inactive if s in members} if members else set(inactive))
    return sorted(syms)
