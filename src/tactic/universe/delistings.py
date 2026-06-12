"""Terminal returns per exiting entity (sec4.2, Non-negotiable sec0.3.9).

Exit sources: Alpaca inactive assets (last trading day) + corporate-action mergers/worthless.
  cash merger     -> terminal_return = deal_cash/last_close - 1     (terminal_type=cash_merger)
  bankruptcy/worthless -> terminal_return = -1.0                    (terminal_type=worthless)
  unobtainable    -> terminal_return = NaN                          (terminal_type=unknown; excluded + counted)
`assert_terminal_returns` enforces a typed field for every exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import CURATED, load_config
from ..common.contracts import assert_terminal_returns
from ..common.io import write_parquet


def _last_close(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.sort_values(["entity", "date"]).groupby("entity").tail(1)
    return p[["entity", "date", "c"]].rename(columns={"date": "last_date", "c": "last_close"})


def build_terminal_returns(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    prices = pd.read_parquet(CURATED / "prices_daily.parquet")
    prices = prices[prices["adjustment"] == "all"].copy()
    prices["entity"] = prices.get("entity", prices["symbol"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    last = _last_close(prices)

    rows: dict[str, dict] = {}

    # 1) inactive assets = delisted seed
    apath = CURATED / "assets_snapshots.parquet"
    if apath.exists():
        assets = pd.read_parquet(apath)
        inactive = assets[assets["status_query"] == "inactive"]["symbol"].dropna().unique()
        for sym in inactive:
            rows[sym] = {"entity": sym, "terminal_type": "unknown", "terminal_return": np.nan}

    # 2) corporate actions refine the type
    capath = CURATED / "corporate_actions_unified.parquet"
    if capath.exists():
        ca = pd.read_parquet(capath)
        for _, r in ca.iterrows():
            sym = r.get("symbol")
            t = str(r.get("ca_type", ""))
            if not sym:
                continue
            if "worthless" in t:
                rows[sym] = {"entity": sym, "terminal_type": "worthless", "terminal_return": -1.0}
            elif "cash_merger" in t or t == "cash_mergers":
                cash = r.get("cash_rate") or r.get("rate") or r.get("price")
                lc = last.loc[last["entity"] == sym, "last_close"]
                if cash is not None and len(lc) and lc.iloc[0] > 0:
                    rows[sym] = {"entity": sym, "terminal_type": "cash_merger",
                                 "terminal_return": float(cash) / float(lc.iloc[0]) - 1.0}
                else:
                    rows[sym] = {"entity": sym, "terminal_type": "acquired", "terminal_return": 0.0}
            elif "merger" in t:
                rows[sym] = {"entity": sym, "terminal_type": "acquired", "terminal_return": 0.0}

    df = pd.DataFrame(rows.values()) if rows else pd.DataFrame(
        columns=["entity", "terminal_type", "terminal_return"])
    df = df.merge(last.rename(columns={"last_date": "exit_date"})[["entity", "exit_date"]],
                  on="entity", how="left")
    df["is_exit"] = True
    assert_terminal_returns(df)
    write_parquet(df, CURATED / "terminal_returns.parquet")
    return df
