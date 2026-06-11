"""Alpaca assets master (§3.2): active + inactive (inactive = delisted-symbol seed)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ..common.config import CURATED
from ..common.io import write_parquet
from . import ALPACA_TRADING_URL, alpaca_headers, get_json, raw_dir


def fetch_assets(status: str) -> pd.DataFrame:
    url = f"{ALPACA_TRADING_URL}/v2/assets"
    data = get_json(url, alpaca_headers(), {"status": status, "asset_class": "us_equity"})
    df = pd.DataFrame(data)
    if len(df):
        df["status_query"] = status
    return df


def ingest() -> pd.DataFrame:
    frames = [fetch_assets(s) for s in ("active", "inactive")]
    df = pd.concat(frames, ignore_index=True)
    snap = dt.date.today().isoformat()
    write_parquet(df, raw_dir("assets") / f"assets_{snap}.parquet")
    # monthly snapshot table
    df["snapshot_date"] = snap
    write_parquet(df, CURATED / "assets_snapshots.parquet")
    return df


def delisted_seed() -> list[str]:
    """Inactive-symbol seed for the delisted-recovery set (§3.3c)."""
    df = ingest() if not (CURATED / "assets_snapshots.parquet").exists() \
        else pd.read_parquet(CURATED / "assets_snapshots.parquet")
    return sorted(df[df["status_query"] == "inactive"]["symbol"].dropna().unique().tolist())


if __name__ == "__main__":
    print(ingest().shape)
