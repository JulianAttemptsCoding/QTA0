"""SEC EDGAR cross-check (§3.4): company_tickers.json -> ticker/CIK + symbol-change map.

Compliant User-Agent ("Name email") is required by SEC.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from ..common.config import CURATED, load_dotenv
from ..common.io import write_parquet
from . import raw_dir

load_dotenv()
_URL = "https://www.sec.gov/files/company_tickers.json"


def _headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError("SEC_USER_AGENT not set (format: 'Name email')")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def ingest() -> pd.DataFrame:
    resp = requests.get(_URL, headers=_headers(), timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    (raw_dir("edgar") / "company_tickers.json").write_bytes(resp.content)
    df = pd.DataFrame(raw.values())
    df = df.rename(columns={"cik_str": "cik", "ticker": "symbol", "title": "name"})
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    write_parquet(df, CURATED / "edgar_tickers.parquet")
    return df


if __name__ == "__main__":
    print(ingest().head())
