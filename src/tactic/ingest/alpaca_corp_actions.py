"""Alpaca corporate actions (§3.2): splits, dividends, mergers, spin-offs, symbol changes.

KNOWN LIMIT (handled, not hidden): coverage starts ~2020-04; older delistings/symbol-changes
are absent and are reconciled in Phase 2 (universe/delistings.py). Raw JSON is stored, then
normalized into corporate_actions_unified.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from ..common.config import CURATED
from ..common.io import write_parquet
from . import ALPACA_DATA_URL, alpaca_headers, get_json, raw_dir

# Alpaca request types are SINGULAR; response groups them under PLURAL keys.
_TYPES = [
    "forward_split", "reverse_split", "cash_dividend", "stock_dividend",
    "spin_off", "cash_merger", "stock_merger", "stock_and_cash_merger",
    "unit_split", "redemption", "name_change", "worthless_removal",
    "rights_distribution", "contract_adjustment", "partial_call", "reorganization",
]


def fetch(symbols: list[str], start: str, end: str) -> dict:
    url = f"{ALPACA_DATA_URL}/v1/corporate-actions"
    headers = alpaca_headers()
    out: dict[str, list] = {}
    page_token = None
    while True:
        params = {"symbols": ",".join(symbols), "types": ",".join(_TYPES),
                  "start": start, "end": end, "limit": 1000}
        if page_token:
            params["page_token"] = page_token
        data = get_json(url, headers, params)
        ca = data.get("corporate_actions", {})  # keys are plural response groups
        for key, items in ca.items():
            out.setdefault(key, []).extend(items or [])
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return out


def normalize(raw: dict) -> pd.DataFrame:
    rows = []
    for t, items in raw.items():
        for it in items:
            rows.append({"ca_type": t, **it})
    return pd.DataFrame(rows)


def ingest(symbols: list[str], start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    end = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    # chunk symbols to keep URLs sane
    merged: dict[str, list] = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        raw = fetch(chunk, start, end)
        for key, items in raw.items():
            merged.setdefault(key, []).extend(items)
    (raw_dir("corporate_actions") / "raw.json").write_text(json.dumps(merged), encoding="utf-8")
    df = normalize(merged)
    write_parquet(df, CURATED / "corporate_actions_unified.parquet")
    return df
