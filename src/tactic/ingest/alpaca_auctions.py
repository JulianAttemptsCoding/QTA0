"""Alpaca historical auctions (§3.2): official open/close auction prints per (symbol, day).

Output schema: auctions_daily(symbol, date, open_auction_px, open_auction_vol,
open_auction_ts, close_auction_px, close_auction_vol, close_auction_ts, source_flag).
Where multiple prints exist, prefer the official (condition-coded) print, else last print.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ..common.config import CURATED, load_config
from ..common.io import write_parquet
from . import ALPACA_DATA_URL, alpaca_headers, get_json, raw_dir

# Alpaca auction condition codes for official opening/closing prints.
_OFFICIAL_OPEN = {"Q", "O"}
_OFFICIAL_CLOSE = {"M", "6"}


def _pick(prints: list[dict], official: set[str], px_key: str = "p") -> dict | None:
    if not prints:
        return None
    for pr in prints:
        if pr.get("c") in official or (pr.get("cc") and set(pr["cc"]) & official):
            return {"px": pr.get(px_key), "vol": pr.get("s"), "ts": pr.get("t"), "flag": "official"}
    last = prints[-1]
    return {"px": last.get(px_key), "vol": last.get("s"), "ts": last.get("t"), "flag": "last_print"}


def fetch_symbol_auctions(symbol: str, start: str, end: str, feed: str = "sip") -> pd.DataFrame:
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/auctions"
    headers = alpaca_headers()
    rows: list[dict] = []
    page_token = None
    while True:
        params = {"start": start, "end": end, "feed": feed, "limit": 10000}
        if page_token:
            params["page_token"] = page_token
        data = get_json(url, headers, params)
        for a in data.get("auctions") or []:
            o = _pick(a.get("o") or [], _OFFICIAL_OPEN)
            c = _pick(a.get("c") or [], _OFFICIAL_CLOSE)
            rows.append({
                "symbol": symbol,
                "date": pd.to_datetime(a.get("d")).date() if a.get("d") else None,
                "open_auction_px": o["px"] if o else None,
                "open_auction_vol": o["vol"] if o else None,
                "open_auction_ts": o["ts"] if o else None,
                "close_auction_px": c["px"] if c else None,
                "close_auction_vol": c["vol"] if c else None,
                "close_auction_ts": c["ts"] if c else None,
                "source_flag": f"{o['flag'] if o else 'na'}/{c['flag'] if c else 'na'}",
            })
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return pd.DataFrame(rows)


def ingest(symbols: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    start = start or str(cfg["panel"]["start"])
    end = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    frames = []
    for sym in symbols:
        df = fetch_symbol_auctions(sym, start, end)
        if len(df):
            write_parquet(df, raw_dir("auctions") / f"{sym}.parquet")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    union = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"])
    write_parquet(union, CURATED / "auctions_daily.parquet")
    return union
