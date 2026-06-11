"""Alpaca daily bars (§3.2): feed=sip, 1Day, paginated, raw + all adjustments.

Stores both adjustment variants. Resumes from a manifest so a re-run only fetches missing
(symbol, adjustment) pairs. Minute bars are fetched lazily in Phase 4 (not here).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..common.config import CURATED, load_config
from ..common.io import write_parquet
from . import ALPACA_DATA_URL, alpaca_headers, get_json, raw_dir

_BARS_FIELDS = {"t": "t", "o": "o", "h": "h", "l": "l", "c": "c", "v": "v", "n": "n", "vw": "vw"}


def _manifest_path() -> Path:
    return raw_dir("bars") / "manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_manifest(m: dict) -> None:
    _manifest_path().write_text(json.dumps(m, indent=2), encoding="utf-8")


def fetch_symbol_bars(symbol: str, start: str, end: str, adjustment: str,
                      feed: str = "sip", timeframe: str = "1Day") -> pd.DataFrame:
    """Fetch all daily bars for one symbol, following pagination tokens."""
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
    headers = alpaca_headers()
    rows: list[dict] = []
    page_token = None
    while True:
        params = {"timeframe": timeframe, "start": start, "end": end, "feed": feed,
                  "adjustment": adjustment, "limit": 10000}
        if page_token:
            params["page_token"] = page_token
        data = get_json(url, headers, params)
        for b in data.get("bars") or []:
            rows.append({"symbol": symbol, **{k: b.get(k) for k in _BARS_FIELDS}})
        page_token = data.get("next_page_token")
        if not page_token:
            break
    if not rows:
        return pd.DataFrame(columns=["symbol", *_BARS_FIELDS])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"]).dt.date
    df["adjustment"] = adjustment
    df["feed"] = feed
    return df


def ingest(symbols: list[str], start: str | None = None, end: str | None = None,
           adjustments: tuple[str, ...] = ("raw", "all")) -> Path:
    """Fetch bars for `symbols` x `adjustments`, write raw per-symbol parquet + curated union.

    Returns the curated prices_daily directory.
    """
    cfg = load_config()
    start = start or str(cfg["panel"]["start"])
    end = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    manifest = _load_manifest()
    bars_raw = raw_dir("bars")

    all_frames: list[pd.DataFrame] = []
    for sym in symbols:
        for adj in adjustments:
            key = f"{sym}:{adj}"
            out = bars_raw / f"{sym}_{adj}.parquet"
            if manifest.get(key) == "ok" and out.exists():
                all_frames.append(pd.read_parquet(out))
                continue
            df = fetch_symbol_bars(sym, start, end, adj)
            if len(df):
                write_parquet(df, out)
                all_frames.append(df)
                manifest[key] = "ok"
            else:
                manifest[key] = "no_data"
            _save_manifest(manifest)
    _save_manifest(manifest)

    if not all_frames:
        raise RuntimeError("no bars fetched for any symbol")
    union = pd.concat(all_frames, ignore_index=True)
    union = union.drop_duplicates(subset=["symbol", "date", "adjustment"]).sort_values(
        ["symbol", "date", "adjustment"]
    )
    out_dir = CURATED / "prices_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(union, out_dir / "prices_daily.parquet")
    return out_dir


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ingest Alpaca daily bars (SIP).")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    out = ingest(args.symbols, args.start, args.end)
    print(f"wrote curated bars -> {out}")


if __name__ == "__main__":
    _cli()
