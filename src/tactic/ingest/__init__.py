"""Ingestion (Phase 1, §3): raw immutable copies -> typed curated tables.

Shared HTTP helpers live here: a retrying requests session and Alpaca auth headers.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from ..common.config import RAW, load_dotenv

load_dotenv()

ALPACA_DATA_URL = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"


def alpaca_headers() -> dict[str, str]:
    key = os.environ.get("APCA_API_KEY_ID", "")
    sec = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not sec:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (see .env.example)")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec, "accept": "application/json"}


def get_json(url: str, headers: dict, params: dict | None = None,
             max_retries: int = 6, base_sleep: float = 1.0) -> dict:
    """GET with exponential backoff on 429/5xx (§3.2)."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep = base_sleep * (2 ** attempt)
            time.sleep(min(sleep, 60))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"GET {url} failed after {max_retries} retries (last={resp.status_code})")


def raw_dir(name: str) -> Path:
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)
    return d
