"""Stooq bulk pre-2016 daily bars (§3.4, OPTIONAL).

Writes ONLY to data/diagnostic_quarantine/. There is NO join path from here into curated
features — enforced by tests/test_quarantine_isolation.py. Diagnostics only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.config import QUARANTINE


def ingest_local_zip(zip_path: str | Path) -> Path:
    """Parse a locally-downloaded Stooq bulk zip into the quarantine zone.

    Stooq bulk data is downloaded manually from https://stooq.com/db/h/ (free, but
    rate-limited / captcha), so this consumes a local file rather than hitting the network.
    """
    import zipfile

    zip_path = Path(zip_path)
    out = QUARANTINE / "stooq_daily.parquet"
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            with z.open(name) as fh:
                try:
                    df = pd.read_csv(fh)
                except Exception:
                    continue
            df["symbol"] = Path(name).stem.upper()
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no parseable files in {zip_path}")
    union = pd.concat(frames, ignore_index=True)
    union.to_parquet(out, index=False)
    return out
