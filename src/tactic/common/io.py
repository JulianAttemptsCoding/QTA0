"""IO layer: parquet read/write with hash sidecars + the holdout lock (§0.3.7).

The holdout (`data/holdout/`, final 18 months) is written once at ingestion and is
read-protected by a `.LOCKED` sentinel until Phase 13 unlocks it after the pre-registration
hash is committed. Every curated loader routes through `load_curated`, which refuses any
requested date inside the holdout window while the sentinel exists.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .config import CURATED, HOLDOUT, HOLDOUT_LOCK
from .hashing import write_sidecar

_WINDOW_FILE = HOLDOUT / "window.json"


class HoldoutLocked(RuntimeError):
    pass


# --- holdout lock ---------------------------------------------------------------
def is_locked() -> bool:
    return HOLDOUT_LOCK.exists()


def set_holdout_window(start: dt.date, end: dt.date) -> None:
    """Record the holdout date window (inclusive). Written once at ingestion."""
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    _WINDOW_FILE.write_text(
        json.dumps({"start": start.isoformat(), "end": end.isoformat()}), encoding="utf-8"
    )


def holdout_window() -> tuple[dt.date, dt.date] | None:
    if not _WINDOW_FILE.exists():
        return None
    d = json.loads(_WINDOW_FILE.read_text(encoding="utf-8"))
    return dt.date.fromisoformat(d["start"]), dt.date.fromisoformat(d["end"])


def lock_holdout() -> None:
    HOLDOUT.mkdir(parents=True, exist_ok=True)
    HOLDOUT_LOCK.write_text(
        f"LOCKED {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        "Do not unlock except via the Phase-13 pre-registration protocol (PLAN.md §15).\n",
        encoding="utf-8",
    )


def unlock_holdout(prereg_committed: bool) -> None:
    """Remove the lock. Refuses unless the pre-registration commit flag is asserted."""
    if not prereg_committed:
        raise HoldoutLocked("refusing to unlock holdout: PREREG.md not committed (§15.5)")
    if HOLDOUT_LOCK.exists():
        HOLDOUT_LOCK.unlink()


def _overlaps_holdout(start: dt.date | None, end: dt.date | None) -> bool:
    win = holdout_window()
    if win is None:
        return False
    h0, h1 = win
    s = start or dt.date.min
    e = end or dt.date.max
    return not (e < h0 or s > h1)


# --- parquet io -----------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: str | Path, hash_sidecar: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if hash_sidecar:
        write_sidecar(path)
    return path


def load_curated(
    table: str,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    base: Path = CURATED,
    date_col: str = "date",
) -> pd.DataFrame:
    """Load a curated table, refusing holdout-window dates while the sentinel exists.

    `table` is a subdirectory (or file stem) under data/curated/. Dates are filtered on
    `date_col` if present.
    """
    if isinstance(start, str):
        start = dt.date.fromisoformat(start)
    if isinstance(end, str):
        end = dt.date.fromisoformat(end)
    if is_locked() and _overlaps_holdout(start, end):
        raise HoldoutLocked(
            f"requested range [{start}, {end}] overlaps the LOCKED holdout window "
            f"{holdout_window()}; unlock only via §15 protocol"
        )
    p = base / table
    if p.is_dir():
        df = pd.read_parquet(p)
    elif p.with_suffix(".parquet").exists():
        df = pd.read_parquet(p.with_suffix(".parquet"))
    else:
        raise FileNotFoundError(f"curated table not found: {p}")
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col]).dt.date
        if start is not None:
            df = df[d >= start]
        if end is not None:
            df = df[d <= end]
    return df.reset_index(drop=True)
