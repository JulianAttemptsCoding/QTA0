"""Trial ledger — the 'honest N' machinery (§0.3.8, §2.4, Appendix D).

Every design decision, hyperparameter, threshold, expert, and structure choice that is
evaluated against data appends a row here. `current_N()` feeds the Deflated Sharpe Ratio.
The ledger is APPEND-ONLY: the writer refuses any write that would drop or mutate a row
that already exists (enforced by trial_id prefix check).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import LEDGER_PATH

Category = Literal["design", "hpo", "threshold", "expert", "structure"]
_VALID = {"design", "hpo", "threshold", "expert", "structure"}

_COLUMNS = [
    "trial_id", "ts_utc", "category", "description",
    "config_hash", "study_id", "charged_to_N",
]


class LedgerError(RuntimeError):
    pass


def _read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_COLUMNS)


def _assert_append_only(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """`new` must contain `old` as an unchanged prefix (same trial_ids, same content)."""
    if len(new) < len(old):
        raise LedgerError("ledger write would shrink the ledger (append-only violation)")
    prefix = new.iloc[: len(old)].reset_index(drop=True)
    if not prefix[_COLUMNS].equals(old[_COLUMNS].reset_index(drop=True)):
        raise LedgerError("ledger write would mutate existing rows (append-only violation)")


def log_trial(
    category: Category,
    description: str,
    config_hash: str,
    study_id: str | None = None,
    charged_to_N: bool = True,
    path: str | Path = LEDGER_PATH,
) -> int:
    """Append a trial row; return its integer trial_id. Thread-safety is process-level only."""
    if category not in _VALID:
        raise ValueError(f"category must be one of {_VALID}, got {category!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _read(path)
    trial_id = int(old["trial_id"].max()) + 1 if len(old) else 0
    row = {
        "trial_id": trial_id,
        "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "category": category,
        "description": description,
        "config_hash": config_hash,
        "study_id": study_id if study_id is not None else "",
        "charged_to_N": bool(charged_to_N),
    }
    new = pd.concat([old, pd.DataFrame([row], columns=_COLUMNS)], ignore_index=True)
    _assert_append_only(old, new)
    tmp = path.with_suffix(".parquet.tmp")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return trial_id


def current_N(path: str | Path = LEDGER_PATH) -> int:
    """Count of trials charged to N (drives DSR). 0 on an empty/missing ledger."""
    df = _read(Path(path))
    if len(df) == 0:
        return 0
    return int(df["charged_to_N"].sum())


def category_breakdown(path: str | Path = LEDGER_PATH) -> pd.Series:
    df = _read(Path(path))
    if len(df) == 0:
        return pd.Series(dtype=int)
    return df[df["charged_to_N"]].groupby("category").size()


def read_ledger(path: str | Path = LEDGER_PATH) -> pd.DataFrame:
    return _read(Path(path))
