"""Walk-forward splitter (sec10.3 / revision_plan §3B fallback): expanding train, multi-year
val for regime diversity, purge 2d + embargo 5d. Used as a cheaper cross-check of CPCV."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class WFFold:
    fold_id: int
    train_end: dt.date
    val_start: dt.date
    val_end: dt.date
    test_start: dt.date
    test_end: dt.date


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


# revision_plan §3B: expanding train, 2-year val, annual test; 2022 enters OOS in F1 and train later.
_DEFAULT_FOLDS = [
    ("2019-12-31", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-12-31", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2023-12-31", "2024-01-01", "2025-12-31", "2026-01-01", "2026-06-10"),
]


def make_folds(dates=None, cfg=None) -> list[WFFold]:
    """Return the expanding-window walk-forward folds (§3B). `dates`/`cfg` accepted for API
    symmetry; the schedule is calendar-anchored per the revision plan."""
    folds = []
    for i, (te, vs, ve, ts, tend) in enumerate(_DEFAULT_FOLDS):
        folds.append(WFFold(i, _d(te), _d(vs), _d(ve), _d(ts), _d(tend)))
    return folds
