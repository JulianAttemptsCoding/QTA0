"""Trading calendar (§3.2): Alpaca /v2/calendar 2016->present, used everywhere.

`TradingCalendar` loads the curated session index; a Mon-Fri business-day fallback is used
only when no curated calendar exists (tests / offline), and emits a warning so it never
silently masks a missing ingest.
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import pandas as pd

from .config import CURATED

_CAL_PATH = CURATED / "calendar.parquet"


class TradingCalendar:
    def __init__(self, sessions: pd.DatetimeIndex):
        self.sessions = pd.DatetimeIndex(sorted(pd.to_datetime(sessions).normalize().unique()))

    @classmethod
    def load(cls, path: Path = _CAL_PATH) -> "TradingCalendar":
        if path.exists():
            df = pd.read_parquet(path)
            return cls(pd.to_datetime(df["date"]))
        warnings.warn(
            f"no curated calendar at {path}; using Mon-Fri business-day fallback "
            "(run `make data-alpaca` for the real session index)",
            stacklevel=2,
        )
        return cls(pd.bdate_range("2016-01-01", dt.date.today()))

    def is_session(self, d: dt.date | pd.Timestamp) -> bool:
        return pd.Timestamp(d).normalize() in self.sessions

    def slice(self, start: dt.date | str, end: dt.date | str) -> pd.DatetimeIndex:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self.sessions[(self.sessions >= s) & (self.sessions <= e)]

    def shift(self, d: dt.date | pd.Timestamp, n: int) -> pd.Timestamp:
        """Session `d` shifted by n sessions (n>0 future, n<0 past). Raises if out of range."""
        t = pd.Timestamp(d).normalize()
        idx = self.sessions.searchsorted(t)
        if idx >= len(self.sessions) or self.sessions[idx] != t:
            # d not a session: snap to nearest forward session for forward shifts
            if n >= 0:
                idx = idx  # first session >= d
            else:
                idx = idx - 1
        target = idx + n
        if target < 0 or target >= len(self.sessions):
            raise IndexError(f"session shift out of range: {d} + {n}")
        return self.sessions[target]

    def next_session(self, d: dt.date | pd.Timestamp) -> pd.Timestamp:
        return self.shift(d, 1)

    def prev_session(self, d: dt.date | pd.Timestamp) -> pd.Timestamp:
        return self.shift(d, -1)
