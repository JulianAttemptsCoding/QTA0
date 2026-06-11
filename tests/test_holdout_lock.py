import datetime as dt

import pandas as pd

from tactic.common import io


def test_load_curated_refuses_locked_holdout(tmp_path, monkeypatch):
    curated = tmp_path / "curated"
    holdout = tmp_path / "holdout"
    curated.mkdir()
    holdout.mkdir()
    monkeypatch.setattr(io, "CURATED", curated)
    monkeypatch.setattr(io, "HOLDOUT", holdout)
    monkeypatch.setattr(io, "HOLDOUT_LOCK", holdout / ".LOCKED")
    monkeypatch.setattr(io, "_WINDOW_FILE", holdout / "window.json")

    df = pd.DataFrame({"date": pd.to_datetime(["2024-06-01", "2025-06-01"]).date,
                       "x": [1, 2]})
    io.write_parquet(df, curated / "prices.parquet")

    io.set_holdout_window(dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    io.lock_holdout()
    assert io.is_locked()

    # request inside holdout window -> refused
    try:
        io.load_curated("prices", start="2025-03-01", end="2025-09-01", base=curated)
        assert False, "expected HoldoutLocked"
    except io.HoldoutLocked:
        pass

    # request outside window -> allowed
    out = io.load_curated("prices", start="2024-01-01", end="2024-12-31", base=curated)
    assert len(out) == 1


def test_unlock_requires_prereg(tmp_path, monkeypatch):
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    monkeypatch.setattr(io, "HOLDOUT", holdout)
    monkeypatch.setattr(io, "HOLDOUT_LOCK", holdout / ".LOCKED")
    io.lock_holdout()
    try:
        io.unlock_holdout(prereg_committed=False)
        assert False
    except io.HoldoutLocked:
        pass
    io.unlock_holdout(prereg_committed=True)
    assert not io.is_locked()
