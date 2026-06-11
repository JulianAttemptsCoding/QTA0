import numpy as np
import pandas as pd

from tactic.common.calendar import TradingCalendar
from tactic.features.wavelets import modwt_energy


def test_modwt_zero_series():
    e = modwt_energy(np.zeros(128), levels=(2, 3, 4))
    assert set(e) == {2, 3, 4}
    assert all(v == 0.0 for v in e.values())


def test_modwt_short_series_safe():
    e = modwt_energy(np.array([1.0, 2.0, 3.0]), levels=(2, 3, 4))
    assert all(v == 0.0 for v in e.values())


def test_modwt_detects_structure():
    rng = np.random.default_rng(0)
    noisy = rng.normal(size=256)
    e = modwt_energy(noisy, levels=(2, 3, 4))
    assert all(v > 0 for v in e.values())


def test_calendar_shift():
    cal = TradingCalendar(pd.bdate_range("2024-01-01", "2024-02-01"))
    d = pd.Timestamp("2024-01-10")
    assert cal.is_session(d)
    assert cal.next_session(d) == pd.Timestamp("2024-01-11")
    assert cal.prev_session(d) == pd.Timestamp("2024-01-09")
    assert cal.shift(d, 5) == pd.Timestamp("2024-01-17")
