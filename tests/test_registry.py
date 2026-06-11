import pandas as pd
import pytest

from tactic.common import registry


def test_ledger_roundtrip_and_N(tmp_path):
    led = tmp_path / "trial_ledger.parquet"
    assert registry.current_N(led) == 0
    t0 = registry.log_trial("design", "first decision", "deadbeef", path=led)
    t1 = registry.log_trial("hpo", "trial 1", "cafe", study_id="s1", path=led)
    assert (t0, t1) == (0, 1)
    assert registry.current_N(led) == 2
    df = registry.read_ledger(led)
    assert list(df["trial_id"]) == [0, 1]
    assert df.loc[1, "study_id"] == "s1"


def test_not_charged_trials_excluded_from_N(tmp_path):
    led = tmp_path / "l.parquet"
    registry.log_trial("design", "frozen const", "h", charged_to_N=False, path=led)
    registry.log_trial("expert", "f4", "h2", path=led)
    assert registry.current_N(led) == 1


def test_append_only_refuses_shrink(tmp_path):
    led = tmp_path / "l.parquet"
    registry.log_trial("design", "a", "h", path=led)
    registry.log_trial("design", "b", "h", path=led)
    old = registry.read_ledger(led)
    shrunk = old.iloc[:1]
    with pytest.raises(registry.LedgerError):
        registry._assert_append_only(old, shrunk)


def test_append_only_refuses_mutation(tmp_path):
    led = tmp_path / "l.parquet"
    registry.log_trial("design", "a", "h", path=led)
    old = registry.read_ledger(led)
    mutated = old.copy()
    mutated.loc[0, "description"] = "TAMPERED"
    with pytest.raises(registry.LedgerError):
        registry._assert_append_only(old, mutated)


def test_invalid_category(tmp_path):
    with pytest.raises(ValueError):
        registry.log_trial("bogus", "x", "h", path=tmp_path / "l.parquet")
