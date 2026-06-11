import pandas as pd
import pytest

from tactic.common import contracts


def test_assert_pit_ok():
    df = pd.DataFrame({
        "feature_available_ts": pd.to_datetime(["2020-01-01 16:00"]),
        "decision_ts": pd.to_datetime(["2020-01-01 16:00"]),
        "entry_ts": pd.to_datetime(["2020-01-02 09:30"]),
        "exit_ts": pd.to_datetime(["2020-01-02 16:00"]),
    })
    contracts.assert_pit(df)


def test_assert_pit_violation():
    df = pd.DataFrame({
        "feature_available_ts": pd.to_datetime(["2020-01-03 16:00"]),
        "decision_ts": pd.to_datetime(["2020-01-01 16:00"]),
    })
    with pytest.raises(contracts.ContractViolation):
        contracts.assert_pit(df)


def test_assert_feed_sip():
    contracts.assert_feed_sip(pd.DataFrame({"feed": ["sip", "sip"]}))
    with pytest.raises(contracts.ContractViolation):
        contracts.assert_feed_sip(pd.DataFrame({"feed": ["sip", "iex"]}))


def test_fit_scope():
    class Est(contracts.FitScopeMixin):
        pass
    e = Est()
    with pytest.raises(contracts.ContractViolation):
        contracts.assert_fit_scope(e, "train")  # never fit
    e._record_fit_fold("val")
    with pytest.raises(contracts.ContractViolation):
        contracts.assert_fit_scope(e, "train")  # fit on val = leak
    e._record_fit_fold("train")
    contracts.assert_fit_scope(e, "train")


def test_terminal_returns():
    ok = pd.DataFrame({
        "is_exit": [True, True, False],
        "terminal_type": ["bankruptcy", "unknown", ""],
        "terminal_return": [-1.0, None, None],
    })
    contracts.assert_terminal_returns(ok)
    bad = pd.DataFrame({
        "is_exit": [True],
        "terminal_type": ["cash_merger"],
        "terminal_return": [None],  # non-unknown must have a return
    })
    with pytest.raises(contracts.ContractViolation):
        contracts.assert_terminal_returns(bad)
