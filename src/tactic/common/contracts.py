"""Runtime contracts — enforced in code, not comments (§0.3).

These are called by loaders and writers throughout the pipeline. A violation raises
ContractViolation, which fails the build / CI.
"""
from __future__ import annotations

import pandas as pd


class ContractViolation(RuntimeError):
    pass


# --- §0.3.1 Temporal contract ---------------------------------------------------
# feature_available_ts <= decision_ts(close t) < entry_ts(open t+1) < exit_ts
_TS_ORDER = ["feature_available_ts", "decision_ts", "entry_ts", "exit_ts"]


def assert_pit(df: pd.DataFrame) -> pd.DataFrame:
    """Assert point-in-time temporal ordering on whichever ts columns are present.

    At minimum, if `feature_available_ts` and `decision_ts` are both present they must
    satisfy feature_available_ts <= decision_ts. Any consecutive pair among
    (feature_available_ts, decision_ts, entry_ts, exit_ts) that is present is checked.
    """
    present = [c for c in _TS_ORDER if c in df.columns]
    if "feature_available_ts" in df.columns and "decision_ts" in df.columns:
        bad = df["feature_available_ts"] > df["decision_ts"]
        if bool(bad.any()):
            raise ContractViolation(
                f"PIT violation: feature_available_ts > decision_ts in {int(bad.sum())} rows"
            )
    for a, b in zip(present, present[1:]):
        # strict for decision<entry<exit; non-strict only for the feature<=decision pair
        strict = not (a == "feature_available_ts")
        cmp = df[a] >= df[b] if strict else df[a] > df[b]
        if bool(cmp.any()):
            raise ContractViolation(
                f"PIT violation: {a} {'>=' if strict else '>'} {b} in {int(cmp.sum())} rows"
            )
    return df


# --- §0.3.3 SIP-only features ---------------------------------------------------
def assert_feed_sip(df: pd.DataFrame, feed_col: str = "feed") -> pd.DataFrame:
    """Every row's feed must be 'sip'. IEX bars entering curated/ fail this."""
    if feed_col not in df.columns:
        raise ContractViolation(f"feed column {feed_col!r} missing; cannot prove SIP provenance")
    feeds = set(df[feed_col].astype(str).str.lower().unique())
    if feeds - {"sip"}:
        raise ContractViolation(f"non-SIP feed(s) present: {sorted(feeds - {'sip'})}")
    return df


# --- §0.3.2 Fit/transform discipline -------------------------------------------
class FitScopeMixin:
    """Estimators mix this in to record the fold they were fit on.

    Call `self._record_fit_fold(fold)` inside fit(); `assert_fit_scope` then verifies
    the object was only ever fit on a training fold.
    """

    _fit_fold: str | None = None

    def _record_fit_fold(self, fold: str) -> None:
        self._fit_fold = fold


def assert_fit_scope(obj: object, fold: str) -> object:
    """Assert `obj` was fit on a training fold, and that `fold` is a train fold.

    `fold` is the fold the data being transformed belongs to; an estimator fit on a
    non-train fold (val/test/holdout leaking into fit) fails CI.
    """
    fit_fold = getattr(obj, "_fit_fold", None)
    if fit_fold is None:
        raise ContractViolation(f"{type(obj).__name__} has no recorded fit fold; cannot prove fit scope")
    if fit_fold != "train":
        raise ContractViolation(
            f"fit-scope violation: {type(obj).__name__} was fit on fold={fit_fold!r}, must be 'train'"
        )
    return obj


# --- §0.3.9 Delisting / terminal-return contract --------------------------------
_VALID_TERMINAL_TYPES = {"cash_merger", "bankruptcy", "worthless", "exchange_delist", "acquired", "unknown"}


def assert_terminal_returns(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Every exiting entity must have a populated terminal_return and a valid terminal_type.

    `unknown` is a valid type (it is reported in the survivorship bound), but the typed
    field must be present and non-null for every exit.
    """
    exits = universe_df[universe_df.get("is_exit", False) == True]  # noqa: E712
    if "terminal_type" not in universe_df.columns or "terminal_return" not in universe_df.columns:
        raise ContractViolation("universe missing terminal_return/terminal_type columns")
    missing_type = exits["terminal_type"].isna() | (exits["terminal_type"] == "")
    if bool(missing_type.any()):
        raise ContractViolation(f"{int(missing_type.sum())} exits lack a terminal_type")
    bad_type = ~exits["terminal_type"].isin(_VALID_TERMINAL_TYPES)
    if bool(bad_type.any()):
        raise ContractViolation(f"invalid terminal_type(s): {sorted(set(exits.loc[bad_type,'terminal_type']))}")
    # terminal_return may be null ONLY when type == unknown (asset-day excluded + counted)
    needs_ret = exits["terminal_type"] != "unknown"
    missing_ret = needs_ret & exits["terminal_return"].isna()
    if bool(missing_ret.any()):
        raise ContractViolation(f"{int(missing_ret.sum())} non-unknown exits lack terminal_return")
    return universe_df
