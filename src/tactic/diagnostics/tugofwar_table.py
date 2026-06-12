"""Tug-of-war table (sec7.4): the direct existence test for the Pass-1 thesis on our sample.

Decile sorts on past-1-month overnight return and past-1-month intraday return; report the
subsequent overnight and intraday mean returns (Lou-Polk-Skouras pattern). Opposite-signed
component continuation across deciles is the predictability signal feeding gate G1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _past_sum(g: pd.Series, window: int = 21) -> pd.Series:
    return g.rolling(window, min_periods=window // 2).sum()


def tugofwar_table(labels: pd.DataFrame, lookback: int = 21, n_deciles: int = 10) -> pd.DataFrame:
    """Returns a table: sort_leg x decile -> mean subsequent r_on and r_in (next day)."""
    df = labels.sort_values(["entity", "date"]).copy()
    df["past_on"] = df.groupby("entity")["r_on"].transform(lambda s: _past_sum(s, lookback))
    df["past_in"] = df.groupby("entity")["r_in"].transform(lambda s: _past_sum(s, lookback))
    df["next_on"] = df.groupby("entity")["r_on"].shift(-1)
    df["next_in"] = df.groupby("entity")["r_in"].shift(-1)

    out = []
    for leg in ("past_on", "past_in"):
        sub = df.dropna(subset=[leg, "next_on", "next_in"]).copy()
        if len(sub) < n_deciles * 5:
            continue
        # cross-sectional deciles per date
        sub["decile"] = sub.groupby("date")[leg].transform(
            lambda s: pd.qcut(s.rank(method="first"), n_deciles, labels=False) if s.nunique() >= n_deciles else np.nan)
        grp = sub.dropna(subset=["decile"]).groupby("decile")
        tab = grp[["next_on", "next_in"]].mean()
        tab["sort_leg"] = leg
        tab["n"] = grp.size().to_numpy()
        out.append(tab.reset_index())
    if not out:
        return pd.DataFrame(columns=["sort_leg", "decile", "next_on", "next_in", "n"])
    return pd.concat(out, ignore_index=True)


def continuation_spread(table: pd.DataFrame, leg: str, comp: str) -> float:
    """Top-minus-bottom decile spread of subsequent `comp` for sort `leg`."""
    t = table[table["sort_leg"] == leg].sort_values("decile")
    if len(t) < 2:
        return np.nan
    return float(t[comp].iloc[-1] - t[comp].iloc[0])
