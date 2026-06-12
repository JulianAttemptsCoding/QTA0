"""OOS backtest from model predictions + QuantConnect-style metrics + plots (sec13/sec16).

Long-only/long-flat cross-sectional: each decision day t, rank entities by the predicted median
(q50), hold a buy/hold band (enter rank<=K_buy, hold while rank<=K_hold), equal-weight the held
set, gross=100%. Fills at next session open (OO structure, 1-day hold); costs = sum|dw|*0.5*s_blend.
Benchmark = SPY buy-and-hold over the same OOS window. All returns net unless noted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common.config import CURATED
from ..models.train import TAUS

ANN = 252


# ---------- realized returns ----------
def _open_to_open(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (entity,date t) realized return over [open(t+1), open(t+2)] (aligns to decision t)."""
    p = prices[prices["adjustment"] == "all"].copy()
    p["entity"] = p.get("entity", p["symbol"])
    p["date"] = pd.to_datetime(p["date"]).dt.date
    piv = p.pivot_table(index="date", columns="entity", values="o").sort_index()
    roo = piv.shift(-2) / piv.shift(-1) - 1.0
    return roo


def _spread(entity_dates) -> dict:
    spath = CURATED / "spread_surface.parquet"
    if not spath.exists():
        return {}
    s = pd.read_parquet(spath)
    s["date"] = pd.to_datetime(s["date"]).dt.date
    return {(r.entity, r.date): r.s_blend for r in s.itertuples()}


# ---------- strategy ----------
def backtest(preds: pd.DataFrame, prices: pd.DataFrame, k_buy=5, k_hold=8,
             no_trade_bps=25) -> pd.DataFrame:
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"]).dt.date
    roo = _open_to_open(prices)
    sprd = _spread(preds[["entity", "date"]])

    dates = sorted(preds["date"].unique())
    held_w: dict = {}
    rows = []
    for d in dates:
        day = preds[preds["date"] == d]
        if len(day) < k_buy + 1 or d not in roo.index:
            continue
        ranked = day.sort_values("q50", ascending=False).reset_index(drop=True)
        ranks = {e: i for i, e in enumerate(ranked["entity"])}
        # buy/hold band: keep currently-held while rank<k_hold; add top until k_buy held
        keep = {e for e in held_w if ranks.get(e, 1e9) < k_hold}
        for e in ranked["entity"]:
            if len(keep) >= k_buy:
                break
            keep.add(e)
        new_w = {e: 1.0 / len(keep) for e in keep} if keep else {}

        # no-trade band: ignore tiny weight changes
        tw = dict(new_w)
        for e in set(held_w) | set(new_w):
            if abs(new_w.get(e, 0.0) - held_w.get(e, 0.0)) < no_trade_bps / 1e4:
                tw[e] = held_w.get(e, 0.0)
        ssum = sum(tw.values())
        if ssum > 0:
            tw = {e: w / ssum for e, w in tw.items()}

        turnover = 0.5 * sum(abs(tw.get(e, 0.0) - held_w.get(e, 0.0))
                             for e in set(tw) | set(held_w))
        cost = sum(abs(tw.get(e, 0.0) - held_w.get(e, 0.0)) * 0.5 * sprd.get((e, d), 0.0008)
                   for e in set(tw) | set(held_w))
        gross = sum(tw.get(e, 0.0) * (roo.loc[d, e] if e in roo.columns and pd.notna(roo.loc[d, e]) else 0.0)
                    for e in tw)
        spy = roo.loc[d, "SPY"] if "SPY" in roo.columns and pd.notna(roo.loc[d, "SPY"]) else 0.0
        rows.append({"date": d, "ret_gross": gross, "cost": cost, "ret_net": gross - cost,
                     "turnover": turnover, "spy": spy, "n_held": len(tw)})
        held_w = tw
    return pd.DataFrame(rows).dropna(subset=["ret_net"])


# ---------- metrics ----------
def _max_dd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def metrics(bt: pd.DataFrame) -> dict:
    r = bt["ret_net"].to_numpy()
    spy = bt["spy"].to_numpy()
    eq = np.cumprod(1 + r)
    eq_spy = np.cumprod(1 + spy)
    n = len(r)

    def stats(x, eqx):
        mu, sd = x.mean(), x.std(ddof=1)
        downside = x[x < 0].std(ddof=1) if (x < 0).any() else np.nan
        sharpe = mu / sd * np.sqrt(ANN) if sd > 0 else np.nan
        sortino = mu / downside * np.sqrt(ANN) if downside and downside > 0 else np.nan
        cagr = eqx[-1] ** (ANN / n) - 1
        return {"total_return": float(eqx[-1] - 1), "cagr": float(cagr),
                "ann_vol": float(sd * np.sqrt(ANN)), "sharpe": float(sharpe),
                "sortino": float(sortino), "max_drawdown": _max_dd(eqx),
                "win_rate": float((x > 0).mean())}

    strat, bench = stats(r, eq), stats(spy, eq_spy)
    # beta/alpha vs SPY
    if spy.std() > 0:
        beta = np.cov(r, spy)[0, 1] / np.var(spy)
        alpha_ann = (r.mean() - beta * spy.mean()) * ANN
    else:
        beta, alpha_ann = np.nan, np.nan
    return {"strategy": strat, "spy": bench,
            "beta": float(beta), "alpha_ann": float(alpha_ann),
            "avg_turnover": float(bt["turnover"].mean()),
            "avg_cost_bps": float(bt["cost"].mean() * 1e4),
            "n_days": n, "avg_names_held": float(bt["n_held"].mean())}


def predictor_stats(preds: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Predictor diagnostics on the OOS set.

    `y_true` in preds is the model's standardized target (open-to-open / sigma), so pinball and
    interval coverage are computed on the SAME scale the quantiles live on. IC/hit-rate are
    rank/sign-based and identical whether scored against standardized or raw realized returns.
    """
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "y_true" not in df.columns:
        return {}
    df = df.dropna(subset=["y_true", "q50"])
    if df.empty:
        return {}
    # summed pinball on the standardized scale
    pin = 0.0
    for t in TAUS:
        q = df[f"q{int(t*100):02d}"].to_numpy()
        e = df["y_true"].to_numpy() - q
        pin += np.maximum(t * e, (t - 1) * e).mean()
    ics = [g["q50"].corr(g["y_true"], method="spearman")
           for _, g in df.groupby("date") if len(g) >= 5]
    ics = [x for x in ics if np.isfinite(x)]
    hit = float((np.sign(df["q50"]) == np.sign(df["y_true"])).mean())
    cov80 = float(((df["y_true"] >= df["q10"]) & (df["y_true"] <= df["q90"])).mean())
    cov90 = float(((df["y_true"] >= df["q05"]) & (df["y_true"] <= df["q95"])).mean())
    return {"pinball": float(pin), "ic_mean": float(np.mean(ics)), "ic_std": float(np.std(ics)),
            "ic_ir": float(np.mean(ics) / (np.std(ics) + 1e-9)),
            "hit_rate": hit, "coverage_80": cov80, "coverage_90": cov90, "n_obs": len(df)}
