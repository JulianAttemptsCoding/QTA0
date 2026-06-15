"""Re-backtest run2.1's EXISTING deep predictions with the SPY-beating trading algo.

No retraining. Reuses results/run2_1_chrono/oos_predictions.parquet (the deep chronological
walk-forward predictor output). Applies the *old* long-only/long-flat top-K buy/hold-band trader
(git 18e1896 src/tactic/backtest/run.py: k_buy=5, k_hold=8, no_trade_bps=25, next-open fills) that
produced the f4_vertex_83names headline (2023 CAGR 41.9% vs SPY 24.1%).

Benchmarks: SPY (the old algo's native benchmark) AND equal-weight pool buy-hold (firewall-correct
for a survivorship-biased deep pool). NOT deployable; deep pool is survivorship-biased.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEEP = ROOT / "data" / "deep_panel" / "curated"
RUNDIR = ROOT / "results" / "run2_1_chrono"
ANN = 252


def _open_to_open(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices[prices["adjustment"] == "all"].copy()
    p["entity"] = p.get("entity", p["symbol"])
    p["date"] = pd.to_datetime(p["date"]).dt.date
    piv = p.pivot_table(index="date", columns="entity", values="o").sort_index()
    return piv.shift(-2) / piv.shift(-1) - 1.0


def _spread() -> dict:
    spath = DEEP / "spread_surface.parquet"
    if not spath.exists():
        return {}
    s = pd.read_parquet(spath)
    s["date"] = pd.to_datetime(s["date"]).dt.date
    return {(r.entity, r.date): r.s_blend for r in s.itertuples()}


def backtest(preds, prices, k_buy=5, k_hold=8, no_trade_bps=25):
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"]).dt.date
    roo = _open_to_open(prices)
    sprd = _spread()
    dates = sorted(preds["date"].unique())
    held_w: dict = {}
    rows = []
    for d in dates:
        day = preds[preds["date"] == d]
        if len(day) < k_buy + 1 or d not in roo.index:
            continue
        ranked = day.sort_values("q50", ascending=False).reset_index(drop=True)
        ranks = {e: i for i, e in enumerate(ranked["entity"])}
        keep = {e for e in held_w if ranks.get(e, 1e9) < k_hold}
        for e in ranked["entity"]:
            if len(keep) >= k_buy:
                break
            keep.add(e)
        new_w = {e: 1.0 / len(keep) for e in keep} if keep else {}
        tw = dict(new_w)
        for e in set(held_w) | set(new_w):
            if abs(new_w.get(e, 0.0) - held_w.get(e, 0.0)) < no_trade_bps / 1e4:
                tw[e] = held_w.get(e, 0.0)
        ssum = sum(tw.values())
        if ssum > 0:
            tw = {e: w / ssum for e, w in tw.items()}
        turnover = 0.5 * sum(abs(tw.get(e, 0.0) - held_w.get(e, 0.0)) for e in set(tw) | set(held_w))
        cost = sum(abs(tw.get(e, 0.0) - held_w.get(e, 0.0)) * 0.5 * sprd.get((e, d), 0.0008)
                   for e in set(tw) | set(held_w))
        gross = sum(tw.get(e, 0.0) * (roo.loc[d, e] if e in roo.columns and pd.notna(roo.loc[d, e]) else 0.0)
                    for e in tw)
        spy = roo.loc[d, "SPY"] if "SPY" in roo.columns and pd.notna(roo.loc[d, "SPY"]) else 0.0
        # equal-weight pool buy-hold (firewall benchmark): mean OO return across all pool names that day
        ewp = float(np.nanmean([roo.loc[d, e] for e in roo.columns if pd.notna(roo.loc[d, e])]))
        rows.append({"date": d, "ret_gross": gross, "cost": cost, "ret_net": gross - cost,
                     "turnover": turnover, "spy": spy, "ewpool": ewp, "n_held": len(tw)})
        held_w = tw
    return pd.DataFrame(rows).dropna(subset=["ret_net"])


def _max_dd(eq):
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _stats(x):
    x = np.asarray(x); eq = np.cumprod(1 + x); n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    dn = x[x < 0].std(ddof=1) if (x < 0).any() else np.nan
    return {"total_return": float(eq[-1] - 1), "cagr": float(eq[-1] ** (ANN / n) - 1),
            "ann_vol": float(sd * np.sqrt(ANN)),
            "sharpe": float(mu / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
            "sortino": float(mu / dn * np.sqrt(ANN)) if dn and dn > 0 else np.nan,
            "max_drawdown": _max_dd(eq), "win_rate": float((x > 0).mean())}


def _beta_alpha(r, b):
    r, b = np.asarray(r), np.asarray(b)
    if b.std() == 0:
        return np.nan, np.nan
    beta = np.cov(r, b)[0, 1] / np.var(b)
    return float(beta), float((r.mean() - beta * b.mean()) * ANN)


def main():
    preds = pd.read_parquet(RUNDIR / "oos_predictions.parquet")
    prices = pd.read_parquet(DEEP / "prices_daily.parquet")
    bt = backtest(preds, prices)
    strat = _stats(bt["ret_net"]); spy = _stats(bt["spy"]); ewp = _stats(bt["ewpool"])
    b_spy, a_spy = _beta_alpha(bt["ret_net"], bt["spy"])
    b_ew, a_ew = _beta_alpha(bt["ret_net"], bt["ewpool"])
    out = {"strategy": strat, "spy": spy, "ewpool_buyhold": ewp,
           "beta_vs_spy": b_spy, "alpha_ann_vs_spy": a_spy,
           "beta_vs_ewpool": b_ew, "alpha_ann_vs_ewpool": a_ew,
           "avg_turnover": float(bt["turnover"].mean()),
           "avg_cost_bps": float(bt["cost"].mean() * 1e4),
           "n_days": int(len(bt)), "avg_names_held": float(bt["n_held"].mean())}
    bt.to_csv(RUNDIR / "backtest_oldtrader_daily.csv", index=False)
    (RUNDIR / "metrics_oldtrader.json").write_text(json.dumps(out, indent=2))
    # per-year
    bt["year"] = pd.to_datetime(bt["date"]).dt.year
    print(json.dumps(out, indent=2))
    print("\n=== per-year (strat ret / spy ret / ewpool ret) ===")
    for y, g in bt.groupby("year"):
        sr = float(np.cumprod(1 + g["ret_net"].to_numpy())[-1] - 1)
        sp = float(np.cumprod(1 + g["spy"].to_numpy())[-1] - 1)
        ew = float(np.cumprod(1 + g["ewpool"].to_numpy())[-1] - 1)
        print(f"{y}: strat {sr:+.1%}  spy {sp:+.1%}  ewpool {ew:+.1%}")


if __name__ == "__main__":
    main()
