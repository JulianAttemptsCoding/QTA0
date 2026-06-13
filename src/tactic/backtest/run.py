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
             no_trade_bps=25, start_date=None, end_date=None) -> pd.DataFrame:
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"]).dt.date
    if start_date is not None:
        preds = preds[preds["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        preds = preds[preds["date"] <= pd.to_datetime(end_date).date()]
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
    cols = ["date", "ret_gross", "cost", "ret_net", "turnover", "spy", "n_held"]
    return (pd.DataFrame(rows).dropna(subset=["ret_net"]) if rows
            else pd.DataFrame(columns=cols))


# ---------- dollar-neutral cross-sectional long-short (revision_plan §5.1) ----------
def backtest_long_short(preds: pd.DataFrame, prices: pd.DataFrame, k: int = 35,
                        no_trade_bps: int = 0, start_date=None, end_date=None,
                        roo: pd.DataFrame | None = None, sprd: dict | None = None) -> pd.DataFrame:
    """Dollar-neutral book: long top-k by q50, short bottom-k, equal weight, gross=1 (net=0).

    Judged on alpha, not on out-running a cap-weighted index — the correct test for a weak
    cross-sectional signal harvested with breadth (IR ~ IC*sqrt(breadth))."""
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"]).dt.date
    if start_date is not None:
        preds = preds[preds["date"] >= pd.to_datetime(start_date).date()]
    if end_date is not None:
        preds = preds[preds["date"] <= pd.to_datetime(end_date).date()]
    if roo is None:
        roo = _open_to_open(prices)
    if sprd is None:
        sprd = _spread(preds[["entity", "date"]])

    dates = sorted(preds["date"].unique())
    held_w: dict = {}
    rows = []
    for d in dates:
        day = preds[preds["date"] == d]
        if len(day) < 2 * k or d not in roo.index:
            continue
        ranked = day.sort_values("q50", ascending=False).reset_index(drop=True)
        longs = ranked["entity"].iloc[:k].tolist()
        shorts = ranked["entity"].iloc[-k:].tolist()
        new_w = {e: 0.5 / k for e in longs}
        new_w.update({e: -0.5 / k for e in shorts})

        cost = sum(abs(new_w.get(e, 0.0) - held_w.get(e, 0.0)) * 0.5 * sprd.get((e, d), 0.0008)
                   for e in set(new_w) | set(held_w))
        turnover = 0.5 * sum(abs(new_w.get(e, 0.0) - held_w.get(e, 0.0))
                             for e in set(new_w) | set(held_w))
        gross = sum(w * (roo.loc[d, e] if e in roo.columns and pd.notna(roo.loc[d, e]) else 0.0)
                    for e, w in new_w.items())
        spy = roo.loc[d, "SPY"] if "SPY" in roo.columns and pd.notna(roo.loc[d, "SPY"]) else 0.0
        rows.append({"date": d, "ret_gross": gross, "cost": cost, "ret_net": gross - cost,
                     "turnover": turnover, "spy": spy, "n_held": len(new_w)})
        held_w = new_w
    cols = ["date", "ret_gross", "cost", "ret_net", "turnover", "spy", "n_held"]
    return (pd.DataFrame(rows).dropna(subset=["ret_net"]) if rows
            else pd.DataFrame(columns=cols))


def ew_universe_returns(prices: pd.DataFrame, entities=None, roo: pd.DataFrame | None = None,
                        exclude=("SPY",)) -> pd.Series:
    """Equal-weight, daily-rebalanced return of the traded universe (isolates selection skill
    from the EW-vs-cap-weight factor). Per date: mean open-to-open return across present names."""
    if roo is None:
        roo = _open_to_open(prices)
    cols = [c for c in roo.columns if c not in exclude]
    if entities is not None:
        cols = [c for c in cols if c in set(entities)]
    return roo[cols].mean(axis=1)


def ew_pool_buyhold_nav(prices: pd.DataFrame, pool=None) -> pd.Series:
    """RUN 2 benchmark (§1B.2): equal-weight BUY-AND-HOLD of the pool on Adj Close (adjustment=='all'
    is total-return-adjusted). At t0 assign 1/M; no rebalance (weights drift). NAV_t = mean_n
    AdjClose_t / AdjClose_t0. Survivorship bias is identical to the strategy's, so it cancels."""
    p = prices[prices["adjustment"] == "all"].copy()
    p["entity"] = p.get("entity", p["symbol"])
    p["date"] = pd.to_datetime(p["date"]).dt.date
    piv = p.pivot_table(index="date", columns="entity", values="c").sort_index()
    if pool is not None:
        piv = piv[[c for c in piv.columns if c in set(pool)]]
    piv = piv.ffill()
    base = piv.apply(lambda col: col / col.dropna().iloc[0] if col.dropna().size else col)
    return base.mean(axis=1)


def ew_pool_monthly_nav(prices: pd.DataFrame, pool=None) -> pd.Series:
    """RUN 2 secondary benchmark: equal-weight, MONTH-END rebalanced pool NAV (Adj Close)."""
    p = prices[prices["adjustment"] == "all"].copy()
    p["entity"] = p.get("entity", p["symbol"])
    p["date"] = pd.to_datetime(p["date"]).dt.date
    piv = p.pivot_table(index="date", columns="entity", values="c").sort_index().ffill()
    if pool is not None:
        piv = piv[[c for c in piv.columns if c in set(pool)]]
    rets = piv.pct_change()
    idx = pd.to_datetime(piv.index)
    month = idx.to_period("M")
    nav, val, w = [], 1.0, None
    prev_m = None
    for i, d in enumerate(piv.index):
        r = rets.iloc[i]
        present = r.dropna().index
        if w is None or month[i] != prev_m:
            w = pd.Series(1.0 / max(len(present), 1), index=present)
            prev_m = month[i]
        port_r = float((w.reindex(present).fillna(0) * r.reindex(present).fillna(0)).sum())
        val *= (1 + port_r)
        # drift weights within the month
        gr = (1 + r.reindex(w.index).fillna(0))
        w = (w * gr); w = w / w.sum() if w.sum() > 0 else w
        nav.append(val)
    return pd.Series(nav, index=piv.index)


# ---------- regime tagging (revision_plan §2) ----------
_REGIMES = [
    ("2016-01-01", "2016-06-30", "2016H1 oil-mini-bear"),
    ("2016-07-01", "2017-12-31", "2016H2-17 low-vol bull"),
    ("2018-01-01", "2018-09-30", "2018 volmageddon"),
    ("2018-10-01", "2018-12-31", "2018Q4 correction"),
    ("2019-01-01", "2019-12-31", "2019 rate-cut bull"),
    ("2020-01-01", "2020-03-31", "2020 COVID crash"),
    ("2020-04-01", "2020-12-31", "2020 V-recovery"),
    ("2021-01-01", "2021-12-31", "2021 melt-up"),
    ("2022-01-01", "2022-12-31", "2022 rate-hike BEAR"),
    ("2023-01-01", "2023-03-31", "2023Q1 bank crisis"),
    ("2023-04-01", "2023-12-31", "2023 narrow AI rally"),
    ("2024-01-01", "2024-12-31", "2024 broad bull"),
    ("2025-01-01", "2025-12-31", "2025 tariff vol"),
    ("2026-01-01", "2026-12-31", "2026H1 elevated vol"),
]
# deep (Run 2) pre-2016 regimes prepended for §2 stratification
_REGIMES_DEEP = [
    ("2000-01-01", "2002-12-31", "2000-02 dot-com bust"),
    ("2003-01-01", "2007-09-30", "2003-07 recovery bull"),
    ("2007-10-01", "2009-03-31", "2008 GFC"),
    ("2009-04-01", "2011-06-30", "2009-11 QE recovery"),
    ("2011-07-01", "2011-12-31", "2011 EU crisis"),
    ("2012-01-01", "2015-06-30", "2012-15 bull"),
    ("2015-07-01", "2016-06-30", "2015-16 selloff"),
] + _REGIMES[1:]


def regime_of(d, deep: bool = False) -> str:
    d = pd.to_datetime(d).date()
    table = _REGIMES_DEEP if deep else _REGIMES
    for s, e, lab in table:
        if pd.to_datetime(s).date() <= d <= pd.to_datetime(e).date():
            return lab
    return "other"


def metrics_by_regime(bt: pd.DataFrame, bench_col: str = "spy", deep: bool = False) -> pd.DataFrame:
    bt = bt.copy()
    bt["regime"] = [regime_of(d, deep) for d in bt["date"]]
    rows = []
    for reg, g in bt.groupby("regime"):
        r, b = g["ret_net"].to_numpy(), g[bench_col].to_numpy()
        eq, eqb = np.cumprod(1 + r), np.cumprod(1 + b)
        rows.append({"regime": reg, "n_days": len(g),
                     "strat_return": float(eq[-1] - 1), "bench_return": float(eqb[-1] - 1),
                     "strat_sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(ANN)) if r.std() > 0 else np.nan,
                     "strat_maxdd": _max_dd(eq)})
    order = {lab: i for i, (_, _, lab) in enumerate(_REGIMES_DEEP if deep else _REGIMES)}
    return pd.DataFrame(rows).sort_values("regime", key=lambda s: s.map(lambda x: order.get(x, 999)))


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


def metrics_vs(ret_net: np.ndarray, bench: np.ndarray, label: str = "bench") -> dict:
    """Full QuantConnect-style metrics for a net-return series vs an arbitrary benchmark series."""
    r = np.asarray(ret_net, float); b = np.asarray(bench, float)
    eq, eqb = np.cumprod(1 + r), np.cumprod(1 + b)
    n = len(r)

    def stats(x, eqx):
        mu, sd = x.mean(), x.std(ddof=1)
        downside = x[x < 0].std(ddof=1) if (x < 0).any() else np.nan
        return {"total_return": float(eqx[-1] - 1), "cagr": float(eqx[-1] ** (ANN / n) - 1),
                "ann_vol": float(sd * np.sqrt(ANN)),
                "sharpe": float(mu / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
                "sortino": float(mu / downside * np.sqrt(ANN)) if downside and downside > 0 else np.nan,
                "max_drawdown": _max_dd(eqx), "win_rate": float((x > 0).mean())}

    if b.std() > 0:
        beta = np.cov(r, b)[0, 1] / np.var(b)
        alpha_ann = (r.mean() - beta * b.mean()) * ANN
    else:
        beta, alpha_ann = np.nan, np.nan
    return {"strategy": stats(r, eq), label: stats(b, eqb),
            "beta": float(beta), "alpha_ann": float(alpha_ann), "n_days": n}


def metrics_by_year(bt: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year strategy vs SPY (CAGR-equivalent annual return, Sharpe, maxDD)."""
    bt = bt.copy()
    bt["year"] = pd.to_datetime(bt["date"]).dt.year
    rows = []
    for y, g in bt.groupby("year"):
        r, spy = g["ret_net"].to_numpy(), g["spy"].to_numpy()
        eq, eqs = np.cumprod(1 + r), np.cumprod(1 + spy)
        rows.append({
            "year": int(y), "n_days": len(g),
            "strat_return": float(eq[-1] - 1), "spy_return": float(eqs[-1] - 1),
            "strat_sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(ANN)) if r.std() > 0 else np.nan,
            "spy_sharpe": float(spy.mean() / spy.std(ddof=1) * np.sqrt(ANN)) if spy.std() > 0 else np.nan,
            "strat_maxdd": _max_dd(eq), "spy_maxdd": _max_dd(eqs),
        })
    return pd.DataFrame(rows)


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
