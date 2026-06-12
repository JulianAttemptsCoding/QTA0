"""Render OOS results for a training run (sec16): plots + RESULTS.md + metrics.json.

Consumes reports/runs/<run_id>/{loss_history.csv, oos_predictions.parquet} and produces:
  equity_curve.png, drawdown.png, loss_curve.png, calibration.png
  RESULTS.md (QuantConnect-style table), metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..backtest.run import backtest, metrics, predictor_stats  # noqa: E402
from ..common.config import CURATED, REPORTS_DIR  # noqa: E402


def _plot_equity(bt: pd.DataFrame, path: Path):
    eq = (1 + bt["ret_net"]).cumprod()
    eqg = (1 + bt["ret_gross"]).cumprod()
    spy = (1 + bt["spy"]).cumprod()
    x = pd.to_datetime(bt["date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, eq, label="TACTIC-MoB (net)", lw=2)
    ax.plot(x, eqg, label="TACTIC-MoB (gross)", lw=1, ls="--", alpha=0.6)
    ax.plot(x, spy, label="SPY buy & hold", lw=2, color="grey")
    ax.set_title("OOS equity curve (growth of $1)")
    ax.set_ylabel("equity"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_drawdown(bt: pd.DataFrame, path: Path):
    eq = (1 + bt["ret_net"]).cumprod().to_numpy()
    dd = eq / np.maximum.accumulate(eq) - 1
    x = pd.to_datetime(bt["date"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(x, dd, 0, color="firebrick", alpha=0.5)
    ax.set_title("OOS strategy drawdown"); ax.set_ylabel("drawdown"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_loss(hist: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, g in hist.groupby("seed"):
        ax.plot(g["epoch"], g["train_pinball"], color="tab:blue", alpha=0.5,
                label="train" if seed == hist["seed"].min() else None)
        ax.plot(g["epoch"], g["val_pinball"], color="tab:orange", alpha=0.7,
                label="val" if seed == hist["seed"].min() else None)
    ax.set_title("Training vs validation pinball loss per epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("summed pinball loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_calibration(preds: pd.DataFrame, prices: pd.DataFrame, path: Path):
    from ..backtest.run import _open_to_open
    roo = _open_to_open(prices)
    pr = preds.copy(); pr["date"] = pd.to_datetime(pr["date"]).dt.date
    xs, ys = [], []
    for d, day in pr.groupby("date"):
        if d not in roo.index:
            continue
        for r in day.itertuples():
            if r.entity in roo.columns and pd.notna(roo.loc[d, r.entity]):
                xs.append(r.q50); ys.append(roo.loc[d, r.entity])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=6, alpha=0.2)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("predicted median (standardized)"); ax.set_ylabel("realized open-to-open return")
    ax.set_title("OOS prediction vs realized"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fmt_pct(x):
    return f"{x*100:.2f}%" if x == x else "n/a"


def render_run(run_id: str) -> dict:
    run_dir = REPORTS_DIR / "runs" / run_id
    hist = pd.read_csv(run_dir / "loss_history.csv")
    preds = pd.read_parquet(run_dir / "oos_predictions.parquet")
    prices = pd.read_parquet(CURATED / "prices_daily.parquet")

    bt = backtest(preds, prices)
    mt = metrics(bt)
    ps = predictor_stats(preds, prices)

    _plot_equity(bt, run_dir / "equity_curve.png")
    _plot_drawdown(bt, run_dir / "drawdown.png")
    _plot_loss(hist, run_dir / "loss_curve.png")
    _plot_calibration(preds, prices, run_dir / "calibration.png")
    bt.to_csv(run_dir / "backtest_daily.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": mt, "predictor": ps}, indent=2),
                                          encoding="utf-8")

    s, b = mt["strategy"], mt["spy"]
    md = [
        f"# OOS Results — run `{run_id}`",
        "",
        f"Out-of-sample window: **{bt['date'].min()} → {bt['date'].max()}** ({mt['n_days']} sessions). "
        "Long-only/long-flat top-K cross-sectional, fills at next open, net of estimated costs.",
        "",
        "## Performance vs SPY buy-and-hold (out-of-sample)",
        "",
        "| Metric | TACTIC-MoB (net) | SPY B&H |",
        "|---|---|---|",
        f"| Total return | {_fmt_pct(s['total_return'])} | {_fmt_pct(b['total_return'])} |",
        f"| CAGR | {_fmt_pct(s['cagr'])} | {_fmt_pct(b['cagr'])} |",
        f"| Ann. volatility | {_fmt_pct(s['ann_vol'])} | {_fmt_pct(b['ann_vol'])} |",
        f"| Sharpe | {s['sharpe']:.2f} | {b['sharpe']:.2f} |",
        f"| Sortino | {s['sortino']:.2f} | {b['sortino']:.2f} |",
        f"| Max drawdown | {_fmt_pct(s['max_drawdown'])} | {_fmt_pct(b['max_drawdown'])} |",
        f"| Win rate (daily) | {_fmt_pct(s['win_rate'])} | {_fmt_pct(b['win_rate'])} |",
        "",
        f"- Beta vs SPY: **{mt['beta']:.2f}**, annualized alpha: **{_fmt_pct(mt['alpha_ann'])}**",
        f"- Avg one-sided turnover/day: {_fmt_pct(mt['avg_turnover'])}; "
        f"avg cost: {mt['avg_cost_bps']:.2f} bps/day; avg names held: {mt['avg_names_held']:.1f}",
        "",
        "## Predictor statistics (out-of-sample)",
        "",
        f"- Summed pinball loss (standardized scale): **{ps.get('pinball', float('nan')):.4f}**",
        f"- Rank IC (mean daily Spearman, q50 vs realized): **{ps.get('ic_mean', float('nan')):.4f}** "
        f"(IR {ps.get('ic_ir', float('nan')):.2f})",
        f"- Directional hit rate: {_fmt_pct(ps.get('hit_rate', float('nan')))}",
        f"- 80% interval coverage: {_fmt_pct(ps.get('coverage_80', float('nan')))} (target 80%); "
        f"90% coverage: {_fmt_pct(ps.get('coverage_90', float('nan')))} (target 90%)",
        f"- OOS observations: {ps.get('n_obs', 0)}",
        "",
        "## Training",
        "",
        f"- Final val pinball: {hist['val_pinball'].min():.5f}; epochs run: {hist['epoch'].max()+1}; "
        f"seeds: {hist['seed'].nunique()}",
        "",
        "## Figures",
        "",
        "![equity](equity_curve.png)",
        "![drawdown](drawdown.png)",
        "![loss](loss_curve.png)",
        "![calibration](calibration.png)",
        "",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[results] wrote {run_dir/'RESULTS.md'}")
    print(f"[results] strat CAGR {_fmt_pct(s['cagr'])} Sharpe {s['sharpe']:.2f} maxDD "
          f"{_fmt_pct(s['max_drawdown'])} | SPY CAGR {_fmt_pct(b['cagr'])} Sharpe {b['sharpe']:.2f} "
          f"| IC {ps.get('ic_mean', float('nan')):.4f}")
    return {"metrics": mt, "predictor": ps, "run_dir": str(run_dir)}
