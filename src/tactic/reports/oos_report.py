"""Combined-OOS portfolio + firewall + report (revision_plan A5-A7 / B4-B6).

Runs LOCALLY after the CPCV Vertex shards finish and aggregate_combined() has produced
oos_predictions.parquet + paths.csv. Backtests both books (dollar-neutral L/S + long-flat breadth)
against the run's benchmarks (Run 1: SPY + EW-universe; Run 2: EW buy-and-hold of own pool + monthly
EW), computes per-year / per-regime tables, the firewall numbers (DSR, PBO, DM, path-Sharpe
distribution), renders plots, and writes RESULTS_RUN{1,2}.md. Gates are reported, not enforced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.run import (backtest, backtest_long_short, ew_universe_returns,
                            ew_pool_buyhold_nav, ew_pool_monthly_nav, metrics_vs,
                            metrics_by_regime, _open_to_open, _spread, predictor_stats, ANN)
from ..validation.dsr import deflated_sharpe
from ..validation.pbo import pbo
from ..validation.dm_test import dm_test
from ..common.registry import current_N


def _ann_sharpe(r):
    r = np.asarray(r, float); sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")


def _book_daily(preds, prices, roo, sprd, kind, **kw):
    if kind == "ls":
        bt = backtest_long_short(preds, prices, roo=roo, sprd=sprd, **kw)
    else:
        bt = backtest(preds, prices, **kw)
    return bt


def build_report(run_dir, prices_path, run_label="RUN1", deep=False,
                 pool_csv=None, k_book=35, out_md=None) -> dict:
    run_dir = Path(run_dir)
    preds = pd.read_parquet(run_dir / "oos_predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"]).dt.date
    prices = pd.read_parquet(prices_path)
    roo = _open_to_open(prices)
    sprd = _spread(preds[["entity", "date"]])

    # ---- books ----
    ls = backtest_long_short(preds, prices, k=k_book, roo=roo, sprd=sprd)
    lf = backtest(preds, prices, k_buy=20, k_hold=60)         # long-flat breadth
    ls = ls.sort_values("date").reset_index(drop=True)
    lf = lf.sort_values("date").reset_index(drop=True)

    # ---- benchmarks aligned to the L/S date axis ----
    benches = {}
    if deep:
        nav_bh = ew_pool_buyhold_nav(prices)
        nav_mo = ew_pool_monthly_nav(prices)
        bh_ret = nav_bh.pct_change().reindex(pd.to_datetime(ls["date"])).to_numpy()
        mo_ret = nav_mo.pct_change().reindex(pd.to_datetime(ls["date"])).to_numpy()
        benches["EWpool_buyhold"] = np.nan_to_num(bh_ret)
        benches["EWpool_monthly"] = np.nan_to_num(mo_ret)
        primary = "EWpool_buyhold"
    else:
        ew = ew_universe_returns(prices, roo=roo)
        ew_ret = ew.reindex(pd.to_datetime(ls["date"]).dt.date).to_numpy()
        benches["SPY"] = ls["spy"].to_numpy()
        benches["EWuniverse"] = np.nan_to_num(ew_ret)
        primary = "SPY"

    # ---- metrics (both books vs each benchmark) ----
    report = {"run_label": run_label, "deep": deep, "k_book": k_book}
    report["books"] = {}
    for name, bt in (("long_short", ls), ("long_flat", lf)):
        r = bt["ret_net"].to_numpy()
        # align each benchmark (built on the L/S date axis) to this book's length
        vs = {}
        for b, bench in benches.items():
            n = min(len(r), len(bench))
            vs[b] = metrics_vs(r[:n], np.asarray(bench)[:n], label=b)
        report["books"][name] = {
            "ann_sharpe": _ann_sharpe(r),
            "avg_turnover": float(bt["turnover"].mean()),
            "avg_cost_bps": float(bt["cost"].mean() * 1e4),
            "n_days": int(len(bt)),
            "vs": vs,
        }

    # ---- predictor diagnostics ----
    report["predictor"] = predictor_stats(preds, prices)

    # ---- per-year / per-regime (on L/S vs primary benchmark) ----
    ls2 = ls.copy()
    ls2["bench"] = benches[primary]
    ls2["year"] = pd.to_datetime(ls2["date"]).dt.year
    yr = []
    for y, g in ls2.groupby("year"):
        rr, bb = g["ret_net"].to_numpy(), g["bench"].to_numpy()
        yr.append({"year": int(y), "n": len(g),
                   "strat_ret": float(np.prod(1 + rr) - 1), "bench_ret": float(np.prod(1 + bb) - 1),
                   "strat_sharpe": _ann_sharpe(rr)})
    report["by_year"] = yr
    reg = metrics_by_regime(ls2.rename(columns={"bench": primary}), bench_col=primary, deep=deep)
    report["by_regime"] = reg.to_dict("records")

    # ---- firewall ----
    paths = pd.read_csv(run_dir / "paths.csv") if (run_dir / "paths.csv").exists() else pd.DataFrame()
    path_sharpes = paths["path_sharpe"].dropna().to_numpy() if "path_sharpe" in paths else np.array([])
    N = max(current_N(), 1)
    var_sr = float(np.var(path_sharpes / np.sqrt(ANN))) if len(path_sharpes) > 1 else 0.0
    rnet = ls["ret_net"].to_numpy()
    try:
        dsr = deflated_sharpe(rnet, var_sr_trials=var_sr, n_trials=N)
    except Exception as e:
        dsr = {"error": str(e)}
    # PBO over a small book-config ensemble on the combined OOS daily returns
    books_matrix = []
    for k in (20, 35, 50):
        b = backtest_long_short(preds, prices, k=k, roo=roo, sprd=sprd)
        if not b.empty:
            books_matrix.append(b.set_index("date")["ret_net"].rename(f"ls{k}"))
    if not lf.empty:
        books_matrix.append(lf.set_index("date")["ret_net"].rename("long_flat"))
    M = pd.concat(books_matrix, axis=1).dropna().to_numpy() if len(books_matrix) >= 2 else np.empty((0, 0))
    pbo_out = pbo(M) if M.shape[0] >= 8 and M.shape[1] >= 2 else {"pbo": float("nan"), "note": "insufficient"}
    # DM: strategy vs each benchmark (loss = -return; negative stat => strat better)
    dm = {}
    for b, bench in benches.items():
        n = min(len(rnet), len(bench))
        try:
            dm[b] = dm_test(-rnet[:n], -np.asarray(bench)[:n])
        except Exception as e:
            dm[b] = {"error": str(e)}
    report["firewall"] = {
        "honest_N": N, "path_sharpe_n": int(len(path_sharpes)),
        "path_sharpe_mean": float(np.mean(path_sharpes)) if len(path_sharpes) else float("nan"),
        "path_sharpe_std": float(np.std(path_sharpes)) if len(path_sharpes) else float("nan"),
        "path_sharpe_q": (np.quantile(path_sharpes, [0.1, 0.5, 0.9]).tolist()
                          if len(path_sharpes) else []),
        "path_sharpe_frac_pos": float((path_sharpes > 0).mean()) if len(path_sharpes) else float("nan"),
        "dsr": dsr, "pbo": pbo_out, "dm_vs_bench": dm,
    }

    report["benchmarks"] = list(benches.keys())
    report["primary_benchmark"] = primary
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # plots + markdown
    try:
        _plots(run_dir, ls, lf, benches, primary, path_sharpes, deep)
    except Exception as e:
        print(f"[oos_report] plot skipped: {e}")
    md = _markdown(report, run_dir, pool_csv, deep)
    out_md = Path(out_md) if out_md else (run_dir / f"RESULTS_{run_label}.md")
    out_md.write_text(md, encoding="utf-8")
    print(f"[oos_report] wrote {out_md}")
    return report


def _plots(run_dir, ls, lf, benches, primary, path_sharpes, deep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dts = pd.to_datetime(ls["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dts, np.cumprod(1 + ls["ret_net"]), label="L/S (dollar-neutral) net")
    ax.plot(pd.to_datetime(lf["date"]), np.cumprod(1 + lf["ret_net"]), label="long-flat net", alpha=0.8)
    for b, bench in benches.items():
        ax.plot(dts, np.cumprod(1 + np.nan_to_num(np.resize(bench, len(ls)))), label=b, alpha=0.7)
    ax.set_yscale("log"); ax.legend(); ax.set_title("Combined-OOS equity (log)")
    fig.tight_layout(); fig.savefig(run_dir / "equity.png", dpi=110); plt.close(fig)

    eq = np.cumprod(1 + ls["ret_net"].to_numpy())
    dd = eq / np.maximum.accumulate(eq) - 1
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(dts, dd, 0, color="crimson", alpha=0.5)
    ax.set_title("L/S drawdown"); fig.tight_layout()
    fig.savefig(run_dir / "drawdown.png", dpi=110); plt.close(fig)

    if len(path_sharpes):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(path_sharpes, bins=min(20, len(path_sharpes)), color="steelblue")
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"CPCV path-Sharpe distribution (n={len(path_sharpes)})")
        fig.tight_layout(); fig.savefig(run_dir / "path_sharpe_hist.png", dpi=110); plt.close(fig)


def _fmt_pct(x):
    return f"{x*100:+.1f}%" if x == x else "n/a"


def _markdown(rep, run_dir, pool_csv, deep) -> str:
    L = []
    if deep:
        L += ["# RESULTS — RUN 2 (DEEP, EXPLORATORY)", "",
              "> **SURVIVORSHIP-BIAS CAVEAT (read first).** This run uses free yfinance daily history",
              "> for a pool of names that *survived to today*; delisted/dead tickers are absent, so",
              "> absolute returns are upward-biased. It is benchmarked ONLY against equal-weight",
              "> buy-and-hold of its OWN pool (bias cancels in the relative comparison). It is",
              "> **NOT deployable** and is **never** to be headlined vs SPY.", ""]
    else:
        L += ["# RESULTS — RUN 1 (CLEAN, deployable-grade)", "",
              "Alpaca SIP 2016-2026, 196-name universe, CPCV combined out-of-sample.", ""]
    fw = rep["firewall"]
    L += [f"- Primary benchmark: **{rep['primary_benchmark']}** | books: dollar-neutral L/S + long-flat breadth",
          f"- Honest N (registry trials): **{fw['honest_N']}**",
          f"- CPCV paths: n={fw['path_sharpe_n']}, mean Sharpe {fw['path_sharpe_mean']:.3f} "
          f"(std {fw['path_sharpe_std']:.3f}), frac>0 {fw['path_sharpe_frac_pos']:.2f}", ""]

    L += ["## Books vs benchmarks (combined OOS)", ""]
    for bk, d in rep["books"].items():
        L += [f"### {bk}  (ann Sharpe {d['ann_sharpe']:.3f}, turnover {d['avg_turnover']:.3f}, "
              f"cost {d['avg_cost_bps']:.1f}bps, {d['n_days']} days)"]
        L += ["", "| benchmark | strat CAGR | bench CAGR | strat Sharpe | alpha (ann) | beta | maxDD |",
              "|---|---|---|---|---|---|---|"]
        for b, m in d["vs"].items():
            s = m["strategy"]
            L.append(f"| {b} | {_fmt_pct(s['cagr'])} | {_fmt_pct(m[b]['cagr'])} | "
                     f"{s['sharpe']:.2f} | {_fmt_pct(m['alpha_ann'])} | {m['beta']:.2f} | "
                     f"{_fmt_pct(s['max_drawdown'])} |")
        L.append("")

    p = rep["predictor"]
    if p:
        L += ["## Predictor diagnostics (OOS)", "",
              f"- IC mean {p.get('ic_mean', float('nan')):.4f} (IR {p.get('ic_ir', float('nan')):.3f}), "
              f"hit {p.get('hit_rate', float('nan')):.3f}, pinball {p.get('pinball', float('nan')):.4f}",
              f"- coverage 80/90: {p.get('coverage_80', float('nan')):.3f} / {p.get('coverage_90', float('nan')):.3f}, "
              f"n_obs {p.get('n_obs', 0)}", ""]

    L += ["## Per-year (L/S vs primary)", "", "| year | strat | bench | strat Sharpe |", "|---|---|---|---|"]
    for y in rep["by_year"]:
        L.append(f"| {y['year']} | {_fmt_pct(y['strat_ret'])} | {_fmt_pct(y['bench_ret'])} | {y['strat_sharpe']:.2f} |")
    L += ["", "## Per-regime (L/S vs primary)", "",
          "| regime | strat | bench | strat Sharpe | maxDD |", "|---|---|---|---|---|"]
    for r in rep["by_regime"]:
        L.append(f"| {r['regime']} | {_fmt_pct(r['strat_return'])} | {_fmt_pct(r['bench_return'])} | "
                 f"{r['strat_sharpe']:.2f} | {_fmt_pct(r['strat_maxdd'])} |")

    dsr = fw["dsr"]; pboo = fw["pbo"]
    L += ["", "## Firewall (reported, NOT enforced)", ""]
    if "dsr" in dsr:
        L.append(f"- **DSR** = {dsr['dsr']:.3f} (SR {dsr['sr']:.4f}, SR0 {dsr['sr0']:.4f}, "
                 f"N {dsr['N']}, T {dsr['T']}) — gate dsr_min 0.95 → {'PASS' if dsr['dsr']>=0.95 else 'FAIL'}")
    else:
        L.append(f"- DSR error: {dsr.get('error')}")
    L.append(f"- **PBO** = {pboo.get('pbo')} (configs {pboo.get('n_configs','?')}) — "
             f"gate pbo_max 0.20" + (f" → {'PASS' if (pboo.get('pbo')==pboo.get('pbo') and pboo['pbo']<=0.20) else 'FAIL'}" if pboo.get('pbo')==pboo.get('pbo') else " → n/a"))
    for b, d in fw["dm_vs_bench"].items():
        if "p_value" in d:
            L.append(f"- **DM vs {b}**: stat {d['stat']:.3f}, p {d['p_value']:.3f}, "
                     f"meanΔ {d['mean_diff']:.2e} (neg stat = strat better)")
    L += ["", "![equity](equity.png)", "", "![drawdown](drawdown.png)", "",
          "![paths](path_sharpe_hist.png)", ""]

    if pool_csv and Path(pool_csv).exists():
        pool = pd.read_csv(pool_csv)
        L += [f"## Pool ({len(pool)} survivor names)", "", "See `pool.csv`. First-date sample:", "",
              ", ".join(pool["symbol"].head(40).tolist()), ""]
    return "\n".join(L)
