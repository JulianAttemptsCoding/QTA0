# TACTIC-MoB — Regime-Robust Re-Validation: Full Report

**Date:** 2026-06-15 · **Author:** automated run pipeline + analysis
**Scope:** four runs — run0 (historical SPY-beater), run1 (clean CPCV), run2 (deep CPCV, *in
progress*), run2.1 (deep chronological). Each run also lives on its own git branch.

> **TL;DR.** A daily cross-sectional US-equity mixture-of-experts forecaster. The one historical
> configuration that "beat SPY" (**run0**) did so on **CAGR only** (+2.6%/yr) with a **worse
> Sharpe** and **worse drawdown**, driven by near-full-long market exposure plus a small but
> *statistically real* stock-selection edge (hit 52.5%, t=3.49). Every **regime-fair** re-validation
> (run1 clean CPCV; run2.1 deep chronological) is an **honest NEGATIVE**: no net-of-cost edge. The
> deep predictor (run2.1) is a literal **coin-flip** (hit 49.7%). **Conclusion: not deployable.** The
> historical win was modest, mostly beta, and a product of single-window selection — exactly what the
> deflated/combinatorial tests are designed to expose.

---

## 0. Runs at a glance

| run | branch | data | universe | train → OOS | validation | benchmark | verdict |
|---|---|---|---|---|---|---|---|
| **run0** | `run0` | Alpaca SIP | 83 | 2016-21 → **2023-26** | single chronological | SPY | CAGR-beat only (not Sharpe) |
| **run1** | `run1` | Alpaca SIP | 196 | CPCV(8,2) **2016-26** | 28 combinatorial folds | SPY + EW-196 | **NEGATIVE** |
| **run2** | `run2` | yfinance | 153 | CPCV(10,2) **2000-26** | 45 folds (*partial: 0-19*) | EW-pool buy-hold | *in progress* |
| **run2.1** | `run2.1`/`main` | yfinance | 153 | 2000-15 → **2018-26** | single chronological | EW-pool buy-hold | **NEGATIVE (coin-flip)** |

`run0` is the public GitHub front page (default branch `showcase-83name-2023-2026`). Run 2 is
**survivorship-biased and never deployable** — benchmarked only against its own pool so the bias
cancels in the relative comparison.

---

## 1. run0 — the historical "SPY-beater" (honest read)

f4 (hybrid cross-sectional attention), 83 Alpaca-SIP names, trained 2016-2021 / val 2022, OOS
**2023-01-12 → 2026-02-25** (782 sessions, 64,906 obs). Trading = long-only/long-flat **top-5**
buy/hold band (`k_buy=5, k_hold=8, no_trade_bps=25`), next-open fills.

| metric | run0 (net) | SPY B&H |
|---|---|---|
| Total return | **+92.9%** | +80.5% |
| CAGR | **+23.6%** | +21.0% |
| Ann. vol | 21.2% | 15.7% |
| **Sharpe** | **1.10** | **1.29** |
| Sortino | 1.59 | 1.69 |
| Max drawdown | **−26.3%** | −19.8% |
| Beta / alpha | 0.94 / **+4.3%** | — |
| Avg names / turnover | 5.0 / 6.2%/day | — |

**Predictor:** hit **52.5%**, IC +0.0143 (IR 0.073), pinball 1.686.

**Honest interpretation.** run0 *beats SPY on CAGR by +2.6%/yr* but **loses on Sharpe (1.10 vs 1.29)
and drawdown (−26% vs −20%)**. It is essentially a fully-invested (β 0.94) long book with a small,
genuine selection alpha (+4.3%/yr). The famous **2023-only 41.9% CAGR** figure was a *different,
narrower* slice: beta 1.25 × a +24% bull ≈ 30% from market + ~9% residual in a single year. Over the
full 3-year window the edge is real but **modest**, and not a risk-adjusted improvement on just owning
the index.

---

## 2. run1 — clean deployable-grade CPCV (NEGATIVE)

Alpaca SIP 2016-2026, 196 names, **CPCV(8,2) = 28 combinatorial-purged folds** (purge 2d, embargo
5d, nested 15% embargoed val), combined-OOS averaged across folds. Two books: dollar-neutral
long-short and long-flat breadth. Benchmarks SPY + equal-weight 196-universe.

| book | ann Sharpe | CAGR | vs SPY | alpha | maxDD |
|---|---|---|---|---|---|
| long_short | **−1.49** | −13.4% | −28.3% | −11.1% | −76.5% |
| long_flat | 0.77 | +11.4% | −3.5% | −0.1% (β 0.79) | −30.3% |

- Predictor OOS: IC +0.0200 (IR 0.093), hit 0.511, raw-return IC +0.0081.
- 28 CPCV path Sharpes: mean **−1.77**, frac>0 = **0.00**.
- Firewall: DSR ≈ 0 (≪0.95) FAIL; PBO 0.0 (stably bad, not overfit-lucky); DM vs SPY p≈0 *worse*.

**Read:** the cross-sectional signal is weakly positive but **economically negligible**. The L/S book
is already **negative gross** (−4.4%/yr) before costs; ~10%/yr daily-rebalance cost ⇒ −13.4% net. The
long-flat book makes money only as down-levered market beta (α≈0). Across all 28 regime-fair paths the
net Sharpe is negative. **Clean NEGATIVE — do not deploy.**

---

## 3. run2.1 — deep chronological walk-forward (NEGATIVE, coin-flip)

yfinance daily 2000-2026, 153 survivor names, **strict causal split** trained 2000-2015 / val
2016-2017 / **OOS 2018-2026** (2,112 sessions). Benchmark = equal-weight buy-hold of its **own pool**
(survivorship bias cancels). *This is the split the owner specifically asked to test ("just train
2000-15, val 16-17, test 18-26").*

| book | ann Sharpe | CAGR | bench (EWpool) | alpha | maxDD |
|---|---|---|---|---|---|
| long_short | **−1.97** | −16.6% | +19.0% | −18.1% (β 0.02) | −78.8% |
| long_flat | 0.57 | +8.1% | +19.0% | +8.6% (β 0.03 — artifact) | −35.6% |

- Predictor OOS: IC +0.0106 (IR 0.048), **hit 0.497 (below coin-flip)**, pinball 1.708.
- Firewall: DSR 0.000 FAIL; PBO 0.0; DM vs pool p≈0 *worse*.
- Per-year L/S negative **every** year; only "wins" by losing less in 2018Q4 / 2022 bear.

**Key result:** the strict chronological split returns the **same negative verdict** as the
combinatorial CPCV folds — so the bad OOS is **not** a CPCV artifact. The daily cross-sectional signal
genuinely has no net edge, even on 25 years of deep history.

### 3.1 Swapping in run0's winning trader (no retrain)
`scripts/rerun_run21_oldtrader.py` applies run0's top-5 long-only band trader to run2.1's *existing*
predictions over 2018-2026:

| | strat | SPY | EW-pool |
|---|---|---|---|
| CAGR | **+0.10%** | +14.1% | +15.3% |
| Sharpe | 0.10 | 0.81 | 0.90 |
| maxDD | −48.2% | −32% | −34% |
| alpha/yr | — | −7.4% | −10.2% |

**The trading algo was not the secret sauce.** A good low-cost trader on a no-edge predictor yields
~0% over 8 years. (It beats run2.1's brutal dollar-neutral daily L/S, but loses to owning the pool,
and even to run2.1's own broader long-flat book — concentrating into 5 names without skill is pure
idiosyncratic noise = low transfer coefficient.)

---

## 4. The core question — why run0 has skill and run2.1 is a coin-flip

### 4.1 It is the predictor, not the regime
Directional hit rate on the **identical 2023-2026 window**:

| year | run0 (83, train 2016-21) | run2.1 (153, train 2000-15) |
|---|---|---|
| 2023 | 52.2% | 49.2% |
| 2024 | 53.1% | 49.6% |
| 2025 | 52.4% | 49.1% |
| 2026 | 52.2% | 49.4% |
| **all** | **52.5% (t=+3.49, sig)** | **49.4% (t=−2.03, below coin-flip)** |

Same window, same stocks moving → the **entire gap is the predictor**. run0 has genuine cross-sectional
skill every year; run2.1 has none. (Graph: `results/run2_1_chrono/accuracy_run0_vs_run21.png`.)
*Significance is day-clustered (cross-sectional obs are not independent); per-obs t-stats hugely
overstate it.*

### 4.2 Ranked hypotheses (with tests)
1. **Stale training window / concept drift (HIGHEST).** run0 trained 2016-21 → tested adjacent
   2023-26; run2.1 trained 2000-15 → tested 2018-26 (3-11yr forward). The feature→winner map drifts
   across market-structure eras. *Test: retrain deep model rolling-recent (2014-17 → 2018).*
2. **Degraded free-data features (HIGH).** yfinance has no trade-count ⇒ feature ch05 forced equal to
   ch04 (dead duplicate). run0 on Alpaca SIP had real trade intensity. *Test: ablate ch05 on SIP.*
3. **Universe quality, not size (MEDIUM).** 153 yfinance survivors vs 83 curated SIP; breadth helps
   only if IC>0. *Test: deep model on the same 83 SIP names.*
4. **Survivorship ⇒ degenerate cross-section (MEDIUM).** Names that all survived to 2026 ⇒ 2000-15
   cross-section dominated by eventual winners ⇒ "everything rises," little relative signal.
   *Test: inspect q50 dispersion / ranking degeneracy.*
5. **Target/label mismatch (LOW-MED).** Cross-sectional demeaning differences change what sign-hit
   means. *Test: identical target definition.*

### 4.3 Theory
- **Fundamental Law** (Grinold-Kahn): `IR = IC·√breadth`. IC≈0 ⇒ no trader/universe manufactures
  return — directly explains §3.1.
- **Transfer coefficient** (Clarke-de Silva-Thorley, *FLoAM Redux*): `IR = TC·IC·√breadth`. A top-5
  long-only band has low TC (ignores 148/153 names, can't short) — fine with strong IC, disastrous
  with IC≈0.
- **Deflated Sharpe / backtest overfitting** (Bailey-López de Prado): best-of-N selection inflates the
  max Sharpe even on noise; overfit strategies underperform OOS. run0 widened 15→83 names "until it
  worked"; its DSR was never computed. CPCV + DSR here are precisely the deflated test, and they go
  negative.

---

## 5. Methodology notes
- **CPCV** (de Prado §15.1): N contiguous date groups, K-group test combos, purge + embargo, nested
  embargoed val, combined-OOS averaging. Code `src/tactic/validation/cpcv.py`.
- **S1 sanity gate**: model val pinball must beat the unconditional train-quantile baseline by ≥1%
  rel. (Passes on all runs — the model *does* condition on features; the edge is just too weak to
  monetize.)
- **Firewall (reported, not enforced per owner):** DSR (honest-N from registry), PBO (CSCV), DM test,
  CPCV path-Sharpe distribution.
- **Quarantine isolation:** Run 2 deep panel lives under `data/deep_panel/` (env `TACTIC_DATA_DIR`),
  never touches the clean curated panel; isolation enforced by tests.

## 6. Reproduce
```
# run2.1 finisher (single chronological OOS)
python vertex/finish_run.py --run_id run2_1_chrono --label RUN2.1 --deep --single \
  --panel_dir data/deep_panel/curated --pool_csv results/run2_deep_20260613/pool.csv
# run0 trader on run2.1 predictions
python scripts/rerun_run21_oldtrader.py
# accuracy graph
python scripts/graph_accuracy.py
```

## 7. Status / next
- **run2 deep CPCV** (45 folds) still computing (folds 0-19 done; 20-45 running on Vertex). This
  report + the `run2` branch will be finalized when all folds land (`finish_run.py` without
  `--single`).
- Highest-value follow-up: **hypothesis #1** — retrain the deep model on a recent rolling window and
  re-measure hit rate (isolates concept-drift from a genuinely dead signal).
