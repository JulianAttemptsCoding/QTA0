# QTA0 — Quant Trading Algorithm 0: Full Report

**Project:** TACTIC-MoB v4 — a daily cross-sectional US-equity mixture-of-experts forecaster, built,
trained on Vertex AI, and **regime-robustly re-validated**.
**Last updated:** 2026-06-17 · automated run pipeline + analysis.

> **TL;DR.** One historical config "beat SPY" (**run0**) — but on **CAGR only** (+2.6%/yr), with a
> *worse* Sharpe (1.10 vs 1.29) and *worse* drawdown, driven mostly by near-full-long market beta plus
> a small but *statistically real* stock-selection edge (test hit 52.5%, t=+3.49). Every
> **regime-fair** re-validation (run1 clean CPCV; run2 deep CPCV; run2.1 deep chronological) is an
> **honest NEGATIVE**: no net-of-cost edge. Test directional accuracy degrades cleanly the further the
> test set sits from training in time — 52.5% → 51.1% → 50.5% → **49.7%** — the fingerprint of concept
> drift, not a CPCV artifact. **Conclusion: not deployable.** The historical win was modest, mostly
> beta, and a product of single-window selection — exactly what the deflated/combinatorial tests expose.

---

## How to read this repository (learning guide)

| You want to learn… | Read |
|---|---|
| The whole story + every result | **this file (REPORT.md)** |
| Raw chronological thinking, every bug, every idea | **notes.md** |
| The original build spec (18-ch features, experts, decision layer) | `PLAN.md` |
| The re-validation design (why CPCV, two runs, benchmarks) | `revision_plan.md` |
| Data-source research (why free deep history is survivorship-biased) | notes.md §2026-06-13 |
| The code | `src/tactic/` (models, validation, backtest, reports) |
| How training ran in the cloud | `vertex/` |
| Per-run numeric outputs | `results/<run>/RESULTS_*.md`, `report.json` |
| One-off analysis scripts | `scripts/` |

**Repo map.** `src/tactic/` = library (features, experts f3/f4, train, CPCV, backtest, firewall,
reports). `vertex/` = cloud submit/finish. `configs/v1.yaml` = all hyperparameters. `data/` =
panels (gitignored, local only). `results/` = run artifacts. `tests/` = the test suite.

---

## 1. What the system actually is (algos + data)

**Data.** Two strictly-isolated panels:
- **Clean / deployable:** Alpaca SIP, delisting-aware, 2016-01 → 2026-06, up to 196 liquid US names
  (run0 used an 83-name subset; run1 the full 196). Daily OHLCV + opening/closing auctions.
- **Deep / exploratory:** yfinance free daily 2000 → 2026, **153 survivor names** (continuous-data
  filter). **Survivorship-biased** (dead tickers absent) → benchmarked ONLY against its own pool, never
  SPY, never deployable. Lives under `data/deep_panel/` (env `TACTIC_DATA_DIR`); a test enforces it
  never touches the clean panel.

**Features (PIT, leakage-checked).** Per name per day: 18 sequence channels (length-64 daily window:
returns, vol, ranges, volume/vw, trade-intensity, momentum, etc.), 12 static descriptors, 5
market-state context features. Leakage QA: features at *t* are invariant to future bars (max-abs-diff
0.0 on a truncation test). On the deep panel yfinance has **no trade-count**, so channel ch05 is forced
equal to ch04 (a dead duplicate) — a known feature degradation.

**Experts.**
- **f3 (CI-TCN):** causal dilated TCN (6 blocks, k5, dilations 1…32, 64ch) + non-crossing quantile head.
- **f4 (hybrid cross-sectional attention):** f3 encoder → `[enc, static]` tokens → ONE multi-head
  self-attention (4 heads, d64) **across the N entities trading that day** → quantile heads. ~0.26M
  params. f4 is the headline model; an ablation showed the attention layer is statistically justified
  (DM f4 vs f3 p=0.0099) — the signal is **cross-sectional**, not per-asset temporal.

**Target.** Vol-standardized open(t+1)→open(t+2) return (1-day hold, next-open fills), cross-sectionally
demeaned. Loss = pinball (quantile) loss; q50 is the point forecast / ranking score.

**Trading books (the "algo" on top of the predictor).**
- **long-only top-K buy/hold band** (run0's trader): hold top-5, replace only when a name falls out of
  top-8, 25bps no-trade band, next-open fills. Low turnover, long-biased.
- **dollar-neutral long-short** (k=35 each side): daily rebalance — high transfer coefficient but
  high cost.
- **long-flat breadth** (k_buy=20 / k_hold=60): broad, low-turnover, down-levered long.

Costs from a fitted spread surface; all fills at the next opening auction; NAV identity enforced.

---

## 2. Runs at a glance

| run | data | universe | train → OOS | validation | benchmark | test hit | verdict |
|---|---|---|---|---|---|---|---|
| **run0** | Alpaca SIP | 83 | 2016-21 → **2023-26** | single chronological (adjacent) | SPY | 52.5% (t=+3.49) | CAGR-beat only (not Sharpe) |
| **run1** | Alpaca SIP | 196 | CPCV(8,2) **2016-26** | 28 combinatorial folds | SPY + EW-196 | 51.1% (t=+5.36) | **NEGATIVE** |
| **run2** | yfinance | 153 | CPCV(10,2) **2000-26** | 45 combinatorial folds | EW-pool buy-hold | 50.5% (t=+5.91) | **NEGATIVE** |
| **run2.1** | yfinance | 153 | 2000-15 → **2018-26** | single chronological (distant) | EW-pool buy-hold | 49.7% (t=−1.56) | **NEGATIVE (coin-flip)** |

Run 2 / 2.1 are **survivorship-biased, never deployable** — benchmarked only against their own pool so
the bias cancels in the relative comparison.

---

## 3. run0 — the historical "SPY-beater" (honest read)

f4, 83 Alpaca-SIP names, trained 2016-2021 / val 2022, OOS **2023-01-12 → 2026-02-25** (782 sessions,
64,906 obs). Trading = long-only top-5 buy/hold band.

| metric | run0 (net) | SPY B&H |
|---|---|---|
| Total return | **+92.9%** | +80.5% |
| CAGR | **+23.6%** | +21.0% |
| Ann. vol | 21.2% | 15.7% |
| **Sharpe** | **1.10** | **1.29** |
| Max drawdown | **−26.3%** | −19.8% |
| Beta / alpha | 0.94 / **+4.3%** | — |

Predictor: test hit **52.5%** (day-clustered t=+3.49, significant), IC +0.0143 (IR 0.073), pinball 1.686.

**Honest interpretation.** run0 beats SPY *on CAGR by +2.6%/yr* but **loses on Sharpe and drawdown**. It
is a near-fully-invested (β 0.94) long book with a small, genuine selection alpha (+4.3%/yr). The famous
2023-only 41.9% CAGR was a narrower slice: β 1.25 × a +24% bull ≈ 30% market + ~9% residual in one year.
Over the full window the edge is real but **modest**, and not a risk-adjusted improvement on the index.

---

## 4. run1 — clean deployable-grade CPCV (NEGATIVE)

Alpaca SIP 2016-2026, 196 names, **CPCV(8,2) = 28 combinatorial-purged folds** (purge 2d, embargo 5d,
nested 15% embargoed val), combined-OOS averaged. Benchmarks SPY + EW-196.

| book | ann Sharpe | CAGR | vs SPY | alpha | maxDD |
|---|---|---|---|---|---|
| long_short | **−1.49** | −13.4% | −28.3% | −11.1% | −76.5% |
| long_flat | 0.77 | +11.4% | −3.5% | −0.1% (β 0.79) | −30.3% |

- Predictor: IC +0.0200 (IR 0.093), test hit 0.511 (t=+5.36). Val S1 gate pass **22/28** folds.
- 28 CPCV path Sharpes: mean **−1.77**, frac>0 = **0.00**.
- Firewall: DSR ≈ 0 (FAIL), PBO 0.0 (stably bad, not overfit-lucky), DM vs SPY p≈0 *worse*.

**Read:** the signal is weakly positive but **economically negligible**. L/S is negative gross before
costs; long-flat makes money only as down-levered beta (α≈0). **Clean NEGATIVE — do not deploy.**

---

## 5. run2 — deep combinatorial CPCV, 45 folds (NEGATIVE)

yfinance 2000-2026, 153 survivors, **CPCV(10,2) = 45 folds**, combined OOS = **995,418 obs over 6,506
sessions** (2000-07-25 → 2026-06-08). Bench = EW buy-hold of own pool.

| book | ann Sharpe | CAGR | bench (EWpool) | alpha | beta | maxDD |
|---|---|---|---|---|---|---|
| long_short | **−1.83** | −15.6% | +14.9% | −16.7% | 0.01 | **−98.8%** |
| long_flat | 0.71 | +9.8% | +14.9% | +10.8% | −0.02 | −43.7% |

- Predictor: test hit **0.5054 (day-clustered t=+5.91)**, IC +0.0214 (IR 0.109), pinball 1.674.
- Val S1 gate pass only **32/45** folds (13 fail — model barely beats a constant-quantile guess).
- **45 CPCV path Sharpes: mean −2.66, frac>0 = 0.00** — every regime-fair path negative.
- L/S negative **every year** 2000-2026. long_flat "alpha" is a β-artifact (β≈0); it still *loses*
  to owning the pool (+9.8% vs +14.9%).
- Firewall: DSR 0.000 (FAIL), PBO 0.0, DM vs pool p≈0 *worse*.

**The key nuance.** run2's combined-OOS predictor is *statistically positive* (t=+5.91) — unlike
run2.1's coin-flip. This is **not** deployable skill: in CPCV every test fold is predicted by a model
trained on **purged-adjacent** date groups (temporally bracketing the test), so each prediction enjoys a
near-in-time training set. Remove that adjacency (run2.1's honest forward split) and the edge vanishes.
And even the CPCV "edge" is economically dead: IC +0.0214 → Fundamental Law gives ~zero portfolio IR;
net Sharpe −1.83. **t=+5.91 is significant only because n=6,506 days is huge; 50.5% is unmonetizable.**

---

## 6. run2.1 — deep chronological walk-forward (NEGATIVE, coin-flip)

yfinance 2000-2026, 153 survivors, **strict causal split** trained 2000-2015 / val 2016-2017 / **OOS
2018-01-11 → 2026-06-08** (2,112 sessions). This is the split the owner specifically asked to test.

| book | ann Sharpe | CAGR | bench (EWpool) | alpha | beta | maxDD |
|---|---|---|---|---|---|---|
| long_short | **−1.97** | −16.6% | +19.0% | −18.1% | 0.02 | −78.8% |
| long_flat | 0.57 | +8.1% | +19.0% | +8.6% | 0.03 | −35.6% |

- Predictor: test hit **0.4968 (below coin-flip, day-clustered t=−1.56)**, IC +0.0106 (IR 0.048),
  pinball 1.708.
- Firewall: DSR 0.000 (FAIL), PBO 0.0, DM vs pool p≈0 *worse*. L/S negative every year.

**Key result:** the strict chronological split returns the **same negative verdict** as the
combinatorial folds — so the bad OOS is **not** a CPCV artifact. The daily cross-sectional signal has no
net forward edge even on 25 years of deep history.

### 6.1 Swapping in run0's winning trader (no retrain)
`scripts/rerun_run21_oldtrader.py` applies run0's top-5 long-only band to run2.1's *existing* predictions
over 2018-2026: **CAGR +0.10%**, Sharpe 0.10, maxDD −48.2%, alpha −7.4%/yr vs SPY, −10.2%/yr vs EW-pool.
**The trading algo was not the secret sauce.** A good low-cost trader on a no-edge predictor yields ~0%
over 8 years (Fundamental Law: IC≈0 ⇒ IR≈0; concentrating into 5 names without skill = pure noise = low
transfer coefficient).

---

## 7. Train / Val / Test prediction accuracy (cross-cut)

> **Honest scope caveat.** Directional **hit rate** was persisted only for the **test** split, and **no
> model checkpoints were saved** → train/val hit cannot be recomputed without retraining. What IS saved:
> test hit (all runs), val pinball + the S1 gate (all runs), and train pinball (run0, run2.1 only — the
> single/walk runs that logged loss curves; the CPCV runs logged only per-fold val pinball in meta).

### 7.1 Test directional hit — the one valid cross-run accuracy metric

| run | split type | train→test temporal gap | **test hit** | day-clust t | IC |
|---|---|---|---|---|---|
| run0 | chronological, **adjacent** | 2016-21 → 2023-26 (~1 yr) | **52.5%** | +3.49 | 0.0143 |
| run1 | CPCV, **interleaved** | purged ±2-5d neighbors | 51.1% | +5.36 | 0.0200 |
| run2 | CPCV, **interleaved** | purged ±2-5d neighbors | 50.5% | +5.91 | 0.0214 |
| run2.1 | chronological, **distant** | 2000-15 → 2018-26 (3-11 yr fwd) | **49.7%** | −1.56 | 0.0106 |

**Dominant pattern: test accuracy falls monotonically with train→test temporal distance.** Adjacent
chronological (run0) → genuine edge. Interleaved CPCV (run1/run2) → whisker-thin positive. Distant
forward (run2.1) → coin-flip. This is the **concept-drift / covariate-shift signature**.

### 7.2 Val skill — the S1 gate (model pinball vs naive train-quantile baseline, within-split = valid)

| run | val_pinball | baseline_pinball | S1 margin | S1 pass |
|---|---|---|---|---|
| run0 | 1.669 (best) | — | passed | ✅ |
| run1 | 1.4705 | 1.4874 | +1.13% | **22/28 folds** |
| run2 | 1.4455 | 1.4617 | +1.11% | **32/45 folds** |
| run2.1 | 1.4268 | — | passed | ✅ |

The conditional model beats a constant-quantile guess by ~1% on val — and fails the gate outright in
13/45 (run2) and 6/28 (run1) folds. The val signal itself is marginal.

### 7.3 Train vs val pinball (run0, run2.1) — with a scale warning

| run | train pin | val pin | test pin |
|---|---|---|---|
| run0 | 1.601 | 1.669 (best) / 1.749 (final) | 1.686 |
| run2.1 | 1.672 | 1.427 | 1.708 |

⚠️ **Pinball levels are NOT comparable across splits.** Pinball scales with each period's return
volatility, and the splits cover different calendar eras (run2.1's val 2016-17 is calm; its train
2000-15 includes the dot-com bust and the GFC). run2.1's val "improvement" (1.43 < 1.67) is a
period-scale artifact, **not** generalization. Only within-split model-vs-baseline (the S1 gate, §7.2) is
interpretable. run0's train 1.601 vs test 1.686 ≈ 5% optimism gap = mild, benign overfit (test hit still
good).

---

## 8. Core analysis — why skill degrades with temporal distance

1. **Concept drift / stale training window (HIGHEST confidence).** The feature→winner map is
   non-stationary across market-structure eras. CPCV's positive test t is partly a *near-in-time
   training advantage* (every fold trains on data bracketing its test groups). Honest forward deployment
   (run2.1) removes it → edge gone. **CPCV measures conditional-on-recent skill, not deployable forward
   skill** — the central methodological lesson of this project.
2. **Statistical ≠ economic significance.** run1/run2 test t exceed +3 (run2 +5.91 > Harvey-Liu-Zhu's
   t>3 multiple-testing bar), yet hit is 50.5-51.1% and IC≈0.02 → near-zero portfolio IR, net Sharpe
   −1.8. The t is huge only because n_days is huge. DSR≈0 deflates it to nothing.
3. **The trading algo is second-order.** run0's exact trader on run2.1's preds = +0.10% CAGR.
   Predictor skill is first-order; once IC≈0, no book recovers return.
4. **Degraded free-data features (deep runs).** yfinance has no trade-count ⇒ ch05 is a dead duplicate;
   run0 (Alpaca SIP) had real trade intensity. Plausible contributor to the deep predictor's weakness.
5. **Survivorship → degenerate cross-section (deep runs).** A pool that all survived to 2026 has a
   2000-15 cross-section dominated by eventual winners ("everything rises"), eroding relative signal.

---

## 9. Methodology

- **CPCV** (de Prado §15.1): N contiguous date groups, K-group test combos, purge + embargo, nested
  embargoed val, combined-OOS averaging. Code `src/tactic/validation/cpcv.py`,
  `src/tactic/models/train_cpcv.py`.
- **S1 sanity gate:** model val pinball must beat the unconditional train-quantile baseline by ≥1% rel.
- **Firewall (reported, not enforced):** DSR (honest-N from registry), PBO (CSCV), DM test, CPCV
  path-Sharpe distribution.
- **Quarantine isolation:** the deep panel never touches the clean panel; enforced by a test.
- **No OOS peeking:** models/hyperparameters were never tuned on the test split.

---

## 10. Theory + literature

- **López de Prado, *Advances in Financial ML* (2018), ch.7,12** — CPCV, purge/embargo; OOS as a
  *distribution* of paths; adjacency leakage inflates OOS. Grounds the CPCV-vs-forward gap (§8.1).
- **Bailey, Borwein, López de Prado, Zhu (2014), "Probability of Backtest Overfitting"** + **Bailey &
  López de Prado (2014), "Deflated Sharpe Ratio"** — best-of-N selection inflates Sharpe on noise; our
  DSR≈0, honest-N=17 deflate the "significant" CPCV t to nothing.
- **Harvey, Liu, Zhu (2016), "…and the Cross-Section of Expected Returns"** — t>3.0 bar under multiple
  testing; run1/run2 clear it on directional t but fail on economic magnitude.
- **Gu, Kelly, Xiu (2020), "Empirical Asset Pricing via ML" (RFS)** — canonical train/val/test scheme;
  ML edges decay OOS and recent data dominates → supports the concept-drift reading of run2.1.
- **Grinold (1989) + Grinold-Kahn; Clarke-de Silva-Thorley (2002), "FLoAM Redux"** — IR = TC·IC·√breadth;
  IC≈0.02 ⇒ no trader/universe manufactures return; a top-5 long-only band has low TC.
- **Quiñonero-Candela et al. (2009), *Dataset Shift in ML*** + **Hastie-Tibshirani-Friedman, *ESL*** —
  train-error optimism + covariate shift = the temporal-distance degradation in §7.1.
- **Arnott, Harvey, Markowitz (2019), "A Backtesting Protocol in the Era of ML"** — mirrors our
  firewall checklist.

---

## 11. Where I am unsure

1. **Train/val directional hit is unknown** — not persisted, no checkpoints. All train/val "accuracy"
   here is pinball-based proxy, not hit. (Biggest gap; fixable only by retraining with hit logging.)
2. **How much of run2's +5.91 t is genuine vs CPCV adjacency advantage** — I lean mostly adjacency
   (run2.1 forward = coin-flip is the tell), but can't quantify the split without a rolling-recent
   retrain (hypothesis #1's direct test).
3. **IC higher for run1/run2 than run0 yet test hit lower** — likely because rank-IC and sign-hit weight
   different parts of the cross-section; not fully pinned down.
4. **Pinball scale confound** — confident it breaks cross-split comparison; not cleanly de-confounded.

---

## 12. Reproduce

```bash
# run2 deep CPCV finisher (after 45 Vertex folds land in GCS)
python vertex/finish_run.py --run_id run2_deep_20260613 --label RUN2 --deep \
  --panel_dir data/deep_panel/curated --pool_csv results/run2_deep_20260613/pool.csv
# run2.1 chronological finisher
python vertex/finish_run.py --run_id run2_1_chrono --label RUN2.1 --deep --single \
  --panel_dir data/deep_panel/curated --pool_csv results/run2_deep_20260613/pool.csv
# run0 trader on run2.1 predictions
python scripts/rerun_run21_oldtrader.py
# accuracy graph (run0 vs run2.1 over OOS time)
python scripts/graph_accuracy.py
```
> Note: local `gcloud` may crash with "untrusted mount point" if an OpenAI/Codex `bin` is on PATH;
> strip it from PATH for gcloud, or use `gsutil` for GCS transfers.

---

## 13. Verdict

A genuine but **modest and mostly-beta** historical win (run0) that does **not** survive regime-fair
re-validation. Across clean CPCV (run1), deep CPCV (run2), and deep chronological (run2.1), the daily
cross-sectional signal is at best a statistically-significant-but-economically-dead whisker, and at worst
a coin-flip — with test accuracy degrading exactly as concept-drift theory predicts. **Not deployable.**

**Highest-value follow-up:** hypothesis #1 — retrain the deep model on a recent rolling window
(e.g. 2014-17 → test 2018) and re-measure hit + log train/val hit, to cleanly separate concept-drift
from a genuinely dead signal and fill the §7 train/val gap.
