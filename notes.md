# TACTIC-MoB v4 — Build / Train / Backtest Notes

Append-only working log. Newest entries at the bottom. Nothing is ever deleted.

---

## 2026-06-12 — Session start: finish model → train/val/test → full backtest vs SPY

### Goal
Implement remaining PLAN.md phases, train experts (train/val/test), full backtest on Vertex,
QA/QC every step, produce QuantConnect-style results + graphs, compare vs SPY buy-and-hold OOS.

### Environment confirmed
- Python 3.12, torch 2.4.1 **CPU only** (cuda=False) locally → neural training is slow locally;
  use Vertex GPU image (`pytorch-xla.2-4`) for full f3/f4 training.
- lightgbm 4.6.0, matplotlib 3.10.8, bidask, scipy, sklearn, statsmodels, pywt all present.
- gcloud SDK 555, **active account juliansjuan08@gmail.com**, project
  `project-c779f701-1a49-4a58-b54`, region us-central1.
- GCS bucket `gs://gmda-vertex-c779f701-uscentral1` readable; has configs/ data/ packages/
  runs/ reports/ etc. → Vertex submission path viable.

### Status at start (already done + tested, 135 tests green, pushed 8ea7789)
- Phase 0 core; Phase 1 ingestion (live-verified); Phase 2 universe+delistings; Phase 3
  spreads+cost model; Phase 4 labels; Phase 5 diagnostics + gate G1.
- Real slice ingested: 15 symbols (SPY QQQ AAPL MSFT NVDA AMZN GOOGL META JPM XOM JNJ WMT PG
  KO HD), daily bars + auctions 2019–2023.

### Plan for this session (execution order)
1. `features/build_features.py` — 18-ch sequence tensor + static[12] + market-state[5], PIT.
2. `models/panel_dataset.py` — one-date batches, masks.
3. Experts: f2 LightGBM (CPU, fast), f3 CI-TCN, f4 hybrid attention (quantile heads, pinball).
4. `models/train.py` — walk-forward annual refits, train/val/test, early stopping, loss curves.
5. Decision layer: score → band → weights, turnover budget, costs (simplify GP/RC-Kelly first,
   then add if time).
6. `backtest/engine.py` — daily fills at next-open auction, explicit costs, NAV identity.
7. Metrics + plots: equity vs SPY, drawdown, train/val loss/epoch, test predictor stats
   (pinball, IC, hit-rate, calibration), QuantConnect-style table (CAGR/Sharpe/Sortino/maxDD/...).
8. Validation: PBO/DSR/SPA where feasible.
9. Scale full training to Vertex; wait with ScheduleWakeup cooldowns; retrieve; final report.

### Design decisions / simplifications (will mark spec-deviations explicitly)
- Iterate locally on a SMALL config (few epochs, 15-symbol universe) to QA correctness and get
  REAL OOS-vs-SPY numbers fast; then scale to Vertex for the full run.
- Risk-constrained Kelly (cvxpy) and full GP partial-adjustment are complex; first ship a
  capped score-weight + buy/hold-band + turnover-cap decision layer (a valid PLAN.md fallback,
  §12.6 "Fallback if infeasible: capped score weights"), then upgrade.

---

## Phase 4 features — built + QA'd

- `features/build_features.py` implemented: 18 sequence channels, 12 static u, 5 market-state m.
- **Bug found & fixed:** bars ingest manifest keyed only on `symbol:adjustment`, ignoring date
  range → re-ingesting a wider range hit the stale 40-day smoke cache for AAPL/SPY. Fixed:
  manifest now records {status,start,end} and only resumes if the cached range covers the
  request. Re-ingested: all 15 symbols now have the full 1258 sessions (2019-01-02..2023-12-29).
- **Bug found & fixed:** `_market_state` raised "All arrays must be of same length" (breadth
  pivot had a different date index). Fixed by reindexing all M-series to `piv.index`.
- Rebuilt full panel: spreads 18,870 entity-days; labels 18,870 (99.94% auction); features
  18,870 (16,755 fully populated, all 18 ch + 12 u + 5 m non-null).
- **Leakage QA (sec6.4):** recomputed AAPL channels on full series vs truncated@600; 2880/2880
  finite cells, **max abs diff = 0.0** → features at t invariant to future bars. PIT confirmed.
- Note/deviation: only SPY,QQQ pass the universe S&P-500-membership filter (no wiki membership
  table loaded), giving 1537 tradable entity-days. For THIS 15-liquid-name modeling experiment
  the trainable universe = all 15 names passing price/ADV/age (they are all S&P 500 mega-caps);
  the membership gate is a realism constraint relaxed here and noted.

---

## Modeling stack — built + smoke-validated

- `panel_dataset.build_panel`: one-date batches; 1115 dates, 15 names/date, 2019-07-25..2023-12-27
  (starts after 64-seq + 252-day mom warmup). Target = vol-standardized open(t+1)->open(t+2)
  return (OO structure, 1-day hold), PIT-safe (no overlap with x_seq).
- `experts/tcn_ci.py` f3: causal dilated TCN (6 blocks, k5, dil[1..32], 64ch, GELU, dropout0.1)
  + non-crossing QuantileHead (median + softplus increments).
- `experts/hybrid_xs.py` f4: f3 encoder -> [enc,u] tokens -> ONE MHSA(4 heads,d64) over the
  date's N_t entities + residual -> quantile heads. **Params = 259,400** (~0.26M; PLAN target
  0.15-0.25M, under the config param_cap 300k). Forward shape (N,7), non-crossing verified.
- `models/train.py`: walk-forward split train(<=2021)/val(2022)/test-OOS(2023) with purge2+embargo5;
  channel standardization fit on TRAIN only (fit-scope); AdamW+cosine, grad-clip 1.0, early-stop
  patience 5 on val pinball; per-epoch train/val loss recorded; seeds quantile-averaged.
- Smoke (3 epochs, 1 seed): 616/244/241 train/val/test batches; train pinball 1.737->1.647,
  val 1.672->1.659->1.776 (early-stop would trigger). ~23 s/epoch CPU. OOS preds: 3615 rows /241
  dates. Artifacts in reports/runs/<run_id>/ (loss_history.csv, oos_predictions.parquet).
- Compute estimate: full local run 25 epochs x 5 seeds ~= 50 min CPU. Vertex GPU will be faster;
  will submit the full f4 + f3-ablation run to Vertex and wait with cooldowns.

---

## Backtest + results pipeline + Vertex submission

- `backtest/run.py` + `reports/results.py`: OOS backtest vs SPY, QuantConnect-style metrics,
  4 plots (equity, drawdown, loss curve, calibration), RESULTS.md + metrics.json.
- **Smoke (3-epoch) OOS 2023:** TACTIC-MoB net 46.7% total / Sharpe 1.99 vs SPY 22.9% / 1.68;
  alpha 13.5%, beta 1.28; rank-IC 0.025. (Undertrained; full run pending.)
- **Bug fixed:** predictor coverage compared standardized-scale quantiles to RAW returns -> 100%.
  Now scored against the stored standardized `y_true` (pinball + coverage correct scale).
- config.py made container-safe: `$TACTIC_CONFIG_DIR/DATA_DIR/REGISTRY_DIR/REPORTS_DIR` overrides
  + repo/CWD fallbacks (so the pip-installed package finds configs on a Vertex worker).

### Vertex
- `models/vertex_entry.py`: downloads panel+config from GCS, trains, renders, uploads artifacts.
- `vertex/train_on_vertex.py`: uploads curated panel + v1.yaml + sdist, submits custom job.
- **Bug fixed:** Windows subprocess can't exec gcloud/gsutil `.cmd` without a shell -> switched to
  `gcloud storage cp` via shell=True.
- **SUBMITTED full f4 run on Vertex:** job id `5641414924543459328`,
  run_id `f4_vertex_20260612_104701`, machine n1-standard-16, image pytorch-xla.2-4,
  epochs=25 seeds=5. Artifacts -> gs://gmda-vertex-c779f701-uscentral1/runs/f4_vertex_20260612_104701.
- Parallel safety net: full f4 (25ep x 3seed) also training locally for immediate results.
- Next: wait (cooldown) for Vertex job; poll status; download artifacts; render; compare to local.

---

## Full local f4 run — HONEST RESULT (15-name universe)

Run f4_20260612_104415 (25 epochs, 3 seeds, early-stopped, val_pinball 1.633). OOS = 2023.

| Metric | TACTIC-MoB (net) | SPY B&H |
|---|---|---|
| Total return | 20.7% | 22.9% |
| CAGR | 21.8% | 24.1% |
| Sharpe | 1.30 | 1.68 |
| Sortino | 2.02 | 2.75 |
| Max drawdown | -16.7% | -9.6% |
| Win rate | 53.1% | 51.5% |

Predictor (OOS): pinball 1.655, **rank-IC 0.0023 (IR 0.007)**, hit 47.6%, coverage 73.7%/85.9%.

**Interpretation (important, not cherry-picked):** with proper training the model minimizes val
pinball but OOS cross-sectional **IC ~ 0** -> NO reliable edge on 15 mega-caps; it slightly
*under*performs SPY net. The earlier 3-epoch "46% / Sharpe 1.99" was noise from an undertrained
model that happened to overweight 2023 winners. This is the anti-overfitting framework working
as designed: a tiny universe has almost no cross-sectional breadth (Fundamental Law: IR ~ IC*sqrt(BR)).
Coverage 74/86% shows the quantile heads are roughly calibrated. Results copied to
results/f4_local_15names/ (committed).

**Action:** scale the universe to ~80 liquid S&P names for real cross-sectional breadth; retrain.
Ingest of ~80 names launched in background.

## Vertex — fix + resubmit
- First full job FAILED: image pytorch-xla.2-4 is **Python 3.10**, pyproject required >=3.11 ->
  pip refused. Vertex also installs the package `--no-deps`.
- Fix: requires-python ">=3.10"; vertex_entry now pip-installs its light runtime deps
  (pandas/pyarrow/pyyaml/scipy/matplotlib) defensively. Rebuilt + reuploaded sdist.
- Smoke validation job submitted (job 7466498673535352832, n1-standard-4, epochs8 seeds2) against
  the already-uploaded 15-name data to confirm the fixed path returns artifacts.

---

## Scaled to 83-name universe + Vertex iteration

- Ingested 83 liquid S&P names (bars+auctions 2019-2023, 208,828 price rows). Rebuilt panel:
  spreads/labels/features = 104,414 entity-days, 83 entities (features 14.5s).
- Vertex smoke (fixed py-version) got PAST pip install and ran, but hit **NumPy 2.0 ABI break**:
  my defensive `pip install pandas pyarrow scipy matplotlib` pulled numpy 2.x, incompatible with
  the image's prebuilt torch/pyarrow ("_ARRAY_API not found"). Cancelled it.
- Fix: defensive install now just `pyyaml matplotlib "numpy<2"` (image already has pandas/
  pyarrow/numpy/torch compiled together); plotting falls back to local render of downloaded preds.
- **Submitted 83-name training on Vertex:** job 2698031093080129536, run_id f4_vertex_20260612_111310,
  n1-standard-16, epochs=20 seeds=3. data_uri repo_inputs/20260612_111310,
  out_uri runs/f4_vertex_20260612_111310. ~83 names should give real cross-sectional breadth.

---

## VERTEX SUCCESS — 83-name f4 headline result

- Job 2698031093080129536 **SUCCEEDED** (the "ERROR" log lines were pip's non-fatal dependency-
  resolver warning). numpy<2 fix worked; trained on 83 names, in-container render produced plots.
- Downloaded to results/f4_vertex_83names/. **OOS 2023 vs SPY:**
  CAGR 41.9% vs 24.1%, Sharpe 1.82 vs 1.68, maxDD -9.2% vs -9.6%, alpha +9.1%, beta 1.25.
  Predictor: pinball 1.670, **rank-IC 0.0157 (IR 0.072)**, hit 52.4%, coverage 76.3%/87.8%
  (well-calibrated, slightly conservative). 20,003 OOS obs.
- Train/val curves (3 seeds, early-stop @12/6/8 epochs): train pinball 1.69->1.58, val ~1.68.
- **Breadth thesis confirmed:** 15 names IC~0 (trails SPY) vs 83 names IC 0.0157 (beats SPY).
  Fundamental Law IR~IC*sqrt(breadth). The 3-epoch 15-name "win" was undertrained noise.
- Wrote results/FINAL_REPORT.md (QuantConnect-style table + method + honest caveats) + README headline.
- Submitted f3 ablation (no cross-sectional attention) job 6993902187638161408 for the f4-vs-f3
  comparison (gate G5). Polling.
- Honest caveats logged: 83 << 400-700 spec breadth; single OOS year (2023, momentum-friendly);
  CPCV/DSR/SPA at scale + RC-Kelly/GP decision layer are the documented next steps before any
  deployability claim.

---

## f3 ablation result + gate G5 (f4 vs f3)

- f3 ablation job 6993902187638161408 SUCCEEDED on Vertex. Downloaded to results/f3_vertex_83names/.
- **f4 (attention) vs f3 (no attention), 83 names, OOS 2023:**
  f4: CAGR 41.9%, Sharpe 1.81, IC +0.0157.  f3: CAGR 1.7%, Sharpe 0.19, IC -0.0081 (negative).
- **Gate G5 PASS:** DM on daily net P&L f4 vs f3 = stat -2.60, **p=0.0099** -> attention layer
  statistically justified. Confirms PLAN thesis: signal is cross-sectional, not per-asset temporal.
- **f4 vs SPY DM: stat -1.38, p=0.17** -> economically beats SPY (CAGR/Sharpe) but NOT stat-sig
  over one 241-day OOS year. Honest: PLAN-grade G3 needs multi-year + Romano-Wolf + DSR at full
  breadth (library functions exist; not yet run at scale). Logged in FINAL_REPORT section 2.5.

## SUMMARY OF DELIVERABLE
Full pipeline ingest->features->experts->train(Vertex)->OOS backtest vs SPY, with graphs, train/val
curves, predictor stats, ablation, and DM significance. f4 beats SPY economically (CAGR 41.9% vs
24.1%, Sharpe 1.82 vs 1.68, alpha +9.1%) and beats f3 significantly (G5 p=0.0099). Breadth thesis
demonstrated (15 vs 83 names). All honest caveats stated. 141 tests green. Pushed to GitHub.

---

## Deflated Sharpe Ratio (gate G8) on 3yr OOS — FAILS (honest negative result)

DSR on 3yr daily net returns (SR_daily 0.070, skew 0.21, kurtosis 8.35, T=765; var_sr_trials est
from observed variant Sharpes). DSR by N: 5->0.73, 10->0.57, 20->0.43, 50->0.28, 100->0.19. All
< 0.95 -> **G8 FAIL**. SR0(N=10)=0.064 daily (~1.01 ann) ~= observed -> Sharpe indistinguishable
from best-of-N luck. Strategy Sharpe 1.12 < SPY 1.36 anyway. Ledger auto-N=7 (undercounts; Vertex
trials logged ephemerally). CONCLUSION: positive raw-return alpha but NO risk-adjusted skill that
survives multiple-testing deflation. Valid PLAN negative result (do not deploy). This is exactly
what DSR/anti-overfitting gates exist to catch.

---

## 2026-06-12 — UNIVERSE + HISTORY EXPANSION (user req: ignore gates, note only)

**Goal:** train data as early as possible; OOS = 2022-01 → now (~2026-06); clean train/val/test;
≥150 stocks/ETFs; train until val pinball rises while train falls (overfit divergence, keep best-val);
all heavy on Vertex; gates ignored (noted only).

**Plan/split (new):** train ≤2020-12-31, val 2021 (calendar), test/OOS 2022-01-01 → now.
Earliest data 2016-01-01 (cfg panel.start; Alpaca SIP floor). ~5yr train / 1yr val / 4.4yr OOS.
Purge=2 + embargo=5 between regimes (unchanged).

**Universe:** expanded 83 → **196** liquid US large/mid-caps + 17 ETFs (incl SPY benchmark),
broad sector coverage, no leverage/inverse. List = candidate_symbols.LIQUID_BIG.

**Steps:** (1) ingest Alpaca SIP bars+auctions 2016→now for 196 syms; (2) rebuild costs→labels→
features; (3) QA panel split; (4) raise epochs/patience to expose overfit divergence, set new
split, submit f4 to Vertex; (5) poll→download→backtest OOS 2022→now vs SPY (overall + annualized)
+ predictor stats + graphs. Gates computed & noted, not enforced.

### Ingest + panel rebuild results
- Bars: 196 syms, **2016-01-04 → 2026-06-11**, 1,022,982 rows, SPY 2625 days, 0 syms <500 days.
- Auctions: 511,248 rows through 2026-06-11.
- costs/labels/features rebuilt: 511,491 entity-day rows, 39 feat cols.
- **Bugfix:** features ch06 had 398 `-inf` from `log(vw)` where vw=0. Guarded vw<=0/NaN → close.
  Rebuilt → inf=0.
- **Panel split QA (build_panel + _split, purge2+embargo5):**
  - train 2016-07-26 → 2020-12-31 = 1118 date-batches
  - val   2021-01-13 → 2021-12-31 = 245
  - OOS   2022-01-12 → 2026-06-09 = 1105 (216,280 entity-rows)
  - entities/day min/med/max = 192/195/196 (breadth 83→~195).
- Warmup start 2016-07-26 (126d from 2016-01-04). Clean. Ready for Vertex.

### Vertex job submitted (overfit-divergence run)
- run_id **f4_vertex_20260612_230413**, jobId 7559341435384758272, n1-standard-16, pytorch-xla py310.
- f4, epochs=60, seeds=3, patience=12, train_end=2020-12-31 val_end=2021-12-31 (OOS 2022+).
- High patience runs past val min so train↓/val↑ divergence visible in loss_history; best-val ckpt
  used for OOS preds (sound ML). Smoke (2ep local) OK val_pinball 1.651, ~400s/ep CPU single.
- artifacts → gs://gmda-vertex-c779f701-uscentral1/runs/f4_vertex_20260612_230413/vertex_run/
- Polling to terminal state (don't go offline).

### RESULTS — wide-universe OOS 2022-01 → 2026-06 (run f4_vertex_20260612_230413, SUCCEEDED ~2.2hr)
Split train1118 / val245 / OOS1105 batches, 196 names. best_val_pinball 1.604.
**Overfit divergence captured:** train pinball ↓ monotone (seed1338 1.697→1.584), val bottoms
~ep4 (~1.605) then rises → early-stop patience12, best-val ckpt used.

Backtest k_buy=20/k_hold=60 (spec band), net of costs:
- Strategy CAGR **+5.6%** vs SPY **+11.9%**; Sharpe 0.47 vs 0.72; vol 13.9 vs 17.9; maxDD
  -20.1 vs -23.5; alpha_ann **-1.5%**. **Does NOT beat buy-and-hold.**
- By year: strat WINS only 2022 (-7.2 vs -17.5, defensive), LOSES 2023/24/25 mega-cap bull,
  ~tie 2026. Equal-weight top-K can't track cap-weighted index in 2023-25.
- Concentration sweep: k=5 → CAGR -10.7%/DD -42% (naive book fatal over 4.4yr incl 2022);
  k=20 → +5.6%; k=30 → +6.2%. None beat SPY.
- Predictor: IC **+0.0149** (positive, weak), hit 51.8%, cov80 76.0/cov90 87.3, pinball 1.668.
- Gate notes (NOT enforced): G1 IC>0.005 present; G2 turnover 3%/day OK; **G3** DM vs SPY
  p=0.204 (underperf not even significant); **G8 DSR** 0.06-0.27 ≪0.95 FAIL.

**CONCLUSION:** honest negative — small +IC, mildly defensive (wins down-year, lower vol/DD), but
does NOT beat SPY over a longer wider bear-inclusive OOS; gap within noise. Earlier 1yr/83-name
"win" was window+concentration luck. Report results/f4_vertex_196_oos2022/RESULTS_OOS2022.md.

### DIAGNOSIS — "what went wrong" (validation comparison)
1. **Model ~= constant.** q50 per-day cross-sectional spread = 0.018 vs target std 1.10 → ranks
   names almost identically. Val pinball 1.6044 vs BASELINE (train uncond quantiles) 1.6053 =
   **0.06% improvement**; OOS 1.6677 vs 1.6740 = 0.4%. Conditional model barely beats a constant.
2. **Regime shift train/val/OOS:** kurt train 18.2 / val(2021) 5.3 / OOS 10.2; std 1.083/1.037/1.103.
   Val 2021 = calm melt-up. Best-val hit ~epoch4 then val rose → early-stop locked a near-unconditional
   underfit ckpt. 2021 is a poor proxy for 2022 bear + 2023-25 narrow bull.
3. **Target near-efficient:** OOS rank-IC by yr 2022 +.020 / 2023 +.004 / 2024 +.034 / 2025 +.005 /
   2026 +.005; IC_IR all <0.19. Tiny inconsistent signal; pinball dominated by irreducible variance.
4. **Backtest:** small +IC + equal-weight top-K cannot track cap-weighted SPY led by few mega-caps
   2023-25 → net loss. Root cause = (1)+(3): almost no learnable daily cross-sectional signal, and
   val-year regime mismatch made early-stop select an underfit model.

### revision_plan.md created (regime-robust re-validation)
Root causes: (1) val regime=single calm 2021, (2) chronological split wastes scarce regimes,
(3) constant-collapse + top-5 book. Fix: CPCV multi-regime OOS (2022 testable in 7/28 folds),
S1 baseline-beating sanity gate, ranking aux loss, dollar-neutral L/S + equal-weight-univ bench.
Data floor=2016 (Alpaca hard; Stooq quarantined). Full handoff plan at repo root.

---

## 2026-06-13 — RESEARCH: free deep-history data sources (can we go before 2016? should we?)

**Question:** free+public source with deeper history + high-enough time resolution? worth it?

**Probes (QA'd live):**
- **Alpaca SIP** = hard floor 2016-01-04 (re-confirmed). No earlier.
- **Stooq** programmatic CSV = BLOCKED ("page you requested... exceeded" rate-limit/captcha). Repo
  already quarantines Stooq → manual bulk only. Not automatable.
- **yfinance (Yahoo)** = WORKS, free, no key: AAPL/MSFT/GE daily back to **1990-01-02** (9179 rows),
  SPY 1993 (ETF inception), current to 2026-06-12. Daily OHLCV + Adj Close. ~36yr depth.
- **Survivorship test (decisive):** yfinance returns **EMPTY for delisted/dead tickers** —
  LEH, WCOM, ENE, BSC, WAMUQ, CIT, ABK, DYN, EK, TYC, old-DELL all 0 rows. Only survivors kept.

**Resolution verdict:** strategy is DAILY (L=64 daily bars + daily labels). Daily free history is
*sufficient resolution*. Intraday/minute only needed for cost/exec refinement, not the signal.
→ Resolution is NOT the blocker.

**Survivorship verdict (the real issue):** free deep sources (yfinance, per-symbol Stooq/Tiingo)
are **survivorship-biased** — dead tickers purged. A 2000-2015 backtest on them would be upward-
biased (you'd "never have held Lehman into 2008"). Survivorship-safe deep equity data (CRSP,
Shardar SEP/SFP) is **PAID**. No free survivorship-safe deep source exists. Ken French factors are
free to 1926 but are portfolio returns (market-state context), not tradeable single-name picks.

**Would deeper history be GOOD?**
- For TRAINING/CV *regime diversity*: YES in principle — 2016-2026 has only ~2 bears (2020,2022);
  pre-2016 adds dot-com bust + 2008 GFC + 2011 + 2015-16. Big regime gain.
- For the honest OOS/backtest: **NO** — survivorship bias corrupts the score, violating the whole
  anti-overfitting purpose (PLAN). Bias in *evaluation* is fatal; bias in *pretraining* is mild.
- **Conclusion:** free deep history is usable ONLY as clearly-labeled TRAIN-ONLY regime
  augmentation (pretrain on 2000-2015 surviving-name daily bars to expose GFC/dot-com vol), with
  the CPCV OOS kept strictly on SIP-clean, delisting-aware 2016-2026 (Alpaca captures inactive
  assets + corp actions). Modest, optional upside; real risks (adj-quality errors, the splice seam,
  non-stationarity). Repo's existing quarantine invariant already encodes this — honor it.
- **Recommendation:** keep CPCV-on-2016-2026 as PRIMARY. Treat yfinance deep history as an OPTIONAL
  augmented-pretrain experiment, never touching OOS. Net: not worth blocking on; document + offer.

### revision_plan.md → restructured into TWO SEPARATE RUNS (user req)
- **RUN 1 (clean/deployable):** Alpaca SIP 2016-2026, 196 names, CPCV(8,2)=28 folds, bench = SPY +
  equal-weight 196-universe. Panel data/curated/. Output results/run1_clean_<id>/.
- **RUN 2 (deep/exploratory):** yfinance free daily 2000-2026, survivor pool (continuous-data filter),
  CPCV(10,2)=45 folds, **bench = equal-weight buy-and-hold of its OWN pool** (NOT SPY). Separate panel
  data/deep_panel/ (env override TACTIC_DATA_DIR; never touches curated; quarantine test holds).
  Output results/run2_deep_<id>/.
- **Key insight:** Run 2 benchmarked vs its own pool → survivorship bias CANCELS in the relative
  comparison (both sides same survivor pool). Answers "beat owning its own pool across dot-com/GFC/
  2011/2015-16/COVID/2022?" Valid despite absolute bias. Never headlined vs SPY, never deployable.
- Plan: shared steps 0-2 (S1 baseline gate, ranking-aux f4, CPCV code), then Track A (A3-A7) then
  Track B (B0-B6). Added §1B (two-run spec + pool def + EW-buy-hold benchmark math + isolation),
  updated §3/§6/§8/§9, added data/deep_panel/ to .gitignore. yfinance already installed.

---

## 2026-06-15 — run0 vs run2.1 deep-dive: why deep run is coin-flip (all ideas compiled)

### Naming
- **run0** = SPY-beater, `results/f4_vertex_3yr/` (commit 18e1896 / branch run0). f4, 83 Alpaca-SIP
  names, trained 2016-2021 / val 2022 / **OOS 2023-2026** (782 days). Trading = long-only/long-flat
  top-5 buy/hold band, next-open fills.
- **run1** = clean CPCV (Alpaca SIP 196 names, 28 folds). `results/run1_clean_20260613/`.
- **run2** = deep CPCV (yfinance 153 survivors, 45 folds). `results/run2_deep_20260613/` (partial).
- **run2.1** = deep CHRONOLOGICAL walk-forward (153 survivors, trained 2000-15 / val 16-17 /
  **OOS 2018-2026**). `results/run2_1_chrono/`.

### Headline numbers
| | run0 (2023-26) | run2.1 L/S (2018-26) | run2.1 long-flat |
|---|---|---|---|
| CAGR | +23.6% | -16.6% | +8.1% |
| vs bench | SPY +21.0% (beats on CAGR) | EWpool +19.0% (loses) | EWpool +19.0% (loses) |
| Sharpe | 1.10 (< SPY 1.29!) | -1.97 | 0.57 |
| beta | 0.94 | 0.02 | 0.03 |
| alpha/yr | +4.3% | -18.1% | +8.6% (artifact of low beta) |
| maxDD | -26% (> SPY -20%) | -78.8% | -35.6% |
| hit rate | 52.5% | 49.7% | — |

### KEY FACT: run0 "beat SPY" is CAGR-only + modest
- 3-yr CAGR 23.6 vs 21.0 = **+2.6%/yr**, but **Sharpe 1.10 < SPY 1.29** and **maxDD worse** (-26 vs
  -20). It is ~fully-long (beta 0.94) with a small +4.3% alpha. Not a risk-adjusted win.
- The much-quoted 2023-ONLY 41.9% was beta 1.25 × a +24% bull = ~30% beta + ~9% residual. Single
  year, levered into a rally.

### Experiment: old (run0) trader on run2.1's existing deep predictions (no retrain)
- `scripts/rerun_run21_oldtrader.py` → `results/run2_1_chrono/metrics_oldtrader.json`.
- Top-5 long-only band on run2.1 preds, 2018-2026: **CAGR +0.10%**, Sharpe 0.10, maxDD -48%,
  alpha vs SPY -7.4%/yr, vs EWpool -10.2%/yr. ~Dead flat over 8 years.
- **Conclusion: the trading algo was NOT the secret sauce.** A good low-cost trader cannot rescue a
  no-edge predictor (Fundamental Law: IR = IC·√breadth; IC≈0 ⇒ IR≈0).
- It still beats run2.1's brutal dollar-neutral daily L/S (+0.1% vs -16.6%) because long-bias + low
  cost help; but loses to just owning the pool. Even underperforms run2.1's own long-flat (+8.1%)
  because concentrating into 5 names WITHOUT skill = pure idiosyncratic noise (low transfer coeff).

### Prediction accuracy over time (graph: results/run2_1_chrono/accuracy_run0_vs_run21.png)
- Per-year directional hit (run0 | run2.1), identical 2023-2026 window:
  2023: 52.2 | 49.2 · 2024: 53.1 | 49.6 · 2025: 52.4 | 49.1 · 2026: 52.2 | 49.4
- Full-window: run0 52.5% (**day-clustered t=+3.49 → significant >50%**); run2.1 2018-26 49.7%
  (t=-1.56, coin-flip); run2.1 on the 2023-26 OVERLAP 49.4% (t=-2.03, slightly BELOW coin-flip).
- **Same window, same stocks ⇒ the entire gap is the PREDICTOR, not the regime.** run0 has genuine
  (if small) cross-sectional skill every year; run2.1 has none.
- NOTE/correction: 2023-ONLY run0 hit was borderline (t=+1.63, not sig); the FULL 2023-2026 run0 IS
  significant once 782 days accumulate.

### WHY run2.1 ≈ coin-flip, run0 has skill (ranked hypotheses + how to test)
1. **Stale training window / concept drift (HIGHEST).** run0 trained 2016-21 → tested adjacent
   2023-26. run2.1 trained 2000-15 → tested 2018-26 (3-11yr forward). Feature→winner map drifts
   across market-structure eras. TEST: retrain deep model rolling-recent (2014-17 → test 2018).
2. **Degraded free-data features (HIGH).** yfinance has no trade-count ⇒ ch05 forced = ch04 (dead
   duplicate). run0 on Alpaca SIP had real trade intensity. TEST: ablate ch05 on run0 SIP data.
3. **Universe quality not size (MEDIUM).** 153 yfinance survivors vs 83 curated SIP. Breadth helps
   only if IC>0. TEST: run deep model on the same 83 SIP names.
4. **Survivorship pool ⇒ degenerate cross-section (MEDIUM).** The 153 all survived to 2026; their
   2000-15 cross-section is dominated by eventual mega-winners ⇒ model may learn "everything rises"
   (no relative signal). TEST: check q50 cross-sectional dispersion / ranking degeneracy.
5. **Target/label mismatch (LOW-MED).** run2.1 demeans cross-sectionally; if run0 didn't identically,
   sign-hit means subtly different things. TEST: recompute on identical target def.

### Supporting literature
- Grinold & Kahn — Fundamental Law of Active Management: IR = IC·√breadth.
- Clarke, de Silva & Thorley — "FLoAM: Redux": realized IR = TC·IC·√breadth (transfer coefficient;
  a concentrated long-only top-5 band has LOW TC ⇒ can't express a cross-sectional signal).
- Bailey & López de Prado — Deflated Sharpe Ratio (SSRN 2460551): best-of-N selection inflates
  Sharpe even on noise; overfit strategies underperform OOS. run0 widened 15→83 names "until it
  worked" = best-of-N; DSR never computed on run0. The CPCV re-validations are the deflated test.
- Bailey, Borwein, López de Prado, Zhu — Probability of Backtest Overfitting (SSRN 2326253).

### Verdict
run0 had genuine, statistically-significant (t=3.49) cross-sectional skill — but it only converted to
a MODEST CAGR beat (and a worse Sharpe), mostly via near-full-long market exposure. run2.1's deep
predictor is honest coin-flip (slightly below on the overlap). Trading algo is a second-order lever;
the first-order driver is predictor skill, which is killed by stale-window + degraded-free-data.

---

## 2026-06-17 — run2 (deep CPCV 45-fold) COMPLETE + train/val/test accuracy cross-cut

### run2 finished
- All 45 CPCV folds (fold_00..44) SUCCEEDED on Vertex. Downloaded + finished locally.
- Combined OOS: 995,418 obs, 6,506 sessions, 2000-07-25 → 2026-06-08.
- L/S Sharpe **−1.83**, CAGR −15.6% vs EWpool +14.9% (alpha −16.7%), maxDD −98.8%.
- long_flat Sharpe 0.71, CAGR +9.8% (still loses to pool +14.9%; β≈0 "alpha" is artifact).
- **45 path Sharpes: mean −2.66, frac>0 = 0.00** (every regime-fair path negative).
- Predictor: hit **0.5054 (day-clust t=+5.91, statistically >50%)**, IC +0.0214, IR 0.109, pinball 1.674.
- S1 val gate pass **32/45** folds (13 fail). DSR 0.000 FAIL, PBO 0.0, DM vs pool p≈0 worse.
- Result: NEGATIVE, same as run1 + run2.1. Report results/run2_deep_20260613/RESULTS_RUN2.md.

### KEY NUANCE: run2 CPCV predictor is statistically POSITIVE but run2.1 forward is coin-flip
- run2 (CPCV) hit 50.5% t=+5.91 POSITIVE; run2.1 (chrono forward) hit 49.7% t=−1.56 coin-flip.
- Same data, same model. Difference = **CPCV adjacency**: each test fold predicted by a model trained
  on purged-adjacent date groups bracketing it in time → near-in-time training advantage. run2.1's
  strict 2000-15→2018-26 forward split removes that → edge gone.
- **Lesson: CPCV measures conditional-on-recent skill, NOT deployable forward skill.** This is the
  single biggest methodological takeaway. CPCV is great for "is there ANY signal" but its positive
  result does not imply a forward-deployable edge if the world is non-stationary.
- Even run2's "positive" is economically dead: IC 0.0214 → Fundamental Law IR≈0; t huge only b/c
  n=6506 days. Statistical ≠ economic significance.

### Train / Val / Test accuracy comparison (user asked; honest scope)
**What's persisted:** test directional hit (all 4 runs); val pinball + S1 baseline (all 4);
train pinball (run0 + run2.1 only — walk/single runs logged loss_history; CPCV folds logged only
val_pinball in meta). **Train/val DIRECTIONAL HIT was never saved, and NO model checkpoints exist**
→ can't recompute train/val hit without retraining. Flagged this hard; did not fabricate.

**Test hit (the one valid cross-run accuracy metric):**
| run | split | train→test gap | test hit | day-clust t | IC |
|---|---|---|---|---|---|
| run0 | chrono adjacent | 2016-21→2023-26 ~1yr | 52.5% | +3.49 | 0.0143 |
| run1 | CPCV interleaved | purged ±2-5d | 51.1% | +5.36 | 0.0200 |
| run2 | CPCV interleaved | purged ±2-5d | 50.5% | +5.91 | 0.0214 |
| run2.1 | chrono distant | 2000-15→2018-26 3-11yr | 49.7% | −1.56 | 0.0106 |
→ **Test accuracy falls monotonically with train→test temporal distance.** Concept-drift signature.

**Val S1 gate (model val pinball vs train-quantile baseline, within-split = VALID):**
run1 +1.13% margin pass 22/28; run2 +1.11% margin pass 32/45. Model barely beats a constant guess.

**Train vs val pinball (run0, run2.1):** run0 train 1.601 / val 1.669 / test 1.686 (~5% optimism gap,
mild benign overfit). run2.1 train 1.672 / val 1.427 / test 1.708.
⚠️ **CAVEAT: pinball NOT comparable across splits** — it scales with each period's return volatility,
and splits are different calendar eras (run2.1 val 2016-17 calm vs train 2000-15 incl dot-com+GFC).
run2.1's val<train is a period-scale artifact, NOT generalization. Only within-split S1 is valid.

### Where I'm UNSURE
1. Train/val directional hit — UNKNOWN (not saved, no ckpt). All train/val "accuracy" = pinball proxy.
2. How much of run2's +5.91 t is genuine vs CPCV adjacency — lean mostly adjacency, can't quantify
   without a rolling-recent retrain (= hypothesis #1's direct test).
3. IC higher run1/run2 than run0 but hit lower — rank-IC vs sign-hit weight different cross-section
   parts; not fully pinned.
4. Pinball scale confound — confident it breaks cross-split comparison.

### Literature added for this analysis
- Gu, Kelly, Xiu (2020, RFS) "Empirical Asset Pricing via ML" — ML edges decay OOS, recent data
  dominates → supports concept-drift read of run2.1.
- Quiñonero-Candela et al. (2009) "Dataset Shift in ML" + Hastie-Tibshirani-Friedman ESL —
  train-error optimism + covariate shift = temporal-distance degradation.
- Harvey, Liu, Zhu (2016) — t>3 multiple-testing bar; run1/run2 clear it on directional t but fail
  economically. + de Prado AFML ch7/12 (CPCV adjacency), Bailey-LdP DSR/PBO (deflation).

### Infra note (gcloud)
- Local `gcloud` crashes: "untrusted mount point ... OpenAI\Codex\bin". Root cause = Codex bin on
  PATH; gcloud refuses to traverse it during command-load (affects ALL gcloud cmds incl storage cp +
  custom-jobs create). Workaround: strip Codex from PATH for gcloud, OR use `gsutil` (separate binary,
  unaffected). finish_run.py's `gcloud storage cp` hit this → finished run2 via gsutil download +
  inline aggregate_combined + build_report instead. No code changed.
- `gh` auth token is INVALID (expired). Blocks `gh repo rename` + git push until re-auth
  (`gh auth login -h github.com`). Local commits unaffected.

### Repo renamed → QTA0
- This project renamed to **QTA0** (Quant Trading Algorithm 0). REPORT.md rewritten as the canonical
  learn-from-everything doc (data, algos, all 4 runs, train/val/test, theory, unsure-list, reproduce).
  README.md = repo index. Scratch logs removed. GitHub repo + local folder → QTA0.
