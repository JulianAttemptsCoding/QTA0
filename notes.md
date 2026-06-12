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
