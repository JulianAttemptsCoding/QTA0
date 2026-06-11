# PLAN.md — TACTIC-MoB v4

**Transaction-cost-Aware, Component-decomposed, Throttled Implementation of Calibrated Mixture-of-Betting experts.**

> This document is a complete, self-contained build specification: from an empty repository to a finished, validated backtest with all anti-overfitting gates evaluated. It is written to be executed by an autonomous coding agent (Claude Code) in one pass, halting only at pre-registered kill gates or missing credentials. Live/paper trading is OUT of scope for this build (backtest-complete is the deliverable); the live-loop spec is included as Appendix F for the next milestone.

---

## 0. MISSION, DEFINITIONS, NON-NEGOTIABLES

### 0.1 Mission
Build and rigorously validate a daily-frequency, long-only/long-flat, cross-sectional US equity strategy whose forecasts are produced by a small pooled-panel mixture of experts, whose probabilities are conformally calibrated and monitored by e-value test-martingales, whose trading is governed by alpha-decay-matched partial adjustment with a buy/hold band under a hard turnover budget, and whose costs are estimated per-asset per-day from OHLC data. The null hypothesis (no deployable edge) is presumed true until every gate in §17 passes.

### 0.2 Definitions
- **Edge**: positive expected net-of-cost return relative to the net-of-cost equal-weight investable universe, surviving multiple-testing-corrected significance on a locked, never-touched holdout.
- **Trial**: any design decision, hyperparameter setting, threshold choice, or model variant that was evaluated against data. Every trial increments the integer `N` in the trial ledger (Appendix D). `N` feeds the Deflated Sharpe Ratio.
- **Component returns**: overnight `r_on` and intraday `r_in` legs of the daily return (formulas in Appendix A.1).
- **PIT**: point-in-time. No table may answer a historical query using information not available on that date.

### 0.3 Non-negotiables (the agent must enforce these in code, not comments)
1. **Temporal contracts**: `feature_available_ts <= decision_ts(close t) < entry_ts(open t+1) < exit_ts`. Runtime asserts; violation fails the build.
2. **Fit/transform discipline**: every scaler, HMM/BOCPD prior, covariance estimator, calibrator, and model is `fit()` on training folds only. A single `fit` call touching validation/test/holdout data fails CI.
3. **SIP-only features**: all research features/labels from `feed=sip`. Any IEX-fed bar entering `curated/` fails the feed contract test.
4. **No ticker embeddings / no per-ticker models**: identity information may never enter a model. One pooled global model; the full cross-section is processed per date (panel batching, §10.4).
5. **One training objective**: summed pinball loss on a non-crossing quantile grid. No auxiliary loss terms, no loss-weight hyperparameters.
6. **Calibration is post-hoc only** (conformal PID); never a training-loss term.
7. **Holdout lock**: `data/holdout/` (final 18 months) is written once during ingestion and is read-protected (chmod 000 + a `.LOCKED` sentinel checked by all loaders) until Phase 13 explicitly unlocks it after the pre-registration hash is committed.
8. **Honest N**: every hyperparameter search trial, expert addition, threshold, and structure choice is auto-appended to `registry/trial_ledger.parquet`. DSR is computed from this ledger, never from a hand-entered number.
9. **Delisting/terminal-return contract**: every asset that exits the universe must have a populated `terminal_return` and `terminal_type`, or the build fails. If unobtainable from free data, the asset-day is excluded AND counted in the survivorship-bias bound report (§5.6).
10. **Kill gates halt the pipeline**: `make all` stops at the first failed gate and writes `reports/KILL_REPORT_<gate>.md`. A halted build is a *successful* run of this plan.

### 0.4 Final architecture decision (input format)
**One pooled model, all assets at once per date.** Temporal encoding is per-asset with **shared weights** (channel-independent for noise robustness); cross-sectional interaction happens in **exactly one** thin attention layer over the date's asset tokens (variable token count handles daily-changing universe membership). Per-ticker local models are banned: they multiply parameters by N, memorize identities, and cannot produce the cross-sectional rank that is the prediction target. Theoretical basis: global/pooled models match or beat collections of local models even on heterogeneous series (Montero-Manso & Hyndman 2021); channel-independent temporal encoders are more robust than fully channel-dependent stacks (PatchTST vs. Crossformer-family evidence); thin variate-attention on top recovers cross-sectional structure (iTransformer-style), and its token count is flexible at inference.

---

## 1. DATA SOURCES (HUMAN TASKS BEFORE RUNNING — the only manual steps)

| # | Source | What it provides | How to get it | Cost |
|---|--------|------------------|---------------|------|
| 1 | **Alpaca Market Data API** (required) | Daily + minute OHLCV bars (SIP feed, ~2016→present, incl. `vwap`, `trade_count`), **historical opening/closing auction prices**, corporate actions (≥ ~2020), assets master (incl. `status=inactive` → delisted symbol list), calendar | Create a free account at https://alpaca.markets → dashboard → generate API key/secret (paper account is fine). Put in `.env`: `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`. Historical SIP data older than 15 minutes is available on the free plan; the optional Algo Trader Plus subscription is only needed for live real-time SIP later. | Free (research); ~$99/mo later for live SIP |
| 2 | **Ken French Data Library** (required) | Daily factor returns: Mkt-RF, SMB, HML, RMW, CMA, RF; Momentum (UMD); Short-Term Reversal (STR) — used by the attribution gate | Auto-downloaded by `make data-factors` from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (CSV zips; no key needed) | Free |
| 3 | **Wikipedia S&P 500 constituents — revision history** (required) | Point-in-time large-cap universe reconstruction (membership by date) | Auto-downloaded by `make data-universe` via the MediaWiki revisions API for page "List of S&P 500 companies" (set a descriptive `User-Agent`) | Free |
| 4 | **SEC EDGAR** (required) | `company_tickers.json` + submissions index for ticker/CIK cross-checks and symbol-change reconciliation | Auto-downloaded with a compliant `User-Agent: name email` header from https://www.sec.gov | Free |
| 5 | **Stooq bulk US daily data** (optional) | Pre-2016 daily bars for *diagnostics only* (quarantined table; never joined into curated features) | https://stooq.com/db/h/ bulk download | Free |
| 6 | **Norgate Data or Sharadar/Nasdaq Data Link (SEP)** (optional upgrade, recommended for the paper) | True survivorship-bias-free prices incl. delisting returns | norgatedata.com / data.nasdaq.com (Sharadar Equity Prices) | ~$30–60/mo |

`.env` template (created by Phase 0; never committed):
```
ALPACA_API_KEY_ID=
ALPACA_API_SECRET_KEY=
ALPACA_DATA_URL=https://data.alpaca.markets
EDGAR_USER_AGENT="YourName your@email"
```

---

## 2. PHASE 0 — REPO SCAFFOLD, ENVIRONMENT, REGISTRY, CI

### 2.1 Objective
Deterministic, testable skeleton with the trial ledger, config registry, gates engine, and negative-control CI in place **before any data exists**.

### 2.2 Repo layout (create exactly)
```
tactic_mob/
  PLAN.md  SOURCES.md  README.md  Makefile  pyproject.toml  .env.example  .gitignore
  configs/            v1.yaml  (Appendix B verbatim)
  src/tactic/
    common/           contracts.py  hashing.py  registry.py  gates.py  io.py  calendar.py
    ingest/           alpaca_bars.py  alpaca_auctions.py  alpaca_corp_actions.py
                      alpaca_assets.py  factors_french.py  wiki_constituents.py
                      edgar_xcheck.py  stooq_diag.py  day1_audits.py
    universe/         build_universe.py  delistings.py  survivorship_bound.py
    costs/            spreads.py  cost_model.py
    labels/           build_labels.py
    features/         build_features.py  wavelets.py
    diagnostics/      variance_ratio.py  ic_decay.py  auction_gap.py  tugofwar_table.py
    regime/           bocpd_t.py
    risk/             nls.py  dcc_nl.py  hrp.py
    models/           panel_dataset.py  experts/{enet.py,gbm.py,tcn_ci.py,hybrid_xs.py,fm_zero.py}
                      quantile_heads.py  train.py  hpo_local.py  hpo_vertex.py
    uq/               conformal_pid.py  evalue_monitor.py
    agg/              aa.py  gate_prior.py  universal_portfolio.py
    portfolio/        gp_aim.py  band.py  rc_kelly.py  qp_challenger.py  throttle.py
    backtest/         engine.py  fills.py  accounting.py  baselines.py
    validation/       walk_forward.py  cpcv.py  pbo.py  dsr.py  spa.py  dm_test.py
                      romano_wolf.py  attribution.py
    stress/           perturb.py  cost_shock.py  timing_shock.py  feed_contract.py
                      placebos.py  leak_canary.py
    reports/          render.py  dashboards.py  kill_report.py
  tests/              (unit + contract tests, Appendix C)
  data/               raw/  curated/  features/  labels/  holdout/  diagnostic_quarantine/
  registry/           trial_ledger.parquet  config_hashes/  prereg/
  reports/
  vertex/             (optional submission stubs: Dockerfile, pipeline.py, vizier_spec.yaml)
```

### 2.3 Environment spec
- Python 3.11; `uv` or `pip`. Pinned deps: `numpy pandas pyarrow polars scipy scikit-learn lightgbm torch cvxpy statsmodels PyWavelets alpaca-py requests bidask matplotlib pyyaml pytest hypothesis tqdm`.
- Global determinism: `PYTHONHASHSEED=0`, torch/np seeds from config, `torch.use_deterministic_algorithms(True)` where supported. Every artifact written with a content hash sidecar (`<file>.sha256`).
- `Makefile` targets (execution order in §18): `setup, test, data-alpaca, data-factors, data-universe, day1-audits, universe, costs, labels, features, diagnostics, baselines, infra, train, aggregate, decide, backtest, stress, firewall, reports, all`.

### 2.4 Registry & gates engine (implement first)
- `registry.py`: `log_trial(category, description, config_hash, study_id=None) -> trial_id` appends a row to `trial_ledger.parquet`. Categories: `design`, `hpo`, `threshold`, `expert`, `structure`. `current_N()` returns the row count of rows with `charged_to_N=True`.
- `gates.py`: declarative gate table loaded from config (§17); `evaluate(gate_id, value) -> PASS/FAIL`; on FAIL writes `reports/KILL_REPORT_<gate>.md` (inputs, value, threshold, interpretation, recommended next step) and raises `KillGate`.
- `contracts.py`: `assert_pit(df)`, `assert_feed_sip(df)`, `assert_fit_scope(obj, fold)`, `assert_terminal_returns(universe_df)`. All loaders call these.

### 2.5 QA checklist (Phase 0 done when)
- [ ] `make setup && make test` green on the empty skeleton (contract tests use synthetic fixtures).
- [ ] `holdout/.LOCKED` sentinel exists; `io.load_curated()` raises if any requested date falls inside the holdout window while locked.
- [ ] `log_trial` round-trips; `current_N()==0` at start; ledger file is append-only (writer refuses overwrite).

---

## 3. PHASE 1 — INGESTION (ALPACA + SUPPLEMENTS)

### 3.1 Objective
Raw, immutable, hash-manifested copies of all source data in `data/raw/`, then typed curated tables in `data/curated/` (Parquet, partitioned by date).

### 3.2 Alpaca ingestion spec
- **Bars (daily)**: `GET /v2/stocks/bars`, `feed=sip`, `timeframe=1Day`, paginated, for the candidate symbol list (=. all assets from §3.3, active + inactive), **twice**: `adjustment=raw` and `adjustment=all`. Start `2016-01-01`, end = today − 1. Store both. Fields kept: `t,o,h,l,c,v,n,vw`. Chunk requests (≤ 1000 symbols × paged), exponential backoff on 429, resume from manifest.
- **Bars (minute)**: same endpoint, `timeframe=1Min`, **only** for the dates/symbols required by the overnight-leg 15:30 snapshot (§7.4) — fetched lazily in Phase 4 if and only if diagnostics keep the ON leg alive (saves ~50× volume otherwise).
- **Auctions**: historical auctions endpoint (alpaca-py `StockHistoricalDataClient` auctions request, or REST `/v2/stocks/auctions`) → `auctions_daily(asset_id, date, open_auction_px, open_auction_vol, open_auction_ts, close_auction_px, close_auction_vol, close_auction_ts)`. Where multiple auction prints exist, take the official auction print (condition-coded), else last auction print of the window; record `source_flag`.
- **Assets master**: `GET /v2/assets?status=active` and `status=inactive` (the inactive list is the delisted-symbol seed). Snapshot monthly into `assets_snapshots`.
- **Corporate actions**: corporate-actions endpoint for all types (splits forward/reverse, cash/stock dividends, mergers, spin-offs, symbol changes, worthless removals). Store raw JSON; normalize into `corporate_actions_unified`. KNOWN LIMIT (handle, don't hide): coverage starts ~April 2020 and older delistings/symbol-changes are absent → reconciliation in Phase 2 fills the gap.
- **Calendar**: `GET /v2/calendar` 2016→present → trading-day index used everywhere.

### 3.3 Candidate symbol list
Union of: (a) all PIT S&P 500 members 2016→present from `wiki_constituents.py` (MediaWiki revisions API: fetch every revision of "List of S&P 500 companies", parse the constituents table per revision date, forward-fill membership between revisions); (b) a fixed liquid-ETF list (config: SPY,QQQ,IWM,DIA + the 11 SPDR sectors; **leveraged/inverse ETFs excluded**); (c) all Alpaca `inactive` assets whose last trade date ≥ 2016 and that ever matched (a) — this is the delisted-recovery set.

### 3.4 Supplements
- `factors_french.py`: download + parse FF5 daily, Momentum daily, ST Reversal daily → `factors_daily(date, mkt_rf, smb, hml, rmw, cma, rf, umd, str)`.
- `edgar_xcheck.py`: tickers/CIK json; build symbol-change map cross-check.
- `stooq_diag.py`: optional; writes ONLY to `data/diagnostic_quarantine/` (loader physically separate; no join path to curated — enforced by a test).

### 3.5 Day-1 audits (`day1_audits.py`, run by `make day1-audits`, output `reports/day1_audit.md`)
1. **Delisted-bar coverage**: for 50 known 2017–2024 delistings (hardcoded list incl. bankruptcies and cash mergers), query daily bars; report which have bars through their last trading day and whether a final-day price exists. → defines `delisting_coverage_ratio`.
2. **Auctions coverage**: % of (asset,day) in the candidate universe with both auction prints, by year and liquidity decile.
3. **History start**: earliest SIP bar per symbol; effective panel start date.
4. **Feed sanity**: pull one date both `feed=iex` and `feed=sip`; assert volumes differ by >10× on liquid names (verifies we are truly getting SIP).
5. **`opg`/`cls` order support**: documented check only in this milestone (no orders are sent).

### 3.6 QA checklist
- [ ] Raw manifests complete (every requested symbol either has data or an explicit `no_data` record).
- [ ] `curated/prices_daily` row counts per year monotone-plausible; zero duplicate (asset,date) keys; OHLC sanity (l<=o,c<=h; v>=0).
- [ ] Auctions reconciliation table built: distribution of `ln(open_auction_px) − ln(first_print_open)` exists (consumed in Phase 5).
- [ ] Holdout window rows physically copied to `data/holdout/` and removed from the working curated partitions; `.LOCKED` intact.

---

## 4. PHASE 2 — UNIVERSE, CORPORATE ACTIONS, DELISTINGS

### 4.1 Objective
A PIT tradable universe with terminal returns, and a quantified survivorship-bias bound.

### 4.2 Spec
- **Adjustment policy**: adjusted (`adjustment=all`) prices define returns; raw prices feed spread estimators and auction reconciliation. Cross-check: rebuild adjusted closes from raw + Alpaca split/dividend records where available; mismatches > 25 bps flagged to `reports/adjustment_anomalies.csv`.
- **Symbol continuity**: build `entity_map(entity_id, symbol, start_date, end_date)` from Alpaca symbol-change actions + EDGAR cross-check + Wikipedia table footnotes. All curated tables key on `entity_id`.
- **Terminal returns** (`delistings.py`): for each exiting entity — cash merger: terminal price = deal cash (from CA record) else last trade; bankruptcy/worthless: terminal_return = −100%; exchange delist to OTC with no data: mark `terminal_type=unknown`, exclude from tradables on/after exit, count in bias bound. Contract `assert_terminal_returns` enforces completeness of the *typed* field (unknown is a valid type but is reported).
- **PIT universe** `universe_membership(entity_id, date, tradable, reason_codes)`: tradable iff — PIT S&P-500 member OR config ETF; price > $5; ADV21 > $5M; blended spread (Phase 3) < 40 bps; listed ≥ 126 days; asset status active on that date. Target panel width 400–700.
- **Survivorship bound** (`survivorship_bound.py`): run every baseline (Phase 6) on (i) survivor-only universe (names alive today) and (ii) the reconstructed universe; report the return gap per year as the bias bound. This table ships in the paper.

### 4.3 QA checklist
- [ ] No entity appears in two symbols on the same date; no membership without a price row.
- [ ] Universe width time series plotted; no cliff artifacts at Wikipedia revision boundaries (interpolation correct).
- [ ] Bias-bound table renders; `unknown` terminal types < 5% of exits or a warning gate (G0b, soft) fires.

---

## 5. PHASE 3 — SPREAD SURFACE & COST MODEL

### 5.1 Spec
`spreads.py` computes, per (entity, day), on RAW prices, rolling 21-day estimates:
- **Corwin–Schultz** (Appendix A.4): negative estimates set to 0 (log the rate).
- **Abdi–Ranaldo**: `s_AR = 2*sqrt(max(0, E[(c_t − m_t)(c_t − m_{t+1})]))` with `c=ln close`, `m=(ln high + ln low)/2` (use the 21-day mean version).
- **EDGE** (Ardia–Guidotti–Kroencke): call the `bidask` PyPI package (`edge(open, high, low, close)`) on rolling windows.
- **Blend**: `s_blend = median(s_cs, s_ar, s_edge)`, floored at `2*fee_floor` (config: fee_floor = 0.5 bps) and capped at 500 bps; winsorize at the 99.5th pct per day.
`cost_model.py`: cost per auction crossing
```
c_hat(i,t) = 0.5 * s_blend(i,t) + c_impact * sqrt(trade_dollars / ADV21_dollars(i,t))
c_impact = 0.10 (config; in sqrt-units of participation, i.e. 10 bps at 1% of ADV)
```
Holding structures pay: OO (open→open): 2 crossings per round trip amortized by holding period; ON (close→open): 2 crossings/day; ID (open→close): 2 crossings/day. The engine charges per executed trade, not per formula shortcut.

### 5.2 QA
- [ ] Spread surface sanity: cross-sectional median by liquidity decile is monotone; large-cap median ∈ [0.5, 5] bps; small-cap tail visible.
- [ ] Estimator agreement: pairwise rank-corr of the three estimators > 0.5 on the pooled panel (else investigate before proceeding).

---

## 6. PHASE 4 — LABELS & FEATURES

### 6.1 Labels (`build_labels.py`; formulas Appendix A.1)
Per (entity, day): `r_on, r_in, r_cc, r_oo` from **auction prices** where available (`label_source='auction'`), else first/last prints (`'print'`). Top-bucket indicators `z_top_on, z_top_in` at p=0.20 within PIT universe. EWMA sigma (λ=0.94, on r_cc, floor 0.2%/day, fit-scope = expanding within train folds, frozen for val/test). Vol-standardized targets `y_* = r_*/sigma`.

### 6.2 Features (`build_features.py`) — FROZEN v1, exactly these
**Sequence tensor** `x_seq ∈ R[L=64 × C=18]` per (entity, decision-day t), all computed through close t on adjusted SIP bars:
1 log close return; 2 intraday log range `ln H−ln L`; 3 overnight gap `ln O_t − ln C_{t−1}`; 4 volume z (63d); 5 trade-count z (63d); 6 VWAP deviation `ln C − ln vw`; 7–9 realized vol {5,21,63}; 10–12 momentum (cum log ret) {5,21,63}; 13–15 drawdown from rolling max {21,63,126}; 16–18 MODWT (db4, PyWavelets, reflect padding **left-only**, computed on the trailing window only — no future contamination) energy at levels 2,3,4 of the return series.
**Static vector** `u ∈ R[12]` per (entity, t): cross-sectional pct-ranks of {mom21, mom63, mom252−21, vol21, dd63, ADV21}; `s_blend`; ln(ADV21); and 4 interactions `rank_mom21*M_t` components.
**Market state** `M_t ∈ R[5]`: 21d market (universe EW) realized vol; cross-sectional dispersion (std of daily returns); breadth (% above 63d MA); EW-index drawdown from 252d max; top-eigenvalue share `λ1/tr(Σ̂_63d)`.
All standardizations use train-fold statistics only. `feature_available_ts = close(t) + 30min` recorded per row.

### 6.3 ON-leg snapshot (conditional)
If diagnostics (Phase 5) keep the overnight leg alive, build the 15:30 ET feature snapshot: aggregate minute bars 09:30–15:30 into pseudo-daily fields for day t, recompute channels 1–6 with the 15:30 partial bar as "today", and set `decision_ts=15:30`, `entry = cls auction(t)`. This is a *separate* feature version (`feature_set_version='v1_on1530'`) with its own contracts.

### 6.4 QA
- [ ] Leakage spot check: for 100 random (entity,t), recompute every feature with data truncated at t and byte-compare.
- [ ] Wavelet test: shifting the input series by 1 day shifts features by 1 day (no edge effects from padding).
- [ ] Label identity: `r_cc == r_on + r_in` to 1e-12; `r_oo == r_in(t+1)+r_on(t+2)` indexing test.

---

## 7. PHASE 5 — DIAGNOSTICS (GATE G1; run before any model training)

All pre-registered, non-selective. Output `reports/diagnostics.md` + parquet tables.

1. **Variance ratios** (`variance_ratio.py`): Lo–MacKinlay heteroskedasticity-robust VR(q), q∈{2,5,10,20}, per component {on,in,cc}, pooled and per BOCPD-regime bucket (regimes from Phase 7 run in diagnostic mode on train years only).
2. **IC + IC decay** (`ic_decay.py`): daily cross-sectional Spearman IC of every feature vs every component at horizons h=1..20; fit `rho(h)=rho1*exp(−phi*h)`; output table (feature, component, rho1, phi, t-stat of mean IC with Newey–West).
3. **Auction gap** (`auction_gap.py`): distribution of `ln A_open − ln first_print` by liquidity decile → execution-noise floor `epsilon_exec` (90th pct abs gap, large-cap decile) used by gate G11 later and by the fills model noise term.
4. **Tug-of-war table** (`tugofwar_table.py`): decile sorts on past 1-month overnight return and past 1-month intraday return; report subsequent overnight and intraday mean returns (replicating the Lou–Polk–Skouras pattern on 2016+ data). This is the direct existence test for the Pass-1 thesis on our sample.

**GATE G1 (kill):** proceed iff [max over (feature, component) of |mean IC| with NW t>2] ≥ 0.005 **or** any VR rejects random walk at 1% in any regime bucket **or** the tug-of-war table reproduces opposite-signed component continuation at t>2. Else HALT → negative-result report (that report is a deliverable, not a failure of the plan).

**GATE G1b (structure choice):** among holding structures {OO, ON, ID}, keep those whose alive component(s) clear G1 AND whose break-even arithmetic (Appendix A.10: required gross alpha = crossings/day × mean c_hat of the would-be top-K) is < 50% of the component's gross decile spread measured in the tug-of-war table. Log structure choices as trials.

---

## 8. PHASE 6 — BASELINE LADDER & FORECAST TESTS

`baselines.py` implements, per surviving component/structure: (1) net-of-cost equal-weight universe at monthly rebalance (THE benchmark); (2) SPY buy-and-hold; (3) 12-1 momentum decile long-only, monthly, with buy/hold band; (4) 1–5d reversal decile; (5) elastic-net panel on `u`+summary stats; (6) LightGBM on same. All run through the same backtest engine (Phase 11) and cost model — no shortcut analytics.
`dm_test.py`: Diebold–Mariano on daily loss differentials (pinball for forecasts; net P&L for strategies), Newey–West lag = 10. `romano_wolf.py`: stepdown FWER across the full ladder × component grid, stationary bootstrap B=2000, mean block 10.
**GATE G3 definition recorded here** (evaluated in Phase 13): TACTIC-MoB must beat rung 6 net OOS, DM p<0.05 after Romano–Wolf.

QA: every baseline's turnover, cost drag, and survivor-vs-reconstructed gap tabulated; EW benchmark is net-of-cost at its own rebalance frequency (fairness check in test suite).

---

## 9. PHASE 7 — INFRASTRUCTURE MODELS (REGIME, RISK, CALIBRATION, MONITORS)

### 9.1 BOCPD (`bocpd_t.py`)
Inputs: z-scored `M_t` (train stats). Independent Normal-Inverse-Gamma per dim (μ0=0, κ0=1, α0=2, β0=1) ⇒ Student-t predictive; constant hazard H ∈ {1/63, 1/126, 1/252} selected on **train predictive log-likelihood only** (3 logged trials). Standard Adams–MacKay recursion with run-length pruning at P<1e-6. Outputs per t: `p_fresh = P(run_length < 10)`; run-length-weighted posterior mean of M_t. Consumers: gate prior (9.4-adjacent), exposure multiplier `g_t = 1 − beta_g * p_fresh`, beta_g=0.5 (config constant).

### 9.2 Covariance (`nls.py`, `dcc_nl.py`)
Ladder: (a) Ledoit–Wolf nonlinear shrinkage (analytic/QIS variant) on 252d window; (b) DCC-NL: univariate GARCH(1,1) per asset (composite/pairwise likelihood for the DCC params), correlation targeting matrix = NLS of devolatilized residuals; (c) DCC-NL with OHLC regularized returns (range-based vol proxy × smoothed return sign). Selection per refit on realized OOS GMV portfolio variance (never Frobenius). Each rung = 1 trial.

### 9.3 Conformal PID (`conformal_pid.py`)
Per component, per interval level α∈{0.2, 0.1} (i.e., 80%/90% intervals): score = pinball nonconformity of the model's α/2 and 1−α/2 quantiles; online update with **P+I terms only**:
```
q_{t+1} = q_t + eta_P * (err_t − alpha) + eta_I * sum_{s<=t} (err_s − alpha)
eta_P = 0.05 (on standardized score scale), eta_I = 0.005
```
(no scorecaster/D-term — it voids the coverage guarantee). Initialized on the validation year. Also produces the calibrated predictive CDF used by sizing (quantile-interpolated empirical distribution).

### 9.4 e-value monitors (`evalue_monitor.py`)
For binary `z_top` forecasts q (per component): online-isotonic recalibrated competitor q̄ (fit prequentially); per-step e-factor
```
E_t = [z_t*q̄_t + (1−z_t)*(1−q̄_t)] / [z_t*q_t + (1−z_t)*(1−q_t)],  e_t = Π_{s<=t} E_s
```
Merged across monitors by averaging. Throttle `scale_t = clip((20/e_t)^0.5, 0.25, 1.0)`; ALARM (report-only in backtest; human decision live) at e_t ≥ 20 (α=5%, anytime-valid by Ville).

QA: synthetic tests — feed perfectly calibrated simulated forecasts: e-process must stay O(1) (median < 3 over 2,500 steps across 200 sims); feed 10%-biased forecasts: e-process must cross 20 within 500 steps in >80% of sims. PID: empirical coverage within ±2% of target on simulated shifting AR data.

---

## 10. PHASE 8 — EXPERTS & TRAINING

### 10.1 Expert set (K=4 + 1 control; each addition beyond this list = pre-registered trial)
- `f1` ElasticNet (multi-output: 7 quantiles × live components) on `u` + 12 summary stats of `x_seq` (means/last/std per channel group). Quantile version: per-quantile linear pinball regression (sklearn QuantileRegressor or LightGBM linear).
- `f2` LightGBM, `objective='quantile'`, one booster per quantile per component (constrained to monotone non-crossing by post-sort + isotonic across the grid), inputs = `u` + the same summaries + 8 raw recent returns.
- `f3` CI-TCN ("Stage A only"): per-asset TCN encoder, shared weights — 6 residual blocks, kernel 5, dilations [1,2,4,8,16,32], 64 channels, GELU, dropout 0.1 → mean-pool → MLP(64) → quantile heads. **No cross-sectional layer.** This is the ablation twin.
- `f4` Hybrid: `f3` encoder → token `h_i=[enc_i, u_i] ∈ R^76` → **one** MHSA layer over the date's N_t tokens (4 heads, d_model=64, pre-LN) + residual → quantile heads + gate logits. Param count target 0.15–0.25M (assert in test).
- `f0` (control, no training): frozen zero-shot TS foundation model (Chronos-Bolt-small or TimesFM) producing per-asset quantiles from the raw return sequence; pre-registered expectation: loses; informative either way. 1 trial.

### 10.2 Heads & loss (all experts)
Quantile grid τ∈{0.05,0.1,0.25,0.5,0.75,0.9,0.95}; network predicts median + softplus increments (non-crossing by construction); loss = Σ components Σ τ pinball on vol-standardized targets; component weights fixed 1:1 (design constant). `q_top` derived from the predictive CDF: `q_top(i,t) = 1 − F_i,t(Q_{1−p} of the cross-section's predictive medians)` — computed, not separately trained.

### 10.3 Walk-forward & refits (`walk_forward.py`)
Annual refits. For test year Y: train = panel start→Y−2; val = Y−1 (calibration init: PID, AA priors, κ; early stopping); test = Y. Purge 2d, embargo 5d around fold boundaries. Holdout = final 18 months, untouched until Phase 13.

### 10.4 Panel batching (the "all at once" implementation)
`panel_dataset.py`: a batch element is **one date**: tensors `(N_t × 64 × 18)`, `(N_t × 12)`, masks for variable N_t; batch = 8 dates. f4's attention runs within each date over its N_t tokens. Optimizer AdamW lr 1e-3 cosine, wd 1e-4, clip 1.0, ≤30 epochs, early stop patience 5 on val pinball. **5 seeds per neural expert, quantile-averaged** (fixed policy, 0 extra trials).

### 10.5 HPO (`hpo_local.py` / `hpo_vertex.py`)
Budget (hard, in config): f3/f4 → 30 trials per refit over {lr∈[3e-4,3e-3] log, dropout∈[0,0.3], channels∈{32,64}, L∈{32,64,128}}; f1/f2 → 20 trials over standard grids. Local: Optuna TPE. Vertex (optional): Vizier study; either way **every trial auto-calls `log_trial(category='hpo', study_id=...)`** — this is mandatory plumbing, tested.

QA: overfit canary (train on 1 date, loss→~0); seed-variance report; f4 attention ablated to identity must reproduce f3 within tolerance (architecture wiring test); throughput sanity.

---

## 11. PHASE 9 — AGGREGATION

`aa.py`: Vovk Aggregating Algorithm over the quantile-forecast game. Implementation: maintain expert weights `w_k ∝ prior_k(t) * exp(−eta * cum_pinball_k)`, eta=1 on standardized losses with weight floor 0.01; aggregated forecast = weighted quantile-average with substitution-step re-sort (non-crossing). `gate_prior.py`: small MLP on `[M_t posterior, p_fresh]` → prior over experts, trained on val year only (its 2 hyperparams inside the f4 Vizier budget). `universal_portfolio.py`: Cover–Ordentlich CRP-with-side-information (side info = regime bucket) over the experts' implied top-K portfolios — benchmark only.
**GATES recorded:** G4 (translation): AA-aggregate vs best-single-expert-on-val, DM on net daily P&L, p<0.05, else AA is demoted to robustness layer and headline = best single expert. G5 (cross-section value): f4 vs f3 net P&L DM p<0.05, else cut the attention layer.

---

## 12. PHASE 10 — DECISION LAYER

Order of operations per decision day t (per surviving structure):
1. Calibrated predictive per asset (PID) → robust edge `mu_tilde = sign(mu)*max(0,|mu|−kappa*sigma_pred)`, κ from val (1 trial), in raw return units.
2. Score `s = q_top_cal * (mu_tilde − c_hat_roundtrip) / sigma_pred^0.5`; eligibility: `q_top_cal > 0.55`, `mu_tilde > c_hat`, spread < 40bps, ADV share of intended trade < 1%.
3. **Buy/hold band** (`band.py`): enter only rank ≤ K_buy=20; hold while rank ≤ K_hold=60.
4. **GP partial adjustment** (`gp_aim.py`): aim = score-proportional capped weights (w_max=7.5%) over the band-eligible set; per-asset trade rate τ_i from the discrete-time Gârleanu–Pedersen solution with quadratic cost coefficient λ_i calibrated to the local cost curve (fit quadratic to `c_hat` around typical trade size), risk γ=5, and signal decay φ = the Pass-8 measured decay of the dominant signal for the structure. Unit tests: λ→0 ⇒ τ→1; λ→∞ ⇒ τ→0; φ→∞ ⇒ aim→current Markowitz.
5. **No-trade band**: skip trades with |Δw| < 25 bps.
6. **Sizing apex / gross**: risk-constrained Kelly (`rc_kelly.py`, cvxpy): maximize Σ_j π_j ln(r_j^T b) over J=500 scenarios drawn from the joint calibrated predictive (Gaussian copula with DCC-NL correlations + conformal marginals), subject to `log_sum_exp(log π_j − λ_K * log(r_j^T b)) ≤ 0` with λ_K solving target P(DD>30%)≤5% on val; cap gross at 100%, per-name 7.5%. Fallback if infeasible: capped score weights.
7. **Throttles**: multiply gross by `g_t` (BOCPD) and `scale_t` (e-monitor, floor 0.25).
8. **Turnover budget**: hard projection so realized one-sided turnover ≤ 50%/month (scale Δw down pro-rata if exceeded; count occurrences).
`qp_challenger.py`: full cost-aware QP rung; must beat GP net OOS (DM) to be used.

---

## 13. PHASE 11 — BACKTEST ENGINE

`engine.py`: vectorized daily event loop, but fills and accounting are explicit:
- Decision at close t (or 15:30 for ON structure) → target weights.
- Fills next session at `open_auction_px` (or `close_auction_px` for ON entry), price perturbed by `±epsilon_exec` uniform noise (from Phase-5 measurement) in stress mode; cost per executed dollar = `0.5*s_blend + impact(participation)`; impact uses actual trade size vs ADV.
- Accounting in adjusted-return space (dividends embedded), costs subtracted explicitly; cash earns RF (French series). Delisting days: position exits at terminal_return.
- Outputs per run: daily returns (gross/net), positions, trades, turnover, cost drag, exposure, per-asset attribution → `reports/runs/<run_id>/`.
QA: conservation test (NAV identity: ret_net = ret_gross − costs to 1e-10); a hand-computed 3-asset, 10-day fixture must match to the cent; baseline rungs re-verified through this engine.

---

## 14. PHASE 12 — STRESS & NEGATIVE CONTROLS

`stress/`: (G6) cost shock — rerun at 1.5× and 2× spread surface; must beat benchmark at 1.5×. (G7) timing shock — fills at {auction, first print, daily VWAP (vw column)}; must beat benchmark at the worst. (G-perturb) OHLCV tick-scale jitter; top-K Jaccard ≥ 0.8 day-over-day median. (feed_contract) recompute one test-year on IEX bars: pipeline must REFUSE (contract test). (G10 negative controls, all required green): leak canary (inject r_{t+1} into a feature → IC alarm >0.15 must fire, build fails if detector silent); label-shift placebo (+1 day → all |t|<1); ticker-shuffle placebo (within-date permutation → cross-sectional alpha dead). Seed stress: report dispersion of net Sharpe across the 5 seeds; flag if max−min > 0.5.

---

## 15. PHASE 13 — VALIDATION FIREWALL & HOLDOUT PROTOCOL

1. **CPCV** (`cpcv.py`): 8 groups choose 2 → 28 paths over train+val years; purge=2d, embargo=5d; report the path distribution of net Sharpe. **PBO** via CSCV on the strategy-configuration set: require PBO < 0.20.
2. **DSR** (`dsr.py`, Appendix A.9): N = `registry.current_N()` (machine-read). Require DSR > 0.95 vs the EW benchmark.
3. **SPA** (`spa.py`): Hansen SPA, stationary bootstrap B=2000, vs benchmark; require p < 0.05. Romano–Wolf across the ladder.
4. **Attribution** (`attribution.py`): daily strategy returns on FF5+UMD+STR **plus in-house overnight-momentum and intraday-momentum factors** (built from our panel; 2 trials); require residual alpha t > 2.
5. **Holdout unlock protocol**: (i) write `registry/prereg/PREREG.md` (frozen config hash, gate table, trial ledger snapshot, hypothesis text from §0.1); (ii) git-commit; (iii) only then remove `.LOCKED` and run the frozen bundle ONCE on the holdout; (iv) all gates G2,G3,G4,G5,G6,G7,G8(PBO/DSR/SPA),G9(attribution) evaluated; (v) re-lock; the holdout is never run again under this config lineage.

---

## 16. PHASE 14 — REPORTS (deliverables)

`reports/FINAL_REPORT.md` containing: G1 diagnostics; survivorship bound; baseline ladder table with DM/RW; expert comparison incl. f4-vs-f3 and AA translation results; calibration (PID coverage paths, e-process plots); cost/turnover decomposition; stress grid; CPCV path distribution; PBO/DSR(N printed)/SPA; attribution; the Fundamental-Law ceiling box (IR ≈ IC·√BR·TC with measured numbers); the kill/ship verdict; and the full trial ledger as an appendix. Plus `reports/PAPER_TABLES/` (LaTeX-ready CSVs).

---

## 17. MASTER GATE TABLE (pre-registered; copied into config)

| Gate | Phase | Test | Threshold | On fail |
|---|---|---|---|---|
| G0a | 1 | Day-1 audits complete | report exists | halt |
| G0b | 2 | unknown terminal types | <5% (soft) | warn+bound |
| G1 | 5 | predictability exists | §7 criteria | HALT, negative-result report |
| G1b | 5 | structure break-even | §7 | drop structure |
| G2 | 10/13 | turnover one-sided | ≤50%/mo | scale down / fail |
| G3 | 13 | beat LightGBM rung net | DM+RW p<0.05 | ship rung 6 instead |
| G4 | 9/13 | AA beats best expert (net P&L) | DM p<0.05 | demote AA |
| G5 | 9/13 | f4 beats f3 (net P&L) | DM p<0.05 | cut attention layer |
| G6 | 12 | 1.5× cost stress | net > benchmark | fail |
| G7 | 12 | worst-timing fills | net > benchmark | fail |
| G8 | 13 | PBO / DSR / SPA | <0.2 / >0.95 / <0.05 | fail (do not trade) |
| G9 | 13 | residual alpha after attribution | t > 2 | fail |
| G10 | 12 | negative controls | all green | build fails |
| G11 | (live, App. F) | paper-trade e-process + fill tracking | no alarm; ≤ noise floor | no capital |

---

## 18. EXECUTION ORDER (`make all`)
```
setup → test → data-alpaca → data-factors → data-universe → day1-audits [G0a]
→ universe [G0b] → costs → labels → features → diagnostics [G1, G1b]
→ baselines → infra (bocpd, cov, pid, e-monitors) → train (experts, HPO, seeds)
→ aggregate [record G4/G5 inputs] → decide → backtest → stress [G6,G7,G10]
→ firewall (cpcv, pbo, dsr, spa, attribution; holdout unlock protocol) [G2..G9]
→ reports
```
Estimated wall time (single workstation w/ 1 GPU): ingestion 4–10 h (API-bound), features 1–2 h, diagnostics <1 h, baselines 1 h, training ~(4 refits × 30 HPO × ~10 min + 5 seeds × final) ≈ 1–3 GPU-days, everything else <1 day.

---

## APPENDIX A — FORMULAS (implement exactly)

**A.1 Labels.** `r_on(i,t+1)=ln A_O(i,t+1) − ln A_C(i,t)`; `r_in(i,t+1)=ln A_C(i,t+1) − ln A_O(i,t+1)`; `r_cc=r_on+r_in`; `r_oo(i,t+1)=r_in(t+1)+r_on(t+2)`. `z_top = 1{r ≥ Q_{0.8,t+1}(universe)}`. `sigma_t^2 = 0.94*sigma_{t−1}^2 + 0.06*r_cc,t^2`, floor 0.002. `y=r/sigma`.

**A.2 Pinball.** `L_tau(y,q)= max(tau*(y−q), (tau−1)*(y−q))`; total `Σ_c Σ_tau L_tau`.

**A.3 BOCPD.** Run-length posterior recursion: `P(r_t=r_{t−1}+1) growth ∝ (1−H)*pred_t; P(r_t=0) ∝ H*Σ ...` per Adams–MacKay (2007) Eq. 3–7 with NIG-Student-t predictive.

**A.4 Corwin–Schultz.** With `beta = Σ_{j=0,1}[ln(H_{t+j}/L_{t+j})]^2`, `gamma=[ln(H_{t,t+1}/L_{t,t+1})]^2`, `alpha=(sqrt(2β)−sqrt(β))/(3−2√2) − sqrt(γ/(3−2√2))`, spread `S=2(e^α−1)/(1+e^α)`; average overlapping-day estimates over 21d; negatives→0.

**A.5 e-process.** As §9.4; Ville: `P(sup e_t ≥ 1/α) ≤ α` under calibrated null.

**A.6 Conformal PID.** As §9.3 (P+I only).

**A.7 AA.** `w_k,t ∝ prior_k,t * exp(−η L_k,1:t−1)`; aggregate quantiles = Σ w_k q_k,τ then re-sort; η=1 on standardized loss.

**A.8 GP.** `x_t = x_{t−1} + τ (aim_t − x_{t−1})`; implement the discrete-time closed form of Gârleanu–Pedersen (2013), §1 (their Eq. (5)–(8)) with Λ=λΣ; aim weights ∝ φ-discounted Markowitz targets; verify limits in tests.

**A.9 DSR.** `DSR = Φ( ((SR − SR0)·sqrt(T−1)) / sqrt(1 − γ3·SR + ((γ4−1)/4)·SR²) )`, `SR0 = sqrt(V[SR_trials])·((1−γE)·Φ^{-1}(1−1/N) + γE·Φ^{-1}(1−1/(N·e)))`, γE=0.5772, N from ledger; γ3,γ4 = skew/kurtosis of daily net returns.

**A.10 Break-even.** `required_gross_daily_alpha = crossings_per_day × mean(c_hat over intended top-K)`; structure viable iff measured component decile spread × plausible capture (≤30%) > required.

## APPENDIX B — configs/v1.yaml (verbatim starting config)
```yaml
seed: 1338
panel: {start: 2016-01-01, holdout_months: 18, price_min: 5.0, adv_min_usd: 5_000_000,
        spread_max_bps: 40, min_age_days: 126, top_p: 0.20}
labels: {ewma_lambda: 0.94, sigma_floor: 0.002, quantiles: [0.05,0.1,0.25,0.5,0.75,0.9,0.95]}
costs: {fee_floor_bps: 0.5, c_impact: 0.10, spread_cap_bps: 500}
structures: [OO, ON, ID]            # pruned by G1b
bocpd: {hazards: [0.015873, 0.007937, 0.003968], nig: {mu0: 0, kappa0: 1, alpha0: 2, beta0: 1}}
pid: {alphas: [0.2, 0.1], eta_p: 0.05, eta_i: 0.005}
emonitor: {alarm: 20.0, floor: 0.25, beta: 0.5}
experts: {seeds: 5, hpo_budget: {f1: 20, f2: 20, f3: 30, f4: 30}, param_cap: 300000}
decision: {kappa: null,  # set on first val year, then frozen; logged as trial
           k_buy: 20, k_hold: 60, w_max: 0.075, gross_cap: 1.0, no_trade_bps: 25,
           gamma_risk: 5.0, dd_target: 0.30, dd_prob: 0.05, turnover_budget_m: 0.50}
gates: {pbo_max: 0.20, dsr_min: 0.95, spa_p: 0.05, dm_p: 0.05, ic_min: 0.005,
        cost_stress: 1.5, jaccard_min: 0.8, alpha_t_min: 2.0}
validation: {refit: annual, purge_days: 2, embargo_days: 5, cpcv_groups: 8, cpcv_test: 2,
             bootstrap_B: 2000, nw_lags: 10}
```

## APPENDIX C — MODULE CONTRACTS & REQUIRED TESTS (abridged signatures)
Every module exposes typed functions; `tests/` must include at minimum: contract tests (PIT, feed, fit-scope, terminal-return), golden-master mini-backtest (1y × 50 names, byte-compared), engine conservation fixture, BOCPD/e-monitor/PID synthetic-validity tests (§9 QA), GP limit tests, non-crossing tests, panel-batch mask tests, ledger append-only test, holdout-lock test, and the three placebo/canary harnesses. Target: `pytest -q` < 10 min, 100% of contracts covered.

## APPENDIX D — TRIAL LEDGER RULES (honest N)
Charged to N: every HPO trial (auto), every expert in the roster (5+1), every structure candidate (3), every hazard (3), every covariance rung (3), every sizing-ladder rung (4), κ (1), band pair (1), λ_K (1), in-house factors (2), gate-threshold alternatives if ever evaluated (each 1). Not charged: fixed design constants never evaluated against data (declared in config as `frozen:`). The DSR section of the final report prints the ledger row count and a category breakdown.

## APPENDIX E — QA/QC LOG OF THIS PLAN (performed before output)
1 Label identity & indexing cross-checked (A.1 vs §6.4 tests). 2 ON-leg lookahead resolved via 15:30 minute-bar snapshot with its own decision_ts. 3 Feed contract enforced by test, not convention. 4 Corporate-action gap (pre-2020) handled by reconciliation + bias bound, not assumed away. 5 Holdout protected by filesystem lock + loader check + unlock protocol. 6 All previously-identified λ-surfaces removed; remaining knobs enumerated in Appendix D. 7 Calibration never appears in a loss. 8 EW benchmark fairness test included. 9 Engine accounting conservation test included. 10 GP closed-form risk mitigated by limit tests + paper-equation reference. 11 e-monitor & PID validated on synthetic nulls before use. 12 Turnover budget enforced by projection, not hoped. 13 Negative controls test the detectors themselves. 14 Per-ticker models and ticker embeddings prohibited in code review checklist + tests. 15 Trial ledger plumbed through HPO automatically. 16 Kill gates write reports and halt; a halt is a defined success path. 17 Minute-bar volume fetched lazily to keep ingestion tractable. 18 Survivorship bound is a deliverable table. 19 Compute estimates sanity-checked vs model sizes. 20 Every formula in Appendix A has an owning module and test.

## APPENDIX F — LIVE/PAPER LOOP (next milestone; out of scope for this build)
16:05 ET: pull SIP daily bars + auctions; update features/PID/e-monitors/BOCPD; frozen bundle → targets via GP+band+RC-Kelly; 09:25 ET: submit `time_in_force=opg` (fallback marketable limit); log fills vs auction px; weekly e-process & fill-tracking report; ≥3 months paper with no e-alarm and fill tracking error ≤ epsilon_exec before any capital (Gate G11).
