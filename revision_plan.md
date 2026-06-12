# REVISION PLAN — TACTIC-MoB v4 regime-robust re-validation

**Author:** prepared for a takeover agent. Self-contained. Execute top to bottom.
**Date:** 2026-06-13. **Repo root:** `currect QR`. **Branch:** work on `main` or a feature branch.
**Caveman note:** repo owner runs sessions in "caveman" comms mode; that affects chat prose only,
NOT code/commits/this doc. Write normal code and commit messages.

> **Prime directive — no OOS peeking.** Every design choice below is justified by economic/finance
> principles and *ex-ante* data availability ONLY. Do NOT read, tune to, or select models on any
> 2022–2026 backtest result while building. OOS metrics are computed ONCE at the end, reported, and
> never fed back into model/feature/hyperparameter choices. The prior run's OOS numbers in
> `results/` and `notes.md` are off-limits as a selection signal.

---

## 0. Why we are redoing this (root-cause, principle-level)

The previous chronological run (train 2016–2020 / val 2021 / OOS 2022→2026) produced a model that
**collapsed to a near-constant predictor** and did not beat SPY. Three principle-level causes — all
fixable without looking at OOS:

1. **Validation regime ≠ deployment regime.** A single calm calendar year (2021, a low-vol melt-up)
   was the entire early-stopping signal. Early-stop on one placid regime selects an under-fit model
   and gives no robustness to the bear/high-rate regime that any honest OOS must contain. *Principle:
   validation must span multiple regimes, or you are tuning to one regime by construction.*
2. **Single chronological OOS wastes the regime budget.** With only ~10.5 years of usable data, a
   chronological split spends each regime exactly once (either train XOR test). You cannot both
   "train on a bear" and "test on a bear" if you only have ~two bears. *Principle: with scarce
   regimes, use combinatorial cross-validation (de Prado CPCV) so every regime appears in OOS in
   some fold while the model is trained on the others.*
3. **Objective/portfolio mismatch let the model be lazy.** Summed pinball on vol-standardized
   open→open returns is dominated by irreducible variance; predicting the unconditional quantiles is
   ~optimal, so the optimizer parks there (cross-sectional spread of the median prediction was
   ~1.6% of the target's own scale). A top-5 equal-weight book then cannot express a weak,
   broad cross-sectional signal. *Principle: if the signal is weak-but-real (low IC), you must (a)
   force the model to rank (cross-sectional objective) and (b) harvest IC with breadth (many small
   bets), not concentration. Fundamental Law: IR ≈ IC·√breadth.*

This plan fixes all three: **multi-regime validation (CPCV) + a baseline-beating sanity gate +
a ranking objective + a breadth/market-neutral portfolio.**

---

## 1. Hard constraints (verified, do not relitigate)

- **Data floor is 2016-01-04.** Alpaca SIP returns nothing before it (probed AAPL at 2004/2010/2015
  → earliest bar always 2016-01-04). This is a hard vendor floor.
- **No pre-2016 history in the OOS/clean panel.** Deep history (Stooq) in
  `data/diagnostic_quarantine/` is **forbidden** from joining curated features — enforced by
  `tests/test_quarantine_isolation.py` and PLAN §0.3. Do NOT wire deep history into the OOS panel.
  Free deeper daily data *does* exist (yfinance, 1990+) but is **survivorship-biased**; it may be
  used ONLY for an optional, clearly-labeled train-only pretraining ablation (see **§1A**), never for
  scoring. A survivorship-safe deep source (CRSP/Sharadar) is paid and out of scope.
- **Usable span = 2016-01-04 → 2026-06-11** (already ingested for all 196 names; bars+auctions+
  labels+features+spreads present). User target end 2026-06-10 is already covered.
- **PIT / fit-scope discipline.** Feature standardization and any target normalization are fit on
  the *training rows of each fold only*. Purge 2 days + embargo 5 days around every train/test seam.

### Verdict on "include 2022 in OOS"
- **It is NOT stupid — but only via CPCV, not via a naive chronological split.** With CPCV, 2022 is a
  test block in several folds while the model trains on the other regimes (2018-Q4 selloff, 2020
  COVID crash, 2025–26 volatility). That is a *fair* test of "did the model learn bear dynamics from
  other bears and generalize to 2022."
- A naive "train≤2021 / OOS 2022+" is the design that just failed and is **rejected**: training then
  contains no sustained bear (only the 3-week 2020 crash) and the lone val year is calm. Do not
  rebuild that.

---

## 1A. Free deep-history data — research findings + decision (2026-06-13)

**Question asked:** can we get *free, public* data deeper than 2016 at high-enough time resolution,
and would it help? **Answer: deeper free daily data exists, but it is survivorship-biased, so use it
(at most) for TRAIN-ONLY augmentation — never for the OOS.**

**What was probed (live, QA'd):**
| Source | Free? | Depth | Resolution | Verdict |
|---|---|---|---|---|
| Alpaca SIP | yes | **2016-01-04 floor** | daily/minute | hard floor; primary clean source |
| Stooq (per-symbol CSV) | yes | deep | daily | **blocked** programmatically (rate-limit/captcha); manual bulk only; repo quarantines it |
| **yfinance (Yahoo)** | yes, no key | **1990+** (AAPL/MSFT/GE 9179 daily rows; SPY 1993) | daily (+Adj Close) | works; deep; **survivorship-biased** |
| Tiingo free | key | deep | daily | active-ticker focused; partial delisted; not survivorship-safe |
| Nasdaq Data Link "WIKI" | yes | frozen 2018 | daily | discontinued + biased; dead end |
| Ken French factors | yes | **1926+** | daily | portfolio returns (regime/market-state context), NOT single-name tradeable |
| CRSP / Sharadar SEP-SFP | **NO (paid)** | deep, survivorship-safe | daily | the only clean deep option; out of scope (not free) |

**Resolution is sufficient, and is NOT the blocker.** The strategy is daily (L=64 daily bars + daily
open→open labels). Free daily history (yfinance, 1990+) has the native resolution the model needs.
Minute data would only refine the cost/execution layer, not the signal.

**The blocker is survivorship bias (proven).** yfinance returns **0 rows for delisted/dead tickers**
— tested LEH, WCOM, ENE, BSC, WAMUQ, CIT, ABK, DYN, EK, TYC, old-DELL → all EMPTY; only survivors
remain. A pre-2016 backtest on such data is upward-biased (you would "never have held Lehman into
2008"). **No free survivorship-safe deep equity source exists** (CRSP/Sharadar are paid; WIKI/Tiingo
are biased too). Alpaca *does* capture delistings (inactive `assets_snapshots` + corporate actions)
but only from 2016.

**Would it be good to have?**
- *Regime diversity (train/CV):* genuinely yes — 2016-2026 has only ~2 sustained bears (2020, 2022);
  pre-2016 adds the dot-com bust, **2008 GFC**, 2011 EU crisis, 2015-16 selloff. More stress regimes
  to learn from.
- *Honest evaluation:* **no** — survivorship bias in the OOS would fake the score and defeat the
  entire anti-overfitting purpose (PLAN §0.3). Bias in *evaluation* is fatal; bias in *pretraining*
  is mild, and is further neutralized by the per-date **cross-sectional demeaning** in §4.3 (which
  removes the survivor up-drift; the label is a 1-day cross-sectional relative return, not a
  long-horizon absolute one, so survivorship sensitivity is low to begin with).

**DECISION:**
1. **Primary stays CPCV on the SIP-clean, delisting-aware 2016-2026 window.** Do not put free deep
   history into the OOS. Keep `tests/test_quarantine_isolation.py` honored.
2. **OPTIONAL experiment (clearly labeled, train-only):** an *augmented-pretrain* path that pretrains
   f4 on yfinance 1990-2015 **surviving-name** daily bars (features+demeaned labels only), then
   fine-tunes + validates + tests strictly on the clean 2016-2026 CPCV. Report as an ablation:
   "does GFC/dot-com pretraining improve clean-window CPCV metrics?" If yes, modest robustness win;
   if no, drop it. This NEVER touches OOS scoring and must be flagged as survivorship-biased input.
3. **Not worth blocking on.** The clean-data CPCV redesign (§3) is the main lever. Pretraining is a
   nice-to-have ablation, not a dependency.

---

## 2. Realistic regime map (from public market history, NOT from our backtest)

Use this to (a) sanity-check that CPCV folds mix regimes and (b) stratify the final OOS report.
Calendar → dominant regime label (ex-ante, common knowledge):

| Window | Regime label | Why it matters |
|---|---|---|
| 2016 H1 | oil-crash mini-bear / recovery | early-2016 selloff, energy stress |
| 2016 H2–2017 | ultra-low-vol bull | record-low VIX, trend |
| 2018 Q1 | "volmageddon" vol spike | Feb VIX blowup |
| 2018 Q4 | tightening correction (~−20%) | Fed hikes, growth scare |
| 2019 | rate-cut bull | mid-cycle easing |
| 2020 Feb–Mar | COVID crash | fastest bear on record |
| 2020 Apr–Dec | V-recovery / ZIRP | liquidity melt-up |
| 2021 | melt-up + meme/SPAC | low-vol, retail froth |
| 2022 | rate-hike **bear** (~−25%) | inflation peak, duration repricing |
| 2023 Q1 | regional-bank crisis | SVB/credit stress |
| 2023 | narrow mega-cap AI rally | breadth collapse, cap-weight dominance |
| 2024 | broad bull / soft landing | wide breadth |
| 2025 | policy/tariff volatility | choppy, regime-uncertain |
| 2026 H1 | elevated volatility (current) | per owner; treat as stress |

**Principle:** 2016–2026 contains ≥2 sustained bears (2020, 2022), multiple corrections (2018-Q4,
2023-Q1, 2025), ZIRP→high-rate transition, and narrow-vs-broad breadth regimes. That is enough
diversity for CPCV to be meaningful. The scarcity is *number of independent bears* (≈2), which is
exactly why combinatorial (not chronological) CV is required.

---

## 3. Target design — primary and fallback

### 3A. PRIMARY: Combinatorial Purged Cross-Validation (CPCV) — de Prado §15.1 / PLAN Phase 13
- **Blocks:** split the ordered trading-day axis (2016-01-04 → 2026-06-11, ~2625 sessions) into
  `N_GROUPS = 8` contiguous, roughly equal groups (~328 sessions ≈ 1.3 yr each). Record each group's
  calendar span and tag it with the §2 regime(s) it covers.
- **Folds:** every combination of `K_TEST = 2` groups held out as OOS → `C(8,2) = 28` folds. In each
  fold the other 6 groups are training. Config already declares `validation.cpcv_groups: 8`,
  `cpcv_test: 2` in `configs/v1.yaml`.
- **Purge + embargo:** drop training rows whose label window overlaps any test group (purge = 2
  trading days, the label horizon) and embargo 5 trading days *after* each test group before training
  resumes. Use `validation.purge_days: 2`, `embargo_days: 5`.
- **Nested early-stop val (no leakage, no regime-mono):** inside each fold's 6 training groups, hold
  out the **chronologically last 15%** of training *as an embargoed val slice* for early-stopping
  ONLY. Because the 6 groups are drawn from across 2016–2026, this val slice is regime-mixed, not a
  single calendar year. Purge+embargo it from both the kept-train and the test groups.
- **Combined OOS path:** each session appears in the test set of multiple folds (each group is in
  `C(7,1)=7` of the 28 folds). Average the per-fold OOS quantile predictions for each (date,entity)
  across the folds in which it was OOS → one **combined out-of-sample prediction series spanning the
  entire 2016–2026 window, including 2022**. This is the headline OOS.
- **Path distribution:** CPCV also yields a *distribution* of fold-level net Sharpe (28 paths). Report
  its mean/std/quantiles and feed it to PBO (§6).
- **2022 guarantee:** the group(s) covering 2022 are OOS in 7 of 28 folds while trained on the other
  regimes → 2022 is genuinely out-of-sample and bear-tested.

### 3B. FALLBACK: expanding-window walk-forward with multi-year val (if CPCV proves too costly)
- Folds (annual refit, expanding train, **2-year** val for regime diversity, purge2/embargo5):
  - F1: train 2016–2019 · val 2020–2021 · test 2022
  - F2: train 2016–2020 · val 2021–2022 · test 2023
  - F3: train 2016–2021 · val 2022–2023 · test 2024
  - F4: train 2016–2022 · val 2023–2024 · test 2025
  - F5: train 2016–2023 · val 2024–2025 · test 2026-H1
- This puts **2022 in OOS (F1)** AND, in later folds, **2022 in train** so the model learns bear
  dynamics. Stitch test years into one OOS path. Note F1's train still lacks a long bear (principle
  weakness) — that is why CPCV is preferred; F1 is reported but flagged.
- Multi-year val (2 calendar years each) directly satisfies "include more val data" and mixes
  regimes for early-stopping.

**Recommendation:** implement **3A (CPCV)** as primary; keep **3B** as a cheaper sanity cross-check.
Both reuse the same model/feature/portfolio code; only the splitter differs.

---

## 4. Modeling remediation — escape the constant-collapse (root cause #3)

Do these regardless of splitter. Each has an explicit, OOS-free acceptance check.

1. **Unconditional-quantile baseline + sanity gate (MUST implement first).**
   - Implement `baseline_pinball(y_train, y_eval, taus)` = pinball of the *train empirical quantiles*
     (constant per τ) on the eval set. Add to `src/tactic/models/baselines.py` (new) or
     `validation/`.
   - **Gate S1 (internal, val-only):** the trained model's val pinball must beat the baseline val
     pinball by **≥ 1%** (relative). The prior run beat it by 0.06% → would FAIL S1, catching the
     collapse *before* any OOS. Log to the registry; if it fails, the model is not promoted.
2. **Cross-sectional ranking objective (fix the lazy optimum).** Keep pinball but add a
   per-date ranking term so the loss rewards ordering names, not just matching the marginal:
   - Option A (preferred, simple): add a **per-date soft-rank / pairwise logistic** auxiliary on the
     median head: for each date, encourage `q50_i > q50_j` when `y_i > y_j`. Weight `λ_rank` (start
     0.1–0.5, tune on val pinball+val IC, never OOS).
   - Option B: **listwise** (Spearman-surrogate / ListMLE) on q50 per date.
   - Implement in `src/tactic/models/train.py` `pinball_torch` caller; keep non-crossing quantile
     constraint intact.
   - **Check:** cross-sectional std of q50 per date must rise materially above the prior ~0.018 (it
     was ~1.6% of target σ). Target: median per-date q50 dispersion ≥ ~0.1·σ_target on val. (Diagnostic,
     not an OOS read.)
3. **Per-date target conditioning (optional, principled).** The label is already vol-standardized;
   additionally **cross-sectionally demean the target within each date** (subtract the date's mean y)
   so the model is explicitly trained on *relative* returns — which is all a cross-sectional book can
   monetize. Refit any scaler on train rows only.
4. **Capacity / schedule sweep (val-selected only).** Sweep on val pinball (NOT OOS):
   LR ∈ {3e-4, 1e-3}, weight_decay ∈ {1e-4, 3e-4}, attention heads {4} (keep param_cap 300k),
   patience {10}. Keep seeds=3 quantile-averaged. Use `experts.hpo_budget` ledger; log every trial
   to `registry` (honest-N).
5. **Feature sanity audit (no model needed).** Confirm the 12 static cross-sectional features and
   the market-state channels actually vary across entities per date (no all-equal columns after the
   ch06 vw fix). Assert per-date cross-sectional std > 0 for each static feature on a sample of dates.

---

## 5. Portfolio / decision — harvest weak IC with breadth (root cause #3, cont.)

The previous top-5 long-only book cannot express a broad weak signal and is dominated by cap-weighted
SPY in narrow-breadth bull years. Replace/extend with:

1. **Primary book: dollar-neutral cross-sectional long–short.** Long the top decile by q50, short the
   bottom decile (or long top-K / short bottom-K with K≈30–40 of ~196), equal-weight or IC/score-
   weighted, gross capped per `decision.gross_cap`, `w_max` per name. This is *market-neutral*, so it
   is judged on alpha, not on out-running a cap-weighted index — the correct test for a
   cross-sectional signal. Net of costs via existing `spread_surface`.
2. **Secondary book: long-flat breadth.** Keep a long-only variant but at breadth (k_buy ≈ 20–40,
   k_hold ≈ 60–80) for comparability with the index and with prior results.
3. **Benchmarks:** (a) SPY buy-and-hold, (b) **equal-weight of the 196-name universe** (this isolates
   selection skill from the equal-weight-vs-cap-weight factor that hurt us in 2023–24). Report against
   both.
4. **Sizing:** start with the PLAN §12.6 capped-weight fallback (already used). RC-Kelly/GP partial-
   adjustment (PLAN §12) is a later upgrade; do not block on it.

**Principle:** a long–short, breadth-driven book is the honest vehicle for a low-IC daily signal; if
the signal is real, IR≈IC·√breadth shows up as positive market-neutral alpha across CPCV folds. If it
does not, that is a clean negative result.

---

## 6. Firewall / significance (compute, report; gates noted not enforced unless owner says otherwise)

All of these are currently **stubs** — implement what you use:
- `validation/cpcv.py::cpcv_paths` — STUB → implement (drives §3A).
- `validation/walk_forward.py::make_folds` — STUB → implement (drives §3B).
- `validation/pbo.py` — STUB → implement **PBO** (probability of backtest overfitting) over the CPCV
  path matrix (de Prado). Gate G8 threshold `pbo_max: 0.20`.
- `validation/dsr.py` — DONE. Compute **Deflated Sharpe** on the combined OOS net returns with honest
  N = number of model/HPO trials from the registry. Gate `dsr_min: 0.95`.
- `validation/dm_test.py` — DONE. Diebold-Mariano vs SPY and vs equal-weight universe.
- `validation/spa.py`, `romano_wolf.py`, `firewall.py`, `attribution.py` — STUBS; implement SPA /
  Romano-Wolf stepdown if you want G3/G9 verdicts, else explicitly mark "not run".
- **Report every gate's number (G1,G2,G3,G8,G9, S1) but do NOT halt on them** unless the owner
  re-enables enforcement. The owner's current instruction is "note gates, don't enforce."

---

## 7. Exact execution checklist (do in order; QA/QC after each)

> Commands are Windows-friendly. Heavy training runs on **Vertex AI** (see existing
> `vertex/train_on_vertex.py`); light work local. Submit to Vertex and poll to terminal state
> (`gcloud ai custom-jobs describe ... --format='value(state)'`), do not go offline.

**STEP 0 — refresh data to 2026-06-10 inclusive + verify.**
- Data already spans 2016-01-04→2026-06-11. Re-run only if stale:
  `python -c "from src.tactic.common.config import load_dotenv; load_dotenv(); from src.tactic.ingest.candidate_symbols import LIQUID_BIG; from src.tactic.ingest import alpaca_bars, alpaca_auctions; alpaca_bars.ingest(LIQUID_BIG, start='2016-01-01', end='2026-06-10'); alpaca_auctions.ingest(LIQUID_BIG, start='2016-01-01', end='2026-06-10')"`
- Rebuild panel: costs → labels → features (functions:
  `costs.spreads.build_spread_surface`, `labels.build_labels.build_labels`,
  `features.build_features.build_features`).
- **QA0:** assert 196 entities; date max ≥ 2026-06-10; `features` has 0 `inf` and the ch06 vw guard is
  present; per-date entity count ≥ 150 for all dates in 2017+ (warmup tail excluded).

**STEP 1 — baseline + sanity gate (§4.1).**
- Implement `baseline_pinball`; compute train-quantile baseline pinball on a quick chronological val.
- **QA1:** baseline numbers logged; S1 wiring unit-tested (a constant predictor FAILS S1).

**STEP 2 — modeling fixes (§4.2–4.5).**
- Add ranking auxiliary to training; add optional per-date target demeaning; feature-variance audit.
- **QA2 (val-only):** on a quick chronological train (2016–2020) / val (2021) smoke, confirm (a) model
  beats baseline val pinball by ≥1% (S1 PASS), (b) per-date q50 dispersion ≥ ~0.1·σ_target. If S1
  still fails, iterate on `λ_rank`/capacity (val-selected) before spending Vertex on CPCV. **Never
  consult OOS to make these choices.**

**STEP 2.5 — OPTIONAL augmented-pretrain ablation (§1A, train-only, skippable).**
- Only if pursuing the deep-history experiment. Build a SEPARATE quarantined panel from yfinance
  1990-2015 for the survivors of the 196-name list (`yf.download(sym, start='1990-01-01',
  end='2016-01-01', auto_adjust=False)`); compute the same features + **cross-sectionally demeaned**
  labels; keep it in `data/diagnostic_quarantine/` (do NOT join to curated). Pretrain f4 on it, then
  load weights as init for the §3 CPCV training on clean 2016-2026.
- **QA2.5:** assert the pretrain panel never enters any OOS/test set; assert the quarantine-isolation
  test still passes; report it as an ablation row ("pretrain on/off") on clean CPCV metrics only.
- If it does not improve clean-window metrics, DROP it. Never block the main path on this.

**STEP 3 — implement CPCV splitter + driver (§3A).**
- Implement `cpcv_paths` (returns the 28 train/test group partitions with purge+embargo masks).
- Add a CPCV training driver `src/tactic/models/train_cpcv.py` that, per fold: builds the panel,
  applies fold masks, fits channel-stats on fold-train only, carves the nested embargoed val slice,
  trains f4 (seeds=3, ranking-augmented), predicts the held-out test groups; then aggregates the
  combined-OOS predictions and saves per-fold net-return paths.
- **QA3:** unit-test that (a) no test row appears in any same-fold train set, (b) purge+embargo gaps
  hold (no train label window within 2d of a test group; ≥5d embargo after), (c) every session is OOS
  in exactly `C(7,1)=7` folds, (d) combined-OOS covers 2016→2026 incl all of 2022.

**STEP 4 — run on Vertex.**
- Extend `vertex_entry.py` / `train_on_vertex.py` to accept `--cv=cpcv` (and pass through
  `n_groups`, `k_test`, `patience`, ranking weight). 28 folds × 3 seeds is heavy → either one job
  looping folds on `n1-standard-16`, or fan out folds as separate jobs. Estimate runtime from a
  1-fold timing, set polling cadence, **stay online until SUCCEEDED**.
- **QA4:** confirm artifacts: combined `oos_predictions.parquet`, per-fold `paths.csv`
  (28 net Sharpes), `loss_history.csv` per fold/seed, `config.json` with the exact CV spec.

**STEP 5 — portfolio + metrics (§5).**
- Backtest the combined-OOS predictions with BOTH books (dollar-neutral L/S; long-flat breadth) vs
  BOTH benchmarks (SPY; equal-weight universe). Produce overall + **per-calendar-year** +
  **per-regime** (using §2 map) metrics: total, CAGR, vol, Sharpe, Sortino, maxDD, turnover, IC,
  hit-rate, coverage, pinball.
- **QA5:** NAV identity holds; costs > 0; coverage near nominal; IC computed per date then averaged.

**STEP 6 — firewall (§6).**
- DSR on combined-OOS net returns (honest-N from registry). PBO over the 28 paths. DM vs each
  benchmark. CPCV path-Sharpe distribution. Report all; do not halt.
- **QA6:** N used in DSR equals registry trial count; PBO in [0,1]; path count = 28.

**STEP 7 — report + graphs + commit.**
- Write `results/<run_id>/RESULTS_REVISION.md`: split design, regime map, equity vs both benchmarks,
  drawdown, per-year + per-regime tables, train/val loss curves (show no collapse: S1 margin), CPCV
  path-Sharpe histogram, PBO/DSR/DM, honest verdict. Graphs as PNG.
- Append everything to `notes.md` (append-only; never delete).
- **Security:** before any commit verify `.env` stays gitignored and NO `.env`/`data/curated`/raw
  `*.parquet` are staged (only `results/**` artifacts + code). Commit + push to
  `https://github.com/JulianAttemptsCoding/cutie-QT-`.

---

## 8. Acceptance criteria (definition of done)

1. Data verified to 2026-06-10+, 196 names, 0 inf, panel rebuilt. (QA0)
2. S1 sanity gate implemented and **model PASSES S1 on val** (beats unconditional baseline ≥1%);
   per-date q50 dispersion materially > prior 0.018. (QA2) — proves the collapse is fixed *without*
   OOS.
3. CPCV implemented and **unit-tested** for leakage/purge/embargo/coverage; 2022 confirmed OOS. (QA3)
4. Vertex run completes; combined-OOS spans 2016→2026 incl 2022; 28 fold paths saved. (QA4)
5. Dollar-neutral L/S + long-flat books backtested vs SPY AND equal-weight universe, with per-year
   and per-regime breakdowns. (QA5)
6. DSR + PBO + DM computed and reported (noted, not enforced). (QA6)
7. `revision_plan.md` followed; `RESULTS_REVISION.md` + graphs written; `notes.md` appended; pushed;
   `.env` never committed.
8. **Honest verdict** stated either way. A clean negative (no market-neutral alpha across regimes)
   is a successful, publishable result per PLAN §0.3.10 — do not manufacture a win.

---

## 9. File-change manifest (quick index for the takeover agent)

| File | Action |
|---|---|
| `src/tactic/models/baselines.py` | NEW — `baseline_pinball`, S1 gate helper |
| `src/tactic/models/train.py` | EDIT — add ranking aux (`λ_rank`), optional per-date target demean, S1 logging |
| `src/tactic/validation/cpcv.py` | IMPLEMENT `cpcv_paths` (stub today) |
| `src/tactic/validation/walk_forward.py` | IMPLEMENT `make_folds` (stub; for §3B fallback) |
| `src/tactic/validation/pbo.py` | IMPLEMENT PBO (stub today) |
| `src/tactic/models/train_cpcv.py` | NEW — CPCV driver + combined-OOS aggregation |
| `src/tactic/ingest/yfinance_deep.py` | NEW (OPTIONAL, §2.5) — yfinance 1990-2015 survivors → `data/diagnostic_quarantine/` only; pretrain panel. Never joins curated. |
| `src/tactic/backtest/run.py` | EDIT — add dollar-neutral L/S book + equal-weight-universe benchmark + per-regime grouping |
| `vertex/vertex_entry.py`, `vertex/train_on_vertex.py` | EDIT — `--cv=cpcv` passthrough, fold fan-out/loop |
| `configs/v1.yaml` | (already has cpcv_groups/cpcv_test/purge/embargo) — add `lambda_rank` if desired |
| `tests/` | NEW — CPCV leakage/coverage tests; S1 constant-predictor test |
| `results/<run_id>/RESULTS_REVISION.md` | NEW — final report |
| `notes.md` | APPEND — running log |

## 10. Pitfalls / gotchas (learned, save the next agent time)
- Vertex image is **Python 3.10**, installs pkg `--no-deps`; entry must `pip install pyyaml
  matplotlib "numpy<2"` (numpy<2 avoids ABI break with prebuilt torch/pyarrow). Already handled in
  `vertex_entry.py`; preserve it.
- Windows can't exec `gcloud/gsutil` `.cmd` via argv list → submitter uses `shell=True` with
  `gcloud storage cp`. Keep that.
- Curated bars are a **flat file** `data/curated/prices_daily.parquet` (a `.sha256` sidecar once
  broke directory reads). Keep flat.
- Bars manifest resumes only if cached range covers the request; widening dates forces a refetch
  (intended).
- features ch06 = `log(vw)`; guard `vw<=0 → close` (already fixed) or you reintroduce `-inf`.
- Do NOT join `data/diagnostic_quarantine/` (Stooq) into features — a test enforces this.
- Backtest `_open_to_open` aligns label at decision t to `open(t+1)→open(t+2)`; keep the 2-day purge
  consistent with this horizon.
- Honest-N: log EVERY model/HPO trial to `registry` so DSR's N is truthful (Vertex trials were
  previously logged ephemerally and undercounted N).
