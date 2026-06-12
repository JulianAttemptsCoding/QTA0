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
  Free deeper daily data *does* exist (yfinance, 1990+) but is **survivorship-biased**; it is used
  ONLY inside the **separate exploratory RUN 2**, which is benchmarked against equal-weight
  buy-and-hold of its own pool (so the bias cancels in the comparison) and is never deployable and
  never mixed into Run 1 (see **§1B**). A survivorship-safe deep source (CRSP/Sharadar) is paid and
  out of scope.
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
- *Honest evaluation vs SPY:* **no** — survivorship bias against an external SPY benchmark would fake
  the score and defeat the anti-overfitting purpose (PLAN §0.3). **This is why Run 2's benchmark is
  its OWN pool, not SPY** (§1B): both sides share the identical survivor pool, so the bias cancels in
  the difference. The residual survivor up-drift in training is further neutralized by the per-date
  **cross-sectional demeaning** in §4.3 (the label is a 1-day cross-sectional *relative* return, not
  a long-horizon absolute one, so survivorship sensitivity is low to begin with).

**DECISION — build TWO SEPARATE, INDEPENDENT RUNS (both are deliverables; see §1B for full specs):**
1. **RUN 1 (clean / honest):** CPCV on the SIP-clean, delisting-aware **2016-2026** Alpaca window.
   Benchmarks = SPY + equal-weight of the 196-name universe. This is the deployable-grade result.
   Keep `tests/test_quarantine_isolation.py` honored — the clean curated panel never sees deep data.
2. **RUN 2 (deep / exploratory):** a **completely separate** end-to-end run on free **yfinance deep
   daily history (2000-2026)** for a fixed pool of long-lived survivor names. It has its OWN panel
   (in a separate directory, NOT the clean curated one), its OWN train/val/test, and is benchmarked
   against **equal-weight buy-and-hold of every stock in its own pool** (NOT SPY). The matched-pool
   benchmark is the whole point: both the strategy and the benchmark draw from the identical
   survivor pool, so **the survivorship bias cancels in the relative comparison** — Run 2's "did the
   model beat simply owning its own pool" question is meaningful even though absolute returns are
   inflated. Run 2 is exploratory (NOT deployable; bias acknowledged), reported separately, and must
   never contaminate Run 1.

---

## 1B. THE TWO RUNS — read this carefully, they are independent deliverables

You will produce **two separate runs**, each with its own data, its own model training, its own
backtest, and its own report. **Do not mix them.** Run 1 is the honest/clean result; Run 2 is the
deep-history exploratory result. Everything in §3–§6 (CPCV, ranking loss, S1 gate, firewall) applies
to BOTH unless noted; only the data, the time span, and the benchmark differ.

| | **RUN 1 — CLEAN (primary, deployable-grade)** | **RUN 2 — DEEP (exploratory, NOT deployable)** |
|---|---|---|
| Data source | Alpaca SIP (already ingested) | **yfinance** (free, no key) |
| Time span | **2016-01-04 → 2026-06-10** | **2000-01-01 → 2026-06-10** (optionally 1990) |
| Universe / pool | the 196 `LIQUID_BIG` names | survivor pool = §1B.1 below |
| Delistings handled? | yes (Alpaca inactive assets + CA) | **NO — survivorship-biased** (acknowledged) |
| Panel location | `data/curated/` (clean) | **`data/deep_panel/` (separate; never `curated/`)** |
| Auctions / SIP spreads | yes | none → OHLC open→open labels + OHLC-derived spread surface |
| Splitter | CPCV 8 groups, C(8,2)=28 folds | CPCV **10 groups, C(10,2)=45 folds** (longer span) |
| **Benchmark** | **SPY + equal-weight 196-universe** | **equal-weight buy-and-hold of the entire Run-2 pool** (NOT SPY) |
| Output dir | `results/run1_clean_<id>/` | `results/run2_deep_<id>/` |
| Verdict use | deployable read | exploratory only; bias flagged in every artifact |

**Why Run 2's benchmark is the pool itself, not SPY:** Run 2's pool contains only names that survived
to today, so its absolute returns are inflated by survivorship. Comparing the strategy to **equal-
weight buy-and-hold of the same pool** makes both sides carry the *identical* bias, so it cancels in
the difference. The honest question Run 2 answers is therefore: *"Does the model's selection/timing
beat naively owning an equal slice of its own pool, net of costs, across 2000-2026 regimes (dot-com
bust, GFC, 2011, 2015-16, COVID, 2022)?"* That relative-alpha question is valid despite the bias.
Run 2 must NOT be compared to SPY as a headline and must NOT be called deployable.

### 1B.1 Run-2 pool definition (deterministic, no look-ahead beyond "exists today")
- Candidate names = the 196 `LIQUID_BIG` list **plus** any other liquid large-caps you want, BUT the
  pool is fixed by a single mechanical rule: **keep a name iff yfinance returns continuous daily data
  from `POOL_START` (default 2000-01-01) through 2026-06-10 with < 2% missing sessions.** Drop ETFs
  that postdate POOL_START if you want a pure-equity pool, or keep them — document the choice.
- This yields a *survivor* pool (≈100-160 names depending on POOL_START). That is expected and is
  exactly why the benchmark is the pool itself. Record the final pool list + each name's first date
  to `results/run2_deep_<id>/pool.csv`.
- **QA(pool):** assert every pool name has ≥ 0.98 × (#NYSE sessions in span) rows; assert no name
  starts after POOL_START + 5 sessions; print pool size.

### 1B.2 Run-2 equal-weight buy-and-hold benchmark (exact definition)
- Use yfinance **Adj Close** (total return incl. dividends/splits). At `POOL_START`, assign weight
  `1/M` to each of the M pool names; **buy and hold with no rebalance** (weights drift) → this is the
  honest "own the pool" line. NAV_t = mean over names of (AdjClose_t / AdjClose_start). Names that
  (despite the filter) have a gap are forward-filled within the gap only.
- **Secondary benchmark:** equal-weight **monthly-rebalanced** (reset to 1/M each month-end) — report
  both; the strategy net must beat at least the buy-and-hold line to be interesting.
- The strategy in Run 2 trades the same pool, long-flat AND dollar-neutral L/S variants (§5), net of
  the OHLC-derived spread costs.

### 1B.3 Run-2 isolation (do not break Run 1 or the quarantine test)
- Build Run-2's panel into a **separate directory** using the config env overrides (already
  supported): set `TACTIC_DATA_DIR=<repo>/data/deep_panel` before building features/labels/spreads so
  nothing is written to `data/curated/`. The yfinance raw pulls live under
  `data/diagnostic_quarantine/yfinance/`.
- `tests/test_quarantine_isolation.py` must still pass — Run 2 uses its own panel dir and never joins
  quarantined data into `data/curated/`. Add a test asserting `data/deep_panel/` is the only sink for
  deep data.

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

> The CPCV machinery below is written for **Run 1** (2016-2026, `N_GROUPS=8`). **Run 2** reuses the
> *identical* code with `N_GROUPS=10, K_TEST=2` over 2000-2026 and `--panel_dir=data/deep_panel/curated`
> (see §1B / Track B). Implement once, parametrize the span and group count.

### 3A. PRIMARY: Combinatorial Purged Cross-Validation (CPCV) — de Prado §15.1 / PLAN Phase 13
- **Blocks:** split the ordered trading-day axis (Run 1: 2016-01-04 → 2026-06-11, ~2625 sessions) into
  `N_GROUPS` contiguous, roughly equal groups (Run 1: 8 groups ≈ 1.3 yr each). Record each group's
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
- `validation/dm_test.py` — DONE. Diebold-Mariano: **Run 1** vs SPY and vs equal-weight 196-universe;
  **Run 2** vs the equal-weight-buy-and-hold-of-pool benchmark (NOT SPY).
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

Shared steps 0-2 above are done ONCE and serve both runs (the splitter/driver/portfolio/firewall
code is reused; only data span + benchmark differ). Then execute **TRACK A (Run 1)** fully, then
**TRACK B (Run 2)**. Do not interleave their data directories.

### TRACK A — RUN 1 (clean, 2016-2026, Alpaca)

**A3 — CPCV splitter + driver (§3A).**
- Implement `cpcv_paths(returns_index, n_groups=8, k_test=2, purge=2, embargo=5)` → the 28
  train/test group partitions with purge+embargo masks.
- Add CPCV driver `src/tactic/models/train_cpcv.py`: per fold build panel from `data/curated/`, apply
  fold masks, fit channel-stats on fold-train only, carve the nested embargoed val slice, train f4
  (seeds=3, ranking-augmented), predict held-out test groups; aggregate combined-OOS predictions;
  save per-fold net-return paths.
- **QA-A3:** unit-test (a) no test row in same-fold train, (b) purge+embargo gaps hold, (c) every
  session is OOS in exactly `C(7,1)=7` folds, (d) combined-OOS covers 2016→2026 incl all 2022.

**A4 — run on Vertex.**
- Extend `vertex_entry.py`/`train_on_vertex.py` with `--cv=cpcv --panel_dir=...` (+ `n_groups`,
  `k_test`, `patience`, `lambda_rank`). 28 folds × 3 seeds is heavy → one job looping folds on
  `n1-standard-16` or fan out. Estimate from a 1-fold timing; **stay online until SUCCEEDED**.
- **QA-A4:** artifacts present: combined `oos_predictions.parquet`, `paths.csv` (28 net Sharpes),
  `loss_history.csv` per fold/seed, `config.json` with exact CV spec. Output → `results/run1_clean_<id>/`.

**A5 — portfolio + metrics (§5).** Backtest combined-OOS with BOTH books (dollar-neutral L/S;
long-flat breadth) vs **SPY + equal-weight 196-universe**. Overall + per-year + per-regime (§2)
metrics. **QA-A5:** NAV identity; costs>0; coverage near nominal; IC per-date then averaged.

**A6 — firewall (§6).** DSR (honest-N from registry), PBO over the 28 paths, DM vs each benchmark,
path-Sharpe distribution. Report, do not halt. **QA-A6:** N=registry count; PBO∈[0,1]; 28 paths.

**A7 — report.** Write `results/run1_clean_<id>/RESULTS_RUN1.md` + graphs (equity vs SPY & EW-univ,
drawdown, per-year/per-regime tables, loss curves showing S1 margin, path-Sharpe histogram,
PBO/DSR/DM). Honest verdict.

### TRACK B — RUN 2 (deep, 2000-2026, yfinance, benchmark = own pool)

**B0 — deep data ingest (NEW `src/tactic/ingest/yfinance_deep.py`).**
- Set `TACTIC_DATA_DIR=<repo>/data/deep_panel` for ALL of Track B so nothing writes to `curated/`.
- Derive the pool per §1B.1 (continuous yfinance daily data POOL_START=2000-01-01→2026-06-10, <2%
  missing). Pull `yf.download(sym, start='2000-01-01', end='2026-06-11', auto_adjust=False)` for each;
  store raw under `data/diagnostic_quarantine/yfinance/`. Map to the repo's bar schema
  (symbol,date,o,h,l,c,v,vw≈c,adjustment): build an `all`-adjusted set from Adj Close ratio and a
  `raw` set; write `data/deep_panel/curated/prices_daily.parquet`. No auctions.
- Build spreads (OHLC-derived CS/AR/EDGE — works without SIP), labels (open→open, no auction), and
  features into `data/deep_panel/curated/`. Save pool list → `results/run2_deep_<id>/pool.csv`.
- **QA-B0:** pool filter asserts (§1B.1); 0 inf in features; date span 2000→2026-06-10; quarantine
  test still passes; `data/curated/` untouched (diff its mtime/hash before & after).

**B1 — sanity + model.** Reuse the S1 gate + ranking-augmented f4 unchanged (they are data-agnostic).
Quick chronological smoke (train 2000-2015 / val 2016-2017) to confirm S1 PASS on this pool before
spending Vertex. **QA-B1:** S1 margin ≥1%; q50 dispersion healthy.

**B2 — CPCV on the deep span.** Same `cpcv_paths` but `n_groups=10, k_test=2` → C(10,2)=45 folds over
2000-2026 (~2.6yr/group). Driver points `--panel_dir=data/deep_panel/curated`. Combined-OOS spans
2000→2026 incl dot-com bust, GFC, 2011, 2015-16, COVID, 2022.
- **QA-B2:** coverage = each session OOS in `C(9,1)=9` folds; purge/embargo hold; 2008 GFC present in OOS.

**B3 — run on Vertex.** Same job, `--panel_dir=data/deep_panel/curated`, output → `results/run2_deep_<id>/`.
Stay online until SUCCEEDED. **QA-B3:** artifacts present (as A4).

**B4 — portfolio + the pool benchmark (§1B.2).** Backtest combined-OOS with dollar-neutral L/S AND
long-flat books, net of OHLC-derived costs, on the deep pool. Implement the **equal-weight
buy-and-hold of the entire Run-2 pool** benchmark (Adj Close, no rebalance) + secondary monthly-
rebalanced EW. Strategy net must be compared to THESE, not SPY. Overall + per-year + per-regime.
- **QA-B4:** benchmark NAV uses Adj Close total return; pool size M logged; strategy & benchmark share
  the identical pool (survivorship cancels in the spread); NAV identity holds.

**B5 — firewall.** DSR (honest-N), PBO over 45 paths, **DM vs the equal-weight-pool benchmark**,
path-Sharpe distribution. **QA-B5:** 45 paths; PBO∈[0,1]; N=registry count.

**B6 — report.** Write `results/run2_deep_<id>/RESULTS_RUN2.md`: pool definition + `pool.csv`,
prominent **survivorship-bias caveat** at the top, equity vs **equal-weight-pool** (and monthly-EW),
drawdown, per-year/per-regime tables incl GFC, loss curves, path-Sharpe histogram, PBO/DSR/DM-vs-pool,
honest verdict ("beat / did not beat owning its own pool"). NEVER headline vs SPY; NEVER call
deployable.

**STEP FINAL — append `notes.md` (append-only) for both runs; commit + push.**
- **Security:** before any commit verify `.env` stays gitignored and NO `.env` / `data/curated` /
  `data/deep_panel` / `data/diagnostic_quarantine` / raw `*.parquet` are staged (only `results/**`
  artifacts + code). Commit + push to `https://github.com/JulianAttemptsCoding/cutie-QT-`.

---

## 8. Acceptance criteria (definition of done) — BOTH runs required

**Shared:**
1. S1 sanity gate implemented; **model PASSES S1 on val** (beats unconditional baseline ≥1%) and
   per-date q50 dispersion materially > prior 0.018. (QA2) — proves the collapse is fixed *without* OOS.
2. `cpcv_paths` implemented and **unit-tested** for leakage/purge/embargo/coverage.
3. DSR + PBO + DM computed and reported for each run (noted, not enforced).
4. `.env` never committed; no raw/curated/deep `*.parquet` staged; only `results/**` + code pushed.
5. **Honest verdict per run, stated either way.** A clean negative is a successful result
   (PLAN §0.3.10) — do not manufacture a win.

**RUN 1 (clean):**
6. Data verified to 2026-06-10+, 196 names, 0 inf. CPCV(8,2)=28 folds; 2022 confirmed OOS.
7. Backtested (L/S + long-flat) vs **SPY AND equal-weight 196-universe**, per-year + per-regime.
8. `results/run1_clean_<id>/RESULTS_RUN1.md` + graphs written.

**RUN 2 (deep, separate):**
9. `data/deep_panel/` built from yfinance 2000-2026; `data/curated/` provably untouched; quarantine
   test passes; `pool.csv` saved with the survivor pool + first dates.
10. CPCV(10,2)=45 folds; combined-OOS spans 2000→2026 incl 2008 GFC.
11. Backtested vs **equal-weight buy-and-hold of its OWN pool** (+ monthly-EW secondary); per-year +
    per-regime incl GFC. DM is vs the pool benchmark, not SPY.
12. `results/run2_deep_<id>/RESULTS_RUN2.md` with a top-of-file **survivorship-bias caveat**, never
    headlined vs SPY, never labeled deployable.

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
| `src/tactic/ingest/yfinance_deep.py` | NEW (RUN 2, §1B/B0) — yfinance 2000-2026 survivor pool → `data/deep_panel/` (separate). Pool filter + schema map. Never joins `curated/`. Needs `yfinance` (already installed). |
| `src/tactic/backtest/run.py` | EDIT — add equal-weight-buy-and-hold-of-pool benchmark (Run 2) + EW-universe benchmark (Run 1) + dollar-neutral L/S book + per-regime grouping |
| `.gitignore` | VERIFY `data/deep_panel/` + `data/diagnostic_quarantine/` ignored (deep raw data must not be committed) |
| `results/run1_clean_<id>/RESULTS_RUN1.md`, `results/run2_deep_<id>/RESULTS_RUN2.md` | NEW — one report per run |
| `vertex/vertex_entry.py`, `vertex/train_on_vertex.py` | EDIT — `--cv=cpcv --panel_dir=...` passthrough, fold fan-out/loop |
| `configs/v1.yaml` | (already has cpcv_groups/cpcv_test/purge/embargo) — add `lambda_rank` if desired |
| `tests/` | NEW — CPCV leakage/coverage tests; S1 constant-predictor test; deep-panel isolation test |
| `notes.md` | APPEND — running log (both runs) |

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
