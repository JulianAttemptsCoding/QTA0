# TACTIC-MoB v4 — Wide-universe OOS report (2022-01 → 2026-06)

f4 (hybrid cross-sectional attention) trained on **Vertex AI** (`n1-standard-16`, pytorch-xla),
run `f4_vertex_20260612_230413`. Universe **196 liquid US names + ETFs**; history from 2016.
Out-of-sample = **2022-01-12 → 2026-06-09** (1105 sessions, ~4.4 yr, never trained on).
Benchmark = **SPY buy-and-hold**, identical window. Gates are **computed and noted only**
(not enforced) per request.

## Split (clean, walk-forward, purge 2 + embargo 5)
| | dates | date-batches |
|---|---|---|
| Train | 2016-07-26 → 2020-12-31 | 1118 |
| Val   | 2021-01-13 → 2021-12-31 | 245 |
| **OOS/Test** | **2022-01-12 → 2026-06-09** | **1105** |
Entities/day 192–196 (median 195). Channel standardization fit on TRAIN only. 60 epochs ×
3 seeds, early-stop patience 12 on val pinball, seeds quantile-averaged.

## Headline (spec decision band k_buy=20 / k_hold=60, equal-weight, next-open fills, net of costs)
| Metric | TACTIC-MoB (net) | SPY |
|---|---|---|
| Total return | +27.2% | **+63.6%** |
| CAGR | +5.6% | **+11.9%** |
| Ann. volatility | **13.9%** | 17.9% |
| Sharpe | 0.47 | **0.72** |
| Sortino | 0.61 | **0.97** |
| Max drawdown | **−20.1%** | −23.5% |
| Daily win rate | 53.1% | 53.5% |
| Beta / ann. alpha | 0.62 / **−1.5%** | — |
| Avg 1-sided turnover | 3.0%/day | — |

**Verdict: does NOT beat buy-and-hold over this OOS window.** Lower volatility and shallower
drawdown (defensive), but CAGR/Sharpe trail SPY; annual alpha −1.5%.

## Annual breakdown
| Year | Strat ret | SPY ret | Strat Sharpe | SPY Sharpe | Strat maxDD | SPY maxDD |
|---|---|---|---|---|---|---|
| 2022 | **−7.2%** | −17.5% | −0.35 | −0.70 | **−20.1%** | −23.5% |
| 2023 | +4.1% | **+24.6%** | 0.40 | 1.71 | −8.0% | −9.6% |
| 2024 | +21.8% | **+26.5%** | 1.85 | 1.93 | **−6.2%** | −9.1% |
| 2025 | +3.1% | **+18.2%** | 0.28 | 0.93 | −16.4% | −19.8% |
| 2026* | +4.8% | +6.4% | 1.15 | 1.16 | −4.9% | −8.1% |
*2026 partial (109 sessions). The strategy **won only the 2022 bear** (cut the drawdown roughly
in half) and trailed in every mega-cap-led bull year — an equal-weight top-K subset cannot keep
up with a cap-weighted index dominated by the largest names in 2023–2025.

## Concentration sensitivity (why the naive k=5 book blew up)
| k_buy / k_hold | CAGR | Sharpe | maxDD |
|---|---|---|---|
| 5 / 8   | −10.7% | −0.56 | −42.4% |
| 10 / 20 | −2.2% | −0.07 | −22.8% |
| **20 / 60** | **+5.6%** | **0.47** | −20.1% |
| 30 / 80 | +6.2% | 0.51 | −19.8% |
The earlier single-year "headline" used k_buy=5; on a 196-name universe across a 4.4-yr window
including 2022 that concentration is fatal (−42% DD). Breadth in the book (k≥20) is required even
to get a positive return. None of the bands beat SPY.

## Predictor statistics (OOS, 216,280 obs)
| Statistic | Value |
|---|---|
| Summed pinball (std scale) | 1.668 |
| **Rank IC** (mean daily Spearman) | **+0.0149** (IR 0.072) |
| Directional hit rate | 51.8% |
| 80% interval coverage | 76.0% |
| 90% interval coverage | 87.3% |
Small but **positive** cross-sectional IC — real, weak signal — yet not enough net edge to beat
a strong-trend index long-only over this window.

## Train/val curves — overfit divergence (as requested)
`loss_curve.png` / `loss_history.csv`: per seed, **train pinball falls monotonically** (seed1338
1.697→1.584) while **val pinball bottoms ~epoch 4 (≈1.605) then rises/oscillates** — the classic
overfit onset. Early-stop (patience 12) kept the best-val checkpoint for OOS predictions.

## Gate notes (computed, NOT enforced)
- **G1 predictability:** IC 0.0149 > ic_min 0.005 → present.
- **G2 turnover:** 3.0%/day one-sided < 0.50 budget → OK.
- **G3 beat benchmark net:** strategy underperforms; DM on daily P&L (loss=−ret) stat +1.27,
  **p=0.204** → underperformance not even statistically significant (within noise). FAIL-to-beat.
- **G8 DSR:** daily SR 0.0294; DSR(N=5)=0.27, (N=10)=0.13, (N=20)=0.06 — all ≪ 0.95 → FAIL.

## Honest conclusion
Wider breadth (196 vs 83) and a 4.4-yr OOS that includes the 2022 bear give the **honest** read:
the model has a small positive cross-sectional IC and is **mildly defensive** (best in the down
year, lower vol/DD throughout), but it **does not beat SPY buy-and-hold** over 2022–2026 at any
sane book concentration, and the gap is within statistical noise. The flashy one-year/15-name and
one-year/83-name "wins" do not survive a longer, wider, bear-inclusive test. Valid negative result.

Figures: `equity_k20.png`, `drawdown_k20.png`, `by_year_k20.png`, `loss_curve.png`,
`calibration.png`.
