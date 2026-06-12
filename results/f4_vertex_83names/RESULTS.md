# OOS Results — run `vertex_run`

Out-of-sample window: **2023-01-12 → 2023-12-27** (241 sessions). Long-only/long-flat top-K cross-sectional, fills at next open, net of estimated costs.

## Performance vs SPY buy-and-hold (out-of-sample)

| Metric | TACTIC-MoB (net) | SPY B&H |
|---|---|---|
| Total return | 39.79% | 22.90% |
| CAGR | 41.94% | 24.06% |
| Ann. volatility | 20.46% | 13.38% |
| Sharpe | 1.81 | 1.68 |
| Sortino | 2.90 | 2.75 |
| Max drawdown | -9.19% | -9.59% |
| Win rate (daily) | 56.43% | 51.45% |

- Beta vs SPY: **1.25**, annualized alpha: **9.07%**
- Avg one-sided turnover/day: 2.86%; avg cost: 1.46 bps/day; avg names held: 5.0

## Predictor statistics (out-of-sample)

- Summed pinball loss (standardized scale): **1.6698**
- Rank IC (mean daily Spearman, q50 vs realized): **0.0157** (IR 0.07)
- Directional hit rate: 52.39%
- 80% interval coverage: 76.30% (target 80%); 90% coverage: 87.75% (target 90%)
- OOS observations: 20003

## Training

- Final val pinball: 1.66439; epochs run: 13; seeds: 3

## Figures

![equity](equity_curve.png)
![drawdown](drawdown.png)
![loss](loss_curve.png)
![calibration](calibration.png)
