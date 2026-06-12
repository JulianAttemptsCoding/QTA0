# 3-Year OOS Results — f4 (Vertex), 2023-01-12 → 2026-01-30 (765 sessions)

Model frozen: trained on data ≤2021, validated 2022; **2023–2026 entirely out-of-sample.**
Vertex job 668174298531233792. Universe 83 liquid S&P-500 names. Net of estimated costs.

## Overall (whole 3-year period)
| | TACTIC-MoB (net) | SPY B&H |
|---|---|---|
| **Total return** | **92.1%** | 84.0% |

## Annualized
| Metric | TACTIC-MoB | SPY |
|---|---|---|
| CAGR | **24.0%** | 22.2% |
| Ann. volatility | 21.3% | 15.7% |
| Sharpe | 1.12 | **1.36** |
| Sortino | 1.61 | **1.77** |
| Max drawdown | -26.3% | -19.8% |
| Win rate (daily) | 54.0% | 55.3% |
| Beta vs SPY | 0.95 | — |
| Alpha (annualized) | **+3.5%** | — |
| Avg turnover/day | 6.3% | — |

## Per calendar year (return / Sharpe / maxDD)
| Year | Strat ret | SPY ret | Strat Sharpe | SPY Sharpe | Strat maxDD | SPY maxDD |
|---|---|---|---|---|---|---|
| 2023 | 13.7% | 21.3% | 0.80 | 1.57 | -11.0% | -9.6% |
| 2024 | 36.0% | 26.5% | 1.63 | 1.93 | -11.9% | -9.1% |
| 2025 | 19.4% | 18.2% | 0.85 | 0.93 | -26.3% | -19.8% |
| 2026* | 4.0% | 1.4% | 3.31 | 1.64 | -2.8% | -2.3% |

\*2026 = 20 sessions only.

## Predictor (OOS, 63,495 obs)
Rank-IC 0.0140 (IR 0.071), hit 52.5%, pinball 1.684, interval coverage 74.8%/86.1%.

## Honest read
- Higher **cumulative + CAGR** than SPY (92% vs 84%; 24.0% vs 22.2%), but **lower risk-adjusted**
  return (Sharpe 1.12 vs 1.36) — the concentrated 5-name long book runs higher vol (21% vs 16%)
  and a deeper 2025 drawdown (-26% vs -20%).
- Beats SPY on raw return in 2024/2025/2026, **trails in 2023**. Note: an earlier 1-year run
  (model retrained, same split) printed +39.8% for 2023 vs +13.7% here — **high run-to-run
  variance** (seed + early-stop), i.e. the edge is real-but-fragile, not a stable Sharpe>SPY.
- Consistent positive rank-IC (~0.014) and calibrated intervals across 3 years.
- Verdict: modest positive alpha (+3.5%/yr), beta ~1; outperforms on return, not on Sharpe.
  Deployability still needs the full PLAN firewall (CPCV/DSR/SPA + RC-Kelly sizing to cut vol).

## Deflated Sharpe Ratio (gate G8) — FAILS

DSR (de Prado, Appendix A.9) deflates the observed Sharpe for (a) number of trials tried,
(b) non-normal returns, (c) sample length. Computed on the 3-year daily net returns
(daily SR 0.070 ≈ 1.12 ann.; skew 0.21, **kurtosis 8.35**, T=765; trial-Sharpe dispersion
estimated from the variants actually evaluated).

| N (trials) | SR0 (daily) | DSR | G8 (>0.95)? |
|---|---|---|---|
| 5 | 0.048 | 0.73 | FAIL |
| 10 | 0.064 | 0.57 | FAIL |
| 20 | 0.077 | 0.43 | FAIL |
| 50 | 0.092 | 0.28 | FAIL |
| 100 | 0.102 | 0.19 | FAIL |

**Verdict: DSR < 0.95 at every plausible trial count → gate G8 FAILS.** After accounting for
the ~7–10 model/universe/window variants tried and the fat-tailed daily returns, the OOS
Sharpe (1.12) is statistically indistinguishable from the best a lucky search would produce
(SR0 ≈ 1.01 ann. at N=10). The strategy's raw Sharpe is already below SPY's (1.36), so there is
no evidence of risk-adjusted skill. Per PLAN §0.3.10 this is a valid **negative result** — the
anti-overfitting firewall working as designed: do not deploy.
