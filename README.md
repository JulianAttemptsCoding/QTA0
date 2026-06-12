# TACTIC-MoB v4

**Transaction-cost-Aware, Component-decomposed, Throttled Implementation of Calibrated
Mixture-of-Betting experts** — a daily-frequency, long-only/long-flat, cross-sectional US
equity strategy with conformally-calibrated forecasts, e-value monitoring, alpha-decay-matched
partial-adjustment trading under a hard turnover budget, and per-asset/day cost estimation.

The full build specification is in [`PLAN.md`](PLAN.md). The null hypothesis (no deployable
edge) is presumed true until every kill gate in §17 passes. **A halted build is a successful
run** (§0.3.10).

> Scope of this milestone: empty repo → validated backtest with all anti-overfitting gates
> evaluated. Live/paper trading is out of scope (Appendix F).

## Status

| Layer | State |
|-------|-------|
| Phase 0 — scaffold, registry, gates, contracts, IO, calendar | **implemented + tested** |
| Phase 1 — ingestion (Alpaca bars/auctions/assets/CA, French factors, Wikipedia, EDGAR), Day-1 audits (G0a) | **implemented** (live-verified) |
| Phase 2 — universe (PIT membership), delistings/terminal returns (G0b) | **implemented + tested** |
| Phase 3 — spread surface (Corwin-Schultz, Abdi-Ranaldo, EDGE blend) + cost model/break-even | **implemented + tested** |
| Phase 4 — labels (A.1 components, z_top, EWMA σ, vol-standardized y) | **implemented + tested** |
| Phase 5 — diagnostics (variance ratio, IC decay, tug-of-war) + gate G1/G1b | **implemented + tested** |
| Math libs — MODWT, quantile/pinball, DSR, Diebold-Mariano, NAV accounting | **implemented + tested** |
| Phase 4 features tensor, Phases 6–14 (baselines, experts/training, aggregation, decision, backtest, stress, firewall, reports) | **scaffolded** (typed stubs; halt at boundary) |
| Vertex AI submission (custom-jobs, Vizier spec, packaging) | **implemented** |

`make all` runs the §18 order and stops cleanly at the first scaffold boundary or kill gate.
End-to-end verified on a 15-symbol × 5-year real slice: ingest → costs → labels (99.9% auction
coverage) → universe → diagnostics (G1 passes, max |IC| ≈ 0.025).

> Calibration note: the spread blend is the spec'd median of three *daily-OHLC* estimators
> (PLAN.md §5.1); Corwin-Schultz/Abdi-Ranaldo run high on liquid daily bars, so blended
> medians are conservative. Tighten with minute bars or an EDGE-weighted blend when wiring
> the live loop.

## Quickstart

```bash
# 1. install (editable)
uv pip install -e ".[dev]"        # or: pip install -e ".[dev]"

# 2. credentials
cp .env.example .env              # fill APCA_API_KEY_ID / APCA_API_SECRET_KEY / SEC_USER_AGENT

# 3. tests (green on the scaffold)
python make.py test               # Windows; or `make test` on Linux/macOS

# 4. run the pipeline (smoke symbol set, fast)
python make.py setup
python make.py data-factors
python make.py data-alpaca        # add --full for the whole candidate universe (hours)
python make.py day1-audits
python make.py all                # full §18 order; halts at the scaffold boundary
```

On Linux/macOS with GNU make installed, use `make <target>` instead of `python make.py <target>`.

## Heavy compute → Vertex AI

GPU/large-CPU work runs on Vertex AI; light work runs locally. See [`vertex/README.md`](vertex/README.md).

```bash
python vertex/submit.py --module tactic.models.hpo_vertex --gpu --args expert=f4 refit=2021
```

## Layout

```
configs/            v1.yaml (Appendix B + gate table + vertex)
src/tactic/
  common/           config, hashing, registry (honest-N ledger), gates, contracts, io, calendar
  ingest/           alpaca_*, factors_french, wiki_constituents, edgar_xcheck, day1_audits
  universe/ costs/ labels/ features/ diagnostics/ regime/ risk/
  models/ (experts/) uq/ agg/ portfolio/ backtest/ validation/ stress/ reports/
  cli.py            pipeline driver (make targets)
tests/              contract + unit tests (pytest)
vertex/             submit.py, vizier_spec.yaml, Dockerfile
data/ registry/ reports/   (generated; gitignored)
```

## Non-negotiables (enforced in code)

Temporal PIT contracts, fit-scope discipline, SIP-only feed, no per-ticker models, single
pinball objective, post-hoc-only calibration, filesystem-locked 18-month holdout, automatic
honest-N trial ledger, populated terminal returns, and halting kill gates. See `PLAN.md §0.3`.
