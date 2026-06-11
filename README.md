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
| Phase 1 — ingestion (Alpaca bars/auctions/assets/CA, French factors, Wikipedia, EDGAR), Day-1 audits | **implemented** |
| Costs (cost model, break-even), labels math, MODWT, quantile/pinball, DSR, DM, accounting | **implemented + tested** |
| Phases 2–14 — universe, spreads surface, features, diagnostics, experts/training, aggregation, decision, backtest, stress, firewall, reports | **scaffolded** (typed stubs; raise at the boundary) |
| Vertex AI submission (custom-jobs, Vizier spec, packaging) | **implemented** |

`make all` runs the §18 order and stops cleanly at the first scaffold boundary or kill gate.

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
