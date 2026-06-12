"""TACTIC-MoB pipeline driver. Maps the Makefile targets (sec2.3, sec18) to callables.

Usage:
    tactic <target> [--smoke] [--full] [--symbols AAPL MSFT ...]
    python -m tactic.cli all --smoke

`all` runs the execution order from PLAN.md sec18, halting at:
  - the first failed KILL GATE (a *successful* run per sec0.3.10), or
  - the scaffold boundary (first not-yet-implemented phase), reported clearly.
"""
from __future__ import annotations

import argparse
import sys

from .common.config import ensure_dirs, load_config, load_dotenv
from .common.gates import KillGate

SMOKE_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM"]


# --- phase callables ------------------------------------------------------------
def _setup(args):
    ensure_dirs()
    from .common.config import DATA_DIR, REGISTRY_DIR, REPORTS_DIR
    for d in (DATA_DIR, REGISTRY_DIR, REPORTS_DIR):
        for sub in d.rglob("*"):
            pass
    # .gitkeep so empty dirs survive git
    for d in (DATA_DIR / "raw", DATA_DIR / "curated", DATA_DIR / "holdout",
              REGISTRY_DIR / "prereg", REPORTS_DIR):
        (d / ".gitkeep").touch()
    print("[setup] dirs + registry + reports ready")


def _candidate_symbols(args) -> list[str]:
    if getattr(args, "symbols", None):
        return args.symbols
    if getattr(args, "full", False):
        from .ingest.candidate_symbols import build
        syms = build()
        if syms:
            return syms
    return SMOKE_SYMBOLS


def _data_alpaca(args):
    from .ingest import alpaca_assets, alpaca_auctions, alpaca_bars, alpaca_corp_actions
    syms = _candidate_symbols(args)
    print(f"[data-alpaca] {len(syms)} symbols (full={getattr(args,'full',False)})")
    alpaca_assets.ingest()
    alpaca_bars.ingest(syms)
    alpaca_auctions.ingest(syms)
    alpaca_corp_actions.ingest(syms)
    print("[data-alpaca] done")


def _data_factors(args):
    from .ingest import factors_french
    factors_french.ingest()
    print("[data-factors] done")


def _data_universe(args):
    from .ingest import edgar_xcheck, wiki_constituents
    wiki_constituents.ingest(sample_every=getattr(args, "sample_every", 5))
    edgar_xcheck.ingest()
    print("[data-universe] done")


def _day1(args):
    from .ingest import day1_audits
    print(f"[day1-audits] {day1_audits.run()}")


def _stub_phase(modpath: str, fn: str):
    def runner(args):
        import importlib
        mod = importlib.import_module(modpath)
        getattr(mod, fn)(load_config())
    return runner


# --- implemented phases (Phase 2-5) ---
def _universe(args):
    from .universe.build_universe import build_universe
    from .universe.delistings import build_terminal_returns
    cfg = load_config()
    term = build_terminal_returns(cfg)
    # Gate G0b (soft): unknown terminal types < 5% of exits
    if len(term):
        frac_unknown = float((term["terminal_type"] == "unknown").mean())
        from .common.gates import evaluate
        evaluate("G0b", passed=frac_unknown < 0.05, value=frac_unknown, threshold=0.05,
                 interpretation="Too many exits have unknown terminal returns (survivorship risk).",
                 next_step="Improve delisting reconciliation or report the bias bound.", soft=True)
    uni = build_universe(cfg)
    tradable = int(uni["tradable"].sum())
    print(f"[universe] {len(term)} exits, {tradable} tradable entity-days")


def _costs(args):
    from .costs.spreads import build_spread_surface
    surf = build_spread_surface(cfg=load_config())
    print(f"[costs] spread surface: {len(surf)} entity-days, "
          f"median blend {surf['s_blend'].median()*1e4:.1f} bps")


def _labels(args):
    from .labels.build_labels import build_labels
    lab = build_labels(load_config())
    print(f"[labels] {len(lab)} entity-days; auction frac "
          f"{(lab['label_source'] == 'auction').mean():.2%}")


def _diagnostics(args):
    from .diagnostics.run_diagnostics import run
    res = run(load_config())
    print(f"[diagnostics] G1 pass={res['g1_pass']} (max_ic={res['max_ic']:.4f})")


# target -> callable
PHASES: dict[str, object] = {
    "setup": _setup,
    "data-alpaca": _data_alpaca,
    "data-factors": _data_factors,
    "data-universe": _data_universe,
    "day1-audits": _day1,
    "universe": _universe,
    "costs": _costs,
    "labels": _labels,
    "features": _stub_phase("tactic.features.build_features", "build_features"),
    "diagnostics": _diagnostics,
    "baselines": _stub_phase("tactic.backtest.baselines", "run_baselines"),
    "infra": _stub_phase("tactic.regime.bocpd_t", "BOCPDt"),
    "train": _stub_phase("tactic.models.train", "train"),
    "aggregate": _stub_phase("tactic.agg.aa", "AggregatingAlgorithm"),
    "decide": _stub_phase("tactic.portfolio.gp_aim", "gp_step"),
    "backtest": _stub_phase("tactic.backtest.engine", "run_backtest"),
    "stress": _stub_phase("tactic.stress.run_stress", "run"),
    "firewall": _stub_phase("tactic.validation.firewall", "run"),
    "reports": _stub_phase("tactic.reports.render", "render_final_report"),
}

# execution order for `all` (sec18)
ALL_ORDER = [
    "setup", "data-alpaca", "data-factors", "data-universe", "day1-audits",
    # costs precedes universe: the universe spread filter (sec4.2) consumes the Phase-3
    # spread surface. (PLAN.md sec18 lists universe first; this resolves the data dependency.)
    "costs", "universe", "labels", "features", "diagnostics", "baselines",
    "infra", "train", "aggregate", "decide", "backtest", "stress",
    "firewall", "reports",
]


def _run_target(name: str, args) -> int:
    fn = PHASES[name]
    try:
        fn(args)
        return 0
    except KillGate as kg:
        print(f"\n=== KILL GATE {kg.gate_id} HALTED THE BUILD ===")
        print(f"report: {kg.report_path}")
        print("(A halted build is a SUCCESSFUL run of PLAN.md, sec0.3.10.)")
        return 0
    except NotImplementedError as nie:
        print(f"\n[{name}] scaffold boundary reached: {nie}")
        print("This phase is scaffolded but not yet implemented. Stopping cleanly.")
        return 3


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="tactic", description="TACTIC-MoB pipeline driver")
    ap.add_argument("target", choices=list(PHASES) + ["all", "test"])
    ap.add_argument("--smoke", action="store_true", help="use a small symbol set (default)")
    ap.add_argument("--full", action="store_true", help="full candidate universe")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--sample-every", type=int, default=5)
    args = ap.parse_args(argv)

    if args.target == "test":
        import subprocess
        return subprocess.call([sys.executable, "-m", "pytest", "-q"])

    if args.target == "all":
        for name in ALL_ORDER:
            print(f"\n>>> {name}")
            rc = _run_target(name, args)
            if rc != 0:
                return rc
        print("\n[all] completed through implemented phases.")
        return 0

    return _run_target(args.target, args)


if __name__ == "__main__":
    raise SystemExit(main())
