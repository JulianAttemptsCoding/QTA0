"""RUN 2 deep-panel builder (revision_plan B0). Sets TACTIC_DATA_DIR=data/deep_panel so NOTHING
writes to data/curated. Builds prices->spreads->labels->features into the separate panel, with QA.

    TACTIC_DATA_DIR=<repo>/data/deep_panel python scripts/build_deep_panel.py
(the script sets the env itself if unset)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEEP = ROOT / "data" / "deep_panel"
os.environ.setdefault("TACTIC_DATA_DIR", str(DEEP))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tactic.common.config import load_dotenv, CURATED, ensure_dirs  # noqa: E402

RUN2_ID = "run2_deep_20260613"
POOL_CSV = ROOT / "results" / RUN2_ID / "pool.csv"


def main():
    load_dotenv()
    ensure_dirs()
    assert "deep_panel" in str(CURATED), f"CURATED must point to deep_panel, got {CURATED}"
    print(f"[deep] CURATED={CURATED}")

    from tactic.ingest.yfinance_deep import ingest_deep
    print("[deep] 1/4 ingest yfinance survivor pool ...")
    r = ingest_deep(pool_csv=POOL_CSV)
    pool = r["pool"]
    print(f"[deep] pool={len(pool)} dropped={len(r['dropped'])} ref={r['ref_symbol']}")

    # QA(pool) §1B.1
    n_sessions = r["n_sessions"]
    for p in pool:
        assert p["coverage"] >= 0.98, f"{p['symbol']} coverage {p['coverage']} < 0.98"
    assert len(pool) >= 50, f"pool too small: {len(pool)}"
    print(f"[deep] QA pool OK (>=0.98 coverage, n_sessions={n_sessions})")

    from tactic.costs.spreads import build_spread_surface
    from tactic.labels.build_labels import build_labels
    from tactic.features.build_features import build_features
    print("[deep] 2/4 spreads ...")
    build_spread_surface()
    print("[deep] 3/4 labels ...")
    build_labels()
    print("[deep] 4/4 features ...")
    feats = build_features()

    # QA(features) §B0
    num = feats.select_dtypes("number").to_numpy()
    n_inf = int(np.isinf(num).sum())
    dmin, dmax = feats["date"].min(), feats["date"].max()
    print(f"[deep] features: ents={feats['entity'].nunique()} dates={dmin}..{dmax} inf={n_inf}")
    assert n_inf == 0, f"features have {n_inf} inf"
    assert pd.to_datetime(dmax).year >= 2026 and pd.to_datetime(dmin).year <= 2001, \
        f"deep span wrong: {dmin}..{dmax}"
    print("[deep] DONE. pool.csv ->", POOL_CSV)


if __name__ == "__main__":
    main()
