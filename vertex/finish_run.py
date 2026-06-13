"""Download CPCV shard artifacts from GCS, aggregate combined-OOS, build the report (locally).

    python vertex/finish_run.py --run_id <id> --label RUN1 [--deep] [--panel_dir data/curated]
                                [--pool_csv results/run2_deep_<id>/pool.csv]

Heavy training already ran on Vertex; this step is light (portfolio + firewall + plots)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactic.common.config import VertexConfig, load_config, load_dotenv  # noqa: E402


def _run(cmd):
    print("  $", cmd); subprocess.check_call(cmd, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--label", default="RUN1")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--panel_dir", default="data/curated")
    ap.add_argument("--pool_csv", default=None)
    ap.add_argument("--out", default=None, help="local results dir (default results/<run_id>)")
    ap.add_argument("--k_book", type=int, default=35)
    a = ap.parse_args()
    load_dotenv()
    v = VertexConfig.from_config(load_config())

    out = Path(a.out) if a.out else ROOT / "results" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    src = f"{v.bucket}/runs/{a.run_id}/vertex_run/*"
    print(f"[1/3] download {src} -> {out}")
    _run(f'gcloud storage cp "{src}" "{out}/"')

    # deep run reads its panel via TACTIC_DATA_DIR (spread surface lookups in _spread)
    if a.deep:
        os.environ["TACTIC_DATA_DIR"] = str((ROOT / "data" / "deep_panel").resolve())

    from tactic.models.train_cpcv import aggregate_combined
    from tactic.reports.oos_report import build_report
    print("[2/3] aggregate combined OOS")
    agg = aggregate_combined(out)
    print("   ", agg)

    print("[3/3] build report")
    prices_path = ROOT / a.panel_dir / "prices_daily.parquet"
    rep = build_report(out, prices_path, run_label=a.label, deep=a.deep,
                       pool_csv=a.pool_csv, k_book=a.k_book)
    fw = rep["firewall"]
    print(f"done. DSR={fw['dsr'].get('dsr')} PBO={fw['pbo'].get('pbo')} "
          f"pathSharpe mean={fw['path_sharpe_mean']:.3f}")


if __name__ == "__main__":
    main()
