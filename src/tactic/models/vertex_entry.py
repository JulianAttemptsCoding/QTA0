"""Vertex AI custom-job entry point for TACTIC-MoB training (sec10.5).

Runs inside a Vertex worker. Pulls the curated panel (features/prices/labels) from GCS into the
expected local CURATED path, runs the walk-forward training, renders OOS results, and pushes all
artifacts (loss history, predictions, metrics, plots, RESULTS.md) back to GCS.

Args are passed as `--args=key=value,...` by the submitter, e.g.
  data_uri=gs://.../repo_inputs/<ts> out_uri=gs://.../runs/<run_id> model=f4 epochs=25 seeds=5
GCS transfers use gsutil (present in Vertex prebuilt containers).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _gsutil(*args):
    subprocess.check_call(["gsutil", "-q", "-m", *args])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_uri", required=True, help="gs:// dir holding features/prices/labels parquet + v1.yaml")
    ap.add_argument("--out_uri", required=True, help="gs:// dir to receive run artifacts")
    ap.add_argument("--model", default="f4")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train_end", default="2020-12-31")
    ap.add_argument("--val_end", default="2021-12-31")
    ap.add_argument("--patience", type=int, default=5)
    # CPCV (revision_plan §3A) passthrough
    ap.add_argument("--cv", default="walk", choices=["walk", "cpcv"])
    ap.add_argument("--n_groups", type=int, default=8)
    ap.add_argument("--k_test", type=int, default=2)
    ap.add_argument("--purge", type=int, default=2)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--fold_start", type=int, default=0)
    ap.add_argument("--fold_end", type=int, default=-1)
    ap.add_argument("--lambda_rank", type=float, default=0.3)
    ap.add_argument("--demean", type=int, default=1)
    ap.add_argument("--k_book", type=int, default=35)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    a = ap.parse_args(argv)

    # Vertex installs the package with --no-deps; the pytorch-xla image already has
    # numpy/pandas/pyarrow/torch compiled together. Only add the missing pure deps and PIN
    # numpy<2 so nothing drags in NumPy 2.0 (which breaks the prebuilt torch/pyarrow ABI).
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "matplotlib", "numpy<2"])

    # point the package at writable container paths BEFORE importing config
    work = Path("/tmp/tactic"); cfg_dir = work / "configs"; data_dir = work / "data"
    (data_dir / "curated").mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TACTIC_DATA_DIR"] = str(data_dir)
    os.environ["TACTIC_CONFIG_DIR"] = str(cfg_dir)
    os.environ["TACTIC_REGISTRY_DIR"] = str(work / "registry")
    os.environ["TACTIC_REPORTS_DIR"] = str(work / "reports")
    _gsutil("cp", f"{a.data_uri.rstrip('/')}/v1.yaml", str(cfg_dir / "v1.yaml"))

    from .train import train
    from ..common.config import CURATED, ensure_dirs
    ensure_dirs()
    for name in ("features.parquet", "prices_daily.parquet", "labels.parquet", "spread_surface.parquet"):
        try:
            _gsutil("cp", f"{a.data_uri.rstrip('/')}/{name}", str(CURATED / name))
        except subprocess.CalledProcessError:
            if name == "spread_surface.parquet":
                continue  # optional
            raise

    if a.cv == "cpcv":
        from .train_cpcv import run_cpcv
        fold_end = None if a.fold_end < 0 else a.fold_end
        r = run_cpcv(model_kind=a.model, n_groups=a.n_groups, k_test=a.k_test,
                     purge=a.purge, embargo=a.embargo, epochs=a.epochs, seeds=a.seeds,
                     patience=a.patience, lambda_rank=a.lambda_rank, demean=bool(a.demean),
                     fold_start=a.fold_start, fold_end=fold_end, k_book=a.k_book,
                     run_id="vertex_run")
        # upload fold artifacts directly (flatten into out_uri so shards co-locate)
        _gsutil("cp", "-r", str(r["run_dir"]), a.out_uri.rstrip("/") + "/")
        print(f"[vertex_entry] cpcv folds {a.fold_start}..{fold_end} done -> {a.out_uri}")
        return

    r = train(model_kind=a.model, epochs=a.epochs, seeds=a.seeds,
              train_end=a.train_end, val_end=a.val_end, run_id="vertex_run",
              patience=a.patience, lambda_rank=a.lambda_rank, demean=bool(a.demean),
              lr=a.lr, weight_decay=a.weight_decay)
    try:
        from ..reports.results import render_run
        render_run(r["run_id"])
    except Exception as e:  # plotting deps optional in-container; predictions still saved
        print(f"[vertex_entry] results render skipped: {e}", file=sys.stderr)

    _gsutil("cp", "-r", str(r["run_dir"]), a.out_uri.rstrip("/") + "/")
    print(f"[vertex_entry] done; artifacts -> {a.out_uri}")


if __name__ == "__main__":
    main()
