"""Submit a full TACTIC-MoB training run to Vertex AI custom-jobs.

Uploads the curated panel (features/prices/labels/spread_surface) + v1.yaml to GCS, builds and
uploads the package sdist, then launches a custom job whose python-module is
`tactic.models.vertex_entry`. Artifacts land in gs://<bucket>/runs/<run_id>/.

    python vertex/train_on_vertex.py --model f4 --epochs 25 --seeds 5
    python vertex/train_on_vertex.py --model f3 --epochs 25 --seeds 5 --smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactic.common.config import VertexConfig, load_config, load_dotenv  # noqa: E402

CURATED = ROOT / "data" / "curated"
DATA_FILES = ["features.parquet", "prices_daily.parquet", "labels.parquet", "spread_surface.parquet"]


def _run(cmd):
    # shell=True so Windows resolves gcloud/gsutil .cmd shims; quote args with spaces.
    line = " ".join(f'"{c}"' if (" " in c or "\\" in c) else c for c in cmd)
    print("  $", line)
    subprocess.check_call(line, shell=True)


def _cp(src, dst):
    _run(["gcloud", "storage", "cp", str(src), dst])


def build_sdist() -> Path:
    dist = ROOT / "vertex" / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "build"])
    subprocess.check_call([sys.executable, "-m", "build", "--sdist", "--outdir", str(dist)], cwd=str(ROOT))
    return sorted(dist.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f4", choices=["f3", "f4"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="n1-standard-4 instead of n1-standard-16")
    ap.add_argument("--gpu", action="store_true", help="use the pytorch-xla image (default; torch present)")
    ap.add_argument("--train_end", default="2020-12-31")
    ap.add_argument("--val_end", default="2021-12-31")
    ap.add_argument("--patience", type=int, default=5)
    # CPCV passthrough (revision_plan §3A)
    ap.add_argument("--cv", default="walk", choices=["walk", "cpcv"])
    ap.add_argument("--n_groups", type=int, default=8)
    ap.add_argument("--k_test", type=int, default=2)
    ap.add_argument("--fold_start", type=int, default=0)
    ap.add_argument("--fold_end", type=int, default=-1)
    ap.add_argument("--lambda_rank", type=float, default=0.3)
    ap.add_argument("--demean", type=int, default=1)
    ap.add_argument("--k_book", type=int, default=35)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--panel_dir", default=None,
                    help="local panel dir to upload (default data/curated; use data/deep_panel/curated for Run 2)")
    ap.add_argument("--run_id", default=None, help="shared run id (shards co-locate under runs/<run_id>)")
    ap.add_argument("--data_uri", default=None, help="reuse an already-uploaded panel uri (skip upload)")
    a = ap.parse_args()

    load_dotenv()
    cfg = load_config()
    v = VertexConfig.from_config(cfg)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = a.run_id or f"{a.model}_vertex_{ts}"
    panel_dir = Path(a.panel_dir) if a.panel_dir else CURATED
    data_uri = a.data_uri or f"{v.bucket}/repo_inputs/{run_id}"
    job_tag = f"{run_id}_f{a.fold_start}-{a.fold_end}" if a.cv == "cpcv" else run_id
    out_uri = f"{v.bucket}/runs/{run_id}"

    if a.data_uri:
        print(f"[1/4] reuse uploaded panel -> {data_uri}")
    else:
        print(f"[1/4] upload panel ({panel_dir}) + config -> {data_uri}")
        for f in DATA_FILES:
            p = panel_dir / f
            if p.exists():
                _cp(p, f"{data_uri}/{f}")
            elif f != "spread_surface.parquet":
                raise FileNotFoundError(f"missing panel file {p}; run the data phases first")
        _cp(ROOT / "configs" / "v1.yaml", f"{data_uri}/v1.yaml")

    print("[2/4] build + upload sdist")
    pkg = build_sdist()
    pkg_uri = f"{v.bucket}/packages/{pkg.name}"
    _cp(pkg, pkg_uri)

    print("[3/4] submit custom job")
    # pytorch-xla image ships torch; CPU n1-standard-16 is ample for this model size.
    image = v.image_gpu
    machine = v.machine_smoke if a.smoke else v.machine_full
    args = (f"--data_uri={data_uri},--out_uri={out_uri},--model={a.model},"
            f"--epochs={a.epochs},--seeds={a.seeds},"
            f"--train_end={a.train_end},--val_end={a.val_end},--patience={a.patience},"
            f"--cv={a.cv},--lambda_rank={a.lambda_rank},--demean={a.demean},"
            f"--lr={a.lr},--weight_decay={a.weight_decay}")
    if a.cv == "cpcv":
        args += (f",--n_groups={a.n_groups},--k_test={a.k_test},"
                 f"--fold_start={a.fold_start},--fold_end={a.fold_end},--k_book={a.k_book}")
    worker = (f"machine-type={machine},replica-count=1,executor-image-uri={image},"
              f"python-module=tactic.models.vertex_entry")
    cmd = ["gcloud", "ai", "custom-jobs", "create",
           f"--region={v.region}", f"--project={v.project_id}",
           f"--display-name=tactic-{job_tag}",
           f"--python-package-uris={pkg_uri}",
           f"--worker-pool-spec={worker}",
           f"--args={args}"]
    _run(cmd)
    print(f"[4/4] submitted. job_tag={job_tag} run_id={run_id}\n  artifacts will land at: {out_uri}")
    print(f"  poll: gcloud ai custom-jobs list --region={v.region} --project={v.project_id} "
          f"--filter='displayName:tactic-{run_id}'")


if __name__ == "__main__":
    main()
