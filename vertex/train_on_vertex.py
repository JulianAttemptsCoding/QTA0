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
    a = ap.parse_args()

    load_dotenv()
    cfg = load_config()
    v = VertexConfig.from_config(cfg)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{a.model}_vertex_{ts}"
    data_uri = f"{v.bucket}/repo_inputs/{ts}"
    out_uri = f"{v.bucket}/runs/{run_id}"

    print(f"[1/4] upload curated panel + config -> {data_uri}")
    for f in DATA_FILES:
        p = CURATED / f
        if p.exists():
            _cp(p, f"{data_uri}/{f}")
        elif f != "spread_surface.parquet":
            raise FileNotFoundError(f"missing curated file {p}; run the data phases first")
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
            f"--epochs={a.epochs},--seeds={a.seeds}")
    worker = (f"machine-type={machine},replica-count=1,executor-image-uri={image},"
              f"python-module=tactic.models.vertex_entry")
    cmd = ["gcloud", "ai", "custom-jobs", "create",
           f"--region={v.region}", f"--project={v.project_id}",
           f"--display-name=tactic-{run_id}",
           f"--python-package-uris={pkg_uri}",
           f"--worker-pool-spec={worker}",
           f"--args={args}"]
    _run(cmd)
    print(f"[4/4] submitted. run_id={run_id}\n  artifacts will land at: {out_uri}")
    print(f"  poll: gcloud ai custom-jobs list --region={v.region} --project={v.project_id} "
          f"--filter='displayName:tactic-{run_id}'")


if __name__ == "__main__":
    main()
