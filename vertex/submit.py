"""Submit TACTIC-MoB heavy jobs to Vertex AI custom-jobs (gcloud CLI wrapper).

Mirrors the submit pattern provided for the project. Builds an sdist, uploads it to GCS,
and launches a custom job whose python-module is a tactic entrypoint (default the Vizier
HPO worker, tactic.models.hpo_vertex).

Examples:
    python vertex/submit.py --module tactic.models.hpo_vertex --smoke
    python vertex/submit.py --module tactic.models.train --args expert=f4 refit=2021

Config comes from configs/v1.yaml[vertex] with env overrides (see .env.example). Requires
`gcloud` authenticated as the configured account and a built sdist (auto-built here).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactic.common.config import VertexConfig, load_config, load_dotenv  # noqa: E402


def build_sdist() -> Path:
    """Build the python package tarball under vertex/dist/."""
    dist = ROOT / "vertex" / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "build"])
    subprocess.check_call([sys.executable, "-m", "build", "--sdist", "--outdir", str(dist)],
                          cwd=str(ROOT))
    tarballs = sorted(dist.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)
    if not tarballs:
        raise RuntimeError("sdist build produced no tarball")
    return tarballs[-1]


def upload(local: Path, bucket: str) -> str:
    uri = f"{bucket.rstrip('/')}/packages/{local.name}"
    subprocess.check_call(["gcloud", "storage", "cp", str(local), uri])
    return uri


def submit(module: str, args_list: list[str], smoke: bool, gpu: bool,
           display_name: str | None = None) -> None:
    load_dotenv()
    cfg = load_config()
    v = VertexConfig.from_config(cfg)
    pkg = build_sdist()
    pkg_uri = upload(pkg, v.bucket)

    machine = v.machine_smoke if smoke else v.machine_full
    image = v.image_gpu if gpu else v.image_cpu
    name = display_name or f"tactic-{module.split('.')[-1]}-{'smoke' if smoke else 'full'}"
    worker = (
        f"machine-type={machine},replica-count=1,"
        f"executor-image-uri={image},python-module={module}"
    )
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--region={v.region}",
        f"--project={v.project_id}",
        f"--display-name={name}",
        f"--python-package-uris={pkg_uri}",
        f"--worker-pool-spec={worker}",
    ]
    if args_list:
        cmd.append(f"--args={','.join(args_list)}")
    print("submitting:\n  " + " \\\n  ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit a Vertex AI custom job for TACTIC-MoB.")
    ap.add_argument("--module", default="tactic.models.hpo_vertex")
    ap.add_argument("--args", nargs="*", default=[], help="key=value args passed to the module")
    ap.add_argument("--smoke", action="store_true", help="n1-standard-4 instead of n1-standard-16")
    ap.add_argument("--gpu", action="store_true", help="use the plan_d pytorch-xla GPU image")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    submit(a.module, a.args, a.smoke, a.gpu, a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
