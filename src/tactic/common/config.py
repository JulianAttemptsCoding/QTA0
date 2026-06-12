"""Config loading + .env. Single source of truth for paths and the active config.

Paths are derived from the repo root (two levels above this file's package), so the
pipeline runs identically on a workstation and on a Vertex worker.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# repo root = .../<repo>/   (src/tactic/common/config.py -> up 4)
ROOT = Path(__file__).resolve().parents[3]


def _resolve_configs_dir() -> Path:
    """Find configs/. Honors $TACTIC_CONFIG_DIR (set by the Vertex entry), else repo root,
    else CWD — so the package works both from a source checkout and pip-installed in a container.
    """
    env = os.environ.get("TACTIC_CONFIG_DIR")
    if env and (Path(env) / "v1.yaml").exists():
        return Path(env)
    for cand in (ROOT / "configs", Path.cwd() / "configs", Path.cwd()):
        if (cand / "v1.yaml").exists():
            return cand
    return ROOT / "configs"


CONFIGS_DIR = _resolve_configs_dir()
DATA_DIR = Path(os.environ.get("TACTIC_DATA_DIR", ROOT / "data"))
REGISTRY_DIR = Path(os.environ.get("TACTIC_REGISTRY_DIR", ROOT / "registry"))
REPORTS_DIR = Path(os.environ.get("TACTIC_REPORTS_DIR", ROOT / "reports"))

RAW = DATA_DIR / "raw"
CURATED = DATA_DIR / "curated"
FEATURES = DATA_DIR / "features"
LABELS = DATA_DIR / "labels"
HOLDOUT = DATA_DIR / "holdout"
QUARANTINE = DATA_DIR / "diagnostic_quarantine"

LEDGER_PATH = REGISTRY_DIR / "trial_ledger.parquet"
HOLDOUT_LOCK = HOLDOUT / ".LOCKED"


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Minimal .env loader (no external dep required). Populates os.environ."""
    path = Path(path) if path else ROOT / ".env"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        out[key] = val
        os.environ.setdefault(key, val)
    return out


def load_config(name: str = "v1") -> dict[str, Any]:
    """Load configs/<name>.yaml as a plain dict. Resolves the configs dir at call time so a
    late-set $TACTIC_CONFIG_DIR (e.g. on a Vertex worker) is honored."""
    p = _resolve_configs_dir() / f"{name}.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class VertexConfig:
    project_id: str
    region: str
    bucket: str
    bucket_fallback: str
    machine_full: str
    machine_smoke: str
    image_cpu: str
    image_gpu: str
    package_module: str

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VertexConfig":
        v = cfg["vertex"]
        env = os.environ.get
        return cls(
            project_id=env("GCP_PROJECT_ID", v["project_id"]),
            region=env("GCP_REGION", v["region"]),
            bucket=env("GCS_BUCKET", v["bucket"]),
            bucket_fallback=env("GCS_BUCKET_FALLBACK", v["bucket_fallback"]),
            machine_full=env("VERTEX_MACHINE_FULL", v["machine_full"]),
            machine_smoke=env("VERTEX_MACHINE_SMOKE", v["machine_smoke"]),
            image_cpu=env("VERTEX_IMAGE_CPU", v["image_cpu"]),
            image_gpu=env("VERTEX_IMAGE_GPU", v["image_gpu"]),
            package_module=v.get("package_module", "tactic.models.hpo_vertex"),
        )


def ensure_dirs() -> None:
    """Create the data/registry/reports tree if missing (idempotent)."""
    for d in (RAW, CURATED, FEATURES, LABELS, HOLDOUT, QUARANTINE,
              REGISTRY_DIR / "config_hashes", REGISTRY_DIR / "prereg", REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
