"""Content hashing + sidecar manifests (Non-negotiable §0.3: every artifact gets a .sha256).

Determinism: config hashes are computed on a canonical JSON serialization so dict
ordering never changes the hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: Any) -> str:
    """Order-independent hash of a config object (dict/list/scalars)."""
    return sha256_bytes(canonical_json(cfg).encode("utf-8"))


def write_sidecar(path: str | Path) -> str:
    """Write `<path>.sha256` next to an artifact; return the digest."""
    path = Path(path)
    digest = sha256_file(path)
    side = path.with_suffix(path.suffix + ".sha256")
    side.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def verify_sidecar(path: str | Path) -> bool:
    """True iff `<path>.sha256` exists and matches current file bytes."""
    path = Path(path)
    side = path.with_suffix(path.suffix + ".sha256")
    if not side.exists():
        return False
    expected = side.read_text(encoding="utf-8").split()[0]
    return expected == sha256_file(path)
