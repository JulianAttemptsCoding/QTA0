"""Smoke: every tactic module imports cleanly (catches stub syntax errors), and the
diagnostic quarantine has no import path into curated/features (sec3.4)."""
import importlib
import pkgutil
from pathlib import Path

import pytest

import tactic

SRC = Path(tactic.__file__).resolve().parent


def _all_modules():
    mods = []
    for m in pkgutil.walk_packages(tactic.__path__, prefix="tactic."):
        mods.append(m.name)
    return mods


@pytest.mark.parametrize("modname", _all_modules())
def test_module_imports(modname):
    importlib.import_module(modname)


def test_quarantine_isolation():
    """stooq_diag (quarantine writer) must not import curated/feature builders."""
    src = (SRC / "ingest" / "stooq_diag.py").read_text(encoding="utf-8")
    for forbidden in ("build_features", "load_curated", "from ..features", "from ..labels"):
        assert forbidden not in src, f"quarantine module references {forbidden!r}"
