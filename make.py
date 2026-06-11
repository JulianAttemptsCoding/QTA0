#!/usr/bin/env python
"""Cross-platform `make` shim for Windows (no GNU make required).

    python make.py setup
    python make.py all
    python make.py test

Delegates to `python -m tactic.cli <target>` (or pytest/ruff for test/lint).
"""
from __future__ import annotations

import subprocess
import sys

PY = sys.executable


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python make.py <target> [args...]")
        print("targets: setup test data-alpaca data-factors data-universe day1-audits")
        print("         universe costs labels features diagnostics baselines infra train")
        print("         aggregate decide backtest stress firewall reports all lint")
        return 1
    target, rest = sys.argv[1], sys.argv[2:]
    if target == "test":
        return subprocess.call([PY, "-m", "pytest", "-q", *rest])
    if target == "lint":
        return subprocess.call([PY, "-m", "ruff", "check", "src", "tests", *rest])
    return subprocess.call([PY, "-m", "tactic.cli", target, *rest])


if __name__ == "__main__":
    raise SystemExit(main())
