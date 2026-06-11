"""Kill-report rendering (sec16). Thin re-export of the gates engine writer so reports/
modules and gate evaluations share one code path. A halted build writes one of these and
is, per sec0.3.10, a successful run of PLAN.md.
"""
from __future__ import annotations

from ..common.gates import KillGate, evaluate, write_kill_report

__all__ = ["KillGate", "evaluate", "write_kill_report"]
