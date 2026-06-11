"""Declarative kill-gate engine (§2.4, §17).

A failed gate writes reports/KILL_REPORT_<gate>.md and raises KillGate. Per Non-negotiable
§0.3.10, a halted build is a *successful* run of the plan — `make all` stops at the first
failed gate by design.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .config import REPORTS_DIR, load_config


class KillGate(RuntimeError):
    """Raised when a kill gate fails. Carries the gate id and context."""

    def __init__(self, gate_id: str, value: Any, threshold: Any, report_path: Path):
        self.gate_id = gate_id
        self.value = value
        self.threshold = threshold
        self.report_path = report_path
        super().__init__(f"GATE {gate_id} FAILED: value={value!r} threshold={threshold!r} "
                         f"-> {report_path}")


def gate_spec(gate_id: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    table = cfg.get("gate_table", {})
    if gate_id not in table:
        raise KeyError(f"unknown gate {gate_id!r}; known: {sorted(table)}")
    return table[gate_id]


def write_kill_report(
    gate_id: str,
    value: Any,
    threshold: Any,
    interpretation: str,
    next_step: str,
    inputs: dict | None = None,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"KILL_REPORT_{gate_id}.md"
    lines = [
        f"# KILL REPORT — {gate_id}",
        "",
        f"- **Timestamp (UTC):** {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- **Observed value:** `{value!r}`",
        f"- **Threshold:** `{threshold!r}`",
        "",
        "## Inputs",
        "",
    ]
    for k, v in (inputs or {}).items():
        lines.append(f"- `{k}` = `{v!r}`")
    lines += [
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Recommended next step",
        "",
        next_step,
        "",
        "> A halted build is a successful run of PLAN.md (Non-negotiable §0.3.10).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evaluate(
    gate_id: str,
    passed: bool,
    value: Any,
    threshold: Any,
    interpretation: str = "",
    next_step: str = "",
    inputs: dict | None = None,
    soft: bool = False,
) -> bool:
    """Return True on PASS. On FAIL write a kill report; raise KillGate unless `soft`.

    The caller decides the PASS/FAIL boolean from gate-specific math (kept in the
    owning module); this engine handles reporting + halting uniformly.
    """
    if passed:
        return True
    report = write_kill_report(
        gate_id, value, threshold,
        interpretation or f"Gate {gate_id} condition not met.",
        next_step or "Inspect the inputs above and the relevant phase in PLAN.md.",
        inputs,
    )
    if soft:
        return False
    raise KillGate(gate_id, value, threshold, report)
