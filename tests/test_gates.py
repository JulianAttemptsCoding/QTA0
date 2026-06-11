import pytest

from tactic.common import gates


def test_pass_returns_true():
    assert gates.evaluate("G1", passed=True, value=0.01, threshold=0.005) is True


def test_fail_writes_report_and_raises(tmp_path):
    path = gates.write_kill_report(
        "G1", value=0.001, threshold=0.005,
        interpretation="no edge", next_step="halt",
        reports_dir=tmp_path,
    )
    assert path.exists()
    assert "KILL REPORT" in path.read_text(encoding="utf-8")


def test_evaluate_hard_fail_raises():
    with pytest.raises(gates.KillGate) as exc:
        gates.evaluate("G1", passed=False, value=0.001, threshold=0.005)
    assert exc.value.gate_id == "G1"


def test_evaluate_soft_fail_returns_false():
    assert gates.evaluate("G0b", passed=False, value=0.07, threshold=0.05, soft=True) is False


def test_gate_spec_lookup():
    spec = gates.gate_spec("G8")
    assert spec["phase"] == 13
