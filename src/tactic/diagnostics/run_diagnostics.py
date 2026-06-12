"""Phase 5 driver (sec7): run diagnostics, evaluate gate G1 (predictability exists) and
G1b (structure break-even). All pre-registered + non-selective. Writes reports/diagnostics.md.

G1 (kill): proceed iff max|mean IC| with NW t>2 >= ic_min, OR any VR rejects RW at 1%, OR
the tug-of-war table shows opposite-signed component continuation. Else HALT -> a
negative-result report (a deliverable, not a failure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import CURATED, REPORTS_DIR, load_config
from ..common.gates import evaluate
from .ic_decay import ic_decay
from .tugofwar_table import continuation_spread, tugofwar_table
from .variance_ratio import variance_ratios


def _simple_features(labels: pd.DataFrame) -> pd.DataFrame:
    df = labels.sort_values(["entity", "date"]).copy()
    g = df.groupby("entity")
    df["mom21"] = g["r_cc"].transform(lambda s: s.rolling(21, min_periods=10).sum())
    df["rev5"] = g["r_cc"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    df["vol21"] = g["r_cc"].transform(lambda s: s.rolling(21, min_periods=10).std())
    return df[["entity", "date", "mom21", "rev5", "vol21"]].dropna()


def _ew_component(labels: pd.DataFrame, comp: str) -> np.ndarray:
    return labels.groupby("date")[comp].mean().to_numpy()


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ic_min = cfg["gates"]["ic_min"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(CURATED / "labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.date

    # 1) variance ratios on EW component series
    vr_rows = []
    vr_reject = False
    for comp in ("r_on", "r_in", "r_cc"):
        for rec in variance_ratios(_ew_component(labels, comp), qs=(2, 5, 10, 20)):
            rec["component"] = comp
            vr_rows.append(rec)
            if np.isfinite(rec["p_value"]) and rec["p_value"] < 0.01:
                vr_reject = True
    vr_df = pd.DataFrame(vr_rows)

    # 2) IC of simple signals
    feats = _simple_features(labels)
    ic_df = ic_decay(feats, labels, horizons=range(1, 6))  # short horizons for the gate
    sig_ic = ic_df.dropna(subset=["ic_mean", "t_nw"])
    ic_pass = bool(((sig_ic["ic_mean"].abs() >= ic_min) & (sig_ic["t_nw"].abs() > 2)).any())
    max_ic = float(sig_ic["ic_mean"].abs().max()) if len(sig_ic) else 0.0

    # 3) tug-of-war
    tow = tugofwar_table(labels)
    tow_on = continuation_spread(tow, "past_on", "next_on")
    tow_in = continuation_spread(tow, "past_in", "next_in")
    tow_signal = bool(np.isfinite(tow_on) and np.isfinite(tow_in)
                      and np.sign(tow_on) != np.sign(tow_in))

    g1_pass = ic_pass or vr_reject or tow_signal

    # report
    lines = [
        "# Diagnostics (sec7) + Gate G1",
        f"- max |mean IC| = {max_ic:.4f} (threshold {ic_min}); NW t>2 present: {ic_pass}",
        f"- VR rejects random walk at 1%: {vr_reject}",
        f"- tug-of-war opposite-signed continuation: {tow_signal} "
        f"(on spread={tow_on}, in spread={tow_in})",
        "",
        "## Variance ratios",
        vr_df.to_markdown(index=False),
        "",
        "## IC (h=1..5, simple signals)",
        sig_ic.to_markdown(index=False),
        "",
        "## Tug-of-war table",
        tow.to_markdown(index=False),
    ]
    (REPORTS_DIR / "diagnostics.md").write_text("\n".join(str(x) for x in lines), encoding="utf-8")

    evaluate(
        "G1", passed=g1_pass, value={"max_ic": max_ic, "vr_reject": vr_reject, "tow": tow_signal},
        threshold=f"IC>={ic_min}&t>2 OR VR p<0.01 OR tug-of-war",
        interpretation="No exploitable predictability found in any pre-registered diagnostic."
        if not g1_pass else "Predictability present.",
        next_step="Write the negative-result report; do not proceed to modeling."
        if not g1_pass else "Proceed to baselines + infrastructure.",
    )
    return {"g1_pass": g1_pass, "max_ic": max_ic, "vr_reject": vr_reject, "tow_signal": tow_signal}
