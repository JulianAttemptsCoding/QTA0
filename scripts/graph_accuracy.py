"""Directional prediction accuracy over OOS time: run0 (SPY-beater, 2023, 83 names) vs
run2.1 (deep chronological, 2018-2026, 153 names).

Accuracy = per-session mean of sign(q50)==sign(y_true) (the same hit-rate the run reports use).
Plots raw daily + 21-session rolling, with the 50% coin-flip line. Panel B zooms to 2023 for an
apples-to-apples calendar comparison.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R0 = ROOT / "r0check" / "run0_3yr_oos_predictions.parquet"   # run0 = 2023-2026 full OOS (showcase)
R21 = ROOT / "results" / "run2_1_chrono" / "oos_predictions.parquet"
OUT = ROOT / "results" / "run2_1_chrono" / "accuracy_run0_vs_run21.png"


def daily_hit(path):
    d = pd.read_parquet(path).dropna(subset=["q50", "y_true"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["hit"] = (np.sign(d["q50"]) == np.sign(d["y_true"])).astype(float)
    g = d.groupby("date")["hit"].mean()
    return g.sort_index()


h0 = daily_hit(R0)
h21 = daily_hit(R21)
m0, m21 = h0.mean(), h21.mean()
roll0 = h0.rolling(21, min_periods=5).mean()
roll21 = h21.rolling(21, min_periods=5).mean()

fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 9))

# Panel A: full timelines
axA.axhline(0.5, color="k", lw=1, ls="--", label="coin flip 50%")
axA.plot(roll21.index, roll21.values, color="crimson", lw=1.2,
         label=f"run2.1 deep 2018-26 (21d roll)  mean={m21:.1%}")
axA.plot(roll0.index, roll0.values, color="seagreen", lw=1.6,
         label=f"run0 SPY-beater 2023-26 (21d roll)  mean={m0:.1%}")
axA.axhline(m21, color="crimson", lw=0.8, ls=":", alpha=0.7)
axA.axhline(m0, color="seagreen", lw=0.8, ls=":", alpha=0.7)
axA.set_title("A. Directional hit rate over OOS time (21-session rolling)")
axA.set_ylabel("daily hit rate"); axA.set_ylim(0.40, 0.60)
axA.legend(loc="upper left", fontsize=9); axA.grid(alpha=0.25)

# Panel B: 2023-2026 overlap head-to-head (apples-to-apples — both runs cover this window)
lo, hi = "2023-01-12", "2026-02-25"
h0o = h0[(h0.index >= lo) & (h0.index <= hi)]
h21o = h21[(h21.index >= lo) & (h21.index <= hi)]
axB.axhline(0.5, color="k", lw=1, ls="--", label="coin flip 50%")
axB.plot(h0o.index, h0o.rolling(21, min_periods=5).mean().values, color="seagreen", lw=1.6,
         label=f"run0 83names trained2016-21 (21d roll)  mean={h0o.mean():.1%}")
axB.plot(h21o.index, h21o.rolling(21, min_periods=5).mean().values, color="crimson", lw=1.2,
         label=f"run2.1 153names trained2000-15 (21d roll)  mean={h21o.mean():.1%}")
axB.set_title("B. Same 2023-2026 window, head-to-head — run0 vs run2.1")
axB.set_ylabel("daily hit rate"); axB.set_ylim(0.40, 0.62)
axB.legend(loc="upper left", fontsize=9); axB.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
def tstat(h):
    return (h.mean() - 0.5) / (h.std(ddof=1) / np.sqrt(len(h)))
print(f"run0  2023-26 mean hit = {m0:.4f}  ndays={len(h0)}  day-clustered t={tstat(h0):+.2f}")
print(f"run2.1 2018-26 mean hit = {m21:.4f}  ndays={len(h21)}  day-clustered t={tstat(h21):+.2f}")
lo, hi = "2023-01-12", "2026-02-25"
h0o = h0[(h0.index >= lo) & (h0.index <= hi)]; h21o = h21[(h21.index >= lo) & (h21.index <= hi)]
print(f"OVERLAP 2023-26: run0 {h0o.mean():.4f} (t={tstat(h0o):+.2f}) | run2.1 {h21o.mean():.4f} (t={tstat(h21o):+.2f})")
print("per-year hit (run0 | run2.1):")
for y in range(2023, 2027):
    a = h0[h0.index.year == y]; b = h21[h21.index.year == y]
    av = f"{a.mean():.4f}" if len(a) else "  -   "
    bv = f"{b.mean():.4f}" if len(b) else "  -   "
    print(f"  {y}: {av} | {bv}")
