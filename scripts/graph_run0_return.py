"""run0 OOS cumulative percent return vs time (2023-2026).

run0 = f4_vertex_3yr (83 Alpaca-SIP names, trained 2016-21, OOS 2023-01-12 -> 2026-01-30).
Daily net returns pulled from the run0 branch (git 18e1896) backtest_2023_2026.csv.
SPY buy-and-hold drawn faintly for reference. Output: run0_oos_return.png at repo root.
"""
from __future__ import annotations
import subprocess
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "run0_oos_return.png"

csv = subprocess.check_output(
    ["git", "show", "18e1896:results/f4_vertex_3yr/backtest_2023_2026.csv"], cwd=str(ROOT)
).decode()
d = pd.read_csv(StringIO(csv))
d["date"] = pd.to_datetime(d["date"])
d = d.sort_values("date").reset_index(drop=True)

strat = (np.cumprod(1 + d["ret_net"].to_numpy()) - 1) * 100
spy = (np.cumprod(1 + d["spy"].to_numpy()) - 1) * 100

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(d["date"], strat, color="seagreen", lw=1.8,
        label=f"run0 strategy (net)  final {strat[-1]:+.1f}%")
ax.plot(d["date"], spy, color="gray", lw=1.3, alpha=0.8,
        label=f"SPY buy-and-hold  final {spy[-1]:+.1f}%")
ax.axhline(0, color="k", lw=0.8)
ax.set_title("run0 — OOS cumulative percent return vs time (2023-01-12 → 2026-01-30)")
ax.set_ylabel("cumulative return (%)"); ax.set_xlabel("date")
ax.legend(loc="upper left"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT, dpi=130)
print("wrote", OUT)
print(f"run0 final {strat[-1]:+.1f}% | SPY final {spy[-1]:+.1f}% | ndays {len(d)}")
