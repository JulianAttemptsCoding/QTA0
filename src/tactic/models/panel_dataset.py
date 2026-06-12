"""Panel batching (sec10.4): one date = one batch element.

Builds, per decision date t, the tensors for every entity that has >=L history and a realized
forward target:
  x_seq  (N_t, 18, L)   trailing L channel rows (channel-first for the TCN)
  u      (N_t, 12)      static features at t
  m      (5,)           market state at t
  y      (N_t,)         vol-standardized forward target (open(t+1)->open(t+2)) / sigma_t
  entities, dates

Target = open-to-open return realized AFTER a next-open entry (OO structure, 1-day hold),
so nothing in x_seq overlaps the target window (PIT-safe).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CH_COLS = [f"ch{i:02d}" for i in range(1, 19)]
U_COLS = [f"u{i:02d}" for i in range(1, 13)]
M_COLS = ["m1", "m2", "m3", "m4", "m5"]
SEQ_LEN = 64


@dataclass
class DateBatch:
    date: object
    x_seq: np.ndarray   # (N, 18, L)
    u: np.ndarray       # (N, 12)
    m: np.ndarray       # (5,)
    y: np.ndarray       # (N,)
    entities: list


def _forward_oo_target(opens: np.ndarray) -> np.ndarray:
    """Return realized over [open(t+1), open(t+2)] aligned to decision day t."""
    lo = np.log(opens)
    r = np.full_like(lo, np.nan)
    r[:-2] = lo[2:] - lo[1:-1]
    return r


def build_panel(features: pd.DataFrame, prices: pd.DataFrame, labels: pd.DataFrame,
                seq_len: int = SEQ_LEN) -> list[DateBatch]:
    feats = features.copy()
    feats["date"] = pd.to_datetime(feats["date"]).dt.date
    prices = prices[prices["adjustment"] == "all"].copy()
    prices["entity"] = prices.get("entity", prices["symbol"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"]).dt.date

    sig = labels.set_index(["entity", "date"])["sigma"]

    ent_data: dict = {}
    for ent, g in feats.groupby("entity", sort=False):
        g = g.sort_values("date")
        pr = prices[prices["entity"] == ent].sort_values("date")
        common = set(g["date"]) & set(pr["date"])
        g = g[g["date"].isin(common)]
        pr = pr[pr["date"].isin(common)].sort_values("date")
        if len(g) < seq_len + 3:
            continue
        dates = g["date"].to_numpy()
        ch = g[CH_COLS].to_numpy(float)
        u = g[U_COLS].to_numpy(float)
        m = g[M_COLS].to_numpy(float)
        opens = pr.set_index("date").loc[dates, "o"].to_numpy(float)
        tgt = _forward_oo_target(opens)
        sigs = np.array([sig.get((ent, d), np.nan) for d in dates])
        ent_data[ent] = {"dates": dates, "ch": ch, "u": u, "m": m, "y": tgt / sigs}

    if not ent_data:
        return []
    all_dates = sorted(set().union(*[set(v["dates"]) for v in ent_data.values()]))
    date_to_pos = {ent: {d: i for i, d in enumerate(v["dates"])} for ent, v in ent_data.items()}

    batches: list[DateBatch] = []
    for d in all_dates:
        xs, us, ms, ys, ents = [], [], [], [], []
        for ent, v in ent_data.items():
            i = date_to_pos[ent].get(d)
            if i is None or i < seq_len - 1:
                continue
            window = v["ch"][i - seq_len + 1: i + 1]
            y = v["y"][i]
            if not np.isfinite(y) or not np.isfinite(window).all() or not np.isfinite(v["u"][i]).all():
                continue
            xs.append(window.T)
            us.append(v["u"][i]); ms.append(v["m"][i]); ys.append(y); ents.append(ent)
        if len(ents) >= 3:
            batches.append(DateBatch(
                date=d, x_seq=np.stack(xs), u=np.stack(us),
                m=np.nanmean(np.stack(ms), axis=0), y=np.array(ys), entities=ents))
    return batches
