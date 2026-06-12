"""Component labels (sec6.1, Appendix A.1) from auction prices, else first/last prints.

Per (entity, date d) realized:
  r_on[d] = ln A_O[d] - ln A_C[d-1]      (overnight into d)
  r_in[d] = ln A_C[d] - ln A_O[d]        (intraday d)
  r_cc[d] = r_on[d] + r_in[d]            (close-to-close; identity enforced)
  r_oo[d] = ln A_O[d] - ln A_O[d-1] = r_in[d-1] + r_on[d]
z_top_{on,in}: 1 if component >= 80th cross-sectional pct that date (p=0.20 top bucket).
sigma: causal EWMA (lambda=0.94) on r_cc, floored at 0.2%/day; y_* = r_*/sigma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import CURATED, load_config
from ..common.io import write_parquet


def _ewma_sigma(r_cc: pd.Series, lam: float, floor: float) -> pd.Series:
    """Causal EWMA volatility: sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r^2. Floored sigma."""
    r2 = (r_cc.fillna(0.0) ** 2).to_numpy()
    var = np.empty(len(r2))
    prev = floor ** 2
    for i, x in enumerate(r2):
        prev = lam * prev + (1 - lam) * x
        var[i] = prev
    return pd.Series(np.maximum(np.sqrt(var), floor), index=r_cc.index)


def _entity_labels(g: pd.DataFrame, lam: float, floor: float) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    ao, ac = np.log(g["a_open"].to_numpy()), np.log(g["a_close"].to_numpy())
    r_on = ao - np.concatenate([[np.nan], ac[:-1]])
    r_in = ac - ao
    r_cc = r_on + r_in
    r_oo = ao - np.concatenate([[np.nan], ao[:-1]])
    out = pd.DataFrame({
        "entity": g["entity"], "date": g["date"],
        "r_on": r_on, "r_in": r_in, "r_cc": r_cc, "r_oo": r_oo,
        "label_source": g["label_source"],
    })
    out["sigma"] = _ewma_sigma(pd.Series(r_cc, index=out.index), lam, floor).to_numpy()
    for comp in ("on", "in", "cc", "oo"):
        out[f"y_{comp}"] = out[f"r_{comp}"] / out["sigma"]
    return out


def build_labels(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    lam = cfg["labels"]["ewma_lambda"]
    floor = cfg["labels"]["sigma_floor"]
    top_p = cfg["panel"]["top_p"]

    prices = pd.read_parquet(CURATED / "prices_daily.parquet")
    prices = prices[prices["adjustment"] == "all"].copy()
    prices["entity"] = prices.get("entity", prices["symbol"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    # auction prices where available, else first/last prints (o, c)
    apath = CURATED / "auctions_daily.parquet"
    if apath.exists():
        auc = pd.read_parquet(apath)
        auc["date"] = pd.to_datetime(auc["date"]).dt.date
        auc = auc.rename(columns={"symbol": "entity"})
        m = prices.merge(auc[["entity", "date", "open_auction_px", "close_auction_px"]],
                         on=["entity", "date"], how="left")
    else:
        m = prices.copy()
        m["open_auction_px"] = np.nan
        m["close_auction_px"] = np.nan
    m["a_open"] = m["open_auction_px"].where(m["open_auction_px"].notna(), m["o"])
    m["a_close"] = m["close_auction_px"].where(m["close_auction_px"].notna(), m["c"])
    m["label_source"] = np.where(m["open_auction_px"].notna() & m["close_auction_px"].notna(),
                                 "auction", "print")
    m = m.dropna(subset=["a_open", "a_close"])
    m = m[(m["a_open"] > 0) & (m["a_close"] > 0)]

    parts = [_entity_labels(g, lam, floor) for _, g in m.groupby("entity", sort=False)]
    lab = pd.concat(parts, ignore_index=True)

    # cross-sectional top-bucket indicators (top_p = top quintile -> 1-top_p quantile)
    for comp in ("on", "in"):
        thr = lab.groupby("date")[f"r_{comp}"].transform(lambda s: s.quantile(1 - top_p))
        lab[f"z_top_{comp}"] = (lab[f"r_{comp}"] >= thr).astype(int)

    write_parquet(lab, CURATED / "labels.parquet")
    return lab
