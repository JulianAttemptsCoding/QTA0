"""FROZEN v1 features (sec6.2). All channels computed through close t on adjusted SIP bars;
nothing uses data after t (PIT). Output is a long table of per-(entity,date) values:

  18 sequence channels ch01..ch18  (panel_dataset slices trailing L=64 of these to form x_seq)
  12 static features u01..u12       (cross-sectional pct-ranks + s_blend + ln ADV + interactions)
   5 market-state m1..m5            (per-date, broadcast to every entity that day)

feature_available_ts = close(t) + 30min (recorded per row). Per-channel standardization for
the neural experts is fit on the TRAIN fold only (handled in models/train.py), not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import CURATED, load_config
from ..common.io import write_parquet
from .wavelets import modwt_energy

SEQ_LEN = 64
_MODWT_WIN = 64  # trailing window for MODWT energy channels


def _entity_channels(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    o, h, l, c = (np.log(g[x].to_numpy()) for x in ("o", "h", "l", "c"))
    v = g["v"].to_numpy(float)
    n = g["n"].to_numpy(float) if "n" in g else np.full(len(g), np.nan)
    vw = np.log(g["vw"].to_numpy()) if "vw" in g else c

    ret = np.concatenate([[np.nan], np.diff(c)])                 # ch01 log close return
    rng = h - l                                                  # ch02 intraday log range
    gap = o - np.concatenate([[np.nan], c[:-1]])                 # ch03 overnight gap
    s = pd.Series(ret)

    def zroll(arr, w=63):
        a = pd.Series(arr)
        return ((a - a.rolling(w, min_periods=w // 2).mean())
                / a.rolling(w, min_periods=w // 2).std()).to_numpy()

    vol_z = zroll(v)                                             # ch04 volume z (63d)
    tcnt_z = zroll(n)                                            # ch05 trade-count z (63d)
    vwap_dev = c - vw                                            # ch06 VWAP deviation
    rv5 = s.rolling(5, min_periods=3).std().to_numpy()          # ch07-09 realized vol
    rv21 = s.rolling(21, min_periods=10).std().to_numpy()
    rv63 = s.rolling(63, min_periods=30).std().to_numpy()
    mom5 = s.rolling(5, min_periods=3).sum().to_numpy()         # ch10-12 momentum (cum log ret)
    mom21 = s.rolling(21, min_periods=10).sum().to_numpy()
    mom63 = s.rolling(63, min_periods=30).sum().to_numpy()
    cl = pd.Series(c)

    def dd(w):
        return (cl - cl.rolling(w, min_periods=w // 2).max()).to_numpy()  # log drawdown (<=0)

    dd21, dd63, dd126 = dd(21), dd(63), dd(126)                 # ch13-15 drawdown

    # ch16-18 MODWT energy (db4) levels 2,3,4 on trailing return window
    e2 = np.full(len(g), np.nan); e3 = np.full(len(g), np.nan); e4 = np.full(len(g), np.nan)
    r_arr = ret.copy()
    for t in range(len(g)):
        if t + 1 >= _MODWT_WIN:
            win = r_arr[t + 1 - _MODWT_WIN: t + 1]
            win = win[np.isfinite(win)]
            if len(win) >= 16:
                en = modwt_energy(win, levels=(2, 3, 4))
                e2[t], e3[t], e4[t] = en[2], en[3], en[4]

    out = pd.DataFrame({
        "entity": g["entity"], "date": g["date"],
        "ch01": ret, "ch02": rng, "ch03": gap, "ch04": vol_z, "ch05": tcnt_z, "ch06": vwap_dev,
        "ch07": rv5, "ch08": rv21, "ch09": rv63, "ch10": mom5, "ch11": mom21, "ch12": mom63,
        "ch13": dd21, "ch14": dd63, "ch15": dd126, "ch16": e2, "ch17": e3, "ch18": e4,
    })
    # raw quantities used by static features
    out["_mom21"] = mom21
    out["_mom63"] = mom63
    out["_mom252_21"] = (s.rolling(252, min_periods=120).sum().shift(21)).to_numpy()  # 12-1 mom
    out["_vol21"] = rv21
    out["_dd63"] = dd63
    out["_adv21"] = (pd.Series(g["c"].to_numpy() * v).rolling(21, min_periods=10).mean()).to_numpy()
    out["_ret"] = ret
    out["_c"] = g["c"].to_numpy()
    return out


def _market_state(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date market-state M[5] from the cross-section of entity returns."""
    piv = df.pivot_table(index="date", columns="entity", values="_ret")
    dates = piv.index
    ew = piv.mean(axis=1)                                        # EW return per date
    m1 = ew.rolling(21, min_periods=10).std()                   # 21d market realized vol
    m2 = piv.std(axis=1)                                        # cross-sectional dispersion
    # breadth: % of entities above their 63d MA (reindex to piv.index to stay aligned)
    cpiv = df.pivot_table(index="date", columns="entity", values="_c").reindex(dates)
    ma63 = cpiv.rolling(63, min_periods=30).mean()
    m3 = ((cpiv > ma63).sum(axis=1) / cpiv.notna().sum(axis=1).clip(lower=1)).reindex(dates)
    ew_idx = (1 + ew.fillna(0)).cumprod()
    m4 = ew_idx / ew_idx.rolling(252, min_periods=60).max() - 1  # EW-index drawdown
    # top-eigenvalue share over trailing 63d
    m5 = pd.Series(index=dates, dtype=float)
    R = piv.to_numpy()
    for i in range(len(dates)):
        if i + 1 >= 63:
            w = R[i + 1 - 63: i + 1]
            w = w[:, ~np.isnan(w).any(axis=0)]
            if w.shape[1] >= 2 and w.shape[0] >= 30:
                cov = np.cov(w.T)
                ev = np.linalg.eigvalsh(cov)
                tr = ev.sum()
                m5.iloc[i] = ev[-1] / tr if tr > 0 else np.nan
    res = pd.DataFrame(index=dates)
    res["m1"], res["m2"], res["m3"], res["m4"], res["m5"] = m1, m2, m3, m4, m5
    return res.reset_index(names="date")


def build_features(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    prices = pd.read_parquet(CURATED / "prices_daily.parquet")
    prices = prices[prices["adjustment"] == "all"].copy()
    prices["entity"] = prices.get("entity", prices["symbol"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    parts = [_entity_channels(g) for _, g in prices.groupby("entity", sort=False)]
    df = pd.concat(parts, ignore_index=True)

    # market state, broadcast per date
    M = _market_state(df)
    df = df.merge(M, on="date", how="left")

    # static u: cross-sectional pct-ranks per date
    def xrank(col):
        return df.groupby("date")[col].rank(pct=True)
    df["u01"] = xrank("_mom21"); df["u02"] = xrank("_mom63"); df["u03"] = xrank("_mom252_21")
    df["u04"] = xrank("_vol21"); df["u05"] = xrank("_dd63"); df["u06"] = xrank("_adv21")
    # s_blend from spread surface
    spath = CURATED / "spread_surface.parquet"
    if spath.exists():
        surf = pd.read_parquet(spath)
        surf["date"] = pd.to_datetime(surf["date"]).dt.date
        df = df.merge(surf[["entity", "date", "s_blend"]], on=["entity", "date"], how="left")
    else:
        df["s_blend"] = np.nan
    df["u07"] = df["s_blend"].fillna(df["s_blend"].median())
    df["u08"] = np.log(df["_adv21"].clip(lower=1.0))
    rank_mom21 = df["u01"]
    df["u09"] = rank_mom21 * df["m1"]
    df["u10"] = rank_mom21 * df["m2"]
    df["u11"] = rank_mom21 * df["m3"]
    df["u12"] = rank_mom21 * df["m4"]

    df["feature_available_ts"] = pd.to_datetime(df["date"].astype(str) + " 16:30")
    df["feature_set_version"] = cfg["frozen"]["feature_set_version"]

    keep = (["entity", "date", "feature_available_ts", "feature_set_version"]
            + [f"ch{ i:02d}" for i in range(1, 19)]
            + [f"u{i:02d}" for i in range(1, 13)]
            + ["m1", "m2", "m3", "m4", "m5"])
    out = df[keep].copy()
    write_parquet(out, CURATED / "features.parquet")
    return out
