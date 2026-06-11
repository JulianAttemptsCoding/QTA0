"""Ken French Data Library (§3.4): FF5 daily, Momentum daily, ST Reversal daily.

Output: factors_daily(date, mkt_rf, smb, hml, rmw, cma, rf, umd, str). Values are decimals
(French publishes percent; divided by 100 here).
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from ..common.config import CURATED
from ..common.io import write_parquet
from . import raw_dir

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
    "str": "F-F_ST_Reversal_Factor_daily_CSV.zip",
}


def _download_csv(zip_name: str) -> bytes:
    url = f"{_BASE}/{zip_name}"
    resp = requests.get(url, timeout=120, headers={"User-Agent": "tactic-mob-research"})
    resp.raise_for_status()
    (raw_dir("factors") / zip_name).write_bytes(resp.content)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        return z.read(name)


def _parse_french(csv_bytes: bytes, value_cols: list[str]) -> pd.DataFrame:
    text = csv_bytes.decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
            try:
                vals = [float(x) for x in parts[1:1 + len(value_cols)]]
            except ValueError:
                continue
            rows.append([parts[0]] + vals)
    df = pd.DataFrame(rows, columns=["date"] + value_cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
    for c in value_cols:
        df[c] = df[c] / 100.0
    return df


def ingest() -> pd.DataFrame:
    ff5 = _parse_french(_download_csv(_FILES["ff5"]),
                        ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    mom = _parse_french(_download_csv(_FILES["mom"]), ["umd"])
    strv = _parse_french(_download_csv(_FILES["str"]), ["str"])
    df = ff5.merge(mom, on="date", how="left").merge(strv, on="date", how="left")
    write_parquet(df, CURATED / "factors_daily.parquet")
    return df


if __name__ == "__main__":
    print(ingest().tail())
