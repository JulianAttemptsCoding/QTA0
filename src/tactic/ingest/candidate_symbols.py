"""Candidate symbol list (§3.3): union of PIT S&P 500 + liquid ETFs + delisted-recovery set."""
from __future__ import annotations

import pandas as pd

from ..common.config import CURATED

# §3.3b fixed liquid-ETF list; leveraged/inverse explicitly excluded.
ETFS = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# Explicit ≥150-name liquid breadth set (large/mid-cap US, broad sector coverage). No PIT
# membership dependency — used directly for the wide-universe ingest. SPY included for benchmark.
LIQUID_BIG = sorted(set(ETFS + [
    # Tech / semis / software
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "AMD", "INTC",
    "TXN", "QCOM", "IBM", "INTU", "NOW", "AMAT", "MU", "LRCX", "KLAC", "ADI", "PANW",
    "SNPS", "CDNS", "ANET", "FTNT", "MCHP", "NXPI", "ON", "MRVL",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "WBD",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "ORLY", "CMG",
    "MAR", "GM", "F", "YUM", "ROST", "AZO", "LULU",
    # Consumer staples
    "PG", "KO", "PEP", "COST", "WMT", "MO", "PM", "MDLZ", "CL", "TGT", "KMB", "GIS",
    "KHC", "STZ", "SYY", "KR",
    # Health care
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN", "BMY", "CVS",
    "MDT", "ISRG", "GILD", "VRTX", "ELV", "REGN", "CI", "HCA", "HUM", "ZTS", "BSX",
    "SYK", "BDX",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C", "SCHW", "BLK",
    "SPGI", "CB", "PGR", "MMC", "PNC", "USB", "TFC", "COF", "AIG", "MET", "AFL", "TRV",
    # Industrials
    "GE", "CAT", "BA", "HON", "UNP", "UPS", "RTX", "LMT", "DE", "ADP", "GD", "NOC",
    "MMM", "EMR", "ETN", "ITW", "CSX", "FDX", "NSC", "WM", "PH", "TDG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB", "KMI",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
    # Real estate
    "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA",
]))


def build(include_inactive: bool = True) -> list[str]:
    syms: set[str] = set(ETFS)
    memb = CURATED / "sp500_membership.parquet"
    if memb.exists():
        syms |= set(pd.read_parquet(memb)["symbol"].unique())
    if include_inactive:
        snap = CURATED / "assets_snapshots.parquet"
        if snap.exists():
            df = pd.read_parquet(snap)
            inactive = df[df["status_query"] == "inactive"]["symbol"].dropna().unique()
            # §3.3c: inactive that ever matched (a) — intersect with seen members if available
            members = set(pd.read_parquet(memb)["symbol"].unique()) if memb.exists() else set()
            syms |= ({s for s in inactive if s in members} if members else set(inactive))
    return sorted(syms)
