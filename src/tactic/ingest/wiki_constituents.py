"""Wikipedia S&P 500 PIT membership (§3.3a) via MediaWiki revisions API.

Fetch every revision of "List of S&P 500 companies", parse the constituents table per
revision, forward-fill membership between revisions. Output:
sp500_membership(symbol, date, member).
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import requests

from ..common.config import CURATED, load_dotenv
from ..common.io import write_parquet
from . import raw_dir

load_dotenv()
_API = "https://en.wikipedia.org/w/api.php"
_PAGE = "List of S&P 500 companies"
_UA = os.environ.get("WIKI_USER_AGENT", "tactic-mob-research example@example.com")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    return s


def list_revisions(sess: requests.Session, max_rev: int = 2000) -> list[dict]:
    revs: list[dict] = []
    cont: dict = {}
    while len(revs) < max_rev:
        params = {
            "action": "query", "prop": "revisions", "titles": _PAGE,
            "rvprop": "ids|timestamp", "rvlimit": "500", "rvdir": "newer",
            "format": "json", **cont,
        }
        data = sess.get(_API, params=params, timeout=60).json()
        pages = data["query"]["pages"]
        for page in pages.values():
            revs.extend(page.get("revisions", []))
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    return revs


def revision_symbols(sess: requests.Session, revid: int) -> set[str]:
    params = {"action": "parse", "oldid": revid, "prop": "wikitext", "format": "json"}
    try:
        data = sess.get(_API, params=params, timeout=60).json()
        text = data["parse"]["wikitext"]["*"]
    except Exception:
        return set()
    syms: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        # constituent rows reference the NYSE/NASDAQ ticker via [[NYSE:XXX]] or a symbol cell
        if line.startswith("|") and ("NYSE" in line or "NASDAQ" in line or "[[" in line):
            for tok in line.replace("|", " ").split():
                t = tok.strip("[]| ")
                if t.isupper() and 1 <= len(t) <= 5 and t.isalpha():
                    syms.add(t)
    return syms


def ingest(sample_every: int = 1, max_rev: int = 1500) -> pd.DataFrame:
    """Build PIT membership. `sample_every` subsamples revisions to bound API calls."""
    sess = _session()
    revs = list_revisions(sess, max_rev=max_rev)
    snapshots: list[tuple[dt.date, set[str]]] = []
    for i, rev in enumerate(revs):
        if i % sample_every:
            continue
        d = pd.to_datetime(rev["timestamp"]).date()
        syms = revision_symbols(sess, rev["revid"])
        if syms:
            snapshots.append((d, syms))
    if not snapshots:
        raise RuntimeError("no S&P 500 revisions parsed")
    # forward-fill daily membership
    rows = []
    snapshots.sort()
    all_dates = pd.bdate_range(snapshots[0][0], dt.date.today())
    si = 0
    cur = snapshots[0][1]
    for day in all_dates:
        while si + 1 < len(snapshots) and snapshots[si + 1][0] <= day.date():
            si += 1
            cur = snapshots[si][1]
        for sym in cur:
            rows.append({"symbol": sym, "date": day.date(), "member": True})
    df = pd.DataFrame(rows)
    write_parquet(df, raw_dir("wiki") / "sp500_snapshots.parquet")
    write_parquet(df, CURATED / "sp500_membership.parquet")
    return df
