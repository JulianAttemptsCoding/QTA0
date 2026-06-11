# Data Sources (PLAN.md §1)

| # | Source | Provides | Access | Cost | Status |
|---|--------|----------|--------|------|--------|
| 1 | **Alpaca Market Data API** (required) | Daily/minute SIP OHLCV bars (~2016→), auction prices, corporate actions (~2020→), assets master (active/inactive=delisted seed), calendar | `.env`: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`. Free research plan (SIP >15min delayed). | Free (research); ~$99/mo live SIP later | wired (`ingest/alpaca_*`) |
| 2 | **Ken French Data Library** (required) | FF5 daily, Momentum (UMD), ST Reversal (STR) — attribution gate | auto-download (`make data-factors`) | Free | wired (`ingest/factors_french`) |
| 3 | **Wikipedia S&P 500 revision history** (required) | PIT large-cap universe membership by date | MediaWiki revisions API (`make data-universe`) | Free | wired (`ingest/wiki_constituents`) |
| 4 | **SEC EDGAR** (required) | `company_tickers.json` for ticker/CIK + symbol-change reconciliation | compliant `SEC_USER_AGENT` | Free | wired (`ingest/edgar_xcheck`) |
| 5 | **Stooq bulk US daily** (optional) | pre-2016 daily bars, diagnostics-only (quarantined) | manual download → `ingest/stooq_diag.ingest_local_zip` | Free | wired (quarantine-isolated) |
| 6 | **Norgate / Sharadar SEP** (optional upgrade) | true survivorship-bias-free prices incl. delisting returns | norgatedata.com / data.nasdaq.com | ~$30–60/mo | not wired |

## Credentials (`.env`, never committed)

```
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
ALPACA_DATA_URL=https://data.alpaca.markets
SEC_USER_AGENT=Name email
FRED_API_KEY=...            # optional RF cross-check
WIKI_USER_AGENT=desc email
```

## Provenance contracts

- **SIP-only**: every curated feature/label row must carry `feed=sip`
  (`contracts.assert_feed_sip`); IEX bars entering `curated/` fail CI.
- **Quarantine isolation**: Stooq data writes ONLY to `data/diagnostic_quarantine/` with no
  import path into curated/features (enforced by `tests/test_imports_and_isolation.py`).
- **Holdout**: final 18 months are filesystem-locked (`.LOCKED` + loader check) until the
  Phase-13 pre-registration protocol unlocks them once.
