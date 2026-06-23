# Funding export for the geo-blocked backtest server

`funding_universe.parquet` — Binance USDM funding rates for the live universe (174 syms),
pulled from **FAPI** on this (Binance-reachable) box on **2026-06-23** for transport to the
backtest server, which is geo-blocked from both FAPI and Vision. Vision funding archives are
monthly and not published intra-June, so they can't supply June either — FAPI is the only source.

## Schema (matches the loader cache)
Long-format, columns: `symbol, calc_time (UTC), interval_hours, funding_rate`.
Window **2026-05-01 → 2026-06-23** (June + ~4w trailing buffer for the 7d funding z-score).
44,377 rows · ~320 settlements/sym for the complete ones.

## Coverage
- **156 / 174 complete through 2026-06-23** — includes every actively-traded / picked symbol.
- **18 syms partial (through 2026-06-06)**: SPX, STABLE, STBL, STRK, TIA, TON, TURBO, USUAL,
  VIRTUAL, VVV, WIF, WLD, WLFI, W, XPL, ZEC, ZEN, ZK. These 403'd in a FAPI rate-limit ban.
  They are scored-but-**never-traded** illiquid names (0 picks across the whole live run); their
  June funding feature forward-fills from 06-06 and does not affect any traded result — and the
  liquidity-filter test excludes them anyway. Re-run `scripts/fetch_funding_fapi.py` once the ban
  clears to top them up (it overwrites only the symbols it successfully fetches).

## Use on the backtest server
```bash
python scripts/unpack_funding_export.py    # merges into data/ml/cache/funding_{sym}.parquet
```
This UPSERTS (concat + dedup on `calc_time`), so any pre-existing history is preserved and June is
added. Afterwards `data_collectors.funding_rate_loader.load_funding_rate()` serves it from cache
with no Binance fetch.
