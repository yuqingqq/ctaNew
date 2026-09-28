# QNT, 2026-09-22 → 09-28 — case study

Every number below is produced by `research/qnt_case_study.py` →
`data/case_qnt/summary.json` (as-of 2026-09-28 09:42 UTC) unless marked
*(hourly table)*, which comes from the same cached 1m data resampled to 1h.
Detector timings are in §5, from `research/replay.py`.

## 1. What happened

| | |
|---|---|
| Pre-move close (09-21) | **$66.87** |
| Catalyst | 09-24: The Clearing House (bank-owned, ~25 US banks) selected Quant for its On-Chain Money Initiative; UK Finance confirmed live tokenised-sterling payments on Quant the same day ([cryptonomist 09-25](https://en.cryptonomist.ch/2026/09/25/quant-qnt-price-surge/), [cryptonomist 09-28](https://en.cryptonomist.ch/2026/09/28/quant-price-surge-banks/)) |
| Peak | **$374.50** high, 2026-09-27 **22:56 UTC** (5.6× the pre-move close) |
| As-of | $205.90 at 09-28 09:42 UTC (−45% from the peak) |

Phases *(hourly table, Binance spot close; volume multiple = Binance USDM hourly quote
volume ÷ the 09-20..21 median hour of $0.217M)*:

| Phase (UTC) | Price | Perp vol × | Notes |
|---|---|---:|---|
| 09-22 12:00–13:00 | 66.6 → 73.3 | 7–8× | pre-news pop, faded |
| 09-24 12:00–16:00 | 72.3 → 82.5 (+14%) | 8–13× | news day; a steady grind, no 1m bar above +1.3% |
| 09-25 03:00–07:00 | 89.6 → 102.1 | 37–175× | OI $4.0M (09-24) → $12.4M; top-trader L/S 2.15 → 0.95 |
| 09-26 14:00–17:00 | 106 → 124 | 31–104× | |
| 09-26 22:00 → 09-27 01:00 | 122 → 182 (+49% in 4h) | 238–648× | ignition 22:12 |
| 09-27 21:00–22:59 | 197 → **374.5** | 908–1632× | blow-off; last leg 255 → 373 in ~16 min |
| 09-28 | 344 → 206 | | unwind; OI $56.6M peak (22:50) → $43.1M by 23:55 |

## 2. Where it traded

Quote volume 09-20..09-27 ($M): Binance USDM **2,932.6**, Bybit linear 564.4, Bitget
perp 319.3, Binance spot 266.6, Coinbase 184.0, KuCoin 29.4, MEXC 17.8, Bitget spot
12.1. **Perps were 88.2% of volume.** Not listed on Upbit, Bithumb or Hyperliquid.

**Same ticker ≠ same asset.** OKX `QNT-USDT-SWAP` is an equity perp (`instCategory`
3) trading ~$48 through the whole event — it is not Quant. Any multi-venue system
keyed on tickers would have mixed it in. movewatch filters OKX to `instCategory 1`
and splits every ticker into price clusters at runtime (`store.refresh_consistency`).

## 3. Who led

Correlation of 1m log returns with Binance USDM, 09-26 12:00 → 09-27 24:00:

| venue | lag 0 | venue leads 1m | venue lags 1m |
|---|---:|---:|---:|
| Binance spot | 0.996 | 0.027 | 0.029 |
| Bybit linear | 0.996 | 0.026 | 0.019 |
| others (Bitget, Coinbase, KuCoin, MEXC) | 0.951–0.983 | ≤ 0.036 | ≤ 0.042 |

**At 1-minute resolution every venue moved together; no venue led.** Lead-lag, if
any, is sub-minute and needs tick timestamps (not measured here). For detection,
this means venue count is a confirmation signal, not an early signal, for a
large-cap move like this one.

## 4. The squeeze signature (descriptive, not predictive)

- **Open interest** $4.0M (09-24 00:00) → $12.4M (09-25 07:00) → **$56.6M** (09-27
  22:50), then −24% in an hour as the top formed.
- **Top-trader long/short** (Binance) fell 2.15 → 0.95 while price rose: positioning
  leaned short into the rally.
- **Funding** (Bitget) turned negative exactly at the legs: −0.0344% (09-25 08:00),
  −0.0193% (09-26 16:00), **−0.0782%** (09-27 00:00). Shorts were paying.
- **Basis** (Binance perp vs spot, daily mean) stayed negative −12.8 → −4.6 bps: the
  perp traded *below* spot, so spot demand was leading and shorts sat on the perp.

That is a textbook short squeeze on top of a real news catalyst. It is recorded here
as context for alert messages (OI / funding lines are on the roadmap), **not** as a
validated predictor: one event is an anecdote, and CLAUDE.md's prior work found
positioning/funding signals thin and era-unstable as alpha.

## 5. When would movewatch have fired?

Replay of the live logic (v4) on Binance USDM minute closes (live evaluates every second,
so live can only be earlier). QNT sits in a window seen during development: these rows
illustrate the presets, they are not out-of-sample evidence (`docs/CALIBRATION.md` 2e).

| preset (Binance-USDM msgs/day) | first alert | the 09-27 blow-off (top $374.50, 22:56) |
|---|---|---|
| strict ≤40 (live default) | 09-27 01:24, $177.55 (+45.8% / 4h) | NEW 22:55 $308.82 → +32% 22:58 → reversal 23:03 |
| ≤80 | 09-26 23:08, $147.67 (+20.4% / 1h) | NEW 21:30 $237.04 → 22:27 / 22:44 / 22:55 → reversal 23:06 |

Neither catches the 09-24 news leg: +13.5% over 4 h at z 5.8 with volume ×13.3 sits below
every 4 h floor. That is the next calibration candidate (a z-weighted path for calm names).
