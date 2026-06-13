# Orthogonal data sources (registry)

External data approved for the optimization loop, beyond the baseline free price/funding. Research
may propose features built from these; Implementation fetches/integrates; Review audits PIT alignment
(an external series must be lagged so a feature at decision-time t uses only data published by t).

## Approved / available
| source | what | access | status |
|---|---|---|---|
| **Deribit DVOL** | BTC & ETH implied-vol index (forward 30d IV), hourly | public API, free, no auth | **APPROVED 2026-05-25** |
| Deribit options | BTC/ETH IV surface, 25Δ skew/risk-reversal, term structure | public API, free | available (heavier fetch) |
| Binance funding term-structure | funding curve slope / cross-exchange basis | free (have) | available |
| **Binance METRICS (positioning)** | open interest + long/short ratios + top-trader + taker buy/sell, 5-min, back to 2021 | free Vision archive (metrics_loader.py) | **APPROVED+FETCHING 2026-05-25 (iter-009): the LEADING-signal candidate — crowded-long positioning builds BEFORE the deleverage unwind, unlike coincident price/vol** |

## Not available (need human-provided key)
| source | what | why blocked |
|---|---|---|
| Glassnode | on-chain (exchange flows, SOPR, stablecoin) | paid; no API key in env |
| Coinglass | aggregated liquidations | paid |

## Why Deribit DVOL/skew is the right first orthogonal bet
The strategy's unsolved problem is a regime-driven drawdown that **price/funding features could not lead**
(iter-001/002/004: all coincident/lagging). Implied-vol level/skew is a *forward-looking* market signal
(option-implied 30d vol, crash-fear via put skew) — a genuine LEADING regime/crowding indicator. It is
market-wide (BTC/ETH), so it conditions the whole book (a regime overlay), not a per-alt rank.

## PIT rules for external data (Review enforces)
- Align the external series to the 4h decision grid by **backward as-of merge** (use the last value
  published at or before the decision time); never forward-fill from the future.
- Lag by at least one bar if there's any publish-latency ambiguity.
- Features = level, expanding-percentile rank (PIT, not full-sample), trailing change/slope.
- A new external feature with IC > +0.10 is a look-ahead red flag — investigate.

## Fetched datasets (cache)
| file | content | window | fetched |
|---|---|---|---|
| `research/convexity_portable_2026-05-20/results/_cache/deribit_dvol.parquet` | BTC+ETH DVOL hourly + 4h-grid features | (see fetch) | iter-005 |
