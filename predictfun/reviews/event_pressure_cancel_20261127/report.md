# Checking pressure between 50 ms decisions

November 19–23, 2026 UTC. Both policies ran on 352 complete markets from the previously consumed five-day block. Every control reproduced its archived nominal ledger. This replay used one additional candidate.

## Paired outcome

| Date UTC | Markets | Grid settlement | Event settlement | Difference | Off-grid cancel requests |
|---|---:|---:|---:|---:|---:|
| 20261119 | 76 | −$101.70 | −$2.17 | $99.53 | 13,120 |
| 20261120 | 59 | $132.21 | $233.10 | $100.89 | 15,212 |
| 20261121 | 66 | −$503.07 | −$438.27 | $64.81 | 13,582 |
| 20261122 | 77 | $818.97 | $930.75 | $111.77 | 9,387 |
| 20261123 | 74 | −$33.79 | $134.93 | $168.72 | 15,143 |
| Pooled | 352 | $312.61 | $858.34 | $545.73 | 66,444 |

The event policy improved settlement on 5 of 5 dates. Its pooled difference after dropping the largest daily improvement was $377.00. The predeclared screen for a further shadow test **passed**.

| Metric | Grid | Event |
|---|---:|---:|
| Completed FIFO turnover P&L | $551.42 | $1,161.92 |
| Residual inventory P&L at payout | −$238.81 | −$303.58 |
| Matched shares | 206,870 | 198,745 |
| Mean absolute ending inventory, shares | 7.60 | 8.01 |
| Submissions | 101,189 | 102,926 |
| Cancel requests, all reasons | 91,024 | 92,731 |
| Same-lane submissions following a cancel | 78,592 | 80,258 |
| Filled shares | 416,416 | 400,310 |
| Shares matched after a cancel request | 39,907 | 35,867 |
| Eligible +1 s markout, cents per share | -1.065 | -0.989 |

The event strategy itself earned a positive settlement on 3 of five days. Removing its best day leaves −$72.41 across the other four.

The 66,444 off-grid requests were issued a mean 25.6 ms before the next grid tick. A transient signal may have cleared by that tick, and later orders can diverge.

## Fill differences at one second

Quantity is matched between arms by public trade, side and fill price. The values below include changes in later orders after the policies diverge.

| Fill quantity present only in | Favorable shares | Favorable value | Adverse shares | Adverse value | Flat shares | Unvalued shares |
|---|---:|---:|---:|---:|---:|---:|
| Grid arm | 10,847 | $236.12 | 28,066 | −$897.35 | 117 | 79 |
| Event arm | 8,724 | $211.93 | 14,201 | −$397.69 | 54 | 25 |

The venue can match until 37 ms after a live cancel request in both arms. Prices, including size-only update opportunities, use the frozen model of delivery at Binance event time plus 4 ms with one observable book per event millisecond. The public trade path is fixed and private fills are simulated. A wallet or live Tokyo latency measurement requires separate evidence.

The 1,440 scheduled markets included 872 missing endpoints, 215 stale books and one missing future book. Economics for those markets are unknown. Dates were previously used in the November holdout. The 4,096 day-level sign flips yield only 32 distinct sign patterns.

[Frozen protocol](protocol.md) · [All metrics and per-market file hashes](result.json)
