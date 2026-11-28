# Does Predict.fun confirmation improve Binance-triggered maker cancels?

The confirmed rule passed the frozen development screen. As of November 28, 2026 UTC, the replay covered 218 eligible five-minute markets of 864 scheduled on three consumed dates. The common census excluded 465 missing endpoints and 181 stale books. All P&L is simulated USD at settlement under zero maker fees and rebates.

| UTC | Markets | Current pressure guard | Binance-only overlay | Order-level confirmation | Change vs current |
|---|---:|---:|---:|---:|---:|
| 20261124 | 73 | +34.06 | +210.56 | +146.58 | +112.52 |
| 20261125 | 62 | -177.32 | -124.21 | -215.67 | -38.36 |
| 20261126 | 83 | +341.87 | +85.71 | +397.92 | +56.05 |
| pooled | 218 | +198.61 | +172.06 | +328.83 | +130.21 |

| Measure | Current | Binance only | Confirmed |
|---|---:|---:|---:|
| Five-minute market Sharpe | 0.0246 | 0.0223 | 0.0406 |
| Mean P&L in worst decile of markets, USD | -69.54 | -65.71 | -69.37 |
| Completed FIFO closes, USD | 359.41 | 358.43 | 476.96 |
| Residual inventory at payout, USD | -160.80 | -186.37 | -148.13 |
| Cancel requests | 57,213 | 101,510 | 57,849 |
| Reposts | 49,335 | 92,079 | 49,921 |
| Filled shares | 246,111 | 259,387 | 242,248 |
| One-second markout, cents per valued share | -1.124 | -0.970 | -1.116 |

Settlement changed by +130.21 USD versus the current policy. Completed closes contributed +117.55 USD and residual inventory contributed +12.67 USD. The confirmed policy beat the current policy in 88 markets, lost in 67, and tied in 63. Sharpe uses mean P&L divided by its sample standard deviation across five-minute markets, unannualized. The three dates do not support a day-cluster interval.

The extra rule sent 1,759 off-grid and 24 grid pulls. At 1,224 off-grid decisions it pulled one of two simultaneously live orders on the same side. It withheld 194 proposed slots. These action counts are on the candidate path. Total cancellation differences also reflect changed fills, inventory and future orders.

| Incumbent exposure | Shares | Shares without +1 s mark | Gross adverse USD | Gross favorable USD | Net cents per valued share |
|---|---:|---:|---:|---:|---:|
| timely | 7,223.9 | 2.0 | -132.73 | +45.03 | -1.214 |
| remaining | 238,887.5 | 184.8 | -4,089.08 | +1,413.26 | -1.121 |

Timely exposure identifies incumbent fills when the combined conditions preceded matching by at least the modeled 37 ms cancel delay, before an incumbent request. Its value describes selection on that original path. The independently replayed policy includes all downstream fills and inventory.

The Binance alert uses its previous score and modeled 4 ms delivery. Each local public settlementsPending increase is linked to an original book frame and read on the normalized Predict.fun receipt clock. Side-specific net depletion and current displayed depth are required at the same quote price. A settlement increase can be reported on either side. The thin-depth condition is a proxy for queue ahead and can miss an order at the front of a deep level. The original 50 ms posting grid and 58 ms placement delay still apply. All three policies assume unchanged external flow.

Each incumbent market matched the archived tape identity and nine archived order or settlement metrics. Fill quantities and settlement reconciled on each paired path. The evaluator rejected a corrupted adverse-markout subtotal on each day. A separate scalar fixture reproduced the local trigger and detected substitution of venue time for receipt time. The first launch failed during module import, before processing data. A loader-only amendment preceded the completed run.

The frozen checks evaluate settlement PASS, sharpe PASS, two days PASS, without best day PASS, less churn than binance PASS.

[Protocol](protocol.md) · [Paired market data](paired_markets.csv) · [Computed results](results.json)
