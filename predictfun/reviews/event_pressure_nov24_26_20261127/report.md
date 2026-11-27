# Event-driven pressure cancels on November 24–26

Receipt prepared 2026-11-27 16:50 UTC. Three additional complete UTC dates selected by file availability before scoring. November 24–25 neutral books also supplied a separate study. Tokyo had previously examined November 26 public trade windows. One candidate, using modeled fills and zero passive fees or rebates.

| UTC day | Eligible / scheduled | Other statuses |
|---|---:|---|
| 20261124 | 73 / 288 | missing_endpoint 159, stale_future_book 56 |
| 20261125 | 62 / 288 | missing_endpoint 158, stale_future_book 68 |
| 20261126 | 83 / 288 | missing_endpoint 148, stale_future_book 57 |

| UTC day | Grid settlement, USD | Event settlement, USD | Difference, USD | Grid / event +1 s, cents/share |
|---|---:|---:|---:|---:|
| 20261124 | -8.29 | +34.06 | +42.35 | -1.085 / -0.987 |
| 20261125 | -285.58 | -177.32 | +108.26 | -1.008 / -0.947 |
| 20261126 | +240.50 | +341.87 | +101.38 | -1.448 / -1.392 |

| Measure | Grid | Event |
|---|---:|---:|
| Settlement, USD | -53.376 | 198.611 |
| Completed FIFO profit, USD | 132.431 | 359.412 |
| Residual at payout, USD | -185.807 | -160.801 |
| One-second markout, cents per valued share | -1.192 | -1.124 |
| Gross adverse +1 s, USD | -4,547.662 | -4,221.806 |
| Gross favorable +1 s, USD | 1,485.352 | 1,458.290 |
| Shares lacking a +1 s mark | 170.627 | 186.848 |
| Filled shares | 257,021.183 | 246,111.374 |
| Mean absolute inventory at payout, shares | 6.204 | 6.356 |
| Cancel requests | 56,166.000 | 57,213.000 |
| Reposts | 48,302.000 | 49,335.000 |

Event minus grid settlement was +251.99 USD. Excluding its best improvement day, the difference was +143.72 USD. The event arm improved 3/3 days and settled positive on 2/3. The directional screen passed and the absolute screen failed.

The two arms generated separate order and inventory paths. The historical feed delay was 4 ms, the nominal venue cancel effect was 37 ms, and placement and reposts remained on the 50 ms grid. The live wallet fill rate and venue removal time remain unresolved.

Event settlement after removing its best day was -143.26 USD and after removing its best market was +98.30 USD.

## Eight archived dates

November 14–18 and 24–26 cover 550 eligible markets. Grid settlement totaled -67.84 USD and event-driven settlement +451.44 USD. The event strategy earned positive settlement on 6/8 dates and beat the grid on 6/8. Removing its best day gives +109.57 USD. These two separately declared archive blocks are pooled for context.
