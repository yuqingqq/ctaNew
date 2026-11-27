# Event-driven pressure cancels on November 14–18

Receipt prepared 2026-11-27 15:19 UTC. Five archived UTC days released before scoring, earlier than the November 19–23 development results. One candidate, using modeled fills and zero passive fees or rebates.

| UTC day | Eligible / scheduled | Other statuses |
|---|---:|---|
| 20261114 | 65 / 288 | missing_endpoint 192, stale_future_book 31 |
| 20261115 | 62 / 288 | missing_endpoint 178, stale_future_book 48 |
| 20261116 | 66 / 288 | missing_endpoint 179, stale_future_book 43 |
| 20261117 | 70 / 288 | stale_future_book 65, missing_endpoint 153 |
| 20261118 | 69 / 288 | stale_future_book 53, missing_endpoint 165, missing_future_book 1 |

| UTC day | Grid settlement, USD | Event settlement, USD | Difference, USD | Grid / event +1 s, cents/share |
|---|---:|---:|---:|---:|
| 20261114 | -4.15 | +20.56 | +24.71 | -0.972 / -0.917 |
| 20261115 | -321.68 | -162.66 | +159.02 | -1.000 / -0.935 |
| 20261116 | -59.07 | +81.07 | +140.13 | -1.106 / -1.077 |
| 20261117 | +69.83 | +58.68 | -11.15 | -0.971 / -0.951 |
| 20261118 | +300.59 | +255.18 | -45.42 | -1.438 / -1.333 |

| Measure | Grid | Event |
|---|---:|---:|
| Settlement, USD | -14.462 | 252.830 |
| Completed FIFO profit, USD | 20.916 | 278.849 |
| Residual at payout, USD | -35.378 | -26.019 |
| One-second markout, cents per valued share | -1.083 | -1.031 |
| Gross adverse +1 s, USD | -7,101.231 | -6,600.044 |
| Gross favorable +1 s, USD | 2,760.567 | 2,623.154 |
| Shares lacking a +1 s mark | 538.035 | 491.264 |
| Filled shares | 401,496.630 | 386,193.838 |
| Mean absolute inventory at payout, shares | 3.739 | 3.952 |
| Cancel requests | 69,598.000 | 70,268.000 |
| Reposts | 59,261.000 | 59,903.000 |

Event minus grid settlement was +267.29 USD. Excluding its best improvement day, the difference was +108.27 USD. The event arm improved 3/5 days and settled positive on 4/5. The directional screen passed and the absolute screen failed.

The two arms generated separate order and inventory paths. The historical feed delay was 4 ms, the nominal venue cancel effect was 37 ms, and placement and reposts remained on the 50 ms grid. The live wallet fill rate and venue removal time remain unresolved.

Event settlement after removing its best day was -2.35 USD and after removing its best market was +144.36 USD.
