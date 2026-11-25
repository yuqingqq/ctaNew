# Keep a quote when it falls behind the best price

Frozen development reading: **reject**. The two arms cover identical tapes and settlement outcomes for 70 complete October 26 markets. The registry also contains 124 markets with missing endpoints and 94 with stale books. The 288-market date is consumed. Only the distance-alone cancellation was disabled; all other ways of cancelling and all risk and order limits operated as before.

| Metric | Current ladder | Keep distant quotes | Change |
|---|---:|---:|---:|
| Settlement | −$2,759.33 | +$1,890.96 | +$4,650.29 |
| Completed FIFO turnover | −$2,302.71 | +$5,937.92 | +$8,240.63 |
| Residual settlement P&L | −$456.62 | −$4,046.95 | −$3,590.33 |
| Completed cents per matched share | -1.189 | +7.003 | +8.192 |
| Matched shares | 193,654 | 84,789 | -108,865 |
| Mean absolute ending inventory (shares) | 41.9 | 186.2 | 144.3 |
| Submissions | 25,746 | 12,434 | -13,312 |
| Cancellation requests | 23,617 | 11,185 | -12,432 |
| Both sides active (fraction) | 0.617 | 0.538 | -0.079 |
| Same-price opening cycles | +$192.97 | +$2,453.11 | +$2,260.14 |
| Through-price opening cycles | −$2,495.68 | +$3,484.81 | +$5,980.49 |
| Same-price opening matched shares | 70,045.2 | 36,950.4 | -33,094.8 |
| Through-price opening matched shares | 123,608.7 | 47,838.8 | -75,769.9 |
| Same-price opening cents per matched share | +0.275 | +6.639 | +6.363 |
| Through-price opening cents per matched share | -2.019 | +7.284 | +9.303 |

Positive settlement markets: 26/70 before and 42/70 after. Within 10 shares of flat: 51/70 before and 30/70 after.

Pulls per full market-minute went from 67.9 to 32.2. Baseline requests solely for distance were 7,900; the alternative had 0, and 2,989 distinct generations reached the exemption at least once. A kept order could still receive a same-tick pressure or inventory request. Pressure pulls went from 5,794 to 5,977.

Paired settlement improved in 46 markets, worsened in 24 and tied in 0. Excluding the single largest improvement gives +$4,193.74. The opening trade class is observed when a fill happens. The class-level differences include changes to future orders, inventory and FIFO closes. They do not measure the causal value of capturing a specific queue position.

The disabled arm reconciled against the previous per-market settlement, positions, fills, turnover, participation and pull counts, plus both opening classes in the previous cycle attribution. The per-tick guard and accounting tests include a falsifier. The gate and candidate were committed before the replay. No other settings were varied, no untouched dates were used and no rebate was credited. External order flow remains fixed.

[Predeclared test](protocol.md) · [Market rows](markets.csv) · [Computed decision](result.json)
