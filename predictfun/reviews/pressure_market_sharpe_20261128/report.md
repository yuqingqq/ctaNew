# Five-minute market Sharpe and the cost of entering a tight book

November 28, 2026 UTC. The unit is the five-minute market. Each eligible market has equal weight, including any that settle at zero. Dollar P&L at constant allocated capital gives the same Sharpe as returns on that capital. Values are unannualized, using a zero benchmark and sample standard deviation.

**Decision.** The single candidate failed the prespecified development screen.

# Population and strategy

The optimization compares 550 paired markets out of 2,304 scheduled on November 14–18 and 24–26. Status counts were 1,332 missing_endpoint, 550 complete, 421 stale_future_book, 1 missing_future_book. Missing markets have unknown P&L.

Both strategies use the existing two-level passive policy, up to 50 shares per quote, its 150-share inventory cap with expiry taper, the same fixed historical order flow, modeled queue and event-driven adverse-pressure cancellations. The historical clock assumes 4 ms for the Binance feed, 58 ms for placement, 37 ms for live cancellation and a 50 ms posting grid. Reported USD assume zero passive fees, rebates and funding.

At a one-tick received spread, the candidate posts at the touch only up to known reducing inventory left after same-side reservations. It can still add exposure at the second level. A resting order runs its original management logic. The rule was frozen before replay. Both recent date blocks had already been examined before this research choice.

# Whole-market results

| Date or block | Markets | Current USD | Candidate USD | Current 5m Sharpe | Candidate 5m Sharpe | Current / candidate worst-decile mean USD |
|---|---:|---:|---:|---:|---:|---:|
| 20261114 | 65 | +20.56 | +53.16 | 0.0082 | 0.0260 | -69.72 / -53.78 |
| 20261115 | 62 | -162.66 | -230.77 | -0.0847 | -0.1315 | -59.13 / -47.88 |
| 20261116 | 66 | +81.07 | -103.40 | 0.0329 | -0.0498 | -55.07 / -55.06 |
| 20261117 | 70 | +58.68 | +69.07 | 0.0216 | 0.0271 | -75.23 / -66.09 |
| 20261118 | 69 | +255.18 | +298.56 | 0.1154 | 0.1665 | -42.64 / -40.14 |
| 20261124 | 73 | +34.06 | -187.64 | 0.0131 | -0.0781 | -62.99 / -57.52 |
| 20261125 | 62 | -177.32 | -116.33 | -0.0716 | -0.0591 | -84.39 / -59.83 |
| 20261126 | 83 | +341.87 | +262.43 | 0.1143 | 0.0987 | -56.64 / -51.27 |
| early | 332 | +252.83 | +86.61 | 0.0214 | 0.0084 | -62.12 / -53.92 |
| late | 218 | +198.61 | -41.54 | 0.0246 | -0.0059 | -69.54 / -57.61 |
| pooled | 550 | +451.44 | +45.07 | 0.0227 | 0.0026 | -65.45 / -55.69 |

The candidate improved 276 markets and worsened 274. Excluding the best improvement date (20261125), the P&L difference was -467.36 USD. The descriptive 95% paired day-bootstrap range for the market Sharpe difference was -0.0531 to +0.0106 using 4,096 draws. It has only eight day clusters. The unit for the point estimate remains the individual market.

The gate outcomes were early False, late False, pooled False, worst_decile True, without_best False.

Across all 22 previously tested dates the current strategy had market Sharpe 0.0753 and USD +4,083.47 over 1,591 eligible markets. Those dates include strategy development. The 50 ms grid comparator had 0.0390. On the recent 550 markets the grid comparator had -0.0034.

# Loss attribution

Across the 274 losing markets the current strategy settled -7,552.23 USD. Completed FIFO buy/sell pairs accounted for -7,295.67 USD and remaining inventory -256.56 USD. 269 of these markets already had negative completed turnover. Across all markets, profitable completed shares gained 10.64 cents per share and losing completed shares lost 13.24 cents per share.

| Strategy and opening cohort | Matched shares | Missing-mid shares | Entry edge USD | Mid move during holding USD | Exit edge USD | Missing-mid net USD | Completed USD | Mean holding s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current / all | 314,803.6 | 556.8 | +2,354.63 | -3,847.59 | +2,212.75 | -81.53 | +638.26 | 34.7 |
| current / losing markets | 147,356.0 | 374.2 | +1,046.47 | -9,281.42 | +992.14 | -52.86 | -7,295.67 | 38.1 |
| current / losing FIFO pairs | 132,323.8 | 489.0 | +968.15 | -19,251.14 | +845.01 | -87.96 | -17,525.95 | 42.4 |
| current / won FIFO pairs | 170,759.2 | 62.7 | +1,304.21 | +15,557.65 | +1,295.92 | +6.43 | +18,164.21 | 29.5 |
| current / one-tick touch | 193,177.4 | 306.1 | +1,211.90 | -1,455.89 | +1,318.50 | -31.15 | +1,043.36 | 36.0 |
| current / other opening quotes | 121,626.2 | 250.6 | +1,142.73 | -2,391.70 | +894.24 | -50.38 | -405.10 | 32.5 |
| candidate / all | 256,547.8 | 628.6 | +2,574.78 | -4,082.54 | +1,844.36 | -90.24 | +246.35 | 33.5 |
| candidate / losing markets | 123,905.3 | 483.1 | +1,225.70 | -8,589.60 | +833.58 | -69.40 | -6,599.72 | 36.2 |
| candidate / losing FIFO pairs | 110,303.3 | 496.2 | +1,085.72 | -16,315.37 | +702.42 | -97.74 | -14,624.97 | 41.2 |
| candidate / won FIFO pairs | 137,188.9 | 108.1 | +1,398.17 | +12,381.20 | +1,084.46 | +7.49 | +14,871.33 | 28.0 |
| candidate / one-tick touch | 0.0 | 0.0 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | — |
| candidate / other opening quotes | 256,547.8 | 628.6 | +2,574.78 | -4,082.54 | +1,844.36 | -90.24 | +246.35 | 33.5 |

Edge uses the most recent valid received book no older than one second at each fill. Signed entry edge plus midpoint movement plus exit edge equals the realized closed profit. Cohorts are descriptive and overlap across different breakdowns. The one-tick group is defined at the opening fill’s original quote proposal. Some orders can change their inventory role before execution.

| Pooled measure | Current | Candidate |
|---|---:|---:|
| Mean USD per market | 0.821 | 0.082 |
| SD USD per market | 36.118 | 31.399 |
| Winning markets | 276.000 | 275.000 |
| Flat markets | 0.000 | 0.000 |
| Gross matched gains USD | 18,164.208 | 14,871.325 |
| Gross matched losses USD | -17,525.947 | -14,624.971 |
| Completed FIFO USD | 638.261 | 246.355 |
| Residual at payout USD | -186.820 | -201.286 |
| Filled shares | 632,305.212 | 515,061.842 |
| Orders submitted | 142,214.000 | 137,033.000 |
| Cancel requests | 127,481.000 | 125,206.000 |
| Mean absolute settlement inventory in shares | 4.905 | 3.575 |
| One-second markout in cents per valued share | -1.067 | -1.159 |

The additional rule withheld 357,575 proposed placements and reduced 12,594 sizes. A repeated withheld proposal can refer to the same empty slot.

Every eligible market reproduced the archived control table, including the two policies, fill values, and off-grid diagnostics. Both trial arms reconciled settlement against signed fills and against matched plus residual value. Each FIFO decomposition reproduced its matched ledger. Fixed external flow and simulated queues limit what this establishes about live execution.

The negative dates break down as follows. 20261115 completed -111.41 USD and residual -51.25 USD, 20261125 completed -223.98 USD and residual +46.66 USD.

Across all eight dates the change in gross winning turnover was -3,292.88 USD, the change in gross losing turnover was +2,900.98 USD and the change in residual was -14.47 USD. These three quantities sum to the settlement difference.

The baseline net settlement edge averaged 0.071 cents per executed share before any unmodeled cost.

## Implication for the next test

On the original policy, pairs opened by a best-price quote submitted at a one-tick spread earned +1,043.36 USD, positive on 7 of 8 dates. All other opening quotes contributed -405.10 USD. The per-date breakdown is a descriptive follow-up. These groups differ in market state and subsequent fills.

Keep the existing best-price entry rule. The next diagnostic should separate second-level openings from best-price openings at wider spreads and measure subsequent inventory movement for each. Use the latest public trades available at the placement decision to distinguish a sweep toward the quote from ordinary flow. This tests whether the losing exposure could be recognized early enough to reduce new inventory while preserving profitable quotes and exits.

[Frozen protocol](protocol.md) · [All paired markets](paired_markets.csv) · [Scores and artifact hashes](result.json) · [Five-minute Sharpe for all 22 dates](historical_sharpe.json) · [FIFO attribution](fifo_attribution.csv)
