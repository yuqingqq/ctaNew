# Cost of withholding fresh inventory late in a market

**Keep the current controller for shadow testing.** The veto reduced average ending inventory on each of the eight dates under both execution assumptions. It also reduced the loss in the worst 10% of markets. Its nominal settlement fell by $1,021.70 across 619 markets, and the nominal gain disappears if the best date is removed. Both execution assumptions returned the predeclared verdict `risk_tradeoff`.

| Measure | Current nominal | New veto nominal | Current short | New veto short |
|---|---:|---:|---:|---:|
| Mean absolute ending shares | 13.35 | 8.47 | 13.86 | 8.95 |
| Mean of the 62 largest ending positions, shares | 98.11 | 72.43 | 99.46 | 75.31 |
| Mean peak position per market, shares | 136.06 | 124.37 | 135.03 | 123.65 |
| Maximum intramarket position, shares | 150 | 150 | 150 | 150 |
| Mean loss of the 62 worst markets | $59.55 | $51.56 | $57.98 | $49.31 |
| Matched shares, counted once per round trip | 270,807 | 227,648 | 252,407 | 211,820 |
| Completed FIFO P&L | +$3,445.47 | +$1,928.02 | +$4,687.66 | +$3,025.90 |
| Leftover inventory P&L at payout | −$1,934.11 | −$1,438.36 | −$2,093.10 | −$1,553.99 |
| **Total settlement P&L** | **+$1,511.36** | **+$489.66** | **+$2,594.56** | **+$1,471.91** |
| Profitable dates | 6/8 | 5/8 | 5/8 | 5/8 |

Under nominal execution, completed trading gave up $1,517.45 and leftover inventory recovered $495.75. Completed profit per matched share fell from 1.27¢ to 0.85¢. Completed trades remained profitable, unlike the earlier combined taper, which lost $5,029 on completed trades. These comparisons measure the feedback of the entire policy, including different fills and subsequent positions. They do not assign those dollars to individual veto decisions.

| UTC date | Markets | Current nominal P&L | Veto nominal P&L | Current ending shares, mean absolute | Veto ending shares, mean absolute |
|---|---:|---:|---:|---:|---:|
| Oct 23 | 70 | +$326 | +$159 | 5.56 | 2.49 |
| Oct 24 | 81 | −$206 | −$202 | 7.05 | 3.10 |
| Oct 25 | 72 | +$303 | +$380 | 9.64 | 5.92 |
| Oct 27 | 58 | +$35 | +$10 | 11.41 | 6.22 |
| Oct 31 | 102 | +$39 | −$236 | 33.35 | 24.80 |
| Nov 1 | 88 | +$665 | +$564 | 17.03 | 11.58 |
| Nov 2 | 72 | −$277 | −$488 | 6.62 | 4.11 |
| Nov 3 | 76 | +$625 | +$304 | 7.47 | 2.47 |

The only intervention was to withhold an entire new bid while long or flat, or ask while short or flat, if projected same-side inventory exceeded the late target. Outstanding reservations counted. An order on the side reducing known inventory retained the original proposal even if a complete fill could cross flat. The veto fired in 597 of 619 markets in each arm. The nominal policy withheld 815,759 bid and 608,508 ask *slot decisions*, including repeated decisions about the same empty slot. It passed through 20,290 late reducing-side proposals. The independent original 150-share whole-order cap and resting-order logic continued to operate.

The test was frozen in 4d281d241 before scoring. Both policies used two 50-share levels per side, passive trading, simulated L queues and fixed public order flow, with zero maker fees and rebates. Nominal execution uses 58 ms activation and 37 ms live cancellation. Short execution ends eligibility immediately at a cancel request and retains the nominal notice schedule. As of November 26, 2026, the consumed-date census contained 2,304 markets, including 619 complete, 938 missing endpoints, 746 stale books and one boundary-censored. All complete markets were included. November 14–18 remains reserved. These are offline development outcomes.

Controls reproduced the original policy on one fixed market per date and execution assumption, and on every market where the new veto never fired. Replay checked same-side reservations and actual peak inventory against the hard cap. Independent read-back verified dataset hashes, market membership, matched-share and settlement identities, all 36 daily and pooled panels, and the decision. Runtime was 20 minutes 12 seconds on one core.

[Full daily results](comparison.csv) · [Receipt and hashes](result.json) · [Frozen test](request.md) · [Earlier combined taper](../urgency_inventory_taper_20261126/report.md)
