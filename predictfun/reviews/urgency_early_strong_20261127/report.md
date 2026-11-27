# Earlier inventory tightening near the BTC opening price

The strong BTC rule passed the prespecified development screen under both execution assumptions. On the same 211 markets, nominal settlement was +$468.54 versus +$132.64 for current, +$185.72 for the mild BTC rule and +$357.68 for strong time-only tightening. The stronger BTC rule improved nominal settlement over current on each date. October 24 still lost money.

| Metric | Current | Mild BTC | Strong time only | Strong BTC |
|---|---:|---:|---:|---:|
| Nominal settlement P&L | $132.64 | $185.72 | $357.68 | $468.54 |
| Completed FIFO P&L | $551.35 | $596.26 | $683.64 | $839.37 |
| Leftover inventory P&L | −$418.70 | −$410.54 | −$325.96 | −$370.84 |
| Shares matched in round trips | 97,750 | 96,891 | 88,249 | 85,024 |
| Mean absolute ending shares | 9.13 | 9.03 | 6.78 | 8.03 |
| Mean of largest 22 ending positions, shares | 68.42 | 68.76 | 56.66 | 66.08 |
| Mean loss of worst 22 markets | $56.04 | $55.09 | $51.95 | $48.07 |
| Maximum intramarket position, shares | 150 | 150 | 150 | 150 |
| Positive dates, nominal | 2/3 | 2/3 | 2/3 | 2/3 |
| Settlement if eligibility ends at cancel request | $481.10 | $512.82 | $673.97 | $687.27 |

| Date | Markets | Current nominal | Mild BTC | Strong time only | Strong BTC |
|---|---:|---:|---:|---:|---:|
| Oct 24 | 81 | −$206.02 | −$187.96 | −$149.88 | −$52.42 |
| Oct 25 | 72 | $303.29 | $330.16 | $407.76 | $360.21 |
| Oct 27 | 58 | $35.37 | $43.53 | $99.80 | $160.74 |

Against current, nominal completed profit increased $288.03 and leftover losses decreased $47.87. The matched quantity fell 13.0%, while completed value rose from 0.56¢ to 0.99¢ per matched share. Earlier tightening alone also helped. The BTC-conditioned rule earned $110.86 more than strong time-only under nominal execution, but just $13.31 more under short eligibility. Time-only ended flatter in both runs. These are net differences between complete feedback policies and cannot identify which rejected order would have lost.

**What changed.** At or inside 0.5 estimated remaining standard deviations from the opening mid, new exposure follows a piecewise linear schedule of 150 shares at 210 seconds remaining, 100 at 90, and zero at 30. At or beyond 1.5 standard deviations, keep the original time curve. Interpolate between the curves at each tick, using the same one-second-lagged BTC feature as the mild pilot. Far-side limits at 90, 60 and 48 seconds are 150 shares; near-side limits are 100, 50 and 30. Both directions use the same rule. Time-only uses their midpoint, also used for missing BTC. Existing resting exits stay under the original review rules, and committed or retiring quotes count against new room.

The intervention affected 200/211 nominal markets (mild: 162/211). Of 174,744 otherwise eligible proposal decisions on the strong path, 2,862 were reduced in size and 157,932 withheld (92.0% combined); the mild path was 6,567 of 25,315 (25.9%). Repeated retries at an empty slot are separate decisions. These percentages measure proposal decisions on each policy's path, not order cancellations, unique fills or time out of market.

**Coverage and evidence.** As of Nov 27, 2026 the three consumed dates had 864 registered markets. All 211 complete markets were paired, 392 lacked endpoints and 261 had stale books. The BTC feature fell back on 5,114 of 759,389 ticks in the active interval. Settings were fixed in a1cc042c8 before replay. Baseline and mild reference runs matched archived figures on one fixed market per date and execution assumption. Read-back verified all 32 aggregate panels, ledger and share identities, file hashes, proposal counts and verdicts. Wall time was 13 minutes 25 seconds on one research core.

These estimates assume fixed external flow, simulated queues and zero passive fees and rebates. Three previously studied dates support a research extension only. Next, carry this exact formula to the other five consumed benchmark dates after reproducing the BTC input there. Keep the reserved November validation block and Tokyo settings intact.

[Formula and predeclared screen](request.md) · [Daily figures and intervention counts](comparison.csv) · [Verification hashes](result.json)
