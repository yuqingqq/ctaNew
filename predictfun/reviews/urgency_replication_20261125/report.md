# October 27 replication of urgency-only exit repricing

The unchanged passive controller earned $904 at settlement on 58 complete October 27 markets. Completed trades earned 2.15¢ per matched share, close to October 26's 2.16¢. Four of the five prespecified criteria passed on October 27. Mean absolute ending inventory missed the ceiling by 5.15 shares (54.96 against 49.81). Both dates were previously consumed in research. As of November 25, coverage was 70 of 288 scheduled markets on October 26 (124 missing endpoints, 94 stale books) and 58 of 288 on October 27 (139 missing endpoints, 91 stale books). These are historical replay point estimates under shared external flow.

| Day and policy | Settlement | Completed FIFO | Residual | ¢ per matched share | Mean absolute ending shares |
|---|---:|---:|---:|---:|---:|
| Oct 26 original | −$2,759 | −$2,303 | −$457 | −1.19 | 41.9 |
| Oct 26 retain all distant | +$1,891 | +$5,938 | −$4,047 | +7.00 | 186.2 |
| Oct 26 urgency | +$1,774 | +$2,260 | −$485 | +2.16 | 38.3 |
| Oct 27 original | −$2,878 | −$2,168 | −$710 | −1.38 | 39.8 |
| Oct 27 retain all distant | +$2,629 | +$5,452 | −$2,823 | +8.49 | 171.1 |
| Oct 27 urgency | +$904 | +$1,769 | −$865 | +2.15 | 55.0 |

On October 27 urgency improved settlement over the original in 43 of 58 markets, by $3,782 overall and $3,321 excluding the largest gain. Unconditional retention earned $1,725 more than urgency and carried 171.1 ending shares on average. The urgency controller's 90th percentile was 227.4 shares and its maximum was 557.3.

## Where inventory cost money

The rows use the **existing 250-share allowance at 60 seconds to expiry**. Positions here are realized on the simulated venue clock. Decisions in a controller must use delivered fills and reservations. Residual is the FIFO cost-basis-to-settlement P&L of the position left at the end. These group differences do not measure the effect of intervening.

| Day | Realized absolute position at 60 s | Markets | Mean absolute ending shares | Completed FIFO | Residual | Settlement |
|---|---|---:|---:|---:|---:|---:|
| Oct 26 | Above 250 | 35 | 67.7 | +$249 | −$466 | −$218 |
| Oct 26 | Within 250 | 35 | 8.8 | +$2,011 | −$19 | +$1,992 |
| Oct 27 | Above 250 | 33 | 89.3 | +$1,159 | −$804 | +$355 |
| Oct 27 | Within 250 | 25 | 9.7 | +$610 | −$61 | +$549 |

Six of the seven largest ending exposures on October 26 and all six on October 27 were already over budget with a minute left. The October 27 over-budget group still earned $355 in total, so its $804 residual loss cannot be counted as recoverable profit.

**Next experiment.** Keep the current controller as the comparator. During the 90-to-60-second window only, cap the risk-adding part of each *new* quote at the upcoming 250-share allowance after client-known inventory and all same-side reservations. An order may still close existing inventory. Keep existing orders under the current cancellation rules so the experimental change does not itself discard their place in the queue. Use the same two consumed cohorts. Compare settlement, completed FIFO, mean and p90 ending shares, and how many markets remain over 250 at 60 seconds. Require higher settlement and lower mean ending exposure on each day with positive completed turnover before pursuing that change. This is a candidate for testing.

The replay was frozen at 488bab589. All metrics of the preselected October 26 reference market reproduced for the three arms; every October 27 tape matched its census identity. Market-level economic and share identities reconcile for each arm; the independent CSV check refused a corrupted residual on each date. [Frozen protocol](protocol.md) · [Daily aggregates and coverage](result.json) · [October 27 market rows](markets.csv) · [Inventory paths](inventory.csv)
