# Did late cancellation cause the losses?

November 28, 2026 UTC. The unchanged event-driven strategy had two negative dates in the last eight. This audit reconstructs every fill in all 124 eligible markets on those dates, including profitable markets. The other 452 scheduled markets on those two days had incomplete or stale inputs. They have unknown economics.

On all eight dates, 57,364.1 of 632,305.2 simulated shares filled strictly after a cancel request (9.07%). Moving pressure cancellation from the 50 ms grid to feed events had improved settlement on the same 550 markets by $519.28.

## Which legs had a cancel in flight?

| Date | Pair classification | Matched shares | Gross losses USD | Gross gains USD | Net completed USD |
|---|---|---:|---:|---:|---:|
| 20261115 | Either leg pending | 6,428.78 | -439.57 | 386.49 | -53.08 |
| 20261115 | Neither pending | 30,688.56 | -1,742.42 | 1,684.09 | -58.33 |
| 20261115 | All | 37,117.34 | -2,181.99 | 2,070.58 | -111.41 |
| 20261125 | Either leg pending | 7,888.26 | -409.56 | 379.37 | -30.19 |
| 20261125 | Neither pending | 28,540.90 | -1,758.29 | 1,564.51 | -193.78 |
| 20261125 | All | 36,429.17 | -2,167.85 | 1,943.87 | -223.98 |
| pooled | Either leg pending | 14,317.04 | -849.13 | 765.86 | -83.27 |
| pooled | Neither pending | 59,229.47 | -3,500.71 | 3,248.59 | -252.12 |
| pooled | All | 73,546.51 | -4,349.84 | 4,014.45 | -335.39 |

A request must precede the match strictly. Fills at the decision timestamp execute first. Live cancel effect is modeled at request +37 ms. Each FIFO tranche is counted once. This split describes the existing feedback path. A changed cancellation time could affect future orders, good fills and inventory.

## Where did the price move?

| Completed cohort | Entry edge USD | First second of holding USD | Later movement until close USD | Exit edge USD | Unmarked pair P&L USD | Total USD |
|---|---:|---:|---:|---:|---:|---:|
| all | 537.84 | -1,170.70 | -149.73 | 480.52 | -33.32 | -335.39 |
| losing_pairs | 233.23 | -718.39 | -4,034.81 | 207.97 | -37.84 | -4,349.84 |
| winning_or_flat | 304.62 | -452.31 | 3,885.09 | 272.54 | 4.52 | 4,014.45 |
| losing_markets | 262.10 | -591.92 | -1,655.22 | 242.89 | -37.51 | -1,779.66 |

Any pair closed within one second has zero later movement. All three marks must be valid and at most one second old. Matched shares missing at least one mark total 183.82 of 73,546.51. The losing-pair breakdown is a hindsight cohort.

## How identifiable were the openings?

| Opening-fill classification | Matched shares | Gross losses USD | Gross gains USD | Net completed USD |
|---|---:|---:|---:|---:|
| cancel_in_flight | 7,778.56 | -456.02 | 371.64 | -84.38 |
| no_prior_live_warning | 65,767.95 | -3,893.82 | 3,642.81 | -251.01 |
| at_or_favorable_to_mid | 68,090.36 | -4,001.99 | 3,754.77 | -247.22 |
| wrong_side_of_mid | 5,456.15 | -347.85 | 259.68 | -88.17 |
| same | 26,970.84 | -1,482.41 | 1,642.74 | 160.32 |
| through | 46,575.66 | -2,867.43 | 2,371.72 | -495.71 |

The warning is the current order side’s 250 ms Binance pressure threshold, checked on grid ticks and off-grid feed evaluations from activation up to, but excluding, the match. A fill can arrive immediately after activation; absence of a warning does not establish time for a feasible cancel. The prefill midpoint uses the strictly preceding received two-sided book within one second. Local midpoint, public execution price and an executable prediction of fair value are different measures. A through-price print crosses our quote on the realized tape.

## One-second fill markout

| Status | Filled shares | Shares with mark | Gross adverse USD | Gross favorable USD |
|---|---:|---:|---:|---:|
| cancel_in_flight | 15,178.74 | 15,178.74 | -345.78 | 80.76 |
| no_prior_live_warning | 132,497.35 | 132,377.72 | -2,054.57 | 945.69 |
| warning_too_late | 21.00 | 21.00 | -15.75 | 0.00 |
| warning_with_lead | 106.05 | 81.05 | -0.83 | 1.05 |
| all | 147,803.13 | 147,658.51 | -2,416.93 | 1,027.50 |

Both days reproduced every archived per-market control column. Economic differences reflect modeled matching, the historical clock, a fixed public tape and zero maker fees or rebates. The eight dates were previously examined. The two-day breakdown is descriptive.

[Protocol](protocol.md) · [Full results](results.json) · [Earlier eight-date attribution](../pressure_market_sharpe_20261128/report.md)
