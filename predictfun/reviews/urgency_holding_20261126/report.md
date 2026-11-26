# Holding time in the 100-share urgency backtest

Across 689 complete markets on nine previously examined dates, a share closed by an offsetting fill spent 68.9 seconds in inventory on average. The median was 55.2 seconds and 69.3% of matched shares took longer than 30 seconds.

| Inventory outcome | Opening shares | Mean | Median | 90th percentile | Held over 30 s | P&L |
|---|---:|---:|---:|---:|---:|---:|
| Closed through an opposite fill | 627,323 | 68.9 s | 55.2 s | 152.4 s | 69.3% | +$28,516 |
| Still open at payout | 32,642 | 137.1 s | 135.1 s | 215.9 s | 99.99% | −$6,271 |
| All inventory opened | 659,965 | 72.3 s | 58.4 s | 157.9 s | 70.9% | +$22,244 |

Matched shares closed after 30 seconds contributed +$21,165 of the +$28,516 completed turnover. Shares closed within 30 seconds contributed +$7,351. Realized holding time is known only after a close. These figures describe the original policy and do not estimate the effect of forcing a longer hold.

The clock starts at the simulated fill that opens or increases net inventory. Each opposite fill closes the oldest open shares first, including partial closes and fills that reverse inventory. The outstanding shares are timed through the five-minute payout boundary, two seconds beyond the final supported simulation endpoint. All durations are quantity weighted. A matched quantity is counted once. Ages exclude time spent waiting for the opening order to fill.

As of November 26, 2026. The simulated fill count, matched and residual quantities, and completed and residual P&L reconcile to the committed size receipt in every market. An independent integral of absolute net inventory over time reproduces total share-seconds. Synthetic controls cover partial closes, a reversal, a short position, the exact 30-second boundary and refusal of reversed timestamps. Raw fill journal SHA-256 is `2c425dbaab1614df4d7e4895a50f39e1cedd621e960de0d3908e24ca4c4abb20`.

[Daily and pooled results](holding.csv) · [Fixed-horizon markouts](../urgency_markouts_20261126/report.md)
