# Limit inventory by repricing exits when over budget

Frozen decision: **pass**. On October 26, the same 70 complete markets were eligible out of 288. Two previously tested controls reproduced their per-market metrics on all 70, and the original ladder comparison is hash-bound. The other 124 markets lacked a supported endpoint and 94 had stale books. The four policies share external flow, settlement outcomes, both passive levels, pressure protection and costs.

| Metric | Original | Keep all distant | Reprice every distant closing quote | Reprice only while over budget |
|---|---:|---:|---:|---:|
| Settlement | −$2,759.33 | +$1,890.96 | −$1,777.94 | +$1,774.49 |
| Completed FIFO | −$2,302.71 | +$5,937.92 | −$1,514.33 | +$2,259.82 |
| Residual inventory P&L | −$456.62 | −$4,046.95 | −$263.61 | −$485.32 |
| ¢ per matched share | -1.189 | +7.003 | -0.872 | +2.157 |
| Matched shares | 193,654 | 84,789 | 173,658 | 104,752 |
| Mean absolute ending shares | 41.9 | 186.2 | 33.1 | 38.3 |
| 90th percentile absolute ending shares | 162.6 | 539.4 | 127.6 | 188.3 |
| Largest absolute ending position | 368.7 | 792.6 | 351.9 | 438.8 |
| Within 10 shares of flat | 51/70 | 30/70 | 52/70 | 53/70 |
| Positive markets | 26/70 | 42/70 | 29/70 | 45/70 |
| All cancellation requests | 23,617 | 11,185 | 20,401 | 13,896 |
| Distance-only requests | 7,900 | 0 | 5,417 | 1,387 |
| Both sides live | 61.7% | 53.8% | 68.6% | 59.0% |
| Same-price opening cycle P&L | +$192.97 | +$2,453.11 | +$66.97 | +$1,399.04 |
| Through-price opening cycle P&L | −$2,495.68 | +$3,484.81 | −$1,581.30 | +$860.77 |

The expiry exposure ceiling is 51.89 mean absolute shares. positive_settlement=True, positive_turnover=True, improved_settlement=True, improved_turnover=True, within_inventory_limit=True.

Settlement versus reference: +$4,533.82; 49 improved, 21 worsened, 0 equal. Omitting the largest improvement: +$4,129.65.
Settlement versus retain: −$116.47; 37 improved, 32 worsened, 1 equal. Omitting the largest improvement: −$544.22.
Settlement versus role: +$3,552.44; 48 improved, 22 worsened, 0 equal. Omitting the largest improvement: +$3,255.62.

## Inventory path

| Seconds to expiry | Target, shares | Keep all distant | Always responsive | Urgency only |
|---:|---:|---:|---:|---:|
| 150 | 1000 | 469.3 | 232.5 | 469.3 |
| 120 | 750 | 530.4 | 285.9 | 496.0 |
| 90 | 500 | 469.0 | 290.6 | 370.6 |
| 60 | 250 | 413.7 | 201.4 | 255.4 |
| 30 | 0 | 247.3 | 91.8 | 94.3 |
| 10 | 0 | 197.5 | 51.7 | 56.1 |
| 2 | 0 | 186.2 | 33.1 | 38.3 |

The target uses client-known net holdings at the decision. The path uses realized holdings, averaged in absolute shares across markets. A decreasing target does not guarantee a passive counterparty. Ending exposures are the last simulated position two seconds before settlement, valued at the recorded payout.

| Policy | Mean seconds over budget | Of those, exit actually live | Of those, exit side under pressure |
|---|---:|---:|---:|
| retain | 89.7 | 78.2 | 2.2 |
| role | 60.2 | 42.3 | 1.5 |
| urgency | 72.6 | 52.3 | 1.8 |

These are paired feedback paths on one previously examined day. No additional days, active taking or parameter search were used. A positive development result warrants a frozen replication; it is not validation.

[Protocol](protocol.md) · [Market rows](markets.csv) · [Inventory traces](inventory.csv) · [Criteria and hashes](result.json)
