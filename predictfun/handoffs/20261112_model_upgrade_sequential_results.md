# Model upgrade — full sequential replay
As of 2026-11-12T22:36:17.206635+00:00; six consumed complete dates; 1,728 windows per policy/scenario.

| Scenario | Policy | Valid / missing marks | Marked USD | Settlement USD | Inventory RMS | Union quote presence |
|---|---|---:|---:|---:|---:|---:|
| X | default | 1656 / 72 | -2596.57 | -2643.22 | 3.971 | 18.83% |
| X | legacy_guard | 1611 / 117 | -35.95 | 42.11 | 3.057 | 2.21% |
| X | causal_guard | 1608 / 120 | -31.24 | 94.86 | 3.138 | 2.09% |
| X | static10 | 1612 / 116 | -752.35 | -821.00 | 3.920 | 5.28% |
| X | history10 | 1586 / 142 | -716.41 | -748.35 | 4.261 | 5.20% |
| stress | default | 1660 / 68 | -2948.67 | -3010.32 | 3.858 | 18.82% |
| stress | legacy_guard | 1596 / 132 | -184.14 | -83.95 | 3.430 | 2.50% |
| stress | causal_guard | 1598 / 130 | -195.07 | -102.44 | 3.434 | 2.37% |
| stress | static10 | 1634 / 94 | -1457.77 | -1521.88 | 3.723 | 6.08% |
| stress | history10 | 1618 / 110 | -1423.14 | -1476.13 | 3.925 | 6.01% |

Raw marked totals use different populations. Paired comparisons below are the decision evidence.

| Comparison | Common / missing windows | Marked delta USD [day interval] | Inventory-square delta % |
|---|---:|---:|---:|
| X:static10-default | 1578 / 150 | +1728.20 [1453.5474087795494, 2016.1191463917773] | -2.55 |
| X:static10-legacy_guard | 1541 / 187 | -677.10 [-801.2618871158296, -565.1666411999778] | +64.40 |
| X:static10-causal_guard | 1539 / 189 | -678.90 [-820.6675147195253, -527.4505273151939] | +56.06 |
| X:history10-default | 1561 / 167 | +1746.76 [1470.6217973105329, 1968.7846606697778] | +15.17 |
| X:history10-legacy_guard | 1522 / 206 | -642.08 [-735.1127049040151, -535.8809885124755] | +94.30 |
| X:history10-causal_guard | 1517 / 211 | -645.80 [-767.6081960089655, -517.4221556768173] | +84.45 |
| X:history10-static10 | 1537 / 191 | +28.08 [-59.10120158374487, 117.42271025064855] | +18.19 |
| stress:static10-default | 1605 / 123 | +1402.19 [1099.0818174158371, 1670.9143492012818] | -6.88 |
| stress:static10-legacy_guard | 1539 / 189 | -1180.41 [-1357.0345332098238, -1027.2634818487836] | +17.83 |
| stress:static10-causal_guard | 1541 / 187 | -1181.34 [-1360.4047254043132, -1010.9179500389679] | +17.56 |
| stress:history10-default | 1585 / 143 | +1411.97 [1022.6790714484363, 1756.0979247196174] | +3.48 |
| stress:history10-legacy_guard | 1538 / 190 | -1170.07 [-1266.3989829755226, -1089.2882978108428] | +30.95 |
| stress:history10-causal_guard | 1538 / 190 | -1153.97 [-1292.1405076776423, -1016.8166662505183] | +30.65 |
| stress:history10-static10 | 1572 / 156 | +17.27 [-104.9297214526855, 161.03423844444544] | +11.14 |

Computed objectives: `{"causal_guard": {"both_scenarios_day_intervals_positive": false, "both_scenarios_marked_up_inventory_not_up": false, "primary_day_interval_positive": false}, "static10": {"both_scenarios_day_intervals_positive": false, "both_scenarios_marked_up_inventory_not_up": false, "primary_day_interval_positive": false}}`

Complete exploratory sequential policies on consumed dates, with inferred execution clocks and queues. Historical legacy guard is not chronological. Common-mark comparisons retain all missing counts; missingness can still be policy-dependent. Inventory risk constraint is reported rather than relaxed. No live change or untouched validation. Timer re-entry and age100ms eligibility remain tested policy limitations.
