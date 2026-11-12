# Quote by default, learn when to pull
As of 2026-11-12T05:42:48.311589+00:00. Stage: final. Six consumed test dates /1,728 windows per arm and scenario. Exploratory only.

Frozen ladder v3 is the operational benchmark. Matched_guard shares the new5-share JOIN/resting architecture, no ladder, and reserved +/-10 inventory range; it retains PM90/G1/G2. Comparing a model against matched_guard isolates the cancellation-policy change more closely than comparing only with the frozen strategy.

| Timing | Policy | Windows / marked | Marked $ | Settlement $ | Carry on marked $ | 1s /5s markout $ | Spread $ | Inventory RMS | End abs mean | Orders |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X | reference | 1728 /1579 | 152.63 | 324.30 | 46.24 | 250.42 /212.22 | 220.82 | 11.800 | 5.737 | 251224 |
| X | matched_guard | 1728 /1611 | -35.95 | 42.11 | 31.74 | 50.14 /36.64 | 45.90 | 3.057 | 1.524 | 100548 |
| X | default | 1728 /1656 | -2596.57 | -2643.22 | 7.98 | -2488.55 /-2593.34 | 687.66 | 3.971 | 0.355 | 90283 |
| X | pressure | 1728 /1644 | -18.14 | 60.10 | 68.76 | -76.76 /-75.14 | 29.65 | 2.862 | 1.213 | 42778 |
| X | floor | 1728 /1583 | -375.45 | -121.77 | 66.42 | -168.29 /-177.49 | 136.98 | 4.513 | 1.729 | 55671 |
| X | market | 1728 /1599 | -637.50 | -686.00 | -13.15 | -733.89 /-766.24 | 89.16 | 4.036 | 0.879 | 221363 |
| X | queue | 1728 /1557 | -661.78 | -735.88 | 19.62 | -742.25 /-808.37 | 93.75 | 4.103 | 0.959 | 222617 |
| X | queue_cost | 1728 /1568 | -633.69 | -730.01 | -2.13 | -751.08 /-798.70 | 97.03 | 4.165 | 0.953 | 221524 |
| stress | reference | 1728 /1559 | -161.00 | -220.66 | 43.09 | -413.10 /-378.83 | 287.12 | 13.671 | 6.725 | 252803 |
| stress | matched_guard | 1728 /1596 | -184.14 | -83.95 | 15.32 | -112.47 /-111.05 | 63.65 | 3.430 | 1.825 | 91655 |
| stress | default | 1728 /1660 | -2948.67 | -3010.32 | 6.97 | -2832.33 /-3031.60 | 748.51 | 3.858 | 0.306 | 86765 |
| stress | pressure | 1728 /1594 | -294.07 | -331.37 | 7.05 | -395.35 /-408.76 | 80.65 | 3.708 | 1.319 | 44539 |
| stress | floor | 1728 /1603 | -896.70 | -796.12 | 10.95 | -700.04 /-708.73 | 215.92 | 4.390 | 1.080 | 51029 |
| stress | market | 1728 /1592 | -1339.20 | -1445.74 | 13.49 | -1330.34 /-1423.18 | 171.10 | 3.811 | 0.518 | 194245 |
| stress | queue | 1728 /1614 | -1356.12 | -1456.15 | 0.69 | -1316.31 /-1396.99 | 165.22 | 3.855 | 0.572 | 192820 |
| stress | queue_cost | 1728 /1604 | -1351.30 | -1447.95 | 12.99 | -1332.69 /-1394.09 | 166.14 | 3.882 | 0.600 | 192581 |

Marks/spread require fresh valid books. Missing populations and statuses are in the JSON; do not compare raw marked totals with different missing populations. Markout and spread are alternative valuations, not additive P&L. Rebates excluded.

| Comparison | Common marked / windows | Paired marked delta $ [day interval] | Settlement delta $ | Exposure delta % |
|---|---:|---:|---:|---:|
| X:reference-matched_guard | 1555 /1728 | +159.61 [-105.08, +416.41] | +282.19 | +1389.811 |
| X:reference-default | 1545 /1728 | +2525.49 [+2164.42, +2849.32] | +2967.51 | +783.118 |
| X:matched_guard-reference | 1555 /1728 | -159.61 [-416.41, +105.08] | -282.19 | -93.288 |
| X:matched_guard-default | 1578 /1728 | +2417.55 [+2087.49, +2696.93] | +2685.32 | -40.723 |
| X:default-reference | 1545 /1728 | -2525.49 [-2849.32, -2164.42] | -2967.51 | -88.676 |
| X:default-matched_guard | 1578 /1728 | -2417.55 [-2696.93, -2087.49] | -2685.32 | +68.699 |
| X:pressure-reference | 1533 /1728 | -153.08 [-391.98, +62.60] | -264.19 | -94.119 |
| X:pressure-matched_guard | 1557 /1728 | +29.98 [-38.66, +120.36] | +18.00 | -12.381 |
| X:pressure-default | 1589 /1728 | +2467.31 [+2147.52, +2744.45] | +2703.32 | -48.062 |
| X:floor-reference | 1501 /1728 | -464.21 [-709.27, -222.40] | -446.06 | -85.371 |
| X:floor-matched_guard | 1527 /1728 | -289.21 [-374.73, -188.79] | -163.87 | +117.940 |
| X:floor-default | 1559 /1728 | +2066.04 [+1741.00, +2368.82] | +2521.45 | +29.189 |
| X:market-reference | 1511 /1728 | -738.96 [-1054.56, -466.78] | -1010.29 | -88.302 |
| X:market-matched_guard | 1534 /1728 | -559.35 [-673.79, -451.66] | -728.10 | +74.271 |
| X:market-default | 1565 /1728 | +1839.32 [+1521.82, +2133.09] | +1957.22 | +3.303 |
| X:queue-reference | 1488 /1728 | -765.04 [-1079.74, -475.51] | -1060.17 | -87.909 |
| X:queue-matched_guard | 1508 /1728 | -592.27 [-694.89, -511.54] | -777.98 | +80.126 |
| X:queue-default | 1534 /1728 | +1738.76 [+1438.98, +2009.02] | +1907.34 | +6.774 |
| X:queue_cost-reference | 1495 /1728 | -732.50 [-1021.42, -472.74] | -1054.30 | -87.540 |
| X:queue_cost-matched_guard | 1516 /1728 | -563.77 [-657.41, -478.00] | -772.11 | +85.635 |
| X:queue_cost-default | 1543 /1728 | +1780.64 [+1477.71, +2023.93] | +1913.21 | +10.039 |
| stress:reference-matched_guard | 1540 /1728 | -8.47 [-317.12, +265.26] | -136.72 | +1488.556 |
| stress:reference-default | 1527 /1728 | +2518.24 [+2057.38, +2973.02] | +2789.66 | +1155.404 |
| stress:matched_guard-reference | 1540 /1728 | +8.47 [-265.26, +317.12] | +136.72 | -93.705 |
| stress:matched_guard-default | 1561 /1728 | +2560.41 [+2195.07, +2889.56] | +2926.38 | -20.972 |
| stress:default-reference | 1527 /1728 | -2518.24 [-2973.02, -2057.38] | -2789.66 | -92.034 |
| stress:default-matched_guard | 1561 /1728 | -2560.41 [-2889.56, -2195.07] | -2926.38 | +26.537 |
| stress:pressure-reference | 1488 /1728 | -122.70 [-495.06, +310.28] | -110.70 | -92.643 |
| stress:pressure-matched_guard | 1514 /1728 | -105.76 [-265.99, +17.74] | -247.42 | +16.869 |
| stress:pressure-default | 1553 /1728 | +2432.09 [+2011.15, +2842.29] | +2678.96 | -7.641 |
| stress:floor-reference | 1503 /1728 | -643.83 [-948.08, -233.53] | -575.46 | -89.688 |
| stress:floor-matched_guard | 1530 /1728 | -663.57 [-803.49, -528.40] | -712.18 | +63.819 |
| stress:floor-default | 1579 /1728 | +1894.32 [+1396.19, +2358.96] | +2214.20 | +29.463 |
| stress:market-reference | 1493 /1728 | -1103.98 [-1463.25, -808.24] | -1225.08 | -92.229 |
| stress:market-matched_guard | 1517 /1728 | -1110.16 [-1260.76, -968.45] | -1361.79 | +23.439 |
| stress:market-default | 1564 /1728 | +1436.22 [+1190.53, +1675.88] | +1564.58 | -2.449 |
| stress:queue-reference | 1503 /1728 | -1098.50 [-1455.33, -781.09] | -1235.48 | -92.048 |
| stress:queue-matched_guard | 1528 /1728 | -1100.40 [-1237.24, -952.70] | -1372.20 | +26.321 |
| stress:queue-default | 1581 /1728 | +1477.27 [+1243.96, +1710.58] | +1554.18 | -0.171 |
| stress:queue_cost-reference | 1495 /1728 | -1086.36 [-1457.74, -759.38] | -1227.29 | -91.935 |
| stress:queue_cost-matched_guard | 1522 /1728 | -1095.40 [-1236.55, -967.01] | -1364.01 | +28.115 |
| stress:queue_cost-default | 1571 /1728 | +1463.05 [+1187.02, +1720.81] | +1562.37 | +1.247 |

Primary: -50ms trade anchor,40/22/100ms place/cancel/notice, queueM. Stress: -250ms,80/100/250ms,queueL. Day intervals resample6UTC days1,000times; they do not correct for selection on consumed history.

| Policy | Neutral actions / pulls | Estimated local benefit c | Matched null mean c | p >= observed | Zero-fill actions |
|---|---:|---:|---:|---:|---:|
| market | 40059 /30003 | 34013.60 | 29493.19 | 0.0020 | 29224 |
| queue | 40059 /30211 | 34585.12 | 30004.03 | 0.0020 | 29224 |
| queue_cost | 40059 /29621 | 34052.46 | 29434.23 | 0.0020 | 29224 |

One independent hash-selected probe per neutral order generation; unfilled outcomes included.500matched-count shuffles by day/side/hour/age/inventory sign. Local labels include effective cancellation delay and a1s wait followed by reset at the same price. Future portfolio responses are approximated locally; the sequential portfolios above evaluate the actual policy. Market control includes order age and guard prediction.

Computed criteria: {"floor": {"matched_guard": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}, "reference": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}}, "market": {"matched_guard": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}, "reference": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}}, "pressure": {"matched_guard": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}, "reference": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}}, "queue": {"matched_guard": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}, "reference": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}}, "queue_cost": {"matched_guard": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}, "reference": {"all_scenarios_trading_up_exposure_down": false, "primary_day_interval_positive": false, "stress_complete": true}}}

No live promotion. Fill/cancel clocks and queue rules remain uncalibrated; protected dates untouched. Complete source/declaration, per-stage checks, statuses and model identities are retained.
