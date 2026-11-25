# Resting-order inventory budget review

On the 70 complete October 26 markets the inventory rule requested 1,574 removals, including 325 while the post was still pending. Another 518 shares matched after an inventory request and before its venue effect. Mean absolute expiry inventory rose from 587.19 to 587.42 shares. The existing $50 projected-leftover budget now governs outstanding orders as well as admissions. This change by itself did not unload more inventory by expiry.

| Paired simulator metric | Admission-only budget | Budget on admissions and resting orders |
|---|---:|---:|
| Settlement P&L | −$2,694.41 | −$2,894.34 |
| Completed FIFO turnover | +$370.00 | +$78.70 |
| Open inventory at settlement, P&L | −$3,064.42 | −$2,973.04 |
| Mean absolute terminal inventory, shares | 587.19 | 587.42 |
| Median peak inventory, shares | 981.80 | 988.68 |
| Submissions / cancellation requests | 30,912 / 27,942 | 32,297 / 29,290 |
| Both sides live, mean grid share | 75.33% | 74.52% |

Settlement fell $199.93. Thirty-five markets improved, 32 worsened and three tied. The inventory component improved $91.38, while completed turnover fell $291.31. These are paired effects of the entire feedback policy. The test does not identify a queue cost for any individual removal.

The synthetic signed 350-share cases verify the decision on both sides. At a 50¢ mid and expected exits of 250, the rule requests cancellation of the 100 outstanding adding shares. If the forecast recovers to 300 before the cancellation is acknowledged, admission still waits. After acknowledgment the controller can post 50 shares. Independent controls cover partial fills, the live and pending cancellation clocks, matching during transit, avoiding repeat cancels and preserving an eligible older order. The disabled replay on market 3789484 reproduced the original recorded metrics. The initial pressure-precedence fixture accidentally shared its pressure array with the opening trigger; separate arrays fixed the fixture before market replay.

All market inputs, the earlier-day capacity fit, pressure rules and the hard position cap match the prior simulation. This is one consumed development day. Of 288 registry markets, 124 lacked the required endpoint and 94 had a stale book. These findings apply to the replayable subset. A separate declining terminal share target and active exits were outside this comparison.

[Controller](policy.py) · [Frozen protocol](protocol.md) · [Paired market results](markets.csv) · [Machine-readable totals](result.json)
