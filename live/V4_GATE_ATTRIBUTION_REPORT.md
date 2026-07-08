# V4 Gate Win/Loss Attribution

Scope: exact cycle-level attribution for the simple gates. Metric is 24h residual-alpha model edge, skipped cycles contribute zero in the gate comparison.

## Win/Loss Buckets

| gate | bucket | n | edge_sum | edge_mean | long_mean | short_mean |
| --- | --- | --- | --- | --- | --- | --- |
| btc_no_bull | traded_winners | 2112 | 1236609.72 | 585.52 | 297.51 | 288.01 |
| btc_no_bull | traded_losers | 1918 | -1052047.12 | -548.51 | -310.81 | -237.70 |
| btc_no_bull | missed_winners | 589 | 384245.89 | 652.37 | 351.41 | 300.96 |
| btc_no_bull | saved_losers | 678 | -468100.83 | -690.41 | -406.39 | -284.03 |
| bucket_report_card | traded_winners | 1649 | 1035775.35 | 628.12 | 338.30 | 289.82 |
| bucket_report_card | traded_losers | 1474 | -869087.74 | -589.61 | -328.60 | -261.01 |
| bucket_report_card | missed_winners | 1052 | 585080.26 | 556.16 | 263.74 | 292.42 |
| bucket_report_card | saved_losers | 1122 | -651060.21 | -580.27 | -345.20 | -235.07 |
| btc_bear_only | traded_winners | 525 | 349208.40 | 665.16 | 336.76 | 328.39 |
| btc_bear_only | traded_losers | 413 | -245527.14 | -594.50 | -310.79 | -283.70 |
| btc_bear_only | missed_winners | 2176 | 1271647.21 | 584.40 | 302.63 | 281.77 |
| btc_bear_only | saved_losers | 2183 | -1274620.81 | -583.88 | -340.50 | -243.39 |
| tree_d2 | traded_winners | 1704 | 962619.31 | 564.92 | 277.72 | 287.20 |
| tree_d2 | traded_losers | 1582 | -864731.63 | -546.61 | -324.79 | -221.82 |
| tree_d2 | missed_winners | 997 | 658236.30 | 660.22 | 363.19 | 297.03 |
| tree_d2 | saved_losers | 1014 | -655416.32 | -646.37 | -352.91 | -293.46 |

## Bucket Report-Card vs BTC Skip-Bull

| case | n | edge_sum | edge_mean | win_rate | long_mean | short_mean |
| --- | --- | --- | --- | --- | --- | --- |
| both_trade | 2520 | 178223.29 | 70.72 | 0.54 | 22.13 | 48.59 |
| bucket_skips_btc_trades | 1510 | 6339.30 | 4.20 | 0.50 | -15.61 | 19.81 |
| bucket_trades_btc_skips | 603 | -11535.69 | -19.13 | 0.48 | 29.41 | -48.54 |
| both_skip | 664 | -72319.25 | -108.91 | 0.45 | -129.95 | 21.03 |

Interpretation: `bucket_skips_btc_trades` is the decisive loss. Those are cycles BTC skip-bull would trade but the bucket gate skipped; their average edge is near flat but total positive, so the bucket gate gave up too much calendar edge. `bucket_trades_btc_skips` is bull exposure reintroduced by the bucket gate; it is negative.

## Bucket Gate: Biggest Missed Winners

| period | macro | trend_bucket | n | edge_sum | edge_mean | win_rate | long_mean | short_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025H2 | side | side_up | 80 | 26338.03 | 329.23 | 0.56 | 176.42 | 152.80 |
| 2026 | side | side_up | 20 | 17184.46 | 859.22 | 0.60 | 146.74 | 712.48 |
| 2026 | bear | bear_deep | 26 | 13134.84 | 505.19 | 0.62 | 394.05 | 111.13 |
| 2025H2 | bear | bear_mild | 22 | 9245.30 | 420.24 | 0.64 | 154.13 | 266.11 |
| 2024 | side | side_up | 136 | 8163.70 | 60.03 | 0.58 | -15.39 | 75.42 |
| 2024 | bull | bull_mild | 121 | 6468.02 | 53.45 | 0.47 | -25.57 | 79.02 |
| 2025H2 | bull | bull_mild | 61 | 5389.15 | 88.35 | 0.48 | 52.43 | 35.92 |
| 2026 | bear | bear_mid | 23 | 5259.09 | 228.66 | 0.57 | 42.36 | 186.29 |
| 2025H2 | side | side_down | 73 | 5006.92 | 68.59 | 0.55 | 33.76 | 34.82 |
| 2026 | side | side_flat | 76 | -1692.59 | -22.27 | 0.58 | -49.29 | 27.02 |
| 2025H2 | bear | bear_deep | 12 | -1744.74 | -145.39 | 0.33 | -161.02 | 15.62 |
| 2024 | bull | bull_hot | 40 | -4268.64 | -106.72 | 0.38 | -37.21 | -69.50 |

## Bucket Gate: Biggest Saved Losers

| period | macro | trend_bucket | n | edge_sum | edge_mean | win_rate | long_mean | short_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025H2 | side | side_flat | 315 | -38686.28 | -122.81 | 0.47 | 51.07 | -173.89 |
| 2025H1 | bull | bull_deep | 92 | -27770.89 | -301.86 | 0.51 | -143.21 | -158.65 |
| 2025H1 | bull | bull_mild | 70 | -25691.43 | -367.02 | 0.36 | -536.38 | 169.36 |
| 2025H1 | side | side_flat | 181 | -10391.23 | -57.41 | 0.48 | -129.09 | 71.68 |
| 2024 | bull | bull_deep | 231 | -9590.30 | -41.52 | 0.45 | -98.08 | 56.57 |
| 2024 | side | side_flat | 300 | -7441.24 | -24.80 | 0.48 | -82.76 | 57.96 |
| 2025H1 | bull | bull_hot | 32 | -7372.23 | -230.38 | 0.41 | -201.86 | -28.53 |
| 2024 | side | side_down | 123 | -7041.12 | -57.24 | 0.45 | -58.82 | 1.57 |
| 2025H1 | side | side_down | 40 | -5322.77 | -133.07 | 0.42 | -52.36 | -80.71 |
| 2025H1 | side | side_up | 60 | -5229.23 | -87.15 | 0.40 | -151.87 | 64.71 |
| 2024 | bull | bull_hot | 40 | -4268.64 | -106.72 | 0.38 | -37.21 | -69.50 |
| 2025H2 | bear | bear_deep | 12 | -1744.74 | -145.39 | 0.33 | -161.02 | 15.62 |

## Bucket Gate: Biggest Traded Losers

| period | macro | trend_bucket | n | edge_sum | edge_mean | win_rate | long_mean | short_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025H2 | bear | bear_mid | 62 | -32133.28 | -518.28 | 0.34 | 0.41 | -518.68 |
| 2024 | bull | bull_mild | 112 | -20105.46 | -179.51 | 0.45 | -131.11 | -48.40 |
| 2024 | side | side_flat | 358 | -13161.15 | -36.76 | 0.49 | -77.26 | 40.50 |
| 2025H1 | side | side_up | 69 | -7708.91 | -111.72 | 0.39 | -260.47 | 148.75 |
| 2024 | bear | bear_deep | 12 | -6341.99 | -528.50 | 0.25 | -182.38 | -346.12 |
| 2026 | bear | bear_mid | 68 | -6265.79 | -92.14 | 0.49 | -187.47 | 95.33 |
| 2024 | bull | bull_deep | 231 | -5894.22 | -25.52 | 0.47 | 112.75 | -138.27 |
| 2025H1 | bear | bear_deep | 11 | -4071.07 | -370.10 | 0.18 | -353.05 | -17.04 |
| 2024 | bull | bull_hot | 37 | -2533.53 | -68.47 | 0.46 | -152.62 | 84.14 |
| 2026 | bull | bull_hot | 41 | -2210.42 | -53.91 | 0.44 | 154.97 | -208.88 |
| 2025H1 | side | side_flat | 112 | -2096.91 | -18.72 | 0.52 | -71.88 | 53.15 |
| 2024 | side | side_down | 153 | -1312.92 | -8.58 | 0.50 | 88.73 | -97.31 |

## Feature Profiles

| group | n | edge_bps | btc_ret_30d | pred_gap | long_ret3d | short_ret3d | long_trail3_resid | short_trail3_resid | xs_ret1d_std | n_symbols |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bucket_traded_winners | 1649 | 628.12 | -0.01 | 0.82 | 0.02 | 0.07 | 26.30 | 288.16 | 0.04 | 130.94 |
| bucket_traded_losers | 1474 | -589.61 | 0.01 | 0.82 | 0.02 | 0.06 | 66.52 | 211.41 | 0.04 | 128.77 |
| bucket_missed_winners | 1052 | 556.16 | 0.06 | 0.82 | 0.01 | 0.08 | 16.26 | 185.15 | 0.04 | 126.39 |
| bucket_saved_losers | 1122 | -580.27 | 0.07 | 0.81 | 0.00 | 0.06 | 17.08 | 207.79 | 0.04 | 125.32 |
| btc_traded_winners | 2112 | 585.52 | -0.04 | 0.82 | 0.01 | 0.06 | 20.38 | 238.61 | 0.04 | 132.64 |
| btc_traded_losers | 1918 | -548.51 | -0.04 | 0.81 | 0.00 | 0.04 | 23.74 | 193.99 | 0.04 | 131.58 |

## Diagnosis

- The bucket report-card gate is quality-positive: active edge rises versus `btc_no_bull`, but it skips too many large winners.
- Its missed winners are mainly side-up and bear buckets, exactly where the residual model can work. This argues against a coarse full-book trade/skip gate.
- It still trades large losing pockets: 2025H2 mid-bear, 2024 bull-mild/deep, 2024 side-flat, and 2025H1 side-up. These are leg/regime-specific failures, not one global output-distribution failure.
- Feature means of missed winners and saved losers are very close; simple averages of pred gap, ret3d, trail3, and rvol do not separate them cleanly. The classifier needs leg-specific or richer state, or should remain conservative.

## Next Test

Test separate gates for long and short legs. The combined edge hides cases where one leg is good and the other is bad. Start with: side long veto, bull short veto, mid-bear short veto, while keeping bear/mild-bear and side-up winners available.
