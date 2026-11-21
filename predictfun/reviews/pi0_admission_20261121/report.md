# PI0-ADMISSION-EXPOSURE-v1 — longer exposure of PI0's own admitted orders, the corrected mean metrics, the cross-tab and RETENTION v2

Review 99fb165d (one next priority). Freeze `6b14953f`, committed before any branch ran. Outputs as of 2026-11-21T11:12:01.503402+00:00. Development session Claude aea5bfb9. One diagnostic, zero fitted candidates, consumed Oct 26–27.

## Verdict (every sentence chosen by a computed or declared predicate)

- **Declared checks:** all true ([checks.json](checks.json)); falsifier passed, 11 cases ([checks_falsifier.json](checks_falsifier.json)).
- **Population:** 329 selected generations of the 689 PI0 POSTs; 329 ok.
- **Reading `admitted_longer_harmful`:** Longer exposure of PI0's own admitted orders loses on both consumed days: later work should focus on admission and remaining-value discrimination, not unconditional duration.
- **Per day (inclusion-weighted mean D_EXPOSURE):** Oct 26 -16.901 c over 166 generations; Oct 27 -2.872 c over 163 generations; 73 generations gain selected-gid fills by keeping.
- **Pooled:** weighted mean D_EXPOSURE -9.367 c (unweighted -7.599 c): selected generation -9.270, focal rest -0.097, opposite +0.000 c. P(fill) of the selected generation 0.066 under PI0 → 0.303 kept; lifetime 136 → 1063 ms (weighted).
- **Concentration (descriptive, computed):** D_EXPOSURE is zero on 259 of 329 generations (31 positive, 39 negative, 20 severe); the top-5 |D| share is 0.37; 3 of 4 day × side cells are negative.
- **First review:** No first-review selection advantage is demonstrated on this sample (not identified when thin or degenerate). PULL on 233 of 299 rows; percentile 0.806; 45 of 91 strata (166 rows) can change assignment; thin false.
- **Entry forecasts at PI0's own admissions:** At PI0's own admissions, V's squared error is not below zero's. Weighted MSE of V 4043.3 c² against zero 3982.0 and edge × E[q] 4098.5; mean error +11.099 c; Brier 0.1525 against 0.2110.
- **First-review forecasts:** weighted MSE of V 3743.4 c² against zero 4018.2 and edge × E[q] 4151.9; mean error +5.694 c.
- **RETENTION v2 (supersedes the MAE-gated reading of `a44f7dd8`):** At the forced entries, V's squared error is below zero's: the earlier MAE-based failure reading does not hold for the mean forecast. At the forced first reviews, V's squared error is above zero's: retention calibration remains a problem. MSE of V / zero / edge × E[q]: entry 4690.44 / 4785.28 / 4985.39; first review 4794.91 / 4568.57 / 4762.42 c². Both-modeled flips 34 of 717. The ENTRY-RETENTION selection result reads 'not demonstrated on this forced sample'.
- **Cross-tab (ENTRY-EXPOSURE by shadow admission, adaptive accounting):** would_POST 43 forks, mean D_EXPOSURE -10.50 c; would_WAIT 712 forks, mean D_EXPOSURE -12.62 c; fallback 1 forks, mean D_EXPOSURE +0.00 c; excluded 12.
- **Scope:** adaptive development description on two consumed days; no interval, no significance, no fit, no profitable-policy claim. KEEP_SELECTED changes every discretionary pull of the selected generation, not only its first. Generations within a world are related.

## Tables (admission_tables.py, copied from the outputs)

### Generation statuses (every selected generation)

| day | status | generations |
|---|---|---|
| 20261026 | ok | 166 |
| 20261027 | ok | 163 |

### Population (the frozen manifest)

| item | value |
|---|---|
| PI0 POST generations (census) | 689 |
| selected generations | 329 |
| nonempty cells | 93 |
| sum of inclusion weights | 689.0 |
| worlds ok | 380 |
| worlds world:no_path:missing_endpoint | 3 |
| worlds world:no_path:stale_future_book | 1 |

### Exposure: D_EXPOSURE = R(KEEP_SELECTED) − R(factual PI0) (c; inclusion-weighted and unweighted)

| day | side | generations | worlds | Σ weight | weighted mean D | mean D | Σ D | weighted D_selected | weighted D_focal_rest | weighted D_opposite | D > 0 | D < 0 | D = 0 | severe (< −30) | top-5 \|D\| share | gain forks | with an override |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | all | 329 | 155 | 689.00 | -9.37 | -7.60 | -2,499.95 | -9.27 | -0.10 | 0.00 | 31 | 39 | 259 | 20 | 0.37 | 73 | 268 |
| 20261026 | 0 | 82 | 53 | 182.00 | -5.38 | -7.58 | -621.74 | -5.12 | -0.26 | 0.00 | 5 | 6 | 71 | 4 | 0.91 | 15 | 69 |
| 20261026 | 1 | 84 | 51 | 137.00 | -32.20 | -11.86 | -996.24 | -32.36 | 0.16 | 0.00 | 9 | 10 | 65 | 6 | 0.74 | 18 | 65 |
| 20261027 | 0 | 87 | 52 | 227.00 | 0.97 | -0.50 | -43.69 | 1.14 | -0.17 | 0.00 | 9 | 7 | 71 | 3 | 0.70 | 17 | 75 |
| 20261027 | 1 | 76 | 49 | 143.00 | -8.97 | -11.03 | -838.28 | -8.96 | -0.02 | 0.00 | 8 | 16 | 52 | 7 | 0.76 | 23 | 59 |

### The selected generation in each arm (inclusion-weighted)

| day | side | lifetime PI0 (ms) | lifetime KEEP (ms) | P(fill) PI0 | P(fill) KEEP | filled PI0 (sh) | filled KEEP (sh) | END value PI0 (c) | END value KEEP (c) |
|---|---|---|---|---|---|---|---|---|---|
| all | all | 135.864 | 1,063.363 | 0.066 | 0.303 | 1.078 | 6.429 | -0.287 | -9.557 |
| 20261026 | 0 | 147.832 | 1,087.633 | 0.063 | 0.201 | 1.573 | 5.259 | 0.594 | -4.527 |
| 20261026 | 1 | 103.754 | 901.009 | 0.057 | 0.339 | 1.245 | 8.220 | 0.414 | -31.950 |
| 20261027 | 0 | 161.606 | 1,282.993 | 0.064 | 0.368 | 0.532 | 5.583 | 0.320 | 1.464 |
| 20261027 | 1 | 110.531 | 839.372 | 0.082 | 0.294 | 1.153 | 7.544 | -3.043 | -12.002 |

### Forecasts against matched neutral-hold targets, inclusion-weighted (c, c²; MAE secondary)

| stage | day | side | n | fill rate | mean P | Brier(P) | Brier(pooled) | MSE V | MSE 0 | MSE edge×E[q] | mean error | MAE V | MAE 0 | mean V | mean value | q\|fill error | qm\|fill error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entry | all | all | 329 | 0.303 | 0.230 | 0.152 | 0.211 | 4,043.273 | 3,981.997 | 4,098.463 | 11.099 | 17.947 | 17.222 | 1.542 | -9.557 | -0.073 | 38.568 |
| entry | 20261026 | 0 | 82 | 0.201 | 0.179 | 0.122 | 0.171 | 2,237.539 | 2,158.225 | 2,263.287 | 5.749 | 11.302 | 10.547 | 1.222 | -4.527 | -2.693 | 29.234 |
| entry | 20261026 | 1 | 84 | 0.339 | 0.263 | 0.162 | 0.226 | 11,824.889 | 11,632.343 | 11,935.575 | 33.437 | 42.612 | 41.584 | 1.487 | -31.950 | -4.250 | 101.643 |
| entry | 20261027 | 0 | 87 | 0.368 | 0.239 | 0.168 | 0.237 | 143.310 | 143.428 | 118.340 | 0.274 | 5.063 | 4.813 | 1.738 | 1.464 | 5.493 | 0.724 |
| entry | 20261027 | 1 | 76 | 0.294 | 0.248 | 0.159 | 0.208 | 5,077.218 | 5,067.203 | 5,243.961 | 13.693 | 23.225 | 22.078 | 1.691 | -12.002 | -4.237 | 52.084 |
| first_review | all | all | 299 | 0.302 | 0.252 | 0.141 | 0.211 | 3,743.448 | 4,018.163 | 4,151.929 | 5.694 | 20.092 | 17.494 | -4.250 | -9.944 | 0.626 | 17.167 |
| first_review | 20261026 | 0 | 76 | 0.197 | 0.218 | 0.127 | 0.169 | 2,297.207 | 2,304.128 | 2,429.974 | 1.734 | 13.979 | 11.089 | -3.263 | -4.997 | -3.351 | 14.892 |
| first_review | 20261026 | 1 | 75 | 0.340 | 0.293 | 0.145 | 0.226 | 12,140.365 | 13,087.192 | 13,471.526 | 30.267 | 47.147 | 45.850 | -6.040 | -36.307 | -3.831 | 88.178 |
| first_review | 20261027 | 0 | 85 | 0.341 | 0.244 | 0.138 | 0.226 | 257.149 | 126.318 | 129.766 | -3.880 | 8.146 | 4.369 | -2.885 | 0.995 | 6.475 | -15.041 |
| first_review | 20261027 | 1 | 63 | 0.340 | 0.273 | 0.160 | 0.226 | 3,790.610 | 4,517.534 | 4,651.809 | 4.282 | 23.613 | 22.300 | -6.370 | -10.652 | -2.516 | 7.480 |
| first_pull | all | all | 268 | 0.304 | 0.259 | 0.133 | 0.212 | 4,116.836 | 4,414.951 | 4,563.667 | 5.590 | 21.728 | 18.928 | -5.456 | -11.046 | 0.638 | 17.372 |
| first_pull | 20261026 | 0 | 69 | 0.165 | 0.213 | 0.105 | 0.157 | 2,474.435 | 2,480.217 | 2,618.482 | 2.126 | 14.405 | 11.272 | -3.800 | -5.927 | -2.835 | 20.358 |
| first_pull | 20261026 | 1 | 65 | 0.377 | 0.328 | 0.154 | 0.240 | 13,983.357 | 15,068.781 | 15,512.542 | 34.767 | 54.152 | 52.590 | -7.259 | -42.027 | -3.612 | 90.889 |
| first_pull | 20261027 | 0 | 75 | 0.345 | 0.252 | 0.127 | 0.228 | 306.451 | 138.962 | 144.698 | -6.021 | 8.976 | 4.690 | -4.720 | 1.300 | 6.109 | -18.370 |
| first_pull | 20261027 | 1 | 59 | 0.354 | 0.272 | 0.163 | 0.231 | 3,930.864 | 4,712.028 | 4,852.763 | 3.737 | 24.132 | 23.220 | -7.332 | -11.069 | -2.248 | 4.356 |

### Forecasts, unweighted

| stage | day | side | n | fill rate | mean P | Brier(P) | Brier(pooled) | MSE V | MSE 0 | MSE edge×E[q] | mean error | MAE V | MAE 0 | mean V | mean value | q\|fill error | qm\|fill error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entry | all | all | 329 | 0.267 | 0.207 | 0.152 | 0.196 | 3,230.151 | 3,177.085 | 3,286.137 | 9.358 | 16.339 | 15.589 | 1.375 | -7.983 | -1.369 | 37.477 |
| entry | 20261026 | 0 | 82 | 0.244 | 0.176 | 0.155 | 0.185 | 3,142.081 | 3,037.333 | 3,194.875 | 8.528 | 15.775 | 15.092 | 1.295 | -7.233 | -2.064 | 36.748 |
| entry | 20261026 | 1 | 84 | 0.262 | 0.222 | 0.142 | 0.193 | 4,770.873 | 4,692.071 | 4,815.560 | 12.921 | 20.964 | 20.084 | 1.234 | -11.686 | -2.211 | 51.281 |
| entry | 20261027 | 0 | 87 | 0.253 | 0.203 | 0.145 | 0.189 | 152.787 | 150.483 | 140.167 | 0.993 | 4.936 | 4.538 | 1.398 | 0.405 | 3.244 | 3.951 |
| entry | 20261027 | 1 | 76 | 0.316 | 0.228 | 0.168 | 0.218 | 5,145.044 | 5,118.073 | 5,295.497 | 15.891 | 24.892 | 23.806 | 1.591 | -14.300 | -4.248 | 56.164 |
| first_review | all | all | 299 | 0.264 | 0.229 | 0.139 | 0.194 | 2,955.433 | 3,210.872 | 3,333.519 | 4.017 | 18.276 | 15.747 | -4.319 | -8.335 | -0.848 | 14.734 |
| first_review | 20261026 | 0 | 76 | 0.237 | 0.202 | 0.154 | 0.181 | 3,193.502 | 3,263.452 | 3,448.750 | 5.190 | 17.899 | 15.929 | -2.852 | -8.042 | -2.733 | 24.614 |
| first_review | 20261026 | 1 | 75 | 0.253 | 0.255 | 0.130 | 0.189 | 4,967.279 | 5,232.141 | 5,378.626 | 8.056 | 24.367 | 21.604 | -5.094 | -13.150 | -1.988 | 33.450 |
| first_review | 20261027 | 0 | 85 | 0.224 | 0.205 | 0.118 | 0.175 | 245.002 | 116.856 | 140.248 | -2.530 | 7.792 | 3.712 | -3.034 | -0.504 | 3.804 | -9.425 |
| first_review | 20261027 | 1 | 63 | 0.365 | 0.261 | 0.160 | 0.242 | 3,930.113 | 4,915.633 | 5,068.238 | 6.625 | 25.626 | 24.794 | -6.899 | -13.524 | -2.275 | 11.498 |
| first_pull | all | all | 268 | 0.272 | 0.236 | 0.136 | 0.198 | 3,294.831 | 3,578.011 | 3,715.809 | 3.614 | 20.020 | 17.276 | -5.732 | -9.346 | -0.614 | 13.225 |
| first_pull | 20261026 | 0 | 69 | 0.217 | 0.197 | 0.133 | 0.173 | 3,508.859 | 3,584.744 | 3,787.202 | 5.692 | 19.032 | 16.893 | -3.383 | -9.075 | -2.253 | 29.198 |
| first_pull | 20261026 | 1 | 65 | 0.277 | 0.284 | 0.136 | 0.200 | 5,736.123 | 6,033.625 | 6,204.391 | 9.080 | 28.031 | 24.697 | -6.323 | -15.404 | -1.538 | 33.700 |
| first_pull | 20261027 | 0 | 75 | 0.227 | 0.208 | 0.114 | 0.177 | 306.000 | 130.189 | 159.796 | -4.917 | 8.760 | 4.028 | -5.321 | -0.404 | 3.409 | -13.408 |
| first_pull | 20261027 | 1 | 59 | 0.390 | 0.266 | 0.167 | 0.252 | 4,154.328 | 5,247.625 | 5,411.011 | 6.008 | 26.665 | 26.390 | -8.348 | -14.356 | -1.797 | 6.470 |

### Probability bins (inclusion-weighted)

| stage | P from | P to | rows | mean P | fill rate |
|---|---|---|---|---|---|
| entry | 0.000 | 0.010 | 0 | — | — |
| entry | 0.010 | 0.020 | 19 | 0.015 | 0.000 |
| entry | 0.020 | 0.050 | 64 | 0.035 | 0.093 |
| entry | 0.050 | 0.100 | 43 | 0.069 | 0.083 |
| entry | 0.100 | 0.200 | 59 | 0.140 | 0.146 |
| entry | 0.200 | 0.400 | 82 | 0.289 | 0.329 |
| entry | 0.400 | 1.000 | 62 | 0.497 | 0.712 |
| first_review | 0.000 | 0.010 | 10 | 0.007 | 0.000 |
| first_review | 0.010 | 0.020 | 32 | 0.016 | 0.075 |
| first_review | 0.020 | 0.050 | 43 | 0.034 | 0.049 |
| first_review | 0.050 | 0.100 | 45 | 0.069 | 0.069 |
| first_review | 0.100 | 0.200 | 37 | 0.140 | 0.107 |
| first_review | 0.200 | 0.400 | 58 | 0.314 | 0.308 |
| first_review | 0.400 | 1.000 | 74 | 0.542 | 0.712 |
| first_pull | 0.000 | 0.010 | 10 | 0.007 | 0.000 |
| first_pull | 0.010 | 0.020 | 28 | 0.015 | 0.000 |
| first_pull | 0.020 | 0.050 | 28 | 0.031 | 0.086 |
| first_pull | 0.050 | 0.100 | 46 | 0.068 | 0.052 |
| first_pull | 0.100 | 0.200 | 31 | 0.136 | 0.085 |
| first_pull | 0.200 | 0.400 | 54 | 0.313 | 0.285 |
| first_pull | 0.400 | 1.000 | 71 | 0.538 | 0.728 |

### First-review selection: PULL against 500 matched draws (weights carried)

| eligible | PULL | PULL mean value | all mean | null mean | null 5% | null 95% | percentile | strata | strata that can change | rows in them | thin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 299 | 233 | -12.378 | -9.944 | -12.699 | -13.264 | -12.165 | 0.806 | 91 | 45 | 166 | false |

### Entry → first review: input changes and new observations (both stages modeled)

| subset | n | market_flow changed | fair changed | order_state changed | time changed | new PF frame (age) | new BN trade | new BN depth | new PF receipt | weighted dV | weighted V entry | weighted V review | flip share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 299 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 | 0.946 | 0.933 | 0.967 | -5.814 | 1.563 | -4.250 | 0.779 |
| any_new_observation | 299 | 1.000 | 1.000 | 1.000 | 1.000 | 0.967 | 0.946 | 0.933 | 0.967 | -5.814 | 1.563 | -4.250 | 0.779 |
| no_new_observation | 0 | — | — | — | — | — | — | — | — | — | — | — | — |

### Cross-tab: the 768 ENTRY-EXPOSURE forks by ENTRY-RETENTION shadow admission (c; unweighted; adaptive accounting)

| day | side | admission | forks | worlds | mean D_EXPOSURE | Σ D_EXPOSURE | mean D_initial | mean D_rest | top-5 \|D\| share |
|---|---|---|---|---|---|---|---|---|---|
| 20261024 | 0 | excluded | 1 | 1 | — | — | — | — | — |
| 20261024 | 0 | would_POST | 8 | 8 | 2.20 | 17.56 | 2.20 | 0.00 | 1.00 |
| 20261024 | 0 | would_WAIT | 87 | 87 | -16.71 | -1,454.07 | -16.86 | 0.15 | 0.53 |
| 20261024 | 1 | excluded | 1 | 1 | — | — | — | — | — |
| 20261024 | 1 | would_POST | 10 | 10 | 0.64 | 6.42 | 7.38 | -6.74 | 1.00 |
| 20261024 | 1 | would_WAIT | 85 | 85 | -4.10 | -348.48 | -4.99 | 0.89 | 0.39 |
| 20261025 | 0 | excluded | 1 | 1 | — | — | — | — | — |
| 20261025 | 0 | would_POST | 8 | 8 | -45.03 | -360.22 | -65.65 | 20.63 | 1.00 |
| 20261025 | 0 | would_WAIT | 87 | 87 | -12.05 | -1,047.99 | -12.05 | -0.00 | 0.52 |
| 20261025 | 1 | excluded | 1 | 1 | — | — | — | — | — |
| 20261025 | 1 | would_POST | 7 | 7 | -16.77 | -117.41 | -19.29 | 2.51 | 1.00 |
| 20261025 | 1 | would_WAIT | 88 | 88 | -3.12 | -274.95 | -3.42 | 0.30 | 0.42 |
| 20261026 | 0 | excluded | 3 | 3 | — | — | — | — | — |
| 20261026 | 0 | would_POST | 2 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 20261026 | 0 | would_WAIT | 91 | 91 | -15.87 | -1,444.44 | -15.92 | 0.05 | 0.45 |
| 20261026 | 1 | excluded | 3 | 3 | — | — | — | — | — |
| 20261026 | 1 | fallback | 1 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| 20261026 | 1 | would_WAIT | 92 | 92 | -7.73 | -710.88 | -7.73 | -0.00 | 0.53 |
| 20261027 | 0 | excluded | 1 | 1 | — | — | — | — | — |
| 20261027 | 0 | would_POST | 4 | 4 | -0.30 | -1.18 | -0.30 | 0.00 | 1.00 |
| 20261027 | 0 | would_WAIT | 91 | 91 | -26.72 | -2,431.41 | -28.70 | 1.98 | 0.55 |
| 20261027 | 1 | excluded | 1 | 1 | — | — | — | — | — |
| 20261027 | 1 | would_POST | 4 | 4 | 0.79 | 3.16 | 3.75 | -2.96 | 1.00 |
| 20261027 | 1 | would_WAIT | 91 | 91 | -14.00 | -1,273.77 | -14.17 | 0.17 | 0.44 |
| all | all | excluded | 12 | 6 | — | — | — | — | — |
| all | all | fallback | 1 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| all | all | would_POST | 43 | 42 | -10.50 | -451.66 | -12.91 | 2.40 | 0.81 |
| all | all | would_WAIT | 712 | 376 | -12.62 | -8,985.99 | -13.06 | 0.44 | 0.12 |

### RETENTION v2: the forced forks' forecasts by mean scores (c²; supersedes the MAE gate)

| stage | day | side | n | MSE V | MSE 0 | MSE edge×E[q] | mean error | Brier(P) | Brier(pooled) | MAE V | MAE 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entry | all | all | 755 | 4,690.44 | 4,785.28 | 4,985.39 | -0.44 | 0.23 | 0.24 | 35.54 | 27.63 |
| entry | 20261024 | 0 | 95 | 3,510.99 | 3,435.98 | 3,649.30 | 5.42 | 0.23 | 0.22 | 30.77 | 22.86 |
| entry | 20261024 | 1 | 95 | 2,871.43 | 2,676.07 | 2,778.60 | -10.56 | 0.23 | 0.25 | 34.16 | 24.12 |
| entry | 20261025 | 0 | 95 | 6,243.99 | 6,252.73 | 6,480.43 | 4.41 | 0.22 | 0.25 | 39.90 | 32.55 |
| entry | 20261025 | 1 | 95 | 2,460.75 | 2,114.96 | 2,286.96 | -10.18 | 0.23 | 0.24 | 31.19 | 21.01 |
| entry | 20261026 | 0 | 93 | 2,516.26 | 2,776.39 | 2,938.23 | 3.38 | 0.23 | 0.24 | 28.29 | 23.38 |
| entry | 20261026 | 1 | 92 | 5,087.71 | 4,837.67 | 4,944.27 | -11.19 | 0.25 | 0.25 | 35.86 | 27.60 |
| entry | 20261027 | 0 | 95 | 10,784.39 | 12,038.57 | 12,501.66 | 17.09 | 0.22 | 0.23 | 47.11 | 41.67 |
| entry | 20261027 | 1 | 95 | 4,014.77 | 4,109.23 | 4,259.31 | -2.15 | 0.22 | 0.24 | 36.90 | 27.77 |
| first_review | all | all | 718 | 4,794.91 | 4,568.57 | 4,762.42 | -7.09 | 0.23 | 0.24 | 39.83 | 27.04 |
| first_review | 20261024 | 0 | 90 | 3,751.38 | 3,395.61 | 3,610.15 | 1.04 | 0.23 | 0.22 | 35.22 | 22.90 |
| first_review | 20261024 | 1 | 90 | 3,188.24 | 2,824.73 | 2,926.89 | -18.80 | 0.23 | 0.25 | 40.16 | 25.46 |
| first_review | 20261025 | 0 | 91 | 6,764.02 | 6,517.68 | 6,775.54 | -1.21 | 0.22 | 0.25 | 45.44 | 33.65 |
| first_review | 20261025 | 1 | 88 | 3,268.69 | 2,257.43 | 2,348.17 | -21.83 | 0.24 | 0.24 | 39.85 | 21.99 |
| first_review | 20261026 | 0 | 89 | 2,348.43 | 2,712.49 | 2,896.94 | -0.80 | 0.21 | 0.24 | 29.05 | 22.27 |
| first_review | 20261026 | 1 | 89 | 3,914.67 | 3,254.45 | 3,380.14 | -14.66 | 0.24 | 0.25 | 37.21 | 23.23 |
| first_review | 20261027 | 0 | 89 | 11,091.37 | 11,735.27 | 12,128.88 | 9.19 | 0.24 | 0.24 | 52.06 | 40.95 |
| first_review | 20261027 | 1 | 92 | 4,026.74 | 3,838.53 | 4,018.92 | -9.81 | 0.22 | 0.24 | 39.59 | 25.76 |
| first_pull | all | all | 714 | 4,789.93 | 4,583.37 | 4,779.15 | -7.66 | 0.23 | 0.24 | 39.69 | 27.03 |
| first_pull | 20261024 | 0 | 88 | 3,669.35 | 3,395.44 | 3,612.80 | -1.20 | 0.23 | 0.22 | 34.24 | 22.49 |
| first_pull | 20261024 | 1 | 90 | 3,123.55 | 2,824.73 | 2,926.33 | -19.71 | 0.23 | 0.25 | 39.51 | 25.46 |
| first_pull | 20261025 | 0 | 89 | 6,909.99 | 6,654.04 | 6,916.50 | -1.84 | 0.22 | 0.25 | 46.14 | 34.07 |
| first_pull | 20261025 | 1 | 88 | 3,248.88 | 2,257.43 | 2,357.40 | -22.22 | 0.23 | 0.24 | 39.58 | 21.99 |
| first_pull | 20261026 | 0 | 89 | 2,348.43 | 2,712.49 | 2,896.94 | -0.80 | 0.21 | 0.24 | 29.05 | 22.27 |
| first_pull | 20261026 | 1 | 89 | 3,914.67 | 3,254.45 | 3,380.14 | -14.66 | 0.24 | 0.25 | 37.21 | 23.23 |
| first_pull | 20261027 | 0 | 89 | 11,091.72 | 11,735.27 | 12,129.05 | 9.17 | 0.24 | 0.24 | 52.08 | 40.95 |
| first_pull | 20261027 | 1 | 92 | 4,027.36 | 3,838.53 | 4,020.11 | -9.89 | 0.22 | 0.24 | 39.67 | 25.76 |

### Declared predicates and readings

| item | value |
|---|---|
| day 20261026 weighted mean D | -16.9007 |
| day 20261027 weighted mean D | -2.8719 |
| gain forks | 73 |
| thin | false |
| reading admitted_longer_beneficial | false |
| reading admitted_longer_harmful | true |
| reading not_identified | false |
| mse_beats_zero entry | false |
| mse_beats_zero first_pull | true |
| mse_beats_zero first_review | true |
| review_discriminates | false |
| retention_v2 entry_mse_beats_zero | true |
| retention_v2 review_mse_beats_zero | false |
| retention_v2 both-modeled flips | 34 of 717 |

## Review 99fb165d (findings folded in)

- **P1:** the MAE gate is superseded by RETENTION v2 above (MSE, calibration, side / stage detail); MAE stays secondary; the perfect-mean mixture is a selftest control; the ENTRY-RETENTION receipt and gate stay as provenance.
- **P2 (masks):** transitions use both-modeled forks (RETENTION v2 flips; this round by construction); the cross-tab separates model-valid WAIT from fallback.
- **P2 (new information):** new observations are classified per source from ages (ms) and the PF receipt time (tables); a group-change indicator is not read as new information.
- **P2 (wording):** "not demonstrated on this forced sample"; this round measures PI0's actual admissions.
- **P3:** the pilot projection is write-once (tools/pilot_projection_adm.py refuses an existing record).

## Provenance

- Mutation checks: the final suite 29 / 29; run 1 28 / 29 (a thin positive case added), run 2 29 / 29 (before the per-generation guard was set to the contract's 120 s). The checker's synthetic selftest passed before the run.
- Pilot (part 0): 111 CPU s over 30 of 329 generations, peak 1156 MiB; projection 0.597 core-hours against the 1 ceiling, so the rest ran.

## Cost

- 0.576 core-hours of CPU over every run_light job under logs/, recursive (the ceiling is 1); 108 jobs, peak 1179 MiB; check G true.

## Limits

- Two consumed days (Oct 26–27) of one adaptive development population; inclusion weights estimate the 689 PI0 POSTs of the 380 replay worlds, not other days.
- Gross values at the common 5 s END mark; remaining queue priority at END has no value.

