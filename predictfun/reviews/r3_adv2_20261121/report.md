# R3 ADV2-RICH v1 — the fixed paired-advantage candidate on consumed Oct 26–27: prediction diagnostic, local control and bounded own-state replay

Review 315979d205e7788daf69e277aa1bd6e0cd881e0e (one next priority). Freeze 1 `551185f2` (evaluation sample), freeze 2 `c8655f75` (candidate, controls, evaluator, replay, declared checks), model hashes `2bec465d` (committed before any Oct 26–27 label was read), freeze 2b `259e6c56` (the replay summary v2). Evidence as of 2026-11-21T06:11:50.521874+00:00. Development session Claude aea5bfb9.

## Verdict (every sentence chosen by a computed predicate)

- **Declared checks (v2):** all true ([checks_v2.json](checks_v2.json)); falsifier v2 passed ([checks_falsifier_v2.json](checks_falsifier_v2.json)). v1 ([checks.json](checks.json), frozen r3_checks.py) has G_replay false: its G required a binding check on every ok reached pair, and 2 ok KEEP pairs at the final tick made none, correctly (no later tick, the other side already decided; 2 zero-callback ok pairs in all). v2 (r3_checks_v2.py) changes only that predicate; v1 stays as written.
- **Scope:** this round measures prediction skill and local decision value on two consumed development days (Oct 26–27) with a model fitted on Oct 24–25. It is development evidence only: no interval (two days), no validation, no promotion, no refit.
- **Incremental prediction (offline, all ok rows, weighted MSE):** RICH is below the zero baseline in 4 of 8 day × side × head cells, below the Oct 24–25 training mean in 4, below FAIR in 6 and below MECH in 6.
- **Plan-v2 screens (RICH, 16 day × side × head × action groups):** PASS 1, FAIL 11, INSUFFICIENT_SUPPORT 4, NOT_EVALUABLE 0. FAIR: PASS 3, FAIL 9, INSUFFICIENT_SUPPORT 4, NOT_EVALUABLE 0. MECH: PASS 1, FAIL 11, INSUFFICIENT_SUPPORT 4, NOT_EVALUABLE 0.
- **Local decision value (pi0-conditional, never a policy return):** RICH's value exceeds its fallback-default policy (never post on the post head, always keep on the keep head) in 4 of 8 cells and the best constant policy in 1; it is above FAIR's value in 3 and above MECH's in 3.
- **Support:** 10 of 18,132 evaluation rows (0.1 %) fall outside the candidate support and take the fallback (outside_training_range 10).
- **Local action-information control (RICH, day × head):** lift above the matched draw mean in 2 of 4 scopes; the realized value is above the draws' 95th percentile in 0 of 4. This is additive local selection, not a p-value.
- **Replay 20261026 (188 worlds, joint own-state 5 s episodes; diagnostic, not the R4 gate):** RICH total -3,216.1 c against PI0 -265.1 c and N0 0 c; paired RICH − PI0 -2,951.0 c (31 worlds above, 56 below, of 188); RICH − ENTRY_PA -514.8 c (20 worlds above, 26 below, of 188); FAIR − PI0 -5,083.8 c (29 worlds above, 84 below, of 188); MECH − PI0 -6,114.1 c (25 worlds above, 77 below, of 188); ENTRY_ONLY − PI0 -2,370.2 c (25 worlds above, 49 below, of 188); RETAIN_ONLY − PI0 -355.4 c (2 worlds above, 5 below, of 188).
- **Replay 20261027 (192 worlds, joint own-state 5 s episodes; diagnostic, not the R4 gate):** RICH total -3,252.7 c against PI0 +31.6 c and N0 0 c; paired RICH − PI0 -3,284.3 c (23 worlds above, 64 below, of 192); RICH − ENTRY_PA -844.2 c (23 worlds above, 33 below, of 192); FAIR − PI0 -4,873.0 c (36 worlds above, 82 below, of 192); MECH − PI0 -5,868.5 c (28 worlds above, 81 below, of 192); ENTRY_ONLY − PI0 -2,409.1 c (18 worlds above, 50 below, of 192); RETAIN_ONLY − PI0 -78.0 c (3 worlds above, 4 below, of 192).
- **State shift (reached RICH states, frozen pi0 arms):** RICH's MSE relative to zero is higher at reached states than on the neutral sample in 3 of 4 day × head cells (reached pairs: no_eligible_decision 4, ok 1,516).

## Which layer limits the candidate (computed readings, no tuning)

| day | side | head | MSE RICH / zero | below zero | below train mean | below FAIR | below MECH | value | fallback-default value | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | keep | 0.993 | true | true | true | true | -0.299 | -0.599 | 0.155 |
| 20261026 | bid | post | 1.068 | false | false | true | false | -0.687 | 0.000 | 0.200 |
| 20261026 | ask | keep | 0.968 | true | true | true | true | 0.149 | -0.838 | 0.296 |
| 20261026 | ask | post | 1.032 | false | false | false | false | -0.949 | 0.000 | 0.352 |
| 20261027 | bid | keep | 0.961 | true | true | true | true | -0.156 | -0.557 | 0.199 |
| 20261027 | bid | post | 1.019 | false | false | true | true | -0.225 | 0.000 | 0.385 |
| 20261027 | ask | keep | 0.976 | true | true | true | true | -0.252 | -1.273 | 0.293 |
| 20261027 | ask | post | 1.001 | false | false | false | true | -0.381 | 0.000 | 0.461 |

| day | head | reached pairs | reached nonzero | reached MSE RICH / zero | neutral MSE RICH / zero (weighted) | neutral (unweighted) |
|---|---|---|---|---|---|---|
| 20261026 | keep | 376 | 9 | 1.012 | 0.977 | 0.976 |
| 20261026 | post | 376 | 18 | 1.046 | 1.044 | 1.021 |
| 20261027 | keep | 380 | 9 | 1.039 | 0.974 | 0.973 |
| 20261027 | post | 384 | 15 | 0.998 | 1.006 | 1.014 |

Readings: a ratio below 1 means the prediction beats predicting zero; the value columns are sampling-weighted cents per opportunity (local, pi0-conditional); the oracle is the value of acting on the realized sign. Two consumed days: no interval.

## Tables (r3_tables.py, copied from the outputs)

### Declared checks

| check | v1 (r3_checks.py, frozen) | v2 (r3_checks_v2.py; G superseded) |
|---|---|---|
| A_extraction | true | true |
| B_fit | true | true |
| C_order | true | true |
| D_sample | true | true |
| E_readings | true | true |
| F_control | true | true |
| G_replay | false | true |
| H_suites | true | true |
| I_resources | true | true |
| all_true | false | true |

Falsifier v1 (each check on a clean copy of its inputs, then after one corruption):

| check | clean copy | corrupted copy |
|---|---|---|
| E | true | false |
| G | false | false |
| H | true | false |
| falsifier_ok | false |  |

Falsifier v2 (G2 on copies of the real parts; E and H as in v1):

| case | corruption applied | G2 |
|---|---|---|
| G2 binding_failure | true | false |
| G2 cap_ok_false | true | false |
| G2 clean | true | true |
| G2 zero_callback_at_a_non_final_tick | true | false |
| falsifier_ok |  | true |

### Prediction error against the baselines (all ok rows of the group; sampling-weighted; cents squared / cents)

| day | side | head | rows | meaningful nonzero originals | MSE zero | MSE train mean | MSE RICH | MSE FAIR | MSE MECH | RICH / zero | MAE RICH | MAE zero | Spearman RICH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | keep | 1,509 | 35 | 156.07 | 155.76 | 154.97 | 156.98 | 156.96 | 0.993 | 2.14 | 0.91 | 0.103 |
| 20261026 | bid | post | 3,018 | 68 | 213.33 | 212.80 | 227.78 | 247.39 | 225.88 | 1.068 | 3.35 | 1.28 | 0.007 |
| 20261026 | ask | keep | 1,509 | 37 | 290.33 | 289.81 | 281.07 | 292.77 | 281.63 | 0.968 | 3.41 | 1.43 | 0.072 |
| 20261026 | ask | post | 3,018 | 86 | 425.57 | 424.57 | 439.22 | 436.78 | 438.24 | 1.032 | 3.79 | 1.86 | 0.006 |
| 20261027 | bid | keep | 1,513 | 35 | 117.00 | 116.73 | 112.47 | 118.47 | 117.40 | 0.961 | 2.09 | 0.96 | 0.029 |
| 20261027 | bid | post | 3,026 | 80 | 229.89 | 229.36 | 234.16 | 260.01 | 241.47 | 1.019 | 3.40 | 1.66 | 0.028 |
| 20261027 | ask | keep | 1,513 | 41 | 602.08 | 601.21 | 587.65 | 603.98 | 597.58 | 0.976 | 3.46 | 1.86 | 0.034 |
| 20261027 | ask | post | 3,026 | 70 | 564.06 | 563.02 | 564.76 | 564.42 | 569.19 | 1.001 | 3.88 | 2.12 | 0.014 |

### Local decision value per opportunity (cents; sampling-weighted; pi0-conditional, never a policy return)

| day | side | head | oracle | always POST / KEEP | RICH | FAIR | MECH | RICH regret | RICH missed + (sum w) | RICH admitted − (sum w) | \|Y\| > 30 c events |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | keep | 0.155 | -0.599 | -0.299 | -0.006 | -0.096 | 0.454 | 1,645.1 | -5,104.1 | 16 |
| 20261026 | bid | post | 0.200 | -0.882 | -0.687 | -0.367 | -0.574 | 0.887 | 767.2 | -12,408.8 | 43 |
| 20261026 | ask | keep | 0.296 | -0.838 | 0.149 | -0.259 | -0.066 | 0.147 | 1,667.4 | -522.9 | 12 |
| 20261026 | ask | post | 0.352 | -1.159 | -0.949 | -0.282 | -0.775 | 1.301 | 2,050.4 | -17,271.8 | 46 |
| 20261027 | bid | keep | 0.199 | -0.557 | -0.156 | -0.264 | -0.046 | 0.356 | 2,537.1 | -2,687.5 | 15 |
| 20261027 | bid | post | 0.385 | -0.887 | -0.225 | -0.483 | -0.516 | 0.609 | 2,989.5 | -5,961.3 | 49 |
| 20261027 | ask | keep | 0.293 | -1.273 | -0.252 | -0.245 | -0.047 | 0.545 | 3,351.4 | -4,653.3 | 17 |
| 20261027 | ask | post | 0.461 | -1.195 | -0.381 | -0.254 | -0.386 | 0.842 | 3,192.6 | -9,181.3 | 40 |

d = 1 means POST (post head) or KEEP (keep head); always-0 (never post / always pull) is 0 by definition.

### Plan v2 prediction screens (per day / side / head / selected group)

| day | side | head | candidate | group | verdict | originals | meaningful nonzero | exact nonzero | mean pred | mean Y | why |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | keep | FAIR | KEEP | INSUFFICIENT_SUPPORT | 593 | 13 | 13 | 0.567 | -0.016 | 593 originals, 13 meaningful nonzero originals |
| 20261026 | bid | keep | FAIR | PULL | FAIL | 914 | 22 | 22 | -1.270 | -0.979 | sign ok, amount outside |
| 20261026 | bid | post | FAIR | POST | FAIL | 1,068 | 37 | 37 | 1.561 | -0.878 | sign wrong, amount outside |
| 20261026 | bid | post | FAIR | WAIT | FAIL | 1,127 | 32 | 32 | -2.180 | -0.885 | sign ok, amount outside |
| 20261026 | ask | keep | FAIR | KEEP | INSUFFICIENT_SUPPORT | 615 | 13 | 13 | 1.078 | -0.635 | 615 originals, 13 meaningful nonzero originals |
| 20261026 | ask | keep | FAIR | PULL | FAIL | 889 | 24 | 24 | -1.599 | -0.983 | sign ok, amount outside |
| 20261026 | ask | post | FAIR | POST | FAIL | 876 | 26 | 26 | 1.473 | -0.877 | sign wrong, amount outside |
| 20261026 | ask | post | FAIR | WAIT | PASS | 1,282 | 66 | 66 | -1.549 | -1.293 | sign ok, amount ok |
| 20261027 | bid | keep | FAIR | KEEP | INSUFFICIENT_SUPPORT | 624 | 12 | 12 | 0.560 | -0.634 | 624 originals, 12 meaningful nonzero originals |
| 20261027 | bid | keep | FAIR | PULL | FAIL | 887 | 21 | 21 | -1.092 | -0.500 | sign ok, amount outside |
| 20261027 | bid | post | FAIR | POST | FAIL | 1,092 | 40 | 40 | 1.477 | -1.192 | sign wrong, amount outside |
| 20261027 | bid | post | FAIR | WAIT | FAIL | 1,148 | 42 | 42 | -1.955 | -0.679 | sign ok, amount outside |
| 20261027 | ask | keep | FAIR | KEEP | INSUFFICIENT_SUPPORT | 618 | 18 | 18 | 1.044 | -0.599 | 618 originals, 18 meaningful nonzero originals |
| 20261027 | ask | keep | FAIR | PULL | PASS | 894 | 23 | 23 | -1.596 | -1.739 | sign ok, amount ok |
| 20261027 | ask | post | FAIR | POST | FAIL | 872 | 31 | 31 | 1.379 | -0.730 | sign wrong, amount outside |
| 20261027 | ask | post | FAIR | WAIT | PASS | 1,258 | 39 | 39 | -1.668 | -1.444 | sign ok, amount ok |
| 20261026 | bid | keep | MECH | KEEP | INSUFFICIENT_SUPPORT | 474 | 6 | 6 | 0.487 | -0.306 | 474 originals, 6 meaningful nonzero originals |
| 20261026 | bid | keep | MECH | PULL | FAIL | 1,033 | 29 | 29 | -1.155 | -0.735 | sign ok, amount outside |
| 20261026 | bid | post | MECH | POST | FAIL | 1,104 | 32 | 32 | 1.439 | -1.176 | sign wrong, amount outside |
| 20261026 | bid | post | MECH | WAIT | FAIL | 1,066 | 36 | 36 | -1.549 | -0.603 | sign ok, amount outside |
| 20261026 | ask | keep | MECH | KEEP | INSUFFICIENT_SUPPORT | 563 | 14 | 14 | 1.123 | -0.178 | 563 originals, 14 meaningful nonzero originals |
| 20261026 | ask | keep | MECH | PULL | FAIL | 941 | 23 | 23 | -1.627 | -1.236 | sign ok, amount outside |
| 20261026 | ask | post | MECH | POST | FAIL | 1,078 | 41 | 41 | 1.504 | -1.737 | sign wrong, amount outside |
| 20261026 | ask | post | MECH | WAIT | FAIL | 1,070 | 49 | 49 | -2.134 | -0.693 | sign ok, amount outside |
| 20261027 | bid | keep | MECH | KEEP | INSUFFICIENT_SUPPORT | 485 | 7 | 7 | 0.510 | -0.138 | 485 originals, 7 meaningful nonzero originals |
| 20261027 | bid | keep | MECH | PULL | FAIL | 1,026 | 26 | 26 | -1.194 | -0.753 | sign ok, amount outside |
| 20261027 | bid | post | MECH | POST | FAIL | 1,149 | 39 | 39 | 1.475 | -1.004 | sign wrong, amount outside |
| 20261027 | bid | post | MECH | WAIT | FAIL | 1,041 | 44 | 44 | -1.466 | -0.763 | sign ok, amount outside |
| 20261027 | ask | keep | MECH | KEEP | INSUFFICIENT_SUPPORT | 592 | 10 | 10 | 1.020 | -0.121 | 592 originals, 10 meaningful nonzero originals |
| 20261027 | ask | keep | MECH | PULL | PASS | 920 | 31 | 31 | -1.611 | -2.014 | sign ok, amount ok |
| 20261027 | ask | post | MECH | POST | FAIL | 1,095 | 35 | 35 | 1.464 | -0.814 | sign wrong, amount outside |
| 20261027 | ask | post | MECH | WAIT | FAIL | 1,020 | 35 | 35 | -2.114 | -1.540 | sign ok, amount outside |
| 20261026 | bid | keep | RICH | KEEP | INSUFFICIENT_SUPPORT | 650 | 10 | 10 | 0.852 | -0.693 | 650 originals, 10 meaningful nonzero originals |
| 20261026 | bid | keep | RICH | PULL | FAIL | 857 | 25 | 25 | -1.857 | -0.529 | sign ok, amount outside |
| 20261026 | bid | post | RICH | POST | FAIL | 1,105 | 33 | 33 | 2.144 | -1.312 | sign wrong, amount outside |
| 20261026 | bid | post | RICH | WAIT | FAIL | 1,043 | 36 | 36 | -2.139 | -0.410 | sign ok, amount outside |
| 20261026 | ask | keep | RICH | KEEP | INSUFFICIENT_SUPPORT | 753 | 12 | 12 | 1.627 | 0.298 | 753 originals, 12 meaningful nonzero originals |
| 20261026 | ask | keep | RICH | PULL | FAIL | 751 | 25 | 25 | -2.626 | -1.985 | sign ok, amount outside |
| 20261026 | ask | post | RICH | POST | FAIL | 1,071 | 38 | 38 | 1.511 | -1.852 | sign wrong, amount outside |
| 20261026 | ask | post | RICH | WAIT | FAIL | 1,054 | 51 | 51 | -2.564 | -0.431 | sign ok, amount outside |
| 20261027 | bid | keep | RICH | KEEP | INSUFFICIENT_SUPPORT | 662 | 9 | 9 | 0.746 | -0.353 | 662 originals, 9 meaningful nonzero originals |
| 20261027 | bid | keep | RICH | PULL | FAIL | 849 | 24 | 24 | -1.624 | -0.714 | sign ok, amount outside |
| 20261027 | bid | post | RICH | POST | FAIL | 1,145 | 34 | 34 | 1.836 | -0.433 | sign wrong, amount outside |
| 20261027 | bid | post | RICH | WAIT | FAIL | 1,053 | 47 | 47 | -1.950 | -1.374 | sign ok, amount outside |
| 20261027 | ask | keep | RICH | KEEP | INSUFFICIENT_SUPPORT | 750 | 11 | 11 | 1.460 | -0.508 | 750 originals, 11 meaningful nonzero originals |
| 20261027 | ask | keep | RICH | PULL | PASS | 762 | 30 | 30 | -1.990 | -2.029 | sign ok, amount ok |
| 20261027 | ask | post | RICH | POST | FAIL | 1,131 | 30 | 30 | 1.514 | -0.705 | sign wrong, amount outside |
| 20261027 | ask | post | RICH | WAIT | FAIL | 996 | 41 | 41 | -2.373 | -1.770 | sign ok, amount outside |

Verdict counts:

| candidate | FAIL | INSUFFICIENT_SUPPORT | PASS |
|---|---|---|---|
| FAIR | 9 | 4 | 3 |
| MECH | 11 | 4 | 1 |
| RICH | 11 | 4 | 1 |

### RICH fixed-bin calibration (supported rows; both days and sides pooled per head by row-weighted sums)

| head | prediction bin | rows | nonzero labels | mean pred (row-weighted over strata) | mean Y (row-weighted over strata) |
|---|---|---|---|---|---|
| post | (-inf, -10] | 336 | 40 | -15.941 | -5.056 |
| post | (-10, -5] | 542 | 35 | -6.845 | -1.894 |
| post | (-5, -2] | 1,445 | 54 | -3.104 | -1.135 |
| post | (-2, -1] | 1,235 | 25 | -1.431 | -0.530 |
| post | (-1, -0.5] | 942 | 17 | -0.741 | 0.118 |
| post | (-0.5, 0) | 1,238 | 12 | -0.232 | -1.373 |
| post | (0, 0.5] | 1,558 | 16 | 0.253 | -0.769 |
| post | (0.5, 1] | 1,357 | 23 | 0.733 | -0.229 |
| post | (1, 2] | 1,660 | 36 | 1.437 | -1.531 |
| post | (2, 5] | 1,448 | 46 | 3.090 | -1.370 |
| post | (5, 10] | 291 | 21 | 6.657 | -2.319 |
| post | (10, inf] | 36 | 1 | 13.046 | -0.477 |
| keep | (-inf, -10] | 87 | 20 | -13.389 | -29.012 |
| keep | (-10, -5] | 257 | 27 | -6.761 | -2.725 |
| keep | (-5, -2] | 626 | 22 | -3.014 | -0.676 |
| keep | (-2, -1] | 695 | 18 | -1.460 | -0.601 |
| keep | (-1, -0.5] | 626 | 8 | -0.748 | -0.196 |
| keep | (-0.5, 0) | 928 | 9 | -0.234 | 0.070 |
| keep | (0, 0.5] | 936 | 17 | 0.249 | -0.802 |
| keep | (0.5, 1] | 644 | 6 | 0.730 | -0.051 |
| keep | (1, 2] | 710 | 7 | 1.412 | 0.397 |
| keep | (2, 5] | 488 | 10 | 2.907 | -0.710 |
| keep | (5, 10] | 36 | 2 | 6.343 | 0.180 |
| keep | (10, inf] | 1 | 0 | 10.014 | 0 |

### Coverage by support dimension (rows and meaningful-nonzero originals; both days and both sides summed; evaluation groups = RICH's selected action)

post head:

| dimension | value | train rows (Oct 24-25) | train meaningful originals | eval rows (Oct 26-27) | eval meaningful originals | eval POST | eval WAIT | eval fallback |
|---|---|---|---|---|---|---|---|---|
| c_gen | 1 | 11,481 | 329 | 11,615 | 283 | 6,122 | 5,493 | 0 |
| c_gen | 2 | 325 | 12 | 255 | 14 | 124 | 131 | 0 |
| c_gen | >=3 | 362 | 8 | 218 | 10 | 104 | 114 | 0 |
| c_inv | flat | 12,037 | 341 | 11,989 | 300 | 6,306 | 5,683 | 0 |
| c_inv | long | 65 | 2 | 55 | 4 | 22 | 33 | 0 |
| c_inv | short | 66 | 3 | 44 | 1 | 22 | 22 | 0 |
| c_other | absent | 11,846 | 338 | 11,860 | 300 | 6,222 | 5,638 | 0 |
| c_other | cancel_pending | 93 | 3 | 67 | 1 | 39 | 28 | 0 |
| c_other | live | 117 | 4 | 71 | 2 | 40 | 31 | 0 |
| c_other | post_pending | 112 | 3 | 90 | 3 | 49 | 41 | 0 |
| c_price | (20,80] | 7,534 | 225 | 7,816 | 226 | 4,079 | 3,737 | 0 |
| c_price | (5,20] | 1,551 | 43 | 1,548 | 41 | 737 | 811 | 0 |
| c_price | (80,95] | 1,728 | 55 | 1,400 | 22 | 753 | 647 | 0 |
| c_price | (95,99] | 602 | 10 | 570 | 7 | 362 | 208 | 0 |
| c_price | [1,5] | 753 | 14 | 754 | 9 | 419 | 335 | 0 |
| c_queue | (0,30] | 1,594 | 124 | 940 | 78 | 427 | 513 | 0 |
| c_queue | (30,150] | 3,285 | 133 | 1,593 | 98 | 705 | 888 | 0 |
| c_queue | >150 | 7,289 | 96 | 9,555 | 134 | 5,218 | 4,337 | 0 |
| c_rem | (20,30] | 12,168 | 346 | 12,088 | 304 | 6,350 | 5,738 | 0 |
| hbin | (0.0, 100.0] | 54 | 2 | 59 | 0 | 29 | 30 | 0 |
| hbin | (100.0, 250.0] | 168 | 4 | 160 | 4 | 84 | 76 | 0 |
| hbin | (1000.0, 2000.0] | 1,241 | 34 | 1,187 | 32 | 638 | 549 | 0 |
| hbin | (2000.0, 3500.0] | 1,784 | 50 | 1,778 | 51 | 932 | 846 | 0 |
| hbin | (250.0, 500.0] | 263 | 10 | 285 | 11 | 136 | 149 | 0 |
| hbin | (3500.0, 5000.0] | 8,092 | 238 | 8,039 | 199 | 4,215 | 3,824 | 0 |
| hbin | (500.0, 1000.0] | 566 | 14 | 580 | 16 | 316 | 264 | 0 |
| source | delayed | 3,110 | 119 | 3,006 | 88 | 1,542 | 1,464 | 0 |
| source | joint_chase | 102 | 1 | 108 | 3 | 57 | 51 | 0 |
| source | joint_post_keep | 66 | 3 | 67 | 8 | 29 | 38 | 0 |
| source | neutral | 6,084 | 185 | 6,044 | 156 | 3,188 | 2,856 | 0 |
| source | pi0 | 2,806 | 61 | 2,863 | 71 | 1,534 | 1,329 | 0 |

keep head:

| dimension | value | train rows (Oct 24-25) | train meaningful originals | eval rows (Oct 26-27) | eval meaningful originals | eval KEEP | eval PULL | eval fallback |
|---|---|---|---|---|---|---|---|---|
| c_gen | 1 | 5,638 | 156 | 5,569 | 129 | 2,610 | 2,951 | 8 |
| c_gen | 2 | 204 | 5 | 201 | 6 | 85 | 115 | 1 |
| c_gen | >=3 | 239 | 9 | 274 | 13 | 120 | 153 | 1 |
| c_inv | flat | 5,576 | 141 | 5,535 | 113 | 2,602 | 2,932 | 1 |
| c_inv | long | 239 | 11 | 243 | 16 | 105 | 135 | 3 |
| c_inv | short | 266 | 18 | 266 | 19 | 108 | 152 | 6 |
| c_other | absent | 5,457 | 152 | 5,399 | 122 | 2,504 | 2,886 | 9 |
| c_other | cancel_pending | 18 | 1 | 17 | 2 | 8 | 9 | 0 |
| c_other | live | 598 | 17 | 618 | 24 | 300 | 317 | 1 |
| c_other | post_pending | 8 | 0 | 10 | 0 | 3 | 7 | 0 |
| c_price | (20,80] | 3,741 | 102 | 3,897 | 105 | 1,790 | 2,101 | 6 |
| c_price | (5,20] | 772 | 17 | 777 | 15 | 373 | 403 | 1 |
| c_price | (80,95] | 851 | 31 | 694 | 20 | 331 | 362 | 1 |
| c_price | (95,99] | 327 | 8 | 292 | 4 | 124 | 168 | 0 |
| c_price | [1,5] | 390 | 12 | 384 | 4 | 197 | 185 | 2 |
| c_queue | (0,30] | 947 | 62 | 570 | 49 | 165 | 401 | 4 |
| c_queue | (30,150] | 1,989 | 50 | 993 | 38 | 443 | 549 | 1 |
| c_queue | 0 | 205 | 29 | 103 | 23 | 29 | 72 | 2 |
| c_queue | >150 | 2,940 | 29 | 4,378 | 38 | 2,178 | 2,197 | 3 |
| c_rem | (0,10] | 50 | 5 | 45 | 9 | 10 | 26 | 9 |
| c_rem | (10,20] | 57 | 5 | 29 | 5 | 12 | 17 | 0 |
| c_rem | (20,30] | 5,974 | 160 | 5,970 | 134 | 2,793 | 3,176 | 1 |
| hbin | (0.0, 100.0] | 162 | 1 | 137 | 1 | 60 | 77 | 0 |
| hbin | (100.0, 250.0] | 235 | 5 | 255 | 5 | 116 | 139 | 0 |
| hbin | (1000.0, 2000.0] | 1,447 | 29 | 1,420 | 28 | 678 | 740 | 2 |
| hbin | (2000.0, 3500.0] | 1,822 | 49 | 1,851 | 57 | 843 | 1,002 | 6 |
| hbin | (250.0, 500.0] | 403 | 12 | 375 | 8 | 182 | 192 | 1 |
| hbin | (3500.0, 5000.0] | 1,213 | 44 | 1,232 | 36 | 583 | 649 | 0 |
| hbin | (500.0, 1000.0] | 799 | 30 | 774 | 13 | 353 | 420 | 1 |
| source | delayed | 4,775 | 129 | 4,698 | 104 | 2,162 | 2,528 | 8 |
| source | joint_chase | 435 | 17 | 466 | 18 | 221 | 245 | 0 |
| source | joint_post_keep | 474 | 9 | 527 | 17 | 247 | 279 | 1 |
| source | neutral | 361 | 12 | 334 | 8 | 178 | 156 | 0 |
| source | pi0 | 36 | 3 | 19 | 1 | 7 | 11 | 1 |

### Local action-information control (500 draws, seed 20261121; anchors: mandatory POST and sampled KEEP picks)

| day | head | candidate | anchors | strata | singleton | no variation | realized | draw mean | lift | draw p05 | draw p95 | share of draws ≥ realized |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | keep | FAIR | 3,011 | 2,222 | 1,620 | 310 | -0.1331 | -0.1658 | 0.0327 | -0.2965 | -0.0405 | 0.358 |
| 20261026 | keep | MECH | 3,011 | 2,222 | 1,620 | 322 | -0.0816 | -0.0543 | -0.0272 | -0.1558 | 0.0168 | 0.722 |
| 20261026 | keep | RICH | 3,011 | 2,222 | 1,620 | 266 | -0.0753 | -0.0672 | -0.0081 | -0.2223 | 0.0650 | 0.480 |
| 20261026 | post | FAIR | 3,018 | 1,832 | 996 | 411 | -0.1104 | -0.1294 | 0.0190 | -0.3786 | 0.1295 | 0.454 |
| 20261026 | post | MECH | 3,018 | 1,832 | 996 | 390 | -0.5217 | -0.3200 | -0.2017 | -0.5845 | -0.0634 | 0.876 |
| 20261026 | post | RICH | 3,018 | 1,832 | 996 | 385 | -0.2171 | -0.2494 | 0.0323 | -0.4899 | -0.0129 | 0.438 |
| 20261026 | both | FAIR | 6,029 | 4,054 | 2,616 | 721 | -0.1326 | -0.1678 | 0.0352 | -0.2918 | -0.0434 | 0.340 |
| 20261026 | both | MECH | 6,029 | 4,054 | 2,616 | 712 | -0.0922 | -0.0574 | -0.0347 | -0.1458 | 0.0109 | 0.776 |
| 20261026 | both | RICH | 6,029 | 4,054 | 2,616 | 651 | -0.0787 | -0.0720 | -0.0067 | -0.2209 | 0.0528 | 0.504 |
| 20261027 | keep | FAIR | 3,023 | 2,216 | 1,589 | 305 | -0.2534 | -0.3676 | 0.1141 | -0.5500 | -0.1767 | 0.296 |
| 20261027 | keep | MECH | 3,023 | 2,216 | 1,589 | 328 | -0.0459 | -0.0521 | 0.0062 | -0.0768 | -0.0221 | 0.340 |
| 20261027 | keep | RICH | 3,023 | 2,216 | 1,589 | 271 | -0.2033 | -0.2667 | 0.0634 | -0.4504 | -0.0478 | 0.400 |
| 20261027 | post | FAIR | 3,026 | 1,822 | 964 | 416 | 0.2779 | 0.0694 | 0.2085 | -0.1775 | 0.2691 | 0.038 |
| 20261027 | post | MECH | 3,026 | 1,822 | 964 | 403 | -0.6498 | -0.5443 | -0.1054 | -0.7537 | -0.2985 | 0.706 |
| 20261027 | post | RICH | 3,026 | 1,822 | 964 | 398 | -0.2406 | -0.2164 | -0.0243 | -0.4313 | 0.0400 | 0.494 |
| 20261027 | both | FAIR | 6,049 | 4,038 | 2,553 | 721 | -0.2406 | -0.3496 | 0.1090 | -0.5374 | -0.1720 | 0.274 |
| 20261027 | both | MECH | 6,049 | 4,038 | 2,553 | 731 | -0.0605 | -0.0639 | 0.0035 | -0.0895 | -0.0352 | 0.410 |
| 20261027 | both | RICH | 6,049 | 4,038 | 2,553 | 669 | -0.2042 | -0.2581 | 0.0540 | -0.4412 | -0.0512 | 0.394 |

### Replay: episode totals per policy (cents; each world valued once; 5 s episodes)

| day | policy | worlds | value | bid value | ask value | fills | filled shares | generations | policy pulls | cancels / 100 filled shares | generations / 100 filled shares | worlds with no entry | worlds with no fill | loss (sum) | V < −30 c | cap_ok false |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | 188 | -2,635.3 | -2,220.2 | -415.1 | 158 | 1,537 | 4,611 | 4,335 | 281.5 | 300.0 | 0 | 116 | 3,896.9 | 21 | 0 |
| 20261026 | ENTRY_PA | 188 | -2,701.3 | -2,250.2 | -451.1 | 150 | 1,389 | 4,741 | 4,498 | 323.0 | 341.4 | 0 | 120 | 3,888.2 | 21 | 0 |
| 20261026 | FAIR | 188 | -5,348.9 | -4,220.2 | -1,128.7 | 334 | 2,976 | 3,177 | 2,706 | 93.4 | 106.8 | 0 | 80 | 6,271.0 | 48 | 0 |
| 20261026 | MECH | 188 | -6,379.2 | -3,481.0 | -2,898.2 | 305 | 2,676 | 3,475 | 3,051 | 115.7 | 129.8 | 0 | 86 | 7,231.3 | 46 | 0 |
| 20261026 | N0 | 188 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | 188 | 188 | 0 | 0 | 0 |
| 20261026 | PI0 | 188 | -265.1 | -286.4 | 21.3 | 29 | 336 | 319 | 251 | 76.4 | 94.8 | 88 | 172 | 370.5 | 3 | 0 |
| 20261026 | RETAIN_ONLY | 188 | -620.5 | -326.8 | -293.7 | 26 | 286 | 323 | 260 | 92.5 | 113.1 | 88 | 172 | 693.7 | 4 | 0 |
| 20261026 | RICH | 188 | -3,216.1 | -2,965.6 | -250.6 | 204 | 2,051 | 3,429 | 3,024 | 149.5 | 167.2 | 0 | 105 | 4,871.2 | 29 | 0 |
| 20261027 | ENTRY_ONLY | 192 | -2,377.5 | -1,676.1 | -701.4 | 172 | 1,387 | 4,745 | 4,482 | 322.1 | 342.2 | 0 | 125 | 3,260.2 | 29 | 0 |
| 20261027 | ENTRY_PA | 192 | -2,408.5 | -1,737.7 | -670.8 | 162 | 1,341 | 4,831 | 4,591 | 341.1 | 360.3 | 0 | 129 | 3,273.7 | 30 | 0 |
| 20261027 | FAIR | 192 | -4,841.4 | -1,284.8 | -3,556.6 | 349 | 3,167 | 3,125 | 2,693 | 87.4 | 98.7 | 0 | 77 | 6,850.1 | 44 | 0 |
| 20261027 | MECH | 192 | -5,836.9 | -2,599.7 | -3,237.2 | 339 | 3,059 | 3,603 | 3,160 | 105.1 | 117.8 | 0 | 85 | 6,902.2 | 49 | 0 |
| 20261027 | N0 | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | 192 | 192 | 0 | 0 | 0 |
| 20261027 | PI0 | 192 | 31.6 | 241.4 | -209.9 | 38 | 326 | 370 | 310 | 96.7 | 113.6 | 95 | 177 | 398.2 | 3 | 0 |
| 20261027 | RETAIN_ONLY | 192 | -46.4 | 395.5 | -441.9 | 35 | 368 | 378 | 308 | 87.3 | 102.8 | 95 | 177 | 487.4 | 3 | 0 |
| 20261027 | RICH | 192 | -3,252.7 | -1,690.3 | -1,562.4 | 223 | 1,976 | 3,401 | 3,004 | 154.3 | 172.1 | 0 | 104 | 4,375.6 | 33 | 0 |

Paired full-policy differences (per world, same tape):

| day | policy | versus | worlds | sum of differences | worlds above | worlds below |
|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | ENTRY_PA | 188 | 66.1 | 4 | 3 |
| 20261026 | FAIR | ENTRY_PA | 188 | -2,647.6 | 43 | 71 |
| 20261026 | MECH | ENTRY_PA | 188 | -3,677.9 | 40 | 67 |
| 20261026 | N0 | ENTRY_PA | 188 | 2,701.3 | 47 | 21 |
| 20261026 | PI0 | ENTRY_PA | 188 | 2,436.3 | 50 | 26 |
| 20261026 | RETAIN_ONLY | ENTRY_PA | 188 | 2,080.8 | 46 | 27 |
| 20261026 | RICH | ENTRY_PA | 188 | -514.8 | 20 | 26 |
| 20261026 | ENTRY_ONLY | PI0 | 188 | -2,370.2 | 25 | 49 |
| 20261026 | ENTRY_PA | PI0 | 188 | -2,436.3 | 26 | 50 |
| 20261026 | FAIR | PI0 | 188 | -5,083.8 | 29 | 84 |
| 20261026 | MECH | PI0 | 188 | -6,114.1 | 25 | 77 |
| 20261026 | N0 | PI0 | 188 | 265.1 | 8 | 8 |
| 20261026 | RETAIN_ONLY | PI0 | 188 | -355.4 | 2 | 5 |
| 20261026 | RICH | PI0 | 188 | -2,951.0 | 31 | 56 |
| 20261027 | ENTRY_ONLY | ENTRY_PA | 192 | 31.0 | 4 | 4 |
| 20261027 | FAIR | ENTRY_PA | 192 | -2,432.9 | 49 | 66 |
| 20261027 | MECH | ENTRY_PA | 192 | -3,428.4 | 37 | 63 |
| 20261027 | N0 | ENTRY_PA | 192 | 2,408.5 | 47 | 16 |
| 20261027 | PI0 | ENTRY_PA | 192 | 2,440.1 | 50 | 18 |
| 20261027 | RETAIN_ONLY | ENTRY_PA | 192 | 2,362.1 | 49 | 19 |
| 20261027 | RICH | ENTRY_PA | 192 | -844.2 | 23 | 33 |
| 20261027 | ENTRY_ONLY | PI0 | 192 | -2,409.1 | 18 | 50 |
| 20261027 | ENTRY_PA | PI0 | 192 | -2,440.1 | 18 | 50 |
| 20261027 | FAIR | PI0 | 192 | -4,873.0 | 36 | 82 |
| 20261027 | MECH | PI0 | 192 | -5,868.5 | 28 | 81 |
| 20261027 | N0 | PI0 | 192 | -31.6 | 9 | 6 |
| 20261027 | RETAIN_ONLY | PI0 | 192 | -78.0 | 3 | 4 |
| 20261027 | RICH | PI0 | 192 | -3,284.3 | 23 | 64 |

### Replay: quoting, retention and inventory (sums over worlds; ms of the 5,000 ms episodes)

| day | policy | bid quoted | ask quoted | either | both | both at best | first-review pulls | reviewed generations | mean \|inventory\| | peak \|inventory\| | worlds with terminal inventory | markout +100 ms (sum) | missing +1 s endpoints |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | 0.207 | 0.197 | 0.360 | 0.044 | 0.040 | 4,293 | 4,374 | 4.03 | 30 | 72 | -1,164.4 | 32 |
| 20261026 | ENTRY_PA | 0.189 | 0.191 | 0.340 | 0.040 | 0.036 | 4,498 | 4,498 | 3.67 | 30 | 68 | -1,254.4 | 32 |
| 20261026 | FAIR | 0.362 | 0.314 | 0.533 | 0.142 | 0.118 | 1,929 | 2,992 | 5.85 | 30.00 | 101 | -3,466.6 | 72 |
| 20261026 | MECH | 0.396 | 0.346 | 0.592 | 0.151 | 0.120 | 2,123 | 3,295 | 6.33 | 30.00 | 95 | -2,404.9 | 79 |
| 20261026 | N0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261026 | PI0 | 0.037 | 0.018 | 0.053 | 0.002 | 0.002 | 223 | 285 | 0.68 | 30 | 16 | 89.0 | 10 |
| 20261026 | RETAIN_ONLY | 0.043 | 0.027 | 0.066 | 0.004 | 0.003 | 180 | 292 | 0.74 | 30 | 16 | 44.0 | 8 |
| 20261026 | RICH | 0.342 | 0.361 | 0.572 | 0.130 | 0.112 | 1,913 | 3,227 | 4.88 | 30 | 79 | -1,487.7 | 46 |
| 20261027 | ENTRY_ONLY | 0.201 | 0.203 | 0.356 | 0.047 | 0.044 | 4,440 | 4,515 | 2.88 | 30.00 | 67 | -1,554.1 | 40 |
| 20261027 | ENTRY_PA | 0.181 | 0.198 | 0.336 | 0.043 | 0.040 | 4,591 | 4,591 | 2.75 | 30.00 | 62 | -1,592.2 | 36 |
| 20261027 | FAIR | 0.343 | 0.289 | 0.500 | 0.133 | 0.113 | 1,979 | 2,966 | 6.29 | 30 | 109 | -2,710.9 | 79 |
| 20261027 | MECH | 0.393 | 0.351 | 0.586 | 0.158 | 0.133 | 2,208 | 3,418 | 6.16 | 30.00 | 101 | -3,042.2 | 77 |
| 20261027 | N0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261027 | PI0 | 0.052 | 0.016 | 0.065 | 0.003 | 0.002 | 266 | 340 | 0.32 | 30 | 15 | -180.2 | 20 |
| 20261027 | RETAIN_ONLY | 0.055 | 0.025 | 0.073 | 0.007 | 0.006 | 223 | 345 | 0.53 | 30 | 14 | -206.9 | 17 |
| 20261027 | RICH | 0.355 | 0.364 | 0.576 | 0.143 | 0.126 | 1,874 | 3,196 | 4.14 | 30.00 | 85 | -1,278.9 | 55 |

### Churn: each reviewed generation's POST and first live review (recorded values and changes; means over generations)

| day | policy | first-review action | generations | mean ms post → review | mean value at post | mean value at review | mid moved (count) | mean \|Δ mid\| | mean Δ fair_edge | mean Δ queue ahead | mean Δ remaining | mean Δ spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | KEEP | 81 | 100 | 1.662 | 1.300 | 15 | 0.346 | 0.176 | -29.82 | 0 | 0.247 |
| 20261026 | ENTRY_ONLY | PULL | 4,293 | 100 | 1.353 | -13.629 | 378 | 0.091 | 0.006 | 4.76 | 0 | -0.003 |
| 20261026 | ENTRY_PA | PULL | 4,498 | 100 | 1.361 | — | 0 | — | — | — | — | — |
| 20261026 | FAIR | KEEP | 1,063 | 100 | 1.941 | 1.265 | 101 | 0.086 | -0.019 | 2.96 | 0 | -0.020 |
| 20261026 | FAIR | PULL | 1,929 | 100 | 2.469 | -1.547 | 147 | 0.072 | 0.006 | -2.08 | 0 | -0.021 |
| 20261026 | MECH | KEEP | 1,172 | 100 | 1.821 | 1.063 | 110 | 0.093 | 0.036 | -1.21 | 0 | -0.042 |
| 20261026 | MECH | PULL | 2,123 | 100 | 2.310 | -1.687 | 201 | 0.095 | 0.003 | -2.39 | 0 | -0.014 |
| 20261026 | PI0 | KEEP | 62 | 100 | 1.106 | 1.165 | 14 | 0.210 | 0.068 | 10.73 | 0 | -0.129 |
| 20261026 | PI0 | PULL | 223 | 100 | 1.246 | -5.855 | 51 | 0.184 | 0.104 | 30.27 | 0 | -0.224 |
| 20261026 | RETAIN_ONLY | KEEP | 112 | 100 | 0.921 | 1.239 | 27 | 0.232 | 0.047 | 55.47 | 0 | -0.268 |
| 20261026 | RETAIN_ONLY | PULL | 180 | 100 | 1.391 | -2.590 | 36 | 0.150 | 0.116 | 7.29 | 0 | -0.133 |
| 20261026 | RICH | KEEP | 1,314 | 100 | 1.318 | 1.262 | 122 | 0.114 | 0.046 | 7.95 | 0 | 0 |
| 20261026 | RICH | PULL | 1,913 | 100 | 1.419 | -1.885 | 186 | 0.098 | -0.028 | -1.60 | 0 | -0.002 |
| 20261027 | ENTRY_ONLY | KEEP | 75 | 100 | 1.956 | 2.005 | 12 | 0.167 | 0.324 | -12.09 | 0 | 0.147 |
| 20261027 | ENTRY_ONLY | PULL | 4,440 | 100 | 1.295 | -13.432 | 338 | 0.083 | 0.004 | 2.82 | 0 | -0.002 |
| 20261027 | ENTRY_PA | PULL | 4,591 | 100 | 1.312 | — | 0 | — | — | — | — | — |
| 20261027 | FAIR | KEEP | 987 | 100 | 2.100 | 1.271 | 88 | 0.087 | -0.012 | -0.25 | 0 | -0.034 |
| 20261027 | FAIR | PULL | 1,979 | 100 | 2.424 | -1.386 | 146 | 0.070 | 0.023 | -2.18 | 0 | -0.005 |
| 20261027 | MECH | KEEP | 1,210 | 100 | 1.783 | 1.018 | 90 | 0.068 | 0.032 | -3.33 | 0 | -0.014 |
| 20261027 | MECH | PULL | 2,208 | 100 | 2.239 | -1.674 | 178 | 0.084 | 0.005 | -0.62 | 0 | -0.010 |
| 20261027 | PI0 | KEEP | 74 | 100 | 1.847 | 2.443 | 9 | 0.074 | 0.069 | 10.46 | 0 | 0.014 |
| 20261027 | PI0 | PULL | 266 | 100 | 1.745 | -5.866 | 60 | 0.179 | 0.078 | 7.64 | 0 | -0.139 |
| 20261027 | RETAIN_ONLY | KEEP | 122 | 100 | 1.819 | 1.435 | 29 | 0.168 | 0.101 | 5.64 | 0 | -0.189 |
| 20261027 | RETAIN_ONLY | PULL | 223 | 100 | 2.054 | -2.201 | 41 | 0.150 | -0.005 | 9.72 | 0 | -0.076 |
| 20261027 | RICH | KEEP | 1,322 | 100 | 1.229 | 1.179 | 108 | 0.084 | 0.015 | 3.55 | 0 | 0.004 |
| 20261027 | RICH | PULL | 1,874 | 100 | 1.437 | -1.674 | 152 | 0.089 | -0.012 | 0.14 | 0 | 0.008 |

value = the deciding policy's prediction (candidates) or the teacher's V (PI0 decisions); ENTRY_PA has none at review.

### RICH's own-state exposure in the replay: callbacks by state and status

| day | state | status | callbacks |
|---|---|---|---|
| 20261026 | absent | cap_blocked | 3,974 |
| 20261026 | absent | model | 12,613 |
| 20261026 | absent | near_end | 103 |
| 20261026 | live | model | 10,359 |
| 20261026 | live | outside_training_range | 918 |
| 20261027 | absent | cap_blocked | 3,960 |
| 20261027 | absent | model | 12,886 |
| 20261027 | absent | near_end | 109 |
| 20261027 | live | model | 11,116 |
| 20261027 | live | outside_training_range | 777 |

RICH model decisions by support dimension against the training rows of the same head (absent = post, live = keep):

| state | dimension | value | RICH decisions | share | training rows | training share |
|---|---|---|---|---|---|---|
| absent | c_gen | 1 | 4,043 | 0.159 | 11,481 | 0.944 |
| absent | c_gen | 2 | 2,979 | 0.117 | 325 | 0.027 |
| absent | c_gen | >=3 | 18,477 | 0.725 | 362 | 0.030 |
| absent | c_inv | flat | 22,361 | 0.877 | 12,037 | 0.989 |
| absent | c_inv | long | 1,890 | 0.074 | 65 | 0.005 |
| absent | c_inv | short | 1,248 | 0.049 | 66 | 0.005 |
| absent | c_other | absent | 10,416 | 0.408 | 11,846 | 0.974 |
| absent | c_other | cancel_pending | 5,198 | 0.204 | 93 | 0.008 |
| absent | c_other | live | 6,262 | 0.246 | 117 | 0.010 |
| absent | c_other | post_pending | 3,623 | 0.142 | 112 | 0.009 |
| absent | c_price | (20,80] | 16,710 | 0.655 | 7,534 | 0.619 |
| absent | c_price | (5,20] | 3,384 | 0.133 | 1,551 | 0.127 |
| absent | c_price | (80,95] | 2,659 | 0.104 | 1,728 | 0.142 |
| absent | c_price | (95,99] | 958 | 0.038 | 602 | 0.049 |
| absent | c_price | [1,5] | 1,788 | 0.070 | 753 | 0.062 |
| absent | c_queue | (0,30] | 2,341 | 0.092 | 1,594 | 0.131 |
| absent | c_queue | (30,150] | 3,765 | 0.148 | 3,285 | 0.270 |
| absent | c_queue | >150 | 19,393 | 0.761 | 7,289 | 0.599 |
| absent | c_rem | (20,30] | 25,499 | 1 | 12,168 | 1 |
| absent | hbin | (0.0, 100.0] | 204 | 0.008 | 54 | 0.004 |
| absent | hbin | (100.0, 250.0] | 604 | 0.024 | 168 | 0.014 |
| absent | hbin | (1000.0, 2000.0] | 4,842 | 0.190 | 1,241 | 0.102 |
| absent | hbin | (2000.0, 3500.0] | 7,685 | 0.301 | 1,784 | 0.147 |
| absent | hbin | (250.0, 500.0] | 1,047 | 0.041 | 263 | 0.022 |
| absent | hbin | (3500.0, 5000.0] | 8,824 | 0.346 | 8,092 | 0.665 |
| absent | hbin | (500.0, 1000.0] | 2,293 | 0.090 | 566 | 0.047 |
| live | c_gen | 1 | 4,079 | 0.190 | 5,638 | 0.927 |
| live | c_gen | 2 | 2,557 | 0.119 | 204 | 0.034 |
| live | c_gen | >=3 | 14,839 | 0.691 | 239 | 0.039 |
| live | c_inv | flat | 19,284 | 0.898 | 5,576 | 0.917 |
| live | c_inv | long | 1,334 | 0.062 | 239 | 0.039 |
| live | c_inv | short | 857 | 0.040 | 266 | 0.044 |
| live | c_other | absent | 8,040 | 0.374 | 5,457 | 0.897 |
| live | c_other | cancel_pending | 4,111 | 0.191 | 18 | 0.003 |
| live | c_other | live | 6,693 | 0.312 | 598 | 0.098 |
| live | c_other | post_pending | 2,631 | 0.123 | 8 | 0.001 |
| live | c_price | (20,80] | 13,166 | 0.613 | 3,741 | 0.615 |
| live | c_price | (5,20] | 2,728 | 0.127 | 772 | 0.127 |
| live | c_price | (80,95] | 2,843 | 0.132 | 851 | 0.140 |
| live | c_price | (95,99] | 767 | 0.036 | 327 | 0.054 |
| live | c_price | [1,5] | 1,971 | 0.092 | 390 | 0.064 |
| live | c_queue | (0,30] | 861 | 0.040 | 947 | 0.156 |
| live | c_queue | (30,150] | 2,608 | 0.121 | 1,989 | 0.327 |
| live | c_queue | 0 | 177 | 0.008 | 205 | 0.034 |
| live | c_queue | >150 | 17,829 | 0.830 | 2,940 | 0.483 |
| live | c_rem | (0,10] | 6 | 0.000 | 50 | 0.008 |
| live | c_rem | (10,20] | 28 | 0.001 | 57 | 0.009 |
| live | c_rem | (20,30] | 21,441 | 0.998 | 5,974 | 0.982 |
| live | hbin | (0.0, 100.0] | 349 | 0.016 | 162 | 0.027 |
| live | hbin | (100.0, 250.0] | 512 | 0.024 | 235 | 0.039 |
| live | hbin | (1000.0, 2000.0] | 4,232 | 0.197 | 1,447 | 0.238 |
| live | hbin | (2000.0, 3500.0] | 6,699 | 0.312 | 1,822 | 0.300 |
| live | hbin | (250.0, 500.0] | 920 | 0.043 | 403 | 0.066 |
| live | hbin | (3500.0, 5000.0] | 6,952 | 0.324 | 1,213 | 0.199 |
| live | hbin | (500.0, 1000.0] | 1,811 | 0.084 | 799 | 0.131 |

### Reached-state labels (RICH's own states; frozen pi0 arms) against the predictions at those states

| day | head | pairs | nonzero | mean Y | MSE RICH | MSE FAIR | MSE MECH | MSE zero | mean RICH prediction |
|---|---|---|---|---|---|---|---|---|---|
| 20261026 | keep | 376 | 9 | 0.409 | 201.57 | 200.03 | 200.81 | 199.12 | 0.081 |
| 20261026 | post | 376 | 18 | -1.990 | 461.56 | 463.89 | 460.84 | 441.42 | -1.239 |
| 20261027 | keep | 380 | 9 | 0.129 | 106.00 | 110.31 | 108.65 | 102.03 | 0.087 |
| 20261027 | post | 384 | 15 | -1.553 | 302.63 | 336.19 | 322.90 | 303.25 | -0.893 |

| day | head | status | pairs |
|---|---|---|---|
| 20261026 | keep | ok | 376 |
| 20261026 | post | ok | 376 |
| 20261027 | keep | no_eligible_decision | 4 |
| 20261027 | keep | ok | 380 |
| 20261027 | post | ok | 384 |

## Suites and provenance

- **Final mutation checks:** sample 19 / 19; adv2 33 / 33; eval 30 / 30; replay 26 / 26; rsummary 10 / 10. The first four are verified per record by check H; the summary v2 (rsummary) by its own summary log, since r3_checks.py is frozen.
- **Run history (kept as *_run1 / *_run2):** run 1 caught adv2 30 / 33, eval 29 / 30, replay 24 / 26; run 2 adv2 31 / 33. Every miss was in the test instrument, and no module changed:
  - crashes at positive controls (a valid fit, ENTRY_PA's valid states, the declared policies' episodes, including the fixed-sign comparisons) now fail those checks;
  - the replay selftest's unguarded lookup of a POST decision is now a failed check;
  - the coverage dimensions are compared with the declared list, not the module's constant;
  - one equivalent mutant (the fingerprint after inner(v)) is replaced by the other side's identity at the same site.
- **The replay summary v2 (freeze 2b):** the frozen `pf_adv2_replay.summarize` wrote the four CSVs and failed writing replay.json (an int count column under sort_keys; [log](logs/r3_replay_summary.log)). The superseding `pf_adv2_replay_summary.py` proved all four CSVs byte-identical and wrote replay.json (380 ok worlds; statuses ok 380, world:no_path:missing_endpoint 3, world:no_path:stale_future_book 1).
- **Generators ([tools/](tools/)):** fill_r3.py wrote this report from the outputs, and fill_state_r3.py the state hunks. r3_collect.sh made the receipt copies and the sha256 manifest of the outputs left in data/ ([checkpoint/](checkpoint/)). r3_worker.sh and r3_finish.sh did the scheduling.
- **The launcher:** launch_r3.sh was extended in place while the label queues ran it. The queues' bytes were restored into their inode before any queue ended, and they read exactly freeze 1's launcher ([evidence](logs/launcher_inode_repair.txt)).

## Review 315979d2 items 1–3

- **Item 1 (support):** per side / head and selected-action coverage by source, horizon and the six support dimensions for the evaluation and training rows (coverage table above; [coverage.csv](eval/coverage.csv), [coverage_train.csv](eval/coverage_train.csv)); RICH's own-state exposure in the replay against the training rows ([exposure.csv](replay/exposure.csv)). Counts, not calibration.
- **Item 2 (arithmetic residues):** the screens count meaningful nonzero originals (|Y| > 1e-6 c) for the 20-original floor; the exact count and the floor under it (`exact_count_support`) are in [screens.csv](eval/screens.csv). Labels, weights and targets unchanged.
- **Item 3 (suite identity):** check H runs pf_mutant_summary.judge on every record against the final sources and requires one record per file; the falsifier shows it false after one copied record.

## Cost

- Labels: 96 partition jobs, 16.06 core-hours, peak 1,203 MiB. Replay: 12 partitions, 1.96 core-hours, peak 1,171 MiB. The round (every run_light job under logs/, check I): 18.63 core-hours of the 40 ceiling, peak 1,260 MiB.

## Limits

- Two consumed days: point estimates only, no interval, no claim of validation. The fit's in-sample readings (fit/fit.json) are not evidence.
- The replay is the bounded diagnostic of review 315979d2 item 11 (≤ 384 worlds), not the R4 gate: same-history NEU2 and the full economic / quoting qualification stay deferred; PI0 is not NEU2.
- The direct regressors output mean advantages: no probability or tail calibration is claimed.
- The churn and exposure tables are recorded state; no cause is inferred from them.

