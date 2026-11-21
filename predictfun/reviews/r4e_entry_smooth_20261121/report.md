# R4E ADV2-ENTRY-SMOOTH-v1 — one declared entry candidate on the existing consumed Oct 26–27 labels and the bounded own-state comparison

Review 371aab568277dea8ed15c0e386b8c07e7bacd340 (one next priority). Freeze `602fdfc5`, model hashes `eaea1161` (before any evaluation), freeze amendment `5c2ec2ca` (the reuse check's reader, before any evaluation output), freeze 2b `c69ea5c6` (the replay summary v2 and the falsifier v2). Evidence as of 2026-11-21T07:20:13.981250+00:00. Development session Claude aea5bfb9.

## Verdict (every sentence chosen by a computed predicate)

- **Declared checks:** all true ([checks.json](checks.json)); falsifier passed ([checks_falsifier.json](checks_falsifier.json)).
- **Scope:** one declared smoother entry candidate, fitted on Oct 24–25 and tested on the existing Oct 26–27 labels (adaptive consumed-day development: these days informed the design). No interval, no validation, no promotion, no refit, no sweep.
- **Reuse:** the evaluation's 12,088 POST rows equal R3's in keys, labels, weights and RICH predictions (all equal; check C true).
- **Prediction (weighted MSE, per day × side):** SMOOTH is below the zero baseline in 2 of 4 cells, below the training mean in 1, and below the old RICH POST head in 4.
- **Declared entry predicates (SMOOTH, 4 day × side cells; 4,122 admissions in all against RICH's 6,350):** selected POST value > 0 in 0, screen PASS in 0, above the matched draw mean in 1 and above its p95 in 0. All three conjoined (the declared demonstration) in 0 of 4. RICH: 0 / 0 / 2 / 0, demonstrated in 0.
- **Local entry value (pi0-conditional):** SMOOTH's value is above never posting (0) in 0 of 4 cells and above RICH's in 3.
- **Replay 20261026 (188 worlds; diagnostic, not the R4 gate):** SMOOTH − ENTRY_ONLY (primary) +163.6 c (42 worlds above, 41 below, of 188); SMOOTH − PI0 -2,206.6 c (18 worlds above, 40 below, of 188); ENTRY_ONLY − PI0 -2,370.2 c (25 worlds above, 49 below, of 188); generations SMOOTH 3,540 against ENTRY_ONLY 4,611. The re-run controls equal R3's episodes in N0 188 of 188, PI0 188 of 188, ENTRY_ONLY 188 of 188.
- **Replay 20261027 (192 worlds; diagnostic, not the R4 gate):** SMOOTH − ENTRY_ONLY (primary) +883.7 c (44 worlds above, 33 below, of 192); SMOOTH − PI0 -1,525.4 c (21 worlds above, 36 below, of 192); ENTRY_ONLY − PI0 -2,409.1 c (18 worlds above, 50 below, of 192); generations SMOOTH 3,712 against ENTRY_ONLY 4,745. The re-run controls equal R3's episodes in N0 192 of 192, PI0 192 of 192, ENTRY_ONLY 192 of 192.
- **Reached SMOOTH entry states (common states; frozen pi0 arms):** SMOOTH's MSE is below RICH's at the same states on 1 of 2 days and below zero on 0 (pairs: ok 760). Few nonzero labels: descriptive only.

## Tables (r4e_tables.py, copied from the outputs)

### Declared checks

| check | value |
|---|---|
| A_fit | true |
| B_order | true |
| C_reuse | true |
| D_readings | true |
| E_control | true |
| F_replay | true |
| G_reached | true |
| H_suites | true |
| I_resources | true |
| all_true | true |

Falsifier (each check on a clean copy of its real inputs, then after each corruption):

| check | case | value |
|---|---|---|
| C | applied | true |
| C | clean | true |
| C | corrupted | false |
| D | applied | true |
| D | clean | true |
| D | corrupted | false |
| F | applied | true |
| F | clean | true |
| F | control_value | false |
| F | trace_fill | false |
| G | applied | true |
| G | arm_fill | false |
| G | clean | true |
| H | applied | true |
| H | clean | true |
| H | corrupted | false |
| falsifier_ok |  | true |

### Reuse of the existing labels (checked before any reading)

| rows | R3's rows | same keys | labels equal | weights equal | RICH predictions equal |
|---|---|---|---|---|---|
| 12,088 | 12,088 | true | true | true | true |

### Prediction error (all ok POST rows; sampling-weighted; cents squared)

| day | side | rows | meaningful nonzero originals | MSE zero | MSE train mean | MSE SMOOTH | MSE RICH | SMOOTH / zero | RICH / zero | MAE SMOOTH | MAE RICH | MAE zero | Spearman SMOOTH | Spearman RICH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | 3,018 | 68 | 213.33 | 212.80 | 214.18 | 227.78 | 1.0040 | 1.0677 | 1.88 | 3.35 | 1.28 | -0.007 | 0.007 |
| 20261026 | ask | 3,018 | 86 | 425.57 | 424.57 | 424.77 | 439.22 | 0.9981 | 1.0321 | 2.68 | 3.79 | 1.86 | 0.073 | 0.006 |
| 20261027 | bid | 3,026 | 80 | 229.89 | 229.36 | 229.91 | 234.16 | 1.0001 | 1.0186 | 2.26 | 3.40 | 1.66 | 0.014 | 0.028 |
| 20261027 | ask | 3,026 | 70 | 564.06 | 563.02 | 561.62 | 564.76 | 0.9957 | 1.0012 | 2.95 | 3.88 | 2.12 | 0.059 | 0.014 |

### Local entry value per opportunity (cents; sampling-weighted; pi0-conditional, never a policy return; never posting = 0)

| day | side | oracle | always POST | SMOOTH | RICH | SMOOTH regret | SMOOTH missed + (sum w) | SMOOTH admitted − (sum w) | RICH admitted − (sum w) | SMOOTH admitted − with \|Y\| > 30 c | \|Y\| > 30 c events |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | 0.200 | -0.882 | -0.471 | -0.687 | 0.672 | 600.5 | -9,376.8 | -12,408.8 | -8,646.7 | 43 |
| 20261026 | ask | 0.352 | -1.159 | -0.002 | -0.949 | 0.354 | 5,107.1 | -152.0 | -17,271.8 | -87.7 | 46 |
| 20261027 | bid | 0.385 | -0.887 | -0.386 | -0.225 | 0.771 | 2,202.8 | -9,118.9 | -5,961.3 | -8,546.9 | 49 |
| 20261027 | ask | 0.461 | -1.195 | -0.231 | -0.381 | 0.692 | 3,496.3 | -6,669.9 | -9,181.3 | -6,667.2 | 40 |

### Declared predicates (per day / side; lower MSE alone is not success; zero admissions = no demonstrated entry skill)

| day | side | candidate | admissions | status | selected POST value > 0 | screen | above draw mean | above draw p95 | realized | draw mean | draw p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | RICH | 1,571 | admitted | false | FAIL | true | false | -0.4067 | -0.6120 | -0.4012 |
| 20261026 | bid | SMOOTH | 1,247 | admitted | false | FAIL | true | false | -0.4859 | -0.4995 | -0.2302 |
| 20261026 | ask | RICH | 1,526 | admitted | false | FAIL | false | false | -0.0224 | 0.1284 | 0.5059 |
| 20261026 | ask | SMOOTH | 802 | admitted | false | INSUFFICIENT_SUPPORT | false | false | 0.0636 | 0.0885 | 0.1382 |
| 20261027 | bid | RICH | 1,605 | admitted | false | FAIL | false | false | -0.7013 | -0.6573 | -0.1501 |
| 20261027 | bid | SMOOTH | 1,184 | admitted | false | FAIL | false | false | -0.3485 | -0.0913 | 0.3253 |
| 20261027 | ask | RICH | 1,648 | admitted | false | FAIL | true | false | 0.2235 | 0.2171 | 0.3824 |
| 20261027 | ask | SMOOTH | 889 | admitted | false | INSUFFICIENT_SUPPORT | false | false | -0.0456 | -0.0336 | 0.0363 |

### Plan v2 screens (POST / WAIT groups)

| day | side | candidate | group | verdict | originals | meaningful nonzero | exact nonzero | mean pred | mean Y | why |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | RICH | POST | FAIL | 1,105 | 33 | 33 | 2.144 | -1.312 | sign wrong, amount outside |
| 20261026 | bid | RICH | WAIT | FAIL | 1,043 | 36 | 36 | -2.139 | -0.410 | sign ok, amount outside |
| 20261026 | ask | RICH | POST | FAIL | 1,071 | 38 | 38 | 1.511 | -1.852 | sign wrong, amount outside |
| 20261026 | ask | RICH | WAIT | FAIL | 1,054 | 51 | 51 | -2.564 | -0.431 | sign ok, amount outside |
| 20261027 | bid | RICH | POST | FAIL | 1,145 | 34 | 34 | 1.836 | -0.433 | sign wrong, amount outside |
| 20261027 | bid | RICH | WAIT | FAIL | 1,053 | 47 | 47 | -1.950 | -1.374 | sign ok, amount outside |
| 20261027 | ask | RICH | POST | FAIL | 1,131 | 30 | 30 | 1.514 | -0.705 | sign wrong, amount outside |
| 20261027 | ask | RICH | WAIT | FAIL | 996 | 41 | 41 | -2.373 | -1.770 | sign ok, amount outside |
| 20261026 | bid | SMOOTH | POST | FAIL | 955 | 24 | 24 | 0.602 | -0.972 | sign wrong, amount outside |
| 20261026 | bid | SMOOTH | WAIT | PASS | 1,217 | 44 | 44 | -0.625 | -0.798 | sign ok, amount ok |
| 20261026 | ask | SMOOTH | POST | INSUFFICIENT_SUPPORT | 628 | 9 | 9 | 0.432 | -0.009 | 628 originals, 9 meaningful nonzero originals |
| 20261026 | ask | SMOOTH | WAIT | FAIL | 1,335 | 77 | 77 | -1.055 | -1.619 | sign ok, amount outside |
| 20261027 | bid | SMOOTH | POST | FAIL | 909 | 29 | 29 | 0.628 | -0.811 | sign wrong, amount outside |
| 20261027 | bid | SMOOTH | WAIT | FAIL | 1,238 | 52 | 52 | -0.625 | -0.956 | sign ok, amount outside |
| 20261027 | ask | SMOOTH | POST | INSUFFICIENT_SUPPORT | 668 | 9 | 9 | 0.463 | -0.761 | 668 originals, 9 meaningful nonzero originals |
| 20261027 | ask | SMOOTH | WAIT | PASS | 1,292 | 61 | 61 | -1.080 | -1.383 | sign ok, amount ok |

### Calibration, pooled over days and sides by each bin's weight (review 371aab56 item 5: w_sum pooling)

| candidate | prediction bin | rows | weight | weighted mean pred | weighted mean Y | nonzero labels |
|---|---|---|---|---|---|---|
| SMOOTH | (-5, -2] | 855 | 3,408.4 | -2.551 | -1.822 | 59 |
| SMOOTH | (-2, -1] | 2,235 | 8,908.3 | -1.410 | -2.071 | 86 |
| SMOOTH | (-1, -0.5] | 2,192 | 9,850.0 | -0.730 | -0.947 | 59 |
| SMOOTH | (-0.5, 0) | 2,684 | 14,043.0 | -0.243 | -0.772 | 48 |
| SMOOTH | (0, 0.5] | 2,319 | 12,802.1 | 0.241 | -0.951 | 42 |
| SMOOTH | (0.5, 1] | 1,169 | 6,476.5 | 0.717 | -0.959 | 21 |
| SMOOTH | (1, 2] | 614 | 3,470.4 | 1.319 | 0.661 | 11 |
| SMOOTH | (2, 5] | 20 | 139.2 | 2.279 | 0 | 0 |
| RICH | (-inf, -10] | 336 | 711.1 | -14.578 | -4.537 | 40 |
| RICH | (-10, -5] | 542 | 2,361.0 | -6.843 | -1.960 | 35 |
| RICH | (-5, -2] | 1,445 | 7,354.1 | -3.104 | -1.148 | 54 |
| RICH | (-2, -1] | 1,235 | 6,400.9 | -1.432 | -0.543 | 25 |
| RICH | (-1, -0.5] | 942 | 4,748.5 | -0.741 | 0.103 | 17 |
| RICH | (-0.5, 0) | 1,238 | 6,589.3 | -0.232 | -1.280 | 12 |
| RICH | (0, 0.5] | 1,558 | 7,620.9 | 0.253 | -0.745 | 16 |
| RICH | (0.5, 1] | 1,357 | 6,559.7 | 0.733 | -0.247 | 23 |
| RICH | (1, 2] | 1,660 | 7,733.1 | 1.436 | -1.569 | 36 |
| RICH | (2, 5] | 1,448 | 7,155.8 | 3.087 | -1.375 | 46 |
| RICH | (5, 10] | 291 | 1,586.3 | 6.661 | -2.384 | 21 |
| RICH | (10, inf] | 36 | 277.1 | 13.159 | -0.496 | 1 |

R3's RICH POST calibration recomputed with sampling-weight pooling from R3's predictions (supersedes the R3 receipt's pooled table, which averaged weighted cell means by row counts):

| prediction bin | rows | weight | weighted mean pred | weighted mean Y |
|---|---|---|---|---|
| (-inf, -10] | 336 | 711.1 | -14.578 | -4.537 |
| (-10, -5] | 542 | 2,361.0 | -6.843 | -1.960 |
| (-5, -2] | 1,445 | 7,354.1 | -3.104 | -1.148 |
| (-2, -1] | 1,235 | 6,400.9 | -1.432 | -0.543 |
| (-1, -0.5] | 942 | 4,748.5 | -0.741 | 0.103 |
| (-0.5, 0) | 1,238 | 6,589.3 | -0.232 | -1.280 |
| (0, 0.5] | 1,558 | 7,620.9 | 0.253 | -0.745 |
| (0.5, 1] | 1,357 | 6,559.7 | 0.733 | -0.247 |
| (1, 2] | 1,660 | 7,733.1 | 1.436 | -1.569 |
| (2, 5] | 1,448 | 7,155.8 | 3.087 | -1.375 |
| (5, 10] | 291 | 1,586.3 | 6.661 | -2.384 |
| (10, inf] | 36 | 277.1 | 13.159 | -0.496 |

### Strata (both days and sides pooled: counts summed, values weight-pooled; no sparse-cell skill claim)

| dimension | value | rows | meaningful originals | SMOOTH posts | RICH posts | SMOOTH value | RICH value | mean Y |
|---|---|---|---|---|---|---|---|---|
| c_gen | 1 | 11,615 | 283 | 3,942 | 6,122 | -0.174 | -0.483 | -0.888 |
| c_gen | 2 | 255 | 14 | 101 | 124 | -2.013 | -2.889 | -4.426 |
| c_gen | >=3 | 218 | 10 | 79 | 104 | -0.851 | 0.083 | -0.837 |
| c_other | absent | 11,860 | 300 | 4,027 | 6,222 | -0.296 | -0.534 | -1.031 |
| c_other | cancel_pending | 67 | 1 | 28 | 39 | 0 | -5.424 | -5.424 |
| c_other | live | 71 | 2 | 35 | 40 | 1.115 | 1.115 | 1.115 |
| c_other | post_pending | 90 | 3 | 32 | 49 | 0 | -0.109 | 0.605 |
| hbin | (0.0, 100.0] | 59 | 0 | 27 | 29 | 0 | 0 | 0 |
| hbin | (100.0, 250.0] | 160 | 4 | 69 | 84 | 0 | -0.514 | -0.514 |
| hbin | (1000.0, 2000.0] | 1,187 | 32 | 532 | 638 | -0.100 | -0.733 | -1.165 |
| hbin | (2000.0, 3500.0] | 1,778 | 51 | 780 | 932 | -0.264 | -0.143 | -0.769 |
| hbin | (250.0, 500.0] | 285 | 11 | 118 | 136 | -0.319 | -1.057 | -1.314 |
| hbin | (3500.0, 5000.0] | 8,039 | 199 | 2,339 | 4,215 | -0.464 | -0.854 | -1.419 |
| hbin | (500.0, 1000.0] | 580 | 16 | 257 | 316 | -0.020 | -0.237 | -0.225 |
| source | delayed | 3,006 | 88 | 1,175 | 1,542 | -0.198 | -0.595 | -1.221 |
| source | joint_chase | 108 | 3 | 35 | 57 | 0 | -3.368 | -3.227 |
| source | joint_post_keep | 67 | 8 | 18 | 29 | 2.134 | 1.952 | 2.239 |
| source | neutral | 6,044 | 156 | 1,767 | 3,188 | -0.206 | -0.229 | -0.305 |
| source | pi0 | 2,863 | 71 | 1,127 | 1,534 | -0.421 | -0.497 | -0.862 |

### Local action-information control (500 draws, seed 20261121; mandatory POST anchors)

| day | side | candidate | anchors | strata | singleton | no variation | realized | draw mean | lift | draw p05 | draw p95 | share ≥ realized |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | bid | RICH | 1,509 | 916 | 498 | 194 | -0.4067 | -0.6120 | 0.2053 | -0.8589 | -0.4012 | 0.058 |
| 20261026 | ask | RICH | 1,509 | 916 | 498 | 191 | -0.0224 | 0.1284 | -0.1508 | -0.2367 | 0.5059 | 0.610 |
| 20261026 | both | RICH | 3,018 | 1,832 | 996 | 385 | -0.2171 | -0.2494 | 0.0323 | -0.4899 | -0.0129 | 0.438 |
| 20261026 | bid | SMOOTH | 1,509 | 916 | 498 | 201 | -0.4859 | -0.4995 | 0.0136 | -0.7112 | -0.2302 | 0.464 |
| 20261026 | ask | SMOOTH | 1,509 | 916 | 498 | 267 | 0.0636 | 0.0885 | -0.0249 | 0.0359 | 0.1382 | 0.624 |
| 20261026 | both | SMOOTH | 3,018 | 1,832 | 996 | 468 | -0.2148 | -0.2105 | -0.0043 | -0.3254 | -0.0714 | 0.508 |
| 20261027 | bid | RICH | 1,513 | 911 | 482 | 199 | -0.7013 | -0.6573 | -0.0440 | -1.0402 | -0.1501 | 0.466 |
| 20261027 | ask | RICH | 1,513 | 911 | 482 | 199 | 0.2235 | 0.2171 | 0.0064 | 0.0493 | 0.3824 | 0.538 |
| 20261027 | both | RICH | 3,026 | 1,822 | 964 | 398 | -0.2406 | -0.2164 | -0.0243 | -0.4313 | 0.0400 | 0.494 |
| 20261027 | bid | SMOOTH | 1,513 | 911 | 482 | 240 | -0.3485 | -0.0913 | -0.2572 | -0.6186 | 0.3253 | 0.776 |
| 20261027 | ask | SMOOTH | 1,513 | 911 | 482 | 251 | -0.0456 | -0.0336 | -0.0120 | -0.1128 | 0.0363 | 0.646 |
| 20261027 | both | SMOOTH | 3,026 | 1,822 | 964 | 491 | -0.1976 | -0.0628 | -0.1348 | -0.3327 | 0.1489 | 0.784 |

### Replay: episode totals per policy (cents; each world valued once; 5 s episodes; R3's 384-world manifest)

| day | policy | worlds | value | bid value | ask value | fills | filled shares | generations | policy pulls | no entry | no fill | loss (sum) | V < −30 c | equal to R3 (controls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | 188 | -2,635.3 | -2,220.2 | -415.1 | 158 | 1,537 | 4,611 | 4,335 | 0 | 116 | 3,896.9 | 21 | 188 |
| 20261026 | N0 | 188 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 188 | 188 | 0 | 0 | 188 |
| 20261026 | PI0 | 188 | -265.1 | -286.4 | 21.3 | 29 | 336 | 319 | 251 | 88 | 172 | 370.5 | 3 | 188 |
| 20261026 | SMOOTH | 188 | -2,471.7 | -1,678.0 | -793.7 | 128 | 1,296 | 3,540 | 3,359 | 0 | 135 | 2,915.7 | 21 | — |
| 20261027 | ENTRY_ONLY | 192 | -2,377.5 | -1,676.1 | -701.4 | 172 | 1,387 | 4,745 | 4,482 | 0 | 125 | 3,260.2 | 29 | 192 |
| 20261027 | N0 | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 192 | 192 | 0 | 0 | 192 |
| 20261027 | PI0 | 192 | 31.6 | 241.4 | -209.9 | 38 | 326 | 370 | 310 | 95 | 177 | 398.2 | 3 | 192 |
| 20261027 | SMOOTH | 192 | -1,493.8 | -1,102.7 | -391.1 | 135 | 1,266 | 3,712 | 3,533 | 0 | 141 | 2,521.9 | 19 | — |

Paired differences (per world, same tape; SMOOTH − ENTRY_ONLY is the primary comparison):

| day | policy | versus | worlds | sum of differences | worlds above | worlds below |
|---|---|---|---|---|---|---|
| 20261026 | N0 | ENTRY_ONLY | 188 | 2,635.3 | 48 | 24 |
| 20261026 | PI0 | ENTRY_ONLY | 188 | 2,370.2 | 49 | 25 |
| 20261026 | SMOOTH | ENTRY_ONLY | 188 | 163.6 | 42 | 41 |
| 20261026 | ENTRY_ONLY | PI0 | 188 | -2,370.2 | 25 | 49 |
| 20261026 | N0 | PI0 | 188 | 265.1 | 8 | 8 |
| 20261026 | SMOOTH | PI0 | 188 | -2,206.6 | 18 | 40 |
| 20261027 | N0 | ENTRY_ONLY | 192 | 2,377.5 | 49 | 18 |
| 20261027 | PI0 | ENTRY_ONLY | 192 | 2,409.1 | 50 | 18 |
| 20261027 | SMOOTH | ENTRY_ONLY | 192 | 883.7 | 44 | 33 |
| 20261027 | ENTRY_ONLY | PI0 | 192 | -2,409.1 | 18 | 50 |
| 20261027 | N0 | PI0 | 192 | -31.6 | 9 | 6 |
| 20261027 | SMOOTH | PI0 | 192 | -1,525.4 | 21 | 36 |

### Replay: quoting, retention, inventory and markouts (sums over worlds; ms of the 5,000 ms episodes)

| day | policy | bid quoted | ask quoted | both | first-review pulls | reviewed generations | peak \|inventory\| | terminal inventory worlds | markout fill | markout +100 ms | markout +1 s | missing +100 ms | missing +1 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | 0.207 | 0.197 | 0.044 | 4,293 | 4,374 | 30 | 72 | 487.8 | -1,164.4 | -2,104.4 | 3 | 32 |
| 20261026 | N0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261026 | PI0 | 0.037 | 0.018 | 0.002 | 223 | 285 | 30 | 16 | 241.0 | 89.0 | -71.0 | 1 | 10 |
| 20261026 | SMOOTH | 0.197 | 0.119 | 0.028 | 3,325 | 3,386 | 30.00 | 51 | 502.7 | -1,022.8 | -1,870.0 | 0 | 34 |
| 20261027 | ENTRY_ONLY | 0.201 | 0.203 | 0.047 | 4,440 | 4,515 | 30.00 | 67 | -17.0 | -1,554.1 | -1,382.4 | 6 | 40 |
| 20261027 | N0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20261027 | PI0 | 0.052 | 0.016 | 0.003 | 266 | 340 | 30 | 15 | 113.5 | -180.2 | -71.0 | 3 | 20 |
| 20261027 | SMOOTH | 0.201 | 0.127 | 0.030 | 3,491 | 3,555 | 30 | 51 | 170.6 | -992.3 | -1,189.8 | 3 | 25 |

### First live reviews (each reviewed generation; the POST advantage at the post and the KEEP advantage at the review are DIFFERENT targets, not one forecast reversing; review 371aab56 item 2)

| day | policy | first-review action | generations | mean ms post → review | mean POST advantage at the post | mean retention value at the review | mid moved (count) | mean \|Δ mid\| | mean Δ queue ahead |
|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | KEEP | 81 | 100 | 1.662 | 1.300 | 15 | 0.346 | -29.82 |
| 20261026 | ENTRY_ONLY | PULL | 4,293 | 100 | 1.353 | -13.629 | 378 | 0.091 | 4.76 |
| 20261026 | PI0 | KEEP | 62 | 100 | 1.106 | 1.165 | 14 | 0.210 | 10.73 |
| 20261026 | PI0 | PULL | 223 | 100 | 1.246 | -5.855 | 51 | 0.184 | 30.27 |
| 20261026 | SMOOTH | KEEP | 61 | 100 | 0.437 | 0.636 | 13 | 0.238 | 8.31 |
| 20261026 | SMOOTH | PULL | 3,325 | 100.0 | 0.442 | -12.657 | 311 | 0.097 | -3.39 |
| 20261027 | ENTRY_ONLY | KEEP | 75 | 100 | 1.956 | 2.005 | 12 | 0.167 | -12.09 |
| 20261027 | ENTRY_ONLY | PULL | 4,440 | 100 | 1.295 | -13.432 | 338 | 0.083 | 2.82 |
| 20261027 | PI0 | KEEP | 74 | 100 | 1.847 | 2.443 | 9 | 0.074 | 10.46 |
| 20261027 | PI0 | PULL | 266 | 100 | 1.745 | -5.866 | 60 | 0.179 | 7.64 |
| 20261027 | SMOOTH | KEEP | 64 | 100 | 0.351 | 1.791 | 6 | 0.070 | -6.69 |
| 20261027 | SMOOTH | PULL | 3,491 | 100 | 0.450 | -12.623 | 284 | 0.082 | 5.28 |

### SMOOTH's own entry decisions in the replay by generation and the other side's state (callback exposure, not independent samples)

| dimension | value | SMOOTH absent model decisions | share |
|---|---|---|---|
| c_gen | 1 | 12,286 | 0.295 |
| c_gen | 2 | 5,699 | 0.137 |
| c_gen | >=3 | 23,615 | 0.568 |
| c_other | absent | 23,378 | 0.562 |
| c_other | cancel_pending | 9,669 | 0.232 |
| c_other | live | 2,365 | 0.057 |
| c_other | post_pending | 6,188 | 0.149 |

### Reached SMOOTH entry states (frozen pi0 arms): both POST heads on identical inputs at the same states

| day | pairs | nonzero | mean Y | MSE SMOOTH | MSE RICH | MSE zero | mean SMOOTH prediction | mean RICH prediction | SMOOTH > 0 | RICH > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261026 | 376 | 15 | -1.932 | 381.21 | 362.80 | 379.07 | -0.602 | -0.485 | 73 | 188 |
| 20261027 | 384 | 14 | -0.533 | 382.63 | 386.96 | 379.38 | -0.537 | -0.575 | 100 | 176 |

| day | status | pairs |
|---|---|---|
| 20261026 | ok | 376 |
| 20261027 | ok | 384 |

## Review 371aab56 checklist

- **Item 1 (entry selection):** this round.
- **Item 2:** the first-review table names the POST advantage at the post and the retention value at the review as different targets. Hybrid losses are not summed into a decomposition. FINDINGS P92's wording is corrected in this commit.
- **Item 3:** strata by generation, source, horizon and other side; SMOOTH's own entry exposure; both heads on identical reached states.
- **Item 4:** every episode and reached pair keeps its full trace. Checks F and G recompute ledgers and required callbacks with the frozen label validator, and the controls' reproduction of R3 directly from R3's parts.
- **Item 5:** calibration bins carry w_sum and are weight-pooled; R3's pooled RICH table is recomputed by weights (tables).
- **Operational:** part records are accepted only with this round's identity; the summary checks complete membership; this round has its own launcher.

## Suites and provenance

- **Mutation checks (final):** entry 16 / 16; entry_eval 15 / 15; entry_replay 19 / 19 (per record: check H).
- **Run history:** replay run 1 caught 18 / 19: one replacement did not compile (harness fault, fixed; every mutant is now compile-checked). The evaluator's first suite (14 / 14) is superseded by the amendment's (15 / 15).
- **Freeze amendment:** the first evaluation refused at the declared reuse check because pandas' default CSV float parser is not exact ([log](logs/r4e_eval_refused_default_parser.log)). The reader now parses round-trip; no reading had been computed.
- **Freeze 2b (two frozen writers failed after the run):** the frozen summarize wrote the four CSVs and failed on a top-level binding_bad that the episodes keep in their traces ([log](logs/r4e_replay_summary.log)); `pf_adv2_entry_replay_summary.py` proved the CSVs byte-identical and wrote replay.json (mutation check 8 / 8). The frozen falsifier crashed (its fill corruption looked in one world without fills; [log](logs/r4e_checks.log)); `r4e_checks_falsify_v2.py` searches every partition and records whether each corruption was applied. The checks A–I themselves (checks.json) are r4e_checks.py's, unchanged.

## Cost

- Replay: 12 partitions, 1.11 core-hours, peak 1,186 MiB. The round (every run_light job, check I): 1.38 core-hours of the 8 ceiling, peak 1,186 MiB.

## Limits

- Two consumed days that informed the design: point estimates only, no interval, no validation. In-sample fit readings are not evidence.
- The replay is a bounded diagnostic, not the R4 gate; PI0 is not NEU2. Reached-state labels are few: descriptive only.
- One smoothing package was tested; a negative result does not show that no entry model can work.

