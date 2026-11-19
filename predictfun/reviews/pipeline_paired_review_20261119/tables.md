### Validity (each predicate computed on every row unless named; a FAIL is an open defect)

| Predicate | Oct 26 | Oct 27 |
|---|---|---|
| V1 labels = a per-row loop over the engine fills (sampled orders, oid % 25 = 0) | PASS: 0 of 51,642 rows | PASS: 0 of 56,366 rows |
| V2 value identity on every label row (NaN fails) | PASS: 0 of 1,299,570 | PASS: 0 of 1,433,878 |
| V3 timing: export rows off / pi0 steps off / kept fills outside [activation, END] | PASS: 0 of 1,300,498 / 0 / 0 | PASS: 0 of 1,434,692 / 0 / 0 |
| V4 populations and statuses | PASS | PASS |
| V5 accounting: ledger = evaluation at B (1e-6 c), labels = PI0-ALL, null mean, first-review shares, input hashes | PASS: 0 of 53 differ | PASS: 0 of 53 differ |
| Open defects | none | none |

**Populations (V4)**

| Count | Oct 26 | Oct 27 |
|---|---:|---:|
| Original opportunities (keep / pull run) | 30,242 | 29,820 |
| Export POST rows (one per original) | 30,242 | 29,820 |
| With a path (eligible + post-only refused) | 29,832 | 29,338 |
| No path (statuses) | 410 | 482 |
| Labelled POST rows | 29,832 | 29,338 |
| Labelled RETAIN rows | 1,269,738 | 1,404,540 |
| Eligible RETAIN export rows | 1,269,738 | 1,404,540 |
| RETAIN steps without a forecast (kept: fallback) | 518 | 332 |
| ... on A's reached path | 27 | 0 |
| ... at A's first review | 0 | 0 |
| A's reached rows | 67,776 | 76,731 |
| A reached orders not admitted | 0 | 0 |
| Refused POST rows with a nonzero label | 0 | 0 |
| Two sources agree: original / with a path | True / True | True / True |
| Admitted (A / AF / AM / PI0 / NEU / E1) | 14,126 / 10,381 / 9,834 / 3,364 / 3,520 / 2,502 | 13,319 / 10,051 / 9,539 / 2,978 / 2,692 / 1,975 |
| Nonfinite predictions (A / AF / AM / pi0 / NEU) | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| Keep / pull statuses | eligible 28,442, missing_endpoint 374, refused_post_only 1,390, stale_future_book 36 | eligible 28,038, missing_endpoint 460, refused_post_only 1,300, stale_future_book 22 |
| Export RETAIN statuses | eligible 1,269,738, feature_unavailable 518 | eligible 1,404,540, feature_unavailable 332 |

### Prediction (A; c per row; eb = episode-balanced; the old gate is the verdict, the plan v2 screen a diagnostic)

| Day, side, group | Rows | Episodes (nonzero) | Mean pred / label | eb pred / label | Sum pred / label | Old gate | Plan v2 screen |
|---|---:|---:|---:|---:|---:|---|---|
| Oct 26 bid POST (pred > 0) | 6,319 | 6,319 (175) | +2.906 / -0.063 | +2.906 / -0.063 | +18,360 / -399 | FAIL | FAIL |
| Oct 26 bid NO_ENTRY (pred < 0) | 8,597 | 8,597 (290) | -3.264 / -1.121 | -3.264 / -1.121 | -28,062 / -9,634 | FAIL | FAIL |
| Oct 26 bid KEEP reached (pred > 0) | 20,097 | 2,293 (158) | +1.369 / -0.240 | +0.974 / +0.205 | +27,521 / -4,830 | FAIL | FAIL |
| Oct 26 bid PULL reached (pred < 0) | 5,734 | 5,734 (184) | -1.173 / -0.663 | -1.173 / -0.663 | -6,727 / -3,801 | FAIL | FAIL |
| Oct 26 bid KEEP neutral (pred > 0) | 255,503 | 11,302 (1,890) | +0.612 / -0.032 | +0.891 / -0.201 | +156,409 / -8,115 | FAIL | FAIL |
| Oct 26 bid PULL neutral (pred < 0) | 369,083 | 13,613 (4,653) | -1.105 / -0.655 | -2.162 / -1.712 | -407,835 / -241,600 | FAIL | FAIL |
| Oct 26 ask POST (pred > 0) | 7,807 | 7,807 (178) | +2.043 / +0.012 | +2.043 / +0.012 | +15,952 / +96 | FAIL | FAIL |
| Oct 26 ask NO_ENTRY (pred < 0) | 7,109 | 7,109 (335) | -2.779 / -2.105 | -2.779 / -2.105 | -19,753 / -14,967 | FAIL | FAIL |
| Oct 26 ask KEEP reached (pred > 0) | 34,769 | 3,493 (171) | +0.571 / +0.171 | +0.629 / -0.041 | +19,843 / +5,942 | FAIL | FAIL |
| Oct 26 ask PULL reached (pred < 0) | 7,176 | 7,176 (172) | -0.949 / -0.286 | -0.949 / -0.286 | -6,808 / -2,053 | FAIL | FAIL |
| Oct 26 ask KEEP neutral (pred > 0) | 353,463 | 12,162 (2,193) | +0.385 / +0.007 | +0.677 / -0.401 | +136,064 / +2,512 | FAIL | FAIL |
| Oct 26 ask PULL neutral (pred < 0) | 291,689 | 13,634 (4,336) | -1.075 / -0.681 | -1.976 / -1.538 | -313,571 / -198,749 | FAIL | FAIL |
| Oct 27 bid POST (pred > 0) | 5,866 | 5,866 (196) | +2.618 / -1.139 | +2.618 / -1.139 | +15,358 / -6,680 | FAIL | FAIL |
| Oct 27 bid NO_ENTRY (pred < 0) | 8,803 | 8,803 (296) | -2.932 / -1.065 | -2.932 / -1.065 | -25,807 / -9,373 | FAIL | FAIL |
| Oct 27 bid KEEP reached (pred > 0) | 20,815 | 2,044 (161) | +1.108 / -0.002 | +0.964 / +0.027 | +23,061 / -46 | FAIL | FAIL |
| Oct 27 bid PULL reached (pred < 0) | 5,323 | 5,323 (161) | -1.049 / -0.213 | -1.049 / -0.213 | -5,582 / -1,132 | FAIL | FAIL |
| Oct 27 bid KEEP neutral (pred > 0) | 280,619 | 11,133 (1,864) | +0.499 / +0.151 | +0.680 / -0.239 | +140,079 / +42,312 | FAIL | FAIL |
| Oct 27 bid PULL neutral (pred < 0) | 422,901 | 13,424 (4,538) | -0.901 / -0.409 | -1.841 / -1.382 | -381,016 / -172,808 | FAIL | FAIL |
| Oct 27 ask POST (pred > 0) | 7,453 | 7,453 (146) | +1.692 / -0.240 | +1.692 / -0.240 | +12,610 / -1,787 | FAIL | FAIL |
| Oct 27 ask NO_ENTRY (pred < 0) | 7,216 | 7,216 (257) | -2.544 / -1.133 | -2.544 / -1.133 | -18,356 / -8,176 | FAIL | FAIL |
| Oct 27 ask KEEP reached (pred > 0) | 43,752 | 3,398 (175) | +0.401 / -0.108 | +0.537 / -0.099 | +17,550 / -4,713 | FAIL | FAIL |
| Oct 27 ask PULL reached (pred < 0) | 6,841 | 6,841 (182) | -0.766 / -0.859 | -0.766 / -0.859 | -5,237 / -5,874 | PASS | PASS |
| Oct 27 ask KEEP neutral (pred > 0) | 391,039 | 11,924 (2,074) | +0.295 / +0.001 | +0.547 / -0.347 | +115,386 / +580 | FAIL | FAIL |
| Oct 27 ask PULL neutral (pred < 0) | 309,981 | 13,379 (4,237) | -0.885 / -0.655 | -1.588 / -1.779 | -274,436 / -203,056 | FAIL | PASS |

**AF and AM (old gate / plan v2 screen per group)**

| Head, day, side | POST (pred > 0) | NO_ENTRY (pred < 0) | KEEP reached (pred > 0) | PULL reached (pred < 0) | KEEP neutral (pred > 0) | PULL neutral (pred < 0) |
|---|---|---|---|---|---|---|
| AF Oct 26 bid | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL |
| AF Oct 26 ask | FAIL / FAIL | PASS / PASS | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | PASS / FAIL |
| AF Oct 27 bid | FAIL / FAIL | PASS / PASS | FAIL / FAIL | PASS / PASS | FAIL / FAIL | FAIL / FAIL |
| AF Oct 27 ask | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL |
| AM Oct 26 bid | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / PASS |
| AM Oct 26 ask | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | PASS / PASS | FAIL / FAIL | PASS / FAIL |
| AM Oct 27 bid | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / PASS |
| AM Oct 27 ask | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | PASS / FAIL |

### Controller (the committed evaluation v2 at B; c at END marks, not P&L)

| Policy | Oct 26 both | Oct 26 bid | Oct 26 ask | Oct 27 both | Oct 27 bid | Oct 27 ask |
|---|---:|---:|---:|---:|---:|---:|
| A | -2,032 | -980 | -1,052 | -11,865 | -9,064 | -2,801 |
| A-PA | -1,053 | -1,160 | +107 | -6,944 | -5,630 | -1,314 |
| PI0 | +478 | +126 | +352 | -2,608 | -2,028 | -579 |
| NEU | +3,540 | +912 | +2,628 | -2,827 | -3,344 | +517 |
| E1 | +277 | -82 | +359 | -824 | -362 | -462 |
| PA | -23,731 | -9,324 | -14,406 | -23,346 | -13,902 | -9,444 |

| Reading | Oct 26 | Oct 27 |
|---|---|---|
| A above N0 / PI0 / NEU / E1 / A-PA at B (both sides) | False / False / False / False / False | False / False / False / False / False |
| Scenarios (of 19) with A above 0 / above PI0 | 4 / 4 of 19 | 3 / 3 of 19 |
| the practical predicates vs A-PA | IMPLIED-BY-NESTING (not evidence) | IMPLIED-BY-NESTING (not evidence) |
| MATCHED CONTROL WITH ITS SPREAD (draws, one-sided p <= 0.05) | {'both': 'PASS', 'bid': 'FAIL', 'ask': 'PASS'} | {'both': 'FAIL', 'bid': 'FAIL', 'ask': 'FAIL'} |
| (evaluation) ECONOMIC GATE (this day, both sides) | FAIL | FAIL |
| (evaluation) MATCHED CONTROL AT B: A above the matched null of its admissions | PASS | PASS |
| (evaluation) PRACTICAL PREDICATES AT B (A vs A-PA): at-best both sides / median lifetime / first-review reversals / cancels per filled share | [True, False, True, True] | [True, False, True, True] |
| (evaluation) PRACTICAL GATE (this day) | FAIL | FAIL |

**The matched null of A with its spread (exact moments; 1,000 seeded draws)**

| Day, side | A (c) | Null mean | Null sd | z | p (normal, approx.) | p (draws) | Draws agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oct 26 both | -2,032 | -11,026 | 2,692 | +3.34 | 0.0004 | 0.0010 | True |
| Oct 26 bid | -980 | -3,573 | 1,510 | +1.72 | 0.0429 | 0.0569 | True |
| Oct 26 ask | -1,052 | -7,452 | 2,228 | +2.87 | 0.0020 | 0.0010 | True |
| Oct 27 both | -11,865 | -13,431 | 3,056 | +0.51 | 0.3041 | 0.3047 | True |
| Oct 27 bid | -9,064 | -7,780 | 2,234 | -0.58 | 0.7174 | 0.6853 | True |
| Oct 27 ask | -2,801 | -5,652 | 2,084 | +1.37 | 0.0857 | 0.0849 | True |

**Quoting (both sides; A vs PI0 / NEU / E1 is not nested)**

| Day, policy | Resting | At best | At best both sides | Median / p90 life (ms) | Pulled at first review | Cancels / 100 filled sh | Attempts | Filled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oct 26 A | 2.5% | 1.7% | 0.3% | 79 / 679 | 57.0% | 126.81 | 14,126 | 490 |
| Oct 26 PI0 | 0.7% | 0.5% | 0.0% | 79 / 579 | 74.3% | 63.17 | 3,364 | 219 |
| Oct 26 NEU | 0.5% | 0.4% | 0.0% | 79 / 379 | 78.3% | 120.16 | 3,520 | 131 |
| Oct 26 E1 | 0.5% | 0.4% | 0.0% | 79 / 479 | 79.2% | 107.63 | 2,502 | 99 |
| Oct 26 A-PA | 0.7% | 0.4% | 0.2% | 79 / 79 | 100.0% | 320.92 | 14,126 | 222 |
| Oct 27 A | 2.9% | 2.0% | 0.3% | 79 / 779 | 57.4% | 121.30 | 13,319 | 477 |
| Oct 27 PI0 | 0.7% | 0.6% | 0.0% | 79 / 779 | 72.9% | 64.26 | 2,978 | 182 |
| Oct 27 NEU | 0.4% | 0.3% | 0.0% | 79 / 359 | 80.9% | 117.68 | 2,692 | 99 |
| Oct 27 E1 | 0.5% | 0.4% | 0.0% | 79 / 589 | 77.7% | 82.09 | 1,975 | 89 |
| Oct 27 A-PA | 0.7% | 0.4% | 0.2% | 79 / 79 | 100.0% | 337.61 | 13,319 | 208 |

| A vs | Oct 26: more resting / more both-sides at best / longer median life / fewer first-review pulls / fewer cancels per filled share | Oct 27: more resting / more both-sides at best / longer median life / fewer first-review pulls / fewer cancels per filled share |
|---|---|---|
| PI0 | True / True / False / True / False | True / True / False / True / False |
| NEU | True / True / False / True / False | True / True / False / True / False |
| E1 | True / True / False / True / False | True / True / False / True / False |
| Pulled at the first review: PI0, every eligible opportunity | 91.4% | 91.7% |
| Pulled at the first review: PI0, its admissions | 74.3% | 72.9% |
| Pulled at the first review: A, every eligible opportunity | 61.1% | 62.0% |
| Pulled at the first review: A, its admissions | 57.0% | 57.4% |

### Risk (the ledger at B; episodes = opportunities with a path; loss = max(-V, 0); severe = V < -30 c)

| Day, policy | Mean loss (c) | P(loss) | P(severe) | Severe episodes | p95 / p99 loss, filled | p95 / p99 loss, loss episodes | Worst 1% share of loss | Largest loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oct 26 A | 0.846 | 0.9% | 0.6% | 167 | 255.0 / 465.0 | 315.0 / 519.0 | 100.0% | 765.0 |
| Oct 26 A-PA | 0.404 | 0.4% | 0.3% | 78 | 283.5 / 428.7 | 345.6 / 458.4 | 100.0% | 495.0 |
| Oct 26 PI0 | 0.279 | 0.4% | 0.2% | 67 | 225.0 / 279.6 | 242.0 / 381.0 | 100.0% | 495.0 |
| Oct 26 NEU | 0.151 | 0.2% | 0.1% | 33 | 180.9 / 312.8 | 242.7 / 426.7 | 100.0% | 555.0 |
| Oct 26 E1 | 0.138 | 0.2% | 0.1% | 32 | 225.0 / 406.2 | 248.5 / 435.0 | 100.0% | 465.0 |
| Oct 26 PA | 1.532 | 1.4% | 0.9% | 272 | 315.0 / 495.0 | 374.1 / 555.0 | 97.8% | 1,065.0 |
| Oct 26 PI0-ALL | 1.993 | 2.0% | 1.3% | 380 | 285.0 / 495.0 | 337.6 / 555.0 | 90.1% | 1,065.0 |
| Oct 27 A | 1.097 | 0.9% | 0.6% | 182 | 285.3 / 736.0 | 388.5 / 1,039.8 | 100.0% | 1,395.0 |
| Oct 27 A-PA | 0.452 | 0.4% | 0.3% | 79 | 255.0 / 634.5 | 373.9 / 1,015.1 | 100.0% | 1,395.0 |
| Oct 27 PI0 | 0.253 | 0.4% | 0.2% | 61 | 223.5 / 362.8 | 285.0 / 373.9 | 100.0% | 465.0 |
| Oct 27 NEU | 0.196 | 0.2% | 0.1% | 33 | 228.0 / 456.6 | 339.0 / 897.1 | 100.0% | 1,395.0 |
| Oct 27 E1 | 0.126 | 0.2% | 0.1% | 32 | 144.0 / 348.6 | 204.7 / 360.3 | 100.0% | 375.0 |
| Oct 27 PA | 1.437 | 1.3% | 0.8% | 237 | 315.0 / 587.1 | 424.9 / 645.5 | 98.7% | 1,395.0 |
| Oct 27 PI0-ALL | 1.849 | 1.9% | 1.1% | 328 | 300.0 / 495.0 | 363.3 / 611.0 | 92.4% | 1,395.0 |

**Where A's loss and severe episodes arise (both sides)**

| Day, stratum | Episodes | Total loss (c) | Severe episodes | Severe excess (c) | Value (c) |
|---|---:|---:|---:|---:|---:|
| Oct 26 side ask | 14,916 | 15,406 | 97 | 11,981 | -1,052 |
| Oct 26 side bid | 14,916 | 9,837 | 70 | 7,258 | -980 |
| Oct 26 phase 1 | 6,326 | 4,902 | 37 | 3,666 | -729 |
| Oct 26 phase 2 | 6,910 | 3,170 | 31 | 2,047 | +1,778 |
| Oct 26 phase 3 | 6,830 | 3,792 | 24 | 2,908 | -1,289 |
| Oct 26 phase 4 | 6,080 | 4,821 | 32 | 3,667 | -1,516 |
| Oct 26 phase 5 | 3,686 | 8,559 | 43 | 6,951 | -276 |
| Oct 26 spread (0, 1] | 28,464 | 22,247 | 152 | 16,765 | -3,866 |
| Oct 26 spread (1, 2] | 934 | 1,922 | 8 | 1,611 | +1,647 |
| Oct 26 spread (2, 5] | 352 | 1,074 | 7 | 863 | +187 |
| Oct 26 spread (5, inf] | 82 | 0 | 0 | 0 | +0 |
| Oct 26 sigma tercile high | 9,934 | 10,495 | 67 | 8,011 | -1,491 |
| Oct 26 sigma tercile low | 10,002 | 6,170 | 42 | 4,649 | +898 |
| Oct 26 sigma tercile mid | 9,896 | 8,578 | 58 | 6,579 | -1,439 |
| Oct 27 side ask | 14,669 | 15,071 | 89 | 11,899 | -2,801 |
| Oct 27 side bid | 14,669 | 17,109 | 93 | 13,765 | -9,064 |
| Oct 27 phase 1 | 6,328 | 6,347 | 49 | 4,618 | -3,553 |
| Oct 27 phase 2 | 6,898 | 4,482 | 31 | 3,302 | -2,291 |
| Oct 27 phase 3 | 6,798 | 5,542 | 31 | 4,466 | -2,915 |
| Oct 27 phase 4 | 5,910 | 6,731 | 33 | 5,543 | -2,169 |
| Oct 27 phase 5 | 3,404 | 9,078 | 38 | 7,735 | -938 |
| Oct 27 spread (0, 1] | 28,180 | 27,748 | 166 | 21,776 | -9,663 |
| Oct 27 spread (1, 2] | 722 | 912 | 7 | 648 | -451 |
| Oct 27 spread (2, 5] | 370 | 3,189 | 8 | 2,939 | -1,421 |
| Oct 27 spread (5, inf] | 66 | 330 | 1 | 300 | -330 |
| Oct 27 sigma tercile high | 9,740 | 7,941 | 52 | 6,093 | +766 |
| Oct 27 sigma tercile low | 9,792 | 10,956 | 60 | 8,821 | -4,383 |
| Oct 27 sigma tercile mid | 9,806 | 13,283 | 70 | 10,750 | -8,249 |

| Day, policy | Window: before the first review | earliest-effect delay | after the earliest effect | Severe: before | Severe: delay | Severe: after | Spread / movement component | Time-integrated inventory (share-s) | Movement per filled share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Oct 26 A | -2,667 | +1,614 | -979 | -6,270 | -5,333 | -12,646 | +5,679 / -7,712 | 45,835 | -0.765 |
| Oct 26 A-PA | -2,667 | +1,614 | +0 | -6,270 | -5,287 | +0 | +2,365 / -3,418 | 20,424 | -0.820 |
| Oct 26 PI0 | -854 | +780 | +552 | -2,241 | -1,390 | -4,129 | +2,647 / -2,169 | 20,243 | -0.466 |
| Oct 26 NEU | +724 | +538 | +2,279 | -967 | -1,165 | -2,053 | +1,650 / +1,890 | 11,989 | +0.711 |
| Oct 26 E1 | -514 | -232 | +1,022 | -1,448 | -1,551 | -870 | +1,344 / -1,068 | 9,512 | -0.519 |
| Oct 27 A | -5,006 | -1,938 | -4,922 | -6,644 | -6,095 | -18,385 | +5,817 / -17,682 | 44,933 | -1.779 |
| Oct 27 A-PA | -5,006 | -1,938 | +0 | -6,633 | -6,085 | +0 | +2,012 / -8,956 | 18,414 | -2.384 |
| Oct 27 PI0 | -1,351 | -1,862 | +605 | -1,913 | -1,920 | -2,978 | +2,344 / -4,952 | 16,447 | -1.233 |
| Oct 27 NEU | -2,898 | -647 | +718 | -2,987 | -1,650 | -882 | +1,241 / -4,067 | 8,932 | -1.957 |
| Oct 27 E1 | -1,248 | +319 | +106 | -1,735 | -360 | -1,354 | +1,217 / -2,041 | 9,355 | -0.962 |

**FINDINGS C1 / C3 / C12 (both sides; c)**

| Day, policy | Age < 100 ms | 100-500 ms | >= 500 ms | Without young fills | B1a-weighted (M0 / M50) | B1a sign | Cancel-pending fills / value | Pending share of gross | WS / E2 / imputed / mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Oct 26 A | -1,850 | -634 | +451 | -183 | -289 / -194 | HOLDS-SIGN | 135 / +1,480 | 27.6% | -1,591 / +651 / +24 / -1,116 |
| Oct 26 A-PA | -1,053 | +0 | +0 | +0 | -87 / -130 | HOLDS-SIGN | 132 / +1,914 | 54.4% | -2,690 / +2,447 / +0 / -810 |
| Oct 26 PI0 | -178 | +280 | +376 | +657 | +315 / +238 | HOLDS-SIGN | 49 / +14 | 18.1% | +1,105 / -39 / +0 / -587 |
| Oct 26 PI0-PA | -74 | +0 | +0 | +0 | -6 / -9 | HOLDS-SIGN | 56 / +774 | 49.9% | +473 / +134 / +0 / -681 |
| Oct 26 NEU | +1,754 | +849 | +937 | +1,787 | +1,049 / +906 | HOLDS-SIGN | 39 / +663 | 24.0% | +2,549 / +68 / +195 / +727 |
| Oct 26 E1 | -546 | +679 | +143 | +823 | +394 / +195 | HOLDS-SIGN | 34 / +483 | 37.8% | -851 / +739 / +195 / +194 |
| Oct 26 PA | -23,731 | +0 | +0 | +0 | -1,967 / -2,928 | HOLDS-SIGN | 374 / -7,339 | 48.1% | -13,921 / -1,866 / +0 / -7,944 |
| Oct 27 A | -6,586 | -4,056 | -1,223 | -5,279 | -3,339 / -2,555 | HOLDS-SIGN | 121 / -2,307 | 24.5% | -4,485 / -8,305 / +209 / +715 |
| Oct 27 A-PA | -6,944 | +0 | +0 | +0 | -576 / -857 | HOLDS-SIGN | 110 / -1,915 | 54.3% | -3,729 / -2,501 / +0 / -713 |
| Oct 27 PI0 | -3,268 | +49 | +611 | +660 | +42 / -98 | SIGN-DIFFERS | 35 / -2,365 | 22.9% | -1,809 / +46 / +0 / -845 |
| Oct 27 PI0-PA | -3,213 | +0 | +0 | +0 | -266 / -396 | HOLDS-SIGN | 37 / -1,862 | 47.3% | -1,296 / -1,420 / +0 / -496 |
| Oct 27 NEU | -3,208 | -139 | +520 | +381 | -99 / -187 | HOLDS-SIGN | 26 / -807 | 24.8% | -1,757 / -435 / +0 / -634 |
| Oct 27 E1 | -1,005 | +167 | +14 | +181 | +15 / -70 | SIGN-DIFFERS | 20 / +78 | 15.5% | -296 / -139 / +0 / -389 |
| Oct 27 PA | -23,346 | +0 | +0 | +0 | -1,935 / -2,880 | HOLDS-SIGN | 315 / -9,028 | 53.1% | -12,875 / -6,373 / +0 / -4,099 |

- Oct 26: A minus A-PA by fill age: lt100 -796 c, 100to500 -634 c, ge500 +451 c
- Oct 27: A minus A-PA by fill age: lt100 +358 c, 100to500 -4,056 c, ge500 -1,223 c

### Computed statements (descriptive; each sentence of the verdicts below is one of these)

| Statement | Value |
|---|---|
| A's old-gate groups failing (of those with rows) | 23 of 24 |
| A POST: mean prediction above 0 and realized label at most +0.02 c per row, every day and side | True |
| Validity PASS (V1-V5) with no open defect, both days | True |
| A below N0, PI0, NEU, E1 and A-PA at B (both sides), both days | True |
| Matched control with its spread (both sides): Oct 26 / Oct 27 | PASS / FAIL |
| Evaluation matched control vs the null mean only: Oct 26 / Oct 27 | PASS / PASS |
| A rests longer and is pulled at the first review less often than PI0, NEU and E1, but cancels more per filled share, both days | True |
| Every policy pulls more than half of its live orders at the first review (median life = the first-review landing), both days | True |
| Fewer than 1% of A's episodes lose (the worst-1% share is then 100% by construction), both days | True |
| A's severe-episode loss is largest after the earliest effect (the continuation window), both days | True |
| A's loss per episode is highest in the market window's last minute (phase 5), both days | True |
| A's loss per episode in phase 5 / the other phases' mean (c) | Oct 26 2.32 / 0.65 ; Oct 27 2.67 / 0.90 |
| Share of A's loss at a spread of at most 1 c | Oct 26 88.1% ; Oct 27 86.2% |
| A's spread component positive and movement component negative, both days | True |
| The sigma tercile carrying A's largest loss: Oct 26 / Oct 27 | high / mid |
| A's time-integrated inventory above PI0's, NEU's and E1's, both days | True |
| Share of A's value from fills under 100 ms old | Oct 26 91.0% ; Oct 27 55.5% |
| A below PI0, NEU and E1 under each B1a age weighting (FINDINGS C1), both days | True |
| A without fills under 100 ms old below 0, both days | True |
| A's E2-timed trade value below its WS-timed, Oct 27 (FINDINGS C12) | True |
