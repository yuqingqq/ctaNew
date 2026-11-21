# PI0-MOVEMENT-SHRINK-v1 — report

2026-11-21 12:43Z, session aea5bfb9. Review 7ad03681's one next priority.
- Freeze `9720813606172cde68ccdf85947c93c4ead28470`: module, selftest, harness, declaration, manifest, checker, launcher; nothing fitted.
- Fit `bd5e2f01`: fit.json on the Oct 26 TRAIN keys only, committed before any Oct 27 score.
- Results: the commit carrying this report.
- Scope: a new adaptive experiment after the movement audit, not a retroactive pass of its gate; chronological consumed-day development (fit Oct 26, measure Oct 27), not unseen validation. One new candidate.

## The computed readings (audit.json `predicates`)

| Head | Decision | Reasons | Prediction (pooled) |
|---|---|---|---|
| entry | **INCONCLUSIVE** | candidate_mask_equals_prior, inadequate_movable_support | movement MSE improves, value MSE improves, movement bias not reduced |
| first_review | **INCONCLUSIVE** | candidate_mask_equals_prior, inadequate_movable_support | movement MSE does not improve, value MSE does not improve, movement bias not reduced |

## The fit (Oct 26 only)

Every group has fit support. The raw weighted least-squares slope exceeds 1 in all four groups, so lambda clips to 1 everywhere: the fitted candidate is the pooled-prior control, on every row.

| Group | status | n | fills | weight | weighted fills | mu (c / fill) | lambda raw | lambda | prior binding (weight share) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entry, side 0 | ok | 82 | 20 | 182.00 | 36.50 | -38.89 | 2.103 | 1.000 | 0.015 |
| entry, side 1 | ok | 84 | 22 | 137.00 | 46.50 | -107.83 | 1.847 | 1.000 | 0.164 |
| first_review, side 0 | ok | 76 | 18 | 170.00 | 33.50 | -39.53 | 1.693 | 1.000 | 0.007 |
| first_review, side 1 | ok | 75 | 19 | 121.50 | 41.25 | -119.85 | 1.752 | 1.000 | 0.214 |

## Prediction on Oct 27 (weighted; c per opportunity; c_m per filled opportunity)

- **entry:** movement MSE 2,248.6 → 2,125.9 c² (improves; zero 2,313.1); value MSE 2,050.2 → 2,010.8 (improves; zero 2,046.4, mechanical 2,099.3); movement bias +6.62 → -7.72 c (not reduced); side 0: movement MSE does not improve, value MSE does not improve; side 1: movement MSE improves, value MSE improves.
- **first_review:** movement MSE 1,578.3 → 1,808.5 c² (does not improve; zero 1,838.6); value MSE 1,495.7 → 1,749.9 (does not improve; zero 1,665.5, mechanical 1,714.8); movement bias -0.63 → -11.01 c (not reduced); side 0: movement MSE does not improve, value MSE improves; side 1: movement MSE does not improve, value MSE does not improve.

| head | side | arm | n | fills (unweighted / weighted) | mean E[qm] | mean M | movement bias | movement MSE | zero-movement MSE | V bias | V MSE | zero | mechanical | c_m (weighted / unweighted) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entry | all | original | 163 | 46 / 125.50 | -1.62 | -8.24 | +6.62 | 2,248.6 | 2,313.1 | +5.46 | 2,050.2 | 2,046.4 | 2,099.3 | +17.91 / +31.19 |
| entry | all | prior | 163 | 46 / 125.50 | -15.95 | -8.24 | -7.72 | 2,125.9 | 2,313.1 | -8.88 | 2,010.8 | 2,046.4 | 2,099.3 | -37.60 / -35.49 |
| entry | all | candidate | 163 | 46 / 125.50 | -15.95 | -8.24 | -7.72 | 2,125.9 | 2,313.1 | -8.88 | 2,010.8 | 2,046.4 | 2,099.3 | -37.60 / -35.49 |
| entry | 0 | original | 87 | 22 / 83.50 | -1.39 | -1.86 | +0.46 | 188.5 | 161.1 | +0.27 | 143.3 | 143.4 | 118.3 | +0.72 / +3.95 |
| entry | 0 | prior | 87 | 22 / 83.50 | -9.27 | -1.86 | -7.41 | 281.5 | 161.1 | -7.60 | 249.7 | 143.4 | 118.3 | -33.71 / -29.40 |
| entry | 0 | candidate | 87 | 22 / 83.50 | -9.27 | -1.86 | -7.41 | 281.5 | 161.1 | -7.60 | 249.7 | 143.4 | 118.3 | -33.71 / -29.40 |
| entry | 1 | original | 76 | 24 / 42.00 | -1.97 | -18.36 | +16.39 | 5,518.9 | 5,729.1 | +13.69 | 5,077.2 | 5,067.2 | 5,244.0 | +52.08 / +56.16 |
| entry | 1 | prior | 76 | 24 / 42.00 | -26.56 | -18.36 | -8.20 | 5,053.8 | 5,729.1 | -10.90 | 4,806.5 | 5,067.2 | 5,244.0 | -45.32 / -41.07 |
| entry | 1 | candidate | 76 | 24 / 42.00 | -26.56 | -18.36 | -8.20 | 5,053.8 | 5,729.1 | -10.90 | 4,806.5 | 5,067.2 | 5,244.0 | -45.32 / -41.07 |
| first_review | all | original | 148 | 42 / 117.25 | -7.26 | -6.63 | -0.63 | 1,578.3 | 1,838.6 | -1.02 | 1,495.7 | 1,665.5 | 1,714.8 | -7.17 / +2.03 |
| first_review | all | prior | 148 | 42 / 117.25 | -17.64 | -6.63 | -11.01 | 1,808.5 | 1,838.6 | -11.39 | 1,749.9 | 1,665.5 | 1,714.8 | -48.12 / -49.34 |
| first_review | all | candidate | 148 | 42 / 117.25 | -17.64 | -6.63 | -11.01 | 1,808.5 | 1,838.6 | -11.39 | 1,749.9 | 1,665.5 | 1,714.8 | -48.12 / -49.34 |
| first_review | 0 | original | 85 | 19 / 76.25 | -6.05 | -2.15 | -3.90 | 318.3 | 207.6 | -3.88 | 257.1 | 126.3 | 129.8 | -15.04 / -9.42 |
| first_review | 0 | prior | 85 | 19 / 76.25 | -9.62 | -2.15 | -7.47 | 326.0 | 207.6 | -7.45 | 250.4 | 126.3 | 129.8 | -33.17 / -24.72 |
| first_review | 0 | candidate | 85 | 19 / 76.25 | -9.62 | -2.15 | -7.47 | 326.0 | 207.6 | -7.45 | 250.4 | 126.3 | 129.8 | -33.17 / -24.72 |
| first_review | 1 | original | 63 | 23 / 41.00 | -9.50 | -14.92 | +5.42 | 3,913.0 | 4,860.9 | +4.28 | 3,790.6 | 4,517.5 | 4,651.8 | +7.48 / +11.50 |
| first_review | 1 | prior | 63 | 23 / 41.00 | -32.48 | -14.92 | -17.56 | 4,555.5 | 4,860.9 | -18.70 | 4,528.4 | 4,517.5 | 4,651.8 | -75.92 / -69.69 |
| first_review | 1 | candidate | 63 | 23 / 41.00 | -32.48 | -14.92 | -17.56 | 4,555.5 | 4,860.9 | -18.70 | 4,528.4 | 4,517.5 | 4,651.8 | -75.92 / -69.69 |
| first_pull | all | original | 134 | 40 / 110.00 | -9.02 | -7.05 | -1.96 | 1,759.5 | 2,010.2 | -2.44 | 1,636.2 | 1,816.7 | 1,872.0 | -9.90 / -1.98 |
| first_pull | all | prior | 134 | 40 / 110.00 | -18.09 | -7.05 | -11.04 | 1,947.8 | 2,010.2 | -11.51 | 1,884.8 | 1,816.7 | 1,872.0 | -49.21 / -50.36 |
| first_pull | all | candidate | 134 | 40 / 110.00 | -18.09 | -7.05 | -11.04 | 1,947.8 | 2,010.2 | -11.51 | 1,884.8 | 1,816.7 | 1,872.0 | -49.21 / -50.36 |
| first_pull | 0 | original | 75 | 17 / 69.00 | -8.15 | -2.02 | -6.13 | 420.9 | 222.4 | -6.02 | 306.5 | 139.0 | 144.7 | -18.37 / -13.41 |
| first_pull | 0 | prior | 75 | 17 / 69.00 | -9.95 | -2.02 | -7.93 | 352.9 | 222.4 | -7.82 | 274.5 | 139.0 | 144.7 | -33.63 / -24.50 |
| first_pull | 0 | candidate | 75 | 17 / 69.00 | -9.95 | -2.02 | -7.93 | 352.9 | 222.4 | -7.82 | 274.5 | 139.0 | 144.7 | -33.63 / -24.50 |
| first_pull | 1 | original | 59 | 23 / 41.00 | -10.51 | -15.73 | +5.22 | 4,069.4 | 5,095.5 | +3.74 | 3,930.9 | 4,712.0 | 4,852.8 | +4.36 / +6.47 |
| first_pull | 1 | prior | 59 | 23 / 41.00 | -32.13 | -15.73 | -16.40 | 4,700.0 | 5,095.5 | -17.89 | 4,663.5 | 4,712.0 | 4,852.8 | -75.43 / -69.47 |
| first_pull | 1 | candidate | 59 | 23 / 41.00 | -32.13 | -15.73 | -16.40 | 4,700.0 | 5,095.5 | -17.89 | 4,663.5 | 4,712.0 | 4,852.8 | -75.43 / -69.47 |

first_pull rows are descriptive (the first-review parameters), never in the main scores.

## Local actions on Oct 27 (the neutral hold labels; stages never summed)

- **entry:** the candidate POSTs on 2 of 163 rows (J +0.000 c per opportunity, mean among the chosen +0.00); original 163 rows, J -3.740; all -3.740; none +0.000; prior 2 rows, J +0.000.
- **first_review:** the candidate KEEPs 6 of 148 rows (J -0.196 c per opportunity, mean among the chosen -4.43); original 35 rows, J -0.852; all -3.087; none +0.000; prior 6 rows, J -0.196.

| head | side | rule | n | chosen | chosen weight | J (c / opp.) | mean among chosen | worlds | Kish n | severe n / sum | top-5 gen / world (weighted) | changed vs original |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entry | all | none | 163 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 163 |
| entry | all | all | 163 | 163 | 370.00 | -3.740 | -3.74 | 78 | 111.4 | 11 / -2,501.2 | 0.574 / 0.644 | 0 |
| entry | all | original | 163 | 163 | 370.00 | -3.740 | -3.74 | 78 | 111.4 | 11 / -2,501.2 | 0.574 / 0.644 | 0 |
| entry | all | prior | 163 | 2 | 2.00 | +0.000 | +0.00 | 2 | 2.0 | 0 / 0.0 | — / — | 161 |
| entry | all | candidate | 163 | 2 | 2.00 | +0.000 | +0.00 | 2 | 2.0 | 0 / 0.0 | — / — | 161 |
| entry | 0 | none | 87 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 87 |
| entry | 0 | all | 87 | 87 | 227.00 | +1.464 | +1.46 | 52 | 57.6 | 3 / -281.2 | 0.542 / 0.624 | 0 |
| entry | 0 | original | 87 | 87 | 227.00 | +1.464 | +1.46 | 52 | 57.6 | 3 / -281.2 | 0.542 / 0.624 | 0 |
| entry | 0 | prior | 87 | 2 | 2.00 | +0.000 | +0.00 | 2 | 2.0 | 0 / 0.0 | — / — | 85 |
| entry | 0 | candidate | 87 | 2 | 2.00 | +0.000 | +0.00 | 2 | 2.0 | 0 / 0.0 | — / — | 85 |
| entry | 1 | none | 76 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 76 |
| entry | 1 | all | 76 | 76 | 143.00 | -12.002 | -12.00 | 49 | 61.1 | 8 / -2,220.0 | 0.768 / 0.829 | 0 |
| entry | 1 | original | 76 | 76 | 143.00 | -12.002 | -12.00 | 49 | 61.1 | 8 / -2,220.0 | 0.768 / 0.829 | 0 |
| entry | 1 | prior | 76 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 76 |
| entry | 1 | candidate | 76 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 76 |
| first_review | all | none | 148 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 35 |
| first_review | all | all | 148 | 148 | 344.50 | -3.087 | -3.09 | 72 | 100.8 | 10 / -2,060.7 | 0.593 / 0.665 | 113 |
| first_review | all | original | 148 | 35 | 71.75 | -0.852 | -4.09 | 28 | 28.4 | 3 / -258.8 | 0.835 / 0.910 | 0 |
| first_review | all | prior | 148 | 6 | 15.25 | -0.196 | -4.43 | 6 | 2.6 | 1 / -67.5 | 1.000 / 1.000 | 41 |
| first_review | all | candidate | 148 | 6 | 15.25 | -0.196 | -4.43 | 6 | 2.6 | 1 / -67.5 | 1.000 / 1.000 | 41 |
| first_review | 0 | none | 85 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 20 |
| first_review | 0 | all | 85 | 85 | 223.75 | +0.995 | +1.00 | 51 | 56.3 | 3 / -281.2 | 0.606 / 0.704 | 65 |
| first_review | 0 | original | 85 | 20 | 47.75 | -0.798 | -3.74 | 17 | 16.6 | 1 / -168.8 | 1.000 / 1.000 | 0 |
| first_review | 0 | prior | 85 | 6 | 15.25 | -0.302 | -4.43 | 6 | 2.6 | 1 / -67.5 | 1.000 / 1.000 | 26 |
| first_review | 0 | candidate | 85 | 6 | 15.25 | -0.302 | -4.43 | 6 | 2.6 | 1 / -67.5 | 1.000 / 1.000 | 26 |
| first_review | 1 | none | 63 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 15 |
| first_review | 1 | all | 63 | 63 | 120.75 | -10.652 | -10.65 | 41 | 50.6 | 7 / -1,779.5 | 0.790 / 0.835 | 48 |
| first_review | 1 | original | 63 | 15 | 24.00 | -0.952 | -4.79 | 12 | 12.9 | 2 / -90.0 | 1.000 / 1.000 | 0 |
| first_review | 1 | prior | 63 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 15 |
| first_review | 1 | candidate | 63 | 0 | 0.00 | +0.000 | — | 0 | 0.0 | 0 / 0.0 | — / — | 15 |

| head | strata | free strata | free rows | chosen | J | null mean | p05 | p95 | percentile | weight-mismatch draws |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entry | 45 | 2 | 6 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | 1.000 | 0 |
| first_review | 45 | 6 | 21 | 6 | -0.196 | -0.061 | -0.227 | +0.000 | 0.244 | 0 |

## Validity

- Checks A–J: 10 / 10 true (`checks.json`), each recomputing its values independently (identity and freeze; membership and every eval row against the audit table with the rowwise identity; the fit; every prediction and mask; every score, calibration and action row; all null draws; the readings; resources and the suite).
- Falsifier: 15 / 15 corruptions of real copies flagged by their check, with clean copies true (`checks_falsifier.json`).
- Mutation suite: 56 / 56 planted errors caught by a selftest check (a first 55 / 56 run kept: a fixture gap).
- Populations as frozen: TRAIN 166 / 151, EVAL 163 / 148, descriptive first pull 134; every eligible row reconciled with the audit's committed table.
- Cost as of 2026-11-21 12:41Z: 121 run_light jobs, 0.064 core-hours, peak 244 MiB (fit 238, score 238 MiB).

## Limits

- One evaluation day; related generations; no interval; the 500 draws are not 500 market samples.
- Entry evaluation filters PI0's own admissions (no evidence for posts PI0 rejected); the review is scored on PI0's reached live states; local choices on neutral labels, not a combined strategy.
- Nothing else is tuned in this round.
