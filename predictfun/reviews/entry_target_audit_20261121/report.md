# ENTRY-TARGET-AUDIT-v1 — the accounting of the existing POST / WAIT labels (no fit, no new label, no replay)

Review d6d5d55e254c507db9799ef2fc1c6129d4657d06 (one next priority). Freeze `811fd7b3`, amendment `6157b1a0` (outputs under distinct names, before any output). Evidence as of 2026-11-21T07:51:47.351480+00:00. Development session Claude aea5bfb9.

## Verdict (every sentence chosen by a computed or declared predicate)

- **Declared checks:** all true ([checks.json](checks.json)); falsifier passed ([checks_falsifier.json](checks_falsifier.json)).
- **Population:** 24,416 POST rows (every status kept; expected 24,416) and 760 reached pairs. The identity Y = V_initial + D_focal + D_opposite holds on every accounted row (check B true).
- **Oct 24–25 (in-sample for the frozen models):** meaningful share 0.030, initial generation with no fill 0.971, Σ w|Y| / Σ w|V_initial| 0.945: rare_execution true, cancellation false, **diagnosis rare_execution**.
- **Oct 26–27 (adaptive development):** meaningful share 0.028, initial generation with no fill 0.974, Σ w|Y| / Σ w|V_initial| 1.141: rare_execution true, cancellation false, **diagnosis rare_execution**.
- **The two sets agree** on the diagnosis.
- **Recommendation (from the adaptive Oct 26–27 diagnosis, by the declared rule):** Rare execution only: the forced 50 ms deviation rarely changes execution, so relabeling the initial-generation component alone will not solve the entry problem.
- **The frozen models' POST groups carry a higher mean V_initial than their WAIT groups** in 3 of 4 day × side cells (SMOOTH) and 2 of 4 (RICH). This is a cross-tab, not a skill claim.
- **Scope:** an accounting of the existing labels. It fits nothing, builds no label and proposes no candidate here; any new target is a new declared candidate. V_initial is not a deployable entry advantage.

## Tables (audit_tables.py, copied from the outputs)

### Declared checks

| check | value |
|---|---|
| A_identity | true |
| B_accounting | true |
| C_predictions | true |
| D_reached | true |
| E_summaries | true |
| F_resources | true |
| G_suites | true |
| all_true | true |

Falsifier (clean copy, then one corruption each):

| check | clean | corrupted | applied |
|---|---|---|---|
| A | true | false | true |
| B | true | false | true |
| C | true | false | true |
| E | true | false | true |
| G | true | false | true |
| falsifier_ok | true |  |  |

### Declared predicates (thresholds: meaningful < 0.10 and no initial fill ≥ 0.50; Σ|Y| ≤ 0.5 Σ|V_initial|)

| set | meaningful share | initial no-fill share | Σ w \|V_initial\| | Σ w \|Y\| | rare_execution | cancellation | diagnosis |
|---|---|---|---|---|---|---|---|
| eval | 0.028 | 0.974 | 89,509.2 | 102,171.1 | true | false | rare_execution |
| train | 0.030 | 0.971 | 116,115.4 | 109,719.0 | true | false | rare_execution |

### Population: POST rows by set, day and status (every status kept)

| set | day | status | rows |
|---|---|---|---|
| eval | 20261026 | ok | 6,036 |
| eval | 20261026 | world:no_path:missing_endpoint | 46 |
| eval | 20261026 | world:no_path:stale_future_book | 8 |
| eval | 20261027 | ok | 6,052 |
| eval | 20261027 | world:no_path:missing_endpoint | 46 |
| train | 20261024 | ok | 6,084 |
| train | 20261024 | world:no_path:missing_endpoint | 30 |
| train | 20261025 | ok | 6,084 |
| train | 20261025 | world:no_path:missing_endpoint | 24 |
| train | 20261025 | world:no_path:stale_future_book | 6 |

### Accounting classes (weighted shares; exact zeros and residues counted; Oct 24–25 in-sample, Oct 26–27 adaptive)

| set | day | side | pick | rows | weight | no_post_fork_fills | economic_identical | same_cash_inventory | offset_by_mark | meaningful | meaningful | positive | negative | exact zero | residue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | all | all | all | 12,088 | 59,097.9 | 0.926 | 0.046 | 0 | 0.000 | 0.028 | 326 | 98 | 228 | 11,762 | 0 |
| eval | 20261026 | bid | all | 3,018 | 14,856.6 | 0.929 | 0.048 | 0 | 0 | 0.023 | 71 | 17 | 54 | 2,947 | 0 |
| eval | 20261026 | bid | extra | 1,509 | 14,485.6 | 0.930 | 0.048 | 0 | 0 | 0.023 | 34 | 5 | 29 | 1,475 | 0 |
| eval | 20261026 | bid | mandatory | 1,509 | 371.0 | 0.898 | 0.075 | 0 | 0 | 0.026 | 37 | 12 | 25 | 1,472 | 0 |
| eval | 20261026 | ask | all | 3,018 | 14,856.6 | 0.923 | 0.044 | 0 | 0.001 | 0.032 | 97 | 31 | 66 | 2,921 | 0 |
| eval | 20261026 | ask | extra | 1,509 | 14,495.2 | 0.924 | 0.043 | 0 | 0.001 | 0.032 | 49 | 14 | 35 | 1,460 | 0 |
| eval | 20261026 | ask | mandatory | 1,509 | 361.4 | 0.895 | 0.071 | 0 | 0 | 0.035 | 48 | 17 | 31 | 1,461 | 0 |
| eval | 20261027 | bid | all | 3,026 | 14,692.4 | 0.925 | 0.046 | 0 | 0 | 0.029 | 85 | 23 | 62 | 2,941 | 0 |
| eval | 20261027 | bid | extra | 1,513 | 14,327.9 | 0.926 | 0.045 | 0 | 0 | 0.029 | 44 | 10 | 34 | 1,469 | 0 |
| eval | 20261027 | bid | mandatory | 1,513 | 364.5 | 0.895 | 0.075 | 0 | 0 | 0.030 | 41 | 13 | 28 | 1,472 | 0 |
| eval | 20261027 | ask | all | 3,026 | 14,692.4 | 0.925 | 0.047 | 0 | 0.000 | 0.028 | 73 | 27 | 46 | 2,953 | 0 |
| eval | 20261027 | ask | extra | 1,513 | 14,330.6 | 0.926 | 0.046 | 0 | 0 | 0.028 | 43 | 14 | 29 | 1,470 | 0 |
| eval | 20261027 | ask | mandatory | 1,513 | 361.7 | 0.896 | 0.081 | 0 | 0.001 | 0.023 | 30 | 13 | 17 | 1,483 | 0 |
| train | all | all | all | 12,168 | 60,115.8 | 0.897 | 0.072 | 0.001 | 0.000 | 0.030 | 369 | 132 | 237 | 11,797 | 2 |
| train | 20261024 | bid | all | 3,042 | 15,203.3 | 0.884 | 0.075 | 0.001 | 0.001 | 0.040 | 108 | 38 | 70 | 2,933 | 1 |
| train | 20261024 | bid | extra | 1,521 | 14,811.6 | 0.885 | 0.073 | 0.001 | 0.001 | 0.040 | 61 | 23 | 38 | 1,460 | 0 |
| train | 20261024 | bid | mandatory | 1,521 | 391.7 | 0.822 | 0.139 | 0.002 | 0.001 | 0.036 | 47 | 15 | 32 | 1,473 | 1 |
| train | 20261024 | ask | all | 3,042 | 15,203.3 | 0.895 | 0.080 | 0.002 | 0 | 0.022 | 88 | 34 | 54 | 2,953 | 1 |
| train | 20261024 | ask | extra | 1,521 | 14,818.9 | 0.897 | 0.079 | 0.002 | 0 | 0.022 | 34 | 8 | 26 | 1,486 | 1 |
| train | 20261024 | ask | mandatory | 1,521 | 384.5 | 0.821 | 0.137 | 0.003 | 0 | 0.039 | 54 | 26 | 28 | 1,467 | 0 |
| train | 20261025 | bid | all | 3,042 | 14,854.6 | 0.910 | 0.064 | 0.000 | 0.001 | 0.026 | 77 | 30 | 47 | 2,965 | 0 |
| train | 20261025 | bid | extra | 1,521 | 14,474.6 | 0.911 | 0.062 | 0 | 0.001 | 0.026 | 39 | 16 | 23 | 1,482 | 0 |
| train | 20261025 | bid | mandatory | 1,521 | 380.0 | 0.855 | 0.115 | 0.001 | 0 | 0.029 | 38 | 14 | 24 | 1,483 | 0 |
| train | 20261025 | ask | all | 3,042 | 14,854.6 | 0.899 | 0.068 | 0 | 0 | 0.033 | 96 | 30 | 66 | 2,946 | 0 |
| train | 20261025 | ask | extra | 1,521 | 14,482.2 | 0.901 | 0.067 | 0 | 0 | 0.033 | 50 | 18 | 32 | 1,471 | 0 |
| train | 20261025 | ask | mandatory | 1,521 | 372.4 | 0.856 | 0.108 | 0 | 0 | 0.036 | 46 | 12 | 34 | 1,475 | 0 |

### The decomposition Y = V_initial + D_focal + D_opposite (cents; weighted means and absolute totals)

| set | day | side | mean Y | mean V_initial | mean D_focal | mean D_opposite | Σ w \|Y\| | Σ w \|V_initial\| | Σ w \|D_following\| | Σ\|Y\| / Σ\|V_initial\| | V_initial, D_following opposite signs | top-5 \|Y\| share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | all | all | -1.031 | -0.909 | -0.124 | 0.003 | 102,171.1 | 89,509.2 | 23,653.0 | 1.141 | 0.002 | 0.143 |
| eval | 20261026 | bid | -0.882 | -0.849 | -0.034 | 0 | 19,060.3 | 18,461.2 | 944.3 | 1.032 | 0.001 | 0.374 |
| eval | 20261026 | ask | -1.159 | -0.911 | -0.248 | 0 | 27,663.2 | 25,699.0 | 6,086.0 | 1.076 | 0.003 | 0.312 |
| eval | 20261027 | bid | -0.887 | -1.028 | 0.141 | 0 | 24,335.4 | 24,359.3 | 6,084.8 | 0.999 | 0.004 | 0.364 |
| eval | 20261027 | ask | -1.195 | -0.849 | -0.356 | 0.010 | 31,112.2 | 20,989.7 | 10,538.0 | 1.482 | 0.002 | 0.372 |
| train | all | all | -0.476 | -0.109 | -0.176 | -0.192 | 109,719.0 | 116,115.4 | 62,129.1 | 0.945 | 0.007 | 0.150 |
| train | 20261024 | bid | -0.502 | -0.375 | -0.281 | 0.154 | 36,867.9 | 35,686.2 | 18,426.3 | 1.033 | 0.010 | 0.359 |
| train | 20261024 | ask | -0.503 | -0.287 | -0.160 | -0.056 | 14,401.8 | 18,577.7 | 10,361.4 | 0.775 | 0.007 | 0.337 |
| train | 20261025 | bid | -0.250 | -0.204 | -0.060 | 0.014 | 22,747.1 | 19,390.6 | 11,238.0 | 1.173 | 0.005 | 0.430 |
| train | 20261025 | ask | -0.650 | 0.439 | -0.199 | -0.890 | 35,702.2 | 42,460.9 | 22,103.4 | 0.841 | 0.005 | 0.267 |

### The forced generation's fills and the WAIT arm's next POST (weighted shares; no refusal or queue cause asserted)

| set | day | side | initial: no fill | partial | full | WAIT next POST ≤ 100 ms | 100–500 ms | 500–1,000 ms | > 1 s | no later POST |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | all | all | 0.974 | 0.015 | 0.011 | 0.048 | 0.052 | 0.044 | 0.154 | 0.703 |
| eval | 20261026 | bid | 0.979 | 0.013 | 0.008 | 0.063 | 0.068 | 0.058 | 0.188 | 0.623 |
| eval | 20261026 | ask | 0.966 | 0.022 | 0.012 | 0.040 | 0.045 | 0.038 | 0.141 | 0.736 |
| eval | 20261027 | bid | 0.973 | 0.014 | 0.013 | 0.051 | 0.044 | 0.037 | 0.154 | 0.713 |
| eval | 20261027 | ask | 0.978 | 0.010 | 0.013 | 0.038 | 0.049 | 0.041 | 0.131 | 0.740 |
| train | all | all | 0.971 | 0.018 | 0.011 | 0.083 | 0.072 | 0.060 | 0.200 | 0.585 |
| train | 20261024 | bid | 0.963 | 0.023 | 0.014 | 0.099 | 0.089 | 0.073 | 0.247 | 0.492 |
| train | 20261024 | ask | 0.977 | 0.015 | 0.008 | 0.079 | 0.071 | 0.056 | 0.179 | 0.616 |
| train | 20261025 | bid | 0.976 | 0.015 | 0.009 | 0.089 | 0.076 | 0.064 | 0.196 | 0.574 |
| train | 20261025 | ask | 0.970 | 0.018 | 0.012 | 0.066 | 0.049 | 0.048 | 0.179 | 0.657 |

### Strata (weighted; sparse cells are counts, not claims)

| set | dimension | value | rows | meaningful | initial no fill | mean V_initial | mean D_following | mean Y |
|---|---|---|---|---|---|---|---|---|
| eval | source | delayed | 3,006 | 0.029 | 0.973 | -1.182 | -0.039 | -1.221 |
| eval | source | joint_chase | 108 | 0.028 | 0.981 | -3.355 | 0.128 | -3.227 |
| eval | source | joint_post_keep | 67 | 0.119 | 0.911 | 0.105 | 2.134 | 2.239 |
| eval | source | neutral | 6,044 | 0.029 | 0.975 | -0.438 | 0.133 | -0.305 |
| eval | source | pi0 | 2,863 | 0.025 | 0.976 | -0.578 | -0.284 | -0.862 |
| eval | hbin | (0.0, 100.0] | 59 | 0 | 1 | 0 | 0 | 0 |
| eval | hbin | (100.0, 250.0] | 160 | 0.025 | 0.975 | -0.514 | 0 | -0.514 |
| eval | hbin | (1000.0, 2000.0] | 1,187 | 0.027 | 0.974 | -1.180 | 0.014 | -1.165 |
| eval | hbin | (2000.0, 3500.0] | 1,778 | 0.029 | 0.973 | -0.915 | 0.146 | -0.769 |
| eval | hbin | (250.0, 500.0] | 285 | 0.039 | 0.961 | -1.014 | -0.300 | -1.314 |
| eval | hbin | (3500.0, 5000.0] | 8,039 | 0.028 | 0.975 | -0.974 | -0.446 | -1.419 |
| eval | hbin | (500.0, 1000.0] | 580 | 0.028 | 0.978 | -0.249 | 0.024 | -0.225 |
| eval | c_gen | 1 | 11,615 | 0.026 | 0.975 | -0.845 | -0.043 | -0.888 |
| eval | c_gen | 2 | 255 | 0.055 | 0.968 | -2.289 | -2.137 | -4.426 |
| eval | c_gen | >=3 | 218 | 0.046 | 0.968 | -0.985 | 0.148 | -0.837 |
| eval | c_other | absent | 11,860 | 0.028 | 0.974 | -0.900 | -0.131 | -1.031 |
| eval | c_other | cancel_pending | 67 | 0.016 | 0.984 | -5.424 | 0 | -5.424 |
| eval | c_other | live | 71 | 0.028 | 0.972 | 0.895 | 0.220 | 1.115 |
| eval | c_other | post_pending | 90 | 0.033 | 0.978 | 0.450 | 0.155 | 0.605 |
| train | source | delayed | 3,110 | 0.038 | 0.965 | 0.582 | -0.831 | -0.248 |
| train | source | joint_chase | 102 | 0.009 | 0.991 | -0.043 | 0 | -0.043 |
| train | source | joint_post_keep | 66 | 0.046 | 0.954 | -10.312 | 2.646 | -7.666 |
| train | source | neutral | 6,084 | 0.035 | 0.967 | -0.753 | 0.403 | -0.350 |
| train | source | pi0 | 2,806 | 0.022 | 0.978 | -0.602 | 0.019 | -0.583 |
| train | hbin | (0.0, 100.0] | 54 | 0.037 | 0.963 | -0.091 | 0 | -0.091 |
| train | hbin | (100.0, 250.0] | 168 | 0.025 | 0.969 | -0.537 | 0.043 | -0.494 |
| train | hbin | (1000.0, 2000.0] | 1,241 | 0.027 | 0.975 | 0.341 | -0.032 | 0.308 |
| train | hbin | (2000.0, 3500.0] | 1,784 | 0.028 | 0.977 | 0.143 | -0.227 | -0.084 |
| train | hbin | (250.0, 500.0] | 263 | 0.038 | 0.962 | -0.632 | -0.348 | -0.981 |
| train | hbin | (3500.0, 5000.0] | 8,092 | 0.035 | 0.965 | -0.428 | -0.876 | -1.304 |
| train | hbin | (500.0, 1000.0] | 566 | 0.025 | 0.976 | -0.303 | 0.235 | -0.069 |
| train | c_gen | 1 | 11,481 | 0.030 | 0.972 | -0.083 | -0.412 | -0.496 |
| train | c_gen | 2 | 325 | 0.037 | 0.956 | -2.955 | 0.585 | -2.370 |
| train | c_gen | >=3 | 362 | 0.021 | 0.981 | 2.050 | -0.532 | 1.518 |
| train | c_other | absent | 11,846 | 0.030 | 0.972 | -0.038 | -0.438 | -0.476 |
| train | c_other | cancel_pending | 93 | 0.032 | 0.958 | -1.678 | 1.366 | -0.311 |
| train | c_other | live | 117 | 0.034 | 0.966 | -6.452 | 1.497 | -4.955 |
| train | c_other | post_pending | 112 | 0.027 | 0.973 | 4.022 | 0 | 4.022 |

### The frozen models' actions against the components (cross-tab only; Oct 24–25 in-sample)

| set | model | day | side | action | rows | weight | mean Y | mean V_initial | mean D_following | meaningful | initial no fill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | SMOOTH | 20261026 | bid | POST | 1,247 | 7,201.6 | -0.972 | -0.908 | -0.064 | 0.022 | 0.980 |
| eval | SMOOTH | 20261026 | bid | WAIT | 1,771 | 7,655.0 | -0.798 | -0.792 | -0.005 | 0.024 | 0.979 |
| eval | SMOOTH | 20261026 | ask | POST | 802 | 4,243.3 | -0.009 | -0.008 | -0.001 | 0.003 | 0.997 |
| eval | SMOOTH | 20261026 | ask | WAIT | 2,216 | 10,613.2 | -1.619 | -1.273 | -0.346 | 0.044 | 0.954 |
| eval | SMOOTH | 20261027 | bid | POST | 1,184 | 6,992.0 | -0.811 | -0.707 | -0.104 | 0.034 | 0.967 |
| eval | SMOOTH | 20261027 | bid | WAIT | 1,842 | 7,700.4 | -0.956 | -1.320 | 0.364 | 0.024 | 0.978 |
| eval | SMOOTH | 20261027 | ask | POST | 889 | 4,451.2 | -0.761 | 0.210 | -0.971 | 0.017 | 0.991 |
| eval | SMOOTH | 20261027 | ask | WAIT | 2,137 | 10,241.2 | -1.383 | -1.309 | -0.074 | 0.033 | 0.972 |
| eval | RICH | 20261026 | bid | POST | 1,571 | 7,771.1 | -1.312 | -1.258 | -0.055 | 0.028 | 0.975 |
| eval | RICH | 20261026 | bid | WAIT | 1,447 | 7,085.5 | -0.410 | -0.400 | -0.010 | 0.017 | 0.984 |
| eval | RICH | 20261026 | ask | POST | 1,526 | 7,613.6 | -1.852 | -1.588 | -0.264 | 0.027 | 0.973 |
| eval | RICH | 20261026 | ask | WAIT | 1,492 | 7,243.0 | -0.431 | -0.200 | -0.231 | 0.038 | 0.959 |
| eval | RICH | 20261027 | bid | POST | 1,605 | 7,611.4 | -0.433 | -0.396 | -0.038 | 0.027 | 0.977 |
| eval | RICH | 20261027 | bid | WAIT | 1,421 | 7,081.0 | -1.374 | -1.708 | 0.334 | 0.031 | 0.969 |
| eval | RICH | 20261027 | ask | POST | 1,648 | 7,936.9 | -0.705 | -0.125 | -0.580 | 0.023 | 0.983 |
| eval | RICH | 20261027 | ask | WAIT | 1,378 | 6,755.5 | -1.770 | -1.700 | -0.071 | 0.034 | 0.972 |
| train | SMOOTH | 20261024 | bid | POST | 829 | 5,285.1 | 2.145 | 1.544 | 0.601 | 0.052 | 0.960 |
| train | SMOOTH | 20261024 | bid | WAIT | 2,213 | 9,918.2 | -1.913 | -1.397 | -0.516 | 0.034 | 0.964 |
| train | SMOOTH | 20261024 | ask | POST | 874 | 4,360.5 | 0.456 | 0.641 | -0.185 | 0.012 | 0.984 |
| train | SMOOTH | 20261024 | ask | WAIT | 2,168 | 10,842.9 | -0.888 | -0.659 | -0.229 | 0.027 | 0.974 |
| train | SMOOTH | 20261025 | bid | POST | 896 | 5,630.0 | 1.158 | 0.691 | 0.467 | 0.029 | 0.975 |
| train | SMOOTH | 20261025 | bid | WAIT | 2,146 | 9,224.6 | -1.109 | -0.750 | -0.359 | 0.024 | 0.977 |
| train | SMOOTH | 20261025 | ask | POST | 1,040 | 4,989.9 | 1.799 | 4.398 | -2.600 | 0.014 | 0.981 |
| train | SMOOTH | 20261025 | ask | WAIT | 2,002 | 9,864.6 | -1.888 | -1.563 | -0.325 | 0.043 | 0.964 |
| train | RICH | 20261024 | bid | POST | 1,450 | 7,096.9 | 1.955 | 1.394 | 0.561 | 0.035 | 0.964 |
| train | RICH | 20261024 | bid | WAIT | 1,592 | 8,106.4 | -2.654 | -1.924 | -0.730 | 0.045 | 0.962 |
| train | RICH | 20261024 | ask | POST | 1,419 | 7,109.8 | 0.411 | 0.354 | 0.057 | 0.014 | 0.982 |
| train | RICH | 20261024 | ask | WAIT | 1,623 | 8,093.6 | -1.306 | -0.849 | -0.456 | 0.030 | 0.973 |
| train | RICH | 20261025 | bid | POST | 1,448 | 7,040.0 | 1.200 | 0.856 | 0.344 | 0.022 | 0.979 |
| train | RICH | 20261025 | bid | WAIT | 1,594 | 7,814.6 | -1.556 | -1.159 | -0.397 | 0.029 | 0.974 |
| train | RICH | 20261025 | ask | POST | 1,443 | 6,826.3 | 1.684 | 3.340 | -1.656 | 0.021 | 0.978 |
| train | RICH | 20261025 | ask | WAIT | 1,599 | 8,028.3 | -2.634 | -2.027 | -0.607 | 0.043 | 0.963 |

### R4E's reached SMOOTH entry pairs (a world / side-balanced diagnostic; unweighted; never mixed with the sample)

| day | pairs | no_post_fork_fills | economic_identical | same_cash_inventory | offset_by_mark | meaningful | mean Y | mean V_initial | mean D_following | initial no fill |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261026.0 | 376 | 340 | 21 | 0 | 0 | 15 | -1.932 | -2.171 | 0.239 | 0.963 |
| 20261027.0 | 384 | 348 | 22 | 0 | 0 | 14 | -0.533 | -1.161 | 0.628 | 0.966 |

### Review d6d5d55e item 2: R4E's first-review table with each leg's score and policy named

| day | policy | entry leg | entry score | mean entry score | review leg | review score | mean review score | first-review action | generations |
|---|---|---|---|---|---|---|---|---|---|
| 20261026 | ENTRY_ONLY | RICH | A: RICH POST − WAIT advantage | 1.662 | PI0 | PI0 V_retain | 1.300 | KEEP | 81 |
| 20261026 | ENTRY_ONLY | RICH | A: RICH POST − WAIT advantage | 1.353 | PI0 | PI0 V_retain | -13.629 | PULL | 4,293 |
| 20261026 | PI0 | PI0 | PI0 V_post | 1.106 | PI0 | PI0 V_retain | 1.165 | KEEP | 62 |
| 20261026 | PI0 | PI0 | PI0 V_post | 1.246 | PI0 | PI0 V_retain | -5.855 | PULL | 223 |
| 20261026 | SMOOTH | SMOOTH | A: SMOOTH POST − WAIT advantage | 0.437 | PI0 | PI0 V_retain | 0.636 | KEEP | 61 |
| 20261026 | SMOOTH | SMOOTH | A: SMOOTH POST − WAIT advantage | 0.442 | PI0 | PI0 V_retain | -12.657 | PULL | 3,325 |
| 20261027 | ENTRY_ONLY | RICH | A: RICH POST − WAIT advantage | 1.956 | PI0 | PI0 V_retain | 2.005 | KEEP | 75 |
| 20261027 | ENTRY_ONLY | RICH | A: RICH POST − WAIT advantage | 1.295 | PI0 | PI0 V_retain | -13.432 | PULL | 4,440 |
| 20261027 | PI0 | PI0 | PI0 V_post | 1.847 | PI0 | PI0 V_retain | 2.443 | KEEP | 74 |
| 20261027 | PI0 | PI0 | PI0 V_post | 1.745 | PI0 | PI0 V_retain | -5.866 | PULL | 266 |
| 20261027 | SMOOTH | SMOOTH | A: SMOOTH POST − WAIT advantage | 0.351 | PI0 | PI0 V_retain | 1.791 | KEEP | 64 |
| 20261027 | SMOOTH | SMOOTH | A: SMOOTH POST − WAIT advantage | 0.450 | PI0 | PI0 V_retain | -12.623 | PULL | 3,491 |

The two legs' scores are different quantities (different targets, alternatives and horizons); their difference is not a forecast change.

R4E's predicates with the two populations named (the control uses the mandatory anchors; the selected mean uses all sampled rows):

| day | side | candidate | admissions | all-sampled selected-POST weighted mean | mandatory-anchor control value | draw mean | draw p95 |
|---|---|---|---|---|---|---|---|
| 20261026 | bid | RICH | 1,571 | -1.3124 | -0.4067 | -0.6120 | -0.4012 |
| 20261026 | bid | SMOOTH | 1,247 | -0.9721 | -0.4859 | -0.4995 | -0.2302 |
| 20261026 | ask | RICH | 1,526 | -1.8519 | -0.0224 | 0.1284 | 0.5059 |
| 20261026 | ask | SMOOTH | 802 | -0.0087 | 0.0636 | 0.0885 | 0.1382 |
| 20261027 | bid | RICH | 1,605 | -0.4334 | -0.7013 | -0.6573 | -0.1501 |
| 20261027 | bid | SMOOTH | 1,184 | -0.8109 | -0.3485 | -0.0913 | 0.3253 |
| 20261027 | ask | RICH | 1,648 | -0.7051 | 0.2235 | 0.2171 | 0.3824 |
| 20261027 | ask | SMOOTH | 889 | -0.7612 | -0.0456 | -0.0336 | 0.0363 |

## Review d6d5d55e

- **Item 1:** the smoothing experiment is closed as failed; candidates and multiplicity are kept; this audit is the next step. No threshold mining or regularization sweep.
- **Item 2:** R4E's first-review table is renamed per leg (A, V_post, V_retain), and its predicates table separates the mandatory-anchor control value from the all-sampled selected-POST mean (tables above). No data rerun.
- **Item 3:** the replay checker is not reused here, so the review's verified workaround stands. A future replay checker adds episode END, settings, fill-time bounds and common marks with single-field falsifiers.

## Provenance

- Mutation checks: the final audit suite 16 / 16. Run 1 (14 / 15) had a crash at a summary fixture, now guarded. Run 2 (15 / 15) preceded the amendment.
- The first audit run was stopped by hand before writing any file: the per-row reached data would have been overwritten by its summary. The amendment gives every output its own name.

## Cost

- 0.022 core-hours of CPU over every run_light job under logs/ (the ceiling is 1); check F true.

## Limits

- In-sample Oct 24–25 and adaptive Oct 26–27 development days only: no interval, no validation.
- The accounting explains the existing label under its frozen continuation. A different decision unit needs its own declared alternative and costs before any label is built.

