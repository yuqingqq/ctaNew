# Coherent paired fill/value model — numerical results
As of 2026-11-12T10:11:07.501416+00:00; 40,059 actions, six consumed dates. Source/declaration, models, calibration and status records are in `reviews/joint_fill_20261112/`.

| Branch | Model | Observed fill rate | Predicted fill rate | Brier | Quantity RMSE |
|---|---|---:|---:|---:|---:|
| keep | market_raw | 0.2705 | 0.3099 | 0.194680 | 2.0895 |
| keep | market_cal | 0.2705 | 0.2769 | 0.193677 | 2.0835 |
| keep | queue_raw | 0.2705 | 0.3070 | 0.193222 | 2.0826 |
| keep | queue_cal | 0.2705 | 0.2749 | 0.192499 | 2.0780 |
| keep | previous_queue | 0.2705 | 0.2930 | 0.185752 | 2.0429 |
| keep | previous_baseline | 0.2705 | 0.2914 | 0.192991 | 2.0809 |
| reset | market_raw | 0.0792 | 0.0897 | 0.072827 | 1.2776 |
| reset | market_cal | 0.0792 | 0.0801 | 0.072809 | 1.2774 |
| reset | queue_raw | 0.0792 | 0.0898 | 0.072824 | 1.2775 |
| reset | queue_cal | 0.0792 | 0.0801 | 0.072806 | 1.2774 |
| reset | previous_queue | 0.0792 | 0.0842 | 0.073523 | 1.2819 |
| reset | previous_baseline | 0.0792 | 0.0892 | 0.073005 | 1.2791 |

| Model | Value RMSE, c | Pulls / keeps | Predicted / realized KEEP advantage on kept subset, c |
|---|---:|---:|---:|
| market_raw | 8.6514 | 34,635 / 5,424 | 2349.97 / -2869.99 |
| market_cal | 8.6459 | 34,639 / 5,420 | 2090.58 / -2869.99 |
| queue_raw | 8.6508 | 34,767 / 5,292 | 2231.92 / -2761.98 |
| queue_cal | 8.6453 | 34,769 / 5,290 | 1988.68 / -2761.98 |
| original_direct | 8.6332 | 30,211 / 9,848 | 6254.70 / -2923.70 |
| previous_components | 8.6518 | 29,577 / 10,482 | 7394.70 / -4330.91 |

| Policy | Matched original-direct increment, c [day interval] | Matched guard-margin increment, c [day interval] |
|---|---:|---:|
| market_raw | -725.63 [-1966.70, +558.46] | -1125.53 [-1897.89, -270.87] |
| market_cal | -725.63 [-1966.70, +558.46] | -1125.53 [-1897.89, -270.87] |
| queue_raw | -615.32 [-2040.85, +791.36] | -1192.83 [-1872.17, -494.63] |
| queue_cal | -615.19 [-2040.34, +798.24] | -1192.83 [-1872.17, -494.63] |

Local values are overlapping short-horizon counterfactual utility, not portfolio P&L. All head values derive from a common mixture of paired valid outcomes. All 160,236 joint forecasts and 320,472 branch forecasts are checked for physical support; detailed computed counts are in JSON. Probability log loss clips only for reporting at 1e-9, with clipped counts recorded; the forecasts themselves are unchanged.

Core training excludes the preceding calibration day, so comparing to older full-training models is not equal-data architectural attribution. Raw versus calibrated variants use exactly the same core model; their difference isolates the calibration step. Day intervals resample six UTC days 1,000 times; nulls use 500 matched-count shuffles. No untouched-validation claim.

Computed predicates: `{"all_calibrators_supported": true, "all_forecasts_coherent": true, "calibrated_queue_brier_beats_previous_both_branches": false, "calibrated_queue_selection_beats_original": false, "calibrated_queue_selection_day_interval_positive": false, "calibration_improves_queue_brier_both_branches": true, "new_candidate_promoted": false, "sequential_portfolio_tested": false}`

The short fill horizon, same-price reset semantics, exact fill timing and live queue/fill calibration remain outside this stage. Read the human review before interpreting candidate quality. No live changes or model promotion.
