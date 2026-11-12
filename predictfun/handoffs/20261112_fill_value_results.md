# Explicit fill opportunity and branch value — first-stage diagnostic
As of 2026-11-12T09:15:02.311650+00:00. Six consumed test dates, 40,059 neutral actions. No untouched validation. All heads include zero-fill actions.

Two feature sets, KEEP/RESET branches, five heads per branch, six expanding folds: 120 models. Original parameters and 300 rounds retained; fill-event heads use binary objective. Training-only state-stratified means are the baseline. Exact definitions and source/input hashes are in `../reviews/fill_value_20261112/declaration.json`.

| Branch | Model | Actions / filled | Observed fill rate | Mean probability | Brier | AUC | Actual / predicted shares | Quantity RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| keep | baseline | 40,059 / 10,835 | 0.2705 | 0.2914 | 0.192991 | 0.5907 | 49661.08 / 53771.67 | 2.0809 |
| keep | market | 40,059 / 10,835 | 0.2705 | 0.2984 | 0.188653 | 0.6456 | 49661.08 / 53386.18 | 2.0574 |
| keep | queue | 40,059 / 10,835 | 0.2705 | 0.2930 | 0.185752 | 0.6641 | 49661.08 / 52344.27 | 2.0429 |
| reset | baseline | 40,059 / 3,171 | 0.0792 | 0.0892 | 0.073005 | 0.5144 | 14698.70 / 16500.14 | 1.2791 |
| reset | market | 40,059 / 3,171 | 0.0792 | 0.0846 | 0.073609 | 0.5569 | 14698.70 / 14591.68 | 1.2826 |
| reset | queue | 40,059 / 3,171 | 0.0792 | 0.0842 | 0.073523 | 0.5621 | 14698.70 / 14489.26 | 1.2819 |

Brier and quantity error describe simulated fill prediction, not calibrated live execution. Baseline means use only earlier training days, matched on side/order age/remaining/displayed depth and shrunk with fixed weight 50. Higher displayed depth is not a measurement of shares ahead. Missing marks and all upstream statuses remain in the machine receipt.

| Model | Actions | KEEP-minus-RESET RMSE, c | Pulls / keeps | Predicted / realized keep advantage on kept subset, c |
|---|---:|---:|---:|---:|
| baseline | 40,059 | 8.6362 | 39,318 / 741 | 106.84 / -228.94 |
| market_component | 40,059 | 8.6472 | 29,571 / 10,488 | 7337.61 / -4817.50 |
| queue_component | 40,059 | 8.6518 | 29,577 / 10,482 | 7394.70 / -4330.91 |
| market_direct | 40,059 | 8.6400 | 30,003 / 10,056 | 6419.22 / -3495.22 |
| queue_direct | 40,059 | 8.6332 | 30,211 / 9,848 | 6254.70 / -2923.70 |

| Pull policy | Pull count | Summed local benefit, c | Increment vs matched old queue, c [day interval] | Increment vs matched guard margin, c [day interval] |
|---|---:|---:|---:|---:|
| market_component | 29,571 | 32691.32 | -1118.89 [-2575.08, +182.66] | -2445.52 [-3589.30, -1579.36] |
| queue_component | 29,577 | 33177.92 | -907.48 [-1978.45, +245.08] | -1108.92 [-2059.18, -212.20] |

Same-count comparisons match day, side, UTC hour, order-age bucket and inventory sign. Guard margin is only the original value-line component, not its complete rules. Intervals resample six UTC days 1,000 times. Each component policy also has a predeclared 500-draw matched-count random null. All values here are overlapping local counterfactual utility, not portfolio P&L.

Computed predicates: `{"live_fill_calibrated": false, "new_candidate_promoted": false, "queue_component_selection_better_than_matched_direct": false, "queue_component_selection_day_interval_positive": false, "queue_component_value_mse_better_than_direct": false, "queue_fill_brier_better_than_baseline_both_branches": false, "queue_fill_brier_day_intervals_below_zero_both_branches": false, "queue_quantity_mse_better_than_baseline_both_branches": false, "sequential_portfolio_tested": false}`

This first stage retains the existing roughly two-second fill horizon and same-price reset after one second. It does not yet value later unfilled queue progress, forecast fill timing or raw incoming volume, or fix label/re-entry mismatch. Probability forecasts are diagnostic and constrain feasible quantity moments; value heads estimate unconditional joint gains/losses directly. There is no independent edge-times-fill-probability assumption and no extra queue bonus. No model promotion or live changes.

Full probability/value calibration bins, paired day differences, individual statuses, baselines, saved models, predictions and numerical checks are in the canonical receipt and data directory. Read the human review for interpretation.
