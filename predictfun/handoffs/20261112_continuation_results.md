# Full-engine continuation — numerical results
As of 2026-11-12T14:25:17.782695+00:00; n=9,917 sampled eligible actions on six consumed dates. See receipt populations for every sampling/exclusion status.

| Horizon | KEEP shares | PULL shares | KEEP-minus-PULL local utility, cents | Nonzero actions |
|---|---:|---:|---:|---:|
| 2s | 37762.38 | 26312.25 | -15247.61 | 3038 |
| 10s | 160665.07 | 151263.89 | -12317.16 | 4442 |
| 30s | 383931.46 | 375279.38 | -11752.07 | 4572 |

| Model | Value RMSE, cents | Pulls | Keeps | Predicted / observed kept value, cents |
|---|---:|---:|---:|---:|
| market_raw | 15.6867 | 9917 | 0 | 0.00 / 0.00 |
| market_cal | 15.6867 | 9917 | 0 | 0.00 / 0.00 |
| queue_raw | 15.6862 | 9917 | 0 | 0.00 / 0.00 |
| queue_cal | 15.6862 | 9917 | 0 | 0.00 / 0.00 |

Computed predicates: `{"coherent_source_models_verified": true, "incoming_flow_10s_beats_baseline": true, "live_calibration_complete": false, "promoted": false, "queue_selection_beats_original": false, "queue_selection_interval_positive": false, "same_side_fill_10s_beats_baseline_both": true, "sequential_portfolio_tested": false}`

Quarter-sampled neutral actions, consumed dates, inherited guard features. Full engine removes same-price reset approximation; observed execution timestamps/queue model remain uncalibrated. First-fill curves are horizon probabilities. No value beyond30s and no new sequential portfolio test. Old direct model uses a different short target and one additional prior training day.

These are overlapping local utility sums, not portfolio P&L. No strategy is promoted. Read the human review before interpreting these diagnostics.
