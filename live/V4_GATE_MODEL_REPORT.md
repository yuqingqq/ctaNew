# V4 Simple Gate Model Comparison

Metric: model-only 24h residual alpha edge in bps. Evaluation is expanding walk-forward periods 2024, 2025H1, 2025H2, 2026. Skipped cycles contribute zero edge.

## Summary

| strategy | eval_cycles | traded_cycles | trade_frac | calendar_edge_bps | active_edge_bps | active_hit_rate | calendar_sh_like | total_edge_bps | random_mean_total | random_p10_total | random_p90_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| btc_no_bull | 5297 | 4030 | 0.761 | 34.843 | 45.797 | 0.524 | 3.445 | 184562.594 | 76980.281 | 40865.559 | 112635.711 |
| bucket_report_card | 5297 | 3123 | 0.590 | 31.468 | 53.374 | 0.528 | 3.347 | 166687.605 | 61008.457 | 19222.619 | 102939.631 |
| btc_bear_only | 5297 | 938 | 0.177 | 19.574 | 110.534 | 0.560 | 3.637 | 103681.260 | 17739.441 | -14186.348 | 51480.810 |
| always | 5297 | 5297 | 1.000 | 19.012 | 19.012 | 0.510 | 1.562 | 100707.655 | 100707.655 | 100707.655 | 100707.655 |
| tree_d2 | 5297 | 3286 | 0.620 | 18.480 | 29.789 | 0.519 | 2.065 | 97887.674 | 61058.361 | 19299.852 | 100151.006 |
| lagged_global_edge_180 | 5297 | 3172 | 0.599 | 5.104 | 8.524 | 0.513 | 0.550 | 27038.008 | 60697.696 | 24690.275 | 102520.036 |
| tree_d3 | 5297 | 3189 | 0.602 | -4.759 | -7.905 | 0.496 | -0.542 | -25210.569 | 62263.397 | 22348.283 | 104463.050 |

## Notes

- `always`: trade every v4 model cycle.
- `btc_no_bull`: hardcoded BTC30 gate; trade bear+side, skip bull.
- `btc_bear_only`: farm only the hardcoded bear macro regime.
- `lagged_global_edge_180`: trade only when the prior closed 180-cycle model edge is positive.
- `bucket_report_card`: trade if the current model-state bucket has positive prior closed edge, falling back to macro regime history.
- `tree_d2/tree_d3`: shallow decision-tree regressors trained on prior periods to predict next-period edge; trade if predicted edge > 0.

Random columns are matched-skip baselines using the same number of traded cycles.
