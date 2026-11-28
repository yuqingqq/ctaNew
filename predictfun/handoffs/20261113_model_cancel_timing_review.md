# Entry exposure before the value policy can cancel

Review as of 2026-11-13T01:58:57.077683+00:00; the accounting receipt is as of
2026-11-13T01:50:14.984899+00:00. Population: six consumed complete dates (Oct26/27/31,
Nov1/2/3), 1,728 windows per policy/scenario. This is a descriptive supplement
to the completed primary/stress study, not a new model test.

The chronological guard can withhold a quote before placement. STATIC10 and
HISTORY10 first become eligible to score an accepted order at 100 ms of age,
and then need the configured cancellation delay. Their earliest possible learned
cancel effect is therefore 122 ms in X and 200 ms under stress. Actual scoring
and cancellation can occur later. Base price, data and inventory cancellations
remain separate mechanisms. The code reference is
[`due()` and the policy callbacks](../research/pf_model_eval_policy.py).

The table shows HISTORY10's realized average one-/five-second fill-mark value
in USD. A fill needs both strict marks for the average; missing observations
remain counted. These are per-fill marks, not portfolio marked value or
settlement P&L. All ages and all three disjoint bands remain in the receipt.

| Scenario | Actual order age at fill | Fills | Both marks known / missing | Net fill-mark value, USD |
|---|---|---:|---:|---:|
| X | All ages | 13,168 | 13,155 / 13 | -724.29 |
| X | <100 ms | 8,039 | 8,032 / 7 | -525.49 |
| X | 100–<122 ms | 1,559 | 1,558 / 1 | -102.23 |
| X | ≥122 ms | 3,570 | 3,565 / 5 | -96.57 |
| stress | All ages | 19,302 | 19,284 / 18 | -1,395.86 |
| stress | <100 ms | 8,400 | 8,386 / 14 | -701.60 |
| stress | 100–<200 ms | 6,777 | 6,773 / 4 | -456.44 |
| stress | ≥200 ms | 4,125 | 4,125 / 0 | -237.82 |

Both bands before the earliest possible learned cancel effect have negative
net fill-mark value in each scenario. The review checkpoint computes this
statement directly from the table. A later cancellation cannot alter those
already executed fills of the same generation. It can change future exposure
through replacement and re-entry, so these figures are not an estimate of
unavoidable strategy losses or the benefit of an alternative entry policy.
The populations are conditional on fills produced by each existing policy;
none of these fill subsets is used as a model-training population.

This identifies an execution-policy issue alongside prediction quality: entry
and re-entry expose the strategy before the post-placement value head can act.
The queued recovery experiment is relevant because it changes re-entry after
an explicit pull. Its recovery estimate remains a neutral resting-order proxy;
it does not identify fresh-order entry value, and initial quotes, replenishment
after fills and other replacements still follow the base policy. Full replay
must show whether its changed future exposure actually improves value, risk
and participation. This accounting does not justify changing its frozen
thresholds, declaring it unsuccessful before running it, or claiming that an
entry gate or a wider ladder would improve performance.

The descriptive instrument was committed before reading this breakdown at
`31ae25eef29351b6c1b643c02373ccc6bd643257`. Its positive boundary/missing-value
controls and nine known-bad refusals pass. Every band reconciles to its total;
the one-/five-second totals also reconcile to the existing per-fill audit.
All 60 day/scenario/policy artifact identities and run receipts were checked.
There is no null test, confidence interval, new candidate or policy selection
in this descriptive supplement. The original primary/stress failure and the
frozen next experiment are unchanged. No live change.

Sources: [declaration](../reviews/model_upgrade_20261112_v2/cancel_frontier_declaration.json),
[controls](../reviews/model_upgrade_20261112_v2/cancel_frontier_controls.json),
[complete accounting](../reviews/model_upgrade_20261112_v2/cancel_frontier_detail.json),
[artifact review](../reviews/model_upgrade_20261112_v2/cancel_frontier_review_checkpoint.json).
