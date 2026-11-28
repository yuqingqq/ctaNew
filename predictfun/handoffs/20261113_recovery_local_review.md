# Recovery model: local forecasts and the selective cancellation rule

The primary selective rule makes fewer local cancellation decisions, but its
advantage over the direct-value model is not established. Its extra short-value
condition also retains actions whose total value is predicted negative. These
are reasons to examine the full replay carefully, not evidence of better
quote uptime or portfolio performance.

The [saved local report](../reviews/model_recovery_20261113/local_result.json)
is as of **2026-11-13T07:02:52.217695+00:00**. It covers six consumed outer dates
(Oct26/27/31, Nov1/2/3): **9,977 sampled actions, 9,945 eligible ten-second
targets and 32 explicit missing continuation marks**. Missing counts by date
are 13 / 3 / 0 / 4 / 6 / 6. All comparisons below use those same 9,945 actions.
Each expanding fit uses only earlier core days; the immediately preceding day
is kept for diagnostic calibration predictions. Parameters and 300 rounds
remain pinned. No model is selected or retuned from these results.

## Forecast quality

The baseline is each fold's earlier-core target mean. Values are cents per
action; compare model and baseline within each row because the heads estimate
different targets. These pooled errors are descriptive, without a declared
forecast-error interval in this local report.

| Head / estimate | RMSE (c) | Core-mean RMSE (c) | RMSE reduction |
|---|---:|---:|---:|
| Direct ten-second value | 28.6239 | 28.8674 | 0.84% |
| Short plus remainder | 28.5984 | 28.8674 | 0.93% |
| Short component | 5.2016 | 5.2045 | 0.06% |
| Remainder | 27.8800 | 28.1007 | 0.79% |

The incremental error reduction is small. The positive-value regions also
show calibration problems. For example, the direct model's `[1, 5)` cent bin
contains 708 actions: mean prediction +2.5447c versus observed -3.1855c.
This is a descriptive conditional mean, not a new threshold-selection rule.
Do not compare these RMSE levels with the previous study's different target
and eligible population.

The **short component here is an expanding, order-state-aware value head**.
It is distinct from the **fixed market-only recovery proxy** trained on the
two seed days. That separate proxy forecast audit is still queued behind the
occupied auxiliary slot; this table does not establish its forecast quality.

## Fewer cancellations do not yet mean better timing

At each sampled neutral-path decision, the fixed policies produce:

| Policy | Pulls / 9,945 | Pull fraction | Keeps |
|---|---:|---:|---:|
| DIRECT10 | 8,754 | 88.02% | 1,191 |
| COMPONENTS10 | 8,749 | 87.97% | 1,196 |
| SELECTIVE10 (frozen primary) | 5,816 | 58.48% | 4,129 |

These are sampled action decisions, not continuous quote uptime or counts of
all cancellations in sequential replay. The short-negative condition reduces
pulling mechanically. To assess timing, the report re-ranks each comparator at
the primary's cancellation count within day/side/hour/order-age/inventory-sign
strata. SELECTIVE10's incremental local utility is:

| Matched-count comparator | Delta (c) | 95% UTC-day bootstrap interval (c) |
|---|---:|---:|
| DIRECT10 | -1,229.12 | [-5,676.52, +2,219.76] |
| COMPONENTS10 | -1,394.77 | [-5,715.81, +1,975.38] |
| Chronological guard score | +1,005.09 | [-3,223.41, +6,113.67] |
| Previous HISTORY10 | +743.28 | [-2,342.56, +4,471.94] |
| Previous STATIC10 | +3,298.72 | [+744.39, +6,558.75] |
| Matched random selection | +5,007.18 | [+1,888.80, +9,126.16] |

The report uses the declared 500 matched null draws and 1,000 day-bootstrap
draws over all six dates. The primary's direct-model point comparison is
negative, and its direct/guard intervals include zero. The saved predicates
`primary_beats_direct_at_matched_count` and
`primary_direct_and_guard_intervals_positive` are both false. Positive results
against random selection or the older static score do not clear those harder
comparisons. This is a comparison with the guard's score ranking at matched
counts, not a measured win over its full entry/cancel policy.

## What the retained actions tell us

For the kept populations above, the report sums predicted and observed total
KEEP-minus-PULL utility:

| Policy | Kept n | Predicted total (c) | Observed total (c) |
|---|---:|---:|---:|
| DIRECT10 | 1,191 | +2,785.98 | -2,814.45 |
| COMPONENTS10 | 1,196 | +2,715.74 | -2,996.11 |
| SELECTIVE10 | 4,129 | -16,526.43 | -17,672.05 |

For the direct and component rules, the positive predicted kept totals turn
negative in observed outcomes. For SELECTIVE10, **both totals are negative**:
the policy deliberately keeps some negative-total estimates when the short
estimate is nonnegative. That aggregate negative prediction is therefore not
itself a calibration bug. The extra condition is a policy trade-off layered on
top of the value forecast, and the local evidence has not shown it improves
timing at the same cancellation count.

All these values are overlapping one-intervention continuations on a neutral
reference path, with the declared inventory penalty and recovery rule. They
are **not additive portfolio P&L**. Full replay must still establish retained
presence, reduced churn, marked value and nonincreasing inventory exposure
under primary, conservative and age500 scenarios. SELECTIVE10 remains the
frozen primary; no post-hoc switch to another arm or threshold is made.

## Evidence and remaining checks

All nine recovery data days completed by 07:02:13.217731 UTC: 2,592 windows,
138,743 neutral generations and 15,432 sampled actions; 15,382 eligible
ten-second targets / 50 missing marks, all short targets eligible, plus four
separate neutral `censored_30s` statuses. The all-data checkpoint passed at
**07:08:13.800158 UTC**, reconstructing every captured target and binding 125
artifacts. It checks recovery timing from recorded estimates; it does not
independently recompute every label-branch recovery estimate from market inputs.
The final full replay/portfolio verifier remains required.

The local report SHA-256 is
`dcb2b192f80698494439dd0c78f0632a0b2d41b25a1380687ce5b0e5b2061c98`;
all three data/review copies match. Its six fit-receipt links, model hashes,
saved prediction hashes and core/calibration chronology were inspected.
The unchanged local reporter also reconstructs saved test predictions when it
runs. The scientific source remains `78285fe02d00962abc5f7148e6001a8f39a0ddc6`.
The [all-data checkpoint](../reviews/model_recovery_20261113/all_recovery_data_evidence.json)
SHA is `ea5b41f304b879ddbbc3c1378620f3cdbcf4d1bc661745e6a8717fae81943f6e`.

All four integration scenarios now pass. The [empty-fill accounting repair](20261113_recovery_accounting_repair.md)
proved exact raw replay parity at 07:31:17.944617 UTC, and the eighteen full
scenario-days started at 07:31:18.208066 UTC. The [fixed-proxy forecast audit](20261113_recovery_proxy_audit_review.md)
is also complete; its day-level improvement interval includes zero. This remains
exploratory consumed-history evidence, with no untouched-day validation, live
calibration, model promotion or live change. Full policy performance is pending.
