# First arrival-time entry prediction test

As of 2026-11-16T03:34:33.413746+00:00. Nine consumed data dates, 387,535 neutral placement
generations; five chronological consumed test dates, 203,153 neutral test
placements and 203,114 scored teacher targets. Replay: 1,434
first-minute windows, six October 31 boundary exclusions, all fills retained.
This is development evidence, not untouched or live validation.
Test placement statuses: {'supported': 191442, 'reject_at_activation': 11672, 'missing_teacher_book': 39}.

## Result

This first causal arrival forecast does **not improve the delayed strategy**.
It predicts the frozen teacher more accurately on neutral placement rows,
but its selected entries remain miscalibrated and interact poorly with the
unchanged resting guard. This finding applies to this entry-only recipe;
it does not reject anticipation of the full zero-delay policy.

The causal entry forecast changes pooled marked trading value by
$-132.74 versus the existing entry rule and improves
1/5 test dates. The future-informed entry diagnostic
changes value by $+328.62. Teacher-value-weighted sign disagreement
changes by -23878.64 cents (lower is better).

Computed descriptive predicates: teacher disagreement improves =
True; pooled forecast PnL improves =
False; future-informed entry PnL improves =
True. These are point comparisons, not significance tests.

| Entry rule, all at 58/37 ms | Marked trading value ($) | Cents/share | Filled shares | Positive dates |
|---|---:|---:|---:|---:|
| Existing entry | +159.56 | +0.3312 | 48,173 | 4/5 |
| Future-informed entry diagnostic | +488.18 | +0.7135 | 68,418 | 4/5 |
| Causal arrival predictor | +26.81 | +0.1011 | 26,519 | 2/5 |

| Date | Existing ($) | Future-informed ($) | Forecast ($) | Forecast minus existing ($) |
|---|---:|---:|---:|---:|
| 20261027 | +200.40 | +400.04 | +134.82 | -65.58 |
| 20261031 | +9.39 | +53.20 | -46.39 | -55.78 |
| 20261101 | +11.32 | +110.43 | +43.23 | +31.92 |
| 20261102 | -62.87 | -89.53 | -74.43 | -11.56 |
| 20261103 | +1.32 | +14.03 | -30.43 | -31.75 |

The earlier +$820.92 zero-modeled-delay example also removed delays for
resting-order decisions and notifications. It is a different clock/policy
trajectory. This experiment changes entry only and does not claim to recover
that whole trajectory or to test cancellation anticipation.

## Frozen target and estimator

The teacher is the exact existing v3 full EV head pair for each chronological
fold, verified against the positive-example receipts. No teacher refit or
optimization. At a current decision t, price is fixed to the actual candidate
we could submit now. The target is that teacher's P(fill) x conditional mean
value for a fresh 30-share order at that SAME price at t+58 ms, with zero
modeled feed delay at the target time and activation-time queue features.
It is never the future best price/rung substituted for the current quote.
Post-only rejection at activation has zero execution value and stays in the
population. Missing current or teacher support is a declared status.

Training rows are one per generation actually posted on the neutral reference
path at 58/37 ms, independent of later fill outcomes. No filled-only training.
Each fold uses its teacher's original earlier training dates; its calibration
date is unused. Two separate side models predict target minus current teacher
score from the 83 current causal features plus that current score. LightGBM
mean regression, pinned parameters, 200 rounds; no parameter, feature, margin
or threshold search. Post iff current score plus predicted residual >= 0.

The future-informed diagnostic uses the arrival target directly for every
current candidate in delayed replay, falling back to the incumbent on missing
teacher support. It is nondeployable and is not an optimal-policy upper bound.
The causal arm uses only current inputs. Both keep the incumbent resting EV
guard. All actual placement/cancel/notification, queue and fill clocks remain
unchanged. Future prices or trades are never supplied to the causal arm.

## Prediction versus persistence

The baseline carries the current teacher score forward. A high association
with the future teacher is insufficient; it must improve on this baseline.
The targets below are model scores in cents per order, not realized profit.
Zero rejection targets count as nonnegative under the frozen score convention;
their errors have zero weight in teacher-value-weighted sign disagreement.

| Predictor | MAE (c/order) | MSE | Sign accuracy | Weighted wrong-sign value (c) |
|---|---:|---:|---:|---:|
| persistence | 3.8204 | 102.6597 | 81.63% | 38676.08 |
| forecast | 2.9280 | 50.4260 | 85.33% | 14797.43 |

Post-hoc diagnostic on supported targets whose sign changes from the current
score (no rejection-zero targets in this subset; no strategy selected from it):

| Date | Changing actions | Forecast sign accuracy | Persistence disagreement (c) | Forecast disagreement (c) |
|---|---:|---:|---:|---:|
| 20261027 | 5935 | 55.79% | 7998.29 | 2237.47 |
| 20261031 | 3359 | 61.00% | 5965.69 | 1735.73 |
| 20261101 | 3792 | 54.61% | 5515.28 | 1446.55 |
| 20261102 | 6559 | 63.88% | 9165.50 | 1619.47 |
| 20261103 | 6368 | 66.50% | 10031.33 | 1479.52 |

Post-hoc fixed-count comparator: rank by the existing current score and
retain exactly the forecast's number of posts within each window/side/rung.
This controls selectivity in the neutral action table, without fitting or
changing a trading policy. Values are teacher scores, not realized PnL.

| Date | Posts | Forecast kept value (c) | Matched current kept value (c) | Forecast advantage (c) | Always-wait disagreement (c) |
|---|---:|---:|---:|---:|---:|
| 20261027 | 3234 | -507.55 | -697.50 | +189.95 | 3547.89 |
| 20261031 | 1216 | -810.58 | -928.66 | +118.08 | 1790.20 |
| 20261101 | 1494 | +81.27 | -91.33 | +172.60 | 2374.93 |
| 20261102 | 2336 | -305.35 | -680.17 | +374.82 | 2611.39 |
| 20261103 | 1955 | +202.32 | -494.74 | +697.06 | 3133.13 |

Post-hoc entry/guard trace on the first twelve registry windows of each date
(60 windows total, selected independently of results):

| Arm | Submitted orders | Current fresh score negative at posting | Pulls at first client review / reviews | Same, physical remainder positive |
|---|---:|---:|---:|---:|
| baseline | 7008 | 0 | 3891/6831 | 3885/6819 |
| arrival_oracle | 13070 | 11508 | 4040/7931 | 4033/7914 |
| forecast | 11485 | 8892 | 8694/11096 | 8684/11085 |

The last column distinguishes an order still exposed from one already filled
at the venue but awaiting notification. These sampled traces describe policy
interaction; they do not attribute the full PnL difference or prove that a
different cancellation model would fix it.

Actual submitted-order score audit on the same fixed 60-window sample:

| Arm's posted orders | Scored n | Mean future teacher (c) | Mean forecast (c) | Mean current score (c) | Future teacher negative | Forecast MAE (c) | Current-score MAE (c) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7008 | -0.5423 | -0.3346 | +0.4889 | 57.33% | 0.8137 | 1.1347 |
| arrival_oracle | 13070 | +0.1781 | -5.1120 | -9.9151 | 0.00% | 5.3643 | 10.1678 |
| forecast | 11485 | -0.2582 | +0.2108 | -0.4546 | 75.42% | 0.5360 | 0.6691 |

This exposes selection effects on each arm's own submitted orders; it is not
a representative full-day calibration claim or an independent validation set.

## What the evidence identifies

There is incremental ranking information: at the same post count within each
window/side/rung, forecast-selected neutral actions have higher arrival-teacher
value than current-score selection on all five dates. However, their total
teacher value is negative on three dates, and pooled teacher disagreement is
worse than always waiting. Lower average error and better relative ranking
therefore do not establish a useful post/wait boundary.

On the forecast's own sampled submissions, the mean forecast is +0.2108 cents
while the arrival teacher is -0.2582 cents; 75.42% have a negative arrival
teacher score. Those are teacher scores, not realized fill losses. This is
direct evidence of poor calibration after the policy selects its orders.
The neutral birth population differs from the repeated entry opportunities
the policy creates, but this audit does not isolate distribution shift from
ordinary regression error as the cause.

The sampled guard pulls 8,684/11,085 still-exposed forecast orders at first
review versus 3,885/6,819 for the baseline. Together with the full replay's
245,475 versus 189,968 submissions and 26,519 versus 48,173 filled shares,
this supports an entry/guard consistency problem. Its contribution to PnL
is not causally isolated; weakening the guard could retain harmful fills.
The shortfall is also present in WS-timed value and after attributing away
fills younger than 100 ms, so uncertain E2 anchors alone do not explain it.

The next experiment should repair and test the post/wait boundary on the
complete causal entry-opportunity population and make its value definition
consistent with the existing guard's continuation. Compare against current
score at matched participation, retain the fixed-price arrival target, and
evaluate selected-order calibration plus delayed replay before optimizing
the teacher. No new threshold, recipe or cancellation policy was selected or
tested here; an untouched validation claim still requires later complete days.

## Accounting, timing and limits

Additive attribution of the same 1,434 M50/queue-M replay windows; age is at
fill time. Removing young fills here is an attribution, not a changed-policy
simulation. WS/E2 are source clocks; imputed fills remain in the totals.

| Arm | Age <100ms ($) | Age 100–500ms ($) | Age >=500ms ($) | Excluding <100ms ($) | WS ($) | E2 ($) | Pending-cancel $ / share of volume |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | +129.37 | +37.22 | -7.04 | +30.19 | +28.21 | +131.34 | -136.99 / 23.51% |
| arrival_oracle | +423.33 | +77.15 | -12.30 | +64.85 | +162.56 | +325.62 | -175.42 / 19.28% |
| forecast | +59.27 | -36.61 | +4.15 | -32.46 | -78.29 | +105.10 | -142.35 / 22.10% |

The original frozen first probe uses M50 and queue M only. M0/M250, queue L,
B1a weighting, and the older frozen guard-ladder strategy are not rerun here;
no robustness or acceptance claim follows from these results. Replay volume
and dollars can be inflated relative to real fills.

Every baseline window must equal the previous recorded-median experiment in
PnL, shares, orders, pulls, fills, max position and rung exposure. First-window
cached/uncached trajectories also match. All lane and terminal accounting
checks pass. The artifact audit rederives all test predictions, verifies model
reloads and feature order, all teacher identities, exact fixed-price identity
and the 58 ms target offset. Wrong price, rounded 100 ms timing, chronology,
nonfinite predictions and corrupted baseline controls are refused.

The marginal student inference benchmark on core 8 is
0.1169 ms median /
0.1313 ms p95 for twelve quote scores
across both sides, over 500 measured calls after warmup. It is a CPU batch
benchmark, not measured end-to-end system latency; the replay does not add
this marginal compute time to 58 ms.

The replay value is cash plus inventory at the fixed 60.2 s mark; it is not
whole-day, executable liquidation or live profit. No new fee/reward model.
Neutral birth states may under-cover some model-driven entry times. The
teacher's frozen local keep-value target is still a diagnostic entry rule;
its calibration and omitted queue-continuation value are not repaired here.
Physical queue/timestamp assumptions, imputed trade times, unknown-depth
queue zero and deterministic medians remain inherited limitations. No causal
attribution from changes in volume, no matched-null or inferential claim.
The noncausal entry rule follows the frozen nonnegative-score convention:
zero execution-value rejection targets can still cause attempted orders that
the venue subsequently rejects. Its value is not a clean upper bound on
profit achievable by forecasting valid future placements.

## Provenance

Source b6db135e; v1 freeze 47384cf9 was superseded before a completed dataset,
fit or economic result when a data-builder memory guard stopped the run.
Source 1b94ec80 copies selected rows instead of retaining full matrices;
freeze v2 3ce50516 preserves all targets and model/evaluation choices. One
student recipe, ten side/fold fits, three replay arms, no teacher optimization.
All jobs used the Codex/core8 run_light slot. Incomplete v1 output is retained.

Data/models/windows: data/predict_fun/research/entry_arrival_20261116_v2/.
Evidence: reviews/entry_arrival_20261116/freeze_v2.json,
results_v1/report.json and audit_v1.json. Full data-to-fit-to-replay pipeline:
research/pf_entry_arrival.py and pf_entry_arrival_fit.py; independent artifact
checker: research/entry_arrival_audit.py.
