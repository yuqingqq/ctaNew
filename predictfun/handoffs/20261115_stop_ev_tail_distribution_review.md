# STOP EV: tail losses and the training/testing gap

The replay has material tail losses, and the current scores identify some of
them before the outcome. However, prediction of average value remains too
optimistic outside the fitting sample. Two effects are demonstrated: fitting
sample metrics overstate forward performance, and later fills have worse
value even within coarse matched groups. This does not establish whether all
of that worsening is real market flow: estimated trade timing remains material.

## Scope and evidence

Distribution audit: **137,703 original-model test actions**, Oct 27/31 and
Nov 1–3, as of **2026-11-15T15:05:27Z**. All nine source dates were already
consumed. Original population is retained for comparison with the preceding
cause analysis; certified-estimated sensitivity is reported separately.

Saved-model audit: completed model v2 from commit `3239732e`, **81,630
certified-estimated test actions**, as of **15:05:47Z**. Training populations
contain 52,184 / 68,028 / 80,870 / 95,148 / 109,999 actions, including
5,808 / 7,541 / 8,770 / 9,994 / 11,267 filled actions, across 3–7 training
days. Calibration-day and test identities, labels, certificate flags and saved
head predictions reproduce exactly. Training matrix hashes and training IC
also reproduce. There are no new fits, selected parameters, policies or raw
data reads in this review. These are descriptive diagnostics, not untouched
validation or an independent causal attribution.

Sources `26d6df4f`, superseding source `23eb4c3e`; freezes `c44e92d4` and
`60732c8e`. Evidence is in
[`reviews/stop_ev_distribution_20261115/`](../reviews/stop_ev_distribution_20261115/).
`report_v2.json` supersedes v1's analytic comparator by matching **hour** as
well as day, side and rung. V1 is preserved. No sampled null or significance
claim is made. Synthetic composition, within-group and tail controls pass;
nonfinite scores and three corrupted prediction identities are refused.

## Tail risk is economically important, and partially predictable

Define a tail action by each fold's **training all-action first percentile**
of value, held fixed for its test day. Thresholds are about −106 to −119
cents/action. There are **1,779** such test actions; their test-day frequency
is 1.02–1.63% and they contribute about 49–65% of each day's gross losses.
Within the original 9,727 positive-prediction actions, **78** exceed these loss
thresholds and lose **$167.28**. This is an economic contribution, not a claim
that those actions could all have been identified or cancelled in time.

| Existing risk score, highest-risk 10% per test day | Selected actions | Tail actions captured | Matched expected capture |
|---|---:|---:|---:|
| Original Huber EV | 13,772 | 696 / 1,779 = 39.1% | 12.2% |
| Controlled squared-error EV | 13,772 | 701 / 1,779 = 39.4% | 12.5% |
| Order-only EV | 13,772 | 393 / 1,779 = 22.1% | 12.2% |
| Fill probability | 13,772 | 601 / 1,779 = 33.8% | 13.0% |

The full scores contain information beyond simple order state. They still
miss most tail actions at this selection size. Selecting the highest-risk
share is a retrospective ranking diagnostic: the full test-day score
distribution is used, so this is not a calibrated deployable threshold or
proof of policy profit. Cancellation also sacrifices positive value: the
Huber-selected group includes $386.91 of positive local value alongside
$2,438.81 of gross losses. All 1%, 5% and 10% readings are preserved.

Tail losses should remain in the training target. Removing or treating them
as noise would recreate the optimistic-EV problem. The relevant model quantity
is their conditional expected cost, including both probability and severity,
for each resting order. Selective pulling needs to compare that cost with
retaining the order, including latency and the value of queue priority.

## The test sample has worse fills, not more fills

The following table uses the original populations, with no omitted test
actions. Each row compares that fold's training sample with its own test day.

| Test day | Train / test actions | Train / test fills | Fill rate train → test | Mean shares per fill train → test | Value cents/filled share train → test |
|---|---:|---:|---:|---:|---:|
| Oct 27 | 92,242 / 30,698 | 10,859 / 3,333 | 11.77% → 10.86% | 21.20 → 21.79 | −1.041 → −1.185 |
| Oct 31 | 127,670 / 19,869 | 15,049 / 1,748 | 11.79% → 8.80% | 21.30 → 22.34 | −1.066 → −1.647 |
| Nov 1 | 158,368 / 21,361 | 18,382 / 1,962 | 11.61% → 9.18% | 21.39 → 22.26 | −1.088 → −1.530 |
| Nov 2 | 178,237 / 33,190 | 20,130 / 3,370 | 11.29% → 10.15% | 21.47 → 21.83 | −1.138 → −1.637 |
| Nov 3 | 199,598 / 32,585 | 22,092 / 3,510 | 11.07% → 10.77% | 21.54 → 22.35 | −1.174 → −1.667 |

Lower fill frequency improves the all-action mean relative to training; worse
value conditional on filling more than offsets it in every fold. Filled size
increases only modestly. On Oct 31/Nov 1, the tail's average loss becomes more
severe; on Nov 2/3, tail frequency rises substantially. Oct 27 does not exhibit
the same tail-severity deterioration. There is no single identical mechanism
across every day.

Matching decision-time side, rung, training price tercile and training sigma
tercile covers **all test actions**. Within-group value worsens on every day;
on Oct 27, Nov 2 and Nov 3 the composition change alone would improve value.
Oct 31 excludes 2,539 training actions without common test support; other
folds have no such training exclusions in this comparison.

A separate filled-action comparison matches side, rung, realized fill-source
category and training filled-size tercile. Within-group deterioration accounts
for **76–94%** of the matched mean gap on the last four days (58% on Oct 27).
Matched train/test fills and exclusions are:

| Day | Matched train / test fills | Excluded train / test fills |
|---|---:|---:|
| Oct 27 | 10,399 / 3,153 | 460 / 180 |
| Oct 31 | 14,630 / 1,710 | 419 / 38 |
| Nov 1 | 18,058 / 1,929 | 324 / 33 |
| Nov 2 | 19,858 / 3,298 | 272 / 72 |
| Nov 3 | 21,846 / 3,442 | 246 / 68 |

These are exact descriptive mean decompositions on common cells. Realized
fill size/source are outcome descriptors, never proposed model inputs.
Coarse cells do not establish identical conditional market states. Unmeasured
flow, within-bin conditions and timestamp reconstruction remain possible
explanations. The deterioration also persists on certified-estimated labels;
excluding uncertified actions alone does not remove it.

## Training performance is measured on fitting data

The source computes `T6_ic_train` by predicting `Xtr` used in fitting. It is
resubstitution performance. Replaying the new v2 saved heads gives:

| Test fold | Training IC | Intermediate reserved day (n), IC | Later test day (n), IC |
|---|---:|---|---|
| Oct 27 | 0.213 | Oct 26 (15,844), 0.133 | Oct 27 (12,842), 0.120 |
| Oct 31 | 0.224 | Oct 27 (12,842), 0.133 | Oct 31 (14,278), 0.097 |
| Nov 1 | 0.224 | Oct 31 (14,278), 0.087 | Nov 1 (14,851), 0.081 |
| Nov 2 | 0.217 | Nov 1 (14,851), 0.080 | Nov 2 (20,079), 0.125 |
| Nov 3 | 0.204 | Nov 2 (20,079), 0.125 | Nov 3 (19,580), 0.129 |

The drop is already present on the intermediate day. The variable called
`cal` is a reserved prediction day in this pipeline: the code saves its
predictions but does not fit a value recalibration on it. This is a missing
calibration step, not evidence that a particular recalibrator will solve the
problem. Those days are now consumed and must not become a new test for a
calibrator chosen after seeing these results.

V2 training mean optimism is only 0.047–0.079 cents/action; test optimism is
0.318–1.495 cents/action. The pooled primary readout's highest predicted-value
decile forecasts **+1.60** but realizes **−0.81 cents/action**. Keeping positive
predictions selects **15,681 / 81,630** actions, with **−$117.57** total local
value; four of five test-day selections lose. Reported positive local *gain*
is improvement relative to keeping everything, not positive kept value.

This is consistent with fitting noise in a high-dimensional, sparse-fill
target plus changing outcome distributions. It does not identify the separate
causal shares of overfitting, missing state, market change and label error.
More rows from a few highly related days are not equivalent to independent
coverage. No controlled training-window experiment was conducted here.

## Feature drift exists, but this is not wholesale unseen input ranges

All **83** declared inputs were checked. Depending on the test day, only
**0.062–4.391%** of actions have any finite feature outside its training
minimum/maximum. This rules out the explanation that most test actions have
entirely unseen marginal values; it does not establish joint support or
stability of conditional outcomes.

Volatility, fair-value sensitivity and trading activity change materially.
Fair-value-input missingness rises from about 13.2% to 19.9% on Oct 31 and
13.3% to 18.6% on Nov 1, but improves on Nov 2/3. In source,
`pf_stop_model.inf_feats` masks these inputs when start price is unknown or a
recent finite fair value is unavailable; `pf_influence.pbn_at` uses a 60 ms
tolerance. This audit does not attribute the missingness counts to each
individual reason, establish a new normalization defect, or prove that
missingness causes the losses.

## What follows from these results

Keep the mean-value objective and tails in the data. First resolve whether
estimated trade anchors align with the book updates carrying the execution;
certificate eligibility does not validate physical E2 timestamps. The v2
timing-invariant subset is only 11,574 test actions, has much weaker rank
skill (about 0.016 versus 0.034 for order-only), and is a different selected
population. It cannot establish that all E2-associated losses are artifacts.

Then evaluate conditional tail probability/severity and calibration across
market regimes, with training choices frozen before new untouched days.
Judge the model on forward expected value near the keep/pull boundary and
incremental value over the order-only control. Rank correlation and in-sample
fit alone cannot support model-driven decisions. T5v2 remains pending in the
other session; this review neither runs nor substitutes for it. No additional
blanket cooldown or indiscriminate cancellation is justified by this audit.
