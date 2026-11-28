# Pull-to-depth evidence and implications for the model goal

Reviewed 2026-11-13; producer report as of **2026-11-13T12:16:21Z**. This is a
source/artifact review with synthetic target and dependency controls, not an
independent historical replay. The study uses nine consumed dates:
Oct 23/24/25/26/27/31 and Nov 1/2/3, first 60 seconds. Its reported forecasting
population is 2,290,802 guard-on grid steps with a valid known book, including
275,122 onsets.

The [review receipt](../reviews/model_recovery_20261113/pull_depth_integration_source_review.json)
binds 200 artifacts: report/log mirrors, all daily outputs and 45 depth models,
nine upstream AV models, upstream fit receipt, source files and producer handoff.
SHA `2eb6dd37993924e5496a18eac5535d26d93e493a25fc9046785afe41962202b6`;
review as of 12:26:24.583002Z. The instrument is committed at 1645133b.
The bounded source/hash/synthetic run used free auxiliary core 9, 38 MiB peak,
0.51s wall. No historical predictions, tables or model fits were recomputed.

The primary producer source is commit 7f5f6aa1. Current source equals the
post-hoc commit 0fb1fa22; the target, rows, fit, evaluation and report function
ASTs are identical between those commits. The upstream FRONT-PULL source
equals df320ba3. These exact identities, rather than the handoff's wording,
anchor the review.

## Useful finding, with a narrower conclusion

The reported results support investigating **price-dependent withdrawal**:
keeping some deeper orders can be useful even when the best-priced order
should be pulled. The learned variants improve the producer's selected
websocket-timed, aged-fill mark sums. Fixed-depth retention performs poorly.
That suggests the geometry and timing of the expected move deserve attention.

It does not establish a profitable or risk-matched multilayer controller.
The primary HELPS predicate uses only websocket-timed fills on orders at least
100ms old. In the published M50 all-timing table, the lower-threshold variant
improves the mark sum but remains negative, while the higher-threshold variant
is worse than the pull-all baseline. These are 1-second fill-mark sums, not portfolio
P&L. The post-hoc all-age split is descriptive, not a sequential age-policy test.

## The forecast target differs from avoidable future harm

`move_ticks` starts after the **last known frame**, not after the decision or
effective cancellation. The source-bound synthetic control reproduces this:

- Known frame: bid 50 at 0 ms.
- Actual book: bid 47 at 90 ms, recovered to 50 at 99 ms.
- Decision: 100 ms; hypothetical cancellation effect: 124 ms; later book flat at 50.
- Producer target: 3 adverse ticks. Observed depth after the cancellation
  effect: 0 ticks.

This is consistent with the producer's explicitly declared known-frame
estimand. It is a mix of delayed-book nowcasting and forecasting. Its AUC
cannot be interpreted directly as skill at identifying losses a new
cancellation can still prevent.

The function also returns a finite zero when the known frame is valid but every
later frame in the horizon is invalid. Such rows pass the fit's finite-target
filter as negatives. The reproducer distinguishes this from an observed flat
book and from an invalid known frame, which returns missing. The number of
historical rows affected is **unknown**; no corrected performance is claimed.
For our next target, unavailable future-book coverage needs an explicit status.

Consequently, “no adverse best-price move” does not by itself establish that a
guard episode was a false alarm. The guard's intended economic harm, its
intervention deadline and observable target coverage must be defined first.

## The outer day is not excluded throughout the model chain

The downstream fit excludes rows whose `day` equals its held-out day. However,
`rows_day(x)` attaches the upstream score from `av_x`, and `fp_fit` trains that
upstream model on every date except x. For an outer test day of Oct 26, training
rows from Oct 23 therefore use an `av_20261023` model trained with Oct 26 data.
The downstream row exclusion does not remove that dependency.

The source-derived dependency control finds this path for every outer fold.
A synthetic label-mutation witness changes training-side upstream predictions
while leaving the outer test-side upstream prediction unchanged. Excluding
both the outer day and each upstream row's fold removes that path in the
control. This establishes the dependency structure, not the magnitude or
direction of its effect on the historical LightGBM results. FRONT-PULL also
documents an inherited guard-model dependency on its held-out dates.

The experiment is additionally leave-one-day-out, with later dates in training,
rather than chronological. Its reported R490 metric is the arithmetic mean of
daily AUCs; it is not a pooled AUC. The header's use of “pooled” should not be
carried into interpretations. None of these results are untouched validation.

## Portfolio and maker comparisons remain limited

- Each rung calls `K.sim_arm` separately with a fresh taker quantity budget.
  There is no joint ladder fill allocation or shared inventory reservation
  budget. Summing those fills does not prove a feasible portfolio advantage.
- Grid steps are a market-forecast population, not neutral cancellable
  generations. Many steps can belong to one exposure or share the same
  subsequent move. The study does not isolate prediction value with controls
  matched on actual withdrawal decisions and inventory risk.
- Missing 1-second marks are removed from reporting tables without corresponding
  removal counts. The headline subset also excludes estimated-time and young
  fills that still affected the simulated path.
- R493 compares counts of first-fill records at adverse-side events with
  share volume across all at-best events. These use different units and
  populations. First observed fill order also depends on ordering within a
  taker; it does not reveal all resting orders or exact queue rank.
- Earlier-fill marks and proof of an earlier fill are descriptive observations
  among fills. They cannot establish top makers' full quote availability,
  cancellation rules, simultaneous layers or risk-adjusted returns.

## What to carry into the active goal

Keep the current size experiment unchanged. Its neutral-action capture,
actual cancellation clocks, notified quantities, complete sequential accounting
and earlier-only model dependencies address different requirements from this
study. The review does not select a new depth threshold or promote either arm.

A later depth-head comparison should retain the useful price-level hypothesis
while using:

1. Future targets beginning at the actual intervention deadline, with coverage
   exclusions and a baseline that separates known movement from new movement.
2. Earlier-only training for the entire dependency chain; inner cross-fitting
   or omission of upstream learned scores when constructing training inputs.
3. Neutral action rows, matched withdrawal-count controls and full sequential
   value at controlled realized inventory risk, with joint ladder reservations
   and a shared taker budget before making any multilayer claim.

The largest unresolved question is whether available signals predict enough
**avoidable economic harm** to justify surrendering the value of a resting
order. This study does not close that question, although it supplies a useful
candidate forecasting task. Recovery remains 8/18 complete; Oct 31/stress and the
verified queued size driver continue. No live change or improved-model claim.
