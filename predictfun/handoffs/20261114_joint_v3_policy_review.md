# Review of JOINT-LADDER v3 and the proposed ORDER-HARM policy

As of **2026-11-14 08:34:52 UTC**. Reviewed v3 `eb735795`, its exposure
control `273bbe8f`, post-hoc readings `87ebe59c`, and policy `041edeee`
(supersedes the unrun policy configuration `73950293`). Source files match
their commits. This is a source and saved-receipt review, with small synthetic
controls; it is not an independent historical replay or PnL recomputation.

The previously demonstrated future-rejection leak is fixed in v3. No further
critical execution-causality defect was confirmed by this review. The main
remaining concerns concern what the next comparison can establish.

## Repair and current execution status

- The actual v3 `sim_joint` passes the earlier two-rung, 30-share-cap example:
  changing only the future activation book leaves the complete t=0 decision
  prefix identical. Both paths reserve the first request and block the second.
  Planting `MUT_REF_RELEASE` restores the old divergence and the checker refuses
  it. Increasing the known cap to 60 shares allows the second request, proving
  the checker is not simply observing a lane that never places it.
- Saved receipts reconcile to 6,912 prefix-invariance checks across three
  consumed dates, 9,835,924 decisions compared, zero differences; 163,296 replay
  runs across nine dates with all seven accounting counters zero; and 15,552
  compat independent-parity checks with zero differences. These controls do
  not prove every upstream feature causal or calibrate the fill model.
- The separate true-exposure check is **7/9 dates complete**, 72,576 runs,
  zero hard-cap breaches; its soft-cap positive control fires on every
  completed date. The original report's `maxexp_hard` was maximum position,
  so the separate check is necessary. It is not yet a completed certification.
- ORDER-HARM policy has source but no saved threshold/result JSON yet. The new
  caps are **300 / 600 / 1,500 shares and uncapped**, with 30 shares per order.
  At a flat position the full six-rung ladder requires 180 shares per side.

All historical dates above are consumed: 2026-10-23..27, 10-31, 11-01..03.

## 1. Cadence confounds incremental model value

`pf_order_harm_policy.oh_dec` decides at the first resting guard-on step, then
on the 250 ms grid, retaining the earlier choice between grid points. FB25
can change its decision every 50 ms. This is explicit policy design, not a
coding mistake, but improvement over FB25 would combine a new model with a
different cancellation cadence.

An actual-function synthetic example supplies the same low score at 50 ms
and high score from 100 ms onward, with unchanged book and thresholds. FB25
pulls at 100 ms; OHH retains its prior keep until 250 ms. A newly observed
order does pull immediately at 100 ms. The cadence checker catches a planted
change to evaluate every step. No historical benefit or cost is assigned to
this example.

Add FB25 with the identical first-rest/grid/episode rules, and a simple
depth/age baseline on that same cadence. Retain the original baseline as
a separate operational comparison. Report actual action counts by side and
time bucket, exposure and order residence, alongside portfolio value.

## 2. The proposed matching and validation remain exploratory

Thresholds are quantiles of predictions pooled across all nine replay dates,
chosen to match FB25's keep fraction on independent reference-order states.
They do not read realized outcomes, but they do use the test-date feature and
prediction distributions. The boosters also use leave-one-day-out training,
which includes later dates when testing earlier ones.

Furthermore, matching a global fraction of overlapping reference states does
not match actual sequential cancellation actions by side/hour as required by
CLAUDE rule 7. Grid-only keep counters omit first-rest off-grid decisions.
They describe a sampling population, not all policy actions. The policy header
acknowledges that reference matching does not imply matching in the joint lane.

Use this run for development. For an improvement claim, train and calibrate
on earlier dates only; freeze the full pipeline, cadence, caps and threshold
rule; evaluate on at least five later untouched complete days. Declare the
matched control/null and multiplicity before those results. Do not describe
the planned nine-date replay as forward validation.

## 3. The success criterion still covers only part of the economics

R630/R631 select on WS-timed, order-age-at-least-100-ms one-second markouts
at M0/M50. All-trades values and M250 are printed but do not affect the
BETTER/PAYS labels. A policy can therefore receive those labels while losing
on the wider population or being highly sensitive to the timing assumption.

This matters in the existing v3 report (as of 07:52:32 UTC, nine consumed
dates): hard10 FB25 is reported at **+100.2/day** on that WS subset but
**-178.5/day** including estimated-time E2 trades, versus LDG **-101.2/day**
on the latter population. At M250, FB25 trails LDG on all nine dates at every
hard cap. These are simulated one-second markout dollars, not realized net
earnings. E2 uncertainty requires sensitivity analysis; it does not justify
silently treating the WS subset as the portfolio.

Keep the diagnostic readings, and predeclare a separate portfolio acceptance
criterion covering all included fills, inventory valuation, cancellation and
re-entry effects, and timing sensitivity. The older settlement-tie conversion
issue remains inherited; this review does not certify the settlement column.

## 4. The label still omits the value of reconsidering

ORDER-HARM v2 labels keeping without a later discretionary cancellation over
two seconds. The candidate policy can reconsider every 250 ms. Thus predicted
KEEP-minus-PULL is not the action value of keeping now and cancelling later
if danger increases. This is already disclosed in the source, and a joint
policy replay is a useful first measurement of its practical importance.

Before scaling model complexity, compare the blind-keep label with a defined
continuation policy under shared inventory and the same cancellation timing.
Keep remaining quantity, client-known order state and queue uncertainty
explicit in that next design. Negative blind-keep labels alone do not reject
adaptive queue preservation.

## Evidence and limits

Instrument: `research/model_joint_v3_policy_review.py`.
Immutable receipts: `reviews/model_joint_v3_policy_review_20261114/`.
The receipts contain source/commit identities, saved-artifact hashes, exact
control counts and the synthetic traces. Prior review receipts remain intact;
the v2 causal-reservation finding is superseded specifically by v3's repair.
No active research source, worker, freeze, or protected forward-day data was
changed or read. The next useful addition is a cadence-matched baseline and
a complete economic decision criterion, before interpreting policy gains as
better predictions.
