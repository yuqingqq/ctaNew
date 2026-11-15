# Entry loader coverage history and size study

The issue inventory is committed at 7a4aa584 and integrated at 58159b5b.
This implementation follows the accepted sequence: loader and label validity,
representative coverage, controlled training history, then model changes.
The user additionally requested larger order sizes. This study adds 50 and
100 shares alongside the existing 30-share control.

## Loader and label checks

The new loader opts into file filtering before any book-part read and has a
second read-time tripwire. Only the seven compatible consumed label dates
are admitted: October 25, 26, 27, 31 and November 1, 2, 3. The first six windows
of October 31 remain excluded because their upstream lookback crosses the
sealed block. Every exclusion remains in receipts or row statuses.

For each day, compare the saved clock shift with a median computed from
admissible parts only. Do not substitute that number into old features or
labels: their clocks must agree. The saved shift is a retrospective daily
calibration. Neither this check nor downstream future perturbations establish
causality of the complete raw-to-step5 pipeline. This remains a declared
qualification of the controlled development comparison.

Recompute the first saved eligible action per side per market window and
require identical features, probe timestamps, terminal labels and decomposed
order/portfolio targets. Dense builds subsequently require exact parity for
EVERY original eligible action. The label continues to mean POST versus WAIT
under neutral KEEP and the frozen bootstrap entry continuation. The separate
selective cancellation model is not integrated or implicitly credited.

## Larger size and queue sensitivity

Run 30, 50 and 100 shares on the same admissible windows, each with its own
neutral POST/KEEP path. Keep the hard cap fixed at 1,500 shares, six rungs,
50 ms decisions and the existing latency assumptions. Size changes can change
action births, so generation IDs are never treated as paired actions across
sizes. Each actual fill is recomputed; no linear P&L scaling.

Run both the original unknown-queue-zero assumption and an unknown-queue-block
sensitivity. In the latter, an unobserved initial queue stays infinite until a
visible frame supplies a bound; a trade through the price can still fill.
The existing next-frame activation correction is preserved. These paths are
sensitivity scenarios, not measured queues or mathematical P&L bounds.

Report all orders, accepted and refused orders, filled orders, repeated fills,
exhaustion, aggregate resting order time, inventory, hard-cap blocks and gross
terminal P&L. Repeated fills divided by all accepted orders is the primary
retention description; the filled-order denominator is secondary and explicit.
No inference about superior entry-model accuracy follows from neutral size
results. Size-specific WAIT/POST labels and models must be rebuilt separately
before recommending a size for the learned policy. Own market impact remains
unmodelled.

## Representative sampling

At 30 shares, initial generations retain inclusion probability 1/2. Later
ones increase from 1/32 to 1/4 using the same stable hash. The old subset is
nested in the new subset. Requested probe age ALWAYS uses the original modulus
(2 or 32), including previously unsampled actions. Each row carries its version,
probability, inverse-probability weight, old inclusion flag, reference and
continuation identifiers, original age and status. No outcome filtering,
clipping, duplication or fill-selected training.

New training weights are 2/4. The sparse control and the unchanged old test
actions retain weights 2/32. The weight validator refuses undeclared versions,
probabilities and incompatible weights. Denser rows do not manufacture new
calendar regimes, joint waiting states or observations missing from the feed.

## Controlled history before model changes

Use the same 351-input CONDITION representation and pinned LightGBM parameters
and rounds. Separate bid/ask and absolute POST/timing heads remain unchanged.
Three predeclared arms on each of five original evaluation dates:

- SPARSE_ALL: original selected training actions, all earlier compatible days.
  Saved predictions must reproduce the original CONDITION model.
- DENSE_ALL: denser training actions, all earlier compatible days.
- DENSE_RECENT2: denser training actions, only the latest two compatible days.

DENSE_ALL versus SPARSE_ALL isolates sampling density and its correct weights.
DENSE_ALL versus DENSE_RECENT2 compares history on identical test actions. The
first fold has only two available dates, so those two dense arms are identical
there. A longer calendar history can also change regimes; no claim that it
creates proportionately independent information is allowed.

Report weighted MSE/R2, prior-mean and training price/rung baselines, counts,
weight concentration ESS, per-day results and selected action value. Local
null: 500 matched draws, preserving day, side, hour, rung, initial/later status,
action count and original evaluation weight mass; Bonferroni across six
arm/side checks. Two new timing candidates plus one existing control; six
size/queue scenarios, descriptive. No intervals or fresh validation claim.
Do not change the learner until this comparison has been reviewed and a
separate experiment frozen.

## Execution and reproducibility

Source and all admissible input identities are committed before historical
execution. The serial runner verifies them before and after each stage,
retains failed attempts, checks saved model reloads, and runs through
run_light.sh in the Codex slot with a 2 GiB RSS guard. A two-window smoke must
pass before the full run. A recorded older supervisor may be held at a stage
boundary; its current worker is never killed, and it is resumed afterward.

The new functions and lane are separate from prior frozen studies. Old
receipts, source files and model artifacts are preserved. Reports explicitly
retain open clock, queue, label-policy, coverage and impact qualifications;
none of these development results authorizes live trading or promotion.
