# Review: influence forecasts and preserving resting orders

Reviewed 2026-11-14 around 05:25 UTC against ROOT commit `54408bcd` and the
saved NEXT-PRICE / INFLUENCE artifacts. This is a source and receipt review,
not a refit, independent replay, or forward validation.

The user's objective is correctly reflected in the direction of the work:
estimate the adverse move's extent and continuation, then make a separate
decision for each resting order. Retaining an order preserves its existing
queue position; it does not establish that the order is first in the queue.
The implementation and economic evidence do not yet fully meet that objective.

## Completed progress and its limits

- NEXT-PRICE, as of 2026-11-14 03:32:29 UTC: across nine consumed dates,
  first-minute guard-ladder simulations at M50, 154,740 eligible fallback
  births (15,702 excluded), our order was present at 7.8%; a previous order
  there had been pulled by the guard at 59.9%. This supports investigating
  cancellation-induced absence. It does not show those cancellations were
  economically wrong, nor measure whole-session live quote uptime.
- INFLUENCE, as of 04:56:46 UTC: 269,938 candidate onsets, 333 missing fair
  price, zero invalid books, 269,605 retained. Among the 257,451 onsets with
  no adverse move before the cancellation deadline, the two-second maximum
  adverse move was zero at 50.0% and one tick at 21.0%. A broad signal often
  does not threaten every deeper price.
- On that latter population, FULL's one-tick AUC was 0.701 versus STEP's
  0.697. On the 267,979 eligible two-tick-target onsets, matching the rung-1
  keep count (81.4%) reduced the reported share subsequently passed through
  from 24.1% to 24.0%. These are small historical gains. New influence inputs
  add less than 0.001 AUC over NOINF for thresholds 1–3.
- The report's pooled onset calibration deciles look close. That does not
  establish calibration at subsequent decision steps or on retained orders.

Dates: 2026-10-23..27, 10-31, 11-01..03. These were already consumed.
Leave-one-day-out fits include later dates when scoring earlier ones; this
is exploratory held-out-day evidence, not chronological forward validation.

## What the code already does correctly

`pf_future_best.py:81` maps the forecast to the order's actual current
distance from the known best. For distance d, it compares the probability
of a move of at least d+1 ticks with a policy threshold. Consequently a
bid at 49c can stay when the current best is 50c and the move stops at 49c.
The code also allows an order originally placed deeper to remain when it
becomes the best, provided predicted continuation risk is low.

`pf_influence.py:115` measures the path at/after the assumed cancellation
deadline, through two seconds; it also records one/three-second peaks,
the three-second endpoint and time to peak. The fitted heads predict only
the two-second exceedance probabilities; endpoint and timing are descriptive.

## Material gaps to resolve

1. **The best-price exception preserves blanket pulling.** In
   `pf_future_best.py:90`, an order originally posted at the best is always
   pulled on a guard signal while still at the best, regardless of its
   forecast. Missing forecasts also pull. This is a deliberately retained
   policy exception, not yet the user's general order-specific rule. Keep
   data-health handling explicit and separately count its cancellations.
2. **The fitted and scored populations differ.** `pf_influence.py:282` fits
   on signal onsets, whereas `pf_target_test.py:155` scores every guard-on
   step. Agreement of recomputed onset predictions cannot validate the
   later continuation decisions. Train/evaluate the residual move from the
   actual decision state, or preserve the onset forecast's original horizon
   and explicitly update it with elapsed time and realized movement.
3. **The target still needs causal action alignment.** Training each head
   filters on D0 < j, a future outcome before cancel effectiveness, then
   predictions are used without that future filter. Also, TARGET-TEST's
   oracle uses `pf_pull_depth.move_ticks`, which includes moves between the
   known frame and cancel effectiveness; this differs from the clean target
   when a move recovers before cancellation. Define labels from the native
   effective deadline, retain every neutral action/status, and use the same
   label definition in oracle and learned-policy comparisons.
4. **Price movement is a proxy for order harm.** A fill can lose value even
   without the best moving through its price; a best-price move need not
   fill our order. The policy ultimately needs expected avoidable fill loss
   versus the value of KEEP, including remaining quantity, inventory and
   the queue/re-entry cost of PULL. Predict depth probabilities as inputs
   to that comparison, rather than treating a mean move as a safe boundary.
5. **The multi-level economic test remains incomplete.** TARGET-TEST runs
   each 30-share rung independently (`pf_target_test.py:256`), rather than
   a joint ladder sharing inventory and trade consumption. Native sequential
   replay with a shared risk limit is still needed. Its old queue model also
   retains P18: unknown depth beyond the shown book can initialize at zero
   queue ahead. Do not promote aggregate ladder profit from this harness.

The report's statement that the other 97% of signals are "Predict-local"
is stronger than the evidence: absence of a fair-price move above 0.5c does
not identify its cause. Likewise, observed pass-through is not an isolated
causal influence estimate because later fair-price changes are not removed.
Conditional future-path prediction is sufficient for the proposed policy;
causal attribution to an individual signal is a separate question.

## Current execution and recommended next comparison

At 05:22:35 UTC TARGET-TEST had scored all nine dates, with zero onset
prediction mismatches, and saved one day evaluation. No final report existed.
It compares old/new forecasts, a simple influence rule, and horizon oracles.
It is an exploratory diagnostic; an oracle winner alone does not select the
most learnable or economically best target for a future policy.

Prioritize a decision-time distribution of remaining adverse depth and timing,
mapped to each existing order. Compare broad guard cancellation with selective
cancellation in the same sequential engine and shared inventory budget. Report
preserved order age/position, presence when a level becomes best, good fills
retained, avoidable loss, cancel/re-entry cost and inventory risk. Keep opposite
side additions separate from deciding whether an existing order should survive.

The separately frozen action-depth experiment already uses neutral size-study
actions, native cancel deadlines and earlier-date training. It remains queued
behind size evidence and is a forecast screen, not a completed selective
multi-level policy. Its freeze is unchanged. Existing background workers were
left running; the conversation goal service currently reports `usageLimited`.

## Verification

The six named NEXT-PRICE / INFLUENCE text/JSON artifacts match their stored
SHA256SUMS and their runtime copies. Nine INFLUENCE row receipts reconcile
the population totals above. Sources were inspected at their actual label,
fit, score and decision functions. See the accompanying review receipt for
hashes and execution status. Saved performance statistics were checked against
their named report artifacts, not independently recomputed from prediction
parquets. No frozen scientific source, trading setting or worker was changed.

Documentation checks: the new status block parses in both copies, and ROOT's
complete status parsed on the final check. The isolated worktree's complete
status has a pre-existing YAML error at line 9, also present in the parent
commit; this review did not modify that other section. Whitespace checks pass.
