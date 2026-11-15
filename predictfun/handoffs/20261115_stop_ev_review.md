# Stop EV design review

STOP-EV has a better economic objective than cancelling on sweep probability
alone: keep an order when the fills available at its current queue position
have positive expected value. The current draft is not ready to implement
literally. Its quantity accounting and relationship between price movement,
execution and markout need correction first. It remains design only.

This is a review of `20261115_stop_ev_design.md` at commit `ff4f7d1b`, the
completed 30-share/unknown-queue-zero STOP-PLAY v6 results copied at `9851287a`,
and the relevant label/replay implementations. No model or running source was
changed. Source identities are in
`../reviews/stop_ev_review_20261115/evidence.json`.

## Latest completed result

As of **2026-11-15T11:24:12Z**, STOP-PLAY contains five consumed chronological
test dates: October 27/31 and November 1/2/3. There are 1,434 included windows
(288/282/288/288/288), six boundary windows excluded, no missing prediction
windows and zero recorded engine or off-parity violations. The copied result
matches the data artifact; each day's values, availability, statistics and
calibration choices reconcile exactly to its underlying day receipt.

The main M50 scenario, 30 shares per rung, has these reported mean daily gross
terminal values after excluding fills on orders younger than 100 ms:

| Policy | Mean daily replay value |
|---|---:|
| Always keep | -$5,873.3 |
| Existing guard | -$173.2 |
| Learned one-second sweep-risk policy | -$134.6 |
| Policy using the true one-second price-move label | +$794.7 |

The learned one-second policy loses on every test date. Its innermost rung
has approximately 9% availability, versus 58% for the guard and 64% for the
true-label policy. Its outer rungs remain much more available; 9% is not the
whole ladder's live fraction. It also gates entry, so this is a combined
cancel/re-entry result, not an isolated test of cancellation. On WS-timed
aged fills it remains negative at -$24.8/day; the loss is not wholly confined
to estimated-time trades. It does not pass the declared across-day test for
beating the guard. At M250 its aged-fill value is -$1,305.7/day.

These are consumed-data simulator values, with no inventory cap in this lane,
estimated queues, inflated replay volume and estimated-time trade uncertainty.
They are not live earning estimates and are not directly comparable with the
separate capped local-entry replay. The true-label policy is a comparator,
not a mathematical upper bound on all policies; shorter-horizon learned arms
already exceed that comparator in some settings.

## Corrections needed before fitting

1. **Conserve the same order's remaining quantity.** The draft's equation at
   line 32 adds stopping-fill gains and subtracts `P(sweep) * Z * v_sweep`.
   On a path where stopping trades fill part of the order first, a later sweep
   can execute only what remains. For example, a hypothetical 30-share order
   filled for 20 shares cannot subsequently lose on another 30 shares.
   Advance one chronological fill ledger, including fills before a cancel can
   land, and cap total quantity by the initial remainder. Both value terms
   must come from that same ledger. The current replay already caps each
   execution by remaining order quantity and shared taker volume
   (`pf_stop_model.py:688`).

2. **Predict fill-weighted value jointly.** `E[F | x] * E[v | x]` generally
   differs from `E[F*v | x]`. High-volume stopping trades can have different
   markouts from small ones, and that dependence can change the sign of the
   decision. A conditional, share-weighted markout can support a valid
   factorization, but the draft's separate state-average VALUES table does
   not establish that property, especially after clipping by q and Z. Prefer
   direct unconditional signed-value labels, or additive stopping/through
   value heads that each train on all actions with zeros when unfilled.
   State the sign convention explicitly: a signed sweep markout is added;
   a sweep loss magnitude is subtracted. Allow profitable reversals rather
   than assuming every through fill loses.

3. **Do not equate a best-price move with a full execution.** The existing v6
   label is computed from best-book frames (`pf_stop_model.py:237`), not
   executed trade volume. A level can disappear after cancellations. Even a
   through trade in the current engine fills only `min(trade volume,
   remaining size)` (`pf_stop_model.py:690`). The existing predictor remains
   a useful movement-risk feature; it is not an identity for full-order
   sweep probability. Distinguish trade-through, price disappearance and
   actual modeled execution in the label ledger, preserving shared-volume
   constraints across the ladder.

4. **Make queue and remaining size part of the target contract.** The draft
   names q as an extra input but uses varying Z in the label and proposes
   both 30/100 shares. Remaining Z must be an input, or every model/size must
   have separately declared support. Reduce q and Z chronologically through
   the cancellation-latency prefix. Current inputs must use only received
   information, not the simulator's true `ahead`, `rem`, delayed trade prints
   or unreceived fills. Public trades can support hypothetical labels, but
   execution still depends on queue evolution, cancellation assumptions,
   shared volume and passive-order impact assumptions. Calling STOP-FLOW
   simulator-free is too strong. The claimed conservative upper bound on q
   also requires proof under the actual timestamp and update rules; the
   rule-M estimate is not a verified queue observation.

5. **Value keeping relative to cancelling with the same executable prefix.**
   Fills before a cancellation can arrive happen under both actions. They
   reduce remaining quantity but must not be counted as avoidable losses.
   If the local target excludes replacements after cancellation, describe it
   as a bounded keep-value surrogate. Retaining a favorable queue position
   is a benefit of keeping; the draft's optional minus queue-option term
   needs a defined sign and must not charge that benefit as a keeping cost.
   Use an explicit keep-on-tie or economically justified hysteresis rule.
   This is different from a fixed retry timer. Replacement and later queue
   option value can remain full-replay questions for the first version.

6. **Fix the decision metric and action population.** The harm-capture
   diagnostic samples overlapping one-second values every 250 ms
   (`pf_stop_model_review.py:975`). Its 3,062,637 M50 rows are order states,
   not independent actions, and its artifact lacks an internal as-of field.
   At the guard's pull share, 71.8% harm avoided and 75.4% good value lost
   have different denominators. Those percentages alone neither establish
   negative economics nor prove that the model merely detects activity.
   Test net signed value on neutral reference actions, matched on side,
   hour, rung, action count and exposure, with at least 200 declared null
   draws. Component skill also needs a baseline conditional on known
   price/edge, queue and size; a per-level base rate is insufficient.

## Simpler target to implement

For a resting order with known state x, queue estimate q and remaining size Z,
predict the expected sum of its signed fill values after a cancellation could
land, over a declared short exposure horizon. Each tranche uses its own fill
time plus the markout horizon:

    Y = sum_i quantity_i * side_sign * (mid(fill_time_i + mark_horizon) - price)

Use one chronological, quantity-conserving order ledger. Include zero outcomes
for unfilled actions and explicit validity statuses. Advance unavoidable prefix
fills before evaluating this sum. No later replacements enter this label.
This is potential local keep value with a declared hold-until-horizon
continuation, not optimal multi-step keep-minus-cancel value. Re-evaluation
every 50 ms and separate future cancellations belong to policy replay.

The direct head is the first candidate; stopping/through component values can
serve as diagnostics or a separately frozen additive candidate. This avoids
requiring separate fill, price-move and markout models to all be calibrated
and mutually consistent before a useful keep score can be produced. It is
closely aligned with the compact local-entry experiment, whose direct head
was the better prediction candidate, but that does not establish the same
outcome for resting-order states.

Use the actual resting queue for keep/cancel. Entry evaluates a new order at
the back with placement latency. A negative value for a new back-of-queue
order must not automatically cancel an older front-of-queue order. Keep the
entry and resting-order policies separate while harmonizing units, horizon,
latency semantics and the state schema before integration.

## First verification cases and evidence limits

Before fitting, verify partial stop then through, complete fill before a later
move, a price disappearance with no trade, a profitable through reversal,
latency-prefix fills, two orders sharing one taker trade, missing/E2 timing
and future perturbations of q/Z inputs. Each checker needs a planted error.
With missing E2 executions, an apparently unfilled WS-only window must not
silently become a zero-fill label; propagate incomplete support to action
validity and report the changed population.

The current shifted-window control does not preserve realized action counts
and uses one shift, so its INFORMATION reading is descriptive rather than the
matched economic null required for the next candidate. Top-maker behavior is
consistent with an economic keep rule, but their internal model is unobserved.
The proposed target is better specified economically; its predictability and
net advantage still require measurement. Conservative-queue and 100-share
STOP-PLAY results were not complete at this review snapshot.
