# Longer-horizon economic value — governing USER direction

2026-11-22. Parent **90e82f8eea0505321f422e0ebda011fd973ae5da**.
USER: “Don't just check 1s or short latency fills ... We need ... longer
horizon economic value projections, don't just use the first fill value
to decide.”

This governs the next target/controller design. It supersedes any use of
the first-fill/+1 s diagnostic as the action objective or as grounds to
reject queue continuity. Earlier frozen measurements remain valid for their
declared estimands. No new model, replay or empirical result is produced here.

## Economic question

Evaluate **the remaining order and strategy continuation**, including multiple
partial fills, preserved priority, later replacement orders and inventory
effects. Early adverse fills can be acceptable if the incremental later
benefit of retaining priority more than pays for them relative to a feasible
avoid-and-re-enter policy. Assess the full entry history when choosing when
to join the queue; assess future incremental consequences at later decisions.

At the same decision state, compare:

- absent: POST with the declared resting continuation versus WAIT with its
  declared causal future-entry rule;
- resting: KEEP that generation/priority versus PULL with its declared causal
  re-entry/continuation rule and actual replacement queue.

Both alternatives must be valued to the **same absolute terminal time T**
under the same declared continuation and risk framework. Do not value POST
over a long life and PULL/KEEP over an unrelated 50 ms or first-fill target.
Do not give a repost the retained order's later fills or priority.

## Target and accounting

For action a at t, forecast the full ledger increment:

    Q_a(s_t; T) = E[(net_cash_T^a + inventory_T^a * mark_T)
                   - (net_cash_t + inventory_t * mark_t)
                   | information available at t, action a, continuation]

Use one consistent fee/cost convention: net cash already includes any
declared cash execution costs; never subtract them a second time. Both
branches share the initial fork and terminal mark. Economic labels include
both sides and all subsequent orders affected by the intervention. Per-gid
components explain the result; they must reconcile with the total ledger.
This is marked economic value, not necessarily settlement profit.

Queue value enters through the difference in future fills/prices/continuation
between branches. Do not add a separate arbitrary queue bonus on top of
future fill profit. If risk penalties are later proposed, declare their
units/coefficients and show them separately from P&L; do not silently invent
one to change the label. Preserve the currently pinned inventory limits and
management mechanism in the first comparison.

The initial common target horizon is the **remaining supported episode to
its predeclared END**, using the same END in paired arms. It must be fixed by
data/market support before inspecting results, not selected to make an early
loss recover. This uses the existing long episode evidence where available;
it does not promise every observation has equal time remaining. Record time
to END and censoring explicitly and include time remaining in any later
prediction inputs. Do not terminate valuation at the first fill or at the
first pull; follow the resulting position and re-entry to the common end.

Fixed 5/30/60-second horizons, only where supported, can show how value
accumulates. Use declared common cohorts for horizon comparisons. They are
diagnostics/sensitivities, not a search for the most flattering horizon.
One-second markouts remain useful for locating adverse selection, not for
deciding that an otherwise valuable long-lived order must be cancelled.

## Continuation and controller consistency

Before building labels, freeze the exact resting/entry/re-entry policy used
after each action, including structural/inventory constraints, partial-fill
behavior and termination causes. The generation must retain priority through
partial fills. Do not recycle a target that forces a 50 ms deviation and
then hands the order back to an incompatible policy that immediately pulls
it; that does not test the USER's queue-continuity hypothesis.

The controller may update rapidly as information arrives, but its values
must cover the same longer continuation. A negative predicted next-fill
markout alone is not a PULL criterion. Compare Q_KEEP with Q_PULL/re-entry,
including the future good fills sacrificed, actual cancel exposure and
changed inventory path. Serious new risk or a material change can still
justify immediate cancellation. A longer target is not a mandatory holding
timer, cooldown, or commitment to ignore changed conditions.

Longer horizon alone does not guarantee less churn. Log post-to-pull
reversals with the changed inputs, predicted future fill/value components,
known queue/remaining quantity, inventory and action-value difference. That
distinguishes genuine deterioration from inconsistent valuation or inaccurate
predictions. Do not claim churn is solved merely because a horizon increased.

## What the next combined deliverable must establish

1. A precise paired-action/continuation contract, common END, cost convention,
   causal feature clock, sampling manifest and bounded pilot projection.
2. Engine-reconciled full-ledger labels, all early and later partial fills,
   actual queue reset on re-entry, and inventory/other-side effects. Include
   admission choices when the early loss precedes any feasible cancellation.
3. Keep the user's categories conditional on this longer economic comparison:
   compensated queue continuity; avoidable early loss; uncompensated exposure;
   incomplete/unidentified. Existing short suffix classes cannot decide these.
4. Required economics: paired total value per opportunity and per day; early
   cost versus incremental later benefit; fill/share/markout decomposition;
   retained/lost priority and re-entry waits; inventory exposure/terminal
   position; quoting participation; cancellations/reposts, lifetime and
   post-to-first-review reversals. Include costs of false cancels, not only
   losses avoided. No interval with fewer than five complete validation days.
5. If a predictor is subsequently fitted, test forecast calibration and
   decision-value ranking against hold/re-entry baselines on chronological
   evaluation, with both actions on matched states and separate sides. Good
   short-horizon fill-probability or first-fill forecasts alone do not meet
   this gate. Neither does a passing simulator test. No model fit is launched
   by this design note.

## Immediate handoff and boundaries

The in-progress saved-data headroom work can finish within its existing
0.05h development / 0.02h review allowance, counting the independent
classification work against it. Treat it as a diagnostic, not the new value
projection or a test of queue necessity. Do not start another tuned variant
or another short-horizon classifier from that receipt.

**ONE next combined deliverable:** close that bounded diagnostic and prepare
the longer-continuation paired-value/controller contract with its pilot cost
and controls. Another session implements; this thread reviews. Bundle useful
corrections, defer minor cleanup, and do not require a review per component.

The standing cumulative ceiling remains **2.50 core-hours**. No new run,
model family, data day, untouched validation, live use or resource extension
is inferred here. A replay launch must have a concrete frozen design and
resource disposition first; prepare those before asking for any necessary
approval. This note changes the objective, not an existing frozen result.
