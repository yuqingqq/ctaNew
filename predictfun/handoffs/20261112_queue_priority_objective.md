# Queue priority objective — user clarification, 2026-11-12

The objective is to quote early enough to earn queue priority, retain useful
resting orders through harmless fluctuations, and withdraw when avoidable
adverse fills outweigh the value of staying. The user is not committed to the
current fitted model. Reintroducing the old entry permission is not the default
research direction: waiting for a favorable entry signal may put the order
behind the queue that has already formed.

The original guard remains a performance benchmark. A stronger guard benchmark
does not establish that its entry architecture is the right way to achieve this
objective. The completed negative default-pull replay and subsequent pull-quality
audit remain valid for the tested implementation; they do not reject the broader
objective.

## Agreed economic comparison

At each decision state, compare expected future net value of KEEP with expected
future net value of PULL followed by the declared re-entry policy. Both paths
start from the same order, inventory, market information and unavoidable fills
before cancellation becomes effective. A pull cannot claim to avoid those
prefix fills.

The comparison must account jointly for filled quantity and fill quality:

- favorable fills earned or forfeited while resting, waiting and rejoining;
- adverse fills suffered or avoided over those same periods;
- inventory consequences under the same cost/risk convention in both paths;
- subsequent opportunities affected by retaining or surrendering queue priority.

Queue position changes which fills occur and when. Its value should emerge from
that paired future-value difference. Do not add a separate queue-loss bonus or
penalty on top of future fills that already price the same effect. If a finite
horizon leaves an unfilled queue position, any terminal continuation estimate
must cover only value beyond that horizon and be separately validated.

Favorable-fill value and adverse-fill loss are reporting components of one net
valuation, not an independent prediction of fill probability multiplied by an
unconditional edge: fill probability, quantity and toxicity can be dependent.
Preserve the distinction between local markout utility and portfolio P&L; do
not add both valuations together. Rebates remain separately reported and the
loss-budget setting remains unchanged.

The policy retains default resting while KEEP has at least as much expected
value as PULL/re-entry, subject to hard price/data/inventory constraints. A
simple count of cancellations or time near the front is not a success metric.
The model, its horizon and the re-entry policy remain design choices to test;
the current regression is not a required component.

## Concrete target limitation found in the current implementation

`pf_default_pull_hooks.py::Hook.label` and `pf_default_pull_shadow.py::paired_leg`
compare retaining the existing queue position with canceling and resetting at
the same price after one second. They already account for queue loss through
the different fills within the local, approximately two-second fill horizon.
The 1s/5s markouts value those fills; they do not extend the fill horizon.

There is no continuation value for an unfilled order's queue position at the
end of that horizon. If neither branch fills, the label is zero even when
KEEP has advanced much closer to the front than RESET. Thus the target can
undervalue patience whose payoff arrives later. This is a verified structural
limitation, not yet a measured attribution of the portfolio losses.

## Proposed next study, not yet declared or run

1. Test the premise directly on neutral decision-time exposures: does retaining
   an earlier order capture more favorable fills than relinquishing priority
   and rejoining later? Include unfilled, adverse and invalidated orders, with
   conservative queue and cancellation timing assumptions.
2. Value KEEP versus PULL using a consistent subsequent policy and a predeclared
   horizon that captures queue progression, or a separately validated terminal
   continuation value. Do not assign an arbitrary reward to order age or queue
   rank. Use identical re-entry price, recovery conditions and latency in the
   labels and sequential replay.
3. Retain default quoting subject to hard data, price and inventory constraints.
   Learn when the avoidable loss of remaining exposed exceeds the expected fill
   value and retained priority. Fast withdrawal can remain optimal near the
   front when imminent flow is adverse. Recovery-based re-entry is a separate
   part of the policy, not proof that a fixed cooldown clears danger.

Judge success by incremental favorable fill value and complete portfolio trading
value at comparable inventory exposure. Resting duration, cancel rate and
estimated queue position are explanatory metrics, not the objective themselves.
Displayed depth/progress is only a queue proxy; the simulator's hidden queue
state must remain label-only. Existing live queue/fill calibration limitations
remain in force. New numerical tests require a source/declaration freeze and
retain the consumed-versus-untouched distinction.

No new model, strategy variant, numerical experiment or live change was launched
for this clarification. Earlier artifacts and conclusions are unchanged.

## Subsequent first-stage result

The explicit fill/value diagnostic is now complete: [review](20261112_fill_value_review.md). It found useful KEEP fill-opportunity signal, weak RESET forecasts, no improvement in pull selection, and inconsistent independent value/quantity components. This does not complete the longer-horizon queue-retention or live-calibration work.
