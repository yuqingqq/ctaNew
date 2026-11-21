# Default-post inventory addendum — USER-directed comparison

2026-11-21. Applies to [the duration/size plan](../20261121_default_post_duration_size/plan_v2.md)
at **a4c0d2880b8ce3039de20aa99af91ced0e9633d9**. The USER asks whether the hard
position cap accommodates larger posts and whether corresponding inventory
management is needed. This authorizes a small inventory-control comparison
alongside the unchanged default-post baselines. No fitted adverse-selection
model, new dates, price-skew search or additional compute allocation.

## The cap and the controller do different jobs

Use client-known inventory I, unacknowledged bid/ask reservations B/A, hard
per-market share cap C, and requested new order quantity q. A new bid needs
`I + B + q <= C`; a new ask needs `-I + A + q <= C`. Do not net the two
unfilled sides against each other. Hidden fills and cancel-pending exposure
stay reserved until the existing notice rules release them.

A larger order is valid only when its whole quantity fits. For example,
with C=100, I=+20 and B=0, a new 100-share bid is blocked; up to 80 shares
would fit. Raising C from 30 to 100 admits larger orders but also permits
larger directional exposure. A share cap limits exposure; it does not make
inventory return toward zero or limit monetary drawdown across markets.

The existing fixed-size policy rejects a whole order when insufficient room
remains. Keep that behavior in the v2 controls. Do not silently change the
baseline to smaller posts while describing it as fixed size.

## Two additional, declared arms at the largest planned size

Keep v2's five scenarios. Add ONLY these two 60-second/Qmax100/C100 arms,
giving seven scenarios total on the SAME at-most-96-world manifest:

1. **ROOM_CLIP_Q100_C100**: neutral desired size Q=100 on either absent side;
   clip it to available reservation room. Existing orders keep their quantity
   and queue. This isolates useful smaller posts from an inventory preference.
2. **INVENTORY_SKEW_Q100_C100**: the same clipping, plus the simple inventory
   rule below. Compare it primarily with ROOM_CLIP and additionally with the
   original fixed-Q100 default policy. Q100 is now a maximum, not a claim that
   every submitted order has 100 shares. Report the actual size distribution.

Use the established exchange minimum/quantity increment if it is available
in the frozen market metadata; otherwise explicitly inherit the research
quantity convention. Never invent a venue minimum. Round down to an admissible
quantity and WAIT with a reason if none fits; never round above the budget.

### Inventory rule: frozen choices, no parameter fitting

Target inventory is zero. Soft trigger S=C/2=50 shares; recovery band R=C/4=25
shares. These are declared diagnostic choices, not calibrated optimal limits.
Desired NEW order sizes outside a restricted mode are:

`bid_desired = Q * clip(1 - I/S, 0, 1)`
`ask_desired = Q * clip(1 + I/S, 0, 1)`.

At positive inventory the bid becomes smaller while the ask remains available;
the rule is symmetric at negative inventory. Then apply the hard room limit:

`bid_new = min(bid_desired, max(0, C - I - B))`
`ask_new = min(ask_desired, max(0, C + I - A))`.

For the example I=+20/C=100/Q=100/B=0, neutral room clipping permits 80 bid
shares; the inventory rule desires 60. The reducing-side order can still
overshoot zero if it fills in full; report position sign changes rather than
claiming this is a guaranteed flattening policy.

Persistent restriction mode, updated from client-known state after notices:
- I >= +S sets LONG_RESTRICTED; I <= -S sets SHORT_RESTRICTED (including a
  direct reversal into the opposite mode when its trigger is reached).
- Otherwise, abs(I) <= R clears the mode. Between R and S retain the previous
  mode. Start unrestricted at I=0. This is inventory hysteresis, not a timer.
- LONG_RESTRICTED: no new bids and request a pull of a live bid; leave the ask
  alone. SHORT_RESTRICTED is symmetric. Outside these restrictions preserve
  existing live/partial orders, even if a fresh-order sizing formula changed.

Respect the existing action-eligible states and cancel latency. A pending
order is not instantly removable, a request is not an acknowledgement, and
an already filled position is not undone by pulling its remainder. Log
inventory-requested removals separately from structural and model removals.
Do not cancel/repost to resize every 50 ms. Keep the reducing-side queue;
new orders resume on the normal grid when inventory recovers and reservations
permit them. Include the mode in branch snapshots, identity and resume state.

This first mechanism changes size/admission and selectively cancels excess
same-direction exposure. It does not change prices, cross the spread, hedge,
liquidate at END or add an inventory cost to reported cash P&L. There is no
promise that it improves return: it can forgo good fills or oscillate through
opposite fills. Those are measured costs of the policy.

## Measurement and acceptance in the same round

Compare the three Q100/C100 arms on common evaluable worlds. Report gross
value, execution edge/markouts, tail losses, peak/integrated absolute inventory,
time above S, terminal position, and time until recovery (censored at END).
Alongside them show filled shares, quote/both-side time, actual submitted
sizes, cap blocks, restriction transitions, inventory pulls, replacement/queue
loss and first-review cancellation counts. Smaller inventory alone is not
economic success. Identify whether clipping or the inventory preference
caused the change. Use only the known-state rule in replay; no ex-post removal
of inventory losses or favorable selection of days/sides.

Add focused controls to the existing batch: absent versus pending room;
same/opposite-side reservations; hidden fills; mirrored long/short cases;
I=0, +/-25 and +/-50 boundaries and mode reversal; no repeated resize churn;
no instant release on pull; partial/full fills at actual submitted size;
serialization of restriction mode; and per-arm cash/position/cap identities.
Do not expand into a general execution rewrite or a new large mutation suite.

Run all seven arms in the existing eight-world pilot before projecting the
full batch. The two-core-hour/remaining-allocation ceiling and memory limits
remain unchanged. If the projection does not fit, report it and stop before
the full run; do not silently add resources, drop a comparator or thin by
outcome. Preserve any already completed v2 baseline artifacts by identity;
the new arms may be added under a separate committed freeze. One combined
result review, without separate reviews for each arm or repair.

The current next priority remains the duration/size/exposure diagnosis, now
with this inventory comparison. Protection-model fitting stays deferred.
Development can implement this addendum within the stated scope without
another confirmation. This planning commit contains no simulation result.
