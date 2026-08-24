# Harmful-flow cancel × skew — offline development protocol

**Frozen before the run on 2026-08-24. Research only.** This protocol enables
an offline stateful composition requested after the harmful-flow v5 gate
failed. It cannot promote, cannot become decision eligible, and adds no live
order, cancel, venue or execution port.

## Population and candidate cells

The replay uses the same one BTC and one ETH window per day selected by adverse
v2–v5: 10 windows over 2026-08-20..24. The v5 models remain unchanged. The two
predeclared diagnostic cells are:

| coin | fill horizon | assumed cancel-effective latency |
|---|---:|---:|
| BTC | 50 ms | 25 ms |
| ETH | 250 ms | 100 ms |

Training days remain 2026-08-20/21/22 and the already-seen development days
remain 2026-08-23/24. The stateful result is descriptive on every day; there is
no promotion test.

## Composition semantics

Five 5-share, no-terminal-abstention arms share the same unlagged PM event tape:

1. `JOIN_ONLY`: two-sided JOIN, no skew or cancellation;
2. `FRONT_ONLY`: symmetric front placement, no cancellation;
3. `SKEW_ONLY`: pessimistic `SKEW_LB`, no cancellation;
4. `CANCEL_ONLY`: two-sided JOIN plus the v5 harmful-flow signal; and
5. `CANCEL_X_SKEW`: pessimistic `SKEW_LB` plus the same signal.

The v5 action schema is JOIN-at-touch behind displayed depth. Consequently,
cancellation may act only while the actual order was last placed or reposted
as JOIN. A fronted order is never scored as cancellable. Under skew this means
the model usually controls the inventory-increasing joined side while leaving
the fronted reducing side intact. This is named `JOIN_SCHEMA_ONLY`; extrapolating
v5 to FRONT is forbidden.

The value-weighted harmful predicate is `q > 0.5`, unchanged from v5. It is
evaluated at the exact-event 10 ms-cooldown feature times. A false value arms a
side; the next true value submits one cancel if the side is joined and resting.
Persistent true values do not repeatedly cancel. The predicate must clear to
re-arm.

Cancellation is bound to the current order generation and becomes effective
after the coin-specific assumed latency. Fills before effective time remain.
At effective time:

- a matching live generation is cancelled and immediately rejoins the back of
  displayed depth at the current same-side touch;
- any partial fill is retained and named `partial_fill_then_cancel`;
- a full fill or natural level replacement changes generation, so the stale
  pending cancel cannot remove the new order; and
- cancel submission and repost are counted as separate actions.

The replay does not impute cancel/rejoin cost, ACK time, or queue state from a
live venue. Latencies remain assumed counterfactual values and are never called
`tau_operative`.

## Accounting and controls

Every arm reports total 5-second gross markout PnL per window, share-weighted
markout, shares and spread captured, terminal net inventory and side-aware cash
at risk. Cancellation arms additionally report signal evaluations, submitted,
effective, stale-generation and partial-fill cancellations, and reposts.

Controls run before the result:

- with cancellation disabled, the new loop must reproduce the existing
  `policy_optimizer.replay_cells` fills and mid path exactly at zero state lag;
- all-false signals must equal the matching no-cancel arm exactly;
- a synthetic joined-side lifecycle must retain a partial fill, reject a stale
  generation and repost behind displayed depth on a matching generation;
- deterministic reruns must have identical fills and cancellation counts; and
- inventory reconciliation must hold independently from the fill stream.

`CANCEL_X_SKEW` is compared with `SKEW_ONLY`; `CANCEL_ONLY` with `JOIN_ONLY`;
and the composition with both single-mechanism arms. Since v5 did not pass its
model gate and only two development days exist, even uniformly improved replay
PnL would be a diagnostic bound, not evidence that cancellation works live.
