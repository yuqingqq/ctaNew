# Harmful-flow cancel-all × skew — offline diagnostic protocol

**Status: frozen before execution. Research only.** This protocol adds one
diagnostic arm to the existing JOIN-schema-only cancel×skew replay. It does not
add a live order, cancel, venue, or execution port.

## Question

What happens if the v5 harmful-flow signal is allowed to cancel every resting
order in `SKEW_LB`, including the inventory-reducing order placed at the front
of the queue?

This is deliberately an out-of-action-schema stress test. The v5 model was
trained on JOIN-at-touch outcomes. Its score is not calibrated for FRONT
orders, so this experiment can describe the replay consequence of the proposed
rule but cannot validate the predictor for that action.

## Frozen population and signal

The data, split, feature clock, model receipts, quote size, candidate cells and
assumed cancel-effective latencies are unchanged from
`CANCEL_SKEW_HARMFUL_PROTOCOL.md`:

| coin | fill horizon | assumed cancel-effective latency |
|---|---:|---:|
| BTC | 50 ms | 25 ms |
| ETH | 250 ms | 100 ms |

The replay uses one exact-event window per coin-day for 2026-08-20 through
2026-08-24. Days 20/21/22 are model-training days and 23/24 are already-seen
development days. There is no independent forward holdout.

The side-specific predicate remains the value-weighted v5 probability
`q > 0.5`, evaluated at exact-event feature times with the fixed 10 ms
cooldown. A false value arms the side; the next false-to-true edge may submit
one cancel. Persistent true values do not submit repeated cancels until the
predicate clears.

## Arms and only changed treatment

The original five arms are replayed unchanged:

1. `JOIN_ONLY`;
2. `FRONT_ONLY`;
3. `SKEW_ONLY`;
4. `CANCEL_ONLY` (JOIN); and
5. `CANCEL_X_SKEW` (JOIN-schema-only cancellation).

One arm is added:

6. `CANCEL_X_SKEW_ALL`: identical to `CANCEL_X_SKEW`, except a harmful-signal
   edge may cancel either a joined order or a fronted skew order.

No threshold, model, horizon, latency, skew band, size, feature, or sample is
retuned for this arm.

## Lifecycle semantics

Cancellation remains bound to the resting order generation and becomes
effective after the coin-specific assumed latency. Fills before effective time
remain. A stale pending cancel cannot remove a naturally replaced order.

At effective cancellation, the remaining order is removed and a fresh 5-share
order immediately JOINs behind displayed depth at the current same-side touch.
This is true even when the cancelled order was FRONT. `SKEW_LB` may request
FRONT again only on a subsequent genuine touch formation; cancellation never
teleports the repost to the front.

Cancel submission and repost are separate actions. ACK time, real
cancel-effective latency, cancel/rejoin cost, and live queue state are not
imputed.

## Controls and reporting

- The original JOIN-only controls must still pass.
- With an all-false signal, `CANCEL_X_SKEW_ALL` must equal `SKEW_ONLY` exactly.
- A unit lifecycle must demonstrate that a fronted order can be submitted,
  cancelled after latency, and reposted behind displayed depth.
- Deterministic reruns must reproduce fills and diagnostics.
- Inventory reconciliation must hold for every arm.

Primary comparison: `CANCEL_X_SKEW_ALL - CANCEL_X_SKEW` by coin and day.
Secondary comparisons are against `SKEW_ONLY` and `CANCEL_ONLY`. Report PnL,
shares, spread capture, terminal inventory risk, cancellation traffic, stale
generations, and partial-fill cancellations.

Regardless of PnL, this arm is non-promotable because it applies a JOIN-trained
predictor to FRONT actions on already-visible development data. A decision-
eligible version requires action-conditioned FRONT labels, a frozen model, and
new forward days.
