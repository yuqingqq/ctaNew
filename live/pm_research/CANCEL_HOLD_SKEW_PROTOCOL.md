# Harmful-flow cancel-and-hold × skew — offline diagnostic protocol

**Status: frozen before execution. Research only.** This protocol changes the
post-cancel lifecycle while preserving the existing signal, data, latency,
inventory skew, quote size, and comparison arms. It adds no live venue, order,
cancel, or execution port.

## Question

Does the current harmful-flow signal become more useful when an effective
cancel actually withdraws the eligible quote until the signal clears, instead
of immediately reposting the quote behind displayed depth?

## Frozen population and signal

The replay uses the same ten exact-event windows, split, v5 model receipts and
candidate cells as the prior cancellation diagnostics:

| coin | fill horizon | assumed cancel-effective latency |
|---|---:|---:|
| BTC | 50 ms | 25 ms |
| ETH | 250 ms | 100 ms |

There is one BTC and one ETH window on each day from 2026-08-20 through
2026-08-24. Days 20/21/22 trained the model; days 23/24 are already-seen
development data. There is no independent forward holdout.

The side-specific value-weighted harmful predicate remains `q > 0.5` on the
unlagged PM/direct-event source profile with the fixed 10 ms decision cooldown.
No threshold, feature, model, horizon, latency, size, skew band, or sample is
retuned.

## Arms

The existing five arms are replayed unchanged:

1. `JOIN_ONLY`;
2. `FRONT_ONLY`;
3. `SKEW_ONLY`;
4. `CANCEL_ONLY`; and
5. `CANCEL_X_SKEW`, whose effective cancel immediately reposts as JOIN.

One arm is added:

6. `CANCEL_HOLD_X_SKEW`, which uses the lifecycle below.

The prior cancel-all FRONT stress test is not an arm here: it already lost to
JOIN-only cancellation on all four development comparisons.

## Cancel-and-hold lifecycle

`CANCEL_HOLD_X_SKEW` protects the inventory-reducing side. A cancellation may
be submitted only when all of the following hold:

- the harmful predicate has a false-to-true edge;
- the actual resting order is JOIN, not FRONT;
- the side is not the inventory-reducing side under the current 5-share skew
  band; and
- no cancellation is already pending for that order generation.

Near flat, both sides are eligible because neither side is designated as the
reducing FRONT side. Beyond the band, only the inventory-increasing joined side
is eligible.

The submitted cancel remains bound to its order generation. Fills before the
assumed effective time remain. A full fill or natural level replacement makes
the pending cancel stale and it cannot remove the replacement order.

At effective time:

- if the generation still matches and the signal remains harmful, the quote is
  removed and enters `HELD_OUT`;
- if the signal cleared before effective time, the unavoidable submitted cancel
  takes effect and immediately reposts JOIN behind current displayed depth;
- if inventory changed so that the side is now inventory-reducing, the cancel
  likewise reposts immediately rather than holding out; and
- partial fills before effective time are retained.

While `HELD_OUT`, the side cannot fill and is not reposted on book changes. It
is released when either the harmful predicate becomes false or inventory makes
the side reducing. Release immediately posts the current skew intent at the
current touch: FRONT if reducing, otherwise JOIN behind displayed depth. A
missing book delays the release repost until the next valid state.

Persistent true signals do not repeatedly cancel. A false value rearms the
side. Cancel submission and release repost are separate actions.

No arbitrary hold timeout is introduced: signal clear or inventory-role change
is the only release condition. This avoids selecting another duration on the
visible development days.

## Reporting and controls

Primary comparison: `CANCEL_HOLD_X_SKEW - CANCEL_X_SKEW` by coin and day.
Secondary comparison: `CANCEL_HOLD_X_SKEW - SKEW_ONLY`.

Report PnL, spread and drift, filled shares, terminal inventory/cash risk,
submitted/effective/stale/partial cancellations, hold entries, held side-time,
releases, and repost traffic.

Required controls:

- the new replay loop exactly reproduces all five existing arms;
- all-false `CANCEL_HOLD_X_SKEW` equals `SKEW_ONLY`;
- a matching harmful JOIN cancellation enters `HELD_OUT` after latency;
- no held quote can fill or reappear on a book change;
- false signal and inventory-reducing transitions release the held quote;
- FRONT and reducing-side orders cannot submit a hold cancellation;
- fills before effective time remain and stale generations are harmless;
- deterministic reruns and inventory reconciliation pass.

This is diagnostic regardless of outcome. The v5 model failed its gate and the
same two development days have already been inspected. Promotion requires a
frozen candidate on new forward days and, later, measured cancel-effective
latency and queue/rejoin economics.
