# Iteration 002 — exact horizon-timer protocol

**Status: FROZEN BEFORE EXECUTION at 2026-08-24T11:19:15Z. Research only.**

## Hypothesis

Iteration 001 rejected a next-event minimum hold, but its release occurred
171-916 ms after the intended deadline on average by coin/day. An executable
decision engine can schedule an internal wake-up. Evaluating release exactly at
cancel-effective time plus the prediction horizon isolates the intended 50 ms
BTC / 250 ms ETH lifecycle without waiting for an unrelated book event.

## One change

Add `QR_CANCEL_MINH_TIMER_X_SKEW`. It is identical to iteration 001's
`QR_CANCEL_MINH_HOLD_X_SKEW` except that a provisional internal timer is
scheduled for `cancel_submit + assumed_L + H`:

- if the matching cancel entered a hold and q is clear at the timer, repost
  exactly at the deadline under current queue-realistic placement;
- if q remains harmful, stay held until a later clear;
- if the cancel was stale, never became effective, or the hold already ended,
  the timer is a diagnosed no-op; and
- inventory-reducing release remains immediate before the timer.

No signal, threshold, H, L, model, features, price, size, skew, queue rule,
market data, or population changes.

## Frozen arms, comparisons, and bars

Reproduce all nine arms from iteration 001 exactly and add the timer candidate
as arm ten. Primary adoption comparison remains candidate minus the original
queue-realistic incumbent `QR_CANCEL_HOLD_X_SKEW`; required economic comparator
remains `QR_SKEW_ONLY`. Iteration 001 is also reported to isolate timer value.

The per-coin adoption bars are unchanged:

- positive candidate-minus-incumbent PnL on both August 23 and 24;
- candidate dev2 mean PnL above `QR_SKEW_ONLY`;
- no increase in dev2 mean terminal absolute inventory;
- no increase in dev2 effective cancels or cancel/repost traffic; and
- every lifecycle, parity, determinism, and provenance control passes.

The same ten windows are development diagnostics only. All-five-day means are
context, not gates; no outcome can repair the failed v5 gate or authorize live
use.

## Required controls

- All nine iteration-001 arms reproduce exactly.
- A disabled exact timer equals iteration 001 exactly.
- A clear before deadline remains held and releases at the exact timer.
- Harm at the timer remains held and releases on a later clear.
- A stale/no-hold timer is a no-op.
- Inventory-reducing transition releases before and invalidates the timer.
- Same-timestamp ordering is state, signal, cancel-effective/timer, with no
  fill possible while held.
- Deterministic rerun and inventory reconciliation pass.

Verdict vocabulary remains `ADOPT_DIAGNOSTIC`, `REJECT`, or `BLOCKED`.
