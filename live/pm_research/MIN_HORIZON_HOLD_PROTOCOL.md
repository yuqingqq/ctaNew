# Iteration 001 — horizon-aligned minimum hold protocol

**Status: FROZEN BEFORE EXECUTION at 2026-08-24T11:08:17Z. Research only.**

## Hypothesis

The v5 signal predicts whether an eligible fill over horizon H is harmful. Once
a harmful cancellation becomes effective, reposting the same
inventory-increasing side before H because the binary q>0.5 predicate briefly
clears is inconsistent with that target and creates avoidable queue churn.

## One change

Add `QR_CANCEL_MINH_HOLD_X_SKEW`. It is identical to the queue-realistic
`QR_CANCEL_HOLD_X_SKEW` incumbent except:

- BTC: after entering a harmful hold, signal-clear release is forbidden for
  50 ms from cancel-effective time;
- ETH: the same minimum is 250 ms;
- at the first replay decision event at or after the deadline, release occurs
  only if the current signal is clear;
- if harm remains true, the hold continues until a later clear; and
- an inventory-increasing side that becomes inventory reducing releases
  immediately, even before the deadline.

There is no maximum hold and no threshold, model, feature, quote-size, skew,
queue, price, latency, or data change. The timer starts when cancellation is
effective, not when it is submitted. No synthetic market event is invented at
the deadline: the recorded release delay beyond the deadline is reported, and
a zero-delay boundary case is tested.

## Frozen arms and population

Reproduce all eight arms from the queue-realistic receipt exactly, then add the
candidate as arm nine. Use the same ten windows, three training days, two
already-seen development days, five-share size/band, q>0.5 signal, zero source
state lag, and assumed cancel-effective latencies.

Primary comparisons by coin/day:

1. candidate minus `QR_CANCEL_HOLD_X_SKEW`;
2. candidate minus `QR_SKEW_ONLY`.

Report PnL, spread, drift, shares, terminal inventory/cash risk, cancel/hold
traffic, suppressed early releases, deadline releases, and actual JOIN versus
price-improve placements/fills.

## Adoption bars

The candidate changes the loop's research incumbent for a coin only if, on
August 23 and 24:

- candidate-minus-incumbent PnL is positive on both days;
- candidate dev2 mean PnL exceeds `QR_SKEW_ONLY`;
- candidate dev2 mean terminal absolute inventory is no greater than incumbent;
- candidate total effective cancels and cancel/repost actions are no greater
  than incumbent; and
- all controls pass.

All-five-day results are context only. This iteration cannot become decision
eligible or repair the failed v5 model gate.

## Required controls

- All eight queue-realistic-receipt arms have exact fill and diagnostic parity.
- A zero-minimum candidate equals `QR_CANCEL_HOLD_X_SKEW` exactly.
- A signal clear before the deadline does not repost.
- A clear at/after the deadline reposts at the first decision event under
  current queue-realistic rules.
- Persistent harm at the deadline remains held.
- Inventory-reducing transition releases immediately before the deadline.
- Held quotes cannot fill; stale-generation and partial-fill safety remain.
- Exact-deadline ordering and deterministic rerun are pinned.

Verdict vocabulary: `ADOPT_DIAGNOSTIC`, `REJECT`, or `BLOCKED`. There is no live
or forward-promotion verdict.
