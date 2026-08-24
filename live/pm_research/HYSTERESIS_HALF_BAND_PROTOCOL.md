# Iteration 004 — hysteresis plus half-order skew-band protocol

**Status: FROZEN BEFORE EXECUTION at 2026-08-24T11:34:22Z. Research only.**

## Hypothesis

Iteration 003's BTC hysteresis candidate passed both development-day PnL bars,
the corrected-skew bar, and both traffic bars, but failed terminal inventory.
The current five-share skew band waits for roughly one complete quote fill
before treating the other side as inventory reducing. Starting reduction at
half a quote should control inventory earlier without changing the signal.

## One change

Add `QR_CANCEL_QHYST_SKEW2P5`, identical to iteration 003's fixed 0.55/0.45 q
hysteresis arm except:

- `skew_band_shares = 2.5` instead of 5.0.

The value is exactly half the frozen five-share quote size and is selected as a
structural unit, not from a sweep. Inventory with absolute Up-equivalent
exposure <=2.5 remains two-sided JOIN. Beyond 2.5, the reducing side follows
the queue-realistic one-tick improvement rule when spread permits and remains
protected from harmful cancellation; the increasing side remains cancellable.

No threshold, model, H, assumed L, features, signal timestamp, price rule,
quote size, data, population, or cancellation lifecycle changes.

## Frozen arms, comparisons, and gates

All eleven iteration-003 arms replay unchanged on independent clocks. The new
candidate uses the same hysteresis schedule and is compared with:

1. `QR_CANCEL_HOLD_X_SKEW`, the unchanged adoption incumbent;
2. `QR_SKEW_ONLY`, the required no-cancel comparator; and
3. `QR_CANCEL_QHYST_X_SKEW`, to isolate the smaller-band effect.

The per-coin adoption bars remain unchanged on August 23/24: positive PnL delta
versus incumbent on both days, dev2 PnL above corrected skew, no increase in
dev2 terminal absolute inventory, no increase in effective cancels or
cancel/repost traffic, and all controls pass.

All-five-day results are context only. No outcome is forward, decision
eligible, or capable of repairing the failed v5 model gate.

## Required controls

- All eleven iteration-003 arms reproduce exactly.
- A five-share-band candidate exactly reproduces the hysteresis parent.
- Synthetic inventory at 2.49 and 2.50 shares stays two-sided JOIN; 2.51 makes
  exactly the correct side reducing.
- One-tick spread still forces JOIN; >=2 ticks permits one-tick improvement on
  the reducing side only.
- Reducing-side cancellation protection and increasing-side eligibility hold.
- Candidate replay is deterministic and reconciles inventory.

Verdict vocabulary remains `ADOPT_DIAGNOSTIC`, `REJECT`, or `BLOCKED`.
