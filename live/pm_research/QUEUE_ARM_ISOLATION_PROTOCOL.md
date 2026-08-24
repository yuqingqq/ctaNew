# Iteration 006 — queue arm-isolation protocol

**Status: FROZEN BEFORE REPLAY at 2026-08-24T12:06:00Z. Research only.**

## Correction

Replay every policy cell through its own `QueueRealisticArm`, state heap,
signal heap, and cancel-effective heap. Combining returned immutable window
results is allowed only after every single-arm replay finishes.

No placement rule, signal value or timestamp, q threshold, H/L cell, quote
size, skew band, cancellation lifecycle, hold/release rule, PM/HF input,
markout, or incentive assumption changes. In particular, post-fill skew intent
continues to become placement-effective at the next event in that arm's own
clock; this iteration does not introduce a new post-fill timing rule.

## Rebuilt pair

- baseline: `QR_CANCEL_HOLD_X_SKEW` with the frozen v5 BTC H50/L25 and ETH
  H250/L100 q>0.5 schedules;
- mandatory comparator: `QR_SKEW_ONLY` under the identical signal timestamp
  clock but cancellation disabled.

The old multi-arm artifact remains immutable and is reported only as a
contaminated historical comparison. The rebuilt artifact supersedes its
performance numbers if all controls pass.

## Required controls

- changing cell order cannot change any fill, midpoint, gap, or diagnostic;
- adding/removing another cell cannot change a single-arm result;
- repeated isolated replay is exact;
- all-false `QR_CANCEL_HOLD_X_SKEW` equals isolated `QR_SKEW_ONLY`;
- fill/inventory reconciliation holds for every window;
- source/model/code/protocol receipts and artifact hash validate.

All five days remain seen development data. This correctness rebuild cannot
make the v5 model, baseline, or any strategy decision eligible.
