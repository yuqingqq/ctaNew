# Iteration 003 — harmful-q hysteresis protocol

**Status: FROZEN BEFORE EXECUTION at 2026-08-24T11:27:39Z. Research only.**

## Hypothesis

The incumbent converts every q crossing of 0.5 into a harmful-state edge. That
creates repeated cancel/repost cycles when a weak score oscillates around its
decision boundary. A fixed symmetric deadband should retain strong harmful
states while preventing weak boundary noise from repeatedly changing quote
eligibility.

## One change

Add `QR_CANCEL_QHYST_X_SKEW`, identical to the queue-realistic incumbent except
for a per-maker-side Schmitt state over the existing v5 harmful probability:

- initial state: clear;
- clear -> harmful only when q>0.55;
- harmful -> clear only when q<0.45;
- q in [0.45, 0.55] retains the prior state; and
- comparisons are strict, so exact boundary equality retains state.

The +/-0.05 symmetric margin is frozen as one conventional, minimally separated
deadband. There is no margin sweep. The v5 model, score timestamps, features,
H, assumed L, queue rules, cancellation lifecycle, quote size, skew band, data,
and population are unchanged. Unlike rejected iterations 001/002, there is no
minimum hold.

## Frozen arms and comparisons

Reproduce all ten iteration-002 arms exactly on their original signal schedules
and replay the hysteresis candidate independently so its different signal clock
cannot contaminate controls. Primary comparison remains candidate minus
`QR_CANCEL_HOLD_X_SKEW`; required no-cancel comparator remains
`QR_SKEW_ONLY`.

Report raw versus hysteretic harmful fraction and false-to-true transitions by
side, PnL/spread/drift, filled shares, terminal inventory/cash risk,
cancel/repost traffic, held fraction, and JOIN/price-improve fills.

## Adoption bars

For each coin on August 23 and 24:

- candidate-minus-incumbent PnL must be positive on both days;
- candidate dev2 mean PnL must exceed `QR_SKEW_ONLY`;
- dev2 mean terminal absolute inventory must not increase;
- dev2 effective cancels and cancel/repost traffic must not increase; and
- all score-reconstruction, parity, lifecycle, determinism, and provenance
  controls must pass.

All-five-day values are mechanism context only. These are repeatedly inspected
development days and the v5 gate remains failed.

## Required controls

- Reconstructing q>0.5 from raw model probabilities exactly equals every
  incumbent boolean schedule.
- Entry=exit=0.5 exactly reproduces the incumbent fills and diagnostics.
- Synthetic paths pin clear entry, deadband retention, harmful exit, side-local
  state, strict equality, and timestamp preservation.
- All ten iteration-002 arms reproduce exactly on independent clocks.
- Deterministic rerun and inventory reconciliation pass.

Verdict vocabulary remains `ADOPT_DIAGNOSTIC`, `REJECT`, or `BLOCKED`.
