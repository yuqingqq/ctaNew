# Codex queue-invariant audit — required strategy arms — 2026-08-29

**Exact reviewed tip:** `28d2f57d51412064ddee094558a8ef445482f4f3`

**Scope:** `QR_SKEW_ONLY`, `QR_CANCEL_HOLD_X_SKEW`, their queue engine, the
state-feature queue inputs, and the exact current fragment exposure. This is a
correctness audit only. No strategy/model score was run and no candidate was
promoted.

## Verdict

**THE REQUIRED QUEUE-REALISTIC ARMS PRESERVE THE SAME-PRICE INVARIANT.** At an
existing BBO, both required arms set `front=False` and initialize queue ahead
to the positive displayed depth. A reducing side may receive zero queue only
when it improves strictly inside a spread of at least two ticks; that is a new
price level, not a same-price/front assumption. A signal-clear or
cancel-effective repost at the existing touch rejoins behind displayed depth.

The current exact exposure also carries queue identity completely. Therefore
this audit does not invalidate an existing receipt or add a blocker to the
pending fragment run.

One future-proofing defect remains in `harmful_state_features`: an absent
source `qahead` is coerced to `0.0` and marked present. It is not instantiated
by the current exposure, but the checker cannot refuse its known-bad and should
be fixed in the next identity-moving source-schema cycle.

## Queue engine evidence

`QueueRealisticArm.desired_order` has two disjoint paths:

- `JOIN_EXISTING`: level equals current bid/ask, queue ahead equals displayed
  bid/ask size, and `front=False`;
- `PRICE_IMPROVE_1T`: reducing side only, valid tick, spread at least two
  ticks, level proven strictly between bid and ask.

Executed committed batteries:

```text
policy_optimizer_queue_realistic --selftest: 16 checks passed
harmful_state_features --selftest:          105 checks passed
```

The queue battery covers one-tick join, two-tick reducing-side improvement,
non-crossing placement, flat-policy join, harmful cancel/hold, signal-clear
rejoin behind displayed depth, and protection of the reducing side.

The inside-spread path still carries the receipt's declared limitation:
competitor response and rejoin queue value are unavailable. It remains a
development diagnostic and cannot be read as robust executable performance.

## Exact exposure census

I streamed every row of
`be_fragment_exposure_rows_v1.json` rather than sampling its header:

```text
rows                         482,224
OK rows                      472,413
GAP_IN_HORIZON                 9,669
TRUNCATED_HORIZON                142
rows missing qahead                 0
rows with explicit qahead=0    67,846
rows with qahead>0             414,378
duplicate (slug,side,gen,t)          0
OK rows missing latency/gate         0
```

The 9,811 non-OK rows omit `latency`/`any_fill_ahead` consistently with their
declared horizon status. No row omits `qahead`, `level`, or `resting`. Thus the
latent source fallback below did not manufacture queue-zero rows in this exact
artifact.

## QI-R1 — missing queue state is encoded as confidently front

`harmful_state_features.features_at` currently uses:

```python
qahead = float(row.get("qahead") or 0.0)
```

and defines `queue_ahead_missing` only from
`qahead + resting <= 0`. With positive resting size, an absent queue field is
therefore encoded exactly like a real zero queue and its missing flag is
`0.0`.

Executed synthetic pair, identical except that one row explicitly carries
`qahead=0.0` and the other omits `qahead`:

```text
                         explicit zero    missing field
queue_ahead_norm             0.0              0.0
queue_ahead_missing          0.0              0.0
queue_ahead_of_level         0.0              0.0
```

This is anti-safe: a future malformed source row would tell the model it is
confidently at the front rather than unknown. The derived-tape field-count
gate cannot recover the source omission because all derived fields are still
present.

## Closure and scope

- At the source-feature boundary, require `qahead` to be present, finite, and
  nonnegative, or represent absence as `None` plus a true missing guard.
- Add a known-bad missing-`qahead` row and a positive explicit-zero control;
  they must produce different outcomes.
- Apply the same source-schema discipline to `level` and `resting`, whose
  absence must not become an executable queue state.
- Because `harmful_state_features.py` is in the model identity lattice, queue
  this with the already-recorded encoder/source-schema identity move; do not
  silently rebind a frozen model for this non-instantiated defect.

This audit leaves the strategic requirements unchanged:
`QR_CANCEL_HOLD_X_SKEW` remains the queue-realistic baseline,
`QR_SKEW_ONLY` remains mandatory, and any performance claim still requires the
full lifecycle/cost replay on independent complete UTC days.
