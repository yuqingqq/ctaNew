# Codex integrated-baseline readiness review — 2026-08-29

**Audit tip:** `4c3a397d3ae5a8d184b734b5427a59114e1ed2d5`  
**Scope:** queue-realistic strategy baselines and the bridge from neutral
opportunities to stateful cancel × skew replay  
**Result-bearing work:** none; synthetic/read-only correctness audit only

## Verdict

**QUEUE ENGINE PASS; ACTIVE INTEGRATION HOLD.**

The committed queue engine continues to preserve the same-price invariant:
joining an existing touch starts behind positive displayed depth, while zero
queue is possible only on a strictly improved inside-spread price. The required
strategy identities remain:

- queue-realistic baseline: `QR_CANCEL_HOLD_X_SKEW`;
- mandatory comparator: `QR_SKEW_ONLY`.

The existing parity artifacts do **not** yet instantiate a queue-aware active
comparison between those strategies. They are correctly labelled inert
lifecycle parity. Treating them as integrated strategy performance would widen
their claim beyond what their event schema can represent.

Do not run or report active seven-arm economics until IR-R1 through IR-R4 below
are closed and re-reviewed.

## Executed checks

At the audit tip:

| Instrument | Result |
|---|---|
| queue-realistic engine selftest | 16 checks, green |
| stateful lifecycle selftest | 83 checks, green |
| independent replay parity battery | 108 checks, green |
| BE inert-arm producer selftest | green; BE and DA stub events agree 6/6 |
| absolute-clock counterexample | **fails**: current producer orders a 7200s event before a 100s event |

The green batteries are useful and remain valid within their declared scope:
disabled/infinite-threshold parity, lifecycle transitions, stale-fill charging,
rate-limit accounting, schema refusal, independent producer agreement and
cross-process determinism. None carries queue-ahead or active model economics.

## IR-R1 — the real inert producer still discards the governing clock

`be_inert_arm_run.opportunities()` copies:

```python
t = float(row["t_start"])
```

and globally sorts opportunities on that window-relative value. It does not
use `t0 + t_start`, even though `t0` is the event's absolute window clock.

Executed two-window fixture:

```text
row A: slug=late,  t0=7200, t_start=0    governing t=7200
row B: slug=early, t0=0,    t_start=100  governing t=100

current producer order: late@0, early@100
governing order:        early@100, late@7200
```

This is not cosmetic. It changes cross-window sequence, hour matching and any
rate-limit or dwell calculation that consumes the exported clock. It was found
in the batch-3 review and remains open in the current source.

**Closure:** emit and sort on the exact governing instant `t0 + t_start`, carry
the clock basis in the artifact, and add this inversion fixture plus an
equal-relative-time/different-window fixture. The real producer and independent
consumer must agree on the corrected absolute trajectory before active use.

## IR-R2 — inert lifecycle parity is not queue-aware replay

The producer's opportunity projection contains only:

```text
slug, side, gen, t, qty, price
```

It drops `qahead`, join-versus-price-improve placement, `front`, and every
fresh-queue state. Its output deliberately bans economics and always declares
`predictor="none"`, `predictor_active=false`. Only two inert compositions are
produced.

That is sufficient for the narrow lifecycle anchor and insufficient to prove
that a cancel/repost re-enters the real queue correctly. The separate queue
audit proves the queue engine's current semantics; it does not prove the
stateful replay is wired to that engine.

**Closure:** build a typed neutral reference directly from the queue-realistic
engine, retaining placement reason, level, displayed depth/queue ahead, genuine
generation bounds and fill tranches. A disabled stateful overlay must serialize
bit-identically to that reference. A same-price repost must receive a fresh
behind-depth queue; an inside-spread placement must prove strict improvement.
Deliberate zero-queue-at-same-price and reused-queue known-bads must refuse.

## IR-R3 — the neutral skew reference is still an unfrozen draft

`SKEW_LANE_NEUTRAL_REFERENCE_FREEZE.md` remains
`DRAFT-FOR-USER-FREEZE` with three unresolved questions. Thus current code is a
useful description, not yet the frozen semantic identity of future results.

Recommended pre-result rulings, because they preserve existing code rather
than inventing an interface:

1. define desired exposure as the existing pair `(displayed size,
   front/back placement intent)`, not a nonexistent scalar object;
2. record marginal inventory-risk value as **NOT PRESENT** today. It cannot
   enter `delta_EV` until a separately preregistered policy-layer estimator
   exists; no docstring rationale becomes a number;
3. keep `charge_reset_cost_at_generation_start` outside the skew freeze, but
   require it to be separately frozen in the lifecycle policy before an active
   run.

Whichever rulings are adopted must be frozen by the user before results. No
band, hysteresis or threshold may be selected on consumed 08-20..25 data.

## IR-R4 — no real generation-tranche artifact feeds the state machine

`harmful_stateful_policy.validate_reference()` requires every generation to
carry its own interval, level, displayed size, status and event-level
`tranches`, each with its own fill time, shares and markout. The 705 MB v3.4
top-up instead carries per-row, per-latency aggregates. Its header is
`harmful_exposure_v3_4_fill_scoped_markout`; the handoff correctly says it
cannot drive the real policy replay.

`phase4_generation_tables.tranche_table()` does not close this. It has no
production consumer and emits one aggregate value/shares triple per generation,
not the event-level tranche list the state machine requires. Its module name
must not be read as evidence that tranches were persisted.

The authoritative primitive already exists:
`harmful_exposure_rows.generation_table()` attributes each fill to its
pre-consume generation and retains the fill's own `t`, `shares`, `level` and
five-second markout, with orphan and wrong-generation reconciliation counts.

**Closure:** commit a generation-reference builder using that primitive,
preserve statuses rather than dropping unvalued/gapped generations, reconcile
every fill/action to the neutral engine, bind source and builder identities,
and add malformed/duplicate/out-of-generation known-bads. The artifact must be
created before any active replay and consumed through the real stateful entry
point in a non-scoring seam.

## Minimum integration sequence

1. Finish the bundled identity cycle already filed for Iteration 011, including
   the missing-queue source guard.
2. Repair the absolute clock and build the real generation-tranche neutral
   reference.
3. Obtain the three skew/lifecycle freeze rulings and commit the frozen
   semantics before results.
4. Prove disabled and infinite-threshold bit parity through the **real** queue
   reference, including same-price queue/repost falsifiers.
5. Wire active scores only after Iteration 011's separate fit/score hold is
   released. Run the seven arms on identical neutral opportunities and clocks;
   controls remain matched on action count, side, hour and budget.
6. Quote no performance conclusion until independent complete post-O1 UTC days
   reach the governed bar. No policy-generated path may become training data.

This work is compatible with the one-hypothesis rule: it is correctness and
interface closure around the already-frozen Iteration 011 line, not another
scored family.
