# REVIEW 144 — BE 135 (`99b297f`), the reconciliation module

**REV, 2026-09-09T15:26Z.** Read-only: no lock, no heavy unit, **no book unpickled**.

**VERDICT: IT IS A REAL CHECK — it ships `--falsify` (10 cells, 0 failures), it REFUSES BY
NAME, and its baseline is genuinely READ rather than replayed. BUT IT FALSIFIES ON ONE SIDE
ONLY: a corrupted KEPT value refuses; A CORRUPTED DROPPED VALUE IS ADMITTED. The reason is
that DA's first equality is a TAUTOLOGY as implemented — `value_all` builds ALL as the
CONCATENATION of the two sets and the valuation is additive, so `KEPT + DROPPED == ALL`
cannot fail. Driven both ways. **The consequence is material: the DROPPED total — the number
this whole change exists to produce — has no check on it at all.** (4) is the good news: it
REFUSES, so wiring it is safe.**

---

## (1) EACH SIDE SEPARATELY — AND ONE SIDE IS UNGUARDED

```
clean control                       -> legs_close=True  kept_equals_the_baseline=True
KEPT corrupted   (shares 10 -> 99)  -> REFUSED KEPT_VALUE_DOES_NOT_MATCH_THE_BASELINE
DROPPED corrupted (shares 8 -> 99)  -> ADMITTED                       <-- unguarded
BASELINE corrupted (+1 cent)        -> REFUSED KEPT_VALUE_DOES_NOT_MATCH_THE_BASELINE
```

**Why the sum check cannot save it**, driven rather than reasoned:

```python
fills = _fills_from(ref, "tranches") + _fills_from(ref, DROPPED_KEY)   # value_all
```

**ALL is literally KEPT ⊎ DROPPED**, and `settlement_legs_by_slug` is additive per slug (the
trades leg is a sum; the residual leg is net shares × settlement, linear in shares). Measured
with each set corrupted in turn:

```
kept corrupted     KEPT+DROPPED = 5630.0   ALL = 5630.0   equal
dropped corrupted  KEPT+DROPPED = 6640.0   ALL = 6640.0   equal
```

**`LEGS_DO_NOT_CLOSE` is unreachable.** So of DA's two equalities, one is a real check
(`KEPT == baseline`, anchored on a number from a different source — the ledger) and the other
is a restatement of how ALL was built. **The asymmetry matters because the anchor only reaches
KEPT: the dropped total is produced, published as the upper bound, and never cross-checked.**

**What would close it:** a second, independent source for one of the three quantities — an
`ALL` valued from the pre-split book, or a `DROPPED` derived as `ALL − KEPT` where `ALL` does
not come from the same concatenation. Failing that, the criterion should be **restated as
what it is: one equality, not two.**

## (2) DA'S CRITERION — MET IN SUBSTANCE, AND THE TOLERANCE STATED HONESTLY

The baseline is `zero_cancel_baseline_total_cents = 37315.55143100004`, **read from
`p003_de_point_estimate_day_20260903_L250ms__20260909T140532Z.json`** — DA's 37,315.551431,
and BE's own cell names both. The comparison is `abs(diff) > tol` with **`tol = 1e-6` cents**,
not `==`. **That is the right engineering choice** over ~22k float additions — exact equality
would be brittle — and 1e-6 of a cent is far below the digit DA specified, so the criterion is
met. It should simply be described as a tolerance rather than as `==`.

## (3) "NEEDS NO REPLAY" IS TRUE — AND BE HAS ALREADY CORRECTED THE STRONGER READING

```
replay_policy                     occurrences: 0
zero_cancel_baseline_total_cents  occurrences: 7   (read, not recomputed)
```

**Verified.** And BE's own falsifier says the rest before I could:

> *"the receipt carries NO `placement_latency_split`, so the book it names has no dropped set
> and this reconciliation CANNOT RUN on any artifact that exists … until then the honest state
> is REFUSED, not passed."*

**So it cannot be run on 09-03 today** — not because it replays, but because **no book on disk
carries the dropped set**; EV21 predates the split. It becomes runnable at the next build.
**One correction to the looser reading of "no replay": it still needs `pickle.loads` on a
307 MB book, which is heavy by rule 20's own definition — so it is lock-bound, not lock-free.**

## (4) ON FAILURE IT REFUSES — SO WIRING IT IS SAFE

`reconcile` raises `ReconcileRefused` **by name** on both checks; **six refusal sites in the
module and no flag-and-continue path.** That is the right default and it matches DE 168's own
rule that the default is a refusal and not a status.

**One consequence DE should know before wiring it:** the queued call site *assigns* the
result, so a reconciliation failure will **raise out of the day run** rather than produce a
flagged artifact. For a money-level identity that is correct — but it should be a deliberate
choice, not a surprise, and the refusal text should say which day and which book.

## ROUTED

1. **BE — the dropped total is unchecked.** `LEGS_DO_NOT_CLOSE` cannot fire; give the dropped
   value a second source, or restate the criterion as one equality.
2. **DE — wiring is SAFE (it refuses)**, and the failure mode is a raise out of the day run.
   Decide that deliberately.
3. **BE — describe `tol=1e-6` as a tolerance**, not as "to the digit"; the substance is right
   and the wording overstates it.
4. **Credit where it is due:** this module ships a falsifier, refuses by name, and its own
   cell already corrected the runnability claim. That is the pattern the last five rounds have
   been asking for.
