# REVIEW 137 — `de_reference_integrity_probe.py` (`c379953`), the instrument behind the nine

**REV, 2026-09-09T12:40Z.** Read-only, and deliberately so: **DA holds the lock for a full
book verification, so I unpickled NOTHING.** Every drive below is on constructed input plus
the probe's own emitted artifact (`be_heavy_run_stdout_be129probe.log`).

**VERDICT: THE NINE STAND. I built the falsifier the probe does not ship, and it fires on all
three kinds and stays silent on the clean control; "zero-length" IS the criterion and
"zero-tranche" is a co-occurring property, not the test; and `--against` does compare like
with like — provably, from the artifact. THREE DEFECTS IN THE INSTRUMENT, NONE OF WHICH MOVES
TONIGHT'S NUMBER: it ships NO falsifier at all (rule 15); `t0 == t1` is tested BEFORE
finiteness, so `None`/`None`, MISSING KEYS and equal STRINGS are all reported as
`ZERO_LENGTH`; and a side outside `HSP.SIDES` is dropped from the total silently. And it
CANNOT answer the 35.**

---

## (1) IT SHIPS NO FALSIFIER — SO I BUILT ONE

`--selftest`: **absent.** `ok(`/assert cells: **none.** 123 lines, no positive control, no
known-bad. **Rule 15 is not met**, and a census never shown to fire cannot support a count of
nine. Driven, one generation per case:

```
clean t0<t1          -> NOT_COUNTED      <- the positive control that must stay silent
ZERO LENGTH t0==t1   -> ZERO_LENGTH
INVERTED t0>t1       -> INVERTED
nan t1               -> NON_FINITE
inf t1               -> NON_FINITE
bool t0 (True==1)    -> NON_FINITE       <- correct; `_finite` excludes bool
```

**The instrument fires on each fault it claims to separate, and does not fire on a clean
generation.** That is the evidence the count of nine needed and did not have.

## (2) THE CRITERION IS ZERO-*LENGTH*, NOT ZERO-*TRANCHE* — AND THE DISTINCTION MATTERS

```
gen 1: t0=10 t1=20, ZERO tranches   -> NOT counted
gen 2: t0=30 t1=30, FIVE tranches   -> COUNTED, ZERO_LENGTH, n_tranches_on_them = 5
```

**A generation with no tranches and a good window is not counted; one with five tranches and
`t0 == t1` is.** So `n_tranches_on_them` is a *property reported about* the refused set, never
the test. **The brief's phrase "nine zero-tranche generations" should be "nine ZERO-LENGTH
generations, which happen to carry no tranches"** — a repair aimed at generations-without-
tranches would target the wrong property.

**And on the real nine the co-occurrence is exact and worth handing on:** all nine carry real
finite floats with `t0 == t1` to the last digit, across **7 slugs and 7 DISTINCT instants** —
two slugs fail on **both sides at the identical instant** (`266.305084409`, `119.221239236`).
That is one tape event per affected slug, not nine independent ones.

## (3) `--against` COMPARES LIKE WITH LIKE — AND THE ARTIFACT PROVES IT

Structurally, both branches call **the same `reference_census` on the same key**,
`bk["fr"]["reference"]`. The per-row/per-generation difference lives in `asm["by_arm"]`, which
`assembly_shape` resolves **by the value's TYPE and explicitly not by a count**.

**And the emitted artifact settles it as a measurement rather than a reading:**

```
corrected book   assembly 350,474 entries  PER_ROW_SCORES   |  reference census 313,149
pre-fix book     assembly 297,379 entries  PER_GENERATION   |  reference census 313,114
```

**If `fr.reference` had become per-row, its census would agree with 350,474 — it does not.**
The reference is per-generation in both books, so 313,149 against 313,114 is a like-for-like
count. The contamination the round was worried about is not present.

## (4) IT CANNOT ANSWER THE 35, AND HERE IS WHY

`reference_census` returns totals plus `first_20` — **no per-generation key set**, so the two
books' generation sets cannot be differenced. The 35 is a count difference; the probe cannot
say whether the nine are among the 35 new generations or were present in the pre-fix book with
`t0 < t1`. **Those are different findings and the instrument cannot separate them.**

**One line closes it:** emit the sorted `(slug, side, gen)` keys (or a per-slug count) for
both books, and the set difference answers both the 35 and whether the nine are new.

## THE THREE DEFECTS, SCOPED

1. **No falsifier** (rule 15). Ship the cells above.
2. **`"ZERO_LENGTH" if t0 == t1` is evaluated BEFORE finiteness.** Driven:
   `None`/`None` → `ZERO_LENGTH`; **t0/t1 keys MISSING** → `ZERO_LENGTH`; `"5"`/`"5"` →
   `ZERO_LENGTH`. **A generation with no timestamps at all is reported as a zero-length
   window.** *This does not touch tonight's nine* — all nine carry real finite floats, read
   from the artifact — but the claim "none non-finite" rests on a classifier that cannot say
   "absent". Test finiteness first.
3. **A side outside `HSP.SIDES` is silently skipped.** Driven: a reference of three
   generations, two of them bad on a third side, reports `n_generations = 1, n_refused = 0` —
   **the bad ones invisible and the total not saying so.** Latent only: I could not check the
   real book's side set without unpickling, and I did not.

## ROUTED

1. **DE — the three defects above**, and the one-line key-set emission that answers the 35.
2. **BE — the diagnostic lead:** 7 slugs, 7 distinct instants, two of them failing on **both**
   sides at the identical `t`. `build_reference` is producing one zero-length window per
   affected slug-instant, not a scatter.
3. **The nine stand as measured**, and DE's refusal to weaken `validate_reference` is the
   right call on this evidence.
