# ITER011 PREREGISTRATION — AMENDMENT A2

**STATUS: DRAFT-FOR-USER-FREEZE. Not in force. Drafted by BE 2026-09-01.**

Rule 4: frozen documents are amended only by the USER; a seat drafts and never
adopts. Nothing in this file changes any committed number, and the artifact
emits both forms until the USER rules.

## What this amends

`ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md` §5(2), frozen text:

> null = window-level sign-flip permutation of per-window paired differences,
> **≥1000 permutations, two-sided p.**

## The discrepancy, stated plainly

`phase2_iter011.sign_flip_null` adjudicates a **one-sided** p
(`alternative: "greater"`, `p_value = (ge + 1) / (n_perm + 1)`). The frozen
text says two-sided. Every committed 011 result was adjudicated one-sided.

**This is a provenance defect, not a result defect** — see "Impact" below.

## Why one-sided, and who decided

Not BE, and not silently:

- **R-286 (USER, 2026-08-28)** routed the gate-vs-table contradiction to
  GATE-GOVERNS: the adjudicated null is the GATE's text.
- **R-288** recorded the closure and stated that `p_two_sided` "stays as a
  REPORTED diagnostic so nothing citing it breaks".

The substantive reason is directional and it is the reason the gate reads
"BEATS": a two-sided test scores `|sum|`, so **a candidate LOSING by 120c earns
the same p as one WINNING by 120c** (measured: both 0.000500, both beside
`CELL_STATUS_OK`). Q4's gate asks whether the candidate beats the incumbent;
a two-sided p cannot express that question's direction.

## Impact — measured, not asserted

Recomputed by the reviewer over all six Q4 cells, with Holm re-run over the
full 24-cell family under the substitution:

| | one-sided (emitted) | two-sided (frozen form) |
|---|---|---|
| best Q4 p | 0.019990 | 0.049975 |
| best Q4 holm | 0.1199 | 0.2999 |
| survivors | the six Q1 cells | **the same six Q1 cells** |

**No verdict in the committed artifact changes under either form.** The
amendment matters PROSPECTIVELY: on a future run a one-sided p can pass where
the frozen two-sided p does not, and then the discrepancy is load-bearing.

## What is asked of the USER

**Ruling requested — one of:**

1. **AMEND §5(2) to one-sided**, with R-286/R-288 as the recorded cause,
   `p_two_sided` retained as a reported diagnostic (this is what the code does
   today and what the artifact now emits); or
2. **RE-ADJUDICATE two-sided**, restoring the frozen text as governing. The
   artifact already carries `p_two_sided_REPORTED_NOT_ADJUDICATED` on every Q4
   cell, so this needs no re-run — only a re-adjudication; or
3. **Something else.** BE does not choose between these.

## What BE did in code, pending the ruling

Nothing adjudicative. `p_two_sided_REPORTED_NOT_ADJUDICATED` is emitted beside
`p_value` on every Q4 cell, and each Q4 cell's detail states the discrepancy,
its cause, and that this amendment is a draft. R-288's promised diagnostic
occurred **zero** times in the previous artifact, so the frozen-form p could
not be recovered from the emission at all; it can now.

## Carried recommendation — the matched-random resolution, declared PROSPECTIVELY

**From the re-review's ruling on F-3 (2026-09-01), preserved here so it is not
lost between batches.** The reviewer ruled BE's refusal right on both halves:
A1.6's `n_perm = 2000` pins the **increment** null (§5(2), "≥1000 declared");
the matched-random null is §5(1) at **≥200**, which 500 satisfies. There was no
violated pin to repair, and raising 500 → 2000 after seeing that the survivors
sit one draw from failing is outcome-dependent and directional — a p at exactly
`1/(n+1)` can only move toward a smaller p as resolution rises, taking `holm`
from 0.0479 to at best 0.0120. That is rule 11.

**The recommendation, which costs nothing now:**

> **Declare the matched-random resolution PROSPECTIVELY — before the next run,
> for the next population.** This family stays adjudicated at **500 draws**,
> with the floor disclosure it now carries (every at-floor cell names its
> one-draw margin, and `draw_counts_are_not_uniform: [500, 2000]` is emitted).

A prospective declaration is not a repair of this family and must not be read
as one: it changes nothing here, and it removes the choice-after-seeing problem
from the next family by making the resolution a pre-registered fact.

## Provenance

- Raised by the reviewer (pm-codex seat) as **F-2** in
  `reviews/REVIEW_011_BATCH_AND_DE_BATCH_2026-09-01.md` at `0bf80f3`.
- Impact table reproduced independently by the reviewer.
- Drafted by BE under rule 4; **adoption is the USER's act, not this file's.**
