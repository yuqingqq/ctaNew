# REVIEW 145 — do today's other checks falsify on every side they claim?

**REV, 2026-09-09T15:32Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. **Every line below is a drive, not a cell list.**

**VERDICT: ALL SIX ARE CLEAN — twenty-three sides driven, twenty-three anchored. THE
RECONCILIATION WAS THE OUTLIER, NOT THE PATTERN. And two of the six close findings I raised
earlier tonight: BE 130 now REFUSES the non-finite bounds that REVIEW 138 found being dropped
as zero-length, and DE 169 has made a FORGED derived set inert — it no longer moves the
predicate at all, which is stronger than the cross-check I asked for.**

---

## 1. ZERO-LENGTH GENERATION EXCLUSION (BE 129/130) — 4 sides, 4 anchored

```
A  t0 == t1                -> DROPPED, counted
B  t0 <  t1                -> KEPT
C  inf/inf, nan/nan, None/None, KEYS MISSING, '5'=='5'
                           -> REFUSED GENERATION_BOUNDS_ARE_NOT_FINITE_NUMBERS
C  t0 > t1                 -> KEPT and NAMED (INVERTED_GENERATION), left for validate_reference
D  50 of 50 zero-length    -> REFUSED ZERO_LENGTH_GENERATIONS_ARE_NOT_A_BOUNDARY_CASE
```

**REVIEW 138's finding is closed at its cause.** The finiteness test now runs before the
equality, so the five malformed shapes that were silently excluded as "zero-length" are
refused by their own name.

## 2. BOOK-CODE PREDICATE (DE 168/169) — 3 sides, 3 anchored

```
clean control                     -> MATCHES / match=True / n=12 of 12
A wrong digest on a NAMED member  -> REFUSED BOOK_BUILT_BY_DIFFERENT_SCORING_CODE
B a member UNNAMED                -> MATCHES_WITH_UNNAMED_MEMBERS / match=False / 11 of 12
C the SET forged to 2 of 12       -> MATCHES / match=True / n=12 of 12   <-- forgery INERT
```

**Side C is anchored by construction rather than by a check**, and that is the better answer:
DE 169 recomputes the set from the code on disk and does not read the receipt's
`derived_closures` at all, so a forged block **cannot move it**. **My REVIEW 132 finding —
"the expected set is the receipt's own and nothing recomputes it" — is closed.**

## 3. `assert_pin_sites_agree` (BE 111) — 5 sides, 5 anchored

```
clean control                          -> ADMITTED
A entry module has no path             -> REFUSED ENTRY_MODULE_ABSENT
B entry path absent from the cascade   -> REFUSED ENTRY_MODULE_NOT_IN_CASCADE
C the two sites disagree               -> REFUSED PIN_SITES_DISAGREE
D both agree but are STALE on disk     -> REFUSED PIN_DOES_NOT_MATCH_DISK
   (require_on_disk=False relaxes ONLY D -> ADMITTED)
E a payload carrying a CLOSURE pin      -> REFUSED SITE_LIST_STALE_PARAMS_CARRY_A_CLOSURE_PIN
```

**Side E is the interesting one and it is the one I got wrong first** (below). It is
*computed*, not read from a literal key — `_closure_identity_sites` walks the payload — so it
fires on a `producing_code` block and on a bare `import_closure` block alike. **That is rule 15
applied to a site list: the guard refuses when its own enumeration goes stale.**

## 4. `committed_state` (DA 156) — 2 questions, both answer

```
own tree    -> COMMITTED_IN_THIS_TREE
AUDIT_ROOT  -> NOT_IN_THIS_TREE
the two disagree IFF the file is outside AUDIT_ROOT -> True
```

Verified again after REVIEW 133; the disagreement is itself the asserted property.

## 5. CANCEL-COUNT INSTRUMENT (DE 159) — 3 sides, 3 anchored

```
per-ROW dicts (the real shape) -> ASSEMBLY_IS_PER_ROW
bare FLOATS (the pre-fix shape)-> REFUSED ASSEMBLY_PREDATES_CAUSAL_SCORING
EMPTY assembly                 -> REFUSED CANCEL_DELTA_NO_ASSEMBLED_SCORES
```

and `_cancels` keys on `int(c["ref_gen"])` — **the reference id space**, which is the space
REVIEW 125 established the claim needs.

## 6. MATCHED-CANCEL CONTROL (DE 160) — 5 sides, 5 anchored

```
A demand within the stratum     -> 2 drawn, times randomised within the generation
B demand EXCEEDS the stratum    -> REFUSED MATCHED_CONTROL_STRATUM_SMALLER_THAN_DEMAND
C demand on an ABSENT stratum   -> REFUSED (the same name; treated as size 0, never clamped)
D demand zero                   -> 0 drawn
E n_draws below the minimum     -> REFUSED MATCHED_CONTROL_BELOW_DECLARED_MIN_DRAWS
E' at the minimum (200)         -> 200 draws
```

**C is worth naming**: an absent stratum refuses rather than silently contributing nothing —
the shape that would otherwise let a control quietly answer an easier question.

## WHY THE RECONCILIATION WAS DIFFERENT, IN ONE LINE

Every check above tests a property of **one object** against a rule, so each side is a
different way for that object to be wrong. The reconciliation tests an **identity between
three quantities**, two of which it computes from the same source — and an identity whose
terms are not independently sourced has fewer sides than it appears to. **That is the
distinguishing feature to look for, not "does it ship a falsifier".**

## MY OWN PROBE WAS WRONG TWICE AND I RECORD BOTH

I first drove side E with a literal `closure_sites` key and it ADMITTED — the trigger is
**computed** by `_closure_identity_sites`, so my probe tested a key nothing reads. Re-driven
with a real `producing_code` block, it refuses. And my first `draw_one` pool used ints where
the function wants `{key: [times]}`, producing a `TypeError` I could have filed as a defect.
**Both were caught by their own controls; every number above is the re-drive.**

## SCOPE

Closed over **the sides each check NAMES** — its own refusal codes and documented faults —
driven one at a time with a clean control beside each. **Not closed over** a side nobody has
named: the same limit as REVIEW 140's scope, and the reason the reconciliation's missing side
was visible only because DA had specified two equalities in writing.

## ROUTED

1. **Coordinator — the five load-bearing checks are two-sided.** Named above with their
   drives, so the wiring decision does not rest on my summary.
2. **Nothing to BE or DE from this sweep.** BE 130 and DE 169 closed my earlier two findings
   in passing, and I record that here rather than leaving them open in my ledger.
