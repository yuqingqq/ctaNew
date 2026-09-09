# REVIEW 124 — finishing the enumeration list: one dirty, one clean, one that splits

**REV, 2026-09-09T08:52Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

**RESULT: ONE DIRTY, ONE GENUINELY CLEAN, AND ONE THAT SPLITS — loud at the window level and
silent at the row level, with the derivation for the silent half now available because of a
fix that landed for a different reason.**

| set | membership operation | built that way? | a miss is |
|---|---|---|---|
| `fit_code_files` (12) | the fitting run's import closure | **no** — typed | **SILENT** |
| `_GEN_REQUIRED` (7) | a field a consumer reads off a generation | **yes** | **LOUD** |
| exclusion vocabulary | every reason a row or window can be dropped | window level: effectively yes; row level: **no** | **LOUD / SILENT** |

---

## (1) `fit_manifest.json`'s TWELVE `fit_code_files` — DIRTY, AND THE DERIVATION DOES NOT EXIST YET

**Membership would be:** *a file whose bytes can change a fit artifact* — the **import closure
of the fitting run**.

**Built that way? No.** It is twelve typed module names (`flow_fill_development.py`,
`flow_intensity.py`, `harmful_action_eval.py`, `harmful_candidate_manifest.py`,
`harmful_exposure_rows.py`, `harmful_fast_compute.py`, `harmful_hazard_model.py`,
`harmful_state_features.py`, `phase2_arms.py`, `phase2_declaration.py`, `phase2_embargo.py`,
`phase2_state_schema_freeze.py`) beside a `fit_code_ref` commit and a
`fit_code_sha256_prefix`.

**A miss is SILENT**: the manifest verifies, `verify_head` passes, and the fit artifact is
served as the declared one while a file that shaped it has moved. Nothing recomputes what the
list omits.

**AND THE POINT THAT SEPARATES THIS FROM `SCORING_PATH_MODULES`: there is no closure to
derive from.** The book can be fixed by deriving from what it already records — 49 modules,
captured by `be_rule22.stamp` at import. **The fitting side records nothing of the kind:**
`harmful_candidate_manifest.py`, `phase2_arms.py` and `phase2_declaration.py` contain **zero**
references to `be_rule22`, `stamp` or `import_closure`.

**So the fix here is two steps, not one:** *record* the fitting closure at fit time, exactly as
rule 22 already does for the book — then *derive* the twelve from it and refuse on drift.
Until the first step exists, "is the twelve complete?" is not answerable by anyone, which is
why BE 113 could only find that surface by accident.

## (2) `_GEN_REQUIRED` — GENUINELY CLEAN, and here is the test

**Membership would be:** *a field a consumer reads off a generation.*

**Built that way? Yes, and it is checkable in one line.** The generation constructor
`harmful_stateful_policy._gen()` returns exactly

```
{"gen", "t0", "t1", "level", "displayed", "status", "tranches"}
```

and `_GEN_REQUIRED` is exactly those seven. **The set is not a sample of the object — it IS
the object's field set.**

And the consumers agree: sweeping the generation-holding modules
(`de_phase4_diag_runner`, `be_daybook_build`, `be_cancel_axis_null`, `da_de53_exclusion`,
`de_section81_arms`, `da_elementwise`, `be_score_coverage`) for fields read off a generation
gives `t0`, `gen`, `tranches`, `t1`, `level`, `status` — **all seven required, and nothing
outside the set** (the other names in that sweep, `side`, `slug`, `hour`, come from the
enclosing loop, not the generation).

**A miss would be LOUD** on both paths: `validate_reference` iterates `_GEN_REQUIRED` and
raises — *I tripped it myself this session*, `ReferenceIntegrityError: s1/BUY_UP: generation
missing 't1'` — and a field read but not required fails at the read with a `KeyError`. **The
`.get()` escape, which is what would make a miss silent, does not occur on a generation field
outside the required set.**

**This one needs nothing. Stating that plainly is the result.**

## (3) THE EXCLUSION-STATUS VOCABULARY — IT SPLITS, AND THAT IS THE USEFUL ANSWER

**Membership would be:** *every reason a row or a window can be dropped.*

**At the WINDOW level it is effectively built that way — because a DENOMINATOR checks it.**
`da_book_verify.py:816` types `excl = ("BINANCE_GAP_EXCLUDED", "NO_REPLAY",
"RECONCILIATION_FAILED")`, but the predicate it feeds is

```
admitted_plus_excluded == windows
```

**A missing reason makes the sum too small and the identity FALSE; a silent drop with no
status at all does the same.** The denominator makes the vocabulary's completeness checkable
without enumerating it — so at the window level **a miss is LOUD**, and this is the same
virtue as rule 20's: a property, not a list.

**At the ROW level there was no denominator, and a miss is SILENT.** The feature pass returns
`drops` as a counter **by reason, aggregate for the whole coin** (REVIEW 107), so a row
dropped for an unnamed reason simply is not there and no sum notices.

**AND THE DERIVATION NOW EXISTS, from a fix that landed for a different reason.**
`rows_in_by_generation(split_of)` — which DE 158 added because REVIEW 107 refuted the claim
that the count existed nowhere — gives rows-IN per generation, and `kept` gives rows-OUT. So:

```
rows_in(g) - kept(g)  ==  sum of the NAMED drop reasons for g       (per generation)
```

**Any residue is an unnamed drop, by construction.** That is the row-level analogue of
`admitted + excluded == windows`, it uses only what the pipeline already computes, and it
turns the row-drop vocabulary from a list into a checked property.

**On DA 147's `BINANCE_GAP_EXCLUDED` hardcoded zero:** *historically* an instance, and
**currently repaired the right way** — `be_gate1_fragment.py` now carries
`BINANCE_GAP_EXCLUDED_BY_THIS_SELECTOR = 0` **beside**
`BINANCE_GAP_EXCLUDED_STATUS = "NOT_APPLIED_ON_THE_DAY_PATH"`, so the zero travels with a
status distinguishing *none found* from *not applied*. That is rule 4's own distinction
(an unrecorded quantity and a zero one are the same number and opposite facts), and it is the
fix rather than the defect.

## SCOPE

Closed over: the three sets this round named, read at the tip; `_GEN_REQUIRED` checked against
its constructor and swept across the seven generation-holding modules; the exclusion
vocabulary checked at its typed site and at the identity that consumes it. **Not closed
over:** the fitting closure itself (it is not recorded, which is the finding), and every other
typed set in the package beyond the six now examined across REVIEW 123 and this one.

## ROUTED

1. **BE — record the fitting import closure at fit time** (rule 22's pattern, already proven
   on the book), then derive `fit_code_files` from it. **Two steps, and the first is the one
   that makes the question answerable at all.**
2. **DE / DA — the row-level residue test**: `rows_in(g) − kept(g) == Σ named drop reasons`,
   using `rows_in_by_generation` which already exists. It closes the silent half of (3).
3. **`_GEN_REQUIRED` needs nothing**, and I say so with the test rather than by omission.
4. **`BINANCE_GAP_EXCLUDED` needs nothing** — the zero already carries its status.
