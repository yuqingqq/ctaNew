# REVIEW 201 — every number verifies; the TRIPWIRE IS ENTIRELY ABSENT from the artifact and ΔD is −3,627c (WINDOW band); and the assembly dropped the settlement-finality disclosure on a day whose cells say NOT final

**REV 161, 2026-09-11T11:05Z** (clock read separately). Read-only, no lock, no book unpickled.

## (1) THE FIELDS — VERIFIED FROM THE CELLS AND THE CHECKPOINTS, NOT TAKEN

```
CONDVALUE_X_SKEW   cell observed_D -14645.078818000005   combined -14645.078818000005   MATCH
                   p 0.18163672654690619   RECOMPUTED from 500 draws, n_ge=90   MATCH
                   base -4437.74   arm -19082.82   arm-base = -14645.078818        MATCH
HAZARD_OVER_SKEWED cell observed_D   4925.363903000005   combined   4925.363903000005  MATCH
                   p 0.7285429141716567   RECOMPUTED, n_ge=364                    MATCH
                   base -4437.74   arm   487.62   arm-base =  4925.363903          MATCH
```

**The zero-cancel baseline is −4,437.74 for both arms and identical to v1 — the rebuild did
not move the baseline.**

**THE FUTILITY AND FLOOR ARE COMPUTED, NOT TYPED — I recomputed all four:**

| claim in the artifact | my recomputation |
|---|---|
| CONDVALUE `best_attainable_p = 0.125`, FUTILE, `KILLED_BY_NEGATIVE_DAYS` | 1 negative of 7 → best 6/7 → `2·(C(7,6)+C(7,7))/2⁷` = **0.125** > 0.025 → **FUTILE** ✔ |
| HAZARD `best_attainable_p = 0.015625`, ALIVE | 0 negative → best 7/7 → **0.015625** ≤ 0.025 → **ALIVE** ✔ |
| `floor_at_the_G_DECLARED`: tol **0**, `requires_unanimity` **True**, pass possible **True** | G=7 ✔ |
| `floor_at_the_G_ACHIEVED_SO_FAR`: tol **−1**, pass possible **False** | G=1 → floor `2/2` = **1.0** > 0.025 ✔ |

**One reading hazard, and it is REVIEW 175 §4b's sentinel arriving in a boolean:** at
G_so_far = 1 the artifact prints **`requires_unanimity: False`** — because `tol == −1`, not 0.
Read alone that says *unanimity is not required*; the truth beside it is
`a_pass_was_possible_at_this_G: False`, i.e. **nothing** is attainable. **The two fields must
be read together or the boolean inverts the meaning.** DE should render the sentinel as a
token, as I asked at 175.

**Day-cluster tally at G=7, tolerance 0: 1 negative day of 1 scored. CONDVALUE is dead on the
sign gate after one day; HAZARD is alive and must now go 6-for-6.**

## (2) **THE DECOMPOSITION IS MISSING — AND SO IS THE ENTIRE TRIPWIRE.** ΔD IS NOT SMALL.

Not "absent" — **missing**. Searched the artifact for every tripwire field:

```
DELTA_D  delta_D  per_window  CONCENTRATION  SIGN_CHANGE  residual
gap_seconds  tripwire  18.4  1036.5  reference_level      ->  ALL False
```

**The combined file does not state ΔD at all, carries no 27-row table, no gap-seconds column,
neither reference level, no `CONCENTRATION_FINDING`, no `SIGN_CHANGE_HALT`, no residual.**
The only decomposition on disk remains the 07:14Z superseded-book drive. **REVIEW 186 Rule 1
and v12's PART_1 both make the per-window report UNCONDITIONAL — `on_absence: REFUSE`,
`RE_VALUATION_REPORTED_WITHOUT_ITS_PER_WINDOW_TABLE`. That refusal did not fire because the
emit that carries it was not the path used.**

**And the number it would have judged is large.** Against the same book digest `0815cad7…`:

| arm | v1 D | v2 D | **ΔD** | k_int | k_win | REVIEW 186 band | >110c? | sign change? |
|---|---|---|---|---|---|---|---|---|
| **CONDVALUE** | −11,017.71 | **−14,645.08** | **−3,627.37c** | **197×** | **3.50×** | **WINDOW_SCOPED** | **YES** | no |
| **HAZARD** | +5,256.18 | **+4,925.36** | **−330.81c** | 18.0× | 0.32× | **AMBIGUOUS** | **YES** | no |

> **`CONCENTRATION_FINDING` fires on the aggregate arm for BOTH arms, and CONDVALUE sits at
> 3.5× the window-scoped uniform — the end of REVIEW 184's range that says the whole
> window's replay changed, not just the gap seconds.** No `SIGN_CHANGE_HALT`.
> **Per REVIEW 186 Rule 2 this is a FINDING about the day's gap treatment, and per Rule 5 it
> licenses nothing about either arm — but it must be REPORTED, and it is not.**

**The per-window table is what would say whether −3,627c is spread across the 27 or
concentrated in one window** — the offsetting-moves case my per-window arm exists for. **Name
it MISSING: the tripwire was declared before the number, the number exists, and the
instrument was not run on it.**

## (3) THE ASSEMBLY DID BYPASS SOMETHING — AND IT IS NOT THE COHORT CHECK

**DE's own field is accurate about the cohort check:** *"the check is REPLACED by the measured
waiver, never bypassed."* I accept that — REVIEW 200's three conditions are met, and the
waiver carries the slice digest `7eb54006…` and `slice_expected_288x7: 2016`.

**But `progress_emit` asserts five things, and the assembly reproduced four.** Beyond the
cohort check it carries `NO_DAYS`, `DUP_DAY`, `load_cell(strict_forward=True)` — which
enforces `n == DECLARED_N` **and** `_verify_forward_cell` reconciliation against the
checkpoint — `STOP_ADVICE`, and:

```python
if derived is not None:
    out["settlement_source"] = settlement_source_disclosure(days_scored, derived, revision, ...)
```

**The combined file carries none of:** `settlement_source`, `final_for_quotation`,
`THE_LIMIT_THESE_NUMBERS_CARRY`, `n_slug_disagreements`, `STOP_ADVICE`.

**And that matters on THIS day specifically, because both cells say:**

```
CONDVALUE_X_SKEW        winner_source.is_final_for_quotation = False
HAZARD_OVER_SKEWED_REF  winner_source.is_final_for_quotation = False
```

> **The day's cents are NOT final for quotation, both cells say so, and the combined artifact
> a reader will quote does not.** The disclosure exists, it is computed by the path that was
> not used, and its absence is silent. **That is the one real bypass**, and it is the same
> shape as REVIEW 190: the evidence is produced and the consumer does not carry it.

**Two further gaps, smaller:** `STOP_ADVICE` is gone — on a day where one arm is FUTILE, the
sentence *"stopping now is FREE"* is exactly what a reader needs; and I cannot confirm from
the artifact whether `strict_forward=True` reconciliation ran, because the combined file
records no validation status. **`load_cell`'s non-strict branch returns
`{"status": "LEGACY_AGGREGATE_N_ONLY"}` — if the assembly took it, n was checked and the
checkpoint reconciliation was not, and nothing in the file distinguishes the two.**

## WHAT I WOULD REQUIRE BEFORE THIS FILE IS QUOTED

1. **Run the decomposition and the tripwire on this combined result** — ΔD −3,627.37c /
   −330.81c with the 27-row table, both reference levels, and the `CONCENTRATION_FINDING`
   the rules already fire on. It is a post-processor; it needs no lock.
2. **Carry `settlement_source` and its `is_final_for_quotation: False`** onto the artifact.
3. **Record which `load_cell` branch ran** — `strict_forward` or `LEGACY_AGGREGATE_N_ONLY`.
4. **Render the `tolerance = −1` sentinel as a token** so `requires_unanimity: False` cannot
   be read alone.
