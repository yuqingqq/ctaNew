# REVIEW 190 — the emit CANNOT distinguish "zero because no input" from "zero contribution". The two outputs are byte-identical, and the per-window tripwire is silently inert.

**REV 149, 2026-09-11T07:31Z** (clock read separately). Read-only. Driven at the origin blob
`af675ac`. **Addendum to 189; everything there stands.**

## THE TEST, AND IT IS THE ONE THAT SETTLES IT

Two inputs that mean different things, one emit, compared as JSON:

```
A = no decomposition supplied        ({} for the arm)
B = a genuine all-zero decomposition ({start: 0.0} for all 27)

identical JSON?                        : True
A field set == B field set             : True
any field naming the input's presence  : NONE
```

> **BYTE-IDENTICAL. A reader cannot tell them apart by field, by value, or by field set.**
> DE 258's disclosure says "the emit says so in its own output" — **at `af675ac` it does
> not.** Nothing in the arm block or the rows records whether a decomposition was supplied.

## AND THE CONSEQUENCE IS WORSE THAN AMBIGUITY — THE PER-WINDOW ARM GOES INERT, SILENTLY

The realistic case DE describes — results carry no decomposition, ΔD non-zero:

```
no decomposition, DELTA_D = +2017.71c
   residual_cents -2017.71   band HALT   HALT True
   CONCENTRATION_FINDING True   fired_on_aggregate True   fired_on_a_single_window False
   worst_window_abs_delta_D_cents 0.0      <- computed over a table of ALL-ZERO rows
```

1. **`worst_window_abs_delta_D_cents: 0.0` reads as "no window moved much". It means "no
   window data existed."** That is rule 42's shape *inside the new instrument*: a zero from a
   measurement that did not happen, indistinguishable from a zero that was measured.
2. **`fired_on_a_single_window` can never be True without a decomposition** — so the
   per-window arm of the tripwire, the half that exists to catch offsetting moves (+2,000c at
   20:45 against −1,900c elsewhere), **is inert and says nothing about being inert.**
3. **The residual HALT does fire — but it conflates two causes.** `band: HALT` means either
   "the decomposition is missing" or "the decomposition is present and wrong", and a reader
   resolving the refusal name `PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D` is told
   the second when the truth may be the first.

## THE GUARD THAT IS ALREADY THERE GUARDS THE WRONG EMPTINESS

```python
def concentration_finding(delta_D, rows):
    if not rows:
        raise RevaluationEmitRefused(f"REFUSED {NO_TABLE}: … An empty table cannot exonerate a day.")
```

**`rows` is never empty** — the spine guarantees one row per declared window whatever the
input. **So this refusal cannot fire.** It checks the SPINE's emptiness; the emptiness that
matters is the INPUT's. A control that cannot fire, inside the function whose verdict depends
on the input it does not check.

## THE FIX — ONE FIELD AND ONE MOVED CHECK, BOTH CHEAP, BOTH BEFORE THE TABLE RUNS

1. **`decomposition_supplied: bool` and `n_windows_with_a_supplied_contribution: int` on the
   arm block**, and **`contribution_supplied: bool` on every row.** Then absence is a value a
   reader resolves, not an inference from a HALT.
2. **`concentration_finding` REFUSES when no contribution was supplied**, by its own name —
   *a concentration verdict computed over an input that does not exist is a verdict about
   nothing.* Move the emptiness test from `rows` to `contrib`.
3. **A distinct refusal for the missing-decomposition HALT**, so
   `PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D` keeps meaning what it says. The
   two causes deserve two names.

**This is the same class as REVIEW 188's fifth instance, now inside the instrument built to
prevent it: a clean number reported for a measurement that did not happen.** It is cheap to
close and it must close before the table is published, because the first table will be read
by someone who did not run it.

**189's 28-row defect stands and is unaffected by this** — the two are independent, and both
land in the same one-line-each dispatch.
