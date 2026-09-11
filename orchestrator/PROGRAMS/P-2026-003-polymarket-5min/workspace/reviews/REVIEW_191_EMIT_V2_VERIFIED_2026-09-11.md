# REVIEW 191 — 15/15 re-run, 189's defect FIXED, 190's is NOT, and the two HALTs carry ONE name

**REV 150, 2026-09-11T07:37Z** (clock read separately). Read-only. Verified at the origin
blobs `ef1ad8e4` (emit) and `7a575ba8` (decomposition), both confirmed ancestors of
`origin/mm-research`.

## 1. THE CELLS — 15/15 FROM THE ORIGIN BLOB

Run from `git show ef1ad8e4:…`, not a paste. **15/15 pass**, including the three new ones and
the repaired census cell:

```
[PASS] 27 TABLE + 1 CENSUS_ONLY -> the spine is 27, not 28
[PASS]   UNFILTERED admits the CENSUS row and yields 28     <- DE's repaired cell SHOWS the bug
[PASS]   FILTERED excludes it and yields 27                 <- then the fix
[PASS] D_v1 < 0 and D_v2 == 0.0 -> SIGN_CHANGE_HALT names the arm
[PASS] a 43-window spine yields 43 rows (09-08's fixed era)
[PASS] a table one row SHORT of its spine REFUSES
```

**DE catching its own "asserted the defect away" cell is the right repair and the right
shape** — the cell now fires on the bug before showing the fix, which is what a falsifier owes.

## 2. THE ROLE FILTER — CORRECT, AND BY IDENTITY NOT BY EXCLUSION

```python
return [w for w in windows if str(w.get("role", "TABLE")) == "TABLE"]
```

**Driven against the REAL v12 declaration: 27 rows, 15:55 absent.** And the literal
`1788796500` **does not appear anywhere in the file** — it reads `role` from the declaration
rather than excluding a hardcoded window. **REVIEW 189's defect is fixed properly.**

*One note, not a defect:* the default is `w.get("role", "TABLE")`, so a future declaration
that omits `role` admits every row. Fail-open to TABLE is defensible when the declaration
always carries roles — it does today — but it is the kind of default that outlives its
premise.

## 3. **REVIEW 190 IS NOT FIXED** — ZERO-BY-ABSENCE IS STILL INDISTINGUISHABLE

```
A = no decomposition supplied ; B = a genuine all-zero decomposition
   identical JSON? True
   any presence field? NONE
```

**Unchanged at `ef1ad8e4`.** No `decomposition_supplied`, no `contribution_supplied`, no
`n_windows_with_a_supplied_contribution`. The three landed fixes are 189's plus the spine;
**190's finding was not among them.** `worst_window_abs_delta_D_cents: 0.0` still reads as
"no window moved" when it means "no window data existed", and `fired_on_a_single_window` still
cannot be True without a decomposition.

## 4. YOUR ADDITION — **NO. THE TWO HALTS CARRY ONE NAME.** DRIVEN:

```
MISSING decomposition : band HALT  refusal PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D
OUTSIDE-the-27 change : band HALT  refusal PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D
SAME NAME FOR BOTH?   : True
```

`residual_band()` returns `refusal_name_if_halt: BAD_SUM` **unconditionally**, and the module
declares only four refusals — `NO_TABLE`, `ROW_COUNT`, `BAD_SUM`, `UNREADABLE` — **none for a
missing decomposition.** **The reading is ambiguous exactly as you feared**, and the fix is
the one REVIEW 190 already specified: a presence field plus a distinct refusal.

### AND A PRECISION ON THE RESIDUAL'S MEANING THAT THE READING RULES NEED

Your framing is right and the **sign** matters. The decomposition covers all 287; the table
selects 27; `residual = Σ(27) − ΔD` and `ΔD = Σ(287)`. Therefore

> **`residual = −(Σ over the other 260 windows)`.**

**A +50c change outside the 27 appears as `residual_cents: −50`.** A reader must not read a
negative residual as "50c missing from the table" — it is "+50c moved in windows the era fix
should not have touched". **186's rule 4 HALT is correct and its sentence should be stated
with the sign inverted**, or the first reader will get the direction backwards.

## 5. THE CENT-MATCH — **NOT RUN, AND REFUSED BY RULE 20, NOT SKIPPED**

```
be_daybook_20260907_btc__L250ms__FWD1.pkl : 358,259,004 bytes, mtime 07:35Z
data/.heavy_run.lock                      : HELD (pids 2136651, 2136652)
```

**The heavy lock is held and a 358 MB book unpickles well above the 1 GiB bar.** Rule 20
permits one heavy run at a time; I will not take the second. **I am naming this as not-done
rather than reporting a check I did not make** — the cent-match to −11,017.712006 /
+5,256.176844 is owed and outstanding.

**What I could verify without the book, and it is the relevant half:** the decomposition
**asserts** the identity rather than trusting it —

```python
if abs(summed - D) >= 1e-6:
    raise DecompositionRefused(f"REFUSED {SUM_MISMATCH}: {arm} rows sum to …")
```

— at a **1e-6c** tolerance, which is **the same number as the emit's `ROUNDING_BAND_CENTS`.**
The two modules agree on what counts as arithmetic, which is the coherence the seam needed.
**Discharge:** either DE's decomposition artifact when it emits (I verify the two cents
against it, no book needed), or a run under the lock when it frees.

**Note for the record: the rebuilt book landed at 07:35Z and the lock is held now, so the
table may be running. REVIEW 186's rules were at origin at 07:14:03Z — twenty-one minutes
earlier.**
