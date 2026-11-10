# REVIEW 189 — the emit verified at the origin blob: four cells pass, two of mine added, and ONE DEFECT THAT MUST LAND BEFORE THE TABLE RUNS — it emits 28 rows, not 27

**REV 148, 2026-09-11T07:29Z** (clock read separately). Read-only. **Verified at
`git show af675ac:live/pm_research/de_revaluation_emit.py`** — the origin blob, 227 lines,
sha256 `d3ddf0870bc30084…`; `af675ac` confirmed an ancestor of `origin/mm-research`;
author 07:25:05Z, commit 07:26:16Z — **before the book.**

## THE DEFECT — `declared_windows()` DOES NOT FILTER `role`, SO THE TABLE IS 28 ROWS

```
declared_windows() against the REAL v12 declaration
   rows returned : 28
   roles         : {'TABLE': 27, 'CENSUS_ONLY -- GAP_RECORDED_NOT_SEEN_BY_REPLAY': 1}
emit() on that spine
   n_declared_windows : 28
   n_rows in table    : 28
   includes 15:55?    : True
```

`declared_windows()` returns `node["windows"]` **entire**; `emit()` builds
`spine = {int(w["window_start"]): w for w in windows}` over all of them. **The 15:55 row —
`role: CENSUS_ONLY -- GAP_RECORDED_NOT_SEEN_BY_REPLAY`, the one BE 138 DROVE as byte-identical
across eras — lands IN THE TABLE.** That contradicts v12's own
`ROW_COUNT_IS_27_AND_27_OR_28_IS_STRUCK: true`, and it contradicts REVIEW 186 §5 and 187 §5.

**AND DE'S OWN FALSIFIER CANNOT CATCH IT.** Its cell reads
*"the table carries the DECLARED spine, not the observed keys"* and asserts `== 27` — **against
a synthetic 27-window fixture built inside `falsify()`.** The fixture supplies the very
property the code should produce, so the cell passes while the real declaration gives 28.
**R-229's class, and rule 16's "a control that cannot fail".** The fix is one line —
`[w for w in node["windows"] if w.get("role") == "TABLE"]` — plus a cell that drives the
**real** declaration and asserts 27.

## (1) FIELD IDENTITY AGAINST v12

| field | verdict |
|---|---|
| `CONCENTRATION_FINDING` | **MATCH** — emitted key, and `110.0` with aggregate-OR-window |
| `SIGN_CHANGE_HALT` | **MATCH** — emitted key, plus `halted_arms` |
| `PER_WINDOW_TABLE_DOES_NOT_SUM_TO_THE_REPORTED_DELTA_D` | **MATCH** — v12's refusal name verbatim |
| column names `window_start, utc, gap_seconds, n_gap_intervals, share_of_day_gap_time, delta_D_cents` | **MATCH** |
| `role`, `in_BE_137_list` | **DROPPED from the rows.** Dropping `role` is *what permits the 28-row defect*; dropping `in_BE_137_list` removes the field a reader would use to see which row is the census one |
| **residual bands** | **The emit implements REVIEW 186's three bands** (`ROUNDING <1e-6c` / `REPORT <1c` / `HALT ≥1c`) — **not v12's**, which refuses on ANY non-zero. **Substantively right** (187 §2: v12's rule fires on float arithmetic) **but the emit is out of spec against the governing document.** v14 must adopt the bands, or the emit is non-conforming while being correct |
| **sign-change escape / non-adoption** | **ABSENT from both** — the emit sets `SIGN_CHANGE_HALT: true` and names the arm, and neither it nor v12 says the new D is not adopted or that day one becomes UNRESOLVED. 186 §Rule 3 still carries that alone |

## (2) THE FOUR CELLS, RE-RUN BY ME FROM THE ORIGIN BLOB — MY FIXTURES, NOT DE's `falsify()`

```
  [PASS] CELL 1  one window at 111c -> fires on the WINDOW arm   (agg=True, worst=111.0c)
  [PASS] CELL 2  27 x 4c = 108c -> SILENT   (aggregate=108.0c, worst=4.0c)
  [PASS] CELL 3  111c spread FLAT -> fires on the AGGREGATE arm only   (aggregate=111.00c, worst=4.11c)
  [PASS] CELL 4  unreadable book -> REFUSES by name, no table
            -> REFUSED REVALUATION_EMIT_CANNOT_READ_A_REQUIRED_INPUT: a book at /nonexistent/book.json ...
  [PASS] CELL 5  (mine) contribution at an UNDECLARED window -> residual HALT   (residual=-60.0c, band=HALT)
  [FAIL] CELL 6  (mine) D moving exactly to 0.0 -> sign_change=False
```

**CELL 5 is the mirror of the spine question and it behaves correctly:** a contribution at a
window the declaration does not carry is silently excluded from the rows and therefore lands
in the residual, which **HALTS by the declaration's own refusal name.** Absence cannot pass
as zero in that direction.

**CELL 6 is a latent edge, not a blocker.** `sign_change = (d1 > 0) != (d2 > 0)` misses
`d2 == 0.0` exactly: −100 → 0.0 reports no sign change. Measure-zero in practice (−11,017.71
→ +0.0001 fires correctly), but the predicate is about *crossing*, and exact zero is a
crossing. One-character fix: `(d1 > 0) != (d2 >= 0)` or an explicit zero case.

## (3) THE REFERENCE LEVELS ARE PRESENT — YOUR GREP MISSED THEM

**Both literals are in the origin blob**, as module constants:

```
26: REFERENCE_INTERVAL_SCOPED_CENTS = 18.4
27: REFERENCE_WINDOW_SCOPED_CENTS = 1036.5
136-137: "reference_levels_cents": {"interval_scoped": …, "window_scoped": …}
```

and the driven emit prints `{'interval_scoped': 18.4, 'window_scoped': 1036.5}`. **Not a spec
defect — spec (c) is satisfied.** They are **DECLARED CONSTANTS, not computed** from the
gap-seconds column; acceptable, with one coupling note: **if the census/table split ever moves
a row, the reference levels will not follow it.** Deriving `interval_scoped` from the TABLE
rows' `gap_seconds` sum would close that, and would also have made the 28-row defect visible
as a changed reference level.

## (4) THE SPINE — CONFIRMED, AND IT IS THE GOOD HALF OF THE SAME CODE

```python
spine = {int(w["window_start"]): w for w in windows}
for start, w in sorted(spine.items()):
    rows.append({..., "delta_D_cents": float(contrib.get(start, 0.0))})
```

**Confirmed: the table is keyed on the DECLARED windows and an absent contribution emits as
`0.0`, not a missing row** — `n_rows` is always `len(windows)`. That is exactly the
absence-vs-zero distinction 186 guards, applied to the instrument. **The defect in §1 is not a
failure of this design; it is the wrong window set fed into a correct spine.**

## WHAT MUST LAND BEFORE THE TABLE RUNS (~08:55)

1. **`role == "TABLE"` filter in `declared_windows()`, plus a cell driving the REAL
   declaration and asserting 27.** Without it the first table published is 28 rows and
   contradicts the declaration on its face.
2. **v14 adopts 186's residual bands**, so the emit stops being out of spec while correct.
3. *(optional, cheap)* the `d2 == 0.0` edge, and carrying `role`/`in_BE_137_list` through to
   the rows so a reader can see the split the declaration made.
