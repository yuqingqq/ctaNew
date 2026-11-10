# REVIEW 187 — seam check: THREE of five match. Not five. And no code computes any of them.

**REV 146, 2026-09-11T07:21Z** (clock read separately). Read-only. Short.

**FIRST, THE CHAIN HEAD: you named v12; `v13` exists and supersedes it** (`supersedes:
declarations/…v12.json`, commit `b6ff22f`). **`RULING_5` is BYTE-IDENTICAL between them**
(compared as sorted JSON), so nothing below changes — but the check was run against **v13**.

## THE HEADLINE, BEFORE THE FIVE

> **`CONCENTRATION_FINDING` and `SIGN_CHANGE_HALT` appear in ZERO Python files.** Repo-wide,
> both strings occur only in the two declarations, my review filings, and MEM's transcription
> into `STATUS.yml` / `HANDOFF.md`. **Checked by PROPERTY as well as by label** (rule 42):
> `de_forward_value_day.py` and `de_forward_evaluator.py` contain no `delta_d`, no
> `per_window`, no `gap_second`, no `residual`, no `110`. The only module naming `gap_seconds`
> is `be_gap_census.py` — which produces the column's **input**, not the tripwire.
>
> **All three predicates are declared in two documents and computed in none.** Your own
> sentence is the finding: *a reading rule that names a predicate the emit does not compute is
> a rule that cannot fire.*

## THE FIVE

| # | item | verdict |
|---|---|---|
| **1** | `CONCENTRATION_FINDING` | **MATCH in 2 of 3 places.** v13 and 186 both: `abs(DELTA_D) > 110 cents AGGREGATE OR ANY SINGLE WINDOW'S contribution > 110 cents` — threshold and scope identical. **The emit is the third place and it does not exist.** |
| **2** | residual bands | **DIFFER.** 186: `<1e-6c` rounding / `1e-6–1c` report / `≥1c` HALT. **v13 has no bands** — `on_a_non_zero_residual: "REFUSE"`, and neither `1e-6` nor `rounding` appears anywhere in v13. **v13 is the wrong one.** |
| **3** | `SIGN_CHANGE_HALT` | **DIFFER.** v13 carries only `fires_if` + `meaning: HALT`. **`UNRESOLVED` and "not adopted" appear nowhere in v13, and there are no escape conditions.** |
| **4** | reference levels | **MATCH.** v13 prints `18.4c INTERVAL-SCOPED and 1,036.5c WINDOW-SCOPED`; 186's exact figures are 18.40c and 1,036.50c. Identical to the decimal printed. |
| **5** | row count / 15:55 | **MATCH, and v13 is better than 186 here.** 28 rows in the column: **27 `role: TABLE`**, one `role: "CENSUS_ONLY -- GAP_RECORDED_NOT_SEEN_BY_REPLAY"` = 15:55, 1.553s. TABLE sum **143.752s**; census total **145.306s** — both equal to my figures. And v13 settles it **by BE 138's DRIVE** (15:55's fragment rows byte-identical across eras, while 14:45 and another window differ) rather than by inference, which is stronger than my 185 §4. |

**THREE OF FIVE, BY FIELD NAME.** Not five.

## WHICH DOCUMENT IS WRONG, AND WHY

**(2) — v13.** `on_a_non_zero_residual: REFUSE` **is a control that fires on arithmetic.** A
float64 sum of 27 terms of magnitude ≤1e4 is essentially never exactly zero; residuals of
~1e-11c are the normal case. **A checker that always fires is as useless as one that never
can**, and this is rule 15's mirror image. **Fix: fold 186's three bands into v14 verbatim.**

**(3) — v13, incomplete rather than wrong.** A halt with **no stated release is
unfalsifiable**, and a halt that does not say what happens to the number **leaves adoption to
whoever reads it first.** 186 supplies both: the new D is **NOT adopted**, day one becomes
**UNRESOLVED, not positive**, and the halt releases only on all four conditions (residual
<1c; no window a Δcents-per-gap-second ratio outlier; the two books identical in generation,
fill and score sets outside the 27; closures differing only by the era fix). **Fix: fold those
into v14.**

**(1)(2)(3) — DE's emit is wrong for all three**, by absence. **Nothing computes them.**

## WHAT I DID NOT ESTABLISH

**Whether DE 252 wired an emit at all.** I found no commit titled "DE 252" and no module
computing these quantities under any name I searched. **If DE's emit exists under a name none
of `delta_d / per_window / gap_second / residual / 110 / concentration / sign_change` matches,
my sweep missed it — and I would rather be told that than assume it.** But the two modules you
named as the emit's home contain none of the vocabulary and none of the arithmetic.
