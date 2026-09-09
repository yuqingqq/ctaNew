# REVIEW 113 — the rewritten sealed-value guard (DE 161, `c0e19ad`)

**REV, 2026-09-09T07:37Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. I drove my own corpus against BOTH guards — the landed one and the old one
extracted verbatim from `c0e19ad^` — in a scratch clone at the tip.

## THE GUARD HAS A HOLE, AND IT IS THE COMMONEST FORM IN ENGLISH PROSE

**A reason that ends a sentence with the sealed value is not tokenised at all.** Driven,
sealed `Z = 3.14`:

| reason text | tokens the guard sees | OLD | NEW |
|---|---|---|---|
| `"the sd was 3.14."` | **`[]`** | REFUSES | **PASSES** |
| `"sd of 3.14. The bar was 5"` | `['5']` | REFUSES | **PASSES** |
| `"the excess was 3.14bps"` | **`[]`** | REFUSES | **PASSES** |
| `"a ratio of 3.14x the floor"` | **`[]`** | REFUSES | **PASSES** |
| `"held 3.14s past the open"` | **`[]`** | REFUSES | **PASSES** |
| `"z_3.14_flag"` | **`[]`** | REFUSES | **PASSES** |

Every one of those writes the sealed value into a reason string. The second line is the
worst: the guard sees only the unrelated `5` and reports no hits, so the leak is invisible
*and* the output looks like it examined something.

**THE CAUSE.** `_NUM_TOKEN`'s boundaries are `(?<![\w.])` … `(?![\w.])`, so a numeric
literal followed by a letter, a digit, an underscore **or a dot** is not a token. A
sentence-ending full stop is a dot; a unit suffix (`bps`, `x`, `s`, `c`) is a letter.

**THE DIAGNOSIS, and it is the part worth carrying: the false positive was a COMPARISON
defect, and the fix corrected the comparison AND narrowed the tokeniser. Only the
comparison needed to change.** `0.05` colliding with a sealed `0.0`, and `21.53` with a
sealed `1.5`, were substring accidents — the numeric comparison alone kills them
(`float("0.05") != 0.0`). The tokeniser's word-boundary exclusions buy nothing on top of
that and cost the entire adjacency class. This is the shape the round was called for: a
guard loosened on the strength of a false positive, letting a real leak through.

**A FIX, DRIVEN IN BOTH DIRECTIONS.** Keep the numeric comparison; make the tokeniser
permissive, excluding only digit/dot continuation:

```python
_NUM_TOKEN = re.compile(
    r"(?<![\d.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)"
    r"|(?<![\d.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\d)")
```

Measured against my corpus: **all six adjacency leaks above are caught, and all four of the
false-positive cases the rewrite existed to fix stay green** — `0.05` vs sealed `0.0`,
`21.53` vs sealed `1.5`, `0.1592` vs sealed `0.159`, and `"count 5 of 200 decisions"`
against a sealed `3.14`. A trailing `.` is not consumed because it is not followed by a
digit, so `"3.14."` yields `3.14`.

## THE REACH THAT SURVIVED — every form the old guard caught is still caught

Driven at sealed `Z = 3.14159` and the integer/thousands cases, both guards refuse:
`str(v)`, `.1f`, `.2f`, `.3f`, `.4f`, `%g`, `round(v,6)`, the thousands-separated integer
`1,234,567`, the negative form `-2.5`, the exponent form `1e-06` (both `str` and `%g`), a
thousands separator inside prose (`"the count was 12,345 rows"`), a leading `+`, and the
value inside brackets, before a comma, before a semicolon, before `%` or a dash. **The new
guard also catches one form the old one missed** — `f"{fv:,.2f}"`, e.g. `1,234.50` for a
sealed `1234.5`, which the old form set never generated. So the rewrite is a widening
everywhere except the adjacency class.

## REV 110's FOUR, AS DE CLOSED THEM — driven, not read

- **(B) the excluded population now travels with the table.** Driven: a reference with two
  generations and one scored gives `n_reference_generations: 2`,
  `n_generations_with_no_scored_rows: 1`.
- **(C) the action invariant fires.** With `replay_policy` forced to return two cancels for
  one generation, `measure_arm` refuses `ONE_CANCEL_PER_GENERATION_VIOLATED`. **A
  correction to my own probe:** I first called `_cancels` directly, saw no refusal and
  nearly reported the assert as unreachable — the check lives in `measure_arm`, which is
  the consumer that depends on the invariant and the right place for it. Calling the inner
  function proved nothing.
- **(A) and (D)** are closed as text: the message and the docstring both now say what the
  code does.

## THE MATCHED-CANCEL BRANCH — sound, and it replaces a control rather than deleting one

Matched on **CANCELS** per the USER's ruling B: `build_pool_from_rows` over the same rows
the null samples, `demand_from_arm` taking the demand from the arm's own cancels,
`assert_random_wrt_arm` as a check on the draw, and the drawn sets **PERSISTED** — the right
call, because a data-dependent draw is not reproducible from a seed, and the receipt says
so in as many words. The receipt NAMES which path ran, so no reader infers it.

The cross-check that would have broken is handled properly:
`reproduces_BEs_draw_null` becomes `NOT_APPLICABLE_MATCHED_ON_CANCELS` with its reason —
BE's `draw_null` samples ROWS at a seed, ruling B samples CANCELS and is data-dependent, so
the sequences differ **by design** — and the provenance moves to the persisted artifact's
digest. **Keeping the old check would have asserted a design the USER replaced; deleting it
would have lost a control. This does neither.**

## ROUTED

1. **DE — the tokeniser, now.** It is the one item that lets a real leak through, and the
   fix is the two-line regex above, with the six adjacency cases as its cells (a
   sentence-ending value is the one to put first).
2. **DE — add a cell for the form the fix ADDED** (`,.2f`), so the widening is recorded
   rather than incidental.
3. Nothing else in 161 blocks: B, C, D and the matched-cancel branch are closed and driven.
