# REVIEW 119 — verifying DA 144 (gate item 8): the shape-conditioned coverage checks

**REV, 2026-09-09T08:18Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. Driven with my own fixtures in a scratch clone at the tip.

**ALL FOUR OF DA'S CLAIMS VERIFY, INCLUDING (c) — the loosening did NOT swallow the real
alarm. One finding, in the shape the round asked me to look for: a book whose scores are
dicts WITHOUT `gen` resolves to `PER_GENERATION_SCORES`. It is FAIL-SAFE for the coverage
predicate, so it is a labelling defect and not a missed alarm — and I say so rather than
dress it up.**

---

## (a) PER_GENERATION STILL VERIFIED AS BEFORE

A healthy per-generation receipt — 360 covered of 360, 360 scored keys — resolves
`PER_GENERATION_SCORES` from `DECLARED_BY_THE_RECEIPT`, applies the **strict** form
`n_scored_keys == n_covered`, holds True, and raises **no** false flags. The strict test is
still the one applied where it belongs.

## (b) THE FALSE ALARM THE REPAIR REMOVED, ON THE SAME BOOK

A healthy per-row receipt — 360 covered of 360, **972 scored keys, 2.7 rows per generation**:

```
new: holds True   form "n_scored_keys >= n_covered"   rows_per_covered_generation 2.7
OLD strict expression on the SAME book: False
```

That is DA's claim reproduced exactly: the old expression flagged a book with nothing wrong
with it, and the shape-conditioned form does not, while reporting the ratio so a reader sees
*why* the counts differ.

## (c) THE ONE THAT MATTERS — DRIVEN HARDEST, AND IT STILL FIRES

The real alarm: **60 generations dropped from the assembly and not the receipt** — 300
covered, 0 uncovered, 360 declared, with `coverage` still claiming 1.0. Under **both**
shapes, across both heads:

| shape | per head | total |
|---|---|---|
| `PER_GENERATION_SCORES` | `coverage_matches`, `covered_plus_uncovered_equals_generations` | **4 FALSE flags** |
| `PER_ROW_SCORES` | `coverage_matches`, `covered_plus_uncovered_equals_generations` | **4 FALSE flags** |

**Four each way, exactly as DA claims.** The two predicates that catch it are independent of
the shape conditioning — they compare covered against declared and the recomputed coverage
against the declared coverage — so the loosening of the *keys-vs-covered* test could not and
did not swallow the *coverage* alarm. **Rule 27 is satisfied by measurement here, not by
argument.**

*My own first count said two, because my fixture had one head; DA's four is across the two.
Recording it, because a fixture artifact reported as a discrepancy is how a verification
becomes a false finding.*

## (d) THE RESOLVER NAMES RATHER THAN COERCES — with one gap

Driven on every shape I could construct:

| values | resolved |
|---|---|
| bare floats | `PER_GENERATION_SCORES` |
| dicts with `gen` | `PER_ROW_SCORES` |
| half bare, half dict | `MIXED_SCORE_SHAPES` |
| dicts, only some with `gen` | `MIXED_SCORE_SHAPES` |
| empty / not a dict | `SHAPE_NOT_DETERMINABLE` |
| **dicts WITHOUT `gen`** | **`PER_GENERATION_SCORES`** |

**The last row is the finding.** `n_dict` counts only values that are dicts *and* carry
`gen`, so `n_dict == 0` means "no per-row values" and is read as "all per-generation
values". The resolver's own docstring says a per-generation value **is a bare number**; a
dict without `gen` is not one, and it is classified by *failing* the per-row test rather
than by *passing* a per-generation test.

**How it cuts, measured rather than assumed:** that mis-label selects the **strict**
predicate, so on a book that is really per-row it returns `holds=False` — **it over-flags,
never under-flags.** For the coverage predicate the mistake is **fail-safe**, and I will not
call it a missed alarm.

**Why it is still worth one line of code:** the label is what a reader and a downstream
consumer act on, and the docstring itself names the consumer —
`be_cancel_axis_null.load()`, which "would take the per-row branch on it and raise at the
first bare float". A shape that is named wrongly is exactly the input that consumer is not
warned about. And this shape is not hypothetical: assembly values have been gaining keys all
week (REVIEW 115: `NULL_DRAW` gained `settle_value`), so a value carrying `score` without
`gen` is a shape the current work could produce.

**Fix:** count the bare numbers too and require `n_num == len(scored)` for
`PER_GENERATION_SCORES`; anything else that is not wholly per-row is `MIXED_SCORE_SHAPES` or
a new name such as `DICT_VALUES_WITHOUT_GEN`. `live/pm_research/da_book_verify.py`,
`assembly_shape_of_values`, the `n_dict == 0` branch. Its cell is
`{("s",0.1): {"score": 0.5}, ("s",0.2): {"score": 0.6}}`, which today returns
`PER_GENERATION_SCORES`.

## ROUTED

1. **DA — the `n_dict == 0` branch**, with the cell above. It is a labelling defect with a
   fail-safe consequence today, and the consumer the docstring names is the reason to close
   it anyway.
2. **Coordinator — gate item 8 verifies**, including the one the round asked me to drive
   hardest: the real alarm still fires four ways under both shapes.
3. **Nothing here invalidates a landed claim.**
