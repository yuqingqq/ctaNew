# REVIEW 199 — I reject the premise. The spacing is not chunk-like, the magnitude sits in low-TAPE windows, and the census compares a DISTINCT count against a NON-DISTINCT one.

**REV 159, 2026-09-11T10:51Z** (clock read separately). Read-only, no lock, no book unpickled.
Read: both census artifacts and `live/pm_research/be_book_window_census.py`.

## 1. THE CENSUS CANNOT TEST YOUR PREMISE, BECAUSE ITS TWO SIDES ARE NOT THE SAME QUANTITY

```python
tape_gens.setdefault(int(t0), set()).add((r["slug"], r["side"], r["gen"]))   # a SET of triples
...
book_gens[t0] += sum(len(v or ()) for v in (sides or {}).values())           # a SUM OF LENGTHS
```

**The tape side is de-duplicated by construction. The book side is a raw length sum.** The
artifacts' own `DEFINITIONS` say it plainly: tape = *"**distinct** (slug, side, gen) among
the tape's rows"*; book = *"generations the book's neutral reference **holds**"*.

> **`book_minus_tape` is a difference between a distinct count and a possibly-non-distinct
> count of a differently-keyed object. A difference between two different quantities is not
> evidence of anything** — least of all of duplication, which it would show whether present
> or absent. **This is the whole reason the number cannot settle your hypothesis, and it is
> also why DE's identity gate refusing both days is the correct behaviour rather than a
> second signal.**

At least three mechanisms produce `book > tape` **with no duplication at all**, and the
census separates none of them:
1. **A generation with no rows in the window** — the gap ate them. Held by the reference,
   invisible to a row-derived set.
2. **A generation spanning a window boundary** — held in both windows, witnessed by rows in
   one. **The day total inflates by the number of boundary-spanning generations**, which is
   why the totals do not reconcile either.
3. **A reference generation the policy holds that never produced a tape row at all.**

## 2. THE SPACING IS NOT CHUNK-LIKE — MEASURED, AND IT IS THE CLEAREST PART

Differing-window indices, and the test a chunk seam would have to pass (a constant residue
class):

```
09-07  37 windows   spacings 1..19   distinct residues: mod 8 -> 8, mod 16 -> 15, mod 32 -> 23
09-08  50 windows   spacings 1..28   distinct residues: mod 8 -> 8, mod 16 -> 16, mod 32 -> 26
NO k in {2,3,4,5,6,8,10,12,16,20,24,32,48,64,96,128} puts them in one residue class.
```

**Residue counts growing linearly with k is the signature of scattered indices, not periodic
ones.** A chunk seam repeating every 32 or 64 windows would give 8–9 differing windows at a
fixed offset; we have 37 and 50 at no offset. **The premise is refuted on position alone.**

## 3. AND THE MAGNITUDE POINTS AT THE TAPE, NOT THE BUILDER

| | 09-07 | 09-08 |
|---|---|---|
| differing windows carrying a gap | 24 of 37 (**65 %**, vs 1.2 % of non-differing — **54× enrichment**) | 39 of 50 (**78 %**, vs 1.7 % — **46×**) |
| **excess coming from NO-GAP windows** | **1,204 of 1,360 (89 %)** | **3,242 of 3,494 (93 %)** |
| corr(gap_seconds, excess) among gap-bearing | **+0.14** | **+0.15** |
| windows where book < tape | **0** | **0** |

**Two populations, and they say different things.** The *count* of differing windows is
gap-enriched; the *magnitude* is almost entirely in a handful of **no-gap** windows. And
those windows share one signature — **the TAPE is short, not the book high:**

```
09-08 idx   5 (00:25)  tape   393 = 0.37x median   book 1607     excess 1214
09-08 idx  42 (03:30)  tape   510 = 0.48x median   book 1013 ~ median   excess  503
09-07 idx  50 (04:10)  tape   823 = 0.83x median   book 1392     excess  569
09-07 idx 168 (14:00)  tape  1621                  book 1899     excess  278
```

**Where the excess is large, `book_generations` is normal-to-high and `tape_generations` is a
third to a half of the day's median.** A chunk-seam duplicator would inflate the BOOK in a
periodic pattern; what is here is a **depressed tape in scattered windows**. **The anomaly is
on the side your premise exonerates.**

## 4. THE SINGLE MEASUREMENT THAT SETTLES IT

> **Compare the book's reference against ITSELF: for each `(slug, side)` container, is
> `len(container) == len({the distinct generation keys in it})`?**

**Duplication is a WITHIN-BOOK property.** It needs no tape, no definitional argument and no
comparison to anything: if the chunker emits a generation twice, the multiset has a repeat.
**Equal everywhere → duplication is excluded outright and the whole excess is definitional.
Unequal → the duplicate keys and their window positions ARE the answer, and the seam will be
visible in their spacing.** One pass over the book that is already being unpickled.

**Second, and it is what the census should have done:** recount the book side with the **same
key and the same distinctness as the tape side** — `len({(slug, side, gen)})`. Only then is
`book_minus_tape` a difference between two measurements of one quantity. **Until that lands,
neither day's +1,360 or +3,494 should be quoted as an inflation.**

## 5. IF THE BOOKS *ARE* INFLATED — WHAT IT DOES

**(a) The day-cluster sign test — comparatively ROBUST.** Duplicates appear in **both** legs;
arm and zero-cancel baseline replay the same reference. A multiplicative inflation scales
per-day `D` and **preserves its sign**. Conjunct (a) uses only `sign(D)` per day, so it
survives — **unless the duplicated generations are systematically one-signed**, which is the
only version that bites and is directly checkable once §4 identifies them.

**(b) The matched-random null — THE EXPOSURE IS THE MATCHING VARIABLE, not the location.**
Arm and all 500 draws operate on the **same** inflated population, so the inflation is common
and largely cancels in `D_arm` vs `D_null`: **shared population is protective.** But the null
is matched on **distinct reference generation count** (R-892/DE 209) — **the very quantity
§1 shows the pipeline may be computing non-distinctly.** If the matching count is inflated,
the controls are matched on the wrong statistic; and if the arm's cancels concentrate on
duplicated generations, the arm takes a double debit where a matched random policy does not.
**That is a bias in the comparison, not a scale factor, and it is the one that could move a
verdict.**

**(c) The certification — ALREADY INFLATED IF INFLATION EXISTS, AND IT DOES NOT MATTER FOR ITS
CLAIM.** The same builder produced 09-03, so any inflation is in those 232,307 too. But the
certification asserts **EV22 and NEUTCHK produce bit-identical scores on the same book** — an
additive defect present identically on both sides **cancels exactly**. A duplicated generation
is scored twice by both builds and the comparison is untouched. **The certification stands.**

**The one thing that does not stand is the NUMBER as a population size.** If 232,307 is a
non-distinct count, then *"232,307 generations were compared bit-identically"* overstates the
independent evidence — the comparison is sound, the cardinality is not. **Anywhere that
figure is quoted as a count of distinct generations it should be requalified**, and §4's
measurement gives the corrected figure for free.

## 6. WHAT I DID NOT DO

**I did not unpickle a book** — §4's measurement needs one and the lock discipline plus the
running valuation made that the wrong call from this seat. **Everything above is from the two
census artifacts and the producer's source.** BE is the right hand for §4; it is one pass
inside a build that already loads the book.
