# DE 158 — a question for the USER: what does the null now sample?

**Not a recommendation.** The coordinator ruled this is the user's to
decide, and no control is drawn on a corrected book until it is. Point
estimates need no control and proceed meanwhile.

## What the unit WAS

`be_cancel_axis_null.load()` built the decision stream as **one row per
generation**, at the generation's start. The null's machinery is built on
that list:

* `_alloc` / `draw_flags` build sampling pools from those rows, per side,
  and draw `by_side[side]` of them without replacement;
* `flagged_stream` sets score 1.0 on the drawn rows and 0.0 on the rest;
* the arm is then matched to the control **on the action count** — the
  number of decisions drawn equals the number the arm made.

With one row per generation, "draw k rows" and "cancel k generations" were
the same statement, so the matched action count and the arm's cancel count
were **one quantity**.

## What it BECOMES

DE 155 (1) makes the assembled scores per ROW, and BE 107 makes `rows`
per row to match — 09-04 measured **15,867 of 40,000 rows (39.7 %)
beginning after their generation's start, across 6,586 of 24,133
generations (27 %)**, up to 60 rows in one generation. So:

* the pools are now **rows**, not generations;
* a draw of k rows touches **at most** k generations and usually fewer,
  because several drawn rows can fall in one generation;
* the policy still issues **one cancel per generation** (the engine's
  `one_cancel_per_generation` invariant, unchanged).

**So "matched on the action count" no longer names one quantity.** Matching
on drawn ROWS gives the control more draws than the arm has cancels;
matching on resulting CANCELS requires a draw whose size is not known in
advance.

## The options, with what each costs

**(A) Match on ROWS (what the code would do untouched).** The control
draws the same number of rows the arm's stream has above theta. Simple, no
new machinery. **Cost:** the control's cancel count is systematically
BELOW the arm's, because its drawn rows clump into fewer generations — the
comparison acquires a bias in the arm's favour that has nothing to do with
the policy.

**(B) Match on CANCELS.** Draw rows until the replay has issued as many
cancels as the arm did. **Cost:** the draw size becomes data-dependent and
so does the RNG consumption, which breaks the seed-reproducibility
property every landed null has (`draw_flags` is the only RNG consumer and
its call count is fixed today). It also makes the control's own action
count a random variable.

**(C) Keep the sampling unit at the GENERATION.** Draw generations as
before and mark, say, that generation's first row (or all its rows) in the
flagged stream. **Cost:** the null then samples a different object from
the one the arm decides on, and the arm's advantage from *when* it
cancels — the whole content of DE 155 (1) — is not represented in the
control at all.

## What is NOT in question

* The arm's own behaviour: one cancel per generation, at the first
  crossing. That is settled and driven.
* Comparability with the days already run: **none of them survives this
  anyway.** Every landed arm result was produced under the look-ahead and
  is retracted (R-834); the nulls beside them were matched under the old
  unit. Whichever option is chosen, the corrected days are a new
  population and cannot be compared draw-for-draw with the old ones.
* Point estimates: they carry no control, so they are unaffected and can
  proceed now.

## What I would need to implement any of them

Only the ruling. (A) is the current behaviour and needs no change; (B)
needs a draw loop whose termination is a cancel count, and a decision
about what "same seed" then means; (C) needs the pools built over
generations while the stream stays per row.
