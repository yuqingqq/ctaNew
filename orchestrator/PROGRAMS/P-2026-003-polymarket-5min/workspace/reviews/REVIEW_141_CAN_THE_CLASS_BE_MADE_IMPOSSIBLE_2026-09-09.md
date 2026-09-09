# REVIEW 141 — can the wrong-block class be made impossible?

**REV, 2026-09-09T15:07Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. **A design answer, not a change — DE and DA own their sites and the endpoint
ruling is the user's.**

**ANSWER: NOT NOW, AND THE HONEST RECOMMENDATION IS YOUR OWN SECOND OPTION — FIX THE SIX AND
MAKE THE MEMBERSHIP TEST A CELL. I prototyped the cell in twenty lines of AST and it finds
15 sites in milliseconds, so the class is CHECKABLE TODAY at near-zero cost. The one option
that would genuinely make it impossible is the single block with an endpoint dimension, and
it is a SCHEMA BREAK ON EVERY LANDED ARTIFACT during a rebuild — right in principle, wrong
this week. And the endpoint TAG is not a candidate at all: IT ALREADY EXISTS AND CAUGHT
NOTHING.**

---

## THE DECISIVE MEASUREMENT: OPTION B IS ALREADY BUILT AND ALREADY INERT

```
economic.endpoint             = None
economic_settlement.endpoint  = 'R-801 SETTLEMENT P&L (trades + residual)'
```

**The tag exists on the ruled block, has existed since R-801, and did not stop a single one
of the six sites** — because no site reads it. That is rule 28 in its own words: *the
evidence is recorded and the check is off*. **Adding a tag to the other block changes
nothing; making a CHECKER read the tag is option E under another name.**

## WHY DISTINCT FIELD NAMES DO NOT WORK — THE FAILURE DIRECTION IS WRONG

Renaming the settlement block's fields (`Z_settle`, `null_mean_settle`) makes it loud to
reach for `economic_settlement["Z"]`. **But not one of the six did that.** All six read
`economic["Z"]` **successfully**, while the quantity they were cited for lives next door.
**No renaming of the ruled block can make a successful read of the diagnostic block fail.**
It fixes the direction nobody travelled, and it costs a rename of every correct settlement
reader. **Rejected on the mechanism, not on the cost.**

## WHY THE ACCESSOR DOES NOT CLOSE IT EITHER

An accessor requiring the endpoint be named (`stat(arm, endpoint, field)`, no default) is
good hygiene and I would take it — **but it closes nothing on its own, because it cannot be
made mandatory.** The artifact is JSON on disk, read by DA, by MEM, by the early-read
verifier and by anyone with the file; the raw keys remain reachable. It prevents the next
site only if the raw keys are *gone*, which is option D. **On its own it is six more fixes
plus an API.**

## THE ONE THAT WOULD WORK, AND WHY NOT NOW

A single `economics: {D_E0_5S: {…}, SETTLEMENT: {…}}` block **would have made all six fail
loudly at the read** — it is the only candidate that attacks the actual failure direction.
Against it:

- **It breaks every landed artifact's shape**, and rule 13's whole reason for in-band
  supersession is that *automated readers resolve receipt fields*. Those are exactly the
  readers a shape change breaks.
- **It pushes the error somewhere worse during the migration**: readers would need a shape
  probe, and the idiomatic probe is
  `(d.get("economics") or {}).get(…) or (d.get("economic") or {})` — **the defensive
  `or {}` chain BE 123 has just finished proving yields a SILENT PASS on absence.** A
  migration would manufacture the very shape this class is made of.
- **The timing is the worst available**: every arm result is retracted, one corrected book
  exists, and the rebuild is mid-flight.

**Worth a ruling AFTER the rebuild, on the same day the endpoint question is settled — the
two are one decision, because a single block forces every reader to name the endpoint and
that is what the multi-day verdict needed.**

## WHAT I RECOMMEND, AND IT IS CHECKABLE TODAY

**Fix the six, and land the membership test as a cell.** The test is mechanical because the
class was enumerable — and a class that is enumerable is a class that is checkable:

```
AST: a literal "economic"/"economic_settlement" key, then one of the five SHARED names
     ['Z','null_draws_summary','null_mean','null_sd','p_location']
-> 15 sites, in milliseconds, ALL of them through `economic`, ZERO through the settlement
   block across the whole package
```

That last number is the finding restated as a measurement: **nothing in `live/pm_research`
reaches a shared statistic through the ruled block by a literal chain.**

**The cell's shape, so it does not become another list:** every site it finds must be on a
declared allow-list **carrying a REASON**, not a path — because the reason is the only thing
that separates `da_early_read_verify.py:1458` (*deliberate: comparing against the D_E0 ledger
draws*) from `da_gate1_day_verdict.py:925` (*a defect*). A new site is RED until someone
writes down which endpoint it meant. **Falsifier: plant a wrong-block read, the cell must
fire; remove one, it must not.**

**And its limit, declared rather than discovered:** it is a spelling sweep. A block reached
through a **variable** is invisible to it — the same limit I put in REVIEW 140's scope. **The
cell must say so in its own output**, or it becomes the next instrument that satisfies the
words.

## THE CHEAPEST REAL IMPROVEMENT, SEPARATE FROM ALL FOUR

**Make the existing tag load-bearing.** Any consumer that carries a `Z` into a verdict should
record the `endpoint` string it came from, beside the number. That costs one field, breaks
nothing, and turns `:2865`'s silent wrong-endpoint aggregation into something a reader can
see in the output even when nothing fails. **The evidence is already in the artifact; only
the carrying is missing.**

## ROUTED

1. **Coordinator — the answer is E, with D as a ruling for after the rebuild**, taken
   together with the endpoint question because they are one decision.
2. **DE / DA — fix the six as routed at REVIEW 140**; the cell is a follow-on, not a
   blocker.
3. **Whoever lands the cell — the allow-list carries reasons, and the cell declares that a
   variable-reached block is outside its reach.**
