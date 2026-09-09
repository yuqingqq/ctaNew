# REVIEW 110 — `de_cancel_count_delta` (DE 159, `703a1be`), adversarial

**REV, 2026-09-09T07:14Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. I RAN it — the battery, the two mutations, the real pre-fix code and the real
stale cache — in a scratch clone at the tip. Every file I touched there is restored
byte-identical to the landed version (`fecd6ad51ec9`).

**VERDICT: SOUND. I tried to break it four ways and could not.** The falsifiers fire and I
proved they can FAIL; the OLD aggregation is not a flattering reconstruction — it is
byte-for-byte the pre-fix stream, which I established by running the pre-fix code; the
claims are computed; and the unit is the action. **Four findings, none blocking, all one
edit each.**

---

## (1) THE FALSIFIERS FIRE — AND I PROVED THEY CAN FAIL, BY MUTATION

Green: `--selftest` → **PASS 4 / 0 disarmed / 0 skipped**. That alone establishes nothing,
so I broke the instrument twice in the scratch copy:

| mutation | what it removes | battery |
|---|---|---|
| **A** — `old_stream` returns `new_stream(scored)` | the two aggregations become one stream | **RED at cell 1**: *"the OLD aggregation cancels 2 generations and the RULED rule 2 (delta 0)"* |
| **B** — `assert_per_row_assembly` returns a shape dict without checking | the known-bad gate | **RED after 2 passes** — cells 3 AND 4 both fail |

So the positive control detects a delta that is not there, and both known-bads detect a gate
that is not there. **Neither is a control that cannot fail.**

**Cell 4's claim, verified independently against the real artifact.** I loaded
`data/pm_5min/derived/de_section81_cache_12.pkl` myself — 29,813 scores, first value a bare
`float` — and drove the gate:

```
assert_per_row_assembly(stale)  -> REFUSED BY NAME: ASSEMBLY_PREDATES_CAUSAL_SCORING
measure_arm({}, stale, {})      -> REFUSED BY NAME: ASSEMBLY_PREDATES_CAUSAL_SCORING
assert_per_row_assembly({})     -> REFUSED BY NAME: CANCEL_DELTA_NO_ASSEMBLED_SCORES
```

**It is a refusal by name, not a delta of 0 dressed as a measurement** — and it is not
bypassable: the public entry refuses with the same name *before* computing anything, and an
empty assembly refuses rather than reading as "nothing changed".

## (2) THE OLD AGGREGATION IS FAITHFUL — I RAN THE PRE-FIX CODE

The only way to answer this was to run the thing it claims to reproduce. I extracted
`de_phase4_diag_runner.py` at **`c501824^`** — the last commit before the repair — and fed
the real pre-fix `generation_scores` + `score_events_for` the same blocks and reference I
gave the post-fix path, then compared its stream with `CCD.old_stream` rebuilt from the
post-fix per-row scores:

```
REAL PRE-FIX stream   : [(100.0,'s1','BUY_UP',0,4.154583719769), (200.0,'s2','BUY_UP',0,1.918692995604)]
CCD.old_stream rebuild: [(100.0,'s1','BUY_UP',0,4.154583719769), (200.0,'s2','BUY_UP',0,1.918692995604)]
IDENTICAL: True
```

Event for event, to twelve decimals. **It is not a reconstruction that flatters the delta;
it is the pre-fix stream.**

**One divergence, and it runs the safe way.** Where the reference carries a generation that
nothing scored, the real pre-fix path **REFUSES** (`DiagRefused: no assembled score for
generation ('s1','BUY_UP',500.0)`) while `old_stream` simply emits nothing for it — I drove
both. Since the same scored set feeds both sides, the delta is unaffected. But see finding
**B**: the instrument does not report that population at all.

## (3) IT COMPUTES ITS CLAIMS

Every number in `measure_arm` is derived from the two replays — the counts, the set
differences, the shift median and maximum, the percentage, the partial-row count against the
tape index. The two prose fields explain a mechanism rather than asserting a value, and
`partial_rows` degrades to `NOT_COMPUTABLE_NO_SPLIT_OF: … is UNKNOWN, not 0` when no tape
index is supplied, so absence never reads as a pass. One caveat, finding **A**.

## (4) THE UNIT IS THE ACTION, AND IT IS THE ONE THE USER RULED

`_cancels` keys by `(slug, side, gen)` — one entry per generation — and the engine's own
invariant is one cancel per generation. So the 39.7 % of rows that begin after their
generation's start **collapse to one cancel by construction, not by this instrument's
arithmetic**, and the table counts cancels, never rows. Under the USER's ruling B (match on
cancels) that is exactly the right unit, and it agrees with what DE 160 is implementing
elsewhere. Driven: `cancels_issued == len(by_gen)`. But see finding **C** — the invariant is
relied on and never asserted.

---

## THE FOUR FINDINGS

**A. Cell 1's message attributes a mechanism the number does not separate.** It reads
*"1 cancelled only under the old aggregation — the crossing falls after the generation
ended, which is the look-ahead's own signature"*, while the table reports
`cancelled_only_under_OLD` as a single count. **I went looking for a second mechanism that
would make that attribution wrong on a real book** — a binding cancel rate cap would do it,
because the old stream clusters every event at a generation start while the ruled stream
spreads them across row times, so the two would differ for a purely mechanical reason.
**It does not exist here: `max_cancels_per_minute` is `inf` both in `cell_params` and in
BE's real `params_for`, and neither arm in params v23 carries a cap-like key.** So on these
params the attribution is sound and the count does mean what the sentence says. Make the
sentence say *why* — "with the rate cap unbounded, only-OLD implies the crossing falls after
the generation's end" — so that if a finite cap is ever declared, the reader knows the number
stopped meaning that.

**B. The excluded population does not travel with the table (rule 4).** `measure_arm`
reports `n_generations_scored`, `n_scored_rows` and `partial_rows`, but **no count of
reference generations with NO scored rows** — the population the real pre-fix path refused
over. `generation_scores` already computes it (`NO_ROWS_KEPT`, and `N_REFERENCE_GENERATIONS`
at the union), so carrying both into the output is free. This table is the evidence a theta
re-fit gets ruled on; a day where a large share of generations went unscored would produce
a delta with no indication of the missing denominator.

**C. The action invariant is relied on and not asserted.** `by_gen` is a dict keyed by
generation, so a second cancel for one generation would be silently overwritten while
`cancels_issued` counted both — and then `cancelled_only_under_*` and `delta.cancels` would
be computed over different populations, one de-duplicated and one not. It holds today (1 = 1
in my drive) because the engine asserts it. One `assert len(by_gen) == cancels_issued` in
`_cancels` makes the table's unit checked rather than inherited.

**D. A stated requirement that is not a checked one.** `measure_book`'s docstring says
*"Requires the heavy lock"* and the module contains **zero** lock references. The discipline
is the launcher's; say so in the docstring, or take it here.

## ROUTED

1. **DE — B and C before the run against the corrected 09-03 book** (both are one line, and
   B changes what the table means when generations are missing).
2. **DE — A and D** whenever convenient; A is a sentence, D is a docstring.
3. **Nothing blocks the run.** The instrument may go against the corrected book as it
   stands, and its refusal will fire correctly if the book's assembly is stale.
