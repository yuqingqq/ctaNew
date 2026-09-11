# REVIEW 175 — the power fields ARE on the failure path (driven), nothing REFUSES a result that drops them, and the routing for the two sentences

**REV 132, 2026-09-11T03:34:41Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`, no arm VALUE on any population day read; every drive
below is in-memory against a synthetic fixture. Tip at write time `e4b36a5`.

---

## 1. FOLLOW-UP 2, DRIVEN NOT ARGUED: **THE EVALUATOR KEEPS THE PROMISE. THE GUARD DOES NOT ENFORCE IT.**

**Split verdict, and the split is the finding.**

### 1a. DE's evaluator — PASS, on a constructed failing day-set

I patched **only the data loader** (`AGG.load_cell`) with synthetic cells — a fixture
supplying INPUT and never supplying the floor block or a verdict, which the code under test
must compute (R-229 class). Seven days, two arms, 500 deterministic draws per cell, no RNG.

| cell | result |
|---|---|
| **FAILING population** (D = +100, −50, +30, −20, +10, −5, +2; unremarkable vs null) → verdict `NO_SETTLEMENT_SKILL_OVER_MATCHED_RANDOM`, both arms futile | **`attainable_minimum_p` PRESENT, `tolerance_negative_days` = 0** |
| **PASSING population** (positive control — presence must not be fail-path-specific) | **both PRESENT** |
| **`progress_emit` after 3 of 7 days**, both arms already dead | **both floors PRESENT** — at `G_ACHIEVED_SO_FAR` and at `G_DECLARED` |
| **KNOWN-BAD: floor block removed** — my checker must fire | **checker reports MISSING** ✔ |

**4/4.** `FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT` is built at the top level of `evaluate()`
with no branch on the verdict, and the drive confirms the emitted object rather than the
source. It carries more than the two numbers asked for:
`holm_threshold_for_the_smaller_p`, **`a_pass_was_possible_at_this_G`**,
**`requires_unanimity`**, and a `the_sentence` field that has arrived independently at the
same place I did:

> *"A test that cannot attain its own threshold has measured nothing about the arms — it
> has measured the calendar."*

**So the sentence is a promise the EVALUATOR keeps. On DE's emitted artifact,
`NOT_ESTABLISHED_AT_THIS_POWER, never NO_EFFECT` is backed by numbers the reader can see.**

### 1b. DA's result guard — the check is off

Driven on `da_forward_result_guard.require_forward_result`:

```
a FAILING result carrying forward_limits + pipeline_provenance_limit
and NO floor block, NO attainable_minimum_p, NO tolerance
  ->  ACCEPTED: {"status": "FORWARD_RESULT_FIELDS_PRESENT", "n_days": 7}
```

and the guard's source contains **zero** occurrences of `attainable`, `tolerance`, `floor`,
`power`, `minimum_p`, `NOT_ESTABLISHED` or `NO_EFFECT`.

> **The evaluator emits the numbers and nothing refuses a result that drops them.** That is
> rule 28's exact shape — *the evidence is recorded and the check is switched off* — and it
> matters here for a specific reason: **a "result section" is by definition assembled by a
> hand other than the evaluator's direct output.** The moment a human or a downstream
> writer composes the result, the guard is the only thing standing between the reader and a
> verdict with its power invisible, and the guard does not look.

**Your framing is right and the answer is a split one:** on the evaluator's own object the
sentence is kept; on anything the guard blesses, it is a promise the artifact does not keep.

## 2. FOLLOW-UP 1 + the fix for §1b — ONE ROUTING TO DA, TWO FIELDS, ONE REFUSAL

The two follow-ups have the same repair, so they should land as one change rather than two.
Routed to **DA** (the guard is DA's surface; I do not edit it):

> **ADD TO `da_forward_result_guard.py`, with its producer in the same commit
> (the DA 218 discipline that just closed REVIEW 169 §6.1):**
>
> ```
> NO_POWER_STATED   = "RESULT_DOES_NOT_STATE_THE_POWER_THAT_PRODUCED_IT"
> NO_SELECTION_READ = "RESULT_DOES_NOT_CARRY_ITS_SELECTION_HISTORY_READING"
> ```
>
> **`RESULT_DOES_NOT_STATE_THE_POWER_THAT_PRODUCED_IT`** — refuse unless the result carries
> `attainable_minimum_p` **and** `tolerance_negative_days` **and**
> `a_pass_was_possible_at_this_G`. Copy them from the evaluator's
> `FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT`; do not recompute them, so the guard checks the
> producer's own numbers rather than agreeing with itself. **Rationale for the reader:
> `NOT_ESTABLISHED_AT_THIS_POWER` is only true if the reader can see the power.**
>
> **`RESULT_DOES_NOT_CARRY_ITS_SELECTION_HISTORY_READING`** — refuse unless the result
> carries the **applicable one** of REVIEW 174 §6's two sentences, verbatim, selected by the
> verdict: the PASS sentence when any arm advances, the FAIL sentence otherwise. The field
> should be a literal, pinned by digest to the review, exactly as BE has just done with
> `WHAT_THIS_DOES_NOT_LICENSE` carrying REVIEW 173 §4 verbatim.
>
> **Both ship falsifiers in both directions** (rule 16): each must FIRE on a result missing
> the field AND ADMIT one carrying it — the boundary positive control, not only the refusal.
> And the sentence-selection needs its own known-bad: **a PASS verdict carrying the FAIL
> sentence must refuse**, or the field becomes a box that is ticked rather than read.

**Why a required field and not a review section, in your own evidence:** this programme's
history is that limits in prose get summarised away, and §1b shows it is not hypothetical —
the two limits that ARE required fields (`forward_limits`, `pipeline_provenance_limit`)
refuse correctly in both directions, and the power numbers, which live only in an emitter,
have no enforcement at all. **The difference between the two is not importance. It is
whether someone wired a refusal.**

## 3. REVIEW 169 §6.1 IS CLOSED — DRIVEN, BOTH DIRECTIONS

Reporting it because I opened it and it should not stay open in the record. DA 218 gave
`QUALITY_DECISION_SAW_AN_OUTCOME` a producer in `da_forward_result_guard.py`. Driven now:

```
quality decision whose sources_read = [.../be_daybook_20260907_btc.pkl]
   -> REFUSED QUALITY_DECISION_SAW_AN_OUTCOME          (fires)
quality decision whose sources_read = [raw/20260907, collector_gaps.jsonl]
   -> ACCEPTED                                          (admits)
```

**A refusal that fires on the bad case and admits the good one.** The declared name now has
a reachable raise; my finding is closed, not merely answered. The two limit refusals were
driven the same way and behave the same way.

## 4. TWO SMALL THINGS FROM THE DRIVE, NEITHER A DEFECT

**(a) An unarranged cross-check of REVIEW 174.** DE's `attainable_min_p` computes
`day_sign_component = 0.015625` and `draw_permutation_component = 0.001996007984…` — the
same two floors I derived independently in REVIEW 174 §2 from `2/2⁷` and `1/501`, by a
different hand from a different direction. It also publishes the Holm-adjusted forms
(0.03125 and 0.003992) against α = 0.05, which is the same test as comparing the unadjusted
values to 0.025 — **amendment 9's arithmetic checks out; I looked for a contradiction there
and there is none.**

**(b) `tolerance_negative_days = −1` is a sentinel, and it will read as nonsense in prose.**
At `G_so_far = 3` the drive returned `−1`, which `tolerance()` uses for *"not attainable at
any number of negative days"* — correct, and correctly paired with
`a_pass_was_possible_at_this_G: False`. But a consumer that formats the number will print
*"tolerance: −1 negative days"*. **Suggest DE render the sentinel as a token
(`NOT_ATTAINABLE_AT_THIS_G`) beside the integer** — the same split DA made when a STATUS
field carried a sentence. Cosmetic, one line, and it is the kind of thing that gets quoted
into a summary.

## 5. SCOPE

Driven: `de_forward_evaluator.evaluate` and `.progress_emit` on synthetic failing and
passing populations with `AGG.load_cell` patched (input only), four cells including a
known-bad; `da_forward_result_guard.require_forward_result` on five constructed results
covering both limit refusals, the quality refusal in both directions, and the
no-power-fields case. **Not driven:** the real forward result, which does not exist; the
downstream writer that will compose the result section, which is where §1b's gap actually
bites and which I cannot test until it exists. **Not mine:** the guard itself — §2 is a
routing, not an edit.
