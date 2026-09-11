# REVIEW 171 — the verdict in one line, the rule the register should carry, and the multiplicity answer

**REV 129, 2026-09-11T03:13:32Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`, no arm VALUE on any population day read. Tip at write
time `d931841`. **REVIEW 170 is already landed and in origin (`4008d2a`)** — this filing
adds the verdict, the general rule, the multiplicity escalation, and a measurement of the
code-line change the USER ruled while REVIEW 170 was being written.

---

## 1. THE VERDICT, ONE LINE

> **09-07 STANDS IN THE PRIMARY. No frozen element of this test was fit, tuned, chosen or
> selected after a 09-07 result existed; the only two things that moved after it are
> USER-authored defect repairs with exactly one correct form each, determined by objects
> fixed before 09-07 existed; and dropping the day would not have touched the one genuine
> post-hoc choice in the design — which is the sharpest sign the drop was aimed at the
> wrong object.**

**That is my verdict, in my words, and it confirms your reading of my pane.** You
recommended the drop, the user challenged it, and on the measurement the user was right.
All three residuals as you have them are confirmed: the traceless read is **equally true of
09-08..09-11, so it is not a reason to treat 09-07 differently**; the screen ran 2026-09-10
on 09-03..09-06 and **both arms FAILED it, so selection was not by passing it**; and the
book still records no model identity — composition 3 verifiable at the run, inferable at
the book, my digest check closing the run side only.

## 2. THE ADDENDUM — FACT CONFIRMED, RANKING CORRECTED

**The fact holds, and I re-verified it at 03:11Z after the tree moved:** the only 09-07
arm-value artifacts that exist are `be_daybook_20260907_btc.pkl` (09-08T01:29Z),
`…__L250ms.pkl` (09-08T06:08Z) and
`p003_de_point_estimate_day_20260907_L250ms__20260908T122617Z.json` (09-08T12:28Z).
**No corrected 09-07 book and no corrected 09-07 point estimate exists** — no `EV20/21/22`
revision for that day, and the one point estimate is named by path and sha256 in
`p003_expected_value_policy_correction_v1.json`'s RETRACTED list. So yes: the unsealed
`arm_total_minus_baseline_total` and `trades_cash_flow_cents` on 09-07 are figures for the
p_fill-only, look-ahead-timed policy — **a number for a decision rule that no longer
exists.**

**But it is NOT a stronger argument than the timestamp ordering, and I would not give it to
the user as the lead.** Correcting that ranking is the point of this section:

- **The timestamp ordering closes the CHANNEL.** Rule 11 forbids *choosing after seeing*.
  If nothing was chosen, the channel is shut whatever was seen. That is what REVIEW 170
  measured, at digests and mtimes, and it does not depend on anyone's account of what a
  number meant.
- **The retraction argument only weakens the INFERENCE.** It says what was seen was for a
  superseded design. It cannot close the channel, because two changes *were* made after T0
  and one choice *was* made after seeing (sidedness, §3). To be decisive it would have to
  establish that the retracted 09-07 figure carried no information about the corrected one
  — and that is false. **Same day, same tape, same fills, overlapping generations**; it is
  the same correlation the declaration already concedes for the markout, and conceding it
  there and denying it here would be having it both ways.

**So: corroborating, not load-bearing.** Give the user the ordering; carry the retraction
as the second line that makes the first one comfortable.

## 3. THE GENERAL RULE, REGISTER-READY

Drafted for the coordinator to land; the register is yours.

> **R-### — A POST-HOC CHOICE IS FATAL WHEN ANTI-CONSERVATIVE AND SURVIVABLE WITH
> DISCLOSURE WHEN CONSERVATIVE (REV 171; the coordinator's ask, from REVIEW 170 §6).**
> Rule 11 says *choosing after seeing voids the test*. It has been applied in this
> programme as a blanket prohibition on touching seen data — the coordinator applied it
> that way at 02:5xZ on 2026-09-11 when recommending that 09-07 be dropped, and the USER's
> challenge was correct. **The data is never the contaminated thing; the DESIGN is.** The
> operative test on any choice made after something was seen is its DIRECTION:
> **a choice that makes a pass EASIER is fatal — it manufactures the result and no
> disclosure repairs it. A choice that makes a pass HARDER is survivable, and what it
> requires is DISCLOSURE, not exclusion**, because it cannot create a false positive.
> **Worked instance, on the record:** the freeze's `frozen_null_construction` states
> `why_two_sided` = *"the direction was chosen AFTER the point estimates were seen."* That
> is a real post-hoc choice. It resolved TWO-SIDED. The exact sign-test floor is `1/2^G`
> one-sided and `2/2^G` two-sided, so at G=6 it is **0.015625 one-sided against 0.031250
> two-sided** — one-sidedness is precisely what would have rescued N=6 against the 0.025
> the smaller p must beat, **and it was not taken**. The choice went against the arms.
> **THREE BOUNDS, so the rule is not a licence:** (i) *conservative* is judged against the
> DECISION the test makes, not against a proxy — and it must be argued, not asserted;
> (ii) it does not license SHOPPING among conservative-looking options until one passes —
> one conservative choice disclosed is survivable, a search over them is a new selection;
> (iii) disclosure means **in the receipt, beside the result**, not in a sidecar. The rule
> separates the choices rule 11 must FORBID from the ones it need only require DISCLOSED,
> and the programme has been conflating them.

## 4. RESIDUAL 2 ESCALATED — WHAT THE FORWARD TEST CAN AND CANNOT DO ABOUT 69

**v3/v5's treatment is CORRECT on the denominator and INCOMPLETE on the prior. It is more
than honest, and less than adequate.** Three separate things are being carried under one
word, and they have different answers.

**(a) The denominator — `holm_m = 2` is RIGHT, and for a reason worth writing down.**
Holm's m is the number of hypotheses tested in the family *on the forward data*. Selection
history does not inflate the type-I error of a test run on data that played no part in the
selection. **That precondition is not assumed here — it is measured**: the two-arm pair was
fixed 2026-09-05T16:25Z (`be_cancel_axis_null_v1.cells`), the screen ran 2026-09-10T17:21–
19:00Z on 09-03..09-06, and the forward population is 09-07..09-13. No forward day entered
the selection. **So `it never enters Holm` is a correct statement and not merely a
comfortable one — and the amendment should cite the precondition it rests on, because that
is the thing a later reader must check rather than the claim itself.**

**(b) The prior — "it corrects the prior, not the denominator" is a statement with no
operation attached, and there IS one available, for one line.** The place 69 actually
bites numerically is the **pre-registered expectation**, which was computed from the
development point estimates — *the very estimates that selection from 69 candidates
inflates*. The freeze already calls that expectation OPTIMISTIC for a different reason
(09-05's −4,941 c sign flip). **The winner's curse is a second, independent reason, and it
is recorded nowhere.** Concrete, checkable, costs one field in a superseding amendment:
*the pre-registered power position is an upper bound ALSO because the point estimates that
produced it were selected from 69 candidates.* Without it, an inconclusive result will be
read as weaker evidence against the arms than it is, and a marginal pass as stronger
evidence for them than it is.

**(c) The one that actually bites, and which 69 does not cover: THIS IS LOOK NUMBER TWO.**
The freeze headline says it — *"TWO ARMS ON A SECOND ATTEMPT"* — but it appears in no
arithmetic anywhere. Holm m=2 controls the family *within this look*; nothing controls the
fact that the same two arms already failed a decision-endpoint screen and are being asked
again. **There is no correction that repairs this after the fact.** The only honest
treatment is the one the freeze already uses: it travels in the same sentence as any
p-value. **That is adequate, and it is the ceiling of what is available.**

**So, plainly: what can the forward test DO about a 69-candidate history? Exactly one
thing, and it is doing it — produce an estimate on data that took no part in the
selection. It does not correct 69; it BYPASSES it, and only for the two arms it tests.**
The moment a result is read as *"this family of policies works"* rather than *"these two
arms did X on seven days"*, the 69 comes straight back, because the family is what the 69
searched. **The asymmetry to state before the look: a PASS requires 7-for-7 two-sided and
should still move the posterior far less than p < 0.025 reads, given a failed screen and a
69-candidate search; a FAIL is nearly uninformative, because the floor foreordains
difficulty. The test is informative in roughly one direction, and that belongs in the
receipt before the first day is read.**

## 5. THE CODE-LINE CHANGE — MEASURED, AND BE'S CERTIFICATION HAS A TWO-FILE TARGET

The USER's ruling (whole pipeline on `7ed5a90`, arms-pin carve-out gone) landed while
REVIEW 170 was being written. Measured rather than assumed:

- **The freeze is ON the ruled line.** `b5f311a`, its amendment `3fc91e7`, and
  `22eee4c` (repo head at freeze) are all **ancestors of `7ed5a90`**.
- **All eight `PIPELINE_AT_THE_FREEZE_COMMIT` digests hold EXACTLY at `7ed5a90`** —
  `de_multiday_gate1_runner`, `de_point_estimate_day`, `de_settlement_control_run`,
  `de_settlement_control_aggregate`, `de_matched_cancel_control`, `be_cancel_axis_null`,
  `be_daybook_build`, `de_asymmetry_null_run`. Nothing in the pipeline moved.
- **The arms pin `adbebf9` is NOT an ancestor of `7ed5a90`.** They diverged at `8e46cc03`
  (2026-09-10T05:20:36Z); the pin side carries 6 commits the tip lacks, the tip 130 the pin
  lacks. That sounds alarming and the content says otherwise:
- **Of the thirteen pinned scoring modules, exactly TWO differ at `7ed5a90`:**
  **`de_head_scoring.py`** and **`de_phase4_diag_runner.py`**. The other eleven are
  byte-identical. **Both differences come from one commit, `a339734` "DE: share diagnostic
  head composition" (2026-09-10T06:48:18Z)** — which landed on the main line *after* the pin
  branch had already diverged.

**Where the certification will pass, and where it can fail — read from the diff:**

| module | + / − | shape | risk |
|---|---|---|---|
| `de_head_scoring.py` | +102 / **−1** | purely additive: new `score_lgbm_condvalue_batch`, `compose_head_inputs_batch`, a bound-check class, new selftest cells. **The single removed line is `EXPECTED_CHECKS = 40`** — a selftest counter. The single-row scorers are untouched. | **negligible; neutral by inspection** |
| `de_phase4_diag_runner.py` | +158 / **−19** | the removed lines are the **per-row scoring loop itself** — `HS.compose_head_inputs(...)` / `HS.score_lgbm_condvalue(...)` inside `for i, r in enumerate(kept)` — replaced by the batch variants, inside `generation_scores` | **this is the whole risk** |

**So tell BE the target is one file and one rewrite**, and the predicate matters more than
the day:

1. **Certify DECISIONS, not scores.** The comparison is `generation_max_score >= theta`. A
   per-row→batch rewrite can move a value in the last ulp; at the boundary that flips a
   cancel. Equality of the **cancel/keep decision per generation** is the property; score
   equality is the evidence.
2. **Report the minimum margin.** On 09-03, publish
   `min |generation_max_score − theta|` across all generations beside the largest observed
   score difference. If the smallest margin is orders of magnitude above the largest
   difference, neutrality generalises with an argued bound; **if any generation sits within
   float noise of theta, the certification is day-specific and must be repeated per day.**
   Without that number, "09-03 matched" is one day's luck, not a proof.
3. **Re-pin after the ruling.** The ruling names `7ed5a90`, but the tip has already moved
   past it (`4008d2a`, `d931841` at this writing). The ruled commit needs the same
   immutability discipline as a declaration: a landing that touches any of the thirteen
   modules after the pin is a re-pin, in the same round (SEAT_PROTOCOL rule 20's chain-head
   discipline applied to a code pin).

**And your conditional is right: if the certification fails, residual 2 gets worse, not
better** — the arm forward-tested would not be the arm the screen scored, the candidate
history gains a code-revision axis on top of the 69, and the freeze's `frozen_code` block
becomes a label rather than a property. **It would also reopen §4(a):** the precondition
that makes `holm_m = 2` correct is that the forward test evaluates *the arms that were
selected*. A behavioural code change between selection and test does not break the
denominator argument, but it does mean the thing being tested is not the thing that was
screened — which is the same defect the 69 describes, arriving through a second door.

## 6. SCOPE

Measured here: the existence (and non-existence) of corrected 09-07 artifacts, re-checked
at 03:11Z; ancestry of `b5f311a`/`3fc91e7`/`22eee4c`/`adbebf9` against `7ed5a90`; all eight
pipeline digests and all thirteen arms-pin digests recomputed from `git show` blobs at both
commits; the full `+`/`−` line census and the identity of every removed line in the two
moved modules. **Not measured:** whether the batch rewrite is numerically neutral — that is
BE's run and I have not pre-empted it; and the book-side model-identity gap, which remains
open exactly as REVIEW 170 left it.
