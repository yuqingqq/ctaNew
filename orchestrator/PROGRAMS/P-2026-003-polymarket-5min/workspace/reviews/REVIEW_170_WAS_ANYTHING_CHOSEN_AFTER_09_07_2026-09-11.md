# REVIEW 170 — nothing fitted, tuned or selected postdates 09-07. Two code repairs do, and neither is a choice. The user is right.

**REV 128, 2026-09-11T03:06:55Z** (clock read separately before composing). Read-only:
no lock, no heavy unit, nothing written under `data/`, no book unpickled, and **no arm
VALUE on any population day read** — every check below is on presence, identity, digest
or timestamp. Tip at write time `7ed5a90`. REVIEW 169's stranded commit `9d21981` has
since landed in origin (via `40cbcdf`); nothing of mine is stranded now.

---

## 0. THE ANSWER

**The coordinator's drop recommendation was wrong, and the user's argument is the correct
one.** Applied as a measurement rather than a reflex, rule 11 asks whether any frozen
element was **chosen, fit, tuned or selected** after a 09-07 result existed. Ordered
against the artifacts:

- **Every FITTED object predates 09-07's close by ten days or more.** Six model files,
  all 2026-08-28T05:50–05:52Z, all six digests verified **equal to the bytes on disk
  today**, directory mtime unchanged since. `models_refit: false` is true at the bytes.
- **Both thetas predate it by two days** and are bit-identical across three independent
  re-derivations, and bit-identical to the `10%` causal threshold inside each arm's own
  08-28 fit artifact. `thresholds_refit: false` is true at the bytes.
- **The score formula, the two heads and the two arm names all predate it** — by eleven,
  six and fourteen/four days respectively.
- **Two things DID move after it, and both are defect repairs authored by the USER**, each
  with exactly one correct form determined by an object fixed *before* 09-07 existed:
  a **units** mismatch and a **look-ahead**. Neither is a free parameter. And the repair
  did not *use* 09-07 — **it RETRACTED it**, by name, in a declaration committed
  2026-09-09T03:05:30Z.
- **The one place where a choice genuinely was available after seeing — the sidedness of
  the test — was resolved in the CONSERVATIVE direction**, and at N=7 it stops being
  load-bearing at all.

**So: no element of the frozen test was selected on a 09-07 outcome, and 09-07 is usable
exactly as the user says.** Three residuals are named in §7; none of them is closable by
dropping the day, which is the point that decides it.

---

## 1. T0 — THE MOMENT A 09-07 RESULT COULD FIRST EXIST, ESTABLISHED THREE WAYS

| bound | when | what it is |
|---|---|---|
| **structural** | **2026-09-08T00:00:00Z** | 09-07 closes. No full-day arm result can exist before it. **This is the bar I use — it is the earliest and therefore the strictest.** |
| first arm-outcome **write** | 2026-09-08T00:21:18Z | `harmful_exposure_rows_v3_gate1_20260907_btc.json` |
| first **unsealed** arm value | 2026-09-08T12:28:11Z | `p003_de_point_estimate_day_20260907_L250ms__20260908T122617Z.json`, `sealed: false` |

There is **no earlier partial-day peek**: the `p003_de_early_read_day_*` family exists for
09-03, 09-04, 09-05 and 09-06 and **not for 09-07** (REVIEW 169's census, path arm).

## 2. THE USER'S PREMISE, VERIFIED RATHER THAN ACCEPTED

`data/pm_5min/raw/20260907`: **2016 files**; oldest 2026-09-07T00:06:30Z, newest
2026-09-08T00:01:30Z (the last window's rotation); **zero files modified after
2026-09-08T00:30Z.** The same holds for 09-08, 09-09 and 09-10. **The tape is immutable
and has not been touched since the day closed** — a rebuild under the frozen builder reads
exactly the bytes it would have read at any time. The user's "what's the issue using 9.7
data" rests on a fact, and the fact checks out.

## 3. THE ORDERING, ELEMENT BY ELEMENT

**Everything above the line was fixed before 09-07 could produce a result.**

| frozen element | when its VALUE was fixed | how established |
|---|---|---|
| **six model files** (`lgbm_haz_btc.txt`, `lgbm_val_btc.txt`, `lgbm_thresholds_btc.json`, `linear_btc.json`, `linear_d_btc.json`, `val_models.json`) | **2026-08-28T05:50:31–05:52:10Z** | file mtimes; **all six sha256 prefixes in params v29 recomputed from the bytes and MATCHING**; `phase2_fits/` directory mtime 2026-08-28T05:52:14Z, so nothing added or removed since |
| **score formula** `p_fill * conditional` | **2026-08-28T09:39:27Z** (`9ace8c1`) | `git log -S` on `phase2_iter011.py` |
| **arm name** `CONDVALUE_X_SKEW` | **2026-08-28T09:42:56Z** (`7815c2f`) | `git log -S`, first introduction |
| **heads** `q1_arrival_composed_lgbm`, `incumbent_linear_d` | **2026-09-02T18:51:05Z** (`47a2ba6`) | creation of `de_score_stream.py`/`de_head_scoring.py` |
| **arm name** `HAZARD_OVER_SKEWED_REF` | **2026-09-04T11:18:25Z** (`7f62c0c`) | `git log -S`, first introduction |
| **theta, BOTH arms** (0.32450609461933483 / 0.43525926488298716) | **2026-09-05T16:25:47Z** | `be_cancel_axis_null_v1.json`'s own `as_of`; **bit-identical in v1 (as-of 09-05T16:25Z), v2 (09-06T02:46Z) and v3 (09-06T04:51Z)**, and bit-identical to the values in the freeze |
| **the two-arm pair** as a pair | **2026-09-05T16:25:47Z** | the same artifact's `cells` are exactly these two arms |
| params v1–v15 | 2026-09-06 04:12–16:44Z | `git log --diff-filter=A` |
| params v16–v19 | **2026-09-07 06:28–08:53Z** — during 09-07, **before its close** | same |
| — | **T0 = 2026-09-08T00:00Z** | |
| **composition 2** (value head bound into the scoring stream) + params v20 | 2026-09-09T02:51:28Z (`1d309bc`) | **§4.1** |
| the retraction declaration naming 09-07 | 2026-09-09T03:05:30Z (`03dbc1e`) | **§4.3** |
| **look-ahead repair** (score each row at its own time; first crossing cancels) | 2026-09-09T04:02:26Z (`c501824`) | **§4.2** |
| **composition 3** + params v21 | 2026-09-09T05:21:50Z (`b50254e`) | **§4.2** |
| params v22–v29 (frozen = **v29**) | 2026-09-09 05:28–09:58Z | `git log --diff-filter=A`; v29 has exactly ONE commit touching it — never edited |
| params v30 | 2026-09-10T17:09:28Z | same |
| **arms pin** `~/ctaNew-wt-arms` HEAD `adbebf9` | 2026-09-10T16:40:29Z | worktree HEAD read directly |
| **the development screen itself** (09-03..09-06) | 2026-09-10T17:21–18:59Z, verdict 19:00:56Z | `derived/settle/` mtimes |
| null construction, decision rule, the freeze | 2026-09-11T02:08:34Z | `b5f311a` |

## 4. THE TWO THINGS THAT MOVED AFTER T0 — AND WHY NEITHER IS A CHOICE

### 4.1 The units defect (`1d309bc`, 2026-09-09T02:51:28Z, author AND committer `yqq`)

`de_score_stream.py`'s own diff states it: *"The value head is part of the policy: the
frozen thresholds were produced from `p_fill * conditional_value`, not from hazard
probability alone."* Before the fix the value head **was not even loaded** — `HEADS` gained
`lgbm_val_{coin}.txt` and `val_models.json` in that commit.

**Why it is determined and not chosen — two independent grounds, neither of which is a
reading of any 09-07 number:**

1. **The threshold's own provenance, fixed 08-28.** Theta is bit-for-bit an order
   statistic of `p_fill × conditional_value` (`freeze_thresholds` emits
   `out[f"{int(b*100)}%"] = xs[k-1]`; DA 212 matched the frozen theta of each arm to the
   `10%` entry of that arm's own 08-28 fit artifact). A threshold in expected-value units
   admits **exactly one** dimensionally coherent score. The corrected formula was already
   determined by an artifact ten days older than 09-07.
2. **The data excludes the alternative.** DA 212, on 09-04's ledger: decision scores reach
   **17.791912 with 17,393 rows above 1.0**. A bare `p_fill` is a probability and cannot
   exceed 1.

### 4.2 The look-ahead (`c501824`, 2026-09-09T04:02:26Z, `yqq`) → composition 3 / params v21

Commit subject: *"the look-ahead in the decision is repaired — each row is scored at its
own time and the FIRST crossing cancels."* Previously `generation_scores()` took the
**maximum across a generation** and `score_events_for()` stamped it at the **generation
start**, so later information triggered an earlier cancel.

**Scoring a row at its own time is causality, not a tuning knob.** There is no second
admissible form of it, and the programme's own point-in-time rule fixed it long before.

**And params v21 says so against itself, which is the honest half:**
`be_module_repoint.EXPECT_THE_NUMBERS_TO_MOVE` = *"NOT behaviour-preserving and must not
be read as one. The cancel decision is re-timed to the first crossing…"*, with
`draw_path_functions_changed = ["generation_scores", "score_events_for", "_head_scorer"]`.
The same file also carries `THIS_IS_A_REPOINT_OF_REPAIRED_CODE_NOT_A_DESIGN_CHANGE` —
*"the ONLY content of v21 is the module digests moving"*. **Both are true, and the first is
the load-bearing one; a reader who takes only the second will understate what changed.**
Worth a wording note in a superseding version, but it changes no verdict here.

### 4.3 THE DECISIVE FACT ABOUT BOTH REPAIRS: THEY RETRACTED 09-07, THEY DID NOT USE IT

`live/pm_research/declarations/p003_expected_value_policy_correction_v1.json`, landed
2026-09-09T03:05:30Z, names
`p003_de_point_estimate_day_20260907_L250ms__20260908T122617Z.json` **by path and
sha256** — as the **last entry in a list of RETRACTED outputs**, beside the 09-03, 09-04,
09-05 and 09-06 point estimates. Its role in that document is the thing being withdrawn,
not the evidence for withdrawing it. The declaration further records
`why_no_numbers_are_reported`: *"No corrected daybook exists. Reporting the p_fill-only
numbers under the expected-value policy would preserve the defect this correction closes."*

MEM's round-288 sweep puts it at its strongest: **"EVERY ARM RESULT THE PROGRAMME HAS
PRODUCED IS RETRACTED — the arms decided on the WRONG QUANTITY and used INFORMATION THEY
COULD NOT HAVE HAD."**

**So the only 09-07 arm number that has ever existed is a number for a policy that no
longer exists** — hazard-probability scoring with a look-ahead in its timing. Whatever was
"seen" on 09-07 is not a result for the frozen arm. That does not make the day unseen, and
I do not claim it does; it makes the correlation between what was seen and the estimand
weaker than the declaration's markout argument, not stronger.

## 5. THE FAILURE MODE THE COORDINATOR NAMED — HUNTED AT THE BYTES, NOT AT THE MTIMES

*"A parameter that LOOKS pinned because its FILE is old, but whose VALUE was re-derived
later."* Four places it could hide, each checked by the property and not the label:

| where it could hide | the label | **the property, measured** |
|---|---|---|
| the six model files | mtime 2026-08-28 | **sha256 of the bytes on disk recomputed and compared to every `model_digests` entry in params v29 — 6/6 MATCH.** An old mtime with re-derived content would have failed here |
| theta | one pin source, mtime 09-06 | **read out of all three `be_cancel_axis_null_v{1,2,3}.json` — 09-05T16:25Z, 09-06T02:46Z, 09-06T04:51Z — bit-identical in all three and equal to the frozen value.** Two later re-derivations reproduced it rather than replacing it |
| params v29 | file dated 09-09 | **exactly one commit touches it in the whole history** (`2969caa`); it was never edited after landing |
| the fits directory | — | **directory mtime 2026-08-28T05:52:14Z** — no file added or removed since the fit |

**And the one that did NOT survive the hunt, reported as a positive finding rather than a
clean sweep: the frozen SCORING CODE is post-T0 bytes.** The freeze pins
`de_head_scoring.py = 53a406a0ae2a11ff…`, and params v21's `modules_moved` names exactly
that digest as the **"after"** of the 2026-09-09 repair. The frozen arm therefore runs code
written after 09-07 was visible. §4 is why that is a repair and not a selection — but the
sentence "everything predates" would have been false, and I am not writing it.

## 6. WHERE A CHOICE WAS GENUINELY AVAILABLE AFTER SEEING — AND HOW IT RESOLVES

The freeze's own `frozen_null_construction` says it out loud:
**`why_two_sided`: "the direction was chosen AFTER the point estimates were seen."**

That is a real post-hoc choice, and the point estimates in question include 09-07's. It
does not sink the test, for a reason that is checkable rather than rhetorical:
**two-sided is the CONSERVATIVE direction.** A post-hoc choice can only void a test if it
makes a pass easier; this one makes it strictly harder. The exact sign-test floor is
`1/2^G` one-sided and `2/2^G` two-sided: at G=6 that is **0.015625 one-sided against
0.031250 two-sided**, so one-sidedness is precisely what would have *rescued* N=6 against
the 0.025 the smaller p must beat — and it was not taken. Freeze amendment 9 (2026-09-11T02:38:43Z) closes it properly:
*"N=7 means no such choice is ever needed"*, and both arms clear the two-sided floor.

**That is the correct shape of the rule-11 test and it is worth stating as the general
form: a choice made after seeing is fatal when it is ANTI-conservative, and survivable —
with disclosure — when it is conservative.** Dropping 09-07 would not have touched this
one, which is the sharpest evidence that the drop was aimed at the wrong object.

## 7. THE RESIDUALS — NAMED, AND NONE OF THEM CLOSED BY DROPPING 09-07

1. **The traceless read.** A seat that read 09-07 and wrote nothing consumed it invisibly.
   Permanently unclosable, unchanged by this review, and **equally true of 09-08..09-11** —
   so it is not a reason to treat 09-07 differently from four of its six companions.
2. **The screen ran on 2026-09-10**, after 09-07 was visible — on 09-03..09-06 only, and
   both arms **FAILED** it. Arm selection was therefore not made by passing the screen, and
   the two-arm pair was fixed on 2026-09-05, before T0. The exposure that remains here is
   BE 114's, already filed: **the multiplicity is 69, not 2.** That is a far larger problem
   for this test than 09-07 is, and it is orthogonal to it.
3. **The book does not record its model identity** (BE 114 §8, confirmed by DA 212 §4):
   `header.score_contracts` carries the formula and kind but **no model digest**. So
   "composition 3 produced these scores" is verifiable at the RUN and only inferable at the
   BOOK. My §5 model-digest check is against **params v29 and the fit files**, which is the
   run-side chain — I did not close the book-side gap and do not claim to.

## 8. FINDINGS 1 AND 2, FILED HARD — REGISTER-READY

The register is the coordinator's surface (SEAT_PROTOCOL, seats table), so I do not write
it. Both rows below are drafted for the coordinator to land, and both are recorded here as
**the coordinator's to own**, in the coordinator's own framing.

> **R-### — A DECLARED REFUSAL THAT CANNOT FIRE, IN THE GOVERNING DECLARATION (REV 169 §6.1,
> REV 170).** `da_forward_test_declaration_v2.json` →
> `POPULATION.a_failed_quality_day.refusal_if_violated` = `QUALITY_DECISION_SAW_AN_OUTCOME`.
> **The string exists in exactly three files in the tree** (grep, whole repo, `.git`
> excluded): declaration v1, declaration v2, and `da_forward_test.py` — the word-checker
> that reads them. **No code raises it.** The instrument that actually enforces the
> property, `da_forward_admissibility.py` (`ba28469`), refuses under
> `ADMISSIBILITY_READ_A_NON_METADATA_SOURCE`. A reader resolving the declaration's field
> finds no producer. **This is rule 15 inside the document that governs the test**: a
> checker that ships no falsifier is a zero from an instrument that never proved it can
> fire. **Remedy, either direction, in the same round:** the instrument raises the declared
> name, or a superseding declaration names the refusal the instrument raises.

> **R-### — "NO `arm_total_cents` ANYWHERE" WAS A LABEL CHECK, AND IT WAS MINE (coordinator,
> REV 169 §6.2).** The claim was made to the user several times about 09-07. **It is true as
> a field name and misleading as a claim.** Verified at
> `p003_de_point_estimate_day_20260907_L250ms__20260908T122617Z.json`: the ESTIMAND claim
> HOLDS — `economic_settlement.status = NOT_VALUED_DAY_NOT_ADMISSIBLE`,
> `null_mean`/`null_sd`/`p_location`/`null_draws_summary.n` all
> `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN`. But the same artifact carries, with
> **`sealed: false`**, for BOTH arms: `absolute.arm.trades_cash_flow_cents`,
> `absolute.zero_cancel_baseline.trades_cash_flow_cents`,
> `absolute.reconciliation.arm_total_minus_baseline_total` and `reconciliation.D_E0` — all
> finite, all "read_from: the same replay that computes D(E0)", present by USER ruling
> R-782. **The settlement LEG was not valued; the trades leg and its arm-minus-baseline
> difference were, and they are unsealed.** Same shape as DA's `builder_commit` label over
> builder bytes and the phantom-unit monitor: **the search matched the vocabulary and missed
> the identity.** The correct statement of 09-07's status is the one in these words, and it
> belongs in the superseding receipt.
>
> **Addendum from REV 170, which changes how that fact should be read rather than
> softening it:** those numbers are for the **retracted** p_fill-only, look-ahead-timed
> policy (`03dbc1e`, 2026-09-09T03:05:30Z, which names this very artifact as a retracted
> output). They are unsealed arm-vs-baseline cash figures for a policy that no longer
> exists.

## 9. SCOPE

- **Closed over:** every element the brief named — both thresholds, theta for both arms,
  the composition (1→2→3 with the model sets that define each), `L_place`, the cancel
  latency's separateness, the protection mode, the repost model, the null construction, the
  combination and decision rule — each dated against T0 at an artifact, a digest or a
  commit; plus the six model files verified **at the bytes**, the three theta artifacts read
  directly, the raw tape's immutability, and the retraction declaration's treatment of
  09-07.
- **Not closed over:** `git log -S` over the whole history for two literals timed out at
  120 s and was not retried; the pathspec-limited searches reported above are what I ran.
  The book-side model-identity gap (§7.3) is confirmed and not closed. And no runtime
  verification that the pinned module digests are what an actual forward run will execute —
  that is BE's `import_closure`/committed-bytes guard, not re-driven here.
