# REVIEW 173 — the certification's bar, declared before the run. And the finding that changes what the certification is FOR: this waiver cannot be made available by any comparator result.

**REV 130, 2026-09-11T03:21:42Z** (clock read separately, before composing). Read-only:
no lock, no heavy unit, nothing written under `data/`, no book unpickled, no arm VALUE on
any population day read. Tip at write time `d17780e`. **Written BEFORE BE's comparator
runs and before any number from it exists** — that is the point of the filing.

---

## 0. THE FINDING THAT COMES FIRST, BECAUSE IT CHANGES THE QUESTION

**I drove the programme's own waiver predicate against the ruled code, and it returns
FALSE for a reason no comparator result can change.**

`de_scoring_path_delta.waiver_available`, fed the real refusal rows (params v29's
`be_cascade` digests as `declared`, the worktree's `7ed5a90` bytes as `actual`):

```
de_head_scoring.py         status A_CHANGE_IS_ON_THE_SCORING_PATH   is_empty: False
   on the path AND changed: ['compose_head_inputs_batch', 'score_lgbm_condvalue_batch']
de_phase4_diag_runner.py   status A_CHANGE_IS_ON_THE_SCORING_PATH   is_empty: False
   on the path AND changed: ['_fragment_chunks', '_stream_fragment_rows',
                             'assemble_streaming', 'generation_scores']

WAIVER AVAILABLE: False
   [OK] every differing module's delta was COMPUTED
   [NO] every intersection is EMPTY
   [NO] every unit ON THE PATH is BYTE-IDENTICAL between book and disk
   [OK] no module-level statement outside defs and assignments differs
   [OK] the function-level walk AGREES with be_producing_closure
```

**`generation_scores` itself is modified and on the path.** Conditions (b) and (c) are
statements about **code identity**, not about outcomes. **Zero decision flips cannot make
an intersection empty and cannot make a changed function byte-identical.** So:

> **THE COMPARATOR CANNOT SUPPORT *THIS* WAIVER. No result it can return — not zero flips,
> not any margin — turns `waiver_available` from False to True.** A green comparator read
> as "the waiver is now supported" would be a category error, and it is the error most
> likely to be made in the next two hours.

The module says as much in its own docstring: *"It is not a licence. The refusal fires
either way... an empty intersection is EVIDENCE FOR A HUMAN DECISION — rebuild, or
supersede with the reason recorded."* Here the intersection is **not** empty, so even that
evidence is unavailable.

**What the comparator CAN support is a different claim, which needs its own name, its own
field and its own bar** — and the rest of this filing declares them. I suggest the name
**`SCORING_PATH_CHANGED_BUT_DECISION_EQUIVALENT_ON_MEASURED_DAYS`**, so that no reader can
resolve it as a waiver.

**And the structurally cleaner route, named because it exists:** a forward book BUILT at
`7ed5a90` records `7ed5a90`'s bytes, so book-vs-disk agrees; the refusal comes from the
comparison against **params v29's pin**. Re-pinning (a params v31 whose `be_cascade` records
`7ed5a90`) removes the refusal without any waiver — but it moves a frozen parameter after
the freeze and is therefore the USER's, not a seat's. **Either way the certification's
substantive job is unchanged: establish that the forward arm is the arm the screen scored.
The waiver field is bookkeeping; that is the question.**

---

## 1. THE EXPOSURE, MEASURED — AND IT IS ASYMMETRIC ACROSS THE TWO ARMS

Confirming DA at the artifacts: params v29's `be_cascade` holds **10** modules;
`941e688` and `adbebf9` are **10/10**; **`7ed5a90` is 8/10**, and the two that differ are
`de_head_scoring.py` and `de_phase4_diag_runner.py`. (REVIEW 171 measured the same two
against a different baseline — the arms pin `adbebf9` — and got the same pair. Two
baselines, one answer.)

Read at the diff, which arm touches which new code:

| arm | head | batched **composition** | batched **scoring** |
|---|---|---|---|
| **CONDVALUE_X_SKEW** | `q1_arrival_composed_lgbm` | **YES** — `compose_head_inputs_batch` | **YES** — `score_lgbm_condvalue_batch` |
| **HAZARD_OVER_SKEWED_REF** | `incumbent_linear_d` | **YES** — `compose_head_inputs_batch` | **NO** — still per-row `score_incumbent_condvalue` in a comprehension |

**Neither arm is unexposed, and neither arm's result transfers to the other.** Certifying
HAZARD alone would leave the LGBM batch-predict path — the one belonging to the arm with
the tighter Holm threshold (0.025) and the one expected to clear — entirely untested.
Certifying CONDVALUE alone would leave the incumbent's per-row scorer *fed by batched
features* untested. **That answers question (2): BOTH ARMS, separately reported, never
pooled, and with the reason each is required stated rather than "both to be safe".**

---

## 2. THE BAR — WHAT CONVERTS THE CLAIM FROM UNSUPPORTED TO SUPPORTED

Declared now, before any number. **All of A–E are required. Any one failing leaves the
claim UNSUPPORTED.**

**A. DECISION IDENTITY, per arm, per generation.**
`cancel_new == cancel_old` for every generation, where `cancel = (gen_max >= theta)` at
that arm's own theta (CONDVALUE 0.32450609461933483, HAZARD 0.43525926488298716).
**Required: zero flips, both arms.** *Necessary, not sufficient — see B.*

**B. THE PERTURBATION BOUND, which is the part that generalises.**
Report, per arm: **δ_max = max over generations of |gen_max_new − gen_max_old|**, absolute
and relative. **This is a property of the CODE and it is the only number from this run that
can say anything about a day the run did not touch.** The flip count is a property of the
DAY.

**C. THE MARGIN STATISTIC, per arm** — question (3), and yes, it must be in the bar:
- **m_min = min over generations of |gen_max_old − theta|**;
- **the ratio m_min / δ_max**;
- **the near-threshold occupancy**: count of generations with `|gen_max_old − theta| < 10^k · δ_max` for **k = 0, 1, 2, 3**;
- **the count of generations with `gen_max_old == theta` exactly** — these are maximally
  fragile, because the comparison is `>=` and a −1 ulp perturbation flips them;
- **n_generations**, per arm.

**The reading rule, declared now so it cannot be chosen later:**
- **m_min > δ_max** → no flip was arithmetically possible on that day. This is the strong
  form, and it is what A+B+C together are for.
- **m_min ≤ δ_max with zero flips** → the day got lucky. Report it as luck. It is not a
  certification.
- **a dense occupancy curve (many generations within 10·δ_max) with zero flips** → strong
  evidence: many chances to flip, none taken.
- **a sparse curve (nothing within 10³·δ_max) with zero flips** → weak evidence: the day
  never put the question.
- **δ_max above `REL_BAR = 1e-9`** → a finding requiring explanation **even with zero
  flips**. On scores that reach 17.79, 1e-9 relative is ~1.8e-8 absolute, which is **seven
  orders of magnitude above a last-ulp difference (~2e-15)**. A δ that large is not float
  noise; it is a behavioural change that happened not to cross theta today. **`REL_BAR` is
  the right REPORTING threshold and the wrong PASS criterion; the pass criterion is A+C.**

**D. RECONCILIATION — question (4).**
`n_generations_compared == n_generations_in_book`, per arm, stated as two numbers, not
asserted. **A partial read is a REFUSAL, not a weaker pass.** Any shortfall is a named
status with a count (rule 4), and the reason is not pedantry: **the generations a
comparator drops are plausibly the pathological ones** — unusual shapes, parse failures,
empty tranches — so a partial read is biased toward clean in exactly the direction that
matters. A comparator that silently skips is the `BINANCE_GAP_EXCLUDED: 0` shape the delta
module itself names.

**E. PROVENANCE OF THE OLD SIDE.**
The reference scores must come from the **existing book's stored per-generation values**,
not from a re-execution of the old code — a re-execution introduces its own environment
differences and would measure the environment, not the rewrite. If the stored book does not
carry per-generation maxima and a re-execution is unavoidable, **both sides must run in one
process, one interpreter, one library build**, and the receipt must say so. Otherwise a
difference cannot be attributed.

**THE FALSIFICATION CONDITION, declared before the number:** **any decision flip, on any
arm, on any compared day, refutes decision-equivalence.** It does not become "one flip out
of 24,000". The forward arm would then not be the arm the screen scored, and the answer is
a rebuild or a re-screen, not a tolerance.

---

## 3. QUESTION (1), ANSWERED DIRECTLY: ONE DAY IS NOT ENOUGH, AND HERE IS THE CHEAP FIX

**No. Zero flips on 09-03 alone does not certify two scoring modules**, for a reason that
is about the mechanism and not about caution. The perturbation from a per-row→batch rewrite
is **deterministic and input-dependent** — float accumulation order, and LightGBM's
matrix-predict versus row-predict code paths. A decision flips **iff** some generation's
max score lands within δ of theta. One day is **one draw of the near-threshold occupancy**,
over ~24k generations (09-04's figure: 24,133). Zero flips on 09-03 says 09-03's
generations did not happen to crowd theta. It says nothing about 09-07..09-13's.

**Two things fix it, and both are cheap:**

1. **RUN IT ON ALL FOUR CONSUMED DAYS — 09-03, 09-04, 09-05, 09-06.** Those books already
   exist and those days are already consumed, so **this costs nothing in population and
   quadruples both the δ sample and the occupancy sample.** If BE runs one day because one
   day was asked for, that is the cheapest available improvement to this certification and
   I recommend it be asked for.
2. **CARRY δ_max FORWARD AS A PER-BOOK GUARD.** Declare `DELTA_MAX_CERTIFIED` (the measured
   bound, per arm) and require **every forward book to publish its own m_min per arm and
   REFUSE by name if `m_min <= K · DELTA_MAX_CERTIFIED`**, K declared in advance (10³ is
   defensible; the number matters less than that it is fixed now). m_min is computed from a
   book already built, so this is near-free.

**That second half is what converts a one-day measurement into a seven-day guarantee.**
The certification bounds the code; the per-book guard checks the day. **Without it, no
number of consumed days licenses a forward day** — and that is the honest answer to the
scope question.

---

## 4. THE SCOPE RULING, IN TERMS THE WAIVER FIELD CAN CARRY

**WHAT A CERTIFICATION ON CONSUMED DAYS LICENSES:**
- a measured bound δ_max on the two modules' numeric divergence, on a real workload of
  ~24k generations per day per arm;
- the statement *"on the compared days, the two code lines produce identical cancel/keep
  decisions for both arms, at each arm's own theta"*;
- with C's ratio, the statement *"on those days no flip was arithmetically possible"* —
  which is stronger than "none was observed" and is the version worth having.

**WHAT IT DOES NOT LICENSE:**
- **that any forward day's decisions are unchanged.** m is a property of the day and the
  forward days' m values do not exist yet;
- that the modules are equivalent in general — reachability is a lower bound and the delta
  module's own `LIMITS` (dynamic dispatch, `getattr`, callback values, a changed import
  rebinding a name) are unclosed here as everywhere;
- **any relaxation of the `BOOK_BUILT_BY_DIFFERENT_SCORING_CODE` refusal**, which fires on
  code identity and is firing correctly (§0).

**SO THE FIELD SHOULD SAY, and this is the measured statement rather than the optimistic
one:**

> *Decision-equivalence between the params-v29 cascade bytes and `7ed5a90` was measured on
> N consumed day(s) [list], both arms, at each arm's own theta: F flips of G generations
> compared of G in book; δ_max = …; m_min = …; m_min/δ_max = …; occupancy within
> 10^0..10^3·δ_max = … . The book-code predicate REFUSES on every forward book and this
> does not waive it: `de_scoring_path_delta.waiver_available` is FALSE on conditions (b)
> and (c), because `generation_scores` is modified and on the scoring path. This
> establishes that the two code lines decided identically on the days measured; it does
> not establish it for any day not measured, and the forward days are guarded instead by
> the per-book `m_min` refusal at K·DELTA_MAX_CERTIFIED.*

---

## 5. THE DIRECT ANSWER TO "NECESSARY BUT NOT SUFFICIENT?"

**Yes — and the sufficient design is available tonight, which is why I am not merely
saying no.**

- A one-day, zero-flip result is **necessary and not sufficient**. Report it that way.
- A **four-consumed-day** result with **A+B+C+D+E** and **m_min > δ_max on every day and
  both arms** is sufficient *for the claim it makes*: the two code lines decided
  identically wherever they have been compared, with the margin showing no flip was
  possible.
- **Sufficiency for the FORWARD days requires the per-book guard** (§3.2). Nothing measured
  on consumed days can substitute for it, because the quantity that decides a forward day's
  answer is that day's own near-threshold occupancy.
- **And none of it makes the waiver available.** If what the artifact needs is
  `waiver_available == True`, the answer is that no comparator can deliver it and the
  routes are a USER-ruled re-pin or a recorded decision to run with the refusal standing.

**If BE returns zero flips on one day and the field is filled as "waiver supported", the
programme will have recorded a conclusion its own predicate contradicts** — rule 10's shape
(a hardcoded verdict beside a table that says otherwise), which has happened here three
times.

## 6. SCOPE OF THIS FILING

Driven: `de_scoring_path_delta.delta` + `waiver_available` on the real refusal rows
(v29 cascade vs worktree, worktree confirmed byte-equal to `7ed5a90` for both modules);
the 10-module cascade check at `941e688` / `adbebf9` / `7ed5a90`; the per-arm head
dispatch read from the diff's added lines. Not driven: BE's comparator, which has not run —
**every number in §2 is a slot, not a value, and that is deliberate.** I have not re-derived
BE's instrument and do not propose to replace `REL_BAR` or the per-generation decision test,
both of which are right; §2's additions are the margin statistic, the reconciliation, the
old-side provenance, and the forward guard.
