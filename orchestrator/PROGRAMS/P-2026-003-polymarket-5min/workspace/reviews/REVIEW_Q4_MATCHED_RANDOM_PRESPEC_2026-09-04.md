# REVIEW — Q4's matched-random null, attacked BEFORE it is run

**Filed** 2026-09-04T12:47Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `10cc87d`** in `~/ctaNew-wt-rev`, clean · every
heavy step under `systemd-run --user --scope --slice=research.slice -p
MemoryMax=8G` · read-only against
`data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json`
(188,119 B, sha256 `ca311c8f24e37564…`, `as_of` 2026-09-02T05:21:34Z) and the
code at this tip · no sealed forward day opened, no write under `data/`, no
other seat's worktree opened.

**ROUTING.** Every numbered finding is **CHECKED** — executed or read at the
artifact this round. Two relayed claims are checked and the verdicts differ:
one confirms BE, one refutes a correction I was told BE had made.

**Everything in §1–§3 is written before the null exists.** §4 is answered from
the artifact as asked.

---

## 0. The fact that governs the whole round, and it should be settled before a single draw is spent

**No outcome of this null can make Q4 pass its declared gate.**

The gate is a **conjunction** — *"beats matched-random AND beats the incumbent
by a preregistered null (§5)"* — and the second conjunct is **already
evaluated and already False**. Measured at the artifact:
`increment_beats_incumbent` occurs at exactly **six sites**, all six Q4 cells,
and **all six read `false`**; there is no site anywhere in the file where it
reads `null`. `incumbent_counterpart_computed` for `Q4_combined_ev` is `true`.
The six raw p's are 0.01999 · 0.04998 · 0.08996 · 0.11094 · 0.13643 · 0.44628
over 2,000 sign flips; Holm gives 0.11994 · 0.24988 · 0.35982 · 0.35982 ·
0.35982 · 0.44628, and `survives_joint_reading_at_0_05` is **False in all six**.

So this run can move a cell from **GATE_PARTIALLY_EVALUATED** to
**GATE_EVALUATED_AND_FAILED**. That is a real gain — it closes an open
conjunct and removes an ambiguity the programme has been carrying — and it is
the *only* gain available. **It cannot produce a positive result, and the
dispatch should say so before the run rather than after**, because the first
reader of a floor-level p on a fresh null will read it as a win.

---

## 1. The null design (your item 1) — rule 7, both halves

### R513-R1 — HIGH — the matching is right and **the POOL is wrong**, and I can predict the answer now (CHECKED)

Rule 7's first half — *matched on the decision variable (action count, side,
hour)* — is **satisfied by the implementation**. `harmful_action_eval.py:172-181`
counts the candidate's cancellations per `(side, hour)` stratum (`:174`) and draws
exactly that many per stratum. Count, side and hour are all held.

**What is not matched is the OPPORTUNITY, and that is what decides this test.**
`strata[key]` is built at `:71-75` over **every** generation, so the null draws
from the whole action universe. Read off `populations.btc.eval`:

| quantity | value |
|---|---:|
| actions in the Q4 population | **177,674** |
| actions with ANY preventable value | **17,604** (**9.908 %**) |
| preventable rows | 33,622 (1.910 per preventable action) |
| mean value of a preventable row | **−1.6364 c** (= the artifact's `conditional_cancel_value`, reproduced) |

**So 90.1 % of the pool is worth exactly zero to cancel, and the null spends
90.1 % of its budget there.** The proposition it tests is therefore *"can the
candidate find actions that have anything to prevent at all"* — not *"can it
cancel the right ones"*.

**PREDICTED, BEFORE THE RUN, from the artifact's own numbers:**

| budget | k | **predicted null mean** | observed candidate | gap |
|---|---:|---:|---:|---:|
| 5 % | 8,883 | **≈ −1,440 c** | +7,869.68 c | +9,310 c |
| 10 % | 17,767 | **≈ −2,881 c** | +12,333.50 c | +15,214 c |
| 15 % | 26,651 | **≈ −4,321 c** | +14,476.99 c | +18,798 c |

(k × 0.09908 × (−1.6364). The 9.908 % is an **upper** bound on P(row 0 is the
preventable row); if it is lower the null mean moves toward 0 and **the gap
grows**, so the prediction is one-sided in the safe direction.)

**I therefore predict p at the floor (1/501) at all three budgets and
`beats_random_max_on_NET` True at all three.** For the observed net to be even
3 SD from the null mean at 10 %, the RMS preventable value would have to be
≈ 121 c against a mean magnitude of ≈ 20 c — a sixfold tail. **And the prior is
not only arithmetic: all five heads that already carry this null in this
artifact came back at exactly the floor** — `Q1_arrival`, `Q2_p_neg`,
`Q2_p_pos`, `Q3_m_good`, `Q3_m_harm`, every one `p_value` 0.001996007984, n_draws
500, `n_movable_strata` 30. **Five for five at the floor, and I am predicting the
sixth.**

**A prediction that comes true is not evidence about the model.** Record the
prediction now; if it lands, the correct reading is *"a uniform-random cancel
policy on this population loses money, which the artifact already said"*, not
*"the candidate beat its null"*.

### R513-R2 — HIGH — the second half of rule 7 fails on a DECLARED choice whose stated direction the artifact inverts (CHECKED)

Rule 7's second half is *compared on the DECISION metric*. Both sides report
net cents, so the metric matches — but **the two sides value the same object by
different rules**:

* the candidate: `cross = next(i for i in gens[gk] if scores[i] >= theta)` then
  `val(cross)` — the **first CROSSING row** (`:152-153`);
* the null: `val(gens[gk][0])` — the **first row** (`:181`).

btc runs 1.754 rows per action (311,640 / 177,674), so these are routinely
different rows. **The choice is declared, and honestly** (`:49-53`):

> *"randoms cancel the same number of gens … each at its FIRST row — the
> earliest decision, i.e. **the most generous preventable window a random
> policy could have**. Declared."*

**But the artifact inverts the direction of that generosity.** A wider
preventable window means more exposure, and the mean value of exposure here is
**negative**: `p_neg` 0.5242 > `p_pos` 0.4733 and `m_good` 20.319 > `m_harm`
19.048, so `conditional_cancel_value` = −1.6364 c. **More exposure makes the
null WORSE, not better** — so the choice made to be conservative is, on this
population, anti-conservative, and it widens the very gap R513-R1 predicts.

Nothing here says the code is wrong; the reasoning was written when the sign
was not yet in hand. **The fix is one line of reporting, not of code**: the
emission must state the null's realised preventable-hit rate beside the
candidate's, so a reader can see what the two sides were actually valuing.

### R513-R3 — MEDIUM — "the two nulls are NOT interchangeable" is not true as stated, and the real obstacle is the interface (CHECKED, and it refutes the relayed structural reason)

The relayed diagnosis is that `matched_random_null` permutes **outcomes within
strata** while Q4 needs the **set-drawing** form, and that the two are not
interchangeable. **For a budgeted SUM they are the same null.** Measured, one
stratum, n=400, k=40, 20,000 draws:

| construction | mean | sd | p05 | p95 |
|---|---:|---:|---:|---:|
| permute values within stratum, sum over the FIXED selection | −43.294 | 117.551 | −236.74 | 151.47 |
| draw a uniform k-subset, sum TRUE values | −43.482 | 117.433 | −235.83 | 149.32 |
| analytic (uniform k-subset without replacement) | **−43.552** | **116.833** | — | — |

Both are the sum of a uniform k-subset drawn without replacement; they agree
with the closed form to Monte-Carlo error.

**The genuine obstacle is narrower and it is an interface fact**:
`matched_random_null` hardwires `metric = auc | calibration_slope`
(`phase2_iter011.py:1060`) over the **full** vector, and Q4's statistic is a
**sum over a budgeted selection**. You cannot pass it. That matters for what
gets built: *"the forms are not interchangeable"* invites a new null with new
properties; *"the interface does not take this statistic"* invites the minimal
extension whose equivalence to the existing family is provable — which is what
the table above establishes. **BE's conclusion (Q4 needs its own
implementation) is right; the stated reason is not, and the right reason is
cheaper to defend.**

---

## 2. Pre-specified acceptance criteria (your item 2), written before the number

### A PASS would be UNBELIEVABLE if any of these holds

1. **It lands where I predicted.** p at the floor at all three budgets with a
   null mean near −1,440 / −2,881 / −4,321 c. That is the pool of R513-R1
   speaking, not the model.
2. **It is reported as rescuing Q4.** It cannot — §0. A conjunct passing
   beside a conjunct that is already False is a gate that failed with more
   detail.
3. **The only statistic emitted is `beats_random_max_on_NET`.** That is a
   boolean against `r_nets[-1]` (`:196`), the **max of 500 draws** — an extreme order
   statistic. It is not a p-value, **Holm cannot consume it**, and the family
   denominator is 24. The run must emit a rank-based p (`(1 + #{draws ≥ obs}) /
   (n + 1)`), not the boolean.
4. **`n_random` is not enforced.** `evaluate_policy` has **no minimum-draws
   guard** — `N_RANDOM = 200` is a default (`:30`, `:40`) and nothing refuses
   below it, while its sibling `matched_random_null` refuses below
   `MIN_DRAWS_011 = 200` by name (`phase2_iter011.py:1049-1054`). The module's
   own selftest runs this null at **`n_random=50`** (`:276`, `:308`) — a draw
   count its sibling would refuse by name. Rule 6 is enforced in one null implementation and not in the one about
   to be used. **Port the refusal before the run**, or the declared n=500 is a
   caller's habit rather than a property.

### A FAIL would be UNINFORMATIVE if any of these holds

5. **The null is given preventability.** Drawing the matched set from
   preventable actions only would hand the null an OUTCOME the policy cannot
   know at decision time; a fail then says only that the candidate lacks
   information no policy could have. **Do not "fix" R513-R1 this way.**
6. **Saturated strata are not counted.** `pick = pool if cnt >= len(pool)`
   (`:180`) — a stratum where the candidate cancelled everything contributes
   **zero variance** and returns the candidate's own value in every draw. Enough
   of those and the null collapses onto the observation and p → 1: *"measured,
   not significant"* when the truth is *"no test was possible"*. This is
   precisely the degeneracy `matched_random_null` guards with `movable == 0`
   (`phase2_iter011.py:1080-1082`) **and `evaluate_policy` does not guard at all.** **Emit
   `n_strata`, `n_saturated_strata`, and the share of observed net carried by
   saturated strata.** With ~14 hours present × 2 sides ≈ 30 strata (the
   artifact's `n_movable_strata` is 30), saturation is unlikely at these
   budgets — which is a reason to report it cheaply, not a reason to skip it.
7. **It is read as more than one day.** `dates_present` 2026-08-25,
   `G_complete_utc_days` 0, `is_a_validation` **false**, `unit_used` window
   against `ruled_unit` UTC day, `weaker_than_ruled` true,
   `intervals_claimable` false. A fail here is one development day, and the
   artifact says so itself.

### What would make a pass INTERESTING

8. Only this: **the null mean lands near zero or positive** while the candidate
   is far above it. That would mean the pool is not doing the work and the
   selection is. I do not expect it, and I have said so above in numbers.

---

## 3. Your falsifier (your item 3) — right gate, wrong target, and one thing it silently assumes

**Is exact reproduction of 7,869.68 / 12,333.50 / 14,476.99 the right gate?**
Yes as a *same-arm* check, and I would keep it. But your stated reason — *"a
re-run at today's tree would compose a DIFFERENT ARM"* — is **measured false
for the code that matters.**

Of the **12** files in the artifact's `identity.fit_code_files`, checked at
three points (artifact hash, blob at `b1f36e21`, file at this tip):

* **11 are byte-identical at all three**, including **`harmful_action_eval.py`
  `2c4e21936e3fc1d2`** (the null) and **`phase2_arms.py` `ab19f5c639333bdc`**
  (the arm);
* **exactly one has drifted since the pin: `flow_intensity.py`**
  `e65c812fd42fc8d7` → `e0b0c5787770ba3d`.

So the pin is still correct discipline and the risk it guards is **one named
file**, which is a far stronger statement than a general worry — and it means
**the null code that will execute is the code reviewed in §1**.

**Is exact reproduction achievable?** Two answers, and the second is the one to
act on.

* **The observed statistic: yes, and trivially.** `net_cents` contains no RNG;
  `sorted` is stable and ties break on a deterministic key order; and the
  artifact already **reconciles internally** — `sum(increment_by_window)` equals
  `net_cents − incumbent_net_cents` in **all six cells** to ≤ 9.1e-13 c
  (166 windows each). The increment is recoverable from the artifact without
  re-running anything.
* **The path that produces it: conditionally.** The arm **re-fits** —
  `fit_seconds` 389.1 (lgbm) and 385.0 (linear) — and `LGBM_PARAMS`
  (`phase2_declaration.py:43-57`) carries `n_jobs: 4`, `random_state:
  20260826`, `subsample: 0.7`, `subsample_freq: 1`, and **neither
  `deterministic` nor `force_row_wise`**. Multithreaded LightGBM is
  bit-reproducible on the same build and thread count and is not guaranteed
  across either. **So an exact-reproduction gate on a re-fitting run can fail
  for a reason that has nothing to do with the arm.**

**Recommendation, and it makes the gate stronger rather than weaker:** do not
re-fit. Score once, persist the score vector, and compute the null over fixed
scores. Then exact reproduction is a property of arithmetic rather than of
LightGBM's threading, and a mismatch means what you want it to mean.

**And the falsifier checks the half that was never at risk.** `net_cents` has
no RNG, so reproducing it says nothing about the null. **The null itself is not
pinned by its seed** in this programme's own sense: `evaluate_policy` iterates
`use.items()` (`:178`) and consumes `strata[key]` in list-append order, **both
unsorted** — while `matched_random_null` consumes strata in **sorted** order and
says why (`phase2_iter011.py:1091`, reason at `:1045`, R-234: *"a seed pins the RNG STREAM, it does not pin
what the stream is applied to"*). **R-234 is honoured in one null and not in the
one about to be used.** A second gate is needed: run the null twice at the same
seed and require the draw vector to match, and sort the containers first.

**One provenance note, so the receipt is unambiguous.** The declared seed
**20260825** is `evaluate_policy`'s own default; the 011 family's null seed is
**`PERM_SEED_011 = 20260828`**. Two different seeds under one phrase
*"the matched-random null at n=500"* — correct, since they are different
instruments, and worth naming in the receipt so no reader reconciles them later.

---

## 4. Your item 4 — the combination is COHERENT, and here is the arithmetic that makes it so (CHECKED)

**It is not an accounting problem. The two numbers have different denominators
and different conditioning, and both are correctly labelled in the artifact.**

* `conditional_cancel_value = −1.6364 c` is the mean value of cancelling
  **conditional on the action having something to prevent** — a mean over the
  **33,622 preventable rows**. Reproduced from the components:
  (15,912 × 19.048 − 17,625 × 20.319) / 33,622 = **−1.6364**, matching the
  artifact to four decimals.
* `net_cents` is the total value of a **selection** over **177,674 actions**,
  of which only **17,604 (9.908 %)** have anything to prevent at all.

A positive selected total over a population whose *conditional* base rate is
negative requires no anomaly: the unconditional expected value of a random
cancel is ≈ 0.09908 × (−1.6364) = **−0.162 c**, so a random policy loses, and a
policy that concentrates its budget on the 9.9 % that matter — and, within
them, on the harmful side — gains. **That is what a working ranker looks like.**

**And the artifact contains the check that makes this more than a story.**
Cancelling **every** preventable row is worth **−55,019 c**. So a policy that
identified preventability perfectly and could not tell harmful from good would
**lose 55,019 c**. The candidate's +12,334 c therefore cannot come from
preventability-detection; **it has to come from sign discrimination inside the
preventable set.** That is the object-level content of the number, it is
computable from the artifact alone, and it is the part worth reporting to the
USER when the null lands.

**Two limits on that reading, both from the artifact:**

* **Scale.** The candidate captures a small share of what is available: the
  positive side alone totals 15,912 × 19.048 ≈ 303,090 c, so +12,334 c is about
  **4 %** of the harm on the table. Real, thin.
* **It is one day.** 2026-08-25, `G_complete_utc_days` 0, window-clustered
  p-values the artifact itself labels optimistic, `is_a_validation` false.

**So your instinct not to report the positive number was right, and for a
sharper reason than the missing null:** the number that will survive contact
with a reader is not +12,334 c, it is **+3,867 c against the incumbent, whose
own null it fails at Holm 0.24988.**

---

## 5. Two relayed claims, checked — one confirms BE, one refutes a correction

### R513-R4 — CONFIRMS BE (CHECKED)

*"Q4_combined_ev is the only head not built by `head_report`, so it never
receives `strata=`."* **True at the artifact.**
`results/btc/<arm>/heads` contains exactly `Q1_arrival`, `Q2_p_neg`,
`Q2_p_pos`, `Q3_m_good`, `Q3_m_harm` — five heads, each carrying
`matched_random` with `status OK`, `n_draws 500`, `n_movable_strata 30`. Q4
lives in `economics`, outside `heads`, and `head_report` is where
`out["matched_random"] = matched_random_null(...)` is attached
(`phase2_iter011.py:973`). The structural diagnosis is exactly right.

### R513-R5 — HIGH — REFUTES the relayed correction, and the correction would delete the programme's only evidenced negative (CHECKED)

I was told BE *"corrected one: `increment_beats_incumbent` is None, NOT
False — never recorded rather than evaluated and lost."*

**At the artifact it is `false`, at all six sites, and `null` at none.** More
importantly, **`false` is the correct value and `None` would be wrong**, by
BE's own supporting facts: `incumbent_counterpart_computed` is `true` for
`Q4_combined_ev`, the increment null WAS run (2,000 sign flips over 166
windows), and no cell clears 0.05 after Holm. A conjunct that was computed and
did not clear its threshold is **evaluated and failed**.

**The consequence is not bookkeeping.** Setting it to `None` would make *both*
Q4 conjuncts read "never evaluated", and the decision metric's failure — the
only evidenced negative this programme holds on its decision metric — would
disappear into an un-evaluated status. **The current encoding is exactly what
R-397 ruling 2 prescribes**: `passed: null` for the joint reading, `false` for
the conjunct that was evaluated and failed, `null` for the one nobody computed.
Three states, three meanings, all correct.

I cannot tell from here whether BE's claim was about a future emission or was
mis-relayed, and I am not guessing. **What is checkable is that no change is
needed, and that making the described change would be a regression.**

---

## 6. What I did not check, and one process note

I did not run the null (it does not exist), did not re-fit any arm, did not
hash the 3.2 GB tape or the 1.2 GB fragment — the run must verify those against
`identity.tape_sha256_prefix c7ab02ebcf27d2fc` and
`fragment_sha256_prefix 19a50195c34d0af2` itself — and did not measure how
often a cancelled generation's first crossing row is not its row 0. **That last
one is the single measurement that would close R513-R2**, it is one pass over
the rows the run already loads, and it should be emitted beside the null.

**Process note, cheap to fix.** SEAT_PROTOCOL 19's refresh command
`git -C <wt> checkout --detach mm-research` lands on the **local** branch ref,
which a plain `git fetch origin mm-research` does not advance. It put me on
`a0e6e57` while origin was on `10cc87d`, two rounds ahead. `checkout --detach
origin/mm-research` is the form that does what the protocol means.
