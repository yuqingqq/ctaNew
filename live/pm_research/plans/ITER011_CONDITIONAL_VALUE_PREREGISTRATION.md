# Iteration 011 — conditional signed-value decomposition: PREREGISTRATION

**STATUS: FROZEN BY USER RULING (R-232). Nothing here is fitted or scored.**
The design below — estimands, four questions and their gates, null designs
and minimum samples, model classes, consumed-day list, multiplicity — is
FIXED as of the ruling and predates every iteration-011 number.
**Authored by:** BE. **Authorised by:** the user's four-lane plan (`d506a06`),
lane 1. **Parent:** `plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` §2.1–2.2, whose
estimand definitions are reproduced verbatim below and are not restated in BE's
own words anywhere in this document.

This document exists to be frozen BEFORE any iteration-011 number exists. That
ordering is the point: §2.1's decomposition is a hypothesis about *why* the
previous model underperformed, and a hypothesis chosen after seeing which
decomposition looks good is not a hypothesis (rule 11).

---

## 0. What iteration 010 actually established, and why 011 exists

Stated first so this preregistration is not read as a fresh start on a clean
slate. Under the declared joint reading of the R-217 increment null
(`e7caaeb`), **10 of 12 cells were indistinguishable from chance**.
`PLUS_PRED_STATE_V1` — the arm whose entire purpose was isolating the state
features — **survived nowhere** (best Holm-adjusted p = 0.0700). Only
btc/`LGBM_PINNED` at the 5% and 10% budgets survived, against an
explicitly-optimistic window-level null, and **both btc arms collapsed at 15%**,
which no mechanism has explained.

Iteration 011 is therefore **not** a refinement of a working model. It is a test
of a specific structural claim: that a single unconditional regression through a
zero-heavy signed target is the wrong estimator, and that separating sign from
magnitude from arrival recovers signal the pooled fit destroys. **That claim may
be false.** This document must make its falsification as easy to report as its
confirmation.

---

## 1. Estimands (parent plan §2.1, verbatim)

```text
p_harm(x) = P(V_cancel > 0 | preventable fill, x)
m_harm(x) = E[V_cancel | V_cancel > 0, preventable fill, x]
m_good(x) = E[-V_cancel | V_cancel < 0, preventable fill, x]

conditional_cancel_value(x)
    = p_harm(x) * m_harm(x) - (1 - p_harm(x)) * m_good(x)

expected_cancel_value(x)
    = p_fill(x) * conditional_cancel_value(x)
```

`p_fill(x)` is the fill-arrival hazard: `P(preventable fill | x)`, where
*preventable* has the latency meaning fixed in §2 below and NOT the colloquial
one.

**Every quantity above is conditional on `x` at decision time.** No estimand in
this iteration may condition on anything observable only after the decision.

---

## 2. Target construction, and which module owns each component

`V_cancel` is the **signed** value of cancelling, built from the neutral
no-cancel `QR_SKEW_ONLY` reference path (parent §3). Positive = cancelling would
have avoided harm; negative = cancelling would have forfeited a good fill.

| component | owning module | notes |
|---|---|---|
| exposure rows, exclusions, statuses | `harmful_exposure_rows.py` | one row per `(decision time, generation, side)` |
| **the latency cut** `t + L` | `harmful_exposure_rows.py` | `cut = t_start + L/1000`; only tranches with `t >= cut` are valued; earlier tranches become `stale_shares` and are **never** valued |
| **the valuation gate** `any_fill_ahead` | `harmful_exposure_rows.py` | single definition since R-228(12); `keptrow` delegates to it |
| `preventable_value_cents` per latency | `harmful_exposure_rows.py` | `sum(-markout_cents_per_share * shares)` over post-cut tranches |
| PM/fine features | `harmful_hazard_model.py` | |
| predicted-state features | `harmful_state_features.py` (DA) | 45 pinned, `phase2_state_schema_freeze` |
| action-unit evaluation, budgets, randoms | `harmful_action_eval.py` | |
| declared constants, arms, budgets | `phase2_declaration.py` | |

**Rows are actions (rule 2).** The unit is the cancellable generation. Any
evaluator that can attribute one outcome to several rows must de-duplicate to
actions or the result is inflated.

**Never train on an outcome-selected population (rule 1).** The training unit is
the decision-time exposure on the neutral path — not completed fills, which
condition on the event the policy exists to prevent.

**`L` is part of the estimand, not a nuisance parameter (rule 7).** `L` is
pinned to `D.TARGET_LATENCY_MS` for every head. The latency grid is a
sensitivity analysis reported beside the result, never a search space.

---

## 3. The four separated questions

Each has its own metric, its own null and its own gate. **A head may not borrow
another head's success.** The parent plan's sentence is the governing one: *a
strong hazard head does not establish useful toxicity discrimination.*

| # | question | head | primary metric | gate |
|---|---|---|---|---|
| Q1 | **fill arrival** — does a preventable fill occur? | `p_fill(x)` | AUC + Brier, action-unit | beats the matched-random null AND beats the incumbent hazard head |
| Q2 | **harmful sign** — given a preventable fill, is cancelling right? | `p_harm(x)` | AUC + Brier, conditioned on preventable fill | same, on the fill-conditional population |
| Q3 | **magnitudes** — how much, given the sign? | `m_harm(x)`, `m_good(x)` | MAE and calibration slope, each reported SEPARATELY | calibration slope CI excludes 0 for each, reported separately |
| Q4 | **combined EV** — the decision quantity | `expected_cancel_value(x)` | **net cents at action unit**, the decision metric | beats matched-random AND beats the incumbent by a preregistered null (§5) |

**Reporting rule.** All four are reported for every candidate, always, including
when a head fails. A candidate advancing on Q4 while failing Q2 must say so in
the receipt; that combination is *interesting*, not disqualifying, but it must
not be presented as toxicity discrimination.

**Q3 is the head most likely to be silently skipped**, because magnitude
regressions on thin conditional populations are noisy. Its gate is therefore
stated in advance and its n reported per cell; a head with too few conditional
observations is reported **UNDERPOWERED**, never omitted.

---

## 4. Population and admissibility

- **Development fit** on the declared development population is **development
  evidence**. It selects; it never validates.
- **CONSUMED, named (rule 11):** `2026-08-20 .. 2026-08-25` are consumed for the
  harmful-fill line. `2026-08-26` is ruled FAIL (Q-DA-72, permanent short tape).
  `2026-08-27` is ruled EXCLUDED (R-222, gap-rate bar). None may be used for
  selection *or* validation in iteration 011.
- **Validation** = later untouched **complete UTC days, G ≥ 5**, per-coin under
  the live per-coin verdict regime. Below G=5: point estimate, **no interval**,
  and the receipt must say G explicitly rather than let its absence imply it
  (this is now generator-computed, R-230(4)).
- **Era purity** is a per-event predicate: sub-second features admissible only
  for `recv_ns >= 1787579334881534478`.
- Every quoted population carries its **n and as-of**; the tape grows during
  measurement.

---

## 5. Null designs — DECLARED BEFORE ANY RESULT (rule 6)

Minimum sample **≥200** permutations/draws for every null; the increment null
below uses **≥1000**.

1. **Matched-random null**, per head: randoms matched on the decision variable
   (action count, side, hour) and compared on that head's metric. Not on a
   proxy.
2. **Incremental-over-incumbent null**, per head, per budget: statistic = the
   head's metric minus the incumbent's on the **identical action population**;
   null = window-level sign-flip permutation of per-window paired differences,
   ≥1000 permutations, two-sided p.
   **This is the null iteration 010 did not have**, and its absence is why
   "beats random" was mistaken for "beats the incumbent" for a full cycle.
3. **Cluster unit.** The ruled unit is the **UTC day** (rule 8). Where G=0 the
   window is used as the finest plausibly-exchangeable substitute, and the
   receipt must state that this is **weaker than the ruled unit and therefore
   OPTIMISTIC** — evidence, never a significance certificate.
4. **Multiplicity, counted over every head.** Cells = candidates × heads (4) ×
   budgets. Read **jointly**, Holm-Bonferroni across the whole family. The count
   is recorded at freeze time and includes heads that fail.
   *A four-head decomposition quadruples the family; that cost is the price of
   the decomposition and must not be paid for by reporting only the head that
   won.*
5. **Baselines must remove the tautology (rule 9).** Skill is reported
   incremental to the input the target derives from, never against a base rate.

---

## 6. Model classes pinned in advance

Declared now, so no capacity search can be run and then reported as a choice:

- **Q1, Q2 (probability heads):** the pinned LGBM classifier params
  (`D.LGBM_PARAMS`, seed pinned) and the weighted linear head. No tuning.
- **Q3 (magnitude heads):** the pinned LGBM regressor params
  (`D.LGBM_VALUE_PARAMS`) and ridge, `lam` pinned. Two classes, both reported.
- **Q4:** composition of the above per §1. **No separately-fitted Q4 model** —
  composing is the hypothesis under test; fitting Q4 directly would answer a
  different question and silently rescue a failed decomposition.
- Hyperparameters are pinned across candidates for fair comparison. Any retune
  is a NEW candidate and increments multiplicity.

---

## 7. The `E[Y|state]` fence (parent §2.2)

The fair-price module owns the unconditional `E[Y | state]`. The toxicity module
estimates a **fill-conditional residual relative to that anchor** and must never
absorb an `E[Y | state, FILLED]` fair price — that puts adverse selection in
both terms and counts it twice.

**Checkable form**, so the fence is a predicate and not a promise: the
iteration-011 feature set must contain **no feature conditioned on the fill
event**, and the build must assert this by name against the pinned schema. A
feature whose construction reads fill outcome is inadmissible regardless of its
IC.

Inventory and lifecycle state remain **policy inputs**, not predictors. They
price whether a cancel is desirable; they do not become features because they
influence the action.

---

## 8. What freezing this document does and does not commit

**Does:** fixes the estimands, the four questions and their gates, the null
designs and minimum samples, the model classes, the consumed-day list, and the
multiplicity accounting — all before any iteration-011 number exists.

**Does not:** commit to a result, to any candidate advancing, or to a forward
race. Per rule 12 a freeze is a commit: candidate = builder file committed with
hash and commit ref in the receipt, full pipeline in the repo, declared nulls
inside the receipt, and multiplicity recorded at freeze time.

**The honest prior**, stated so its absence later is visible: iteration 010's
decomposed predecessor produced 10-of-12 chance cells. This decomposition is a
reasoned response to that, not evidence against it. **If Q1–Q4 come back null,
that is a result and it will be reported as one**, on the same footing as a
positive.

---

## 9. USER RULINGS (R-232) — settled, not open

All four were decided by the user before any iteration-011 number existed. They
are recorded here as constraints on the work, not as preferences.

**9.1 SCOPE = TWO ARMS.** Composed-linear and composed-LGBM. Both compose the
heads per §1; neither fits Q4 directly (§6).

  Holm family = 2 arms x 4 heads x 3 budgets = **24 cells**, read JOINTLY.
  Recorded at freeze time and counted whether or not a head fails.

**9.2 Q4 ALONE MAY NOT ADVANCE A CANDIDATE.** A candidate passing Q4 while
failing Q2 or Q3 requires **explicit user sign-off at that time**; it does not
advance on the strength of the composed number. It is **reported always**,
including when it fails — the reporting duty is unconditional and separate from
the advancement decision.

  This is the parent plan's rule given teeth: *a strong hazard head does not
  establish useful toxicity discrimination.* A composed EV that wins while its
  sign or magnitude head fails is precisely the case where the decomposition has
  not done what it was built to do, and it must not be able to advance quietly
  on the aggregate.

**9.3 NO AUTO-ENTRY TO ANY FORWARD RACE.** Iteration-011 candidates do not enter
a race by performing well. Race admission is a **separate user decision**, taken
with the numbers in front of them.

**9.4 G BAR = >=5 COMPLETE UTC DAYS, PER COIN.** Per-coin under the live
per-coin verdict regime — btc and eth accrue independently, and one coin
reaching the bar does not carry the other. Below the bar, per coin: point
estimate, **no interval**, G stated explicitly (generator-computed, R-230(4)).

---

## 10. What remains BE's to decide, and what does not

**BE decides:** implementation, known-bads and their falsifiers, which module
emits what, how the fence in §7 is asserted, how failures are surfaced.

**BE does not decide:** whether any candidate advances (9.2), whether anything
enters a race (9.3), or what a result means for the programme. Models estimate;
they never decide (rule 14). Where a number is ambiguous BE reports the
ambiguity rather than resolving it in either direction.

**The reporting duty is unconditional.** If Q1-Q4 come back null across both
arms, that is the result and it is delivered with the same prominence a positive
would get. Nothing in the rulings above makes a null harder to report than a
win, and if that ever appears to be the case, say so rather than soften it.
