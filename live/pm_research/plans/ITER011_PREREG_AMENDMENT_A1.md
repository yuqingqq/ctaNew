# Iteration 011 preregistration — AMENDMENT A1

**STATUS: FROZEN — USER RULING 2026-08-28T09:29Z (structured answer via the
coordinator, R-242): §A1.1 OPTION 1 (separate p_positive/p_negative) is
selected; §A1.2–A1.6 frozen as drafted. Q2's cell statistic under A1.4 is
therefore AUC; heads (4), arms (2) and the 24-cell Holm family are UNCHANGED.
No clock effect: iteration 011 has no forward clock (nothing fitted, nothing
scored), and the hazard-candidate race clock (`b3f7f9f`) is untouched.
The draft text below is preserved verbatim as committed at `56f272e`.**
**Amends:** `ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md` (frozen at `3b71d3e`
by user ruling R-232). **Drafted by:** BE. **Frozen by:** the USER, and only the
user — BE does not amend a frozen document it authored.
**Cause:** the system reviewer's pre-fit review (R-238). Every defect below is
**data-independent** and was knowable before any iteration-011 number existed;
none was found by looking at a result.

---

## A1.0 Why this is an amendment and not a fix

The Q4 defect in §A1.1 changes an **estimand**. A fix would silently substitute a
different quantity for the one the user froze; an amendment puts the substitution
in front of them. That distinction is the whole reason the freeze exists, and it
holds even though the flaw is unambiguous — *especially* then, because an obvious
correction is the easiest kind to make without asking.

The remaining items (§A1.2–A1.6) are things the frozen document **left
undefined**. Undefined is not the same as wrong, but an undefined term is
resolved by whoever implements it first, and that is exactly how an
implementation choice becomes a silent design decision. They are surfaced here
rather than settled in code.

---

## A1.1 THE Q4 ALGEBRA IS WRONG WHEN ZERO MASS EXISTS  *(estimand change)*

**Frozen text (parent §2.1, reproduced in the prereg §1):**

```text
conditional_cancel_value(x)
    = p_harm(x) * m_harm(x) - (1 - p_harm(x)) * m_good(x)
```

**The defect.** With `p_harm = P(V>0 | preventable)`, the complement
`(1 - p_harm)` is `P(V<=0)`, i.e. `P(V<0) + P(V=0)`. But `m_good` is defined as
`E[-V | V<0]` — measured on the **strictly negative** subset only. So the
good-side term weights a `V<0` magnitude by all **non-positive** mass:

```text
E[V] = P(V>0)·E[V|V>0] + P(V<0)·E[V|V<0]        (exact; the V=0 term is 0)
     = p_harm·m_harm − P(V<0)·m_good
```

The frozen form uses `(1 − p_harm)` where the algebra requires `P(V<0)`. They
differ by exactly `P(V=0)`, so whenever zero-value preventable fills exist the
composed conditional value is **biased downward by `m_good · P(V=0)`**. The
error is in the estimand, not the estimator: no amount of fitting removes it.

`zero_mass_diagnostic()` (committed at `77cc76c`) already computes `P(V=0 |
preventable)` on both populations and reports whether the frozen form is exact.
It was written to surface this, not to correct it.

### Options for the user

**Option 1 — separate `p_positive` / `p_negative` (minimal change).**

```text
p_pos(x) = P(V>0 | preventable, x)
p_neg(x) = P(V<0 | preventable, x)          # NOT 1 - p_pos
conditional_cancel_value(x) = p_pos(x)·m_harm(x) − p_neg(x)·m_good(x)
```

Exact for any zero mass. `p_zero = 1 − p_pos − p_neg` is implied and reported.
Q2 remains one head reporting two probabilities. Smallest departure from the
frozen text; keeps four heads and the 24-cell family unchanged.

**Option 2 — three-class sign head.** Q2 becomes a 3-class model over
{V>0, V=0, V<0}, with the same composition as Option 1. Estimates the zero class
directly rather than by subtraction. Strictly more model, more capacity to
misfit a class that may be tiny, and it changes Q2's metric from AUC/Brier to a
multiclass equivalent — which changes the metric→cell map in §A1.4.

**Option 3 — keep the frozen form, report the bias.** Defensible only if
`P(V=0)` is measured and negligible. It cannot be known to be negligible before
the run, so choosing this now is choosing it blind.

**BE's recommendation: Option 1.** It is exact, it is the smallest change to a
frozen document, it leaves the head count and the family size untouched, and it
does not add capacity to fit a class whose size is unknown. Option 2 is the
better model only if the zero class turns out to be both large and
predictable — which is a finding, not an assumption, and Option 1 measures it
(`p_zero` is reported) without betting on it.

**This is the user's call.** BE will implement whichever is frozen and will not
proceed on any of the three until one is.

> **RULED 2026-08-28T09:29Z: OPTION 1.** (User, structured answer; recorded by
> the coordinator, R-242. The recommendation and the ruling agree; the ruling,
> not the recommendation, is what authorizes implementation.)

---

## A1.2 `any_fill_ahead` MUST BE NAME-BANNED FROM THE FEATURE FENCE

The reviewer is right and my `FENCE_REVIEWED` entry for it is wrong.

I admitted `any_fill_ahead` with the reason *"the valuation GATE, not a feature;
never in the pin"*. Both clauses are true and neither is a defence. It is an
**outcome field** — it reads whether a fill occurs ahead of the decision — and
the fence's job is to make outcome fields inadmissible **by name**, not to trust
that they never reach the pin. A reviewed admission is a standing permission,
and this one permits the exact class the fence exists to stop.

**Amendment:** remove `any_fill_ahead` from `FENCE_REVIEWED`; the fence bans it
by name. Its use as the valuation gate is unaffected — the gate is not a feature
and never passes through the fence. This tightens §7 of the frozen doc; it does
not relax it, so it is recorded here for completeness rather than as a choice.

---

## A1.3 STRICT TARGET CONSTRUCTION — NO FAIL-OPEN ROWS

The frozen doc says exclusions are statuses, never silent drops (rule 4), but
does not say what happens to a row whose **inputs are malformed**. Current
behaviour fails open: a missing or inconsistent `latency` / `preventable_shares`
/ `preventable_value_cents` field yields a clean `0.0` or a "no fill" row that
is indistinguishable from a genuine one.

**Amendment:** a row whose valuation inputs are missing, non-numeric, or
mutually inconsistent (e.g. positive value with zero preventable shares) is
**REFUSED**, not zeroed. If such rows are expected in the population they must
be given an explicit named status and counted, exactly as gaps and truncations
are. A zero that means "absent" and a zero that means "no harm" must not be the
same number.

Also: `head_populations` currently consumes generators twice; it must build once
and reuse. That is an implementation bug, listed here only because it can change
counts.

---

## A1.4 THE METRIC→CELL MAP, AND A HOLM DENOMINATOR THAT CANNOT SHRINK

The frozen doc declares 24 cells (2 arms × 4 heads × 3 budgets) read jointly
under Holm, but does **not** say which number enters a cell when a head reports
several metrics (Q1/Q2 report AUC *and* Brier; Q3 reports MAE *and* calibration
slope). Undefined, this is resolved by whoever writes the evaluator.

**Amendment — one declared statistic per cell:**

| head | statistic entering the cell | rationale |
|---|---|---|
| Q1_arrival | AUC | discrimination is the question; Brier reported beside it |
| Q2_sign | AUC (Option 1) / macro-AUC (Option 2) | as above |
| Q3_magnitudes | calibration slope | MAE has no scale-free null; slope has a declared null at 1 |
| Q4_combined_ev | net cents at the action unit | the decision metric (rule 7) |

Every other metric is **reported and never adjudicated**. A metric that can be
swapped into a cell after seeing results is a multiplicity leak.

**The denominator is FIXED at 24.** Holm must be computed over the declared
family, not over the cells that happen to have p-values. A cell that is
`UNDERPOWERED`, `NO_INCUMBENT_COUNTERPART` (R-237), or otherwise unevaluable
**still occupies its slot in the denominator**. Allowing the denominator to
shrink to the evaluable subset would make a family smaller by failing to measure
part of it — which rewards exactly the wrong thing.

---

## A1.5 ACTION-UNIT WEIGHTING AND DEDUPLICATION

Rule 2 says rows are actions and an evaluator that can attribute one outcome to
several rows must de-duplicate to actions. The frozen doc inherits this but does
not state how the **heads** treat it. Measured previously: 1.99 rows per fill,
max 23.

**Amendment, to be frozen explicitly:**
- **n for every reported head is the ACTION count**, not the prediction count.
  A head predicting on rows must state its action count beside it.
- Head fitting weights each row by `1 / rows_in_generation`, so a generation
  contributes once regardless of how many decision rows it spans — matching the
  weighting already used by the four-arm stack.
- Q4 economics are evaluated at the action unit with first-crossing dedup, as
  `harmful_action_eval` already does. No head may report an action count larger
  than its population's distinct `(slug, side, gen)` count.

---

## A1.6 REMAINING CONSTANTS TO FREEZE

Stated so they are ruled rather than inherited from whatever the code happens to
contain:

| constant | proposed | note |
|---|---|---|
| `UNDERPOWERED_MIN_N` | 100 | conditional-population floor; below it a head is reported UNDERPOWERED with its n |
| ridge `lam` | 10.0 | matches the four-arm stack; no per-head tuning |
| LGBM params | `D.LGBM_PARAMS` / `D.LGBM_VALUE_PARAMS` | pinned, seeds pinned, no capacity search |
| permutation `n_perm` | 2000 | ≥1000 declared; sorted-key consumption (R-234) |
| `PERM_SEED` | 20260828 | pinned |
| declared outputs | `iter011_conditional_value_v1.json` | a run writing nothing must not exit 0 |
| standalone identity | runner **not** in `CODE_IDENTITY_FILES`; imports lattice, modifies none | asserted by falsifier; recorded in the receipt |

---

## A1.7 What this amendment does NOT do

- It does not touch §0 (what iteration 010 established), §4 (population and
  admissibility), or the consumed-day list. Those stand.
- It does not weaken any gate. §A1.2 and §A1.4 tighten.
- It does not change the arm count (2), the head count (4), or the family size
  (24) under Option 1. Option 2 changes Q2's metric only.
- It does not pre-judge a result. If Q1–Q4 come back null under the amended
  design, that is the result and it is reported with the prominence a positive
  would get.

## A1.8 Sequence, once frozen

1. amendment frozen by the USER
2. strict target construction + fence tightening (§A1.2, §A1.3)
3. row-aligned all-action heads + Q4 composition emitting economics, budgets,
   matched-random, increment null, Holm
4. fixed 24-cell evaluator (§A1.4, §A1.5)
5. identity + run guards + falsifiers
6. reviewer re-review, non-fit

Each step red-first. Nothing is fitted or scored until step 6 clears.
