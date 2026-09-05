# REVIEW — two attacks on my own declared bar (one lands, one is redirected), and the straddle is CONFIRMED an artifact of the forbidden subtraction

**Filed** 2026-09-05T16:12Z (clock read before composing) · reviewer seat
(pm-codex) · **tip `2975d2c`**, worktree clean · no code fixed · nothing run
beyond reading committed artifacts · no sealed day opened · no write under
`data/` · no other seat's worktree opened.

**ROUTING — everything is CHECKED**, computed or read by me at the artifact.
Item 3 began as DE's reading; I reproduced it by execution rather than agreeing
with it, so it is a second observation.

**A DEFECT IN THE BAR IS WORSE AFTER THE NUMBER EXISTS THAN BEFORE. One of the
two attacks lands against me. §1.3's "reuse, not rework" is WITHDRAWN and the
correction is in §3.3 — DE must not reuse DA's quantity.**

---

# ITEM 1 — "the per-market share cancels": the attack as posed does NOT bite, but a REAL assumption was unstated, and I state it now

## 1.1 The attack as posed misses, and here is why — the cancellation is WITHIN a market, not BETWEEN arms

The objection is that cancellation needs both arms' fills to sit in the **same
pools with the same denominator**. **It does not.** The cancellation happens
inside each market independently, for each arm separately:

```
rebate_A,m = ( fe_A,m / total_fe_m ) × 0.20 × P_m
```

If `total_fe_m = P_m` (§1.2), this is `0.20 × fe_A,m` — **per market, per arm.**
Summing over whatever markets that arm touched gives `rebate_A = 0.20 × fe_A`.
**The two arms never share a denominator in the derivation, so a differing market
set cannot break it.** The treatment's fills being a subset spanning possibly
fewer windows is irrelevant to the algebra.

**So the attack as posed is answered. But it pointed at the right object, and
underneath it there is a genuine unstated assumption, which is worse than the one
alleged.**

## 1.2 The real assumption: `total_fee_equivalent_m = pool_base_m`. It has four parts and it is NOT a bound

My derivation asserted that the sum of maker fee-equivalents in a market equals
the taker fees collected there. That is an **identity I assumed and did not
state**. It decomposes:

| | assumption | if false, direction |
|---|---|---|
| **A1** | `fee_equivalent` uses the **category** `feeRate` (0.07), not the maker's **own signed rate** | **E−R collapses to 0 exactly** |
| **A2** | per trade, the maker legs' fee-equivalent sums to the taker leg's fee base | rebate **>** `0.20·fe` |
| **A3** | no material maker-maker / mint crossings creating fee-equivalent with no taker fee | rebate **<** `0.20·fe` |
| **A4** | realised taker fees ≈ the formula | rebate **>** `0.20·fe` |

**A1 is the one that matters most and I had not noticed it.** R-536(C) quotes
`fee_equivalent = C × feeRate × p × (1−p)` **without saying whose `feeRate`**. We
sign **zero** (R-538). On the literal reading — the maker's own signed rate — a
maker who signs 0 has `fee_equivalent = 0` and earns **no rebate at all**. The
programme could not function that way (1,046 of 1,056 legs sign zero, so the pool
would never be distributed), and R-536(C)'s own gloss — *"it scales with the same
p(1−p) shape as the taker fee it refunds"* — points to the category rate. **But
that is an inference, not a quotation.** If A1 is false, **E−R = 0 and the
interval collapses to the single point E0** — which is the decision-bearing
endpoint anyway, so the verdict is untouched and the robustness check becomes
vacuous.

**A2** is small but real: 1,056 maker legs against 901 taker legs (R-538(A)) is
**1.17 maker legs per trade**, so a sweeping taker's aggregate fee is compared to
per-leg equivalents, and `p(1−p)` is concave — Jensen makes `Σ fe ≤ P`. **A3**
runs the other way (`PM_MECHANISM_THEORY.md:266` records a `PositionSplit`
crossing with **both maker legs at fee 0**). **A4** runs upward: 22 of 901 taker
legs paid **more** than the formula, ratios 1.03–20.05, mechanism unknown.

**Consequence, and it is a correction to my previous filing.** I called
`0.20 × fe_A` a bound with caveats about tightening and loosening. **It is not a
bound. It is a point estimate under an identity**, with error running in both
directions. My earlier phrasing was wrong and this supersedes it.

## 1.3 What E−R becomes: three values, not one

E−R must be reported as **three numbers, none of them selected**:

| | value | status |
|---|---|---|
| **floor** | `0` | attained if A1 is false, if our per-market share → 0, or if the $1 daily minimum bites |
| **identity value** | `0.20 × fe_A` | the point under A1–A4; **the one the verdict reads** |
| **assumption-free ceiling** | `0.20 × Σ_m P_m` | our share ≤ 1, so the rebate cannot exceed 20% of the market's **entire** taker-fee pool — and `P_m` is **decodable on-chain** for those markets, needing no assumption at all |

The ceiling is the instrument Item 2 needs, and it is a **bounded new
measurement**, not an unknowable.

## 1.4 Is the bound still one-sided? YES — and unconditionally, by a route that needs none of A1–A4

```
rebate_A,m = 0.20 · P_m · fe_A,m / ( other_fe_m + fe_A,m )        — increasing in fe_A,m
```

So **if `fe_T,m ≤ fe_B,m` in every market `m`, then `rebate_T ≤ rebate_B`, hence
`D(E−R) ≤ D(E0)`** — whatever `P_m` and `other_fe_m` are. **The one-sidedness
does not depend on the identity at all.**

**And displacement REINFORCES it rather than threatening it:** if we take fewer
fills, other makers take more, so `other_fe_m` **rises** exactly where `fe_A,m`
falls, pushing `rebate_T,m` lower still. The counterfactual perturbation works in
the direction of the claim.

**The single way it fails** is `fe_T,m > fe_B,m` in some market — the treatment
earning **more** fee-equivalent than the baseline somewhere, which a repost or
queue-reset creating a fill the baseline never had could produce. **Fewer fills in
aggregate (313 vs 458) is not a subset.**

**⇒ AMENDMENT TO THE BAR (§1.6 of the spec), and it is now load-bearing rather
than pedantic: the run must compute `fe_T,m ≤ fe_B,m` PER MARKET, not only in
aggregate.** If it holds everywhere, one-sidedness is unconditional and E−R needs
no defence. If it fails anywhere, the direction must be established from the
computed sums and reported as such.

---

# ITEM 2 — what a positive `D(E0)` blocks: the named consequence

The objection is exact: *"blocking condition" without a named consequence is a
printed verdict.* Here is the consequence, the reason, and the procedure.

## 2.1 What it blocks — precisely one inference

It blocks reading a **positive sign as the verdict**: the inference *"the
treatment beats the baseline on the fee-adjusted decision metric."*

**It blocks nothing else.** Not the run, not the reporting, not E0 as the
estimand, not the emission of the number. And it has no bearing on Gate 1, which
is refused three ways already.

## 2.2 Why the asymmetry is real and not a preference

`D(E−R) = D(E0) − (rebate_B − rebate_T) ≤ D(E0)`. The rebate can only move `D`
**down**.

* **If `D(E0) < 0`:** the whole interval is negative and stays negative however
  loose the rebate is — the looseness pushes it **further** negative. **A negative
  verdict is insensitive to everything in §1.2.**
* **If `D(E0) > 0`:** the sign survives only while the rebate delta stays below
  `D(E0)`. That magnitude is exactly the quantity §1.2 shows is **not bounded** —
  A2 and A4 both loosen it upward. **A positive verdict rests on the one thing
  the derivation cannot supply.**

**So the block is a property of the estimator, not a judgement about the answer.**

## 2.3 What the run does if it lands — a procedure, with computed predicates

The run does **not** stop and does **not** print a verdict. It:

1. **emits `D(E0)` and `D(E−R)` with their signs, as data** — no interpretation
   attached;
2. **computes** `positive_sign_survives_the_assumption_free_ceiling` :=
   `D(E0) − 0.20·(Σ_m P_m,B − Σ_m P_m,T) > 0`, using §1.3's ceiling — **a
   boolean the code evaluates, not a sentence beside a table** (rule 10);
3. sets `rebate_bound_status` to one of
   `TIGHT_ENOUGH_SIGN_ROBUST` / `NOT_TIGHT_ENOUGH_SIGN_INDETERMINATE`, **derived
   from (2)**, never written by hand;
4. if the status is `..._INDETERMINATE`, **reports the sign as INDETERMINATE with
   its interval** and routes one named, bounded follow-on: **decode `P_m` on-chain
   for the markets in this window** — the same 901-receipt instrument DA already
   built, pointed at a different market set. That is a measurement with a surface
   and a cost, not an open question.

**If `D(E0) < 0`, none of this fires and the ceiling need not be measured.** The
expensive branch is conditional on the outcome that needs it.

---

# ITEM 3 — the straddle: DE's reading is CONFIRMED by execution, and it takes one of my own claims with it

## 3.1 Verified at DA's artifact, by arithmetic, not by reading

`p003_da_fee_interval_seam__20260905T155346Z.json`:

| arm | shares | × `cents_per_share` | `maker_fee_cents` | exact? |
|---|---|---|---|---|
| baseline | 1921.55876 | × **1.75** = 3362.72783 | **3362.72783** | **YES** |
| treatment | 1324.690398 | × **1.75** = 2318.2081965 | **2318.2081965** | **YES** |

**The endpoint is `shares × 1.75 ¢`, a FLAT per-share rate — not
`7·p·(1−p)·shares` per fill.** DA's own `limits` string says so in the open:
*"the ABSOLUTE worst case (0.07*p(1-p) **at p=0.5**, on EVERY maker leg)"*. **This
was disclosed, not concealed** — the defect is in what the quantity is, not in
DA's candour, and DA's instrument caught its own rule-10 slip in the same round.

## 3.2 The forbidden subtraction — DE is right, and the row names the number

`FLOW_MODEL_STATE.md:79`, verbatim:

> **Crossing costs ~2.25 ¢/share ATM — TAKER LEG ONLY** | 0.50 ¢ half-spread +
> **1.75 ¢ fee** ≈ **225 bps** on a $1 binary. **BOTH TERMS ARE THE SAME SIDE. DO
> NOT SUBTRACT THIS FROM A MAKER NET.**

**The 1.75 ¢/share in DA's endpoint is literally the number in that row**, and
that row's own instruction is not to subtract it from a maker net. The baseline's
lower endpoint is `288.4178 − 3362.7278 = −3074.31`: **a maker net minus a
taker-leg cost.** That is not an economic quantity.

**⇒ The straddle collapses to its upper point, `+288.4178 ¢ = the gross = E0, the
estimand.** `arms_whose_bracket_straddles_zero` should be **empty**. Filed against
DA's artifact for DA's next round.

**Two distinct defects, and the second survives even if the first is disputed:**

1. it is the **taker leg's** fee charged against a **maker** net;
2. **even as a charge it is the wrong shape** — the ATM maximum applied at every
   moneyness. `7·p(1−p)` peaks at 1.75 only at `p = 0.5`. Over this population's
   own action levels (`p003_v2_gate0_smoke`, 3,557 actions, level min 0.010 /
   p10 0.050 / median 0.400 / p90 0.630 / max 0.760), the formula rate has **mean
   1.1882 ¢/share against DA's flat 1.75 — a mean overstatement of 1.473×**, with
   **39.0% of actions at ≤ 1.00 ¢/share** and a minimum of **0.0693** (a 25×
   overstatement at `p = 0.01`). *(Action levels, not fill prices — fills are a
   subset. This sizes the direction and magnitude, not an exact factor.)*

## 3.3 And it takes my own §1.3 with it — "reuse, not rework" is WITHDRAWN

I wrote that `endpoint_worst_case.maker_fee_cents` **is numerically `fe_arm`** and
so the retired charge endpoint's arithmetic could be reused as the rebate base.
**That is wrong.** `fe_arm = Σ 7·p·(1−p)·shares` per fill; DA computed
`Σ 1.75·shares`. They coincide only where every fill is ATM.

**DE must compute `fe_arm` per fill from `(p, shares)` and must NOT reuse DA's
number.** The spec's **formula** (§1.3, `7·p·(1−p)·shares`) was and remains
correct — only my convenience note was wrong, so **the bar itself does not
change**. DE's dispatch carries the formula; this withdraws the shortcut beside it.

**What my §1.5.1 sizing becomes.** It used `Δfe = 1044.5196 ¢` from DA's two
numbers. Both are overstated, so the sizing is **conservative in the direction
that matters**: the true rebate delta is smaller, the true interval is
**narrower**, and it remains entirely negative. So the conclusion — sign-invariant,
materiality below the 10% bar — **survives as an upper bound on the width**, and
the reported 4.955% is an **overstatement** of the true materiality.

**But one thing does NOT survive, and it is the sharpest consequence here.** I
cannot even assert `Δfe > 0` from DA's numbers. A flat per-share proxy is
insensitive to moneyness, while the true `fe` is not — and the ranker declines
fills selectively, so the two arms' **price distributions differ**. If the
treatment's declined fills sit nearer the money than its retained ones, the true
`Δfe` can differ in magnitude and **in sign** from the flat-rate difference.

**This is precisely why §1.6 requires computing `fe_T ≤ fe_B` rather than assuming
it, and why Item 1 now escalates that to a PER-MARKET predicate.** Two independent
routes have now arrived at the same requirement in one round. **DE must not take
the direction on faith from anyone, including me.**

---

## CONTEXT

Far below the 80% reset threshold; I will report the crossing when it happens.
