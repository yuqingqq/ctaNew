# REVIEW 174 — the 69, priced. v3's treatment is merely honest, it is repairable tonight, and here are the two result sentences.

**REV 131, 2026-09-11T03:28:09Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`, no arm VALUE on any population day read. Tip at write
time `724b617`. REVIEW 173 landed in origin. **Every number below is computed from the
DESIGN — thresholds, draw counts, cluster count — and none of it needs a result, which is
why it can be written now and only now.**

---

## 1. THE RULING: MERELY HONEST TODAY, ADEQUATE BY MORNING, AND THE GAP IS ONE FIELD

**The coordinator's critique is exactly right and I adopt it: "corrects the prior" is only
meaningful if someone states what the prior BECOMES, and nobody has. A recorded number that
adjusts an unstated prior by an unstated amount is disclosure, not control.**

But the honest answer is neither of the two you offered. It is not *"already adequate"*, and
it is not *"a 69-candidate search cannot be repaired and the result is simply weak"*.

> **A 69-candidate search cannot be CORRECTED. It can be PRICED — and the price is
> computable tonight, before any number exists, without anyone inventing a prior.**

The device that avoids inventing one is to **invert the question**: rather than assert a
prior and derive a posterior, state **the prior a reader would have to already hold for the
best possible result to leave them more likely than not persuaded.** The reader supplies the
belief; the design supplies the threshold. That is a pre-registerable number, and it is the
missing operation behind "corrects the prior".

## 2. THE EVIDENCE CEILING OF THIS DESIGN — COMPUTED FROM THE DESIGN ALONE

Using the minimum Bayes factor (Sellke–Bayarri–Berger), `BF01_min = −e·p·ln p`, which is
the bound **most generous to the alternative**: the evidence for a real effect is *at most*
`1/BF01_min`. Actual evidence is lower.

| what the p-value is | p | **max evidence for a real effect** |
|---|---|---|
| day-cluster sign gate at its **floor**, 7-for-7 two-sided (`2/2⁷`) | 0.015625 | **5.66 : 1** |
| matched-random endpoint at its **floor**, 500 draws (`1/501`) | 0.001996 | **29.65 : 1** |
| an arm that merely **clears Holm's smaller threshold** | 0.025 | **3.99 : 1** |
| an arm that merely **clears Holm's second threshold** | 0.05 | **2.46 : 1** |

**Read the last two rows first.** They are the likely outcomes if anything clears at all, and
they say that **an arm which just clears is worth about 4:1, and an arm which just clears the
second slot is worth about 2.5:1 — before any penalty for the 69.** That is the ceiling of
what seven days can deliver, and it is a property of the design that no result improves.

**Reverse-Bayes — the number that replaces "corrects the prior":**

| if the decisive endpoint is | BF | **prior a reader must ALREADY hold for posterior > 50 %** |
|---|---|---|
| the sign gate at its floor | 5.66 : 1 | **> 15.0 %** |
| the matched-random endpoint at its floor | 29.65 : 1 | **> 3.3 %** |

**And one anchor, offered as an anchor and not as a claimed prior:** a reader who thought
each of the 69 configurations equally likely to be the one real effect holds **1/69 = 1.45 %**.
Carried through the best possible result, that leaves a posterior of **7.7 %** (sign gate) or
**30.4 %** (matched-random). **Neither reaches 50 %.** A reader wanting to be convinced by
this test must bring a prior an order of magnitude above the flat one — and the two arms
**already failed their development screen**, which is evidence pointing the other way.

**Caveats, because these bounds are only honest with them:** `BF01_min` is a bound over all
alternative priors and so is the most favourable reading available to the arms. The two
conjuncts are computed on the same seven days and the same fills, so they are strongly
dependent: the joint evidence exceeds either alone and falls **far** short of their product,
and **no joint number is available because no joint null is declared**. The two arms are
dependent for the same reason. All of it presupposes the p-value is valid, which rests on
the forward days being untouched (REVIEW 170/171) and the certification holding (REVIEW 173).

## 3. WHAT THE FORWARD TEST CAN AND CANNOT DO ABOUT 69

**CAN — and it is doing both:**
1. **Provide a test whose type-I error is unaffected by the selection.** Selection history
   does not inflate the false-positive rate of a test on data that played no part in it;
   the precondition is measured, not assumed (REVIEW 171 §4a: pair fixed 2026-09-05T16:25Z,
   screen ran 09-10 on 09-03..09-06, forward population 09-07..09-13). **The forward test
   does not correct the 69 — it BYPASSES it, for two arms only.**
2. **Deliver an UNBIASED EFFECT SIZE — and this is the part nobody has claimed.** The
   development point estimate is inflated by selection from 69 (winner's curse); **the
   forward estimate is not.** The forward test *is* the winner's-curse correction for
   magnitude. **So the most durable output of this test may not be the pass/fail at all but
   the MAGNITUDE**, and I recommend the forward effect size with its day-clustered interval
   be reported as a primary output rather than as colour beside the verdict. It is the one
   quantity here that selection cannot corrupt and that survives an inconclusive verdict.

**CANNOT:**
- correct, shrink or retire the 69 — nothing retrospective can;
- license any statement about **the family** the 69 was drawn from. The moment a result is
  read as *"this class of cancellation policy works"* rather than *"these two arms did X on
  seven days"*, the 69 returns in full, because the family is precisely what the 69 searched;
- repair the **second attempt**. These arms failed a decision-endpoint screen and are being
  asked again. Holm's m=2 governs the family *within this look*; nothing governs the fact
  that it is look two. It is not correctable and it belongs in the same sentence as any
  p-value, which the freeze headline already does.

**So v3/v5's `holm_m = 2` is CORRECT on the denominator** — and the amendment should cite
the precondition it rests on rather than the claim, because the precondition is what a later
reader must check.

## 4. WHAT TO ADD, BEFORE THE DATA — THREE PRE-REGISTERABLE NUMBERS

This is the whole repair, and it requires no result:

1. **`evidence_ceiling`** — the four rows of §2's first table, in the declaration.
2. **`prior_required_for_posterior_half`** — 15.0 % / 3.3 %, with the 1/69 = 1.45 % anchor
   and its 7.7 % / 30.4 % posteriors.
3. **`failure_likelihood_ratio_at_declared_power`** — §5's table.

With those three, "corrects the prior" stops being a placeholder and becomes a statement a
reader can check and disagree with. **Without them, the 69 is disclosed and not priced, and
the coordinator's word for that — disclosure, not control — is the right one.**

## 5. THE FAIL DIRECTION — AND IT IS NOT SYMMETRIC

A failure's evidential weight against a real effect is `P(fail | H0) / P(fail | H1)`:

| power | P(fail\|H0) at α=0.025 | P(fail\|H1) | **LR against a real effect** |
|---|---|---|---|
| **0.54** (the figure on record, called an upper bound) | 0.975 | 0.460 | **2.12 : 1** |
| 0.40 | 0.975 | 0.600 | 1.62 : 1 |
| 0.30 | 0.975 | 0.700 | 1.39 : 1 |
| 0.20 | 0.975 | 0.800 | 1.22 : 1 |

**A failure here is worth about 2:1 against the arms, and less.** The 0.54 was quoted at a
different N and DE owns the current figure; the table is given across power precisely so the
conclusion does not depend on which one is right. And 0.54 is an **upper** bound for two
independent reasons: the declaration's own (09-05's −4,941 c sign flip contradicts a
consistent per-day effect) **and the winner's curse** (the point estimate that produced it is
selection-inflated), which is the second reason and is recorded nowhere.

**But the 69 makes a failure MORE informative at the FAMILY level, not less** — this is the
asymmetry, and it is the direct answer to the inverse question. A pass is weak evidence *for*
the family because the family is what was searched; a failure is comparatively stronger
evidence *against* it, because 69 configurations were tried, **none** passed the development
screen, and the best two then failed on days that took no part in the selection. **The
forward test contributes about 2:1; the search history contributes the rest**, and the two
compound in a way neither does alone. No joint number is offered — there is no model for it —
but the direction is the opposite of the pass case and should be stated.

## 6. THE TWO SENTENCES THE RESULT SECTION WILL NEED, WRITTEN NOW

**IF BOTH ARMS CLEAR AT N=7:**

> Both arms cleared. Read it against the search that produced them: **69 configurations were
> scored to select these two, and both had already FAILED their development screen.** The
> best result this design can produce is worth **at most 5.7 : 1** on the day-cluster gate
> (at most 29.7 : 1 if the matched-random endpoint carries it; **at most 4.0 : 1 for an arm
> that merely clears Holm's 0.025 and 2.5 : 1 for one that clears 0.05**). A reader therefore
> had to hold a prior of **3–15 % that this specific arm works, before the forward data**, to
> be more likely than not persuaded by it; a flat "one of the 69" prior of 1.45 % leaves the
> posterior at **8–30 %**. **This result raises the standing of two arms. It does not
> establish them, and it says nothing about the family the 69 came from.** The single most
> durable number here is not the p-value but the forward effect size, which is the first
> estimate of these arms that selection has not inflated.

**IF THEY FAIL:**

> Neither arm cleared. **This is weak evidence against these two arms and stronger evidence
> against the family.** Against the arms: at the power on record — itself an upper bound,
> because the point estimate behind it is both sign-flip-fragile and selection-inflated — a
> failure is only about **2 : 1** against a real effect, and less as power falls. Against the
> family: 69 configurations were searched, none passed the development screen, and the best
> two then failed on untouched days; the forward test contributes little alone and the
> accumulated record is what carries. **The verdict is NOT_ESTABLISHED_AT_THIS_POWER, never
> NO_EFFECT**, and the stopping rule licenses no further N — HAZARD's nine days was recorded
> above the data precisely so its predictable failure could not be used to argue for them.

## 7. SCOPE

Computed here: the four evidence ceilings, the two reverse-Bayes thresholds, the 1/69 anchor
and its posteriors, and the failure likelihood ratios across four power values — all from
the declared design (thresholds 0.025/0.05, G=7 two-sided sign floor 2/2⁷, 500 draws → 1/501
floor). **Not computed, and named rather than finessed:** any joint evidence figure across
the two conjuncts or the two arms, because no joint null is declared and the components are
strongly dependent; and the prior itself, which is the reader's and which I decline to
invent — §2's inversion exists so that nobody has to.
