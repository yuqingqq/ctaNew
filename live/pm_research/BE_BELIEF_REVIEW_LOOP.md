# BE_BELIEF_REVIEW_LOOP — charter

LOOP TERMINATED BY RULING R-69 — 2026-08-23 — **`REFUTED_IN_SUBSTANCE` at Revision 3**

> **THE LOOP DID NOT CONVERGE. IT WAS STOPPED.** Three iterations, **three
> consecutive `REFUTED_IN_SUBSTANCE` verdicts**, stop counter **0 of 2 — never
> reached.** No Revision 4 will be produced.
>
> **What the third iteration established, and it is about the DOCUMENT, not the
> review.** Iteration 3 ran against a **frozen** artifact — Revision 2, 1,585
> lines, `sha256 fd277272a448`, recorded **before** the lenses were dispatched and
> verified by all three independently — under a **frozen** lens set. Compare SP,
> where freezing the instrument took it 12 → 3 → 0. **Here freezing changed
> nothing.** That exonerates the instrument and implicates the plan: BE_BELIEF's
> defects are in the document, not in how it is being read.
>
> **So iterating was the wrong instrument for this problem.** A fourth rewrite
> would be a fourth attempt to repair by revision what three independent lens-sets
> have called unsound in substance — and the loop's own record shows why that
> fails: **48% of iteration 2's findings were damage the previous rewrite did.**
>
> **What is NOT refuted.** `Identity` — ship the venue book unchanged — was
> **never challenged by any lens in any iteration**: nine independent lens-runs,
> zero disputes of the recommendation. Also standing: §1.2's ownership ruling
> (`E[Y|state]`, never `E[Y|state,FILLED]`), §1.3's cycle break, §2.1's staleness
> finding, §5.1's tick floor, and §7's commodity self-description.
>
> **A successor STARTS FROM THIS REFUTATION.** It does not inherit a plan whose
> only claim to soundness would be that iteration stopped before finding the next
> defect. Nothing blocks: E-X1 is `VOID` under R-56 and its successor is
> calendar-blocked.
>
> **BE also killed its own in-flight measurement on this ruling.** A common-vintage
> pooled/verdict fit pair was running to make Revision 3's comparison controlled.
> With no Revision 4 the numbers have no destination, and lens 2 had already
> isolated the confound to a single day — so under **R-61 clause 3, the same rule
> that closed DE_OP**, a measurement whose findings can no longer change a
> decision should not be run. Killed at ~1,175/6,342; both receipts untouched.

Adversarial review loop over `plans/BE_BELIEF_PLAN.md`. Started 2026-08-23 by
**BE**, on the coordinator's ordering under R-54: **review the plan BEFORE
running E-X1**, because reviewing a plan after running its protocol is the wrong
order.

## Why this loop exists, and why it is late

`BE_BELIEF_PLAN.md` is the **last unreviewed plan in the programme.** Every other
plan — SP, DA inventory, the two DE plans, EV-Replay, OP, EV-Gates — has been
through an adversarial loop, several to a two-clean stop. This one has not, and
it has been overdue since this morning.

**It is also load-bearing for a FROZEN protocol about to run.**
`EX1_PREDICTION_PROTOCOL.md` asks whether our forecast beats the venue book as a
predictor of resolution — the programme-identity question, alpha versus pure
market-making. The plan's **§1.2 self-defeat argument** is what makes that
question strict: *"a belief that tracks the book cannot profit from disagreeing
with it."* **If §1.2 is wrong, E-X1's framing is wrong**, and the loop has to
establish that before the protocol runs rather than after.

## The stake is unusual: the plan has already been contradicted by its own probe

`be_belief.py` productionised the plan's §12 steps 1–3, and the results do not
match the plan's §A appendix:

| quantity | plan §A (1,645 windows, 2 days) | probe (4,762 windows, 4 days) |
|---|---:|---:|
| core `b̂` | **1.145** ± 0.042 | **1.037** |
| core `â` | +0.122 | **−0.006** |
| deployable OOS Δlog-loss | **−0.0006** (beats book) | **+0.00013** (indistinguishable) |
| `affine_ab` OOS | −0.0020 (best) | **+0.00142 (WORSE)** |
| `isotonic10` OOS | +0.0012 (worse) | **+0.00481 (worse)** — replicated |

So the loop is not auditing a plan against opinion; it is auditing a plan
against a measurement that already exists and disagrees with it.

## Lens set — FROZEN, R-61 clause 1

**Declared here and not changed again for the life of this loop.** R-61: a loop
whose reviewers may extend the instrument mid-run is **a gate whose bar moves**,
and the streak counts only iterations run under one set.

**BE declares the streak restarts at iteration 3, and says why rather than
quietly benefiting.** Lens 3's scope moved between iterations 1 and 2 — it was
*"adversarial concrete cases and E-X1 readiness"*, and when E-X1 went `VOID`
under R-56 BE re-pointed it to *"adversarial + Revision 1's own new mechanisms"*.
That was an instrument amendment made without noticing it was one. Under R-61
clause 2 it vacates the streak. **The streak was 0 of 2 in any case, so declaring
this costs nothing today — which is exactly why it should be declared today
rather than the first time it is expensive.**

**The three lenses, fixed:**

1. **Completeness — and WHAT WAS NEVER ATTEMPTED.** Both halves; the second is
   explicit per R-53.
2. **Statistical and methodological soundness.**
3. **Adversarial concrete cases, and whether the plan's own mechanisms survive
   contact.** (E-X1 readiness is struck: the protocol is `VOID` and there is
   nothing to be ready for.)

**No fourth lens.** Adding one is permitted but restarts the count, and that cost
is paid deliberately or not at all.

---

## Lenses (original wording, retained as provenance)

Three independent reviewers, dispatched together:

1. **Completeness — and WHAT WAS NEVER ATTEMPTED.** The second half is explicit
   per the standing lesson from R-45/R-51/R-54: *review loops audit what exists;
   nobody audits what was never attempted*, and two of nine mitigation channels
   existed only because someone asked that question. For each absent hypothesis
   the reviewer must distinguish **rejected with stated evidence** from **never
   considered**.
2. **Statistical and methodological soundness** — the estimator, the drift/FLB
   separation, clustering and power, the single-cluster walk-forward, and any
   place a null is stated as an established fact rather than a failure to reject.
3. **Adversarial concrete cases and E-X1 readiness** — does §1.2 survive, does
   the plan survive its own probe, and can E-X1 proceed on this plan as written.

## Finding classes

`MUST-FIX` — would propagate into code, or contradicts a measured fact ·
`SHOULD-FIX` — a gap that invites a future defect · `NOTE`.

**Every finding needs a concrete failure case.** Deference is the failure mode:
sibling loops on comparable documents returned 20–62 MUST-FIX per iteration, and
a thin review here will be read as a clean bill on the one plan that never had
one.

## Binding constraints on fixes

- The plan stays **DESIGN**. No fix may record as settled what a measurement
  could settle.
- `FLOW_MODEL_STATE.md` wins on facts; `contracts.yaml` v22 wins on types.
- **`EX1_PREDICTION_PROTOCOL.md` is FROZEN and APPEND-ONLY (R-28).** If the
  review finds a defect that reaches the protocol, the remedy is an **annotation
  beside it and an escalation**, never an edit — and under R-38 clause (d) an
  amendment buys an obligation to re-measure, never a verdict.
- BE may **refute** the plan. It is a design document, not a result.

## Stop rule

Two consecutive iterations with **zero confirmed MUST-FIX**. Verdict per
iteration: `DEFECTS_FOUND_AND_APPLIED` | `CLEAN` | `BLOCKED(reason)`.

Loop status is **computed, not asserted**:
`python3 live/pm_research/check_loop_log.py live/pm_research/BE_BELIEF_REVIEW_LOOP.md`

---

## Iteration log

### Iteration 1 — 2026-08-23 — **CLOSED** — verdict: `REFUTED_IN_SUBSTANCE`

3 of 3 lenses · **~70 MUST-FIX-class findings** · stop counter **0 of 2**.

**The plan's architecture survives; its inference does not.** What holds: §1.2's
ownership ruling (`E[Y|state]`, never `E[Y|state,FILLED]`), §1.3's cycle break,
§7's commodity self-description, and §3.1's isotonic rejection — **the only §A
claim the probe confirms**, and it replicates 4× larger.

#### The headline fails the plan's OWN declared primary unit — BE verified

§6.1 rule 4: *"the correct unit for power is windows, and above that, **days**."*
§3.3 then claims `b̂ = 1.145 ± 0.042` rejects `b = 1` at *"≈ 3.5 window-clustered
σ"* and calls it *"the strongest thing that can honestly be said."*

Day-clustered on the probe's four days — `0.9894, 0.9917, 1.1203, 0.9530`:

```
mean 1.0136   sd 0.0733   se 0.0367   n_days 4
day-clustered 95% CI = [0.897, 1.130]      CONTAINS 1
```

The plan's central claim does not survive the unit the plan itself declares
primary. Not a reviewer's alternative standard — **the plan's own.**

#### Refuted by its own probe, with nothing in the document reflecting it

| plan claim | probe |
|---|---|
| core `b̂` **1.145** | **1.037**; **0.989** on the one shared day |
| core `â` +0.122 | **−0.006**, per-day sign-alternating |
| deployable OOS **−0.0006** | **+0.00013** → `INDISTINGUISHABLE` |
| `affine_ab` −0.0020 (best) | **+0.00142**, CI excludes 0 on the **wrong side** |
| buy-Up-at-ask positive in every bucket ≥0.4 | **negative in all ten** |
| heterogeneity *"driven entirely by SOL"* | btc **0.948** ties sol; btc/eth **disagree in sign** about `b−1` |

`grep` for `be_belief` / `BE_BELIEF_RESULTS` / `1.037` in the plan returns
**nothing** — no pointer to its own productionised probe. And §11's contract
block, the text destined for `contracts.yaml`, still ships the superseded
−0.0004 gain, the **withdrawn** 6–8 c ATM spread, a `version: 12 → 13` bump
against a file at **v22**, and a `TopOfBook` that cannot be built from the source
the same block mandates.

#### The never-attempted half, which ordinary review would not have reached

Zero hits in 1,165 lines for: `taker`, `tier`, `Hawkes`, `f_r`, `queue`, `EWMA`,
`rolling`, `time-of-day`, `volatility`. **The plan considers exactly two
conditioners — moneyness and `r`.** Never considered at all: liquidity-tier
belief (a 2-group liquid/thin split has ~3.5× the per-cell sample of the 7-way
fit the plan routes to, and is available today); signed taker flow (a fully built
flow model exists and appears nowhere); session/time-of-day (while §4.3's whole
pin rests on "a 20-hour rally" and §6.3's mechanism is a participant mix that is
a function of session); a non-stationary `b` (§7 proposes a decay **monitor** with
no corresponding estimator); shrinkage toward 1 as a third gate outcome.

#### THREE DEFECTS PROPAGATE INTO THE FROZEN E-X1 PROTOCOL

Handled as annotations beside, never edits (R-28), and escalated:

1. **The 1.45× design effect was measured on a logistic slope and frozen into
   E-X1 §3, whose primary statistic is a paired ΔBrier** — where `y` is constant
   within cluster, so the factor approaches **√5 ≈ 2.24**. Measured naive→day is
   ≈2.2×. E-X1 inherited a number from a different estimand.
2. **No MDE exists anywhere in the plan**, while E-X1 §6.1 mandates one. Computed
   from the probe's between-day dispersion: **0.00042 log-loss at the 7-day
   gate** against a claimed 0.0006 effect and an observed **+0.00013 of the wrong
   sign**. The §12 step-5 gate returns `Identity` with high probability whether
   or not the effect exists.
3. **§7 claims the recalibrated book is a mandatory baseline "enforced by
   construction"; E-X1 §3 scores against the RAW mid.** The guard is vacuous —
   currently harmless only because recalibrated ≡ raw.

#### And E-X1 cannot run at all — verified, escalated as `Q-BE-13`

Independent of the plan: the paired population is **0 of 5,796 rows** (route_a_v1
ends 2026-08-20 13:50; the `clob_v3_1` era opens 14:50:21), `n_oos_days: 1`
against a `VOID` threshold of 3, and `route_a_v1` **carries no probability**, so
the challenger `p̂` does not exist.

**BE's own error, third instance of the class:** BE verified the corrected anchor
**exists** and reported the precondition *cleared*. Pairing, day count and the
existence of a probability were never checked. Annotated in the protocol §8a.1.

#### Direction for Revision 1

The plan should be **rewritten around `Identity` plus a monitor** — its own
§10.13 escape hatch, which is where its probe already points — rather than around
a 1.145 that four days have not reproduced.

#### working notes, superseded by the CLOSED iteration-1 entry above

Three reviewers dispatched (completeness+never-attempted / statistical /
adversarial+E-X1-readiness). Findings pending; nothing applied.

**Recorded before the reviewers report, so it is not later credited to a lens:**
BE already knows from running `be_belief.py` that the plan's headline `b̂ = 1.145`
does not hold on the larger sample, that the drift `â` the plan treats as a
persistent up-drift **alternates sign per day**, and that the plan's §10.3 threat
— *"it is the rally… the single largest unresolved threat, and nothing in the
current sample can settle it"* — has **partially materialised**. The reviewers
were told this and asked what survives.
### Iteration 2 — 2026-08-23 — **CLOSED** — verdict: `REFUTED_IN_SUBSTANCE`

Reviewing **Revision 1**, not Revision 0. Three lenses dispatched together
(completeness+never-attempted / statistical / adversarial+own-new-mechanisms).
Each was told the highest-value finding is a defect in **Revision 1's own new
reasoning** — §0, §3.3's withdrawal, §6.5's gate, §11's banner, §13 — because
re-demolishing Revision 0 is worth nothing.

**PRE-REGISTERED BEFORE THE LENSES REPORT, so neither can be credited to one.**
BE checked its own new arithmetic while they ran and found two defects in
Revision 1:

1. **The day-clustered CI that Revision 1's whole argument rests on is a
   t-interval on 3 df, and the plan did not say so.** Same four numbers under a
   normal approximation give `[0.942, 1.085]`, which **excludes 1** — Revision
   1's central conclusion *reverses* on the convention. t is the correct choice
   at n=4 (sd estimated from the same four points) and the unit was fixed by
   §6.1 rule 4 *before* any day-clustered number existed, so the choice is
   defensible — **but undeclared it is the identical defect Revision 1 charges
   Revision 0 with**:choosing among defensible conventions after seeing which one
   supports the conclusion. Fixed by showing both and defending the choice; P2
   now requires **both** intervals to agree, and returns INSUFFICIENT otherwise.
   The honest summary weakens to *"whether b=1 is rejected depends on a
   convention four clusters cannot settle"* — `Identity` on **insufficiency**,
   not on a demonstrated null.

2. **§6.5 P1's rationale was arithmetically false.** It claimed 14 days is "the
   first horizon where the MDE falls below the claimed effect". But
   `MDE(7)=0.00042 < 0.00060` — **7 days already was.** Worse, it powered
   against a number §0 withdraws; against the *observed* +0.00013 the horizon is
   **73 days**, and the quantity is currently the wrong sign, so the study would
   be powered to establish that the recalibration **hurts**. P1 stays at 14 as an
   *interpretability floor* with the power claim removed.

**This also withdraws an iteration-1 finding of BE's own:** that Revision 0's
7-day gate "returns `Identity` with high probability whether or not the effect
exists". It does not — its power against the claimed effect was adequate. The
gate's failure was that **the claim was not reproduced**, which is a different
failure, and iteration 1 named the wrong one. Under R-38(d) none of this moves
the verdict: `Identity` was and remains the recommendation, and a 73-day honest
horizon makes leaving it harder, not easier.

#### Lens 2 (statistical) reported — **13 MUST-FIX**, and the first one is BE's

**BE independently re-derived M1, M2, M3 and M5 before accepting any of them.
All four hold.**

**M1 — BE ASSERTED AN INTERVAL RELATION INSTEAD OF COMPUTING IT, AND IT WAS
FALSE.** Revision 1's brand-new §3.3 box says the z-interval `[0.942, 1.085]`
*"**excludes 1** — and Revision 1's central argument reverses."*

```
0.94176 < 1.00000 < 1.08544        ->  IT CONTAINS 1
```

**How BE produced it.** The verification script printed the literal string
`"<-- does NOT contain 1"` rather than evaluating `lo < 1 < hi`. **A label, in
the position of output.** BE wrote that line in the same hour it fixed
`check_loop_log.py` for asserting instead of detecting, and filed `Q-BE-16`
generalising the lesson. This is R-42's defect — *the check does not ask the rule
what it is; it makes the rule reveal it* — committed by the plane that had just
restated it, inside the instrument used to check its own work.

**And the error runs AGAINST interest, which is the part worth recording.** Both
conventions contain 1, so the conclusion is **robust** to the convention — a
*stronger* result than Revision 1 claimed. BE manufactured a false fragility,
then wrote a long box confessing to Revision 0's convention-shopping sin. **The
confession was the error.** Over-correction is not a safe direction to fail in;
it is just a different way to put a false statement in a plan.

Consequence: **P2's new "by BOTH the t and normal intervals" clause is a no-op** —
`t(ν) > 1.96` at every ν, so the t-interval strictly contains the z-interval
(62.4% wider at n=4, 10.2% at n=14). "Excludes 1 by both" ≡ "excludes 1 by t".
The clause BE added to fix the defect does nothing, because the defect was not
real.

**M3 — the "estimator is not reproducing" claim is refuted by a variance
decomposition.** Re-derived: mean within-day sampling variance `0.005302` vs
observed between-day `0.005374` → **`τ̂² = +0.000072`, `τ̂ = 0.0085 ≈ 0`.** There
is no measurable day-level variance component; a perfectly stable `b` produces
exactly this alternation. The correctly-pooled window-clustered interval is
**`[0.989, 1.085]` — contains 1 at HALF the width.** `Identity` survives on a
*better* argument than the one Revision 1 gave it: the interval is wide because it
burns 3 df estimating a variance that is zero, not because days disagree.

**M2 — the interval is centred on a different estimator than the point estimate
it is attached to.** Its centre is the *unweighted* mean `1.0136`; the headline is
the pooled MLE `1.0367`. The 2026-08-23 stub is **4.6% of rows and 25% of the
interval's weight**. Dropping it: `[0.848, 1.220]` — which **contains 1.145**. So
Revision 1's second load-bearing claim, that the probe refutes 1.145, evaporates
on removing one 3.2-hour fragment.

**M5 — every "day-clustered 95% CI" in evidence is `[min, max]` of three
numbers.** At 3 clusters `P(all three draws = min) = 1/27 = 0.037 > 0.025`, so the
2.5th percentile *is* the minimum. True coverage **0.75, not 0.95**; "CI excludes
0" ≡ "all three days share a sign" = a **25%** event under the null, quoted at 5%.
**This one propagates into code** (`be_belief.py::cluster_bootstrap_delta`) and
P3 mandates the same machinery with no studentization.

Remaining lens-2 findings taken as reported pending re-derivation: M4 (no shared
day with Revision 0 — disjoint populations 4h10m apart across an era change, so
the 0.989-vs-1.113 comparison is void), M6 (§6.5's directionality probe refuses
its own gate — the criterion is mis-specified), M7 (gate `α ≈ 1.5e-06`, ~0 power
against the plan's own point estimate), M8 (P4's quartiles undefined at n=14 and
unimplementable from the receipt), M9 (P5 conflicts with P1), M10 (§6.4's warm-up
rule is unimplemented, and applying it flips §0's headline from
`INDISTINGUISHABLE` to **WORSE**), M11–M12 (§5.2 and §6.3 unrevised and now
self-contradictory), M13 (the MDE and the "73 days" come from a 2-df sd whose own
95% CI spans 12×, putting the horizon in **[11.6, 1696] days**).

#### A process defect BE caused: the document changed under the reviewers

Lens 2 reported *"the plan changed under me mid-review"* — BE applied its two
self-found corrections at 18:27 while three lenses were reading. The reviewer
absorbed it correctly, but that is luck, not method: **a review of a moving
document is not a review of anything.** No further edits to
`BE_BELIEF_PLAN.md` until all three lenses report; findings will be applied in
one pass. Recorded so the next iteration freezes the artifact first.

#### Lens 1 (completeness / never-attempted) reported — **19 MUST-FIX**

**BE re-derived C1, A4, A4b and F5 before accepting them. All four hold.**

**C1 — THE MOST DANGEROUS LINE IN THE DOCUMENT, AND IT IS NEW IN REVISION 1.**
§0 line 73 reads *"Formally, option **(a)** as the deployed map."* §1.1's table:

| option | §1.1 verdict |
|---|---|
| **(a) stream-anchored `p̂`** | **reject as the level** — loses at every horizon, **+0.0201 Brier vs book** |
| **(b) book as-is** | **reject** — *"a belief equal to the price generates zero disagreement"* |

**`Identity` is option (b). BE labelled it (a).** An implementer reading §0 wires
`p_hat` to the stream forecast — the model measured to lose to the book through
three σ generations. And §11's contract `notes` still open *"produces the venue
book RECALIBRATED, not our own forecast"* and *"the recalibration IS the edge or
there is none"*: **three statements of deployed behaviour in one document, none
of them `Identity`.** Revision 1 changed the recommendation and did not sweep the
document for it.

**C2 — and the option Revision 1 actually adopts is rejected by §1.1 for a reason
`FLOW_MODEL_STATE.md` refutes.** (b)'s reject reason is *"All P&L then comes from
spread/rebate."* But spread capture net of adverse selection is **negative on both
verdict coins** (btc −0.532 c [−0.797, −0.287]; eth −1.243 c, §1e), the rebate is
**`Unavailable`** (§2), and the FLB edge is in the **Withdrawn** table (§4). So
§1.2's *"the recalibration IS the edge or there is none"* resolves to **"there is
none"** — and Revision 1 has no stated economic rationale for the module. Nobody
re-derived §1.1 after the recommendation flipped.

**A4 — §13's "available now?" column is FALSE on its first row, in the corpus's
own named failure class.** BE wrote *"liquidity tier — **Yes**, tier is already
computed."* Every `tier` identifier in the codebase is data-lane distillation:

```
tier tier1 tier1_code tier1_lock tier1_manifests tier1_pipeline tier1_root
tier1_v4_r12 tier2 tier2_root tier2_v1        <- no liquidity grouping exists
```

`FLOW_MODEL_STATE.md` §5: *"The name is not the definition. **Five** instances,
**two** self-inflicted."* **Six now, three self-inflicted** — and committed inside
the section written to stop inherited errors.

**A4b — the highest-value item in the review, and it is not in §13.**
`FLOW_MODEL_PROTOCOL_V5.yaml:333-335` **freezes**:

```
verdict_coins: [btc, eth]
descriptive_only_coins: [sol, xrp, doge, bnb, hype]
restriction_reason: BRACKET_WIDER_THAN_ANY_SUPPORTABLE_CONCLUSION
```

Two consequences BE missed. **(i)** §6.5 P4 demands sign agreement across **all
seven coins**, importing five that a frozen protocol bars from carrying a verdict —
a plane-order violation inside BE's own new gate. **(ii) THE BTC/ETH-ONLY FIT HAS
NEVER BEEN RUN.** The pooled `b̂ = 1.037` is an equal-window average over a
population that is 5/7 descriptive-only. **The number that would actually be
deployed on the only coins we may trade does not exist anywhere**, and neither
Revision 0, Revision 1, nor §13 noticed.

**F5 — §5.2's null is contradicted by the plan's own probe, at 4.8× the effect.**
§5.2 (unrevised Revision 0 text) concludes *"the map does not need conditioning on
`r`."* The receipt's r-stratified fit:

```
r=270  b 1.035  |  r=240  1.133  |  r=180  1.073  |  r=120  0.956  |  r=60  0.975
range 0.177   vs pooled b_hat - 1 = 0.037   ->   r-variation is 4.8x the effect
a:  +0.046  +0.020  -0.035  -0.061  -0.120   <- monotone, and unremarked
```

**And this is the review-loop failure in miniature:** §13 omits `r` *because
§5.2 rejected it with stated evidence* — evidence the probe has since superseded.
A conditioner moved from *never considered* → *rejected* → *not re-examined*.

**Structural findings taken as reported pending re-derivation:** A1 (§13's option
space is closed under "add a conditioner" and contains no different anchor, output
type, loss, or module deletion), A2 (§13's closing argument forbids §6.5's own P4
and the shrinkage row it exempts), A3 (ordering is backwards — tier tests whether
the pooled estimand *exists*), B1 (**no materiality condition** — a map can pass
all five and move `p̂` by 0.14 pp against a 1 c tick, i.e. **promotion is a pure
increase in outage modes for zero quotable change**), B2–B8, C3 (**a Binance-lead
anchor was never a candidate** though §10.10 names it and `BINANCE_LEAD_PROTOCOL`
was frozen the same day), C4–C5, D1 (**under `Identity` there is no fit, but
§6.4's warm-up gate keys on `fit_n_days` — so on day 1 the module refuses to emit
a belief definitionally equal to a book it can read**), D2, E1–E5, F1 (`link_eq`
invariant never checked — a logit map against `PathLaw`'s link), F2–F4, F6, **F7
(the affine form gives `p̂_up + p̂_down = 1.058` against a book measured
complementary to exactly 0.00000 across 1,081,800 checks — a phantom 5.8 c/set
arbitrage)**, F8–F9.

#### Lens 3 (adversarial / own-new-mechanisms) reported — **12 MUST-FIX**

**BE mechanised MF-2 and re-derived MF-8 and MF-13 before accepting them.**

**MF-2 — THE GATE DECLARED ITSELF DIRECTIONAL AND TWO OF THREE CONDITIONS WERE
NOT.** BE made the rules reveal it rather than asking them:

```
P2  "CI excludes 1"         forward=PROMOTE  mirror=PROMOTE   SIGN-BLIND
P3  "Δ negative, CI ex. 0"   forward=PROMOTE  mirror=IDENTITY  directional
P4  "signs agree"            forward=PROMOTE  mirror=PROMOTE   SIGN-BLIND
```

`b̂ = 0.86`, CI `[0.80, 0.92]`, all coins below 1, Δ negative, 14 days →
**P1–P5 all PASS and the gate promotes a map that CONTRACTS the book toward 0.5**,
the exact inverse of the mechanism §3.2/§6.3 use to justify the form, **and
reports it as confirming them.** This is the defect `ev_gates.py::assert_directional`
was built to catch, committed in the one gate that declares itself immune, by the
plane that built the instrument. **APPLIED:** P2 → `lo > 1`; P4 → `sign = +1`
scoped to the **verdict coins**.

**MF-8 — the §11 block hard-fails `contract_check`, on the four lines the plan
certifies safe.** Mechanism re-derived: the reference check skips a `consumes`
entry only `if str(item) in mods` — and v22's 25 modules include **none** of
`BE-Target`, `SP-Params`, `SP-Instrument`, `SP-Venue`. The §11 conformance note
cites `contract_check.py:130-131` for safety; that is the **producer-closure**
check at 145-150, which *does* skip `SP-`/`BE-` prefixes. **The plan cites a
different check from the one that fires.** And the banner's claim that Defect 4
*"would have failed a checker rather than a reviewer"* is backwards — Defect 4
lives in a `notes:` string no checker parses, while the four that do fail are
unlisted.

**MF-13 — the `drift_intercept` NullPin declares a polarity its own data
contradicts on half the days.** `â` by day: **−0.069, +0.088, −0.093, +0.026**.
The pin declares `bias_direction: PESSIMISTIC` (*"under-states P(Up)"*); on 08-20
and 08-22 `â < 0`, so pinning `a = 0` **over**-states it. Wrong on 2 of 4 days —
the `detect_idealisation`/`PolarityViolated` class verbatim, with the eth
`SKEW_UB/SKEW_LB` precedent already on disk.

**MF-10 — §1.2 DOES NOT SURVIVE, and Revision 1 lists it as load-bearing.**
*"A belief that tracks the book cannot profit from disagreeing with it. So the
recalibration IS the edge or there is none."* `FLOW_MODEL_STATE.md` §1e: a
simulated two-sided **`JOIN_BBO`** maker — whose belief *is* the book — captures
**+0.642 c/share on btc (n=10,294)** and **+0.778 c on eth**. **Profit with zero
disagreement.** The first sentence is a tautology about takers; the "So…" is a
claim about the source of P&L in a market-making programme and it is false. Per
the charter this is load-bearing for `EX1_PREDICTION_PROTOCOL.md`'s framing →
annotation-beside and escalation under R-28, never an edit.

Also reported: MF-5 (the newly-declared fallback names `route_a_v1`, which **carries
no probability** and cannot acquire one without lifting `PRICING HOLD`), MF-6
(under `Identity` the WARMUP gate **refuses the best available forecast and falls
back to the worst**), MF-7 (**nothing refuses a degenerate book** — 4.13% of
1.94 M quotes are >20 c wide, 100% of them in the core domain; `bid 0.15/ask 0.94`
emits `p̂ = 0.545` with `staleness = 0` and no flag), MF-9 (Defect 4 is real but
mis-stated — sizes *are* recoverable via `book` replay, so an implementer acting
on the stated version shrinks the type when the remedy is the opposite), MF-11,
MF-12 (**three of six "survives unchanged" claims refuted** — §5.1's tick floor
has the **opposite sign** in the probe, and §8's 12–20 s staleness is **3–30×**
too loose), SF-14 through SF-19, NOTE-20.

---

#### Iteration 2 verdict: `REFUTED_IN_SUBSTANCE` — stop counter **0 of 2**

**44 MUST-FIX across three lenses.** The recommendation is not the problem.

**All three lenses independently declined to challenge `Identity`.** What each
challenged is the machinery Revision 1 built to justify and govern it — and
between them they refuted the central evidence (no shared population, τ̂≈0, a
false z-interval), the gate's directionality, its scope, its power, its fallback,
its refusal path, its contract block, and three of the six claims Revision 1
listed as surviving.

**Three defects were BE's own new text, and all three are the same class:** a
property **asserted** where it could have been **computed** — the z-interval
relation printed as a string literal, the gate's directionality declared in prose,
and the MDE threshold-crossing stated without dividing. R-42 names this exactly,
BE restated it in `Q-BE-16` the same hour, and then committed it three times in
one document.

**Direction for Revision 2 — and it is not "apply 44 fixes".** A plan whose every
new mechanism fails first review is reporting something about its size, not its
details. `Identity` ships the book. A module that ships the book needs **no
promotion gate, no fallback ladder, no warm-up bound on a fit it does not
perform, and no recalibration contract**. Revision 2 should **delete** that
machinery rather than repair it — most of the 44 findings then have nowhere to
live. What must be *added* is small and specific:

1. **Run the btc/eth-only fit.** `FLOW_MODEL_PROTOCOL_V5.yaml:333-335` freezes the
   verdict coins. **The number that would actually be deployed on the only coins
   we may trade has never been computed** — not by Revision 0, Revision 1, or §13.
2. **A book-admissibility refusal** (MF-7): the one genuinely missing mechanism,
   and the only place `Identity` can emit something indefensible.
3. **Withdraw §1.2** (MF-10) and annotate the frozen E-X1 protocol beside.
4. **Re-derive §1.1** for the recommendation that actually shipped, since §0 now
   adopts an option the table marks *reject*, on a reason `FLOW_MODEL_STATE.md`
   refutes.

**Applied this iteration** (the three dangerous-if-read items only; the rest are
Revision 2's): C1's option label, MF-2's P2/P4 sign-blindness, and §3.3's false
z-interval box.

#### R-60 diagnostic — **the self-feeding signature is present in this loop**

The coordinator required this of DA before iteration 12 (SP loop MUST-FIX flat to
rising at 10, 7, 9, 13, 12). Run here without waiting to be asked, classifying
every iteration-2 finding by **whether the defect lives in text Revision 1
introduced, or in Revision 0 text Revision 1 left standing**:

```
findings in text REVISION 1 INTRODUCED : 26
findings in ORIGINAL text left standing: 28
                             classified: 54     SELF-INFLICTED SHARE: 48%
iteration 1: ~70 (all original)   ->   iteration 2: 54
```

**Nearly half of iteration 2 is damage the rewrite did.** The count fell 70 → 54,
but only because the *original* half is being worked off; the rewrite replaced
what it fixed with roughly its own weight in new defects. Every new mechanism —
§6.5's gate (11 findings), §13 (4), the §3.3 box (2), the §0 rewrite (3) —
generated findings in rough proportion to its size.

**So the coordinator's stop condition is met and the remedy applies: FREEZE THE
DOCUMENT AND REVIEW THE FROZEN TEXT, do not iterate again.** Iteration 3 against
a Revision 2 would, on this evidence, produce another ~25 self-inflicted findings
and measure BE's drafting rather than the plan.

**This confirms the direction already recorded above from the opposite side.**
BE concluded from the *content* of the findings that Revision 2 should **delete**
machinery rather than repair it. The diagnostic says the same thing from the
*rate*: on this document, adding mechanism costs about one finding per mechanism.
A plan that ships the book needs almost none, and the two arguments converge on
the smallest possible Revision 2.

**Revision 2 is therefore scoped as deletions plus four named additions** (the
btc/eth-only fit, a book-admissibility refusal, §1.2's withdrawal, a re-derived
§1.1) — **and then FROZEN before iteration 3**, so iteration 3 reviews a fixed
artifact instead of chasing a moving one. That also closes the process defect
recorded above, where both lenses reported the plan changing under them.


---


---

### Iteration 3 — 2026-08-23 — **CLOSED** — verdict: `REFUTED_IN_SUBSTANCE`

**First iteration run against a FROZEN artifact and a FROZEN lens set.**

```
document : plans/BE_BELIEF_PLAN.md  Revision 2, 1,585 lines
sha256   : fd277272a448ee788ad15bf7ab08c13f606233a6c974e9cda734a5d450173fce
recorded : BEFORE the lenses were dispatched
```

The hash is re-checked when they report. **A mismatch invalidates the iteration
rather than being explained away** — iteration 2's lenses both reported the plan
changing under them, and R-64 showed the SP loop's entire convergence happened in
the two iterations after its instrument was held still.

**Streak restarts here** (R-61 clause 2, declared in the charter above): lens 3's
scope moved between iterations 1 and 2 when E-X1 went `VOID`, which was an
instrument amendment BE made without noticing it was one.

**Reported against MARGINAL VALUE, not zero** (R-61 clause 3). Every lens was
told to mark each finding **decision changed: yes/no**, and that a true finding
which leaves `Identity` shipping unchanged is a NOTE. The loop terminates when
findings would change no decision — so the most valuable report is a short one.

**Note on ordering:** this log is now **oldest-first**, matching DE. It was
newest-first for iterations 1 and 2, which cost a reader the current state three
times — the coordinator's fifth false result was a `tail` landing on the oldest
heading. The whole file was reordered rather than appending iteration 3 in the
opposite direction, which would have produced a mixed-order log that is worse
than either convention.

#### Lenses 1 and 3 reported. **Freeze held — both verified the hash independently.**

**THE SAME CRITICAL FINDING FROM BOTH, ARRIVED AT SEPARATELY. That is the
strongest evidence available that it is real, and it is against Revision 2's
central new evidence.**

**The two receipts are different data vintages, so the btc/eth-vs-pooled table
varies COIN RESTRICTION CONFOUNDED WITH A DAY'S EXTRA DATA.** BE re-derived it:

```
day          all-coin (SUPERSET)   btc/eth (SUBSET)   ratio
2026-08-20              2,082              607        0.292
2026-08-21              7,798            2,237        0.287
2026-08-22              8,004            2,337        0.292
2026-08-23                871            1,942        2.230   *** SUBSET > SUPERSET ***
```

**A 2-coin subset cannot hold 2.23× the rows of its 7-coin superset.** The
all-coin receipt is 03:17; the btc/eth run is 20:53 — and `be_belief.py` itself
changed at 20:30, a third uncontrolled difference. Three days restrict cleanly at
0.29; day 4 is simply not the same data.

**This is the third instance of BE comparing populations that share a calendar
label** — it voided E-X1 under R-56, BE flagged it as iteration 2's MF-4, and
then BE committed it **in the banner written to justify Revision 2.** Lens 3
checked the direction on the two comparable days: OOS Δ is `+0.000101` pooled vs
`+0.001796` btc/eth = **17.8×, not the quoted 8.1×** — so the *conclusion*
survives and is understated, while **every magnitude in the banner is
unattributable.** The "2.66× as dispersed" is **2.14×** with the stub dropped
from both.

**And the monotone hypothesis is contaminated at its terminal point:** the
verdict series' 1.271 sits on the day the comparison column could not see.

#### BE's own new text, and all four are the class BE keeps generalising about

1. **A TAUTOLOGICAL ZERO, and BE drew an instruction from it.** §6.4 reports
   *"crossed/locked 0 (0.00%) — **genuinely dead, do not code for it**"*. The
   census ran over quotes admitted by `0.0 <= bid < ask <= 1.0` — **which excludes
   crossed and locked by construction.** Counting X inside a set defined by not-X
   returns 0 always. `ev_gates.py:244` — BE's own instrument — refuses exactly
   this shape: *"idealisation is a COMPARISON; one arm cannot show it."* And §11
   mandates a `price_change` ∪ `book` union, where a fresh bid meeting a stale ask
   manufactures crossed pairs; the probe reads `price_change` only, where crossing
   is structurally impossible. **The instruction must be struck.**
2. **The ONE new mechanism is inert where it must work.** The admissibility
   refusal is a **width** condition, calibrated on the `>20 c` tail and
   demonstrated with **`bnb`** — a `descriptive_only` coin. ATM spread is **1 tick
   on btc/eth**. So a threshold from that tail passes ~100% of the verdict coins.
   **BE used V5:333 to invalidate every pooled number in the corpus, then
   calibrated its single new mechanism on the five coins that ruling bars.** The
   binding condition on btc/eth is **depth**, not width: `bid 0.54 / ask 0.55,
   size 5 × 5` is one tick wide, both-sided, fresh — and is $2.75 a side, the
   venue's own p10 touch.
3. **`2.4×` where the division gives `2.64×`** (0.219/0.083). Asserted rather
   than computed, in the document where BE fixed that defect twice.
4. **`P(monotone) = 1/24`** is P(strictly *increasing*); either-direction is
   **2/24 = 0.083**. The label and the number describe different events. And
   *"the worst books are the FRESHEST"* is **asserted and never measured** — in
   the revision that deleted a gate for asserting directionality in prose.

#### Two structural findings that change what Revision 3 should be

- **The monitor has no actuator, and the deferral cannot be executed as written.**
  §6.5 requires a promotion protocol *"frozen before the data that would justify
  it is looked at"* **and** requires the monitor to publish that data daily. **Both
  cannot hold.** There is no reachable state in which the protocol is written
  cleanly, because §6.5 deleted the only moment — now — when it could have been.
  Lens 3's fix is Class-C-safe and BE accepts it: a **calendar** trigger (*"at
  `n_days ≥ 30` a promotion protocol MUST be written and frozen, evaluated only on
  data after the freeze date"*) is a commitment, not a threshold on the effect.
  **BE deleted the gate and pre-registration together; they are not the same
  object.**
- **The deleted mechanisms are still live where they are implemented.** Six plan
  references to the deleted §6.5 gate, §12 step 5 still names it *"the go/no-go
  for this module"*, and **`be_belief.py` still prints that verdict** — so at 7
  days the machine-generated receipt and §6.5 give opposite answers, and the
  machine one is what a reader trusts. §11 meanwhile **ships the deleted warm-up
  and omits the added `BOOK_INADMISSIBLE` entirely.** Deleting prose does not
  delete a rule that is implemented.

#### The number that would settle it is on disk and unread

§6.5 makes *"per-coin `b̂`, verdict coins only"* a mandatory monitor output.
`be_belief.py` computes it; it is in the JSON; **no receipt renders it and
Revision 2 did not look**:

```
btc  n=3,573  b=1.0284      eth  n=3,550  b=1.1430      spread 0.115
```

**The verdict headline 1.083 pools one coin at ~Identity with one at 1.143** —
Revision 2's own sentence, *"pooling made the estimator look more stable than it
is"*, applying verbatim at n=2 and not applied.

#### Lens 2 reported. **Freeze verified a third time. It dissolves the headline.**

**M2(b) — "8.1× WORSE" IS NOT A FINDING ABOUT THE VERDICT COINS. IT IS THE
TRAINING-SET SIZE, TO 4%.** A k-parameter MLE fitted on `n_train` is *expected*
to lose out-of-sample by `k/(2·n_train_eff)`. BE re-derived it:

```
mean(1/n_train)   pooled 0.000840   btc/eth 0.002941   PREDICTED ratio 3.500
observed day-mean Delta ratio                          3.645      gap 4.2%
```

And the 8.31 → 3.65 step is BE's own §6.1 rule 4 again: **the point estimate is a
ROW average while the interval under it resamples DAYS.** So the banner's headline
decomposes completely into row-weighting × training-size, **with nothing left
over.** The verdict coins are not worse. They have 3.5× fewer training windows.

**M3 — and τ̂ is NOT ≈0 on the verdict coins; it is 0.147–0.171.** `Q = 12.71` on
3 df, **p = 0.0053**, 75–77% of dispersion real, robust to a 4.4× miscalibration
of per-row information. So §3.3's *"there is no measurable day-level variance
component"* — which BE wrote in iteration 2 as the *better* argument for
`Identity` — is a null stated as a fact, and it is **refuted on the population
Revision 2 declares primary.** Consequence: the window-clustered interval BE
promoted is invalid on the verdict coins, and the honest horizon is **17–22 days**.

**M8 — the monotone series is UNDER-read, and BE declined it for the wrong
reason.** A weighted trend test gives **z = 3.55, absorbing 99.3% of Q**; either-
direction monotonicity is 2/24 = 0.083, not 1/24; and exchangeability is
implausible at τ̂ ≈ 0.15. **The right reason to decline is the confound BE did not
name:** each day's share drawn from the 14:50–24:00 block runs **100% → 38.2% →
38.2% → 29.0% — monotone decreasing, mirroring `b̂`** — and **76% of the trend's
leverage sits on the two partial days.** §13 lists session/time-of-day as *never
attempted, availability Yes*.

**M6 — on the verdict coins the plan's own rule cannot reject isotonic either.**
All four challengers read `INDISTINGUISHABLE`; the rule fails to detect a model
**6.2 milli-nats worse — 10× the effect the gate exists to find.** So §3.1's
rejection, which iterations 1 and 2 both recorded as *"the one §A claim the probe
confirms"*, is scoped to the pooled population and BE never said so.

**M4 — §6.5's monitor quotes POOLED numbers** (+0.00013, ~73 days) in the revision
that declared the pooled population 5/7 disqualified. Verdict-coin analogues:
+0.00105 and **~17** scored days; MDE(7d) is **5.0×** larger.

**M9 — a live code defect with a 93% fingerprint match.** `fit_isotonic_bins` can
emit exactly 0.0; `predict_isotonic` clamps to 1e-12, so **one** test row adds
27.63 nats = +0.01235 on a 2,237-row day — against the observed btc/eth day-21
isotonic Δ of +0.013208. Lens 2 reproduced the mechanism in isolation and flagged
honestly that the fold-level trace did not finish. **Recorded as reproduced, not
confirmed.**

**And BE's own new bug, introduced TODAY (N5) — FIXED.** `be_belief.py report`
raised `UnboundLocalError`: `out_paths(coins)` read `coins` eleven lines before it
was bound. **The edit that fixed a scope defect introduced a scope bug — in the
one code path that re-renders a receipt WITHOUT re-fitting, i.e. exactly the path
M1's remedy requires.** Both receipts now re-render; selftest 28 OK.

**N1 — two errors in one clause.** *"the range, 0.219, is still 2.4× the pooled
`b̂ − 1` of 0.083"*: the division gives **2.64**, and **0.083 is the btc/eth
figure, not the pooled 0.037** — against which it is **5.92×**.

---

#### Iteration 3 verdict: `REFUTED_IN_SUBSTANCE` — stop counter **0 of 2**

**The termination test is marginal value, and it is NOT met.** All three lenses
returned findings that change what gets built, and the largest are against text
Revision 2 wrote.

**What is now stable, and it is worth stating plainly: `Identity` has not been
challenged by any lens in any iteration — nine independent lens-runs across three
iterations, none of which disputed the recommendation.** Every finding since
iteration 1 has been about the *reasons*, the *machinery*, or the *arithmetic*.

**What Revision 2 got wrong is the same thing Revision 1 got wrong, one level
up.** Revision 1 built machinery to govern a conclusion that needed none.
Revision 2 deleted most of it and then **built an ARGUMENT the conclusion needed
none of** — an uncontrolled comparison, a dissolved ratio, a refuted τ̂ claim, a
tautological zero, and a mechanism calibrated on the coins its own citation bars.
`Identity` was already right on the plainest reading available: **the interval
contains 1 under every convention tried, and the module is not where the edge
is.** Everything added to that has been wrong.

**Direction for Revision 3 — all three lenses converged on it independently:
DELETE MORE.** Lens 1 reached it as *fold BE-Belief into DA-Normalize* (under
`Identity` the module's only non-trivial outputs are `TopOfBook`, whose §11 note
says **"OWNER IS DA-Normalize, not BE-Belief"**, and an admissibility condition
*on `TopOfBook`*). Lens 3 reached it as *delete the types* — no `Recalibration`,
no `BeliefWarmupPolicy`, no `RecalibrationForm`, no `drift_intercept` pin — and
every §11 defect disappears with the types that carry them. **Revision 3 keeps: a
`TopOfBook`, a `PriceSummary`, a three-legged refusal (width ∧ both-sides ∧
DEPTH), and a monitor with a CALENDAR trigger.**

#### Addendum — lens 2's final report. **Verdict unchanged; two findings BE had not captured, and one of them refutes a word BE wrote.**

**N3 — "EVERY direction points at `Identity` harder" IS FALSE, and BE wrote it
after checking four directions.** Per-window up-rate:

```
pooled    0.5204   n=4,762   z=+2.82   p=0.005     <- a real up-drift
btc/eth   0.5045   n=1,784   z=+0.38   p=0.704     <- NO drift
```

**The "20-hour rally" premise behind §4.3's pin-`a`=0 rule is a property of the
five DESCRIPTIVE-ONLY coins.** On the verdict coins there is no measurable drift
at all — so §10.3's *"single largest unresolved threat"* (that the rotation is
entirely the rally) is **weaker** on the population that matters, not stronger.
That is a direction in Revision 2's own new receipt pointing **against** its
banner. **"Every" was a universal claim made from a sample of four**, in the same
sentence-family as the tautological zero and the un-divided ratio: a word doing
work no check supported.

**N4 — §10.2 marks a risk `TESTED — rotation survives` on a test with no power.**
The free-intercept CI on `b_high − b_low` is `[−0.169, +0.349]`:

```
half-width 0.259  ->  se 0.132  ->  MDE at 80% power = 0.370
                      MDE (0.370) > the +0.241 it claims to have dissolved
                      and +0.241 lies INSIDE the CI
```

**The interval widened to contain both zero and the original estimate.** Nothing
collapsed and nothing survived — the data cannot separate the two hypotheses.
Striking a risk through as `TESTED` on that basis is the plan's risk register
recording a *failure to reject* as a *result*, which is the defect §6.1 rule 4
and iteration 1's whole verdict were about.

**Also expanded:** §5.2's null was **already false on its own Revision-0 table**
(max pairwise separation **1.73 se** against a claimed "~1.5", and two of five
values outside the claimed "one se of 1.145") — so it was never supported, on any
data, at any time. And §4.4's arm-slope does not reproduce: btc/eth free-intercept
`b_high − b_low = +0.2268` against pooled +0.0474 and against the +0.099 §4.4 says
a free intercept collapses it to.

#### Addendum 2 — lens 2's confirmatory runs landed. **It withdrew one of its own findings.**

**S9 (the isotonic clamp) DROPS from MUST-FIX to NOTE — the defect did not fire.**
A fold-level trace of the actual walk-forward shows PAVA **pooled bins 0–2 into
one block at 0.189394** on the 607-row fold — exactly the case that avoids a
degenerate bin. Zero clamped bins, zero catastrophic rows, all three folds. BE
re-derived the mechanism against its own patched code and agrees.

**So §3.1's isotonic evidence is NOT contaminated**, and day-21's Δ = +0.013208 is
genuine miscalibration — a map predicting 0.189 for every `m < 0.3` — not a
1e-12 artifact. BE's Laplace fix stands as **hardening**, not as a correction to
any number on the page, and the loop log says so rather than letting a fixed bug
imply a fixed result.

**This is the honest-status flag doing its job.** Lens 2 marked S9
*"reproduced in isolation, not confirmed in fold"* and named the competing
explanation (10-bin AIC optimism) at the time it reported. The confirmatory run
then **resolved against its own finding**, and it said so. A review that only ever
adds findings is not measuring anything; this one subtracted one.

**And S1 is now a CONTROLLED EXPERIMENT rather than an inference.** Same code,
same two coins, 42 minutes after the frozen receipt:

```
days 08-20 / 08-21 / 08-22   b_hat and per-day delta REPRODUCE BIT-FOR-BIT
day  08-23                   1,942 -> 2,018 rows;  b_hat 1.2711 -> 1.2772
```

**Only day 4 moves** — which is what the vintage hypothesis predicts and nothing
else does. So the confound is *isolated* to 2026-08-23: simultaneously the
endpoint that makes the series monotone, the only day with a negative OOS Δ, and
the day the pooled receipt saw as a 3.25-hour stub against the btc/eth receipt's
20.9 hours. **The frozen receipt was already stale 42 minutes after it was
written**, which is the sharpest possible statement of why `days_sampled` is not
a vintage.

**S2 and S6 replicate at the second vintage**: day-weighted ratio **3.62×**
against the convention-invariant prediction of 3.50×; row-weighted 8.09×;
day-clustered sd 2.69×; and **all four challengers INDISTINGUISHABLE** — so the
plan's own rule cannot reject isotonic on the verdict coins, confirmed twice.

**Revised tally: 8 MUST-FIX, 3 SHOULD-FIX, 7 NOTE.** Verdict and stop counter
unchanged.

Nothing applied to the plan from iteration 3 while it was frozen. **Revision 3 is
now being written**, and BE is running its own common-vintage PAIR — lens 2
re-ran only the btc/eth leg and killed the pooled one, so the controlled
pooled-vs-verdict comparison the banner claimed to make still does not exist.
Revision 3 is frozen before iteration 4.
