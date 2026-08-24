# BE-Belief — design plan

> # ⛔ REFUTED IN SUBSTANCE — R-69, 2026-08-23. DO NOT BUILD FROM THIS PLAN.
>
> **Three review iterations, three consecutive `REFUTED_IN_SUBSTANCE` verdicts,
> stop counter 0 of 2 never reached.** The loop was **terminated by ruling, not
> closed by convergence** (`BE_BELIEF_REVIEW_LOOP.md`). No Revision 4 exists and
> none will.
>
> **The refutation is about this document, and that was established rather than
> assumed.** Iteration 3 ran against a frozen artifact whose `sha256` was recorded
> before the lenses were dispatched and verified by all three independently.
> Freezing the instrument took the sibling SP loop from 12 findings to 3 to 0.
> **Here it changed nothing** — so the defects are in the plan, not in how it was
> read.
>
> ## What a successor should take, and what it must not
>
> **TAKE — never challenged by any lens in any iteration (nine lens-runs):**
> - **`Identity`**: BE-Belief produces the executable top-of-book **unchanged**.
> - **§1.2's ownership ruling**: the estimand is `E[Y | state]`, **never**
>   `E[Y | state, FILLED]`. Fill-conditioning is BE-FlowAndFills' term; baking it
>   in double-counts the haircut.
> - §1.3's cycle break · §2.1's staleness finding · §5.1's tick floor ·
>   §7's commodity self-description.
> - **A book-admissibility refusal is genuinely missing** and is the only place
>   `Identity` can emit something indefensible — but as `width ∧ both_sides ∧
>   **DEPTH**`, per-coin, since width alone is inert on 1-tick btc/eth.
>
> **DO NOT TAKE — refuted, and each one was believed here for at least one
> revision:**
> - **`b̂ = 1.145`** (Revision 0's headline). The probe reads 1.037 pooled; the
>   day-clustered interval contains 1 under **every** convention tried.
> - **Any claim that the verdict-coin fit "argues harder"**. Its `8.1×` is row-
>   weighting plus 3.5× fewer training windows — 4% from the value predicted by
>   estimation optimism alone — measured across **two different data vintages**.
> - **"τ̂ ≈ 0 / no day-level variance component."** On the verdict coins
>   `τ̂ = 0.147–0.171`, `Q = 12.71`, `p = 0.005`.
> - **§5.2's "`b` does not depend on `r`."** False on its own table (1.73 se) and
>   on the pooled receipt (2.32 σ, a floor).
> - **§10.2's `TESTED — rotation survives.`** MDE at 80% power is 0.370 against a
>   +0.241 effect that lies **inside** the interval.
> - **Any automatic promotion rule.** Every version was sign-blind, underpowered,
>   or both; the last one survived in code after the prose deleted it.
>
> ## The generalisable finding, which is the part worth keeping
>
> **Revision 1 built machinery a conclusion needed none of. Revision 2 deleted
> most of it and built an ARGUMENT the conclusion needed none of.** `Identity` was
> correct on the plainest reading available from the start — the interval contains
> 1, and the module is not where the edge is. **Every elaboration of that reading,
> across three revisions, was wrong.** 48% of iteration 2's findings were damage
> the preceding rewrite introduced.
>
> Nothing blocks on this: E-X1 is `VOID(NO_PAIRED_POPULATION)` under R-56 and its
> successor is calendar-blocked. Everything below this banner is **provenance**.

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


Program P-2026-003 (Polymarket 5-min crypto binaries). Planner deliverable:
recommendation + canonical contract. **No implementation in this document.**

> # ⛔ REVISION 2 IS FROZEN FOR REVIEW — 2026-08-23
>
> **No edits until iteration 3 closes.** Both lenses in iteration 2 reported *"the
> plan changed under me mid-review"* — BE applied corrections at 18:27 while three
> reviewers were reading. They absorbed it, but **a review of a moving document is
> not a review of anything**, and R-64 showed the SP loop's entire convergence
> happened in the two iterations after its instrument was held still.
>
> **The freeze is DETECTED, not declared.** The sha256 of this file is recorded in
> `BE_BELIEF_REVIEW_LOOP.md` before the lenses are dispatched and re-checked when
> they report. A mismatch invalidates the iteration rather than being explained
> away.
>
> Findings from iteration 3 are applied to **Revision 3**, after the iteration
> closes — never during it.

> ## Revision 2 — 2026-08-23 — the fit nobody had run, and it argues HARDER for `Identity`
>
> **`FLOW_MODEL_PROTOCOL_V5.yaml:333-335` freezes `verdict_coins: [btc, eth]`;
> the other five are `descriptive_only`.** Every belief number this programme has
> ever quoted — Revision 0's `b̂ = 1.145`, the probe's `1.037`, every interval —
> is an equal-window average over a population that is **5/7 barred from carrying
> a verdict**. Revision 2 ran the restricted fit. Receipt:
> `BE_BELIEF_RESULTS__btc-eth.md` (7,123 core rows, 4 days).
>
> | | all coins (pooled) | **VERDICT COINS btc/eth** |
> |---|---:|---:|
> | core `b̂` | 1.037 | **1.083** |
> | core `â` | −0.006 | **−0.062** |
> | per-day `b̂` | 0.989 / 0.992 / 1.120 / 0.953 | **0.827 / 0.973 / 1.147 / 1.271** |
> | day-clustered sd | 0.073 | **0.195 — 2.66× as dispersed** |
> | day-clustered CI, t(3) | [0.897, 1.130] | **[0.745, 1.364]** |
> | ...same, z | [0.942, 1.085] | [0.864, 1.245] |
> | **OOS Δlog-loss, deployable** | **+0.00013** | **+0.00105 — 8.1× WORSE, same wrong sign** |
>
> **Every direction points at `Identity` harder than the pooled figure did.** The
> out-of-sample penalty is eight times larger on the only coins we may trade, the
> day-to-day dispersion is nearly three times larger, and the interval contains 1
> under both conventions with room to spare. **Pooling made the estimator look
> more stable than it is**, by averaging seven series into one — which is the
> same defect §6.3 was warned about from the other side.
>
> **One thing worth carrying, stated as a hypothesis and not a finding.** The
> verdict-coin per-day series is **monotonically increasing** — 0.827 → 0.973 →
> 1.147 → 1.271 — where the pooled series is not. Under exchangeability
> `P(monotone) = 1/4! = 0.042`. **That is four points and it is not significant**;
> it is exactly the kind of pattern that becomes a headline if someone wants one.
> It goes to §6.5's monitor as a question, with the interval convention named and
> the population declared, and nothing is built on it.
>
> *(A related claim was checked and does NOT hold: `b̂` is **not** monotone in `r`
> on the verdict coins — 1.158 / 1.199 / 1.060 / 1.037 / 0.980 declines from
> r=240 onward but r=270 breaks it. The range, 0.219, is still **2.4× the pooled
> `b̂ − 1` of 0.083**, so §5.2's "no `r`-dependence" remains unsupported on this
> population — but it is unsupported, not reversed.)*
>
> **Revision 2's other changes are DELETIONS**, on the R-61/marginal-value
> reading: §6.5's five-condition promotion gate (11 findings, all symptoms of
> automating a once-only decision) is replaced by a **monitor**; §6.4's warm-up
> bound is removed because `Identity` performs no fit and the bound refused the
> best forecast in favour of the worst; §1.2's *"so the recalibration IS the edge
> or there is none"* is withdrawn as refuted by measurement. **One mechanism is
> ADDED**: a book-admissibility refusal, the only place `Identity` can emit
> something indefensible.

> ## Revision 1 — 2026-08-23 — the recommendation changed
>
> Revision 0 recommended **recalibrating** the book by a fitted logit slope
> `b̂ = 1.145`. **Revision 1 recommends `Identity` plus a monitor** — ship the
> book unchanged, and run the rotation estimator as a *watched diagnostic* that
> may promote a recalibration later.
>
> **This is not a reviewer's preference. Three things forced it:**
>
> 1. **The headline fails this plan's OWN declared primary unit.** §6.1 rule 4
>    says *"the correct unit for power is windows, and above that, **days**."*
>    §3.3 rejected `b = 1` at 3.5 **window**-clustered σ. Day-clustered on four
>    days the CI is **[0.897, 1.130] — it contains 1.**
> 2. **`be_belief.py` productionised §12 steps 1–3 and disagrees.** On 4,762
>    windows / 4 days: core `b̂` **1.037** (not 1.145), `â` **−0.006** (not
>    +0.122, and sign-alternating per day), and **every challenger scores WORSE
>    than the raw book out-of-sample.** The deployable map's OOS Δlog-loss is
>    **+0.00013** — indistinguishable, and of the wrong sign.
> 3. **No MDE existed anywhere in Revision 0**, while its own §12 step 5 gates on
>    an effect size. Computed post-hoc from between-day dispersion it is
>    **0.00042 log-loss at 7 days**. *(Iteration 1 read this as "the gate cannot
>    discriminate". **That reading is WITHDRAWN** — 0.00042 < 0.0006, so a true
>    effect of the claimed size **would** have been detected. See the correction
>    under §6.5: the gate's power was adequate for the claim; **the claim was not
>    reproduced.** Detecting the effect actually observed — +0.00013, wrong sign —
>    would take **73 days**.)*
>
> **What survives, unchanged and load-bearing:** §1.2's ownership ruling
> (`E[Y|state]`, never `E[Y|state,FILLED]`), §1.3's cycle break, §2.1's staleness
> finding, §5.1's tick floor, §7's commodity self-description, and §3.1's
> rejection of isotonic — **the one §A claim the probe confirms**, replicated at
> 4× the sample.
>
> **What this costs:** nothing operationally. Revision 0's own §10.13 named
> `Identity` as the escape hatch, and its §12 step 5 already routed there on a
> failed gate. Revision 1 makes the escape hatch the *default* and requires
> evidence to leave it, rather than the reverse. Sections not touched by this
> revision remain Revision 0 text and are marked where they carry superseded
> numbers.

Status of inputs: contracts.yaml v12 defines `BeliefProcess` but has **no
`BE-Belief` module record** (open MUST-FIX M11-4, `PM_STRUCT_ITER11_REVIEW.md:228`).
This plan supplies it.

**§0–§10 are the design. §A is an evidence appendix of measurements run for this
plan (the FLB has never previously been walk-forwarded, and the book series every
prior result used is stale — both are fixed here). §11 is the contract; §12
sequences the remaining work.** Where §A contradicts a number inherited from the
corpus, §A wins and says so.

---

## 0. Recommendation in one line

**BE-Belief produces the executable top-of-book UNCHANGED — `Identity` — under a
standing rotation monitor.** The two-parameter logit map of Revision 0 is
retained in full as the monitor's *estimator*, fitted walk-forward and pooled
across symbols, but it is **not applied to the output** until it clears the
**monitor of §6.5 has been read by a promotion protocol that does not yet exist**
(§6.5; no bar lives in this document). The stream forecast is retained as a weighted constituent
whose weight is *measured* (currently ≈ 0) and as the declared fallback when the
book is Unavailable.

Formally, option **(b)** as the deployed map, with option **(c)** kept live as a
measured, promotable challenger

> **⚠ Revision 1 first wrote "option (a)" here. That was the most dangerous line
> in the document.** §1.1 defines **(a)** as the *stream-anchored forecast* and
> rejects it — it loses to the book by **+0.0201 Brier** through three σ
> generations. `Identity` is **(b)**. An implementer following the original line
> would have wired `p_hat` to the losing model. **And §1.1 still marks (b)
> "reject"** — see the §1.1 annotation: Revision 1 adopts an option its own table
> rejects, for a reason `FLOW_MODEL_STATE.md` refutes. §11's contract `notes`
> likewise still describe a recalibrating module. **Three statements of deployed
> behaviour in this document, none of them `Identity`.** The recommendation
> changed and the document was not swept. — so the rotation is a running diagnostic rather
than an assumption. **The direction of the burden is the whole change:** Revision
0 shipped the map and would have retreated to `Identity` on a failed gate;
Revision 1 ships `Identity` and requires a passed gate to leave it.

**And the size of it, measured on the largest sample available** (`be_belief.py`,
4,762 windows / 23,801 rows / 4 days):

| quantity | Revision 0 (§A, 1,645 windows, 2 days) | Revision 1 (probe, 4 days) |
|---|---:|---:|
| core `b̂` | 1.145 ± 0.042 | **1.037** — day-clustered CI **[0.897, 1.130]**, contains 1 |
| core `â` | +0.122 | **−0.006**, sign-alternating per day |
| deployable OOS Δlog-loss | −0.0006 (beats book) | **+0.00013** → `INDISTINGUISHABLE` |
| best challenger OOS | `affine_ab` −0.0020 | **`affine_ab` +0.00142** — CI excludes 0 on the **wrong side** |
| MDE at the 7-day gate | *not computed* | **0.00042 log-loss** |

**Every challenger is worse than the raw book out-of-sample.** The honest reading
is not "the rotation is absent" — four days cannot establish that, and the point
estimate is still above 1 — it is **"the rotation is not yet distinguishable from
zero at the unit this plan declares primary, and the effect it would buy is below
the MDE the same data supports."** Those are different claims and only the second
is supported.

**BE-Belief is a correctness module, not a P&L module** — that framing survives
Revision 0 intact. What does not survive is *adopting* the map on it. A
correctness module that ships an unsupported two-parameter transform is not more
correct than one that ships the identity; it is more confident.

---

## 1. What BE-Belief should produce, and why

### 1.1 The four candidates against the evidence

| candidate | verdict | evidence |
|---|---|---|
| (a) stream-anchored p̂ | **reject as the level** | loses at every horizon through three σ generations: +0.0291 → +0.0277 → +0.0201 Brier vs book. Uniform, not horizon-specific (`EXP_RESULTS_2026-08-20.md:26-39`, commit `1ec5e95`) |
| (b) book as-is | **reject** | best available forecast, but a belief equal to the price generates zero disagreement. All P&L then comes from spread/rebate; BE-Belief becomes a pass-through and the FLB — the one measured edge — is discarded |
| **(c) book recalibrated** | **ADOPT, with the magnitude downgraded** | the only measured, sign-stable, mechanism-backed disagreement we have — but §A measures it at `b̂ = 1.15 ± 0.04` and an OOS gain of 0.0004–0.0008 Brier, i.e. ≈ 2 probability points, not the 3–7 cents the bucket gaps suggest |
| (d) blend of (a) and (c) | **adopt as the *shape*, not as a claim** | the blend weight ŵ is E-X2's actual question (`PM_MM_PLAN.md:768-786`). Current evidence says ŵ→0. Keeping the slot costs nothing and makes the claim falsifiable rather than assumed |

### 1.2 The self-defeat check, stated honestly

> ### ⚠ REVISION 2 — THE SECOND SENTENCE IS WITHDRAWN. THE THIRD SURVIVES INTACT.
>
> Revision 0 opened: *"A belief that tracks the book cannot profit from
> disagreeing with it. **So the recalibration IS the edge or there is none.**"*
>
> **The first sentence is a tautology. The "So" is a false inference, and
> `FLOW_MODEL_STATE.md` — which wins on facts — refutes it directly.** §1e
> measures a simulated two-sided **`JOIN_BBO`** maker, whose belief *is* the book
> by construction:
>
> ```
> gross spread capture   btc +0.642 c/share (n=10,294)   eth +0.778 c (n=1,999)
>                        "real, positive and stable"
> ```
>
> **Profit generated with zero disagreement with the book.** The tautology is
> about *takers*, for whom edge requires disagreement. A **maker** is paid the
> spread for supplying immediacy, and that payment does not require believing
> anything the book does not.
>
> **What the honest version says, and why it makes the plan STRONGER:** the same
> §1e measures that maker's markout at **−0.532 c btc / −1.243 c eth**, intervals
> excluding zero, 8 of 8 cells negative. So a book-equal maker is paid the spread
> and loses more of it to adverse selection. **The P&L question is spread capture
> versus adverse selection — which is BE-FlowAndFills' term, not BE-Belief's.**
>
> §1.2 located the programme's edge question inside the *belief* module. It is
> not there. **That is exactly what §7's commodity self-description and
> consequence 3 below already say**, so withdrawing the "So" removes a
> contradiction rather than creating one: the plan asserted BE-Belief was the
> edge in §1.2 and denied it in §7, and §7 was right.
>
> **Consequences 1–3 below are independent of the withdrawn sentence and all
> three stand** — in particular consequence 2, which this plan calls its single
> most important line and which iteration 1 confirmed. **Under R-38(d) this moves
> no verdict.** It is escalated as `Q-BE-18` because the loop charter makes §1.2
> load-bearing for `EX1_PREDICTION_PROTOCOL.md`'s framing; that protocol is
> FROZEN and gets an annotation beside, never an edit.

*Revision 0 text, retained as provenance:* "A belief that tracks the book cannot
profit from disagreeing with it. So the recalibration IS the edge or there is
none." Three consequences we accept up front:

1. **The FLB gap is an upper bound on P&L, not P&L.** The measured capture ratio
   is brutal: unconditional gap +1.8 c/share at `t=290s, mid∈[0.95,1.00)` vs
   realised markout **+0.72 c** in the matching state bin (60% haircut); gap
   −9.4 c at `t=30s, mid∈[0.15,0.30)` vs realised markout **+0.25 c**
   (**97% haircut**) — `PM_DEEP_REVIEW.md:177-183`. *Selection destroys most of
   the available edge at every moneyness.*
2. **BE-Belief must produce the UNCONDITIONAL belief `E[Y | book state]`, never
   the fill-conditional `E[Y | book state, FILLED]`.** The fill conditioning is
   BE-FlowAndFills' adverse-selection term. If BE-Belief bakes it in, the two
   modules double-count the same haircut and the program silently under-quotes.
   This is an ownership ruling, and it is the single most important line in this
   plan.
3. **BE-Belief is not the moat.** See §7.

### 1.3 What this makes BE-Belief architecturally

Under (c), `p_hat` is an **algebraic function of an observable price**, not a
model output. That has a welcome side effect: it dissolves the
BE-Uncertainty ↔ BE-Belief cycle recorded at `PM_STRUCT_ITER2_B.md:785` (the
budget needs `G'` to convert X-space variance to p-space; the belief needs σ
from the budget). With `p̂ = G_b(m)` there is **no σ in the level**, so there is
nothing to circle. `BE-Link` (proposed, not adopted) stays unnecessary.

---

## 2. Which price is the input

**Recommendation: the executable pair `(best_bid, best_ask)` at knowledge time,
reconstructed from `price_change` merged with `book` snapshots; the belief's
anchor is a declared `PriceSummary` over that pair, defaulting to `Mid`, with
`Microprice` a gated challenger and `LastTrade` refused.**

### 2.1 The incumbent series is stale, and it is stale by more than the effect

Every experiment in the corpus builds the book from `book` snapshot events only
(`exp_blend_model.py:86`, `book_mid_series` at `:69-100`). But `book` is ~4.1k
events/window against ~149.7k `price_change` deltas
(`PM_MECHANISM_EXPERIMENTS.md:47`) — and **`price_change` carries `best_bid` and
`best_ask` per asset directly**, so top-of-book needs no delta replay at all.

Measured now (`raw/20260820`, 15 window files, 238 asset series, knowledge time):

| quantity | value |
|---|---|
| `book`-snapshot inter-arrival, per token | p10 55 ms · **p50 547 ms · p90 6,240 ms** |
| `\|mid(t) − mid(t−0.5s)\|` | mean 0.0157 · p50 0.0050 · p90 0.0400 |
| `\|mid(t) − mid(t−2s)\|` | mean 0.0306 · p50 0.0150 · p90 0.0800 |
| **`\|mid(t) − mid(t−6s)\|`** | **mean 0.0558 · p50 0.0350 · p90 0.1350** |
| `\|mid(t) − mid(t−15s)\|` | mean 0.0880 · p50 0.0600 · p90 0.2100 |

A mid taken at the p90 of snapshot staleness is wrong by ~5.6 c on average and
13.5 c at p90. **The FLB being fitted is 3–7 c.** The measurement error is
larger than the effect.

And it *moves the answer*. Full sample, both days, 1,645 windows (§A.1), gap =
realised − mid:

| mid bucket | dense (book+Δ) | snapshot-only | spread dense | spread snap |
|---|---|---|---|---|
| 0.1–0.2 | −0.034 | −0.041 | 0.039 | 0.047 |
| 0.2–0.3 | −0.007 | −0.012 | 0.041 | 0.050 |
| 0.3–0.4 | −0.009 | −0.018 | 0.044 | **0.057** |
| 0.4–0.5 | +0.027 | +0.015 | 0.039 | **0.053** |
| 0.5–0.6 | +0.033 | +0.037 | 0.042 | 0.055 |
| 0.6–0.7 | +0.051 | +0.062 | 0.045 | **0.060** |
| 0.9–1.0 | +0.020 | +0.025 | 0.021 | 0.025 |

Two systematic distortions, both in the direction that flatters the thesis:

1. **The stale series inflates the bias.** Fitting the map on each
   (§A.2): `b̂ = 1.145` dense vs **`1.182` snapshot-only** — the stale book
   overstates the excess over 1 by **26%**. Plausibly because `book` snapshot
   arrival is endogenous (p90 6.2 s, and the `1013 slow consumer` losses are
   load-correlated on BTC, `PM_MM_PLAN.md:493-497`), so stale reads concentrate
   in exactly the busy windows.
2. **The stale series inflates the spread** by 1.0–1.5 c ATM, because it pairs a
   bid and an ask that were never simultaneously quotable.

Mean staleness in the dense series is **0.04 s** where the book is active.

**Action item, first and non-negotiable:** rebuild the book series from
`price_change.best_bid/best_ask` ∪ `book`. Every book number in the corpus —
including the FLB table and the book's Brier — must be re-derived. The book's
information advantage over our model is currently *understated*.

### 2.2 Mid is a poor summary, and the corpus's spread number is wrong

Measured spread (`raw/20260820`, from `price_change` best_bid/ask, 238 series):
**p25 0.04 · p50 0.05 · p75 0.08**. By moneyness (book snapshots, 60 files):

| `min(mid, 1−mid)` | n | spread p25 / p50 / p75 |
|---|---|---|
| 0.0–0.1 | 1266 | 0.020 / **0.030** / 0.050 |
| 0.1–0.2 | 2075 | 0.040 / **0.060** / 0.090 |
| 0.2–0.3 | 2583 | 0.040 / **0.060** / 0.100 |
| 0.3–0.4 | 2642 | 0.050 / **0.080** / 0.120 |
| 0.4–0.5 | 3518 | 0.040 / **0.070** / 0.130 |

The corpus asserts "2–4 c" everywhere (`PM_MM_PLAN.md:35`,
`PM_MECHANISM_THEORY.md:53`, `PM_VS_MM_THEORY_DIFF.md:22`) — inherited from
Dubach 2026, never measured here; the one previously-measured figure was 1.1–1.8 c
(`PM_DEEP_REVIEW.md:150`). **Both are wrong for the ATM region where the drift-
adjusted FLB is largest.** ATM the book is 6–8 c wide.

> **REFUTED FOR BTC/ETH, 2026-08-21 (U8).** Measured on 2.29 M executable
> `price_change` quotes: the ATM median spread is **1 tick (0.0100)** with p90
> 0.020 on btc, and ATM spread in ticks runs **btc 1 · eth 1 · sol 3 · doge 3 ·
> xrp 5 · bnb 5 · hype 7**. "6–8 c" holds only for the thinnest coins — hype has
> 61.8 % of ATM quotes at >= 5 ticks. The corpus figure was a **pooling
> artefact**: btc supplies 2.28 M of 3.73 M quotes, so a pooled spread largely
> reports btc, and a pooled "modal 1 tick" and a per-coin "6–8 c" were both

> **REFUTED FOR BTC/ETH (U8):** ATM median spread is **1 tick** on btc/eth (2.29 M quotes); "6–8 c" holds only for the thin coins, and the fill question is now restricted to btc/eth. See the note above.

> being quoted from the same data. **Every conclusion below that rests on a wide
> ATM book applies to the thin coins only.**

That reframes the whole edge. A 3–7 c gap *at the mid* is 0 to −4 c *at the ask*.
So:

> **The map must be fitted on the mid but VALIDATED at the executable prices.**
> The trading question is never "realised vs mid"; it is "realised vs ask"
> (buy) and "bid vs realised" (sell).

Re-expressed on the full sample (§A.1), in cents earned by the *aggressor* —
positive = profitable before any haircut:

| mid bucket | buy Up at ask (`realised − ask`) | sell Up at bid (`bid − realised`) |
|---|---|---|
| 0.1–0.2 | −5.4 | **+1.5** |
| 0.2–0.3 | −2.7 | −1.4 |
| 0.3–0.4 | −3.2 | −1.3 |
| 0.4–0.5 | **+0.7** | −4.6 |
| 0.5–0.6 | **+1.2** | −5.4 |
| 0.6–0.7 | **+2.8** | −7.3 |
| 0.7–0.8 | **+3.3** | −7.4 |
| 0.8–0.9 | **+1.4** | −6.0 |
| 0.9–1.0 | **+0.9** | −3.1 |

**This is the most uncomfortable table in the plan, and it is not the FLB.**
A symmetric favourite–longshot harvest would show profit on *both* wings —
buy the favourite, sell the longshot. Instead: **buying Up at the ask is
profitable in every bucket at or above 0.4, and selling Up at the bid is
profitable in exactly one bucket.** That is a one-sided, directional
long-Up tilt, which is precisely what the sample's +1.9 pp up-drift produces
and what a rotation does not.

It does not invalidate `b̂ > 1` — the drift is separately identified and the
rotation survives once `a` is free (§4.4) — but it does mean **the executable,
cent-denominated version of this edge is currently indistinguishable from
"the market went up for 20 hours".** Any P&L narrative built on this table is
a rally narrative until a down-drift period is observed.

And all of it sits *before* the 60–97% selection haircut of §1.2. That haircut,
not the gap, is the number that decides deployment, and it is not BE-Belief's to
produce.

### 2.3 Microprice: a gated challenger, not the default

No microprice, imbalance or OFI measurement exists anywhere in the program
(`PM_MM_PLAN.md:697`, `PM_VS_MM_THEORY_DIFF.md:66`) — zero data. Three measured
facts argue it will be weak *here*, unlike in a continuous market:

1. Top-of-book size is quantised at the reward-program minimums — measured p10/p50/p90
   of the touch = **4.98 / 29.98 / 50.0 shares** against `orderMinSize=5` and
   `rewardsMinSize=50` (`PM_MM_PLAN.md:677-679`). Touch size is largely
   reward-farming inventory, not informed demand.
2. Depth deliberately sits *outside* the touch: BTC median depth within 1 c of
   mid is **138 shares/side (~$69)** vs **1,290 within 4.5 c**
   (`PM_SKETCH_REVIEW_ITER1_M.md:150-152`). An imbalance computed at the touch is
   computed on the least informative 10% of the book.
3. The tick regime switches: `tick_size_change` 0.01 → 0.001 fires **328 times
   across 130 windows**, never reverting (`PM_MECHANISM_EXPERIMENTS.md:44,348-351`).
   Size-at-the-touch is not comparable across regimes.

So `PriceSummary` is a **registry-open variant** and `Microprice(alpha)` must
beat `Mid` out-of-sample on the same walk-forward protocol before it can be
declared. It is not the default.

### 2.4 last_trade: refused as the anchor

Two independent reasons, both already on the record:

- The trade-price conditioning was **explicitly retracted**: *"trade price is an
  outcome of the fill, not the state a maker chooses to rest in, and a sweep
  prints far from the mid it started at"* (`PM_DEEP_REVIEW.md:120-126`). The
  retracted number (+3.6 c/share at p∈[0.15,0.35)) still headlines
  `PM_MM_PLAN.md:960` — do not carry it into BE-Belief.
- Feed-ordering defect: *"the `price_change` carrying best_bid/ask for a match
  can be emitted **before** the `last_trade_price` for the same match"*
  (`PM_DEEP_REVIEW.md:149-157`). Any last-trade constituent must therefore carry
  a **declared frozen lag** (e.g. mid as of `t_trade − 250 ms`), which is a
  `LastTrade(lag: Duration)` variant, never a bare read.

### 2.5 There is exactly one price — no cross-check exists

Measured now: the Up and Down books are **exact complements**.
`price_change` entries for the two assets of a market mirror algebraically
(observed: asset A bid 0.43 / ask 0.63, asset B bid 0.37 / ask 0.57 ⇒
`bid_A = 1 − ask_B` exactly), and paired `book` snapshots taken < 2 s apart give
`mid_up + mid_down`: **p10 0.9900 · p50 1.0000 · p90 1.0100 · mean 1.0001**
(n = 3,003).

Consequence to state loudly: **the obvious robustness test — "does the Down
token show the same FLB?" — is algebraically vacuous.** There is no redundant
observation of the price, no cross-book arb residual, and no independent
validation of the anchor. Any "confirmation" from the Down side is a tautology.
This kills a test a reviewer will otherwise demand, and it removes a data source
someone will otherwise assume exists.

---

## 3. Functional form

**Recommendation: one-parameter anchored logit map on the direction-folded
coordinate.**

```
  deployed:   logit p̂ = b · logit m            ⇔   p̂ = m^b / (m^b + (1−m)^b)
  estimated:  logit p̂ = a + b · logit m        (a reported, NOT deployed — §4)
```

### 3.1 Why logit-affine and not isotonic

| form | dof | verdict |
|---|---|---|
| **`b` only, anchored** | **1** | **ADOPT.** Rotation about p=0.5; drift-neutral by construction |
| `(a, b)` free | 2 | estimate and report; `a` is the drift channel, deliberately not deployed |
| `b(r)`, 2 knots | 3 | challenger, gated on ≥7 days (§5) |
| isotonic | ~#level-sets | **reject as a deployable at this n**; keep as an overfitting *reference* |
| parametric non-logit (probit/power-of-odds) | 1–2 | equivalent at this precision; logit chosen because it makes drift and FLB orthogonal |

Isotonic is the wrong tool here, and this is now measured rather than argued
(§A.4): a 10-bin walk-forward isotonic fit is **worse out of sample than the raw
book** — Δlog-loss **+0.0012**, ΔBrier **+0.0009** — while the 1-parameter map is
better (−0.0006 / −0.0004). Isotonic does not merely fail to help; it destroys
value at this n.

The reason is visible in the data. The *drift-adjusted* gap profile the programme
reports is −0.066, −0.043, −0.076, −0.034, −0.015, +0.034, +0.012, +0.021 —
**not monotone**. An isotonic fit on such a profile produces a staircase whose
step boundaries are set by *which way the noise happened to fall*, and it will
pool adjacent buckets that differ by 8 cents.
`PM_DEEP_REVIEW.md:394-403` correctly names isotonic as the *competitor's*
one-hour tool — that is an argument about how cheap the edge is, not an argument
that isotonic is the right estimator at n = 1,645 windows.

Isotonic also **cannot separate drift from FLB**: it absorbs both into one
unconstrained curve, which is precisely the failure mode the brief warns about.

### 3.2 Why the logit parameterisation is the drift separation

In logit space the two confounds are geometrically distinct:

- **FLB = rotation** about `p = 0.5` → the slope `b`. `b > 1` ⇔ underconfident
  book ⇔ longshots overpriced *and* favourites underpriced, symmetrically.
- **Drift = translation** → the intercept `a`. A +4.5 pp up-drift shifts every
  bucket the same way in logit space and is largest in probability space ATM —
  the hump the brief describes.

They are orthogonal by construction, so the separation is *structural*, not an
ex-post subtraction. This is strictly better than the incumbent method
(`exp_blend_v2.py:100-110`), which subtracts `mean(φ(d)·μ/φ(0))` per bucket. That
adjustment assumes drift enters through *our Gaussian model's* `d`. But we are
recalibrating the **book**, and the book's drift response is φ(d) only if the
book is exactly Gaussian in d — which is the very hypothesis the FLB rejects.
A free intercept absorbs drift with no such assumption.

### 3.3 Effect size: predicted, then measured

*Predicted* from the drift-adjusted profile: `m = 0.15 → p ≈ 0.10` gives
`b = logit(0.10)/logit(0.15) = 1.27`; `m = 0.65 → p ≈ 0.70` gives `b = 1.37`.

*Measured* (§A.2, dense book, core domain, both days, window-clustered):
**`b̂ = 1.145 ± 0.042`.** So the bucket-gap reading **overstates the fitted,
drift-controlled rotation by roughly 2×**. In cents at the moneyness where it is
largest: `m = 0.65 → p̂ = 0.671`, i.e. **+2.1 pp against an ATM half-spread of
≈ 2.2 c**. The map is worth about one half-spread at its best point, and less
everywhere else — before the 60–97% selection haircut of §1.2.

> ### ⚠ Revision 1 — THIS SUBSECTION'S HEADLINE IS WITHDRAWN
>
> Revision 0 continued: *"The null `b = 1` is rejected in-sample at ≈ 3.5
> window-clustered σ, and `b` is stable across the two days (day 1: 1.184 ±
> 0.078; day 2: 1.113 ± 0.067). That is the strongest thing that can honestly be
> said on 20 hours of data."*
>
> **It was not the strongest thing that could honestly be said; it was stronger
> than could honestly be said, and by this plan's own rule.**
>
> **§6.1 rule 4 of this document:** *"the correct unit for power is windows, and
> above that, **days**."* The rejection above is quoted in **window**-clustered
> σ. Day-clustered on the four days `be_belief.py` covers — `0.9894, 0.9917,
> 1.1203, 0.9530`:
>
> ```
> mean 1.0136   sd 0.0733   se 0.0367   n_days 4
> t-interval, 3 df (t=3.182):  [0.897, 1.130]   CONTAINS 1
> normal z=1.96:               [0.942, 1.085]   EXCLUDES 1
> ```
>
> **The central claim does not survive the unit this plan declares primary.** Not
> a reviewer's alternative standard — the plan's own, applied to the plan.
>
> **⚠ REVISION 1's OWN ARITHMETIC HERE WAS FALSE. CORRECTED.**
>
> Revision 1 wrote that the z-interval *"**excludes 1** — and Revision 1's central
> argument reverses."* **It contains 1:** `0.94176 < 1.00000 < 1.08544`. BE's
> verification script *printed the literal string* `"does NOT contain 1"` instead
> of evaluating `lo < 1 < hi` — **a label in the position of output**, written in
> the same hour BE fixed another instrument for asserting instead of detecting.
>
> **The error ran AGAINST interest, and that is the part worth keeping.** Both
> conventions contain 1, so the conclusion is **robust** to the convention — a
> *stronger* statement than the one Revision 1 made against itself. Revision 1
> manufactured a false fragility, then wrote a long confession to Revision 0's
> convention-shopping sin. **The confession was the error.** Over-correction is
> not a safe direction to fail in; it is another way to put a false statement in
> a plan.
>
> **AND THE WHOLE DAY-CLUSTERED ARGUMENT IS THE WRONG INSTRUMENT.** A variance
> decomposition on the same four days:
>
> ```
> mean WITHIN-day sampling variance  0.005302     (from §A.2's clustered se, scaled by n)
> observed BETWEEN-day variance      0.005374
> tau^2 = +0.000072   ->   tau = 0.0085   ~  ZERO
> ```
>
> **There is no measurable day-level variance component.** A perfectly stable `b`
> with one true value produces exactly this much apparent alternation. So §3.3's
> *"`b̂` per day alternating around 1 … the estimator is not reproducing across
> samples, which is the finding"* is **refuted** — 98.6% of the dispersion is
> within-day sampling noise. The correct statistic is the window-clustered
> interval on all 18,755 core rows:
>
> ```
> b_hat = 1.0367 +/- 1.96(0.0245)  =  [0.989, 1.085]     CONTAINS 1, at HALF the width
> ```
>
> **`Identity` survives on a better argument than the one Revision 1 gave it.**
> The four-day interval is wide because it burns 3 df estimating a variance that
> is zero, not because days disagree.
>
> **AND "the one day shared with Revision 0's sample" SHARES ZERO WINDOWS.**
> Revision 0's §A spans `2026-08-19 14:25 → 2026-08-20 10:40`; the `clob_v3_1` era
> opens `2026-08-20 14:50:21`. **Disjoint, 4h10m apart, across a collector-era
> boundary** that `FLOW_MODEL_STATE.md` §5 forbids pooling across. The
> 0.989-vs-1.113 comparison is between two populations sharing a calendar label
> and nothing else — the same defect that voided E-X1 under R-56, committed again
> in the section written to correct Revision 0's inference.
>
> **What honestly remains of §3.3:** `b̂ ≈ 1.04` with a window-clustered interval
> containing 1; no evidence of day-level instability; and no comparison to
> Revision 0 that survives the era boundary. That supports `Identity` and supports
> nothing else.

*The Revision 0 text above is retained as provenance. Its measurements are not
disputed; its inference from them is.*

---

## 4. Drift separation — the design

Four layers, in increasing strength.

### 4.1 Structural (always on)
Logit parameterisation, §3.2. `a` = drift, `b` = FLB.

### 4.2 Direction-folding — as a DIAGNOSTIC, not as the estimator

For each `(window, decision time)` define the **longshot coordinate**
`q = min(m, 1−m)` and `y_q = 1` iff the *longshot side* won. Then:

- a pure **FLB** (rotation) predicts `E[y_q | q] < q` **uniformly, regardless of
  whether the longshot happened to be Up or Down**;
- a pure **drift** predicts `E[y_q|q] > q` when the longshot is Up and
  `E[y_q|q] < q` when the longshot is Down — *opposite signs*.

**Correction to the obvious design.** Folding is **not** "drift-free by
construction", and I initially wrote that it was. Folding does not remove drift;
it *relocates* drift into a **disagreement between the two arms**, and cancels it
in the pooled fold only to first order and only when the arms are balanced.
Measured (§A.3): the folded arms give `b = 1.027` (longshot = Up) versus
`b = 1.268` (longshot = Down) — the exact signature of an up-drift, and a 2.4σ
apparent rejection of the rotation hypothesis.

**That apparent rejection is itself an artifact**, and diagnosing it is the most
important thing this analysis did — see §4.4.

So: fold as the *diagnostic*; estimate with the free intercept.

### 4.3 Deployment policy: pin `a = 0`

The deployed map carries no intercept. Justification: `a` is the net drift of a
20-hour rally in which **every coin is up** — measured up-rates btc 0.5375,
eth 0.5667, sol 0.5417, xrp 0.5500, doge 0.5154, bnb 0.5374, hype 0.5551;
pooled 0.5436 (n = 1,641). Deploying `a` is a bet that the rally continues, i.e.
a directional alpha claim — exactly the claim the ŵ→0 result says we cannot make.

Declared as a field-level `NullPin`:
```
  field: BeliefProcess.recalibration.drift_intercept
  assumption: a = 0 (drift-neutral anchor at p = 0.5)
  bias_direction: PESSIMISTIC     # under-states P(Up) in an up-drifting sample
  declared_by: BE-Belief
```
`a` is still *estimated and reported* every fit, because a large stable `a` is
either drift leaking in or a genuine venue-wide Up-token bid, and only calendar
time separates those.

### 4.4 Asymmetry test — RUN, and it is the load-bearing result

A pure rotation implies `b_low = b_high` when fitted separately on `m < 0.5` and
`m ≥ 0.5`. Run two ways, and the two ways disagree — which is the whole point:

| specification | `a` | `b_low` | `b_high` | `b_high − b_low` |
|---|---|---|---|---|
| **pinned `a = 0`** (deployment form) | — | 1.027 ± 0.063 | 1.268 ± 0.069 | +0.241, ≈ 2.4σ — **apparent rejection** |
| **free common `a`** (estimation form) | +0.087 ± 0.078 | 1.096 ± 0.073 | 1.195 ± 0.080 | +0.099, 95% CI **[−0.169, +0.349]** — **contains 0** |

**Reading.** When the intercept is pinned at zero, the sample's up-drift has
nowhere to go and it leaks into the arm slopes — inflating the favourite arm and
deflating the longshot arm, producing a spurious asymmetry that looks exactly
like "the FLB is really drift". Give drift its own parameter and the asymmetry
collapses to within noise. **The rotation hypothesis survives; the drift is
separately identified at `a ≈ +0.09…+0.13`; and the two are cleanly separable in
this parameterisation.**

This validates the §3.2 design choice empirically, and it yields the single
sharpest rule in this plan:

> **Estimate with a free intercept. Deploy with the intercept pinned to zero.
> These are different operations and conflating them corrupts the slope.**

An estimator that pins `a = 0` (including *any* anchored or folded one-parameter
fit, and including isotonic-through-the-origin variants) will systematically
mis-attribute drift to the FLB. The contract encodes this: `RecalibrationForm`
carries both `LogitAffine` (fit) and `LogitAnchored` (deploy), and
`Recalibration.drift_intercept` is a `NullPin`, not an absence.

Residual caveat: `a` is only 1.1σ from zero in the 3-parameter fit, so "drift is
present" is itself weakly established. Both readings — real drift, or no drift —
leave the recommendation unchanged, because we refuse to deploy `a` either way.

### 4.5 A drift control that is knowledge-time legal

Conditioning on the window's own realised drift is look-ahead. The legal
version: stratify by the **prior hour's** realised return of the same coin,
read at `t0` from the settlement stream. If `b` is stable across those strata and
`a` tracks them, the separation works. If `b` moves with the prior-hour drift,
the FLB estimate is drift-contaminated and must be fitted conditionally.

---

## 5. Conditioning on time-in-window `r`

**Recommendation: `r` enters as a REGIME SPLIT and an exclusion, not as a free
slope — because the mechanism changes with `r`, it does not merely scale.**

### 5.1 The tick floor manufactures the favourite side

The book's Brier collapses to 0.0174 at `r = 30 s` — it knows the answer
(`EXP_RESULTS_2026-08-20.md:33`). At that point the *only* thing left to
mis-price is the last cent, and the grid decides it. On a 0.01 grid the best
quotable ask is 0.99, so a contract whose true probability is 0.998 is
mechanically under-priced by 0.8 c. Measured: bucket 0.9–1.0 has mid 0.967,
**ask 0.975**, realised 0.988 — an "edge" of exactly the size of the grid.

This matters enormously for the fit, because **the extreme buckets dominate the
sample**: 1,609 + 1,916 = **3,525 of 8,862 samples (40%)** sit in 0.0–0.1 or
0.9–1.0 (`EXP_RESULTS_2026-08-20.md:45,54`), and in logit space they carry by far
the largest `(logit m)²`, hence most of the Fisher information for `b`. **A
pooled fit is driven by the 40% of the sample where the bias is mechanical, not
behavioural, and where a 2 c edge cannot be quoted against a 1 c tick.**

**Measured — and this partly exonerates the extremes.** I predicted the extreme
buckets would dominate the fit through their large `(logit m)²`. They do not:
core `n = 6,390`, se(`b`) **0.042**; extreme `n = 1,818`, se(`b`) **0.065** — the
core carries ~2.4× the information, because the extreme cells also have tiny
`p(1−p)`, which cancels the leverage. And the two domains agree on the slope:
`b_core = 1.145` vs `b_extreme = 1.168`.

What the extremes *do* have is a far larger intercept: `a_extreme = +0.428` vs
`a_core = +0.122`. So the extreme-domain distortion is a **level** effect —
exactly the shape a grid floor produces — not a slope effect, and it is
therefore already handled by refusing to deploy `a`.

Rulings (softened accordingly):
1. Fit the behavioural `b` on the **core domain** `|logit m| ≤ 3`
   (`m ∈ [0.047, 0.953]`), declared as `Recalibration.domain`. This is now a
   *hygiene* choice, not a rescue: it changes `b̂` by 0.02.
2. Fit the extreme domain under its **own** intercept and label that intercept
   `tick_floor`, not `drift`. Refit within the 0.001 tick regime as the control:
   if `a_extreme` collapses toward `a_core` there, it is the grid.
3. Do **not** deploy any extreme-domain intercept until (2) returns.

### 5.2 Does `b` depend on `r`? — measured: **no, not detectably**

The apparent `r`-dependence in the corpus is confounded: as `r` falls the mid
distribution migrates to the extremes, so a fixed-bucket comparison across `r`
compares different populations. `b` is scale-free and *is* comparable across `r`,
which is the right diagnostic — and another reason to prefer logit-affine over
isotonic. Run (core domain, dense, window-clustered):

| `r` (s remaining) | n | `a` | `b` | se(`b`) |
|---|---|---|---|---|
| 60 | 539 | +0.168 | 1.203 | 0.095 |
| 120 | 1,112 | +0.041 | 1.115 | 0.062 |
| 180 | 1,471 | +0.119 | 1.083 | 0.059 |
| 240 | 1,629 | +0.140 | 1.152 | 0.070 |
| 270 | 1,639 | +0.138 | 1.256 | 0.081 |

**No monotone trend, no pair separated by more than ~1.5 se, every value within
one se of the pooled 1.145.**

> **⚠ Revision 3 — THIS SENTENCE WAS FALSE ON THE TABLE DIRECTLY ABOVE IT, BEFORE
> ANY NEW DATA EXISTED.** Two of its three assertions fail against §5.2's own
> Revision-0 numbers:
>
> ```
> max pairwise separation:  r=270 (1.256, se .081) vs r=180 (1.083, se .059)
>                           diff 0.173, se_diff 0.100  ->  1.73 se     (claim: "~1.5")
> "within one se of 1.145":  r=270 is 1.37 se away    VIOLATES
>                            r=180 is 1.05 se away    VIOLATES   (2 of 5)
> ```
>
> And against the **pooled receipt** — the larger of the two — the max pair is
> `r=240` vs `r=120`: **2.32 σ**, and that is a *floor*, because the per-`r` fits
> share window outcomes so `Cov > 0` shrinks the true `se_diff`. On the verdict
> coins the max pair is 1.40 σ: **underpowered, which is not the same as absent.**
>
> **So the header's "measured: no, not detectably" is a null stated as an
> established fact, and it was never supported — on any data, at any time.**
> Revision 2 annotated this only in its banner and only against the *smaller*
> receipt, which left the section itself asserting the null. §10.8 mandates
> *"always report `r`-stratified"*; §6.5 now records the `r`-split accordingly. So the answer to the brief's question is: **the map
does not need conditioning on `r`, and adding it now would spend 2–4 dof on
noise.** The book's Brier collapse to 0.017 at `r = 30 s` is a statement about
*how much is left to forecast*, not about *how biased the forecast is* — those
are different quantities and `b` is the one that matters here.

Note this also removes the mechanism-conflict worry: if the behavioural FLB fell
with `r` while the tick-floor bias rose, we would see a U or an inverted-U in the
core-domain `b`. We see neither.

**Gate:** `b(r)` with 2 knots is deployed only if it beats pooled `b` on
out-of-sample day-clustered log-loss at ≥ 7 days. On current evidence, expect it
to fail that gate.

---

## 6. Fitting and validation protocol

### 6.1 Hard rules

1. **Walk-forward only.** Fit on days strictly `< d`; score day `d`. Never refit
   within a test day. An in-sample isotonic fit on 1.4 days will look
   spectacular and mean nothing.
2. **Knowledge time only.** All reads via `recv_ns`. Measured total observation
   lag on the settlement stream is **p50 1,700 ms / p95 2,330 ms**, of which 85%
   is PM-side publication delay (`PM_DEEP_REVIEW.md:35-45`). For `r < 1.7 s` we
   have observed *none* of the final segment — the steady state, not a gap.
3. **Admissibility.** Use the canonical four-condition rule
   (`PM_MECHANISM_EXPERIMENTS.md:113-118`), not the weak TWAP-only rule actually
   implemented at `exp_blend_v2.py:45-47` / `exp_blend_v3.py:66-67`. Report the
   excluded fraction and its bias direction with every table; exclusions are
   load-correlated (`1013 slow consumer` disconnects on BTC, which carries 85%
   of notional — `PM_MM_PLAN.md:493-497`), so all numbers are calm-market
   numbers until proven otherwise.
4. **One outcome per window.** The 5–6 grid points per window share a single
   Bernoulli draw. **Every n in the corpus is inflated 5–6×.** Cluster the
   likelihood and every CI on `window`, then on `day`. The correct unit for
   power is *windows*, and above that, *days*.
5. **Effective breadth ≈ 1–2, not 7** (`PM_MM_PLAN.md:747-749`). Seven crypto
   majors on one beta.

### 6.2 The challenger ladder (all scored on the same walk-forward split)

| # | model | status | result / expectation |
|---|---|---|---|
| 0 | raw book (`b=1, a=0`) | run | the null to beat |
| 1 | **anchored `b`, core domain — the deployable** | run | −0.0006 log-loss / −0.0004 Brier OOS |
| 2 | `(a, b)` free — the **estimator** | run | −0.0020 / −0.0008 OOS; `a` diagnoses drift; **not deployed** |
| 3 | `b_low`, `b_high` with common `a` | run | §4.4 — rotation survives, diff CI contains 0 |

> **⚠ Revision 3 — `TESTED` IS WITHDRAWN. THE TEST HAD NO POWER.** The
> free-intercept CI on `b_high − b_low` is `[−0.169, +0.349]`:
>
> ```
> half-width 0.259  ->  se 0.132  ->  MDE at 80% power = 0.370
> MDE (0.370) EXCEEDS the +0.241 this row claims to have dissolved
> and +0.241 lies INSIDE the interval
> ```
>
> **The interval widened to admit both zero and the original estimate.** Nothing
> collapsed and nothing survived — the data cannot separate the two hypotheses.
> Striking a risk through as `TESTED` on that basis records **a failure to reject
> as a result**, which is the defect this plan's own iteration-1 verdict convicted
> Revision 0 of. Status reverts to **OPEN, underpowered**.
| 4 | `b(r)`, 2 knots | screened | §5.2 — flat in `r`; expect it to fail its gate at 7 days |
| 5 | isotonic | run | **worse than the raw book** (+0.0012 / +0.0009) |
| 6 | per-coin `b` (shrunk) | screened | §6.3 — heterogeneity marginal; gated to ≥30 days |
| 7 | blend: `logit p̂ = (1−w)·b·logit m + w·logit p̂_stream` | not run | ŵ expected ≈ 0 |
| 8 | `Microprice(α)` anchor vs `Mid` | not run | §2.3 gate |

Metrics: paired Δlog-loss and ΔBrier vs model 0, day-clustered; plus the
executable re-expression of §2.2 (`realised − bid`, `ask − realised`) per cell.
Log-loss is primary — it is the fit's own objective and it penalises the
overconfidence a too-large `b̂` would create.

### 6.3 What pools and what does not

| parameter | scope | why |
|---|---|---|
| `b` (FLB slope) | **pooled across all 7 coins** | the FLB is a property of the venue's participant mix (retail lottery-ticket demand + reward-farming makers), not of the coin. There is no coin-specific mechanism. With effective breadth 1–2, seven per-coin fits are seven noisy readings of one number — and that is what the data shows (below) |
| `a` (drift) | per-coin, **diagnostic only** | per-coin up-rates span 0.5154–0.5667 — that spread is drift, and it is exactly what we refuse to deploy. Measured `a` tracks it: ETH has the highest up-rate (0.5667) *and* the highest `a` (+0.318) |
| domain / tick regime | per-instrument | `SP-Venue.tick_grid` is state-dependent; the 0.01↔0.001 switch is per market |
| spread / anchor | per-instrument, **observed not fitted** | different coins have different spread distributions; handled by reading the live pair, not by splitting `b` |

Measured per-coin (core domain, dense, window-clustered), against the pooled
`b̂ = 1.145`:

| coin | n | `a` | `b` | se(`b`) |
|---|---|---|---|---|
| bnb | 886 | +0.096 | 1.290 | 0.127 |
| btc | 947 | +0.046 | 1.181 | 0.131 |
| doge | 862 | −0.026 | 1.099 | 0.107 |
| eth | 933 | **+0.318** | 1.189 | 0.140 |
| hype | 899 | +0.069 | 1.408 | 0.150 |
| **sol** | 951 | +0.142 | **0.908** | 0.082 |
| xrp | 912 | +0.158 | 1.065 | 0.102 |

Heterogeneity χ² ≈ 13.9 on 6 df (p ≈ 0.03) — **marginal, and driven entirely by
SOL** (2.9σ below pooled; note SOL is also the coin measured at −231 bps to the
maker, `PM_DEEP_REVIEW.md:601`). And these se's are themselves understated: they
cluster on window but windows within a coin-day ride one price path. So
per-coin heterogeneity is **not established**, pooling is the correct default,
and per-coin `b` enters as a *shrinkage challenger* (model 6) at ≥30 days, never
as the default. If SOL survives as an outlier at 30 days, that is a
liquidity/spread story to be tested by adding spread as a covariate (§10.12) —
not a licence to fit seven slopes.

### 6.4 Refit cadence and artifact discipline

Daily refit at the UTC day boundary on all admissible history; the map is an
immutable artifact with `artifact_id`, `fit_data_through`, `fit_n_windows`,
`fit_n_days`, and the parameter covariance. `ParamValue` already carries
`fit_data_through` and `artifact_id` (`contracts.yaml:251-259`) — **the fitted
`a`/`b` live in `SP-Params` keyed by `(ParamId, ScopeKey)`, not restated on
`Recalibration`** (R-SSOT; same precedent as `loss_limit` and `rewards_band`).
`Recalibration` holds *references*.

Fail-loud conditions → `Unavailable`:
- ~~`fit_n_days < 2` or `fit_n_windows < 300` → `Unavailable(WARMUP)`~~
  **DELETED IN REVISION 2 — under `Identity` THERE IS NO FIT, and this condition
  refused the best available forecast in favour of the worst.** Concrete: a fresh
  deploy or artifact-store reset leaves `fit_n_days = 0`, so the module returns
  `Unavailable(WARMUP)` for two days even though `p̂ = mid` is computable from the
  live book and is, per §1.1 row (b), *"the best available forecast"*.
  `UnavailableAction = FallBack` then routes to the stream forecast, which §1.1
  row (a) measures as losing to the book by **+0.0201 Brier**. **The bound belongs
  on `RecalibrationForm.LogitAnchored`, not on `BeliefProcess`** — a warm-up
  guards a *fit*, and Revision 1 left it guarding a module that no longer performs
  one.
- **NEW IN REVISION 2 — book INADMISSIBLE → `Unavailable(BOOK_INADMISSIBLE)`.**
  The one genuinely missing mechanism, and the only place `Identity` can emit
  something indefensible. Measured over **1,940,224** admissible quotes on
  2026-08-23:

  ```
  spread > 20 c   80,118  (4.13%)   100% of them inside the core domain
  spread > 50 c   14,720  (0.76%)   100% inside the core domain
  best_bid = 0.00 71,628  (3.69%)   -- no bid side at all
  crossed/locked       0  (0.00%)   -- SEE THE CORRECTION BELOW: THIS ZERO IS TAUTOLOGICAL
  ```

  Concrete failure this closes: `bnb-updown-5m-1787446500` quotes `bid 0.15 /
  ask 0.94`. The boundary rule `0.0 ≤ bid < ask ≤ 1.0` **admits** it; age is 0 s
  so `max_book_staleness` **cannot** fire; `m = 0.545` is inside
  `Recalibration.domain` so `on_out_of_domain` **cannot** fire. Under `Identity`
  the module emits **`p̂ = 0.545, staleness = 0`, no flag**, from a 79-cent-wide
  book with essentially nothing in it — and DE sizes against it. `bid 0.00 /
  ask 0.14` likewise emits `p̂ = 0.07` from a book with no bid side.

  **Note the perverse direction, because it is why staleness cannot substitute:
  the worst books are the FRESHEST.** A wide book is wide precisely because it is
  being churned, so every staleness bound passes exactly when admissibility
  fails. The two conditions are independent and the plan had only one.

> **⚠ Revision 3 — TWO CORRECTIONS, AND THE MECHANISM AS SPECIFIED WAS INERT.**
>
> **(a) The crossed/locked zero is TAUTOLOGICAL.** The census ran over quotes
> admitted by `0.0 ≤ bid < ask ≤ 1.0` — a rule that **excludes crossed and locked
> by construction** (the strict `<` is itself the guard). Counting them among
> admitted quotes returns 0 always. `ev_gates.py:244` refuses exactly this shape:
> *"idealisation is a COMPARISON; one arm cannot show it."* **The instruction
> "do not code for it" is struck.** It is also wrong for the object §11 mandates:
> `TopOfBook` is `price_change` ∪ `book`, and a fresh `price_change` bid meeting a
> `book` ask up to 6 s stale **manufactures crossed pairs** — where the probe,
> reading `price_change` only, cannot see one because both sides arrive in the
> same message. Re-run the census on the **unfiltered, merged** stream.
>
> **(b) A WIDTH condition does not bind on the coins that carry verdicts.**
> Revision 2 calibrated this on the `>20 c` tail and demonstrated it with
> **`bnb`** — one of the five `descriptive_only` coins the same banner cites
> V5:333 to disqualify. **ATM spread is 1 tick on btc/eth** (§2.2 U8), so any
> threshold drawn from that tail passes ~100% of the verdict coins. Concrete state
> still admitted: `bid 0.54 / ask 0.55, size 5 × 5` — one tick wide, both sides
> present, age 0, **$2.75 a side**, which is the venue's own p10 touch (§2.3).
>
> **The condition is `width ∧ both_sides ∧ DEPTH`**, and its census must be run
> **per coin with the verdict coins reported separately** — a pooled 4.13% is the
> pooling artefact this plan condemns twice elsewhere. `TopOfBook` already carries
> `bid_size`/`ask_size`, so depth is buildable today.
>
> **(c) The threshold is a POLICY value, not a Class-C estimand.** Revision 2 said
> *"Class-C — measured, adopted, never chosen"*. No estimator returns "refuse
> above X cents": Class C is publish-then-adopt for quantities an estimator
> produces. Calling a policy value Class-C does not defer the choice, it conceals
> it — **and the protection was already spent**, because this document publishes
> the distribution at two candidate cut points. Set it from a stated *principle*
> (refuse when the half-spread exceeds the map's maximum measured effect, ≈2 pp)
> and record that the value is chosen.

  Admissibility is a **width, both-sides and DEPTH** condition on `TopOfBook`,
  evaluated before any map. Its threshold is **Class-C — measured, adopted, never chosen**:
  it is not set in this document, because setting it here against a distribution
  already seen is the Class-D move R-6 forbids and §6.5 was just deleted for.
- book absent or `staleness > threshold` → `Unavailable(STALE_BOOK)`, cause
  propagated
- `m` outside `Recalibration.domain` → the identity map, flagged, never
  extrapolated
- `source_events = SNAPSHOT_ONLY` → belief is emitted but **marked**, because
  §2.1 shows it is a materially different object

---

### 6.5 What would change our mind — a MONITOR, not a gate (Revision 2)

> **Revision 1 put a five-condition automatic promotion gate here. Revision 2
> DELETES it.** Three review lenses returned **eleven** findings against it: two
> of its three "directional" conditions were **sign-blind** (it would have
> promoted a book-*contracting* map and reported it as confirming the opposite
> mechanism); it demanded sign agreement across seven coins when
> `FLOW_MODEL_PROTOCOL_V5.yaml:333` bars five of them from carrying a verdict; its
> `α` was ~1.5e-06 with ~0 power against the plan's own point estimate; its P4
> was undefined at the `n_days` it mandated; P5 contradicted P1; and its
> anti-ratchet could oscillate the belief 1.4 c/share at a UTC boundary with no
> knowledge-time footprint.
>
> **Those are eleven symptoms of one mistake: automating a decision that happens
> at most once.** A gate must be sound against every input it could ever see,
> including adversarial ones — that is why it accumulated conditions and why each
> condition brought its own defect. **A monitor only has to report.**

**The rotation estimator runs and is recorded. It is never applied to the output,
and no rule promotes it automatically.**

Each refit records `b̂`, its interval, the per-coin split restricted to the
**verdict coins**, and the out-of-sample Δlog-loss against `Identity`. That is
the whole mechanism.

> **⚠ Revision 3 — AS WRITTEN THIS DEFERRAL COULD NEVER BE EXECUTED.** §6.5
> required a protocol *"frozen before the data that would justify it is looked
> at"* **and** required the monitor to publish that same data at every refit.
> Both cannot hold: at any time `T`, all data through `T` has been looked at, by
> the monitor. There was no reachable state in which the protocol could be written
> cleanly, because Revision 2 deleted the only moment — before the monitor's first
> reading — when it could have been. **Revision 2 deleted the gate and
> pre-registration together; they are not the same object.** The eleven iteration-2
> findings were against the gate's *conditions*, never against pre-registering.
>
> **THE TRIGGER IS A CALENDAR COMMITMENT, WHICH IS R-6-SAFE PRECISELY BECAUSE IT
> IS NOT A THRESHOLD ON THE EFFECT:**
>
> > **At `n_days ≥ 30` the programme MUST write and freeze a promotion protocol,
> > evaluated only on data AFTER its freeze date.**
>
> A date is not a Class-D value — it cannot be tuned toward a result, and nothing
> about the effect is chosen by picking it. The monitor's readings up to the
> freeze date inform *whether it is worth writing*; the protocol scores only what
> comes after. **A monitor with no actuator is cost with no path to value**, which
> is the R-61 marginal-value test applied to a mechanism rather than a finding.

**Leaving `Identity` requires a NEW FROZEN PROTOCOL, frozen on the calendar
trigger above and evaluated only on post-freeze data.** Not a threshold in this
document.
The reasons are the programme's own:

- **A bar written now, against a result already visible, is a Class-D value moved
  after measurement** — which R-6 forbids and which BE has a standing instruction
  to refuse. Revision 1's gate was exactly that: BE saw `b̂ = 1.037` and then
  chose the conditions under which 1.037 would not promote.
- **Every failure above came from the gate having to answer in advance.** A
  protocol written when the question is live can state its population, its
  interval convention, its arm, and its power against the effect actually
  observed — none of which Revision 1's gate could do, because none of it was
  knowable when the gate was written.

**What the monitor must report, so the eventual protocol has honest inputs:**

| | and why this one |
|---|---|
| `b̂` with the interval method **named**, **`n_clusters`**, and the **data-vintage cut** | §3.3: at small `n_days` the convention decides the verdict. **`n_clusters` because at `k≤3` the "95% CI" IS `[min,max]` of the per-day deltas** — verified on all 8 cells — so without `k` a reader cannot tell an interval from a range. **The vintage cut because two receipts printed BYTE-IDENTICAL `days_sampled` across samples 1,071 rows apart**: a day list is not a vintage |
| **the between-day sd of Δ** | without it the MDE row below is **not computable from this table**, which is how Revision 2 shipped a monitor that could not produce its own item 4. The two available values differ **4.97×** |
| **the fit scope and the scored population, named** | rows here otherwise permit three different populations in one report and name none. And the same Δ reads **8.3× differently** row-weighted vs day-weighted, so **report both** |
| **`b̂_t` and a trend statistic, session-stratified** | a weighted trend on the verdict days absorbs **99.3%** of between-day heterogeneity (z=3.55). A trending `b̂` makes a single pooled `b̂` the wrong summary — and the trend is confounded with UTC-block coverage running 100%/38%/38%/29%, so the split is mandatory, not optional |
| **the arm-slope split and its interval** | §4.4's "load-bearing result" is otherwise uncheckable — and it does **not** reproduce: btc/eth free-intercept `b_high − b_low = +0.227` against pooled +0.047 |
| per-coin `b̂`, **verdict coins only** | V5:333. The pooled figure averages a population that is 5/7 descriptive-only |
| OOS Δlog-loss vs `Identity`, sign included | the deployable currently scores **+0.00013** — indistinguishable, wrong sign |
| `n_days`, and the MDE **at that `n_days`** | Revision 1 quoted an MDE against a withdrawn effect; against the observed one the horizon is ~73 days |
| the excluded-quote fraction and its bias | §6.1 rule 3, which the probe itself violated: 5.2% dropped, tail-concentrated, unreported |

**`Identity` is a legitimate terminal answer for this module** — §7 already argues
the venue book is very hard to beat, and a module that confirms that has done its
job. Nothing here is a holding pattern.

---

## 7. Competitor baseline

The corpus's position is that this bias is *"public, monotone and capturable by
an isotonic map that any competitor can fit in an hour"* and that this is
bearish (`PM_DEEP_REVIEW.md:694-698`, `:374-381`). I agree with the fact and
partly disagree with the inference.

**Where it is right.** The recalibration is not a moat. It must not be the
justification for the programme, and it must not be the thing we spend budget
making fancier. Making the map more flexible buys nothing a competitor has not
already priced — which is a *second*, independent argument for the 1-parameter
form of §3.

**Where the inference needs care.** "Anyone can fit it" does not imply "it is
gone"; it implies it is competed down to the point where the marginal harvester
breaks even *after their costs*. The measured `b` is therefore already the
post-competition residual — and its size is consistent with that reading: a
**0.0004 Brier** walk-forward gain and a **≈2 pp** shift against a **4–5 c**
spread is roughly what "competed to the marginal harvester's break-even" looks
like. The bias has not been eliminated because eliminating it is not free; it has
been eliminated *down to the cost of eliminating it*. That is the same conclusion
as "it is gone" for anyone whose costs are average, and it means our only
possible advantage is being cheaper than average — a claim about execution, not
about belief. Two operational consequences:

1. **Persistence is an empirical question with a decisive test: is `b_t`
   declining in calendar time?** Build that monitor from day 1. A positive but
   *decaying* `b` invalidates deployment even while the level is positive.
2. **It changes what BE-Belief is FOR.** BE-Belief becomes a *quoting-placement
   input* — it tells DE where resting is systematically wrong-sided — not an
   alpha source. The moat, if the programme has one, must live in
   BE-FlowAndFills (surviving the 60–97% selection haircut) or in the incentive
   term. BE-Belief's honest self-description: *"I am a commodity input. I remove
   a known bias so downstream sizing is not systematically wrong. I am not the
   edge."*

It also reinstates the guard that was demoted (`PM_MECHANISM_EXPERIMENTS.md:715`,
originally `PM_SKETCH_REVIEW_ITER1_S.md:144-146`): **the walk-forward
recalibrated book is now a mandatory baseline for any future alpha claim.** Any
model that beats the raw mid but not the recalibrated mid has demonstrated
public recalibration, not information. Under this plan that baseline is not a
deferred diagnostic — it *is* BE-Belief, so the guard is enforced by construction.

---

## 8. Interaction with BE-Uncertainty

Stated as a contract, so the σ planner can design against it.

**Under the recommendation, BE-Belief needs σ for the LEVEL in exactly one
place: the fallback.** Everything else is dynamics.

| what BE-Belief needs from BE-Uncertainty | why | if absent |
|---|---|---|
| **nothing for `p_hat` on the main path** | `p̂ = G_b(m)` is algebraic in an observed price. This is the cycle-breaker of §1.3 | n/a |
| `path_law.var_of_increment(h)` | carried through to the sniping / adverse-selection and inventory-horizon terms in DE and BE-FlowAndFills. BE-Belief **carries, does not compute** | `NullPin` at field granularity (NF-2, `PM_STRUCT_ITER2_A.md:254-261`) |
| `jump_tail(m, h)` | same — carried. Note the open defect: `contracts.yaml:385` still types it `float` with two unnamed arguments and no declared unit (`PM_STRUCT_ITER2_B.md:664-666`) | `NullPin`, bias OPTIMISTIC (a missing tail reads as zero) |
| σ_eff **only** for the fallback forecast | when the book is `Unavailable`, **and that is not rare**: even in the dense series, mean top-of-book age is 12–20 s in the ATM and extreme buckets (§A.1), because a quiet book emits no events. There `w` goes 0 → 1 and the stream forecast is the only level available | `Unavailable`, and the belief refuses rather than guessing |

So: **if BE-Uncertainty concludes σ is needed only for DYNAMICS, that is fully
consistent with this plan and no rework follows.** The blend weight `w` is not a
free tuning knob; it is 0 on the main path (measured, not assumed) and 1 in the
declared `FallbackPolicy`.

**Ownership boundary (R-SSOT).** BE-Belief owns the **parameter uncertainty of
its own fit** — `Var(b̂)` propagated to `Var(p̂)` by the delta method. That is not
a variance component of the *outcome* and must not be registered with
BE-Uncertainty's `VarianceGroup`. Conversely BE-Belief owns no σ_eff, no
`w_hat`, no `VarianceComponent`. Per `PM_STRUCT_ITER1_B.md:470`, if BE-Belief's
anchor choice creates outcome variance (e.g. a microprice anchor's estimator
noise), **BE-Belief registers that component into the budget BE-Uncertainty
owns** — data flows up, ownership stays put.

`Var(p̂)` has one documented use and only one:
`w = var_p / (var_p + var_book)` (`PM_STRUCT_ITER2_B.md:660-663`). It is exposed
as an aggregate `estimator_var`, with per-source marginals on `constituents`.

**BE may not read EV** — σ_book is an EV object (`PM_STRUCT_ITER2_B.md:786`).
BE-Belief's `consumes` list contains no `EV-*`.

---

## 9. Power: what is claimable now, at 7 days, at 30 days

Current inventory (measured on disk now): `markets.jsonl` 1,669 rows;
`resolutions.jsonl` 1,641 final, zero duplicate slugs; `window_start` span
**2026-08-19 14:25 → 2026-08-20 10:40 UTC ≈ 20.25 h**; 2 UTC dates; 1,744 raw
window files; 7 coins. (The corpus's "~1.4 days" label is not an elapsed time —
`PM_DEEP_REVIEW.md:11` gives 13 h 25 m and `PM_QUANT_REVIEW.md:899` says 13.5 h.
Read "1.4 days" as "spans two UTC dates, one walk-forward test day".)

Effective sample, after the §6.1 corrections: **1,645 windows with book coverage,
one outcome each** (day 20684: 762, up-rate 0.5505; day 20685: 882, up-rate
0.5340), ~1.5 effective coin-factors, **one walk-forward test day = one cluster**.

The pseudo-replication penalty is now measured rather than assumed: on the same
sample, se(`b̂`) is **0.028 naive** and **0.040–0.042 window-clustered** — a
**1.45× inflation**, i.e. every unclustered t-statistic in the corpus is ~45%
too large. (Less than the naive √5 because the 5 grid points probe different
mid values and so are not pure replicates; more than 1 because they share one
Bernoulli outcome.) The *day*-clustered inflation on top of that is unmeasurable
with two days.

| | claimable | not claimable |
|---|---|---|
| **Now (2 days, 1 test day)** — *all of these are DONE, §A* | pooled `b̂ = 1.145 ± 0.042` and its day-to-day stability; a **point** OOS Δlog-loss / ΔBrier vs raw book on the single test day; the §4.4 asymmetry decomposition; the §5.2 `r`-flatness; the §2.1 stale-vs-dense rebuild, which is a data-quality fix and needs no power at all | **any CI on the OOS delta** (one cluster); per-coin `b`; `b(r)`; a *confident* drift/FLB separation (`a` is only 1.1σ); anything about P&L; any statement that survives a down-drift period |
| **7 days (~10.3k windows, 6 test days)** | day-clustered paired Δ vs raw book with a real (wide, 5 df) interval; the folded/unfolded asymmetry test with a CI; `a`-stability across days; the first read of `b_t` drift; the core-vs-extreme domain split | per-coin `b`; `b(r)` with 3+ knots; a capture-ratio CI |
| **30 days (~44k windows, 29 test days)** | usable day-clustered CIs; `b(r)` with 2–3 knots; per-coin `b` as a shrinkage challenger; a `b_t` trend test (§7); **and the actual gate — a markout-based capture-ratio estimate with a CI** | nothing about a regime we have not observed |

**The falsification that only calendar time can deliver:** 30 days is the
shortest window likely to contain a *down*-drift period. The drift/FLB
separation is now supported by *within-sample* evidence (§4.4: the rotation
survives a free intercept), but every observed day drifts the same way, so the
separation has never been tested against a sign flip. **Until one is observed,
`b̂ > 1` and "the market went up" are not fully distinguishable in cents (§2.2).**
State it that way in every result table.

**And the one thing more days does NOT fix:** the capture ratio. The 60–97%
haircut is a selection effect, not a sample-size effect. No amount of data makes
the unconditional gap harvestable.

---

## 10. Ways this design could be wrong

Ranked by how much they would change the answer.

1. **The FLB is a tick-floor artifact.** *Partly tested, §5.1 — downgraded.*
   40% of the sample sits where the grid mechanically caps the quote (bucket
   0.9–1.0: ask 0.975 vs realised 0.988, an edge the size of the grid). But the
   extremes do **not** dominate the fit (se 0.065 vs 0.042 core) and they agree
   on the slope (1.168 vs 1.145). The distortion is a level effect
   (`a_extreme = +0.428`), already neutralised by refusing to deploy `a`.
   *Remaining test:* refit inside the 0.001 tick regime.
2. ~~**It is a translation, not a rotation.**~~ **TESTED, §4.4 — rotation
   survives.** With a free common intercept, `b_high − b_low = +0.099`,

   > **⚠ Revision 3 — `TESTED` WITHDRAWN, status reverts to OPEN (UNDERPOWERED).**
   > The free-`a` CI on `b_high − b_low` is `[−0.169, +0.349]`: half-width 0.259 →
   > se 0.132 → **MDE at 80% power = 0.370, which EXCEEDS the +0.241 this row
   > claims to have dissolved** — and **+0.241 lies inside the interval**. The
   > interval widened to admit both zero and the original estimate. Nothing
   > collapsed and nothing survived. **Striking a risk through as `TESTED` on a
   > failure to reject is the defect this plan's own iteration-1 verdict convicted
   > Revision 0 of**, sitting in its risk register for three revisions.

   95% CI [−0.169, +0.349]. The apparent 2.4σ asymmetry is an artifact of
   pinning `a = 0` during *estimation*. Retained as a standing regression test,
   not as an open risk.
3. **It is the rally — and in cents it looks entirely like the rally.** Two days,
   every coin up, pooled 0.5436; measured `a ≈ +0.09…+0.13`. Worse, the
   *executable* re-expression (§2.2) is **one-sided**: buying Up at the ask pays
   in every bucket ≥ 0.4 while selling Up at the bid pays in exactly one bucket.
   A rotation pays on both wings; a drift pays on one. *Mitigation:* pin `a = 0`
   at deployment — which costs two-thirds of the OOS gain (§A.4), and that is the
   honest price of not betting on a rally. *Test:* requires an observed
   down-drift period — 30 days. **This is the single largest unresolved threat,
   and nothing in the current sample can settle it.**
4. **Everything was measured on a stale book, and the stale book flatters us.**
   Snapshot-only series, p90 6.2 s stale, a 6 s-stale mid wrong by 5.6 c mean /
   13.5 c p90. Measured consequence: `b̂` reads 1.182 stale vs **1.145 dense**,
   overstating the excess over 1 by 26%, and the apparent spread is 1.0–1.5 c too
   wide ATM. *Test:* the §2.1 rebuild. Every corpus number must be re-derived.
5. **The mid is not tradeable.** ATM spread 6–8 c, not the 2–4 c the corpus
   assumes; **— REFUTED FOR BTC/ETH (U8): ATM median is 1 tick, so on the two
   liquid coins the mid is roughly half a tick from either side and this item
   does not apply. It stands for sol/doge/xrp/bnb/hype (3–7 ticks). Since the
   fill question is now restricted to btc and eth (`FLOW_MODEL_PROTOCOL_V4`
   `verdict_coins`), this premise is retired for the coins that carry verdicts —** the fitted map moves the belief ~2 pp at its best moneyness, i.e.
   ≈ one half-spread. *Test:* §2.2's `realised−bid` / `ask−realised`
   re-expression, mandatory in every table.
6. **Selection eats it.** 60–97% measured haircut. BE-Belief cannot fix this and
   must not claim the gap as P&L (§1.2). *Owner:* BE-FlowAndFills.
7. **Pseudo-replication.** 5–6 grid points per window share one Bernoulli
   outcome; every n in the corpus is inflated 5–6×. **Measured effect on the
   standard error: 1.45×** (0.028 naive → 0.041 window-clustered), so every
   unclustered t in the corpus is ~45% too large.
8. **Late-window calibration is near-tautological.** At `r = 30 s` the book's
   Brier is 0.0174; "calibration" there is a tick-rounding residual, not a
   belief. *Mitigation:* always report `r`-stratified.
9. **`b` is being competed away.** A positive but decaying `b_t` invalidates
   deployment. *Test:* the §7 monitor, from day 1.
10. **"ŵ → 0" is narrower than it sounds.** It says *our settlement-stream
    forecast at 1.7 s lag* has no edge — 85% of that lag is PM-side publication
    delay we cannot remove. A Binance-lead forecast has never been tested
    (`PM_DEEP_REVIEW.md:648-663`: HYPE has no Binance leg and `data/mm_hf/vision/`
    is aggTrades-only). Do not let `ŵ→0` be quoted as "no alpha exists".
11. **No independent check on the anchor.** `mid_up + mid_down = 1.0000`; the
    Down-token confirmation is algebraically vacuous (§2.5). There is exactly one
    price and no redundancy.
12. **Pooling `b` across coins could be wrong** if the FLB is really a
    liquidity/spread effect rather than a participant-mix effect — spread differs
    materially by coin. *Test:* model 6 at 30 days; earlier, check whether adding
    spread as a covariate absorbs `b`.
13. **The map could be right and nearly useless — and on current evidence it
    largely is.** `b̂ = 1.145` is real in-sample, but the deployed (a = 0) map
    buys **0.0004 Brier** out of sample. If the ≥7-day CI on that delta spans
    zero, the honest output is `RecalibrationForm.Identity` and BE-Belief is a
    pass-through plus a monitor. **The contract must permit that outcome, and it
    does.** Anyone reading `b̂ = 1.15` as "a 15% edge" has misread it.

---

## A. Appendix — measurements run for this plan

All computed at knowledge time (`recv_ns`) on
`data/pm_5min/raw/{20260819,20260820}`, 1,645 windows with book coverage, 8,208
(window, decision-time) rows on the grid {240, 180, 120, 60, 30} s into the
window. Top-of-book rebuilt from `price_change.best_bid/best_ask` ∪ `book`
(§2.1). "Core domain" = `|logit m| ≤ 3` (6,390 rows). All standard errors are
**window-clustered bootstraps** (400 resamples) unless labelled naive. Scripts
are in the session scratchpad, not committed — reproduce them under §12 step 1.

### A.1 Calibration, dense vs snapshot-only (both days, all `r`)

| bucket | n | windows | mid | bid | ask | realised | realised−mid | se (clustered) | realised−bid | ask−realised | spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0–0.1 | 1135 | 616 | 0.037 | 0.026 | 0.047 | 0.042 | +0.006 | 0.008 | +0.016 | +0.005 | 0.021 |
| 0.1–0.2 | 501 | 394 | 0.148 | 0.128 | 0.167 | 0.114 | **−0.034** | 0.016 | −0.015 | +0.054 | 0.039 |
| 0.2–0.3 | 620 | 467 | 0.250 | 0.230 | 0.270 | 0.244 | −0.007 | 0.022 | +0.014 | +0.027 | 0.041 |
| 0.3–0.4 | 733 | 542 | 0.349 | 0.327 | 0.371 | 0.340 | −0.009 | 0.021 | +0.013 | +0.032 | 0.044 |
| 0.4–0.5 | 852 | 607 | 0.451 | 0.432 | 0.470 | 0.478 | +0.027 | 0.026 | +0.046 | −0.007 | 0.039 |
| 0.5–0.6 | 878 | 627 | 0.546 | 0.525 | 0.567 | 0.579 | +0.033 | 0.024 | +0.054 | −0.012 | 0.042 |
| 0.6–0.7 | 753 | 597 | 0.646 | 0.624 | 0.669 | 0.697 | **+0.051** | 0.020 | +0.073 | −0.028 | 0.045 |
| 0.7–0.8 | 659 | 494 | 0.749 | 0.729 | 0.769 | 0.803 | **+0.054** | 0.019 | +0.074 | −0.033 | 0.041 |
| 0.8–0.9 | 681 | 520 | 0.846 | 0.823 | 0.868 | 0.883 | +0.037 | 0.014 | +0.060 | −0.014 | 0.045 |
| 0.9–1.0 | 1396 | 707 | 0.965 | 0.954 | 0.976 | 0.985 | +0.020 | 0.004 | +0.031 | −0.009 | 0.021 |

Pooled realised 0.5422 vs pooled mid 0.5231 — the **+1.9 pp drift** the whole
design has to survive. Note buckets 0.2–0.4 have gaps *within one se of zero*:
the "monotone slope" is a two-sided story about the wings, not a clean ramp.

Mean top-of-book age (`stl`, not shown above) is 0.1–1.2 s in the active buckets
but **12–20 s at 0.0–0.1, 0.4–0.6 and 0.9–1.0** — a quiet book emits no events,
so "fresh" and "live" are different properties. This is why
`BeliefWarmupPolicy.max_book_staleness` is a contract field and not a constant,
and why the σ fallback of §8 is a real code path rather than a formality.

Snapshot-only (incumbent method) on the identical windows differs materially:
0.3–0.4 reads −0.018 vs −0.009 dense; 0.4–0.5 reads +0.015 vs +0.027; and mean
spread is inflated by 1.0–1.5 c ATM (0.057 vs 0.044 at 0.3–0.4) because bid and
ask are observed at different instants.

### A.2 Pooled fit

| sample | n | windows | `a` | `b` | se naive | se clustered |
|---|---|---|---|---|---|---|
| all rows | 8208 | 1644 | +0.128 | 1.147 | 0.028 | 0.040 |
| **core `\|logit m\| ≤ 3`** | **6390** | **1639** | **+0.122** | **1.145** | 0.032 | **0.042** |
| extreme `\|logit m\| > 3` | 1818 | 1141 | +0.428 | 1.168 | 0.058 | 0.065 |
| snapshot-only, all rows | 8208 | 1644 | +0.126 | 1.182 | 0.028 | 0.041 |

Per day: 20684 `a=+0.124, b=1.184 ± 0.078`; 20685 `a=+0.119, b=1.113 ± 0.067`.

The extreme domain carries a much larger intercept (+0.428) for a similar slope
— consistent with §5.1's tick-floor account, and a reason to keep it under its
own parameter.

### A.3 Drift vs rotation

Pinned `a = 0`: `b_low = 1.027 ± 0.063`, `b_high = 1.268 ± 0.069`.
Direction-folded: `b(longshot=Up) = 1.027 ± 0.067`,
`b(longshot=Down) = 1.268 ± 0.073`, pooled fold `1.147 ± 0.046`.
Free common `a`: `a = +0.087 ± 0.078`, `b_low = 1.096 ± 0.073`,
`b_high = 1.195 ± 0.080`, difference 95% CI **[−0.169, +0.349]**. See §4.4.

### A.4 Walk-forward: fit day 20684, score day 20685 (core domain, dense)

Train 3,011 rows / 758 windows → `a = +0.124`, `b = 1.184` (anchored `b = 1.189`).
Test 3,379 rows / 881 windows.

| model | log-loss | Δ | Brier | Δ |
|---|---|---|---|---|
| 0 — raw book (`b=1, a=0`) | 0.5340 | — | 0.1788 | — |
| **1 — anchored `b` (the deployable)** | 0.5334 | **−0.0006** | 0.1784 | **−0.0004** |
| 2 — free `(a, b)` (not deployed) | 0.5320 | −0.0020 | 0.1780 | −0.0008 |
| 5 — 10-bin isotonic | 0.5352 | **+0.0012** | 0.1797 | **+0.0009** |

Three readings, all load-bearing:

1. **The recalibration works, and it is small.** −0.0004 Brier against the 0.0201
   by which our own forecast loses to the book: the map recovers **2%** of that
   gap.
2. **Two-thirds of the gain comes from the drift intercept we refuse to deploy**
   (−0.0020 → −0.0006). Carrying `a` "worked" on day 2 only because the rally
   continued for 20 hours. That is the cost of honesty, paid up front.
3. **Isotonic is worse than doing nothing.** Empirical confirmation of §3.1.

**One test day is one cluster. None of these deltas has a CI.** They are point
estimates whose sign is the only claimable content, and §9's 7-day gate exists
precisely to put an interval around them.

---

## 11. Canonical contract — `BE-Belief`

> ### ⚠ Revision 1 — THIS BLOCK IS NOT SUBMISSION-READY
>
> Revision 0 wrote it against **v12** and proposed a `12 → 13` bump. The file is
> at **v22**, with **v23** in batch consolidation. **Four defects must be cleared
> before this block is proposed to the contract batch, and until they are it is
> design text, not a delta:**
>
> 1. **Version.** `12 → 13` is stale by ten revisions. The bump is against
>    whatever v23 lands at, and the R-SSOT name-collision check must be re-run
>    against **v22/v23**, not v12 — a name free in v12 may well be taken now.
>    *(Unverified as of Revision 1. Do not submit on the v12 check.)*
> 2. **The `−0.0004` Brier gain is superseded.** It is quoted three times below
>    as a settled property. The probe measures **+0.00013 log-loss OOS**
>    (indistinguishable, wrong sign) on 4 days. **A contract must not ship a
>    performance claim the plan's own §0 has withdrawn.**
> 3. **The `6–8 c` ATM spread is WITHDRAWN for btc/eth** — §2.2's own U8 note
>    records the measured value as **1 tick** on 2.29 M quotes, and the fill
>    question is now restricted to btc/eth. The block still carries the refuted
>    figure.
> 4. **`TopOfBook` cannot be built from the source this same block mandates.**
>    The block requires it while the mandated input series does not carry the
>    fields it needs. Either the type shrinks to what the source provides or the
>    source changes — **this is a contradiction inside one block**, and it is the
>    one that would have failed a checker rather than a reviewer.
>
> Defect 4 is load-bearing and 1–3 are stale values. **Under R-38(d) clearing
> them buys an obligation to re-measure, not a verdict** — fixing the numbers does
> not make the recalibration adoptable. **Nothing in this document can** — §6.5's
> gate is DELETED and promotion requires a separate frozen protocol.

Schema style of `live/pm_research/contracts/contracts.yaml` v12. **Not applied to
that file; this block is the proposal.** All additions; per
`contracts/migrations.yaml`, adding types/fields/modules needs no migration
record, so this is a pure `version: 12 → 13` bump. Naming checked against R-SSOT:
no name here is declared elsewhere in v12, and none of `σ_eff`, `w_hat`,
`VarianceComponent`, `K`, `E[X_T]`, `w_declared` or `ScenarioLossLimit` is
touched.

```yaml
# ---------------------------------------------------------------- types (new)
types:

  PriceSummary:
    kind: open_protocol
    registry: PluginRegistry
    protocol:
      summarise: '(TopOfBook) -> float | Unavailable'
      config_schema: JsonSchema
    builtin_ids:
    - mid
    - microprice
    - bid
    - ask
    - last_trade
    notes: 'the scalar the belief is anchored to. `mid` is the DEFAULT. `microprice` is a
      GATED challenger: no imbalance/OFI measurement exists in-programme, touch size is
      quantised at the reward minima (p10/p50/p90 = 4.98/29.98/50.0 shares) and depth sits
      outside the touch (138 shares within 1c vs 1290 within 4.5c), so it must beat `mid`
      out-of-sample before it may be declared. `last_trade` REQUIRES a declared frozen lag:
      the price_change carrying best_bid/ask for a match can be emitted BEFORE the
      last_trade_price for the same match.

      '

  TopOfBook:
    fields:
      best_bid: float
      best_ask: float
      bid_size: float
      ask_size: float
      tick: float
      complement_of: TokenId?
      source_events: enum:SNAPSHOT_ONLY|SNAPSHOT_PLUS_DELTA
      t_known: Timestamp
      provenance: Provenance
    notes: 'OWNER IS DA-Normalize, not BE-Belief (R-SSOT); listed here because BE-Belief is
      its first declared consumer and the type does not yet exist. MUST be built from
      `price_change.best_bid/best_ask` UNIONED with `book` snapshots: snapshots alone are
      ~4.1k events/window vs ~149.7k deltas, p90 inter-arrival 6.24s, and a 6s-stale mid is
      wrong by mean 5.6c / p90 13.5c -- larger than the 3-7c effect being fitted.
      `source_events: SNAPSHOT_ONLY` is a materially different object and MUST be marked.
      The two tokens of a market are exact complements (mid_up + mid_down = 1.0000 median,
      mean 1.0001), so `complement_of` carries no independent information and MUST NOT be
      used as a cross-check.

      '

  RecalibrationForm:
    variants:
    - Identity
    - 'LogitAnchored(b: ParamId, anchor: float)'
    - 'LogitAffine(a: ParamId, b: ParamId)'
    - 'PiecewiseLogitAnchored(b_by_r: ParamId, knots: list[float])'
    - 'Isotonic(artifact: ImmutableId)'
    notes: 'ESTIMATE with LogitAffine, DEPLOY with LogitAnchored. These are different
      operations and conflating them corrupts the slope: pinning a = 0 during ESTIMATION
      leaves the sample drift nowhere to go, so it leaks into b and manufactures a spurious
      asymmetry between the m<0.5 and m>=0.5 arms (measured: b_low 1.027 vs b_high 1.268
      pinned, versus 1.096 vs 1.195 with a free common intercept, difference 95% CI
      [-0.169, +0.349]). LogitAnchored with anchor = 0.5 is a pure ROTATION about 0.5 and is
      drift-neutral by construction, which is why it is the DEPLOYED form. Isotonic is a
      REFERENCE challenger for measuring overfitting, NOT a deployable: measured walk-forward
      it is WORSE than the raw book (Brier +0.0009) while the 1-parameter map is better
      (-0.0004). REVISION 1: that -0.0004 is SUPERSEDED -- the 4-day probe measures
      +0.00013 log-loss OOS, indistinguishable and of the wrong sign. Identity is
      no longer merely "a legitimate outcome"; it is the DEPLOYED DEFAULT, and
      LogitAnchored is promoted onto it only by a SEPARATE FROZEN PROTOCOL with a
      calendar trigger. The five-condition gate this line used to name was deleted
      in Revision 2 and this line kept citing it -- see the Revision 3 sweep.
      The burden runs the other way from Revision 0.

      '

  Recalibration:
    fields:
      form: RecalibrationForm
      input: PluginRef
      fold: enum:NONE|LONGSHOT_FOLDED
      domain: '(float, float)'
      on_out_of_domain: enum:IDENTITY_FLAGGED|REFUSE
      scope: ScopeKey
      drift_intercept: float | NullPin
      fit_data_through: Timestamp
      fit_n_windows: int
      fit_n_days: int
      fit_cluster_unit: enum:WINDOW|DAY
      param_cov: dict[str, float]
      artifact_id: ImmutableId
      provenance: Provenance
    notes: 'the fitted VALUES of a and b live in SP-Params keyed by (ParamId, ScopeKey) with
      their own fit_data_through/artifact_id; this record holds REFERENCES only (R-SSOT,
      same precedent as loss_limit and rewards_band). `fold: LONGSHOT_FOLDED` fits on
      q = min(m, 1-m) with y = 1 iff the longshot side won. Folding is a DIAGNOSTIC, not a
      drift remover: it relocates drift into a disagreement between the two arms (measured
      1.027 vs 1.268) and cancels it in the pooled fold only to first order and only under
      arm balance. `domain` defaults to |logit m| <= 3: 40% of samples sit outside it, where
      the tick grid (0.01, switching to 0.001) mechanically caps the quote and manufactures
      an apparent favourite bias the size of the grid. `fit_cluster_unit` exists because the
      5-6 decision times inside one window share ONE Bernoulli outcome; an unclustered fit
      inflates n by 5-6x.

      '

  BeliefWarmupPolicy:
    fields:
      min_fit_days: int
      min_fit_windows: int
      max_book_staleness: Duration
      on_violation: UnavailableAction
    notes: 'fail-loud thresholds for the belief. Violation yields
      Unavailable(WARMUP|STALE_BOOK) with `cause` propagated, never a silently-degraded
      p_hat.

      '

# ------------------------------------------------------- types (extended)
# additive fields on the existing BE-Belief-owned type; no migration record needed
  BeliefProcess:
    fields:
      p_hat: float
      p_raw: float
      anchor: TopOfBook
      recalibration: Recalibration | Unavailable
      estimator_var: float | NullPin | Unavailable
      link: LinkFunction
      jump_tail: float | NullPin | Unavailable
      path_law: PathLaw | NullPin | Unavailable
      constituents: dict[str, Constituent]
      staleness: Duration
    notes: 'p_hat is the UNCONDITIONAL belief E[Y | book state]. It is NEVER the
      fill-conditional E[Y | book state, FILLED] -- that conditioning is BE-FlowAndFills''
      adverse-selection term, and the measured haircut from unconditional gap to realised
      markout is 60-97%. Baking it in here double-counts. `p_raw` is the un-recalibrated
      anchor, exposed so EV and DE can see the size of the disagreement without recomputing
      it. `anchor` is carried so consumers size against the EXECUTABLE pair: the ATM book is
      6-8c wide (measured p50 spread 0.05, 0.07-0.08 ATM), so a 3-7c edge at the mid is
      0 to -4c at the ask, and no consumer may treat p_hat - mid as tradeable.
      `estimator_var` is Var(p_hat) from the delta method on Var(b_hat) -- the uncertainty of
      BE-Belief''s OWN FIT, not of the outcome; it is NOT a VarianceComponent and is NOT
      registered with BE-Uncertainty''s VarianceGroup. Its single documented use is
      w = var_p / (var_p + var_book). `constituents` is EV-READABLE ONLY; DE MUST NOT branch
      on it. The stream forecast appears as constituents["stream"] with its measured weight
      w (currently ~0); w -> 1 only under the declared FallbackPolicy when the book is
      Unavailable.

      '

# --------------------------------------------------------------- module (new)
modules:

  BE-Belief:
    consumes:
    - DA-Normalize
    - BE-Target
    - BE-Uncertainty
    - SP-Params
    - SP-Instrument
    - SP-Venue
    produces: Known[BeliefProcess] | Unavailable
    requires:
      recalibration: Recalibration
      price_summary: PluginRef
      warmup: BeliefWarmupPolicy
      fallback: FallbackPolicy
      link: LinkFunction
    ports:
    - state_view
    - rng
    - artifact_resolver
    - telemetry_out
    notes: 'produces the venue book RECALIBRATED, not our own forecast: measured
      walk-forward, our stream-anchored p_hat loses to the book at every horizon through
      three sigma generations (+0.0291 / +0.0277 / +0.0201 Brier), so w_hat -> 0 and the
      recalibration IS the edge or there is none. p_hat is ALGEBRAIC in an observed price, so
      the LEVEL needs no sigma -- this dissolves the BE-Uncertainty <-> BE-Belief cycle and
      BE-Link is not needed. sigma enters the level in exactly ONE place: the FallbackPolicy,
      when the book is Unavailable. path_law and jump_tail are CARRIED from BE-Uncertainty,
      never computed here. Consumes no EV-* (sigma_book is an EV object; BE may not read EV)
      and takes NO venue port. t_known is derived per R-KNOW as MAX over live inputs
      (anchor.t_known, BE-Target, BE-Uncertainty); Recalibration.fit_data_through is a
      property of the artifact and does NOT enter t_known. Any future alpha claim must beat
      this module as its baseline, not the raw mid -- a model that beats the raw mid but not
      the recalibrated mid has demonstrated public recalibration, not information.

      '

# ------------------------------------------------------------ null semantics
# NOT a `modules:` entry -- see the conformance notes below. Declared by
# BE-Belief per R-NULL, at FIELD granularity (NF-2).
null_pins:
  BE-Belief:
    drift_intercept:
      field: BeliefProcess.recalibration.drift_intercept
      assumption: a = 0; the map is anchored at p = 0.5 and carries no level shift
      bias_direction: PESSIMISTIC
      declared_by: BE-Belief
    jump_tail:
      field: BeliefProcess.jump_tail
      assumption: no jump component supplied by the active BE-Uncertainty impl
      bias_direction: OPTIMISTIC
      declared_by: BE-Belief
    path_law:
      field: BeliefProcess.path_law
      assumption: no path law supplied; horizon-variance consumers must refuse
      bias_direction: OPTIMISTIC
      declared_by: BE-Belief
```

**Checked mechanically against v12 before writing this block:**

- YAML parses under `StrictLoader` semantics (no duplicate keys).
- **Reference closure holds** — every type reference in every field, variant,
  protocol member and module input resolves to a v12 type, primitive, external,
  or one of the five new types here. (The only "unresolved" tokens are variant
  *constructor* names, which are declarations, not references — same shape as
  v12's `Declared(...)` / `SharedDraws(...)`.)
- **The `BeliefProcess` extension is strictly additive**: adds `p_raw`, `anchor`,
  `recalibration`, `estimator_var`; **removes nothing**. So no `migrations.yaml`
  record is required — a removal or narrowing would need one.
- **Name collisions: none.** `PriceSummary`, `TopOfBook`, `RecalibrationForm`,
  `Recalibration`, `BeliefWarmupPolicy` are absent from v12's `types`,
  `prelude.primitives` and `prelude.external` (the local-AND-external duplicate
  that broke `Hyperedge` cannot recur here).

**Conformance notes for whoever applies this to `contracts.yaml`:**

- `contract_check.py` recognises exactly four module keys — `produces`,
  `consumes`, `requires`, `ports`. `notes:` on a *module* is not flattened into
  the diff, so a notes-only change is invisible; do not encode a decision there
  that has to be enforced.
- The `_null_pins` block above is **not** a recognised key. It is written here so
  the pins are on the record; they must land either as `ModuleManifest.null_semantics`
  (defined at `contracts.yaml:308-316`, currently unused by every module — this
  would be its first user) or as a new `checks:` entry under `R-NULL`.
- `consumes` entries are module references, which the producer-closure check
  skips (`contract_check.py:130-131`). `DA-Normalize` and `BE-Target` /
  `BE-Uncertainty` have no module records yet; naming them here is safe but they
  remain open MUST-FIX M11-4 items.
- `TopOfBook` is declared here only because it does not exist. **Its owner is
  `DA-Normalize`.** Moving it later is a rename, which *does* need a
  `migrations.yaml` record.
- `PriceSummary` is an `open_protocol` + `PluginRegistry` per R-OPEN, so the
  microprice challenger is a plug-in registration rather than a schema edit.
- Bump `contracts.yaml: version: 13` and match it in `PM_ARCHITECTURE.md:3-4`
  and the commit label.

---

## 12. Sequenced work (for the implementer, not done here)

Steps 1–3 were run as part of this planning pass (§A); the implementer's job on
them is to move the throwaway scripts into the repo as a reproducible probe, not
to rediscover the answers.

| # | step | gate | days | status |
|---|---|---|---|---|
| 1 | rebuild `TopOfBook` from `price_change` ∪ `book`; re-derive every book number in the corpus | dense vs snapshot reported; no gate | 2 | **done, §A.1** — must be productionised |
| 2 | re-express all FLB tables at bid/ask, `r`-stratified, window-clustered | mandatory format change | 2 | **done, §A.1** |
| 3 | fit models 0–3 and 5, walk-forward | ~~`b̂ > 1` point estimate~~ **BAR FAILED BY THE PROBE** (pooled 1.037; 3 of 4 days below 1) | 2 | **done, §A.2–A.4** |
| 4 | tick-regime control (0.001 subset) on the extreme domain | extreme bias shrinks ≥5× ⇒ it is the grid | 2–7 | open |
| **5** | ~~day-clustered CI on Δlog-loss vs raw book~~ | ~~CI excludes 0, else ship `Identity`~~ · ~~**7** days~~ | — | **DELETED IN REVISION 3 — see below** |
| 6 | `b(r)` and per-coin challengers; **`b_t` TREND estimator** (§7's monitor, which had none) | beats pooled OOS | **30** | open |
| 7 | capture-ratio markout — **the actual deployment gate for the programme** | owned by BE-FlowAndFills | **30** | open |

> ### ⚠ Revision 3 — STEP 5 IS DELETED, AND IT WAS THE LAST LIVE PROMOTION RULE
>
> §6.5 said *"no rule promotes it automatically"* one revision ago. **Step 5 kept
> saying "the go/no-go for this module", and `be_belief.py` kept EXECUTING it** —
> `STEP5_MIN_DAYS = 7`, `would_ship_today: RECALIBRATION|IDENTITY`. So at 7 days a
> machine-generated receipt would have announced an automatic promotion the plan
> says cannot exist. **Deleting prose does not delete a rule that is
> implemented**; the code is now a monitor and emits no verdict.
>
> **Three reasons the bar was wrong quite apart from being orphaned:**
>
> 1. **Its statistic is a range, not an interval.** At 3 day-clusters the
>    percentile bootstrap returns exactly `[min, max]` of the per-day deltas —
>    verified on all 8 model×population cells. *"CI excludes 0"* means *"all three
>    days share a sign"*: a **25%** event under a symmetric null, quoted at 5%.
> 2. **It tests against the wrong null.** A `k`-parameter map fitted on `n_eff` is
>    *expected* to lose out-of-sample by `k/(2·n_eff)` **even when `b = 1`
>    exactly**. Testing Δ against **0** conflates *"no rotation"* with *"rotation
>    present, not estimable at this `n`"*. Against the correct benchmark the
>    deployable's excess is +0.000042 pooled and +0.000199 on the verdict coins —
>    both deep inside their own intervals.
> 3. **Seven days is not the horizon.** With `τ̂ ≈ 0.15` on the verdict coins the
>    honest figure is **17–22 days** for `b̂`'s interval to exclude 1, and **~50**
>    scored days for Δ to reach an MDE of 0.0006.
>
> **Nothing replaces it in this document.** Promotion requires a separate protocol
> with a **calendar** trigger — *"at `n_days ≥ 30`, write and freeze a promotion
> protocol, evaluated only on data after the freeze date"* — which is a commitment
> rather than a threshold on the effect, and so is the only construction that
> satisfies R-6 while the monitor is publishing daily.

---

## 13. What was never attempted (Revision 1, swept in Revision 3)

Added because R-53 established that **review loops audit what exists and nobody
audits what was never attempted** — two of nine mitigation channels in a sibling
programme existed only because someone asked that question. Iteration 1 of
`BE_BELIEF_REVIEW_LOOP.md` grepped 1,165 lines of Revision 0 and found **zero
hits** for `taker`, `tier`, `Hawkes`, `f_r`, `queue`, `EWMA`, `rolling`,
`time-of-day`, `volatility`.

**Revision 0 considers exactly two conditioners: moneyness and `r`.** Both are
properties of the *contract*. None of the conditioners below is a property of the
contract, and none was rejected — they were never raised.

This section records them **as open, not as recommended.** Each names why it is
plausible and what would kill it, so a successor inherits the question rather
than the absence.

| conditioner | why plausible | available now? | what would kill it |
|---|---|---|---|
| ~~**liquidity tier**~~ **RETIRED** | ~~a 2-group split has ≈3.5× the per-cell sample~~ **Both claims were false.** (a) **No liquidity tier exists** — every `tier` identifier in the codebase is `tier1`/`tier2` data-lane distillation, so "already computed" was the corpus's own *name-is-not-the-definition* class, committed inside the section written to stop inherited errors. (b) **The axis is VACUOUS on the verdict population**: btc and eth are both 1-tick ATM, so a liquid/thin split has *zero variation* inside the only coins that can carry a verdict — and at n=2, per-tier IS per-coin. | ~~Yes~~ **N/A** | `b̂` indistinguishable across tiers at day-clustered CI |
| **signed taker flow** | A **fully built flow model exists** (`flow_intensity.py`, `FLOW_MODEL_STATE.md`) and appears nowhere in this plan. Order flow is the standard conditioner for book mispricing. | **Yes** | no incremental Δlog-loss over moneyness alone |
| **session / time-of-day** | §4.3's entire pin argument rests on *"a 20-hour rally"*, and §6.3's pooling mechanism is a **participant mix that is a function of session**. The plan's own reasoning implies this variable and never tests it. | **Yes** | `b̂` and `â` flat across UTC session blocks |
| **non-stationary `b`** | §7 proposes a decay **monitor** with **no corresponding estimator**. **The anti-ratchet named here was DELETED in Revision 2; this row cited it for a full revision afterwards.** The rate is unestimated, and on the verdict coins a weighted trend absorbs 99.3% of between-day heterogeneity (z=3.55) -- so this is the best-evidenced row in this table, not a low-priority one. | **Yes** | rolling `b̂` within day-clustered noise of its own mean |
| **shrinkage toward 1** | §12 step 5's gate is **binary** — adopt `b̂` or adopt `Identity`. A James–Stein-style `b̃ = 1 + λ(b̂ − 1)` is the natural third outcome and is *exactly* the right shape when the point estimate is above 1 but its interval contains 1 — **the current state.** | **Yes** | λ̂ ≈ 0 out-of-sample, i.e. shrinkage picks `Identity` anyway |
| **realised volatility** | The tick floor of §5.1 manufactures the favourite side; how hard it bites is a function of how far price moves per tick. | **Yes** | tick-floor bias flat in RV |

**None of these is a recommendation to build.** The honest position after
Revision 1 is that the *deployed* map is `Identity` and the *estimated* rotation
is below its own MDE — so adding conditioners to an effect that is not yet
distinguishable from zero would be fitting structure into noise, which is how
Revision 0's headline happened. > **⚠ Revision 3 — P1 AND §6.5's GATE ARE DELETED. This instruction pointed at
> machinery that no longer exists, and it is the ONLY routing decision this
> section makes — so the section written to hand a successor the open questions
> was handing them a dead pointer.**
>
> **The corrected order:**
> 1. **The verdict-coin per-coin split, which is ALREADY ON DISK AND UNREAD** —
>    btc `b̂ = 1.028`, eth `1.143`. The headline 1.083 pools one coin at
>    ≈`Identity` with one at 1.143, so this plan's own sentence *"pooling made the
>    estimator look more stable than it is"* applies verbatim at n = 2 and was not
>    applied. Costs nothing; the number exists.
> 2. **Session / time-of-day**, promoted from this table's bottom rows because it
>    is now a **live confound** for the `b̂_t` trend — UTC-block coverage runs
>    100%/38%/38%/29% against a monotone-increasing `b̂` — not a speculative
>    conditioner.
> 3. **The calendar trigger at `n_days ≥ 30`.**
>
> Conditioners after that, and only if a residual survives.

The shrinkage row is the exception worth flagging: it is the only entry that is
strictly better than what §12 step 5 does today **regardless of sample size**,
because a binary adopt/reject gate throws away the information that the point
estimate is consistently above 1 while its interval is not. That is a **design**
improvement, not a data-dependent one.

---
