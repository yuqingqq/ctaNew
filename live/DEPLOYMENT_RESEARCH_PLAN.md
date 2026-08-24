# Research plan — from marginal edge to tradeable book

Written 2026-08-07 after 18 iterations across three loops (COST_TURNOVER, SIGNAL_DIVERSITY, BEYOND_XS).
Zero adoptions. This plan is built on what those loops *established*, not on hope that another signal exists.

## What is settled (do not re-litigate)

| finding | evidence |
|---|---|
| Cross-sectional prediction on free price/vol data is exhausted | 8 signals with strong standalone IC, all with ~zero incremental value; the 14-feature model spans the information set |
| The target is already factor-neutral | mean pairwise corr +0.0046, n_eff 97/175; book R² on PC1-5 = 4.1%; K-factor residualisation fails the hard split |
| Cost is the binding constraint at 4h | 8.35 bps/unit taker; passive execution moves net +0.85 → ~+1.2 |
| Liquidity provision loses before fees | maker gross −0.6 bps; 0/31 symbols positive; no fee tier rescues it |
| The carry premium has been arbitraged away | median funding negative; max +5.4%/yr; Sharpe ~0.15 |
| Slow horizons are unfalsifiable here | 65 independent 30d blocks in 5.4 years |
| Two sleeves are real but thin | XS reversal (held-out net +0.85, spans 0); 14d skip-recent momentum (held-out IC +0.055 SIG, top-40 OOS spans 0) |
| The literature's own net frontier is ~1.2-2.3 | best properly-costed net Sharpe anywhere = 2.28, reported with no CI; crypto ~1.2-1.45 |

**The implication nobody has acted on:** we are at the frontier the literature describes, and every remaining
iteration of "find a better signal" has negative expected value. The unanswered question is not *is there an
edge* — it is *what is the best book constructible from two thin, real, uncorrelated edges under measured
costs, and what is its honest distribution of outcomes.*

## The plan

### Phase 1 — Consolidate (immediate, no new research)
Assemble the measured components into ONE specification and report its honest distribution, not a point
estimate. Components, each already validated in isolation:
- universe: PIT trailing-ADV top-40 (cost 8.35 → ~3 bps/unit; the ordering replicates across the hard split)
- construction: K+M hysteresis band (turnover 0.40 → 0.26 at equal gross)
- execution: passive/GTX (cost ~4 bps vs 8.35 taker; net +0.85 → ~+1.2)
- sleeve 1: per-symbol Ridge XS reversal, 4h
- sleeve 2: 14d skip-recent momentum (correlation to sleeve 1 measured at −0.05)
Deliverable: hard-split net Sharpe **with CI**, max drawdown, and the probability the true Sharpe is < 0.
This is the number that decides whether to trade at all — and it has never been computed for the *combination*.

### Phase 2 — Paper sweep on a DIFFERENT question
Every prior sweep asked "what predicts returns" and returned 1 survivor from 15. This one asks what the
deployment literature knows that we have not applied:
- sizing and risk under **parameter uncertainty** (the CI spans zero — that is the central fact)
- fractional Kelly / Bayesian shrinkage of expected returns; how much to bet on an edge you cannot confirm
- combining thin uncorrelated sleeves when each individually fails significance
- drawdown control that is not a fitted overlay (this repo has killed several fitted ones)
- what practitioners report that academics do not publish

### Phase 3 — The two mechanisms the last sweep flagged but never tested
1. **Characteristics as conditional factor LOADINGS, not predictors.** In the SOTA model the trading-frictions
   group costs 41% of net Sharpe when dropped *in that role*; we tested the same variables as predictors (null).
2. **Sleeve combination done properly** — the sleeve test used equal-risk weights fixed a priori and was
   dragged down by a sleeve (carry) now known to be dead. Rerun with only the two live sleeves.

### Phase 4 — Forward validation (the standing gap in every document)
The live loop has been down since 2026-07-10. The one dataset we do not own is **real fills**. Post `GTX`
orders in small size through the existing paper harness and record actual fill rates and markouts. This both
validates the passive-execution assumption (currently the weakest link — klines show a price *touched*, not
that we filled) and generates the fill-probability data the cost loop named as missing.

## Priority

Phase 1 first because it is cheap and decision-relevant. Phase 2 in parallel (agents). Phase 3 only if Phase 2
surfaces something. Phase 4 is the highest-value action overall and is operational, not research.

---

## Phase 1 result — **the first configuration in 19 iterations with P(Sharpe < 0) below 5%**
(`live/dp_phase1_consolidate.py`; held-out 2025-01→2026-06, 453 days, every parameter frozen on
2023-06→2024-12)

| execution | sleeve | Sharpe | 95% CI | maxDD | **P(Sh<0)** |
|---|---|---|---|---|---|
| taker | A: XS reversal 4h | +0.79 | [−0.81,+2.31] | −51.5% | 16.3% |
| taker | B: 14d skip-momentum | +1.02 | [−0.75,+2.74] | −44.2% | 14.2% |
| taker | **A+B equal-risk** | **+1.34** | [−0.45,+2.97] | **−25.8%** | **7.3%** |
| passive | A | +1.11 | [−0.43,+2.69] | −41.0% | 8.4% |
| passive | B | +1.04 | [−0.74,+2.75] | −44.1% | 13.4% |
| **passive** | **A+B equal-risk** | **+1.60** | **[−0.13,+3.24]** | **−25.7%** | **3.4%** |

**The combination is the lever, and it works through drawdown, not through the mean.** Sleeve correlation is
**−0.088**. Combining does raise the point estimate (+1.11 → +1.60), but the decisive changes are that maximum
drawdown nearly halves (−41%/−44% → −25.7%) and the probability the true Sharpe is negative falls from 8-13%
to **3.4%**. That is the number that should drive sizing, and no single-component test in this repo could have
produced it.

**Robustness**: fixed 50/50 weights give +1.55 [−0.20,+3.17], P(Sh<0) 4.0%, maxDD −24.0% — so the result is
not an artifact of using held-out volatilities in the risk weighting.

**Scale**: the combined book runs at **+62.4%/yr on 39.1% annualised vol** at unit gross. Scaled to a 10% vol
target (comparable to the benchmark being chased), that is **~16%/yr with ~6.6% maxDD**.

### Honest limits
- The CI still spans zero, if barely (−0.13 at passive). This is "probably positive", not "established".
- 453 daily observations ≈ 15 months. That is the whole held-out sample; it cannot be extended without waiting.
- The passive-execution assumption is the weakest link: klines show a price *touched* a level, not that we
  filled. Phase 4 exists to settle exactly this.
- Sleeve B is a documented sleeve (`docs/CONCLUSION_2026-08-03.md`), re-derived here from the horizon
  direction; it is thin and its top-40 OOS IC cell spans zero.

### What changed versus every prior conclusion
Nothing in this table is a new signal. Every component was already measured. What was never done was
**assembling them and measuring the joint outcome distribution**.

---

## Phase 1 REVIEW — the headline above is OVERSTATED. Corrected numbers below.
(`live/dp_phase1_review.py`) Reviewing my own result adversarially found four problems, three testable.

### R1 — inconsistent P&L accounting. **Material.**
Sleeve A was marked on `alpha_A`, the BTC-beta **residual** return, which silently assumes a beta hedge that
costs nothing to run. Sleeve B was marked on **raw** dollar-neutral returns. Combining them mixed two
accounting bases. Rebuilt on a consistent raw basis:

| sleeve A marked on | A alone | A+B combined | A maxDD | combined P(<0) |
|---|---|---|---|---|
| residual *(what Phase 1 reported)* | +1.11 | **+1.60** | −41.0% | **3.9%** |
| **raw (consistent, conservative)** | **+0.69** | **+1.26** | **−63.7%** | **9.4%** |

The residual marking was flattering sleeve A. The honest headline is **+1.26 with P(Sharpe<0) ≈ 9%**, not
+1.60 with 3.4%. Running the residual version legitimately requires an actual BTC hedge leg whose turnover
and cost are charged nowhere in this book.

### R2 — the passive cost was measured ON the holdout. **Not fatal.**
`maker_exec_probe.py` ran on 2025-01→2026-07, the same window Phase 1 evaluates, so the 4.2 bps figure leaks
holdout information. The whole grid, consistent (raw) accounting:

| cost bps/unit | 0.00 | 4.20 | 6.00 | 8.35 (taker) | 12.00 |
|---|---|---|---|---|---|
| combined Sharpe | +1.51 | +1.26 | +1.15 | +1.01 | +0.79 |
| P(Sh<0) | 5.0% | 8.1% | 11.2% | 13.9% | 21.4% |

It degrades smoothly and stays positive even at full taker cost, so the conclusion does not hinge on the
leaked assumption. Use +1.0 (taker) to +1.26 (passive) as the honest range.

### R3 — return concentration. **This is the serious one.**

| drop best N days | 0 | 1 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| Sharpe | +1.26 | +1.12 | +0.86 | +0.67 | **+0.24** |

**The best 5 days out of 453 carry 50% of total P&L.** And the skew is **−0.60**: the worst days are larger
than the best (worst five −10.2%, −8.0%, −7.7%, −7.7%, −7.2% against best five +8.1%, +8.0%, +5.5%, +4.8%,
+4.5%). An edge concentrated in ~1% of days *with the loss tail fatter than the gain tail* is not a robust
edge — it is closer to a short-volatility profile, and short-vol profiles look excellent right up until they
do not. This materially changes what the strategy is, not just its Sharpe.

### R4 — sleeve selection. **Not quantifiable, must be stated.**
The choice to combine these two sleeves was made after seeing which of five candidates survived. No in-sample
statistic corrects for that; only forward data does.

### Corrected bottom line
**Honest held-out picture: Sharpe +1.0 (taker) to +1.26 (passive), P(Sharpe<0) 8-14%, maxDD −26%, half the
P&L from 5 of 453 days, skew −0.60.** What survives the review intact is the **diversification benefit**:
combining halves drawdown (−63.7% → −26.1%) on both accounting bases, with measured sleeve correlation
−0.088/−0.057. That part is real and robust. The *level* is not established, and the return profile is
concentrated and negatively skewed.

**Implication for Phase 4:** a forward test is no longer merely the highest-value action, it is the only
instrument that can address R3 and R4 — concentration and selection are both invisible to any further
backtest on this sample.

---

## Phase 1 REVIEW 2 — checking what review 1 missed (`live/dp_phase1_review2.py`)

### S1 — the diversification claim REPLICATES. **This is the real finding.**

| window | A: XS reversal | B: 14d momentum | A+B | correlation | worst single maxDD → combined |
|---|---|---|---|---|---|
| SELECT 2023-06→2025-01 (549d) | **+2.21** | +0.45 | +1.85 | +0.036 | −57.6% → **−22.4%** (−61%) |
| HOLDOUT 2025-01→2026-07 (453d) | +0.69 | **+1.04** | +1.26 | −0.057 | −63.7% → **−26.1%** (−59%) |

**~60% drawdown reduction in both windows, with near-zero correlation in both.** That is a replicated
structural property, not a one-sample artifact.

**But the same table kills the return story.** The two sleeves **swap rank between windows**: A is +2.21 then
+0.69; B is +0.45 then +1.04. And the combination does *not* reliably beat the best single — it is worse than
A in SELECT (+1.85 vs +2.21) and better than B in HOLDOUT (+1.26 vs +1.04). Diversification raises the
Sharpe above the *average* sleeve, not above the *best* one.

**⇒ The correct statement of the value of combining: you cannot tell in advance which sleeve will work — they
swapped — and combining halves the drawdown while removing the need to make that bet.** That is genuinely
useful and it is replicated. It is not "the combination is the lever for returns", which is what I said.

### S2 — the dominant days are NOT one episode. **Partially rehabilitates R3.**
Top-10 days span 2025-01-27 → 2026-05-28 across **8 distinct months**, gaps of 11-92 days. Both sleeves
contribute on the biggest days (best day: rev +11.3%, mom +6.2%). So the concentration is real but it is
spread through the sample rather than a single event — a materially better profile than R3 alone implied,
though 50% of P&L from 5 of 453 days remains a short-volatility-shaped return stream.

### S3 — symbol concentration is LOW. **A positive finding.**
115 symbols traded; top-3 contributors are only **10% of gross absolute P&L**, and the big winners
(VVV +26.6%, ZEC +26.3%, IP +24.3%) are roughly offset by the big losers (VIRTUAL −31.0%, FARTCOIN −30.4%,
TAO −19.8%). The book is not carried by a handful of names.

### S4 — I misstated my own statistic.
I called the bootstrap fraction below zero "the probability the true Sharpe is negative". It is not — it
approximates the sampling distribution of the *estimator* under resampling, not a posterior over the
parameter. The correct frequentist reading is simply: **the 95% interval [−0.55,+2.98] contains zero, so the
null of no edge is not rejected.**

### S5 — the selection effect is larger than any of these corrections.
Nineteen iterations, 2-27 cells each: order **200+ configurations examined on substantially overlapping
data**. One surviving configuration at p≈0.05-0.10 is precisely what a search of that size produces under the
null. This cannot be corrected after the fact.

---

## FINAL POSITION after two reviews

**Established and replicated:**
- Two nearly-uncorrelated sleeves (|ρ| < 0.09 in both windows).
- Combining them cuts maximum drawdown ~60% in both windows.
- Which sleeve outperforms is **not predictable** — they swapped rank between windows.
- Symbol-level diversification is good; day-level concentration is high but spread over 8 months.

**Not established:**
- Any Sharpe level. Holdout +1.26 [−0.55,+2.98], interval contains zero.
- That the combination beats the best single sleeve.
- That the result is not a product of a 200-configuration search.

**Therefore the honest recommendation is unchanged and now better supported:** do not size on the point
estimate. The only defensible use of this work is to run the combined book forward, small, with the drawdown
property (the one replicated finding) as the reason to combine rather than pick.

---

# Phase 2 — sizing literature sweep (9 agents) + THIRD correction

## T0 — the drawdown claim is inflated by leverage. Verified independently.

The workflow's screen challenged my banked claim: the combined book is compared at *unit gross*, where it has
**lower volatility** than either sleeve (39.7% vs 76.7% / 46.4%), so of course it draws down less — it is a
smaller bet. I re-measured at **matched 10% annualised vol**:

| window | A rev | B mom | A+B | vs WORST single | vs BEST single |
|---|---|---|---|---|---|
| SELECT | −4.95% | −12.76% | −5.92% | −54% | **+20% WORSE** |
| HOLDOUT | −8.31% | −9.50% | −6.58% | −31% | −21% better |

So the challenge is **partly right**: the true benefit is 31-54% versus the worst sleeve, not the 59-61% I
reported, and against the *best* sleeve the combination is worse in one window and better in the other. The
benefit does not vanish under vol-matching, but it shrinks and it is one-sided.

**Third and final restatement of the only surviving claim:** combining does not reliably improve either
Sharpe or drawdown versus the *best* sleeve. What it reliably does — in both windows, on both metrics — is
land you between the two sleeves and remove the need to guess which one will work, **which you demonstrably
cannot do, since they swapped rank between windows.** That is insurance against selection, not alpha.

## The sizing answer (converges on what the plan already assumed)

- Full Kelly deployed vol **equals** the Sharpe: 126%/yr at S=1.26. A 10%/yr target is **7.9% of Kelly**.
- At that fraction `dlog g / dlog c = 0.96` — **cutting size cuts growth almost 1-for-1.** The "half-Kelly
  keeps 75% of growth" folklore holds only near c=1 and is false here.
- Four independent estimation-risk routes (Bayes posterior τ=0.5 and τ=1.0; Kan-Zhou; drawdown budget)
  converge on **8-12% annualised vol**.
- **Kan-Zhou at t=1.41 with N=2 sleeves gives a multiplier of 0.001** — estimating the sleeve *mix* consumes
  the entire position. The repo already honours this by accident (fixed 50/50 +1.55 ≈ inverse-vol +1.60).
  **Keep the weights fixed. Never fit them.**
- Strict frequentist and multiple-testing-adjusted answers are both **zero**. The only defensible frame is
  deployment at a scale where the null outcome is a rounding error — as a *measurement instrument*.

**Two results that matter more than the sizing number:**
1. **Time does not help.** P(loss) falls only 43.4% (1y) → 40.7% (3y). The uncertainty is in the *drift*, so
   holding longer does not resolve it. "Run it three years and see" buys almost nothing.
2. **Cost of being wrong:** under the null at 10% vol, 3 years costs a median −3.7% (1-in-100: −35%). At unit
   gross it costs a median −27%, 1-in-100 −85%, with P(DD>50%) = 70%. The deployment case rests entirely on
   the first row being tolerable.

## How long a forward test must run — and why Phase 4 is still worth starting

SE(annualised Sharpe) ≈ 1/√T_years (measured 0.890 vs predicted 0.898 — verified on the real series).

| true S | 80% power to reject H0 |
|---|---|
| 1.26 (point estimate) | **3.9 years** |
| 1.01 (taker) | 6.1 years |
| 0.63 (half) | **15.6 years** |

Pooled with the existing holdout, 2 forward years reaches t=2.27 — *only if* the forward Sharpe arrives at the
full +1.26, which §4 says it will not. At the plausible +0.6-0.9, **no feasible horizon resolves it.**

**⇒ Phase 4 is worth starting, but not for the reason I gave.** Do not start it to settle the Sharpe — that is
unattainable. Start it to measure **fills and cost**, which resolves in ~8 weeks with a tight CI and retires
~0.25 Sharpe units of assumption risk, and for implementation integrity. Write that reason down so nobody
later reads the forward P&L as evidence about the edge.

## The base rate — this dominates everything above

| source | n | finding |
|---|---|---|
| QuantConnect | 355 strategies | IS Sharpe 1.574 → OOS **1.049 (−33%)** |
| **Quantopian live** | **888 live algos** | **IS→OOS Sharpe R² = 0.02.** Vol R²=0.67, maxDD R²=0.34 |
| McLean & Pontiff | 97 predictors | −26% OOS, −58% post-publication |
| Harvey & Liu | 316 factors | t=3.0 → 2.0; marginal Sharpes penalised worst |
| **our own** | — | SELECT +1.845 → HOLDOUT +1.259 = **−32%** |

**The Quantopian result is the single most important number in this document: backtest Sharpe explains 2% of
live Sharpe variance. Risk transfers (vol R²=0.67, drawdown R²=0.34); returns do not.** You can forecast what
this book will *risk*, not what it will *earn*.

And the multiple-testing arithmetic settles the rest: under H0 on 453 days, the expected *maximum* Sharpe from
a search of just 10 independent configurations is **+1.38** — above our observed +1.26. We ran ~200.

**Honest live forecast: Sharpe +0.3 to +0.9, with 15-25% posterior mass at or below zero.** At S=+0.6 and 10%
vol: ~+5%/yr, expected 3y max drawdown 14%, and a 19% chance of being underwater after three years.

# FINAL POSITION

1. **Size at 10%/yr vol with fixed 50/50 sleeve weights.** Never fit the mix (Kan-Zhou multiplier 0.001).
2. **Expect Sharpe +0.3 to +0.9, not +1.26.** 15-25% chance the true edge is ≤ 0.
3. **Start Phase 4 for fill/cost measurement (~8 weeks), explicitly NOT to validate the edge.** Record that.
4. **Combine for insurance against sleeve selection, not for return.** The sleeves swap rank; you cannot pick.
5. **Do not run further backtest iterations on this data.** At ~200 configurations examined, additional search
   has negative expected value — it can only manufacture a better-looking number, not a better strategy.
