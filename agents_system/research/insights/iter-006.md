# Research Insights — iter-006 (ROOT-CAUSE DIAGNOSTIC)

**Question (human):** *why is the drawdown so large, what's the reason? how does it relate to our
features/alphas? then generate a possible solution.*

This is a diagnostic iteration, not a proposal-to-implement. All numbers computed, not speculated.
Script: `research/convexity_portable_2026-05-20/scripts/iter006_rootcause.py`. Output parquet:
`results/iter006_rootcause_HL70.parquet`. Engine reused verbatim from X116/X117 (K=5, HOLD=6,
mom-bull / mean-rev-side-BN / flat-bear, trailing-180-bar `.shift(1)` betas for side leg sizing).

---

## TL;DR

The −5,674 bps (−57%) HL70 maxDD is **one episode** (peak 2025-09-30 → trough 2025-12-24, 511 cycles,
folds 5–6). Decomposed to the feature/alpha level, the loss is **NOT a beta-hedge failure**. The beta
hedge actually *works*: the held book's realized net beta in-DD is −0.12 and the directional-beta P&L is
only **−473 bps (8% of the loss)**. **92% of the loss (−5,151 bps) is alpha-residual** — the
cross-sectional **mean-reversion `pred` alpha has no stable cross-sectional edge in the side regime**
(per-fold side XS-IC sign-flips every fold: −0.010 / +0.003 / +0.013 / +0.008 / −0.004 / −0.015 / −0.001,
mean ≈ +0.002), and its near-zero-mean per-cycle PnL has a **fat left tail that fires precisely when the
equal-weight ALT INDEX is in a hidden drawdown that the BTC-only regime gate cannot see** (in-DD: BTC-30d
−7.4% = "sideways", but alt-index 30d **−23.9%**). The "long-leg −6,142" headline from iter-002 is a
leg-attribution artifact: the long leg's −3,344 beta loss is **offset by the short leg** at book level.

**Ranked mechanisms by share of the −5,624 net in-DD loss:**

| rank | mechanism | share of loss | verdict |
|---|---|---|---|
| **1** | **H3/H1 mean-rev alpha is ~zero-edge noise with a fat left tail (book ALPHA-residual)** | **~92% (−5,151 bps)** | **DOMINANT** |
| 2 | H4 regime mislabel (BTC says "side" while alts in −24% bear) = the *condition* under which the tail fires | (the trigger, not separate $) | real but coincident-only forward |
| 3 | H2 residual net-beta (stale/under-hedge) | ~8% (−473 bps) | minor; hedge basically works |

---

## The deep-DD facts (reproduce baseline exactly)

Full = Sharpe +1.93, maxDD −5,674, Calmar +1.68 (matches X117). Deepest DD 2025-09-30→2025-12-24,
511 cycles. Leg attribution in-DD: long **−6,142**, short **+828**, net **−5,624**; of which side-regime
long −5,342 / short +813 / net −4,806 (263 side cycles). So the DD is a side-regime, long-leg-headline
event — consistent with the known finding.

---

## H2 — Stale-beta hedge failure: **REFUTED (explains only 8%)**

The side book sizes legs `a=2·β̄_S/(β̄_L+β̄_S)`, `b=2·β̄_L/(β̄_L+β̄_S)` using **trailing-180-bar
`.shift(1)` betas** — the obvious stale-beta suspect. I computed REALIZED betas (contemporaneous 7d
window, attribution-only) and decomposed.

| window | nbeta TRAILING (hedge believed) | nbeta REALIZED | long-leg β trailing | long-leg β realized |
|---|---|---|---|---|
| out-of-DD | +0.136 | +0.088 | +1.30 | +1.30 |
| in-DD | −0.001 | −0.059 | +0.98 | +0.95 |
| **in-DD side** | **+0.001** | **−0.117** | **+1.70** | **+1.64** |

- **Trailing betas are NOT meaningfully stale**: realized long-leg β 1.64 vs the 1.70 the hedge used
  (gap 0.06). Betas did *not* converge to ~1 and blindside the hedge.
- **The book is beta-neutral in realization** (net −0.12), so `Σ net_β_real·btc_ret` over the whole DD
  window = **−473 bps = 8% of the loss**. The hedge is doing its job.
- The long leg alone holds +1.64 β (it buys "oversold" = recently-fallen = high-β alts), but the **short
  leg holds +1.76 β** (implied) and **cancels it at book level**. The −3,403 "long-leg beta loss" is a
  decomposition mirage; the short leg's +β earns the offsetting +β P&L. H2 is not the cause.

> **Feature/construction note:** the beta-neutral leg sizing is sound — robust beta-neutralization
> (faster/shrunk/correlation-aware β, ex-ante optimizer, BTC overlay) would only address the 8% residual.
> A composition fix on the beta is therefore **not** the high-value lever. This rules out the H2-class
> solution the task flagged as a candidate.

## H1 — Dispersion collapse / correlation spike: **WEAK (the tail, not a collapse)**

| descriptor | in-DD | out | ratio |
|---|---|---|---|
| xs dispersion (target alpha_A) | 0.01576 | 0.01461 | **1.08** |
| xs dispersion (realized ret) | 0.01593 | 0.01484 | **1.07** |
| corr7d (avg alt pair-corr) | 0.668 | 0.592 | 1.13 |

Cross-sectional **dispersion did NOT collapse** — it was 7–8% *higher* in-DD (selloffs are volatile).
There was spread to harvest; the book just harvested it wrong. Correlation rose only 13% and
`corr(corr7d, side long_pnl) = −0.05` (≈0). So "alts all moved together → no spread" is not the mechanism;
this matches iter-002's G4 kill of the corr gate. H1 contributes the *environment* (high-corr selloff) but
is not a separable dollar driver and is not forecastable (iter-002 p27).

## H3 — Alpha sign / decay: **CONFIRMED as the dominant mechanism (with H1 as the trigger)**

- Side-regime XS-IC of `pred` vs forward `alpha_A`: **ALL +0.0024, in-DD −0.0039, out +0.0038**, and the
  **per-fold IC sign-flips every fold** (above). The mean-rev alpha has essentially **no stable
  cross-sectional edge** in side; it is a near-zero-mean noise process.
- Book-level alpha/beta split (net_β_real·btc_ret = beta; rest = alpha): in-DD net −5,624 = beta −473 (8%)
  **+ alpha −5,151 (92%)**; side sub-book net −4,806 = beta −560 (12%) + **alpha −4,246 (88%)**.
- The alpha-residual is a **fat left tail, not a sign inversion**: side in-DD alpha per-cycle mean −16.1 /
  **median only −5.8 / %neg 54%** vs out mean +3.6 / median −1.1 / %neg 51%. The hit-rate barely moves
  (54% vs 51%); the *magnitude* of the losing cycles blows out. Even fold 5 (the worst fold) had **+0.008
  IC** — the signal did not invert, it simply realized its left tail during the alt bear.

**Mechanism in words:** in side, the book ranks by `pred` (mean-reversion) and goes long the most-oversold
alts. Those are the recently-fallen, high-β names. When BTC is flat-ish but the **alt complex is in a
sustained −24% bleed** (Sep–Dec 2025), "oversold" alts keep falling — the mean-rev bet has no
idiosyncratic bounce to harvest because the move is a market-wide alt deleverage, not cross-sectional
mispricing. The near-zero-edge signal then realizes a long string of left-tail cycles. This is the
**central unsolved structural problem**: the side regime is the strategy's 60%-of-cycles, ~zero-alpha
majority whose left tail = the maxDD.

## H4 — Regime mislabel: **CONFIRMED as the trigger condition (but not forecastable forward)**

The BTC-30d gate labeled the DD window "sideways" (BTC −7.4%) while the **equal-weight alt index was
−23.9% over 30d** — a hidden alt bear the BTC-only gate is blind to. **207 / 263** in-DD side cycles have
BTC-sideways but alt-index-30d < −10%, and they carry **−5,211 of the −5,342** side long-leg loss. So the
loss condition is exactly "alts in a bear that BTC doesn't show."

This *is* the cleanest description of WHEN the tail fires — but it is **a coincident descriptor of this one
episode, not a forward separator** (see solution pre-check below). None of the V0 features capture it
because: regime = **BTC-only** trailing-30d (alt-breadth invisible); cohort feats (`rvol_7d`, `ret_3d`,
`btc_rvol_7d`) are vol/own-momentum, not an alt-complex-direction gate; and the target is per-symbol
`alpha_vs_btc` z-scored, which deliberately removes the market-direction the loss rides on.

---

## How it ties to FEATURES / ALPHAS (the broken assumptions)

1. **The mean-rev side alpha assumes cross-sectional mispricing that reverts.** In a correlated alt
   deleverage that assumption breaks: "oversold" = "high-β and still falling," so the alpha is ~0-edge
   and only its left tail shows up. *Feature gap:* nothing in V0 conditions the side leg on whether the
   move is idiosyncratic (mean-revertable) vs market-wide (not). The per-symbol z-scored alpha-residual
   target hides exactly this.
2. **The regime label is BTC-only.** The ±10% BTC-30d gate cannot see an alt-complex bear; it calls the
   most dangerous environment "sideways" and keeps the book on. *Feature gap:* no alt-breadth /
   alt-index-direction regime input.
3. **The beta hedge is fine** (refutes the obvious suspect) — so a beta-construction fix is low-value.

---

## PROPOSED SOLUTION (mechanism-grounded) + HONEST PRE-ASSESSMENT

**Grounded in the dominant cause (H3 fat-tail of a zero-edge side book, triggered by H4 hidden-alt-bear):**
a **2-axis regime gate** that flats the side book when the *alt complex* is in a drawdown the BTC gate
misses — i.e. **`regime=side AND alt_index_30d < THR → FLAT`** (alt-index = PIT trailing 30d cum log-ret of
the equal-weight universe, known at decision time). This is a *composition/regime-definition* change (like
the existing ±10% BTC rule and bear→FLAT), NOT a continuous sizing/vol overlay (that family is dead).

**Pre-registered gates:** objective = raise HL70 Calmar; G2 (in-sample), **G3 nested-OOS (THR tuned)**,
**G4 matched-random side-FLAT placebo ≥ p95**, **G5/LOFO**, G6 paired CI, G7 (S44), G8 (cost).

### HONEST pre-check (run now — this is why it is a DIAGNOSTIC, not an ADOPT candidate)

| test | result | read |
|---|---|---|
| G2 in-sample best (alt30<−0.10) | Calmar **3.91**, maxDD **−2,846 (−50%)**, Sharpe +2.44 | looks great |
| **G4 matched-random placebo** (500 seeds, FLAT 511 random side cyc) | real Calmar **p95** / maxDD **p96** | **right at the bar — the first DD overlay to even touch p95 (vs i1 p0, i2 p27)** |
| **G3 threshold sensitivity** | Calmar **spikes only at −0.10** (−0.08→2.86, −0.10→3.91, −0.12→2.50, −0.15→2.10) | **single-point optimum = over-fit; will fail nested-OOS** |
| **G5 / LOFO** (exclude the deep-DD window) | base 8.31 → gate **7.65, lift −0.67** | **entire win is the one episode — gate HURTS elsewhere** |
| PnL-by-quintile of alt30 (side) | **non-monotone** (q0 −1.6, q1 −4.3, q2 +12.1, q3 −10.9, q4 +4.6) | no usable gradient; alt30 flags benign cycles too |

**Verdict on the solution: it is NOT honestly adoptable as-is.** It clears G4 (p95/p96) — a genuinely
better showing than every prior DD attempt, because the alt-bear condition does carry a real conditional
left-tail — but it **fails G3 (single-point threshold) and G5/LOFO (lift −0.67 without the one episode)**,
the same one-episode/hindsight failure that killed iter-002 (fold-6) and iter-003 (LOFO −0.86). With only
ONE −57% episode in the 402-day OOS, any gate tuned to avoid it is fitting n=1.

**Why I am NOT claiming it as a win, and what would make it real:**
- It is *partly* a "run-smaller in the zero-mean side regime" effect (the iter-001/002 trap) — but unlike
  the corr gate it adds genuine conditional information (clears p95), so it is *not purely* run-smaller.
  The honest blocker is **sample size (n=1 episode), not mechanism**: the mechanism (alt-complex bear
  collapses side mean-rev to a fat left tail) is sound and feature-grounded.
- To become adoptable it needs the condition validated **out-of-the-one-episode**: (a) test the alt-bear
  gate on **S44 and on the longer 23-sym 2021–26 panel** (more alt-bear episodes: 2022 deleverage, 2024
  drawdowns) — does flatting side-during-alt-bear beat matched-random on a *different* episode? If it holds
  on ≥2 independent alt-bear episodes across universes, the n=1 objection dissolves and it can pass G3/G5.
  (b) Replace the tuned scalar THR with a **parameter-free structural form** (e.g. `alt_index_30d <
  BTC_30d − X` as a *relative* alt-underperformance flag, or alt-breadth < 50% — a discrete state like the
  ±10% rule) so G3 is waived rather than failed.

**This is a hypothesis for a FUTURE iteration**, pre-registered above, with the honest read that on the
current single-episode HL70 OOS it fails G3/G5 despite clearing G4 — and the decisive next test is
**multi-episode / multi-universe validation of the alt-bear regime axis**, which is the one thing the prior
5 iterations never had the data to do (HL70 OOS has exactly one big DD).

---

## Relation to the prior 5 (what this adds)

iter-002 found the corr7d separator but its gate died at G4 p27 (random side-flat did better). iter-006
goes one level deeper: the loss is **92% alpha-residual, not beta**, the mean-rev side alpha is **zero-edge
with a fat tail**, and the tail's trigger is a **BTC-invisible alt bear (H4)** — a condition the alt-index
gate captures well enough to *clear G4 (p95)* where corr7d could not (p27). The remaining wall is **n=1
episode** (G3 single-point / G5 LOFO), which only NEW DATA (more alt-bear episodes via longer/other
universes) can break — converging with the iter-004/005 conclusion that the free-data HL70 single-episode
structure is the binding constraint, now pinned to the *specific* missing feature: an **alt-complex-bear
regime axis**, not the per-symbol price/funding features.
