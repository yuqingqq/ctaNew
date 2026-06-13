# Research Insights — iter-008 (NET-SHORT the classified alt-bear — HUMAN idea; NO-CANDIDATE)

**The human idea:** the −57% DD is a correlated ALT-BEAR where the mean-rev book buys "oversold"
alts that keep falling. iter-006/007 tried to FLAT that regime (rejected). This iteration tests the
extension: **classify the alt-bear and FLIP to NET-SHORT-BETA / short-momentum** in it — symmetric
to bull→long-momentum — to profit from the continued fall.

**Verdict: NO-CANDIDATE.** The analysis (computed, not speculated) shows the net-short bet fails the
three pre-registered preconditions decisively, and the reason is mechanistically clean and explains
WHY the strategy flats bear: **the classified alt-bear is NOT forward-separable from the bottom**, so
shorting it shorts into the bounce and gets squeezed. The G4 pre-check is the killer — classified
short ranks **p10–p11** vs matched random-timing short on BOTH HL70 and EXT (random-timing does
*better*), i.e. the classification carries *negative* edge for the short direction.

Script: `research/convexity_portable_2026-05-20/scripts/X123_altbear_short_probe.py` (reuses X122/X117
machinery verbatim: PIT alt-index `.shift(1)`, btc30, held-book engine, matched-placebo loop).
Per-cycle outputs: `results/X123_altbear_short_{HL70,EXT,S44}.parquet`.

The classifier is the SAME parameter-free alt-bear flag from iter-007 (`regime==side AND
alt_index_30d < btc_30d`, PIT, per-universe own alts). Arms tested on the held book (K=5, HOLD=6):
`base` (X117 production) · `flat` (iter-007) · **`shortmom`** (flagged side → net-short top-K by
mom30, no offsetting long) · **`shortbeta`** (flagged side → net-short top-K by trailing β) ·
`longmom_all` (ablation: long-momentum everywhere, out-of-scope reference).

---

## STEP 2.1 — Is the alt-bear FORWARD-classifiable as a PERSISTENT downtrend? **NO.**

A short-momentum bet needs the down-move to *continue* after classification. So I measured the
**forward** alt-index return on the flagged cycles (next HOLD=6 bars = the trade's own horizon, and
next 180 bars = 30d persistence). Mean-reversion/bottom = forward bounces ⇒ short gets squeezed.

| universe | flag fwd_HOLD mean / med / %neg | unflag-side fwd_HOLD mean / med / %neg | read |
|---|---|---|---|
| **HL70** | −0.0002 / **+0.0036** / 47% | −0.0052 / −0.0016 / 53% | flagged BOUNCES (53% up, less negative than unflagged) |
| **EXT** | −0.0022 / −0.0007 / **51%** | −0.0038 / −0.0006 / 51% | flagged %neg identical to unflagged — NO separation |
| **S44** | −0.0028 / −0.0004 / **51%** | −0.0004 / +0.0006 / 48% | flagged barely more negative; coin-flip 51% |

**The classifier does NOT identify a persistent downtrend.** At the trade horizon (next 6 bars) the
flagged alt-bear is a ~coin-flip on direction (47–51% negative) — on HL70 it actually bounces *more*
than the unflagged side cycles. The 30d-forward view is no better at the trade level: on EXT the
flagged 30d-fwd is −0.0005 (essentially flat) while the *unflagged* side cycles are −0.089 (the
genuinely-falling ones are the cycles the flag does NOT catch). The flag fires on the *coincident*
30d-trailing alt drawdown, but by the time it fires the forward move is a coin-flip — classic
"flagging the bottom, not the trend." This is exactly the squeeze risk the task asked to quantify,
and it is fatal to a short bet.

## STEP 2.2 — Does NET-SHORT PAY across MULTIPLE episodes (net of whipsaw)? **NO.**

**Headline by arm @4.5bps (G2/G7):**

| universe | base Calmar | base maxDD | **shortmom** Calmar / maxDD / totPnL | shortbeta Calmar |
|---|---|---|---|---|
| **HL70** (production) | +1.68 | −5,674 | **+0.24 / −8,350 / +2,196** | +0.78 |
| **EXT** (multi-episode) | +0.66 | −4,953 | **+0.11 / −7,911 / +4,038** | +0.30 |
| **S44** (transport) | +2.10 | −4,170 | **+0.81 / −6,381 / +15,033** | +1.28 |

**`shortmom` net-HURTS on ALL THREE universes** — Calmar collapses (1.68→0.24, 0.66→0.11, 2.10→0.81)
and maxDD gets materially **WORSE** (−5,674→−8,350 on HL70; −4,953→−7,911 on EXT). The net-short
introduces directional variance with no edge. It is worse than even the rejected iter-007 `flat` arm.

**EXT per-episode totPnL (bps) — the multi-episode test, with whipsaw:**

| episode | base | flat | **shortmom** | shortbeta | flagged fwd-alt mean | whipsaw (shortmom intra-DD) |
|---|---|---|---|---|---|---|
| 2022_luna | +756 | +1,086 | **+3,316** | +3,615 | −0.0435 (n=35, 80% neg) | −954 |
| 2022_ftx | −2,039 | −681 | **+227** | +1,721 | −0.0042 (51% neg) | −3,511 |
| 2024_summer | −267 | −1,928 | **−1,649** | −1,762 | −0.0027 (54% neg) | −2,929 |
| 2025_q4 | +4,834 | −565 | **−2,779** | +2,661 | −0.0048 (52% neg) | −4,039 |

The net-short payoff **flips sign per episode exactly tracking whether the flagged alts actually kept
falling forward**: it pays big only in **2022_luna** (the one episode where the flagged forward move
was genuinely −0.044 / 80% negative — a true persistent crash — but only n=35 flagged cycles), is
marginal in 2022_ftx, and **loses heavily in 2024_summer (−1,649) and 2025_q4 (−2,779)** — the
episodes where the flagged forward alt-move was near-flat (a bounce/chop), so the short got squeezed.
Critically, **2025_q4 is the SAME episode the iter-007 `flat` arm "won" on HL70** — here on EXT the
net-short version *loses −2,779 with a −4,039 intra-episode whipsaw*, because the alts bounced. The
whipsaw give-back is large in every episode except the genuine luna crash (−2,929 to −4,039 bps
intra-episode maxDD on shortmom). Net of whipsaw, the short does not pay.

**EXT episode-LOFO (G5):** shortmom full lift **−0.56**; dropping each episode the lift stays NEGATIVE
(−0.29 / −0.58 / −0.65 / −0.69) — it uniformly hurts, no single episode rescues it (the inverse of a
one-episode artifact: it is consistently bad). shortbeta same (full −0.36, all-episode-LOFO negative).
**EXT per-fold:** shortmom ≥ base in **3/8** folds (fails 6/9). **S44 per-fold: 2/8.**

## STEP 2.3 — vs FLAT (iter-007) and vs long-momentum-everywhere

- **vs FLAT:** net-short is *worse than* the already-rejected flat arm on every universe (HL70 Calmar
  0.24 < flat 4.73; EXT 0.11 < flat 0.25; S44 0.81 < flat 1.61). Adding the short direction on top of
  the no-edge flag actively destroys value rather than just running smaller.
- **`longmom_all`** (long-momentum everywhere, including the side regime) scores high in-sample
  (HL70 +5.87, EXT +1.25, S44 +2.01) — but that is a *different strategy* (abandons the mean-rev side
  alpha for trend-following everywhere), it is one-episode-fit on HL70 just like flat, and it is
  explicitly out of scope (the strategy's edge is cross-sectional mean-rev in side/bear per the X84/X94
  alpha-vs-beta decomposition). Not a candidate; logged only as an ablation reference.

## STEP 2.4 — G4 PRE-CHECK (the mandated, decisive test): **FAIL p10–p11 on BOTH**

Matched RANDOM-timing short: substitute the SAME net-short book at the SAME COUNT of *random* side
cycles (200 seeds, same construction/decay machinery). Does the CLASSIFICATION beat random-timing
short, or is it just directional variance?

| universe | arm | real(classified) Calmar / totPnL | placebo p50 / p95 | **rank** |
|---|---|---|---|---|
| **HL70** | shortmom | +0.24 / +2,196 | +0.49 / +0.95 | **p11** ✗ (tot p10) |
| **HL70** | shortbeta | +0.78 / +5,602 | +1.20 / +1.91 | **p10** ✗ (tot p28) |
| **EXT** | shortmom | +0.11 / +4,038 | +0.20 / +0.35 | **p10** ✗ (tot p10) |
| **EXT** | shortbeta | +0.30 / +12,926 | +0.37 / +0.57 | **p24** ✗ (tot p46) |

**The classified short is at the p10–p11 of a random-timing short on both universes — random timing
does BETTER than the signal-aligned classification.** This is stronger than the iter-007 flat-gate
G4 (which at least cleared p72 on HL70): here the alt-bear classification carries *negative*
information for the short direction. The "short the classified alt-bear" effect is not skill; it is
worse-than-random directional variance. The flag fires near the bottom, so a *random* side cycle is
on average a better moment to be net-short than a flagged one.

## STEP 2.5 — G8 cost
shortmom loses to base at every cost level on every universe (e.g. EXT @1bp Δ−0.83, S44 @1bp Δ−1.54),
so it cannot be rescued by a cost story — and unlike FLAT (which trades less ⇒ cheaper), the net-short
arm *adds* gross/turnover, so it is worse at high cost too.

---

## STEP 3 — pre-registration & honest decision

**Objective (pre-registered):** raise HL70 Calmar without it being a single-episode artifact;
multi-episode G5 (EXT) + G4 matched-random-timing-short pre-check mandatory; G7 HL70+EXT+S44.

**Pre-committed preconditions for proposing the net-short mode (ALL required):**
(a) alt-bear forward-classifiable as a PERSISTENT downtrend across ≥2 episodes — **FAIL** (forward
move is a coin-flip 47–51% neg at trade horizon; persistent only in 1 episode, 2022_luna n=35);
(b) net-short pays across multiple episodes net of whipsaw — **FAIL** (pays in 1/4 EXT episodes,
loses heavily in 2/4 with −3k to −4k whipsaw; hurts Calmar on all 3 universes; episode-LOFO −0.56);
(c) beats the random-timing short placebo (G4 ≥ p95) — **FAIL** (p10–p11 on both HL70 and EXT;
random-timing short does *better*).

**→ NO-CANDIDATE. No change proposed this iteration.** Champion stays = baseline (Calmar +1.68).

## Why this is the *expected* result, and what it teaches (the value of the negative)

This is the symmetric confirmation of iter-006/007 and explains the bear→FLAT design:

1. **The alt-bear flag is a coincident bottom-detector, not a forward trend-detector.** The 30d
   *trailing* alt drawdown that triggers the flag has, by the time it fires, a forward move that is a
   coin-flip — the genuinely-still-falling cycles (EXT unflagged 30d-fwd −0.089) are the ones the flag
   *misses*. So neither FLAT (iter-007) nor SHORT (here) can extract from it: FLAT just runs smaller on
   a coin-flip (p72 random), SHORT bets directionally into a coin-flip and gets squeezed (p10 random).
2. **This is precisely why the strategy correctly FLATs bear** (X88/X89): in a correlated alt
   deleverage there is no robust *forward* directional or cross-sectional edge — the move is already
   priced/coincident by the time free price/regime observables see it. iter-006 measured per-cycle IC
   predictability R²≈0.005 from regime features; the same unforecastability that kills the long mean-rev
   tail also kills any net-short bet. Shorting "what already fell" is shorting the bottom.
3. **The short side carrying alpha (vBTC: short correct 57%) does NOT transport here.** That finding is
   a *cross-sectional* short-leg effect (short the worst-ranked names within a beta-neutral book), not a
   *net-directional* short of the whole complex. Removing the offsetting long (net-short) strips the
   cross-sectional alpha and leaves naked beta exposure to a coin-flip move — which is why every
   net-short arm degrades. The existing side book already shorts the bottom-K cross-sectionally; making
   it *net* short adds no information and large directional variance.

**Converges with iter-001..007:** the −57% DD is one correlated-alt-bear episode per universe whose
forward path is not separable from a bottom on free observables. FLAT fails (p72/p0), SHORT fails
worse (p10/p10). The regime-axis / composition family is exhausted on free data. Remaining classes
(unchanged from iter-007): a fundamentally different alpha with real edge in correlated selloffs, or
paid *leading* deleverage/liquidation data — or accept the structural DD and live-monitor with a
kill-switch.

Artifacts:
- script: `research/convexity_portable_2026-05-20/scripts/X123_altbear_short_probe.py`
- per-cycle: `results/X123_altbear_short_{HL70,EXT,S44}.parquet`
- full console log reproduces all tables above (forward-classify, per-episode, whipsaw, LOFO,
  per-fold, G4 placebo, G8 cost).
