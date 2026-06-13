# Research Insights — iter-003

## Task
iter-001 (uniform vol throttle) and iter-002 (corr-timed side gate) both REJECTED for the same
reason: a matched-magnitude **random** control of equal flat/skip count does as well — the side
regime is irreducible near-symmetric zero-mean noise, so you cannot TIME it. iter-003 must attack
the side regime's **COMPOSITION / leg structure**, NOT timing, and **PRE-CHECK G4** on any
shrink/flat mechanism before proposing.

All numbers below are computed from the reused iter-002 held-book context parquet
(`research/convexity_portable_2026-05-20/results/iter002_hl70_context.parquet`, exact X116/X117
HL70 book: Sharpe +1.93, maxDD −5,674, Calmar +1.68). Probe script reused:
`scripts/iter002_leg_asym_probe.py`; new analysis run inline (logged below).

---

## Composition finding (data-driven, the core of this iteration)

### A. Leg-asymmetry is DEAD structurally — the side long leg is net-POSITIVE unconditionally
Per-regime leg attribution over the full sample (bps/cyc):

| regime | n | long | short | net | net-beta |
|---|---|---|---|---|---|
| bull | 426 | +8.95 | +16.00 | **+24.56** | +0.586 |
| side | 1,455 (60%) | **+2.93** | **−1.83** | **−0.01** | +0.006 |
| bear | 524 | +0.54 | −0.45 | +0.05 | ~0 (already FLAT) |

The toxic long-leg (iter-002 anatomy: in-DD long −6,142) is a **hi-corr / in-DD episode**, NOT a
structural property: across ALL side cycles the long leg is +2.93 and the short leg is −1.83.
So the candidate mechanisms from the brief fail structurally:
- **Shrink the long leg in ALL side cycles** (structural, not corr-conditioned): `side long*0.5`
  → Sharpe +1.47, Calmar +1.85, but bleeds PnL (+8,344 vs +10,472); `long*0.0` → Calmar +1.08
  (HURTS). You're cutting a net-positive leg.
- **Over-weight the short leg in ALL side cycles**: `side short*1.5` → Calmar +1.45 (HURTS) —
  the short leg is net-negative unconditionally.
- **Beta-cap the long leg**: MOOT. The existing beta-neutral sizing already drives **1,397/1,455
  (96%)** of side cycles to |net-beta|<0.05 (those net +0.2 bps/cyc); only 52 cycles carry
  residual +beta. corr(side net-beta, side PnL) = **+0.01** — no residual-beta channel to cap.
- **corr-conditioned long-leg cut** (iter-002's `iter002_leg_asym_probe.py`): `side hi-corr long*0`
  → Calmar +2.00 but **G4 side-pool placebo p70** (a *timing* claim → dead, per iter-002).

**Verdict: there is no structural leg-asymmetry to exploit.** The long-beta grind is a timing
phenomenon (hi-corr windows), and timing the side regime is the closed direction.

### B. The ONLY composition lever with in-sample bite is a uniform STRUCTURAL side de-weight
Because side nets ≈0 but carries the whole DD (iter-002 Finding 2), shrinking the *entire* side
sleeve cuts DD while keeping ~all PnL (which is 100% bull):

| rule (structural, ALL side cycles) | Sharpe | maxDD | DDcut | Calmar | totPnL |
|---|---|---|---|---|---|
| base (side*1.0) | +1.93 | −5,674 | — | +1.68 | +10,472 |
| **side*0.5** | +2.32 | −3,246 | +43% | **+2.94** | +10,480 |
| side*0.25 | +2.47 | −2,239 | +61% | +4.26 | +10,484 |
| **side*0.0 (side→FLAT)** | +2.52 | −2,239 | +61% | **+4.27** | +10,488 |

PnL is essentially **unchanged** at every fraction (side is net-zero in aggregate) while DD
collapses — the in-sample picture looks spectacular.

---

## G4 PRE-CHECK (mandatory, run BEFORE proposing)

Two placebos, both with the **held book re-weighted** (not row-zeroing), 1,000 seeds.

### G4a — matched-count placebo (de-weight n_side=1,455 *random ACTIVE (bull+side)* cycles)
This is the correct kill test for a "de-weight a regime" claim: does de-weighting THIS regime beat
de-weighting an equal number of *any* active cycles (which would also hit profitable bull cycles)?

| rule | real Calmar | placebo p50 / p95 | **Calmar pctile** | maxDD pctile | totPnL pctile |
|---|---|---|---|---|---|
| side*0.0 | +4.27 | +1.00 / +3.85 | **p97** | p51 | **p100** |
| side*0.5 | +2.94 | +1.61 / +2.27 | **p100** | p86 | **p100** |

**G4a PASSES** (both ≥p95 Calmar, p100 totPnL). This is the first composition lever in three
iterations to clear the matched-count placebo — because, unlike timing, it is NOT selecting *which*
side cycles; the placebo that flats random *active* cycles destroys PnL (median +2,456), whereas
the real rule removes precisely the net-zero regime and keeps the bull PnL.

### G4b — FORWARD / nested-OOS decidability (THE binding risk — also pre-checked)
The G4a win hinges on side being net-zero **in aggregate**. Per-fold it is NOT:

| fold | n_side | side_net (bps) |
|---|---|---|
| 0 | 57 | +103 | 
| 2 | 174 | **+740** |
| 4 | 205 | **−3,337** (the deep-DD episode) |
| 5 | 81 | −1,087 |
| 8 | 179 | **+2,761** |

So I pre-checked the **forward-decidable** versions:
- **Nested-OOS over fraction grid {0,0.25,0.5,0.75,1}** (pick max-Calmar fraction on past folds,
  apply forward): chosen fractions churn `[1,0,1,1,0,0,0,0]` → **nested-OOS Calmar +1.39 < base
  +1.68** (gives up +3,133 PnL in the profitable early side folds).
- **Binary {trade side / flat side}** chosen forward by past Calmar: picks `[trade, flat, trade,
  trade, flat, flat, flat, flat]` → **nested-OOS Calmar +1.39 < base +1.68**.

**G4b is the live failure mode**: the full-sample side→FLAT win is a *non-forward-decidable
aggregate-net-zero artifact* — it implicitly knows the catastrophic side fold (f4, 2025-Q4) cancels
the profitable side folds (f2, f8) over the whole sample, which a forward rule cannot exploit.

### Also ruled out (inert, like iter-001 on HL70)
**Structural regime risk-parity** (PIT de-lever side book to bull-book trailing PnL-vol, cap 1.0):
mean side scale = **1.00** → no change (Calmar +1.68, 0/9 folds). Side PnL-vol ≈ bull PnL-vol
(~150 bps); the side problem is zero-MEAN + long-grind, not high variance. (Confirms why iter-001's
variance throttle was inert on HL70.)

---

## Proposal (ONE change): structural `side → FLAT` regime rule

**The honest call.** Of every composition/leg-structure mechanism, only the structural uniform side
de-weight clears the matched-count G4a placebo (p97/p100). I propose the **parameter-free** end of
that family — extend the existing `bear → FLAT` rule to **`side → FLAT`** — because it is a
*structural regime-map redefinition* (no tuned fraction, no per-cycle signal), exactly the same
class of choice as K=5 and the existing bear-flat rule. Any fraction in (0,1) is a tuned parameter
that I have shown fails nested-OOS (G4b); the only non-tuned points are 0.0 and 1.0, and 1.0 is the
baseline.

**Spec.**
- In the held-book weight construction (`X116/X117` `cyc_w` loop), change the regime map:
  - `BTC-30d > +10%` → **bull** (mom30, unchanged)
  - `BTC-30d < −10%` → **bear → FLAT** (unchanged)
  - **else (side) → FLAT** (emit `{}`; previously: mean-rev pred with beta-neutral leg sizing)
- The held book then only ever holds bull-regime sleeves (overlapping HOLD=6 as today). No new
  feature, no model retrain, no threshold sweep. This is a one-line regime-branch change.

**Hypothesis.** Since 100% of HL70 PnL is the bull regime and side nets ≈0 while supplying the
entire −5,674 DD, never trading side cuts maxDD ~60% with ~0 PnL give-up → Calmar +1.68 → ~+4.3
in-sample.

**Why it is NOT a timing claim.** It does not select *which* side cycles to skip by any signal — it
removes an entire pre-existing regime bucket, parameter-free, like `bear→FLAT`. The G4a placebo
(de-weighting random *active* cycles) destroys PnL, so the skill is "which regime is net-zero,"
which is structural, not "which cycles will lose," which is timing (and dead).

**Pre-registered honest expectation — this is BORDERLINE-to-likely-REJECT on G4b/G3.** I am
flagging up front (per AGENT.md pre-check rule) that the forward-decidable versions land at Calmar
**+1.39 < baseline +1.68**. If Evaluation's honest forward/nested treatment confirms ≤ baseline,
this is a REJECT — and that is itself the decisive, loggable finding: **the side regime is net-zero
only in aggregate; its profitable sub-periods are real and not separable forward from its
catastrophic ones, so even a structural side-flat is not honestly adoptable.** Evaluation should
treat G4a (matched-count placebo, PASS p97/p100) and G4b (forward decidability, FAIL +1.39) as the
two halves of the kill test and decide accordingly.

---

## Pre-registered success criteria (against evaluation_contract.md)
Objective = raise **Calmar on HL70**.

- **G1 look-ahead (PASS required):** the change uses only the existing PIT regime label
  (`BTC trailing-30d return`, already in the baseline). No new feature, no retrain, no full-series
  quantile. Trivially PIT (it deletes a branch). Review should confirm the side branch emits `{}`
  and bull/bear are byte-identical to X117.
- **G2 in-sample:** HL70 Calmar > +1.68. Pre-registered in-sample target **Calmar +4.27, maxDD
  −2,239 (−61%), Sharpe +2.52, totPnL +10,488** (≈ unchanged). G2 will PASS in-sample — that is
  necessary, not sufficient.
- **G3 nested-OOS:** **WAIVED as structural** (side→FLAT is a parameter-free regime-map choice, no
  tuned threshold). BUT I pre-register the **forward-decidability check** as a mandatory substitute:
  the binary {trade-side / flat-side} decision chosen on past folds gives nested-OOS Calmar **+1.39**
  — Evaluation MUST report this. If the structural rule is judged "selected" rather than "structural,"
  G3 fails at +1.39.
- **G4 matched placebo (KILL TEST):** TWO framings, both pre-registered:
  - **G4a** (matched-count de-weight of random ACTIVE cycles, ≥500 seeds): real Calmar must rank
    ≥p95. Pre-check: **p97 Calmar / p100 totPnL → PASS.**
  - **G4b** (forward-decidability of the regime choice): nested binary trade/flat = **+1.39 < +1.68**
    → **this is the gate most likely to fail.** Report both; if G4b loses, REJECT.
- **G5 per-fold/LOFO:** DD improved in **6/9 folds** (pre-checked); but the aggregate PnL-neutrality
  is carried by f4 (−3,337) offsetting f2/f8 (+740/+2,761) → flag LOFO: dropping f8 (profitable side)
  should make side→FLAT look better, dropping f4 should make it look worse — report the LOFO swing.
- **G6 paired CI (block-boot by fold, 2000):** per-cycle PnL-diff (flat − base) mean +0.01 bps, CI
  **[−3.64, +3.98]** (crosses 0 — expected, removing a net-zero regime). For a pure-DD trade this is
  acceptable IF Calmar improves; but the **DD-diff CI is [−0, +8,919]** — lower bound touches zero,
  i.e. the DD benefit is not strictly significant under fold resampling (resamples overweighting f8
  shrink it). Report; borderline.
- **G7 universe:** must hold on HL70 (production). Also replicate the side-regime net-PnL
  decomposition on S44 (`x70_v0_3yr_preds.parquet`) and report whether side is net-zero there too;
  iter-002 already showed the side mechanism is partly HL70-composition-specific.
- **G8 cost:** pre-checked at 1 / 3 / 4.5 bps — Calmar improves at every cost (side*0.0:
  +4.33 / +4.29 / +4.27 vs base +1.99 / +1.81 / +1.68); FLATting side reduces turnover so the
  benefit holds at high cost. **PASS, not cost-dependent.**

## Look-ahead traps & failure modes
- **G4b is the real risk** (documented above): side is net-zero only *in aggregate*; the win is not
  forward-decidable. If Evaluation treats the regime-flat as a *selected* choice, it fails at +1.39.
- **f8 profitable side**: a single fold where side mean-rev genuinely works (+2,761) — side→FLAT
  gives this up. LOFO must show the lift isn't purely the f4 (deep-DD) fold.
- **No PIT trap on the rule itself** (it only deletes a branch using the existing regime label).

## Literature note
Not fetched — mechanism fully pinned from HL70 data. The relevant prior is the repo's own
`bear→FLAT` structural choice; this is its natural extension under the finding that side is also a
net-zero regime that only contributes drawdown.
