# Evaluation iter-003 — Structural `side → FLAT` regime-map change (X121)

**Verdict: REJECT** (fails G3 forward-decidability, G5 LOFO, G6 paired CI, G7 universe).
The in-sample Calmar lift (+1.68 → +4.72) is a **hindsight artifact carried by a single
catastrophic fold (f5)**, is not forward-decidable, and does not transport to S44.

Objective: raise Calmar on HL70. Champion = baseline (Calmar +1.68).

## The change
Parameter-free regime-branch deletion: extend the existing `bear → FLAT` rule to the
sideways regime. `side → FLAT` (was: mean-reversion `pred` with beta-neutral leg sizing).
Held book then only ever holds bull-regime sleeves. No new feature, no retrain, no threshold,
no per-cycle signal. Reproduced X121 from scratch (114s); **base arm matches X117 to the bp**
(Sharpe +1.93 / maxDD −5674 / Calmar +1.68 @4.5bps).

## Headline metrics (HL70 @4.5bps)
| arm | Sharpe | maxDD | Calmar | totPnL | %pos |
|---|---|---|---|---|---|
| base (X117) | +1.93 | −5,674 | +1.68 | +10,472 | 39.9 |
| flat_side | +2.75 (Δ+0.82) | −2,239 (−60.5%) | **+4.72 (Δ+3.04)** | +11,610 | 11.2 |

In-sample it looks like a large, near-free win (DD cut 60%, PnL ≈ flat — side nets ≈zero but
carries the whole drawdown). Every honest gate says otherwise.

## Gate-by-gate

### G1 look-ahead — PASS
Review PASS (handoff fix-round 1). Regime label is the existing PIT BTC-30d return bucketed
±10%, byte-identical to X117. No new info; deleting a branch cannot leak. Independently
confirmed base reproduces X117 exactly. No IC>+0.10 concern (no signal added).

### G2 in-sample objective — PASS (necessary only)
Calmar +1.68 → +4.72 > baseline. Necessary, not sufficient — every prior REJECT also passed G2.

### G3 nested-OOS / forward-decidability — **FAIL** (decisive)
The "which regimes to trade" choice is a selection, so G3 applies. Tested two nested binary
{trade-side / flat-side} rules deciding on past folds, applied forward:

- **Rule B (honest — decide flat when the *claimed mechanism* "side is net-zero" is established
  by past data, i.e. cumulative past side-PnL ≤ 0): nested-OOS Calmar +1.65 < base +1.68 → FAIL.**
  Matches research pre-registration (+1.39). Mechanism: cumulative side-PnL only goes negative
  *after* f5 (the disaster). So a forward rule trading on net-zero **sits through the f5
  catastrophe** (the only thing that makes side look net-zero), flats afterward, captures none of
  the protection, and gives up the genuinely profitable f8 side period.
  - Cumulative side-PnL walking forward: f2 +103 → f3 +99 → f4 +349 → **f5 −2,327** → f6 −1,955
    → f7 −3,289 → f8 −16. The "net-zero" verdict is only visible post-f5.
- **Rule A (Calmar-tiebreak on past folds): nested Calmar +4.67 — but this is a luck artifact, NOT
  a pass.** It flips to "flat" at f4 on a Calmar margin of +10.71 vs +10.54 (1.6%), driven by
  trivial DD noise on f2/f3 where side PnL is ≈0 (give-up only +124 bps). The coin-flip landed on
  "flat" right before f5 by coincidence. It is not forward-meaningful signal — the same
  coincidence LOFO exposes. We do **not** credit Rule A.

The proposal as-stated is "always flat from day 1," which is not a forward decision at all — it
is the full in-sample flat series, whose legitimacy is exactly what G5/G6 adjudicate.

### G4a matched-active placebo — PASS (but does NOT resolve hindsight)
Re-derived held-book PnL under 1,000 random matched-count (1,455) FLATs of **active base**
cycles (not row-zeroing; rebuilt cyc_w and re-ran the mask-agnostic `heldbook`):
- Calmar: placebo mean +1.42 / p95 +3.52 / max +6.60 → **real +4.72 ranks p99**.
- totPnL: placebo mean +2,218 / p95 +4,043 / max +5,362 → **real +11,610 ranks p100**.

Interpretation (per contract & research framing): G4a passing only shows side **is** the
net-zero regime in-sample (random FLATs of active cycles, which include profitable bull, bleed
PnL — so the skill is "which regime is net-zero," not "which cycles lose"). It does **not**
resolve the hindsight question; G3/G5/G7 do, and they fail.

### G5 per-fold + LOFO — **FAIL** (decisive)
Realized HL70 folds are 2–8 (cached preds carry no 0/1/9). flat_side Calmar ≥ base in **4/7
folds** (need 6/9). More importantly the **LOFO collapse**:

| drop | side-PnL | baseCal | flatCal | lift | Δlift vs full |
|---|---|---|---|---|---|
| −f2 | +103 | +0.84 | +4.30 | +3.47 | +0.43 |
| −f3 | −3 | +1.70 | +4.77 | +3.07 | +0.03 |
| −f4 | +249 | +1.61 | +4.62 | +3.01 | −0.03 |
| **−f5** | **−2,675** | **+6.57** | +5.71 | **−0.86** | **−3.90** |
| −f6 | +371 | +1.88 | +5.54 | +3.66 | +0.62 |
| −f7 | −1,334 | +2.17 | +5.50 | +3.33 | +0.29 |
| −f8 | **+3,273** | +0.96 | +4.24 | +3.29 | +0.25 |

Dropping the single worst (deep-DD) side fold **f5** collapses the lift from **+3.04 → −0.86**:
base's own worst DD vanishes (base Calmar jumps to +6.57) and flat_side LOSES. Every other fold
drop leaves the lift intact (+3.0…+3.7). **The entire in-sample Calmar win is one episode.** And
side→FLAT gives up the genuinely profitable f8 side period (+3,273 bps; flat Calmar 28.07 < base
45.44 there). This is a textbook G5 single-fold-carried FAIL.

### G6 paired CI (block-bootstrap by fold, 2000 resamples) — **FAIL**
- Sharpe-diff (flat − base): mean +0.84, **CI95 [−0.57, +2.44] — crosses 0.**
- PnL-diff: mean +1,161, **CI95 [−7,328, +9,389] — crosses 0.**
- DD-improvement (|baseDD|−|flatDD|): mean +2,993, **CI95 [−0, +9,368] — lower bound touches 0.**

Even the drawdown benefit is not strictly significant under fold resampling (one fold = f5 drives
it). No metric clears zero.

### G7 universe robustness — **FAIL** (decisive)
S44 @4.5bps: base Calmar +2.10 → flat_side **+2.09 (Δ−0.02)**, Sharpe +1.84 → **+1.45 (Δ−0.39)**,
totPnL +25,620 → **+16,942 (−34%)**, maxDD −33%. On S44 the side regime is **net-POSITIVE**
(f5 +5,005, f7 +5,043, f2 +1,981) — flatting it bleeds real PnL for a DD cosmetic only.
S44 LOFO full lift is already −0.02 and goes more negative dropping the profitable side folds
(−f7 → +0.47 lift only by removing S44's best side period). The improvement **does not hold on
HL70 as a structural property** — it's HL70-composition-specific (and within HL70, single-fold).

### G8 cost realism — PASS (not cost-dependent), but moot given the above
flat_side Calmar +4.78 / +4.75 / +4.72 @ 1 / 3 / 4.5 bps vs base +1.99 / +1.81 / +1.68.
The (in-sample, hindsight) edge holds at all costs — FLATting side cuts turnover. Robust to cost,
but the edge itself fails the honesty gates.

## Decision
**REJECT.** G2 and G4a look great in-sample; the decisive gates fail:
- **G5 LOFO**: lift +3.04 → −0.86 dropping f5 — the entire win is one catastrophic side fold.
- **G3 forward-decidability**: honest nested rule +1.65 < +1.68 — the net-zero is only knowable
  *after* the disaster, so a forward rule cannot exploit it.
- **G6**: all paired CIs cross zero.
- **G7**: does not transport to S44 (side is net-positive there; −0.39 Sharpe, −34% PnL).

"Always flat the side regime" removes the one episode that happened to blow up *in this sample*
while surrendering genuinely profitable side folds (f2/f8 on HL70, f2/f5/f7 on S44). It is a
hindsight artifact, not a permanent structural improvement.

## Insight for iter-004
- The BTC-side regime is now **closed at three layers**: timing it (iter-002, G4 p27), and
  removing it wholesale (iter-003, G5 LOFO + G7). The side regime's aggregate net-zero is an
  artifact of one deep-DD episode cancelling profitable periods — **not forward-decidable and
  not universe-portable.** Stop attacking the side regime as a block.
- The deep drawdown is *one fold* (f5, side-PnL −2,675), not a chronic property of side trading
  (other side folds are profitable). The unsolved problem is **detecting/de-risking that single
  catastrophic episode in real time** — which iter-001 (vol throttle, inert) and iter-002
  (corr gate, p27) both failed because the episode is not flagged by realized-vol or cross-corr.
  Next research should ask: *what distinguishes f5's side cycles ex-ante from f8's?* If nothing
  PIT-observable separates them, the drawdown is irreducible on this construction and the honest
  move is risk-budgeting (smaller gross / explicit kill-switch), not a regime rule.
- Champion stays = baseline (HL70 Calmar +1.68).
