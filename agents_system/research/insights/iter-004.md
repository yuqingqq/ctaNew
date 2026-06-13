# Research insight — iter-004

**Directive:** two allowed paths. (A) realized-equity-DD circuit-breaker — propose ONLY if its
G4 pre-check passes (be skeptical). (B) PIVOT to Sharpe/alpha: improve the bull-regime momentum
sleeve, OR per-symbol timing with a BTC-beta hedge. Decide from the data.

**Outcome:** I ran G4 / LOFO / signal pre-checks on **all three** concrete options. Every one
fails honestly. I am **NOT proposing an adopt-candidate**; I propose the single remaining
*parameter-free* DD mechanism (Path A, the any-DD breaker) as a **confirmatory final DD test** and
recommend **escalation to human** (the orchestrator's stated trigger: "if iter-004 also rejects on
the DD axis, escalate"). Below is the full evidence — this iteration's value is closing three
directions cleanly.

---

## Setup
Reproduced the X117 held-book engine inline (base @4.5bps = Sharpe +1.93 / maxDD −5,674 /
Calmar +1.68 / totPnL +10,472 — byte-matches baseline). HL70 realized folds = f2…f8.
Scripts: `/tmp/iter004_analysis.py`, `iter004_pathA2.py`, `iter004_pathA3.py`,
`iter004_pathB.py`, `iter004_persym_fast.py` (kept in /tmp; will move to research/ on request).

## Regime anatomy (confirms prior findings)
| regime | cyc | totPnL bps | Sharpe | mean bps |
|---|---|---|---|---|
| bull | 426 | **+10,465** | **+6.03** | +24.6 |
| side | 1455 | −16 | −0.01 | −0.01 |
| bear | 524 | +24 | +0.18 | +0.05 |

100% of PnL is the bull regime; side nets ≈0; the −57% DD lives in the side grind (iter-002/003).

---

## PATH A — realized-equity-DD circuit-breaker. G4 cheap pre-check PASSES, deeper checks FAIL.

Rule: de-gross the NEW sleeve to a floor when the strategy's own trailing realized equity (PIT,
through t−1) is in drawdown beyond a threshold; recover at peak.

**G4 cheap pre-check (matched-COUNT random-timing de-gross to 0.5, 300 seeds):** real beats random
at **p95–p100 at every threshold** — the one path-A result that, taken alone, passes the gate:

| thr (bps) | triggers | real Calmar | placebo mean / p95 / max | real rank |
|---|---|---|---|---|
| 1000 | 54% | +2.37 | +1.63 / +1.97 / +2.54 | **p100** |
| 1500 | 52% | +2.39 | +1.63 / +1.99 / +2.11 | **p100** |
| 2000 | 50% | +2.19 | +1.64 / +1.95 / +2.41 | **p100** |
| 2500 | 47% | +2.08 | +1.64 / +1.95 / +2.37 | **p98** |
| 3000 | 45% | +1.93 | +1.63 / +1.92 / +2.19 | **p95** |

**But the directive said "be skeptical." The deeper checks expose the same f5 artifact as iter-003:**

1. **The winning variants require a TUNED threshold.** The genuinely *parameter-free* form
   (de-gross while underwater at ALL, fixed floor — the textbook "no new risk while underwater"
   rule) gives **Calmar +1.60 (floor 0.5) / +1.65 (floor 0.7) — BELOW base +1.68.** It fails G2.
   Underwater 95% of the time → it's just "run at 0.5× gross," which kills Sharpe (+1.67) and PnL
   (+5,099) for a DD cut that doesn't improve Calmar.

2. **LOFO collapse on the thr=1500 "winner" (the p100 variant):** full lift +0.71 →
   **−0.55 dropping f5 (Δ −1.26).** Every other fold drop leaves the lift ~+0.7. The entire win is
   de-grossing through the single f5 catastrophe — identical signature to iter-003's side→FLAT.

   | drop | lift vs base | Δ vs full |
   |---|---|---|
   | −f5 | **−0.55** | **−1.26** |
   | every other fold | +0.36…+0.86 | ~0 |

3. **The cheap G4 placebo is a WEAK control here.** De-grossing-while-underwater mechanically
   de-grosses *during* the f5 drawdown; a random-count de-gross rarely lands on f5 → real "beats"
   random by exploiting the same one-fold artifact the LOFO exposes. G4 passing ≠ forward skill
   (exactly the iter-003 G4a-passes / G5-LOFO-fails pattern).

4. **Nested-OOS (choose threshold/floor on past folds) reads +2.51 — but it locks onto the TUNED
   (0.5, 1500-bps) config after f4 and rides through f5 by luck.** This is a tuned-parameter
   nested selection, precisely the family (cost-margin swap, decay sleeves) that has repeatedly
   died here; not a structural pass.

**Path A verdict:** the only versions that beat base are tuned to de-gross through f5 (hindsight).
The parameter-free version is Calmar-neutral/negative. → **Do not propose as an adopt-candidate.**

---

## PATH B option 1 — improve the bull momentum sleeve (rank by `pred` instead of `mom30`). FAILS.

**Why it looked promising:** in bull cycles the cross-sectional rank-IC of the *momentum* signal
that currently picks legs is **zero** — `mom30` IC = −0.0016 (t=−0.17), `mom7` = −0.0094 (t=−1.05)
— while the model's `pred` (alpha-residual) has the only positive bull IC, **+0.0107 (t=+1.65)**.
So the construction ranks by a signal with no cross-sectional skill.

**But swapping to `pred` is CATASTROPHIC:** full Sharpe **−1.67** (bull-only −5.71), Calmar −0.67,
totPnL −7,143. A 50/50 rank-blend also hurts (Sharpe +0.48). LOFO of the swap is −2.35 lift,
again worst at f5.

**Mechanism (key insight):** the bull regime's +6.03 Sharpe is **pure long-beta capture, not
cross-sectional alpha.** Ranking by `mom30` puts high-beta trending names long / laggards short →
the basket carries a net long-beta tilt that profits as crypto rises. `pred` is beta-stripped by
construction (target = ret − β·BTC), so ranking by it destroys the beta engine and keeps only a
tiny (+0.01 IC) beta-neutral edge. **Bull "alpha" is not improvable by better stock-selection —
it is a beta bet, and momentum is already the correct lever to express it.** → REJECT.

## PATH B option 2 — per-symbol pred-TIMING sleeve WITH BTC-beta hedge. FAILS.

**Why it looked promising:** the per-symbol *time-series* rank-IC of `pred` (own pred → own fwd
return through time) is **+0.0116 mean / +0.0119 median, SE 0.0026 (t≈4.5), positive in 69% of
symbols (48/70)** — broad-based, not one-fold, not one-symbol. The most robust signal of the
session. (Per-symbol momentum trend-timing is strongly negative, −0.0262, 9% positive — so the XS
momentum sort is NOT a timing signal.)

**But constructing the actual sleeve (long top-10 pred / short bottom-10 pred each cycle, hold 6
sleeves) LOSES, even beta-hedged:**

| arm | Sharpe | maxDD | Calmar | totPnL | folds Sh>0 |
|---|---|---|---|---|---|
| raw (net-beta) | −2.08 | −8,660 | −0.74 | −6,996 | 2/7 |
| **beta-HEDGED** | **−1.97** | −8,189 | −0.74 | −6,631 | **2/7** |

Per-fold PnL: f2 −1,414, f3 −1,153, f4 −448, **f5 −4,104**, f6 +314, f7 −751, f8 +924.

**Mechanism:** an in-sample positive TS-IC does not monetize as a long/short *timing* P&L — the
sign is right on average but the magnitude/turnover and the loss in the same f5 regime swamp it.
The cross-sectional regime book was already the correct (and only profitable) way to use `pred`.
→ REJECT.

---

## Synthesis — DD axis is closed; alpha axis has no free lever on this data
- **Four consecutive DD attacks now fail the same way:** uniform throttle (i1, G4 p0), corr-timing
  (i2, G4 p27), structural side→FLAT (i3, LOFO −0.86), equity circuit-breaker (i4, LOFO −0.55 /
  parameter-free Calmar +1.60 < +1.68). **The −57% DD = one fold (f5), not chronic, not honestly
  reducible by any sizing/timing/regime/equity mechanism.** Every "win" is f5-hindsight; the cheap
  G4 placebo cannot distinguish it because de-grossing during a DD trivially overlaps f5.
- **Alpha axis:** bull PnL is beta, not improvable cross-sectional alpha; the one robust new signal
  (per-symbol pred TS-IC, t≈4.5) does not monetize as a tradable sleeve. Both Path-B levers REJECT.
- This reproduces the manual-research conclusion in MEMORY (43+ directions, free-data Binance/HL
  4h-horizon is a universe-overfit local optimum; DD is structural).

## Recommendation
**ESCALATE to human** (orchestrator's stated trigger met). Champion stays = baseline (Calmar +1.68).
If the framework requires a formal test this cycle, run the **parameter-free any-DD breaker
(floor=0.7)** as the confirmatory final DD probe — pre-registered to FAIL G2 (Calmar +1.65 < +1.68),
which cleanly documents that even the last authorized DD mechanism is non-additive, closing the DD
axis on the record. Do NOT run the tuned-threshold version (it would repeat iter-003's hindsight).

Future directions that need resources beyond this loop's scope (flagged, not proposed): annual
retrain on fresh data with proper winsorization (fix the target_A clip that breaks universe
expansion); orthogonal paid data (Glassnode/Deribit) — but session diagnostics put candidate
orthogonal signals below the value bar; operational forward-test (paper bot) to measure live f5-type
tail frequency rather than try to predict it.
