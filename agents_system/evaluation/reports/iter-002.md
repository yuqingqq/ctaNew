# Evaluation Report — iter-002

**Change:** Correlation-aware sideways regime gate (`X120_corr_regime_gate.py`). In the BTC-sideways
regime, FLAT the held book when the expanding-percentile-rank of trailing 7d mean cross-asset
correlation (`corr7d`, PIT, 1-cycle-lagged) `pr_lag ≥ THR`. **THR is TUNED** ∈ {0.60, 0.70, 0.80},
structural fallback 0.70. Bull (mom30) and bear (FLAT) branches unchanged.

**Verdict: REJECT** — fails the kill test **G4** (HL70 Calmar **p27** / maxDD **p16**, need ≥p95),
**G3** nested-OOS (Calmar +1.79, only +0.11 over baseline and far below in-sample +2.02), **G5**
(lift carried by one fold), **G6** (paired CI crosses zero), and **G7** (S44 strongly negative).
G2 in-sample is the only thing that looks good — exactly the trap the contract exists to catch.

This REJECT is itself the decisive finding: **the BTC-sideways regime is near-symmetric, irreducible
noise** — randomly flatting the same number of side cycles cuts drawdown *as well or better* than the
correlation-timed gate. corr7d does not time the side-regime drawdown.

---

## Metrics table (@4.5 bps, HL70 = PRIMARY)

| universe | arm | Sharpe | maxDD (bps) | Calmar | totPnL (bps) | %pos |
|---|---|---|---|---|---|---|
| **HL70** base / current_best | base | +1.93 | −5,674 | **+1.68** | +10,472 | 39.9 |
| **HL70** gate@0.60 | in-sample | +2.16 | −4,948 | +2.01 | +10,917 | 30.4 |
| **HL70** gate@0.70 (fallback, best Calmar) | in-sample | +2.09 | −4,795 | **+2.02** | +10,639 | 33.0 |
| **HL70** gate@0.80 | in-sample | +2.05 | −5,213 | +1.86 | +10,621 | 35.3 |
| **HL70** | **nested-OOS THR** | +2.00 | −5,213 | **+1.79** | +10,256 | — |
| S44 base | base | +1.84 | −4,170 | +2.10 | +25,620 | 44.8 |
| S44 gate@0.70 | in-sample | +1.26 | −4,181 | +1.31 | +16,001 | 32.9 |
| S44 | **nested-OOS THR** | +1.18 | −4,181 | **+1.22** | +14,914 | — |

In-sample on HL70 every THR raises Calmar (best +2.02 at THR=0.70) and Sharpe (+2.05…+2.16), cutting DD
8–16%. **But the in-sample lift does not survive honest validation** (G3/G4/G6 below).

---

## Gate-by-gate (each with a number)

### G1 — Look-ahead audit: **PASS**
Review handoff PASS (fix-round 1, no required fixes). corr7d window `rt.iloc[lo:i]` strictly trailing
(excludes current cycle); `pr_t` is expanding (`mean(hist[:t] <= x_t)`, append after rank), no
full-series `qcut`/quantile; `pr_lag = pr.shift(1)`; gate reads only `pr_lag[t]`; held-book overlap
re-applies each sleeve's own `flat_mask[k]`, k≤t. No retrain. Base arm reproduces X117 byte-for-byte.
My `heldbook_mask` re-derivation matches the realized parquet series to 1e-18 and the gate masks
exactly (590/446/315 FLATs at THR 0.60/0.70/0.80). PASS.

### G2 — In-sample objective (Calmar on HL70): **PASS (necessary, not sufficient)**
HL70 Calmar +1.68 → **+2.02** at the best in-sample THR=0.70 (Sharpe +1.93→+2.09, maxDD −5,674→−4,795,
−15.5%). Necessary condition met. Note the DD reduction (15.5% at THR=0.70, 12.8% at THR=0.60's best DD
−4,948) is **short of the pre-registered ≥20% target**. THR=0.80 actually has the *worst* DD (−5,213) —
non-monotone in THR, an early sign the relationship is noisy.

### G3 — Nested-OOS (THR tuned → REQUIRED): **FAIL (decisive)**
Select THR by argmax in-sample Calmar over all strictly-earlier folds, apply forward, concatenate.
Chosen THR churns across folds (0.70→0.60→0.80→0.80→0.80→0.70→0.60 — no stable winner).
- **Nested-OOS HL70 Calmar +1.79**, Sharpe +2.00, maxDD −5,213, totPnL +10,256.
- vs base graded +1.68 / +1.93 / −5,674 / +10,472.

Nested Calmar beats baseline by only **+0.11** and collapses from the in-sample +2.02 — the classic
K3/decay-sleeve attenuation pattern (in-sample win evaporates under honest forward THR selection). It is
**far below the pre-registered ≥+2.2 target** and the maxDD reduction shrinks to **8.1%**. The THR that
wins in-sample does not generalize forward. S44 nested-OOS is outright negative (Calmar +1.22 vs base
+2.10). G3 fails to support ADOPT.

### G4 — Matched side-pool placebo (THE KILL TEST, re-derived held book): **FAIL — p27 / p16**
Per the Review's CRITICAL note, the placebo **re-derives the held-book PnL under each randomized FLAT
mask** (FLATting a side sleeve reduces turnover/gross for the next HOLD−1 HOLD cycles) — NOT row-zeroing.
500 seeds, each FLATting **446** randomly-chosen cycles from the `is_side` pool (matched to the real
THR=0.70 gate count), held book re-run with `heldbook_mask`.

| metric | real gate@0.70 | placebo mean | placebo p50 | placebo p95 | placebo max | **real percentile** |
|---|---|---|---|---|---|---|
| Calmar | +2.02 | +2.30 | +2.25 | +3.07 | +4.35 | **p27** |
| maxDD (bps) | −4,795 | −4,311 | −4,305 | −3,458 (best) | −2,711 | **p16** |
| Sharpe | +2.09 | +2.17 | — | +2.46 | — | **p33** |

**The real correlation-timed gate is WORSE than the average random side-FLAT** on Calmar, maxDD, and
Sharpe. A blindfolded control that flats 446 random side cycles delivers Calmar +2.30 / maxDD −4,311 on
average — the corr7d timing adds nothing; it sits at p27/p16, far below the p95 bar. This is exactly the
pre-registered kill (research handoff flagged best-case ~p80/p89 as "most likely to FAIL"; the actual
re-derived placebo is even more damning at p27/p16). **The DD reduction is a "flat fewer side cycles =
run less in a zero-net regime" gross-down artifact, not correlation timing skill.**

### G5 — Per-fold robustness + LOFO: **FAIL (one-fold artifact)**
HL70 @THR=0.70: DD improved in **3/7** folds (need ≥6/9), Sharpe improved 3/7. The entire benefit lives
in **fold 6** (−1,391→−359 DD, +74%, +981 bps; fold 6 = 2025-Q4 grind). LOFO confirms:
full-sample Calmar lift +0.34, but **dropping fold 6 flips the lift to −0.20** (gate HURTS without that
one fold). Two folds (4, 5) are actively negative (−995, −218 bps). The lift is a single-episode artifact.

### G6 — Paired CI (block-bootstrap by fold, 2000): **FAIL (crosses zero)**
HL70 per-cycle (gate − base) diff: mean **+0.069 bps**, 95% CI **[−1.201, +1.176]** — **crosses zero**.
No statistically significant per-cycle edge. (S44 CI [−2.865, −0.179] clears zero on the *negative* side
— gate significantly *worse* on S44.)

### G7 — Universe robustness: **FAIL**
Production (HL70): fails G3/G4/G5/G6 as above. Robustness (S44): the gate is **strongly negative** —
Calmar +2.10→+1.31 (in-sample), nested-OOS +1.22, Sharpe +1.84→+1.26, totPnL halved (+25,620→+16,001).
Every THR loses on S44. The corr mechanism is HL70-composition-specific AND fails its own gates on HL70,
so there is no universe on which it holds. (Even granting HL70-specificity for an HL70-derived anatomy,
the HL70 gates themselves fail — G7 cannot rescue it.)

### G8 — Cost realism: in-sample Calmar improves at every cost, but moot
HL70 base/gate@0.70 Calmar: 1bp +1.99/+2.32, 3bp +1.81/+2.16, 4.5bp +1.68/+2.02 — in-sample the gate
"wins" at all costs and the edge widens slightly at low cost. **But G8 is moot:** since the random-FLAT
placebo (G4) beats the gate at the same FLAT count, the apparent cost benefit is just lower turnover from
flatting side cycles — reproducible by random flatting. Cost is not the deciding gate here.

---

## Why REJECT
The in-sample numbers (G2 Calmar +2.02, DD −15.5%) look like a clean win, but **every honest gate fails**:
the kill test (G4) puts the correlation-timed gate at **p27 Calmar / p16 maxDD** — *below the average*
random side-FLAT of matched count — so corr7d carries no timing skill; nested-OOS (G3) collapses to
+1.79 (+0.11 over baseline, THR doesn't generalize); the lift is a single-fold (fold 6) artifact (G5,
LOFO flips negative); the paired CI crosses zero (G6); and S44 is strongly negative (G7). The objective
(raise HL70 Calmar with gates passing) is not met. Champion stays = baseline.

## Insights for next research cycle
1. **The BTC-sideways regime is irreducible, near-symmetric noise — you cannot *time* it, only avoid it
   wholesale.** This is the decisive, loggable result. G4 shows random side-FLATs (mean Calmar +2.30)
   beat the corr-timed gate (+2.02): the discriminating power the anatomy attributed to corr7d
   (in-DD vs out separation, q0→q4 monotone next-cycle Sharpe) was an **in-sample artifact of the side
   regime's symmetric zero-mean PnL** — any matched-count flat reduces the DD by trimming exposure in a
   zero-net regime. The "info is in corr7d" claim from the research handoff does not survive a re-derived
   matched placebo.
2. **The right structural move is to attack the side regime's net exposure, not its timing.** Since
   random flatting helps as much as corr-timed flatting, the honest equivalent is a **structural,
   parameter-free de-weight of the entire side regime** (e.g. uniformly halve side-regime sleeve gross,
   or skip a fixed fraction of side cycles with NO signal) — but note even that is just the gross-down
   trade the iter-001 lesson already flagged ("lower average gross is not alpha"). A structural side
   de-weight would need to beat its OWN matched random-flat placebo to be more than exposure trimming.
3. **The deep DD is a long-leg, long-beta side-regime grind (research anatomy: in-DD long −6,142 / short
   +828).** Since timing the side regime fails, the next attack should be **the long leg's beta inside
   side**, not when to be in side: e.g. tighten the beta-neutral leg sizing in side (the anatomy showed
   long-beta carries the DD), or asymmetrically shrink the long leg in side regardless of correlation.
   That is a construction-layer change on the *composition* of the side book, orthogonal to the
   (now-dead) correlation-timing axis.
4. **Two consecutive HL70 DD-reduction attempts (iter-001 vol throttle, iter-002 corr gate) have failed
   the same way:** the apparent DD benefit is a gross-down/exposure-trim effect that a matched-magnitude
   random control reproduces. The pattern strongly suggests HL70's −57% DD is **structural to running the
   strategy at all in the side regime**, not a timeable/sizeable sub-episode. Future risk overlays MUST
   pre-register and pass a matched-magnitude random placebo (G4 ≥ p95) before any in-sample number is
   considered — both rejects were predictable from G4 alone.
