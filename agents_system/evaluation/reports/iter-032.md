# iter-032 — EVALUATION: expanded universe (70→156) — does breadth improve the deploy strategy honestly?

**Track:** deployment / universe-construction (decision-support with honest gates).
**Verdict:** **QUALIFIED ADOPT** of a WIDER deploy universe — *with a per-cycle minimum-history gate*.
The breadth=edge principle from iter-031 **CONFIRMS on the independent x132 V0 / 2021-26 panel**
(Sharpe & Calmar rise monotonically with N). BUT the honest gain over the prior 23-sym EXT comes
from (a) the model retrain on more data and (b) the *well-historied* breadth — **NOT** from dumping in
the 88 thin post-2024 names, which DILUTE. The single best honest config is the **history-gated wide
set** (drop names with <30d trailing history per cycle): +stop Sharpe **+1.19 / Calmar +1.33**, vs raw
full-156 +1.03/+1.16 and the 23-sym EXT +0.86/+0.74.

Engine: iter-031 deploy-universe engine VERBATIM (X117 held-book regime-hybrid + X125 iter-012 vol-norm
stop, k=2.0), re-implemented as a fast precompute layer (verified == slow engine to 1e-13). All numbers
@4.5bps with the iter-012 stop unless noted. Script:
`agents_system/research/scratch/iter032_expanded_universe.py`; artifacts `outputs/iter032/`.

---

## 1. Breadth-N sweep on the SAME x132 preds + SAME 2021-26 window (random subsets — clean A/B)

30 random liquidity-agnostic draws per N (full-156 deterministic). Only N varies — same model, same
period. **+stop (deploy) config:**

| N | base Sharpe | base Calmar | +stop Sharpe | +stop Calmar | +stop mDD |
|---|---|---|---|---|---|
| 23 | +0.46 | +0.44 | +0.38 | +0.41 | −3440 |
| 50 | +0.86 | +0.86 | +0.75 | +0.68 | −4002 |
| 100 | +1.02 | +1.28 | +0.91 | +1.16 | −3724 |
| 156 (full) | +1.12 | +2.02 | +1.03 | +1.16 | −3960 |

**Sharpe and Calmar rise monotonically with N** → iter-031's "breadth = edge" reproduces on a fully
independent V0 build over the multi-episode 2021-26 panel. Dispersion-driven, as expected. (Random-N
std also FALLS with N: 0.45→0.29→0.17→0 — wider = more stable composition, fewer catastrophic draws.)

---

## 2. Full-156 vs the 23-sym EXT baseline (same folds). The decomposition matters.

| config (+stop) | Sharpe | maxDD | Calmar | totPnL |
|---|---|---|---|---|
| EXT-23 (x113 preds — prior baseline) | +0.86 | −3000 | +0.74 | +10450 |
| EXT-23 SUBSET of x132 (retrain only, same 23 names) | **+1.06** | −2644 | **+1.08** | +13467 |
| FULL-156 (x132) | +1.03 | −3960 | +1.16 | +21611 |

Base-book (no stop): EXT-23-x113 Calmar +0.66 → full-156 **+2.02** (3×), totPnL +15448 → +40434.

**Decomposition (the honest read):** the headline lift over the prior x113-EXT (+0.86→+1.03 Sharpe,
+0.74→+1.16 Calmar) splits as:
- **Retrain alone** (same 23 names, x132 V0 model): +0.86 → **+1.06** Sharpe — i.e. **~all the Sharpe
  lift is the model retrain**, not the wider universe.
- **Adding 133 more names on top of the retrain:** +1.06 → +1.03 Sharpe — **marginally NEGATIVE** on
  Sharpe (full-156 sits slightly *below* the retrained EXT-23 subset), though it roughly DOUBLES totPnL
  (+13467 → +21611, more capacity / more independent bets) and the base-book Calmar is much higher
  (+2.02). maxDD is worse (−3960 vs −2644).

So expansion adds **capacity/PnL and base-book Calmar**, but the per-cycle risk-adjusted edge over the
retrained narrow set is **not** improved — and is not statistically distinguishable (G6 below).

---

## 3. Per-fold robustness + transport (+stop)

| fold | FULL-156 Sh | EXT-23-x132 Sh | EXT-23-x113 Sh |
|---|---|---|---|
| f1 | +1.23 | +1.11 | +0.92 |
| f2 | +0.38 | +0.23 | −0.05 |
| f3 | **−1.38** | −0.40 | −1.01 |
| f4 | **−0.10** | +1.24 | +1.25 |
| f5 | +3.48 | +1.21 | +1.06 |
| f6 | +1.12 | +1.09 | +1.10 |
| f7 | +0.46 | +0.90 | +0.60 |
| f8 | +1.16 | +2.57 | +2.37 |
| **folds_positive** | **6/8** | **7/8** | 6/8 |
| **AGG Sharpe** | +1.03 | +1.06 | +0.86 |
| **LOFO worst-drop** | f5 → **+0.65** | f8 → +0.87 | f8 → +0.69 |

**The wide book is LESS fold-robust, not more.** FULL-156's aggregate +1.03 leans on f5 (+3.48,
Calmar +10.95, +9765 bps — a single episode); dropping it collapses the book to +0.65. It also has TWO
losing folds (f3 −1.38, f4 −0.10) where the narrower retrained set is positive. The **EXT-23-x132**
subset is the most robust line: 7/8 folds, LOFO +0.87, no fold worse than −0.40. So expansion does NOT
buy cross-episode robustness; it concentrates more of the edge in the best fold.

---

## 4. THIN-HISTORY NOISE CHECK (the watch item) — DECISIVE

**4a. Where do the wide pred tails ([-34,+45]) come from?** median trailing-history bars: ALL rows
3059 vs extreme-|pred| rows 977 (extreme preds DO skew younger); 9.8% of extreme-|pred| rows are
thin (<180 4h-bars ≈ 30d) vs 3.5% baseline. But `corr(|pred|, is_thin) = +0.012` (weak). The single
biggest tail-generator is LITUSDT (full 2021 history, pred std 8.3) — a tiny-realized-vol name whose
per-symbol z-target blows up, **not** a thin name. So the wide tails are a mix, and mostly NOT thin-history.

**4b. Robustness of the wide-book result (+stop):**

| variant | Sharpe | maxDD | Calmar | totPnL |
|---|---|---|---|---|
| full-156 (no mod) | +1.03 | −3960 | +1.16 | +21611 |
| + winsor pred \|3\| | +1.03 | −3960 | +1.16 | +21611 |
| + winsor pred \|1.5\| | +1.03 | −3960 | +1.16 | +21611 |
| **+ min-history 180b (≈30d)** | **+1.19** | −3960 | **+1.33** | +24804 |
| + min-history 540b (≈90d) | +0.98 | −3129 | +1.20 | +17629 |

- **Winsorizing pred does NOTHING** (identical to the digit). The held-book selects top/bottom-K by
  *rank*; monotone clipping of the extreme preds leaves the ranks unchanged → the wide tails are a
  **red herring** for this construction. (Matters only if you ever weight by pred magnitude.)
- **A per-cycle min-history gate (drop names with <30d trailing data) IMPROVES the book** to +1.19 /
  Calmar +1.33 — i.e. the freshly-listed names are mildly NET-NEGATIVE noise. 90d over-prunes (loses breadth).

**4c. Is the breadth gain from the NEW thin names or the wider OLD set? (+stop)**

| set | Sharpe | Calmar | maxDD | totPnL |
|---|---|---|---|---|
| OLD-only (47 syms, ≤2023 listing) | **+1.15** | **+1.37** | −3461 | +22313 |
| FULL-156 (incl. 88 post-2024 names) | +1.03 | +1.16 | −3960 | +21611 |

**DECISIVE: the 47 well-historied names alone BEAT the full 156.** The breadth benefit is the *wider
old set*, not the thin tail. The 88 post-2024 names dilute Sharpe/Calmar and worsen maxDD while adding
almost no PnL. This is exactly the iter-031 risk ("watch item") materializing: **breadth helps only
among names with enough history to estimate the trailing-180 beta/mom and a stable per-symbol z-target.**

---

## 5. Honest gates (deploy = +stop)

- **G1 look-ahead — PASS** (inherited; X132 builder reuses X70 PIT pipeline, target_z `.shift(HORIZON)`,
  defensive z-clip ±50 only, NO clip-at-±5 target hack; engine uses realized-to-t−1 returns/equity).
  Full-sample IC +0.0146 is well below the +0.10 leak flag.
- **G2 in-sample objective — PASS vs prior baseline**: full-156 +stop Calmar +1.16 / Sharpe +1.03 >
  EXT-23-x113 +0.74 / +0.86. min-hist-180b is better still (+1.33 / +1.19).
- **G3 nested-OOS — N/A / WAIVED**: "trade the widest set" is a structural rule with no tuned parameter
  (iter-031 established; liquidity/IC ranking is proven value-negative). The min-history *floor* (30d) is
  a hygiene constant tied to the trailing-180-bar warmup, not a fitted knob.
- **G4 matched placebo — p69 (FAIL ≥p95)**: full-156 +1.03 ranks p69 of 100 random-100-name draws
  (random-100 mean +0.92, p95 +1.26 > full). Breadth beats *truncation* (>p50) but the full set is NOT
  a special composition — consistent with iter-031 (the specific name set is not the edge; breadth is).
- **G5 per-fold — 6/8 folds, FAIL the "more robust" bar**: lift over the narrow set is f5-concentrated
  (LOFO +0.65); two folds turn negative that the narrow set keeps positive. Expansion does not improve
  fold robustness.
- **G6 paired CI — CROSSES ZERO (FAIL)**: full-156 vs EXT-23-x132 per-cycle diff +0.79 bps, 95% block-
  bootstrap CI **[−0.34, +2.28]** → not statistically distinguishable from the retrained narrow set.
- **G7 universe transport — HOLDS**: breadth-monotone shape reproduces on the independent x132 panel and
  matches iter-031's HL70+EXT finding. Same sign, same mechanism.
- **G8 cost — robust**: full-156 +stop Sharpe +1.21 / +1.04 / +1.03 at 1 / 3 / 4.5 bps. Not low-cost-dependent.

---

## VERDICT

**Expanding to ~156 improves the headline deploy metrics vs the prior 23-sym EXT (Sharpe +0.86→+1.03,
Calmar +0.74→+1.16, totPnL ~2×), and the breadth=edge principle is re-confirmed honestly (monotone in
N, transports, cost-robust). This supports deploying a WIDER universe.** BUT three honest caveats sharpen
the recommendation into a *qualified* adopt:

1. **The Sharpe lift over x113-EXT is the MODEL RETRAIN, not the extra names** (retrained EXT-23 = +1.06,
   already above full-156 +1.03). Expansion's real adds are **capacity / totPnL (~2×) and base-book Calmar**.
2. **G4 p69 + G6 CI-crosses-zero + G5 6/8**: the full-156 composition is not statistically better than a
   good narrower retrained set, and it is LESS fold-robust (f5-concentrated). Breadth helps; *this exact
   width* is not special.
3. **Thin-history is a real dilutant**: the 88 post-2024 names lower Sharpe/Calmar (OLD-47 +1.15/+1.37 >
   full-156 +1.03/+1.16); a per-cycle **min-30d-history gate recovers +1.19 / Calmar +1.33** — the single
   best honest config. Pred-winsorizing is unnecessary (rank book is tail-invariant).

**Recommended deploy universe:** trade the **widest set of names that each clear a per-cycle minimum
trailing history (~30d / 180 4h-bars)** in addition to the iter-031 hygiene + execution-liquidity floor.
Do **not** dump in freshly-listed names before they have the history to estimate trailing beta/mom and a
stable z-target. This captures the robust breadth component (+1.19 Sharpe / +1.33 Calmar @4.5bps +stop)
and refuses the fragile thin-tail noise. Forward expectation regresses to the broad-based mean
(~+0.9–1.2 Sharpe on this harder 2021-26 multi-episode panel; the 2025-26-only HL70 reads higher per
iter-031). iter-012 vol-norm stop stays the always-on capital-preservation layer (transports, k unitless).

**Insight for next cycle:** the universe lever is now fully characterized. Breadth helps via dispersion,
but (i) it saturates / dilutes once you include sub-30d-history names, and (ii) it does not buy cross-fold
robustness — the deep-DD / f5-concentration nature of the strategy is unchanged by width. The wide tails
in the V0 z-target are harmless to the rank book but would matter to any magnitude-weighted variant — flag
for any future construction that sizes by |pred|. No further universe-width iterations have positive prior.
