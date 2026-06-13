# Research Insights — iter-002

## Task
iter-001 REJECTED a uniform book-vol throttle on HL70 (DD inert, +0.8%; the whole
vol-scaling family is now exhausted on production). The handoff insight: **HL70 and S44
have structurally different drawdown mechanisms**; the prior X98 DD anatomy was done on
44-sym. So: **redo the drawdown anatomy on HL70 specifically**, then propose ONE
mechanism-targeted change.

All numbers below are computed (not speculated). Scripts:
`research/convexity_portable_2026-05-20/scripts/iter002_*.py`. Context parquet:
`results/iter002_hl70_context.parquet`.

---

## HL70 drawdown anatomy (data-driven)

Reconstructed the exact HL70 regime-hybrid held book (K=5, HOLD=6, 6 sleeves, mom-bull /
mean-rev-side-BN / flat-bear, regime = BTC-30d ±10%, 4.5 bps). Full = Sharpe +1.93, maxDD
−5,674, Calmar +1.68 (matches baseline exactly). Then per-cycle leg attribution + PIT
context series.

### Finding 1 — the deep DD is a LONG-LEG, SIDE-regime event (not a vol-cluster)
Deepest DD episode: **peak 2025-09-30 → trough 2025-12-24 (85 days), −5,674 bps**, recovers
only 2026-05-04. Leg attribution inside vs outside the DD window:

| | long-leg PnL | short-leg PnL | net |
|---|---|---|---|
| in-DD (511 cyc) | **−6,142** | +828 | −5,624 |
| out (1,894 cyc) | +14,493 | +3,093 | +16,096 |

The drawdown is **entirely the long leg** (−12.0 bps/cyc in-DD vs +7.7 out). The short leg
is flat through the DD (+1.6 bps/cyc both in and out). This is NOT a "book P&L-vol runs hot"
episode — which is exactly why iter-001's variance-targeting throttle was inert on HL70.

### Finding 2 — the strategy's PnL is ALL bull; side regime is net-zero + carries the DD
Per-regime decomposition over the full sample:

| regime | n cyc | mean bps/cyc | Sharpe | total bps |
|---|---|---|---|---|
| **bull** | 426 | **+24.6** | **+6.03** | **+10,465** |
| side | 1,455 (60%) | −0.01 | −0.01 | **−16** |
| bear | 524 | +0.05 | +0.18 | +24 (already FLAT) |

**100% of HL70's +10,472 bps comes from the bull regime.** The side regime — the majority of
all cycles — nets ZERO and supplies the entire drawdown. Bear is correctly FLAT.

### Finding 3 — the discriminator of toxic vs benign side cycles is trailing CROSS-ASSET CORRELATION
The standout separator (in-DD vs out) is `corr7d` = average trailing-7d (42-cycle) pairwise
correlation of alt 4h log-returns (strictly trailing → PIT):

| context (in-DD vs out) | in-DD | out | sep (t-like) |
|---|---|---|---|
| **corr7d** (avg alt pair-corr) | +0.668 | +0.592 | **+18.4** |
| BTC 30d ret (b30) | −0.074 | +0.010 | −16.8 |
| net book beta | −0.001 | +0.137 | −17.6 |
| BTC rvol 7d | +0.0100 | +0.0081 | +16.9 |
| xs return dispersion | +0.0158 | +0.0146 | +3.0 |
| breadth (% up) | 0.481 | 0.484 | −0.2 |

`corr7d` is also **PIT-predictive** (lagged t−1, pure out-of-info-set, next-cycle book PnL):

| corr7d quintile (lagged) | q0 | q1 | q2 | q3 | q4 |
|---|---|---|---|---|---|
| mean bps/cyc | +13.8 | +8.4 | +0.7 | −0.4 | −0.6 |
| Sharpe | +4.82 | +3.45 | +0.27 | −0.31 | −0.36 |

Monotone gradient, no look-ahead (signal known strictly before the cycle).

### Finding 4 — the effect is regime-specific and lives in the LONG leg
Within **side** regime: hi-corr (top 30%) = −1.65 Sharpe vs lo-corr (bot 30%) = **+4.99**.
Within **bull**: hi-corr is GOOD (+6.37) — momentum works when alts trend together → a blanket
corr gate would wrongly kill good bull cycles. **The corr effect is side-regime-only.**
2D (btc_rvol × corr) confirms corr is the dominant axis: the corr-"hi" column is negative
across all rvol levels (−6.8 / −1.4 / −3.2 bps), and the rvol effect is largely confounded
with corr.

**Mechanism:** when alts co-move with BTC (high corr7d), the cross-sectional mean-reversion
spread collapses — the long leg is just "long the alts that fell with the market" = long beta
into a grind, with no idiosyncratic bounce. The short leg is unaffected. This is structurally
different from S44's DD (which the book-vol throttle bit on), confirming the handoff hypothesis.

### Finding 5 (negative) — model pred-spread does NOT know when it will lose
Within-cycle selected pred spread (top-K mean − bot-K mean) has **corr +0.001** with next-cycle
PnL (non-monotone quintiles). pred_spread ⊥ pnl. A selection-confidence gate is a dead end; the
information is in corr7d, not in the model's own dispersion. (corr(pred_spread, corr7d) = −0.30:
the spread compresses when corr rises, but that compression is not actionable.)

---

## Why a SIZING/TIMING overlay on corr7d FAILS the kill test (and what survives)
I built the corr signal as every sizing instrument and ran the **matched-eligible-pool
placebo** (the G4 kill test, drawing throttled cycles from the SAME side-regime pool — the
correct control, unlike iter-001's all-cycle pool):

| instrument (best config) | in-sample Calmar | maxDD cut | G4 (side-pool placebo) |
|---|---|---|---|
| blanket corr-throttle thr0.70 FLOOR0 | +2.61 | −30% | p94 (all-pool) / weaker side-pool |
| side-only corr-FLAT thr0.70 | **+2.75** | **−28%** | **p80** |
| side-only corr-FLAT thr0.80 | +1.93 | −11% | p51 |
| long-leg-only cut thr0.70 | +2.00 | −15% | p70 (worse: leaves net-short-beta book) |
| continuous size law λ=1.5 | +2.06 | −12% | p62 |
| side sub-book conditional-mean thr0.70 | kept-mean +1.83 bps (vs ≈0) | — | **p89** |

**None clears p95.** Reason: side regime is symmetric noise (mean ≈ 0), so random throttling
of *any* side cycles cuts DD nearly as well as corr-targeted throttling — the DD benefit is a
"run smaller in the noisy regime" magnitude effect, exactly the iter-001 trap. The corr split
of side **does** shift the kept sub-book mean from ≈0 to +1.83 bps/cyc (a real conditional-mean,
not just variance), and that ranks p89 — the strongest signal in the session — but still under
the kill-test bar.

---

## Proposal (ONE change): correlation-aware regime gate (structural 2D regime)

**Not** a continuous sizing overlay (that family is exhausted + fails G4). Instead, a **discrete
regime-state redefinition** — the same kind of structural choice as the existing ±10% BTC
threshold and `bear→FLAT` rule.

**Spec.** Add a second regime axis = PIT percentile rank of `corr7d`. Redefine the regime map:
- compute `corr7d_t` = mean trailing-7d (42-cycle) pairwise correlation of alt 4h log-returns,
  **strictly trailing** (window excludes the current cycle), then its **expanding PIT
  percentile rank** `pr_t` (rank of `corr7d_t` within its own history, warmup 100 cycles).
  Lag `pr_t` by 1 cycle for use (conservative; the held book overlaps HOLD=6 sleeves but the
  correlation window is already strictly trailing).
- New regime rule:
  - BTC-30d > +10% → **bull** (mom30, unchanged) — corr does NOT gate bull (Finding 4).
  - BTC-30d < −10% → **bear → FLAT** (unchanged).
  - else **side**: if `pr_t ≥ THR` → **side-grind → FLAT** (treat like bear); else → side
    mean-rev BN (unchanged).
- `THR` is the ONE tuned parameter → **nested-OOS** (G3): choose THR on past walk-forward folds
  from a small pre-registered grid {0.60, 0.70, 0.80}, apply forward; report nested-OOS Calmar.
  Default/structural fallback if nested-OOS is degenerate = 0.70 (the in-sample best).

**Where it plugs in.** In the held-book weight construction (X116/X117 `cyc_w` loop): when
regime=="side" AND `pr_t≥THR`, emit `{}` (empty book) for that cycle — identical to the bear
branch. No change to bull or to side leg sizing. No new feature in the model; `corr7d` is a
universe-level descriptor built from the same kline 4h returns already loaded.

**Hypothesis.** Converting the high-correlation half of the (net-zero, DD-bearing) side regime
to FLAT will cut HL70 maxDD by ≥20% and *raise* Calmar (Sharpe goes UP in-sample, +1.93→+2.45,
because we drop a zero-mean/high-vol sub-book) — because the deep DD is a side-regime long-beta
grind that occurs precisely when alts are co-moving.

**Why it may beat the placebo where sizing didn't (and why it might not).** The placebo
neutralizes "skip N random side cycles." This change is framed as a discrete regime *definition*
(structural, like K), and the binding question is whether the corr split carries a real
conditional-mean (it does: +1.83 vs ≈0 bps/cyc, p89 on the side sub-book) rather than just
variance. **Honest expectation: this is borderline on G4** — the strongest variant ranks p80
(whole-book Calmar) / p89 (side sub-book mean). If it lands < p95 it is a REJECT, and that is
itself the decisive finding that **the side regime is irreducible symmetric noise** — at which
point the correct next direction is to attack the *long leg's beta exposure in side* or to
**de-weight the side regime structurally** (e.g. trade only bull+a reduced side), not to time it.

---

## Pre-registered success criteria (against evaluation_contract.md)
Objective = raise **Calmar on HL70**.

- **G1 look-ahead (PASS required):** `corr7d` window strictly trailing (excludes current cycle);
  `pr_t` expanding PIT rank, lagged 1 cycle; no model retrain; regime threshold applied at
  decision time. No feature IC leakage (this is a regime descriptor, not a target-correlated feat).
- **G2 objective:** HL70 Calmar > +1.68. Pre-registered target: **Calmar ≥ +2.2, maxDD reduction
  ≥ 20% (≤ −4,540 bps), Sharpe ≥ +1.93 (must NOT regress; expected ↑).**
- **G3 nested-OOS (THR is tuned):** choose THR ∈ {0.60,0.70,0.80} on past folds, apply forward.
  Nested-OOS Calmar ≥ +1.68 (current_best). If nested-OOS Calmar < in-sample by a wide margin →
  REJECT (same fragility pattern as K3/decay-sleeve).
- **G4 matched placebo (KILL TEST, MANDATORY):** FLAT the same NUMBER of cycles drawn at random
  from the **side-regime eligible pool** (≥500 seeds); real HL70 Calmar must rank **≥ p95**.
  Pre-registered honest note: best variant currently ranks **p80** whole-book / **p89** side
  sub-book → **this is the gate most likely to fail**; report both framings.
- **G5 per-fold/LOFO:** improvement in ≥6/9 folds OR documented LOFO non-concentration (the DD
  lives in the 2025-Q4 fold — check the lift isn't 1-fold-only).
- **G6 paired CI:** block-bootstrap (by fold) the per-cycle PnL diff vs current_best; Calmar/maxDD
  benefit CI should not cross zero on the adverse side. (Sharpe-diff expected positive here,
  unlike iter-001.)
- **G7 universe:** must hold on HL70 (production). Also evaluate on S44 (`x70_v0_3yr_preds.parquet`)
  — replicate the side-regime corr decomposition there; report whether the mechanism generalizes.
- **G8 cost:** report at 1 / 3 / 4.5 bps. FLATting cycles REDUCES turnover, so the benefit should
  hold or widen at higher cost; must not depend on low cost.

## Look-ahead traps & failure modes to watch
- **PIT correlation:** the trailing-corr window must EXCLUDE the current cycle and the percentile
  rank must be expanding (no full-sample quantile). Implementation must not use `qcut` on the full
  series.
- **Sleeve overlap:** held book overlaps 6 sleeves; corr window is strictly trailing so no direct
  leak, but lag `pr_t` by 1 cycle to be safe.
- **G4 is the live risk** (documented above): a count-matched random side-FLAT does ~p80 as well.
  If real < p95, REJECT — do not rationalize.
- **THR overfitting:** in-sample best is 0.70; nested-OOS must confirm. A THR that only works at
  one value is the K3/decay failure mode.
- **G7 generalization:** if the corr mechanism is HL70-composition-specific (the universe is
  known to be overfit at three levels per MEMORY), it may not replicate on S44 — report honestly.

## Literature note
Not fetched — the mechanism is fully pinned from in-sample HL70 data (cross-asset correlation
collapsing the cross-sectional dispersion that long/short factor books harvest is a well-known
"correlation regime" effect in equity stat-arb; the novel, data-driven part here is that on HL70
it is (a) long-leg-only and (b) confined to the BTC-sideways regime). No paid feed needed —
`corr7d` is built from free kline 4h returns already in the pipeline.
