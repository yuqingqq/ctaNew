# iter-021 — Cross-sectional funding-rate as a standalone/ensemble ALPHA (NOT a V0 feature) — NO-CANDIDATE

## Directive
Autonomous loop, lead with online SOTA. Steer AWAY from another DD-leading / construction-weighting
variant (those walls are comprehensively closed). Steer TO a genuinely-different alpha source,
reactive-risk refinement, or signal ensembling. ONE idea, pre-checked against the gates.

## Idea (online SOTA)
**Cross-sectional funding-rate carry / funding-momentum as an alpha signal ENSEMBLED with the V0
mean-reversion pred.** Funding rate is currently only a *V0 input feature* (`funding_rate`,
`funding_rate_z_7d`, `funding_rate_1d_change`) — it has NEVER been tested as a standalone
cross-sectional return predictor nor as an ensemble overlay on `pred`. The hypothesis: funding is a
realized, mechanically-orthogonal return driver (the perp's spot-anchoring cashflow + the crowding it
encodes), distinct from the price-residual mean-reversion `pred`, so blending it could raise the book's
cross-sectional IC/breadth at ~constant turnover (it re-ranks an already-traded book).

### Citations
- *Designing funding rates for perpetual futures in cryptocurrency markets*, arXiv:2506.08573 (2025) —
  cost-of-carry model; 170 return predictors incl. basis/funding; documents funding-fee yield as a
  return component and that "sustained extreme funding has preceded some of the sharpest reversals,"
  i.e. funding↔future-return is real but **regime-dependent with no universal sign/threshold.**
- *Unravelling cross-sectional patterns in cryptocurrencies: a four-factor asset-pricing model*,
  China Acct & Fin Review 27(4) (2024) — cross-sectional crypto factor evidence.
- Short-horizon reversal in perps amplified by leveraged-liquidation cascades (MDPI JRFM 14(5):103, 2025).

## Pre-check (decisive) — `iter021_funding_carry_precheck.py`, `iter021_verify.py`, `iter021_v2.py`, `iter021_ext.py`

### Step 1 — orthogonal cross-sectional IC on HL70 (production), looked GREAT
Merged `funding_rate` onto the cached HL70 V0 pred grid (match rate 1.0).
- **Non-overlapping 4h grid** (n=2405 cycles, no overlap-t-inflation):
  - IC(`funding_rate` → fwd return) = **+0.0133, t=+4.49**
  - IC(`funding_rate` → alpha-residual target) = **+0.0126, t=+4.26**
  - IC(production `pred` → alpha-residual) = **+0.0034, t=+1.31** (reference) — funding IC is ~4× the pred's.
- **Orthogonal to pred**: XS corr(`funding_rate`, `pred`) = **−0.021** (≈0). Genuinely different driver.
- **Ensemble lift is real & monotone** (z-rank blend `(1−w)·z_pred + w·z_fund` → IC vs alpha-residual):
  w=0 (+0.0034 t1.3) → 0.25 (+0.0096 t3.7) → 0.50 (+0.0129 t4.85) → 0.75 (**+0.0139 t5.00**) → 1.0 (+0.0126 t4.26).
  Blending funding lifts combined IC ~4× and t from 1.3→5.0 — the breadth gain the directive targets.
- **Per-fold sign-stable on HL70**: IC(funding→ret) POSITIVE in **7/7 folds** (+0.0016…+0.0289).
- Standalone funding-tilt book GROSS +1.75 bps/cyc, Sharpe +0.56 (re-rank of an existing book, so the
  ensemble's marginal cost is ~0 — turnover unchanged; this is NOT a new standalone-cost book).

### Step 2 — universe transport (G7), the KILL
Same test on the EXT 2021-26 panel (23 syms, different era/composition):
- IC(`funding_rate` → fwd return) = **−0.0109, t=−2.49**  ← **SIGN FLIPS**
- IC(`funding_rate` → alpha-residual) = **−0.0127, t=−2.93**
- Even restricting EXT to 2025 (overlaps HL70 era): IC = **−0.0129, t=−2.51** — still opposite sign to HL70's +0.013.

**The cross-sectional funding sign is NOT stable across universes: +0.013 on HL70 vs −0.011 on EXT.**
On HL70 (sustained-positive-funding 2025-26 bull) funding acts as MOMENTUM (crowded longs keep winning);
on the multi-era EXT panel it acts as carry-REVERSAL (high funding marks crowding that reverses) — exactly
the regime-dependence the arXiv:2506.08573 paper warns about ("no universal funding level signals a
reversal"). A fixed-sign funding tilt fitted on HL70 would actively HURT on EXT.

## Verdict — NO-CANDIDATE (pre-checked dead, no build needed)
Fails **G7 (universe robustness) before building** — the identical signature that killed mom_180d
(iter-015: +HL70 / −EXT), the alt-bear gate (iter-007), and the divergence-cut exit (iter-018):
a real in-sample HL70 IC whose **sign flips on another universe** is a regime-fit, not portable alpha.
The orthogonality (corr −0.02) and HL70 ensemble lift (IC 1.3→5.0 t) are genuine and tempting, but the
sign instability means the blend weight `w` and the funding *sign* are both HL70-2025-26 artifacts; an
honest nested/transport test would zero or invert the lift on EXT.

## Lesson for next iter
The funding↔forward-return relationship is **regime/era-conditional with no stable cross-sectional sign**
on free data — momentum in a one-sided-funding bull, reversal across full cycles. This closes
"funding-as-standalone-alpha" and "funding-pred ensemble" (the breadth lift is sign-unstable). It is the
same wall as price-momentum (i015): any signal whose CS sign is set by the prevailing funding/return
regime cannot be a fixed-weight overlay. Genuinely orthogonal *and sign-stable* free alpha remains
unfound. Remaining productive directions stay NON-free-data: (a) a regime-CONDITIONAL funding sign would
need a forward regime classifier — but iter-005/009 proved the funding/positioning regime itself is only
coincident, so this collapses to the closed DD-leading wall; (b) paid orthogonal flow data; (c) deploy
iter-012 stop + the K=5/6-sleeve book as-is. Champion UNCHANGED (baseline Calmar +1.68 + iter-012 stop).

## Scripts
- `/tmp/iter021_funding_carry_precheck.py` (overlap-grid IC + orthogonality)
- `/tmp/iter021_verify.py` (non-overlap 4h grid + per-fold HL70 sign)
- `/tmp/iter021_v2.py` (alpha-residual IC + ensemble-weight sweep + gross funding-tilt book)
- `/tmp/iter021_ext.py` (EXT universe transport — the sign-flip kill)
