# iter-027 — Phase-2 feature-engineering + pooled-GBM model rebuild — NO-CANDIDATE

First Phase-2 (broadened-scope) research iter: baseline NO longer fixed, feature
engineering + model change + structure rebuild all allowed. Mandate: build a
GENUINELY BETTER ALPHA, transport-first.

## Online SOTA surveyed (May 2026)
- **Order flow / microstructure** (Anastasopoulos & Gradojevic, EFMA 2025 "Order Flow and
  Cryptocurrency Returns"; AEA-2024 "Realized Illiquidity" = realized-Amihud): signed taker
  volume / order-flow imbalance is the most-cited contemporaneous-and-leading microstructure
  return driver; realized Amihud = RV / dollar-volume.
- **Empirical asset pricing via ML** (Gu, Kelly & Xiu 2020, RFS 33(5):2223): pooled gradient-
  boosted trees beat linear OLS in the cross-section by capturing **nonlinear interactions**;
  predictive slope ≈1 vs OLS much smaller. The strongest theoretical case for replacing the
  per-sym Ridge V0 with a pooled LightGBM (+ sym_id).
- **Cross-crypto lead-lag** (J.Econ.Dyn.Control 2024 S0165188924000551; arXiv:2205.00974): BTC
  leads alts; lagged cross-returns predict focal returns.
- **Multi-TF momentum/reversal** (Dobrynskaya SSRN 3913263; FRL S1544612320303135): positive
  momentum ≤2-4 wk, reversal >1mo; short-horizon daily reversal cross-sectionally strong
  (illiquidity-driven). NB: this is iter-022's `rel_ret_1d` family.

Picked: pooled LightGBM (model) + new features {order-flow imbalance, OI dynamics, realized
vol-of-vol, multi-TF momentum/reversal, Amihud}. Tested transport-first.

## STEP-3 transport + marginal-IC table (4h grid, XS Spearman IC vs fwd alpha-residual)
Script: `iter027_feature_transport_precheck.py`. HL70 (71 sym, 2024-12→2026-05, ~2,450 cyc),
EXT (23 sym, 2021-01→2026-05, ~11,560 cyc). PRED-RESID = IC on V0-pred-residualized fwd alpha_A (HL70).

| feature        | HL70 IC | EXT IC  | sign  | PRED-RESID (HL70) | note |
|----------------|---------|---------|-------|-------------------|------|
| rev_2 (8h rev) | +0.0349 | +0.0350 | same  | +0.0359           | = iter-022 reversal |
| rev_6 (1d rev) | +0.0365 | +0.0308 | same  | +0.0355           | corr −0.65 w/ rel_ret_1d (iter-022) |
| mom_6 (1d)     | −0.0365 | −0.0308 | same  | −0.0355           | = −rev_6 |
| mom_18 (3d)    | −0.0229 | −0.0275 | same  | −0.0222           | reversal at 3d |
| mom_42 (7d)    | −0.0160 | −0.0227 | same  | −0.0140           | reversal at 7d |
| **vov** (vol-of-vol) | −0.0311 | −0.0224 | same | −0.0273      | **NEW, transports** |
| vov_n (norm)   | +0.0017 | +0.0004 | same  | +0.0023           | dead |
| amihud_z       | −0.0006 | +0.0020 | **FLIP** | −0.0002        | dead + sign-flip |
| taker_1d (OFI) | −0.0040 | −0.0053 | same  | −0.0052           | tiny, ~noise |
| taker_z7d      | +0.0017 | −0.0043 | **FLIP** | +0.0005        | sign-flip (= iter-009 wall) |
| oi_chg_1d      | −0.0014 | −0.0015 | same  | −0.0019           | tiny, ~noise |
| **pred (V0)**  | +0.0056 |   —     |  —    |  —                | production predictor IC on alpha_A |

Readings:
1. **The microstructure families (order-flow/taker, OI, Amihud) are DEAD**: IC ≈ 0 (|IC|≤0.005)
   and/or sign-FLIP across universes. Reproduces iter-009 (positioning coincident) one layer earlier
   — order-flow at the 4h XS layer carries no transport-stable alpha-residual signal.
2. The only LARGE transport-stable IC is the **short-horizon reversal (rev_2/rev_6/mom_6, ±0.035)** —
   but `rev_6` is corr **−0.65** with iter-022's `rel_ret_1d`: it is the SAME signal that already
   died at the PnL/construction layer (iter-022 REJECT). Its big PRED-RESID IC is the exact iter-022
   trap (orthogonal-to-pred-IC ≠ tradeable).
3. **vov (vol-of-vol)** is the one genuinely NEW transport-stable family (−0.031/−0.022, pred-resid −0.027).

## STEP-3b decisive construction-layer marginal (the iter-022/023 killer)
Script `iter027_construction_marginal.py`: within the pred-conditioned top/bottom-K_POOL=15,
tilt final K=5 by the signal vs 200 matched-RANDOM picks from the SAME pool (need ≥p95).

| signal     | real L-S | rand mean | rank | verdict |
|------------|----------|-----------|------|---------|
| rev_6      | −0.00011 | +0.00001  | p26  | fail (reproduces iter-022) |
| **vov**    | −0.00034 | −0.00002  | **p2** | **fail (tilt HURTS vs random)** |
| vov_resid (vov ⟂ rev_6,mom_6) | −0.00025 | −0.00004 | p14 | fail |

The pred-pool already extracts the XS info; tilting by ANY of these within the pool ≤ random.
4th independent transport-stable signal (after rel_ret_1d/funding/MAX) to die here. Wall #3 holds
even for the new vov family.

## STEP-3c decisive MODEL test — pooled LightGBM vs per-sym Ridge (walk-forward OOS XS-IC)
Script `iter027_pooled_gbm_vs_ridge.py` (+ shallow/deep config sweep). Target = target_z (per-sym
z of 4h alpha-residual), expanding 9-fold, 1d embargo. Features = BASE(13)+cohort+sym_id (+new feats).

| model               | EXT OOS XS-IC | HL70 OOS XS-IC |
|---------------------|---------------|----------------|
| Ridge (V0 baseline) | −0.0074 (t−3.1) | +0.0047 (t+1.5) |
| GBM (base feats)    | −0.0020 (t−0.8) | +0.0013 (t+0.4) |
| GBM (+new feats)    | −0.0037 (t−1.5) | −0.0029 (t−0.9) |
| GBM shallow/heavy-reg | **−0.0142 (t−5.6)** | **−0.0042 (t−1.2)** |

DECISIVE FINDINGS:
- **The raw 4h alpha-residual is near-unpredictable cross-sectionally**: every model's OOS XS-IC is
  |IC| < 0.015. Even the BASELINE Ridge pred FLIPS SIGN between EXT (−0.0074) and HL70 (+0.0047) —
  the predictability is universe/era-conditional, not a stable XS edge. Reproduces the iteration-log
  finding "per-cycle IC R²≈0.005, genuinely unpredictable noise."
- **The pooled GBM does NOT beat Ridge on the production universe.** Default-config GBM is WEAKER on
  both. The heavily-regularized GBM looks strong on EXT (−0.0142, sign-stable all 7 folds) but
  **CATASTROPHICALLY fails transport to HL70**: HL70 IC −0.0042 (insignificant) and SIGN-INCONSISTENT
  per-fold (+0.0075,+0.0029,+0.0009, −0.0173,−0.0106,−0.0139,+0.001) — net near-zero, flips fold to
  fold. This is exactly the **universe-overfit wall (#2 / the Phase-2 #1 killer)**: a richer model
  fits the EXT panel's mean-reversion and produces a HL70 pred that is no better — and arguably worse
  (negative, unstable) — than the Ridge it would replace.
- GBM is also hyperparameter-fragile (IC swings −0.002 → −0.014 across configs on EXT), the
  classic overfit signature; no config wins on BOTH universes simultaneously.

## VERDICT: NO-CANDIDATE
No new feature family AND no model change clears the transport-first bar on the production universe:
- microstructure (order-flow/OFI, Amihud, OI) — IC≈0 / sign-flip (DEAD at the IC layer; iter-009 wall);
- reversal/momentum — strong+transport-stable but = iter-022's already-rejected signal, dies at the
  construction layer (p26);
- vov (the one new transport-stable signal) — dies at the construction layer (p2, tilt HURTS);
- pooled LightGBM — does not beat Ridge on HL70, and the EXT "win" fails transport (sign-unstable HL70).

**Strong finding: the V0 feature/alpha is near the achievable cross-sectional ceiling at 4h even with a
richer model.** The 4h alpha-residual is genuinely unpredictable cross-sectionally (|IC| ~0.005-0.007,
sign era-conditional); a higher-capacity model fits the noise and overfits the EXT panel without
transporting to HL70. The earlier walls (DD-leading-coincident / universe-overfit / marginal-within-
pred-pool) now extend to the FEATURE+MODEL layer of Phase-2.

## What would move the needle (out of free-data / model-capacity scope)
1. **A different HORIZON/target** where alpha is more predictable (the 4h residual is the hard case;
   intraday momentum or multi-day directional may have higher IC — but changes the strategy identity).
   This is the one untested Phase-2 axis with possible headroom; flag for a future iter.
2. **PAID orthogonal LEADING data** (Coinglass liquidations / Glassnode on-chain) — the only mechanism
   for the DD; needs human key.
3. **Genuinely non-cross-sectional structure** (per-symbol time-series + alt-index hedge) — a fresh
   project, not a feature/model swap; prior per-sym-timing tests (iter-004) didn't monetize.

## Scripts
- `research/convexity_portable_2026-05-20/scripts/iter027_feature_transport_precheck.py`
- `research/convexity_portable_2026-05-20/scripts/iter027_construction_marginal.py`
- `research/convexity_portable_2026-05-20/scripts/iter027_pooled_gbm_vs_ridge.py`

## Champion unchanged
BASELINE HL70 regime-hybrid held-book (Calmar +1.68) + iter-012 vol-norm reactive stop (k=2.0).
