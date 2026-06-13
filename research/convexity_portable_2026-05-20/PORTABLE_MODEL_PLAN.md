# Portable Model Extraction Plan — 110-panel

**Date:** 2026-05-20
**Goal:** Train a portable model on 110-panel that AT MINIMUM matches V3.1's HL-50 Sharpe (+3.00), and potentially exceeds it by capturing signal from the 60 extra symbols.

## Architectural commitments (pre-registered)

| Aspect | V3.1 production (51-panel) | Portable model (110-panel) |
|---|---|---|
| Universe | 51 syms (incl BTC) | 110 syms (no BTC) |
| Symbol identity (sym_id) | YES (categorical, in features) | **NO** (portable across syms) |
| Target | target_A on 51-panel (unclipped) | alpha_beta on 110 (PIT 4h forward residual, NO ±5 clip) |
| Per-symbol normalization | NO | YES (per-symbol z via PIT trailing) |
| Cross-sectional features | basket-frame xs-rank | **BTC-frame only**, no basket |
| Preprocessing heavy-tail features | mixed | **rank-transform** pooled fold-0 + z |
| Model class | LGBM 5-seed ensemble | Both Ridge (portable linear) AND LGBM-no-sym_id |
| Feature set | WINNER_21 | 17 BTC-frame V2-equivalent (no basket-frame, no xs-rank) |
| CV | 9-fold walk-forward expanding | same |
| Embargo | 1 day | same |
| Cost convention | 9 bps RT | same |

## Pre-committed feature set (BTC-frame, sym_id-free)

```
return_1d            atr_pct              obv_z_1d
vwap_slope_96        bars_since_high      autocorr_pctile_7d
corr_to_btc_1d       corr_to_btc_change_3d
idio_vol_to_btc_1h   idio_vol_to_btc_1d
beta_to_btc_change_5d  return_8h          vol_zscore_4h_over_7d
dom_btc_z_1d         dom_btc_change_288b
funding_rate         funding_rate_z_7d    funding_rate_1d_change
```

18 features, all BTC-frame, no `sym_id`, no `*_vs_bk`, no `xs_rank`.

## Preprocessing pipeline

- Heavy-tail set (kurt > 50 in fold-0 train, expect: funding × 3, idio_vol × 2):
  pooled rank transform on fold-0 train → re-z-score using fold-0 train rank stats
- Standard set: winsorize p1/p99 on fold-0 train → z-score using fold-0 train stats
- NaN handling: explicit → 0 after preprocessing
- All stats from fold-0 train ONLY (per linear_model arc convention) — no peeking

## Phase X1 — Train portable model

Two model classes in parallel:
1. **Ridge** with RidgeCV α ∈ {0.01, 0.1, 1, 10, 100}, gcv on fold-train
2. **LGBM-no-sym_id**, hyperparameters pinned from production V3.1 (per CLAUDE.md), 5-seed ensemble, NO categorical features

Train on 110-panel target = `alpha_beta` (PIT 4h forward), per-symbol PIT-trailing-z-normalized target (per-symbol scale handling, NOT the ±5 clip).

Output: predictions per (sym, time, fold) — same schema as V3.1's all_predictions.parquet.

## Phase X2 — Apply V3.1 sleeve machinery to new predictions

Same `phase_ah_sleeve.py` machinery used for V3.1 production:
- N=15 rolling-IC universe selector
- K=3 picks per side
- conv_gate + flat_real
- 6-sleeve 24h-hold overlay
- 9 bps RT cost

Two evaluations:
- A. **On full 110-panel** (110 syms tradeable)
- B. **On HL-70 subset** (the 70 HL-tradeable syms — most relevant for production)
- C. **On HL-50 subset** (the 50 shared with current production — apples-to-apples vs V3.1's +3.00)

## Gates (pre-committed, BINARY)

| Gate | Pass criterion |
|---|---|
| **C vs V3.1 on HL-50** | Sharpe ≥ +3.00 (must AT LEAST match — proving portable model can replicate production) |
| **B on HL-70** | Sharpe ≥ +2.50 (within 0.5 of HL-50; portability bonus = extracting signal from extras) |
| **A on 110-panel** | Sharpe ≥ +2.00 (broader universe; less curated, more noise) |
| Folds positive | ≥ 6/9 on all three |
| Matched placebo p95 | PASS on B and C |
| LOFO drop-top-2 sym | Sharpe ≥ +1.5 on B and C |
| Half-of-sample | both halves > 0 on B and C |

**PASS = C ≥ +3.00 AND B ≥ +2.50 AND all sensitivity gates pass.**

**Best-case interpretation**: portable model recovers production AND extracts incremental signal from extras → universe-portable +3 Sharpe strategy → operational deployment with broader universe.

**Worst-case interpretation**: portable model can't even match V3.1 on HL-50 (gates fail) → the panel-specific feature pattern can NOT be captured by a portable architecture → V3.1's edge is inherent to its sym_id + basket-frame specialization → close the portable-extraction direction.

## Anti-patterns to guard against (carried from prior reviews)

1. **One-fold concentration**: report per-fold Sharpe, gate ≥6/9 positive
2. **Half-of-sample fragility**: H1 vs H2 both must be >0
3. **Drop-top-K syms**: must survive drop-top-2 at ≥+1.5
4. **Universe-disjoint test**: train on 110 but evaluate restricted to 50 → if MUCH worse than V3.1, retrain dilutes signal
5. **Target normalization**: per-symbol z (PIT trailing) is the planned fix; if predictions look flat/saturated, the issue is the normalization

## Compute budget

| Step | Time |
|---|---|
| Ridge train + predict | 5 min |
| LGBM-no-sym_id train + predict (5 seed × 9 folds) | 30 min |
| Sleeve evaluation × 3 (full/HL-70/HL-50) × 2 models | 60 min |
| Sensitivity diagnostics + placebo | 15 min |
| Total | ~2 hours |

## Why this is genuinely different from prior attempts

Phase UNI-111 retrained V3.1 with sym_id ON 111 syms and target_A clipped → failed.
R2a added 3 features to 51-panel WINNER_21 → failed.
This plan: **drops sym_id, drops basket-frame, uses unclipped target with per-symbol z, restricts to portable BTC-frame features.** It's a cleaner test of whether portable architecture can extract signal from the broader universe.
