# Linear-model β-residual pipeline — results

Last updated: 2026-05-13
Status: **HISTORICAL — SUPERSEDED**

> ⚠️ **DEPRECATED.** This document describes the original Step 1–6 pipeline (Ridge with sym_id one-hot, threshold-bps gating). That path was rejected (Sharpe −1.62), but the verdict in this file is **not the current state**.
>
> **Current canonical state**: V2 R3_BTC (22 features, NaN-safe rank preprocessing, no sym_id, no basket features) + IC-signed wrapper + V3.1 sleeve gives **+2.19 on 51-panel (placebo-validated)** and **+2.03 on 110-panel** (Step 41 rerun, 2-feature PIT-shift fix + BTC excluded). See `STATUS.md` and `HANDOFF.md` for the live picture.
>
> Read this file only for the original sym-id Ridge rejection and the β-shift leak quantification (-0.06 Sharpe). Do not use it as a verdict on linear models for this problem.

## TL;DR

| variant | model | Sharpe | per-cycle IC |
|---|---|---|---|
| **Production baseline** | LGBM shift(1) leaky | **+0.74** | +0.0157 |
| LGBM shift(49) clean-PIT | LGBM | +0.68 | +0.0159 |
| Ridge through production protocol | Ridge + sym one-hot | **−1.62** | +0.0135 |
| Ridge with threshold-bps gate (best) | Ridge + sym one-hot | −0.46 | +0.0135 |

**Linear model rejected**: Ridge cannot match LGBM on this problem. ~2.3-2.4 Sharpe gap. Threshold-bps gating idea is mechanically sound but Ridge predictions have inverted tails (decile 0 has +6.5 bps mean realized α_β, decile 9 only +1.2 bps), so threshold gates pick the wrong subset.

**Side finding**: production β computation uses `shift(1)` after rolling regression on forward 4h returns — has a 47-bar look-ahead in theory. Empirically quantified at **−0.06 Sharpe** (essentially negligible). Production baseline is approximately PIT-clean.

## Pipeline implementation summary

### Step 1: Target construction (`scripts/01_build_target.py`)
- Rolling 90d × 288-bar PIT β via OLS on forward 4h returns
- α_β = return_pct − β_pit × btc_ret_t
- σ_idio per symbol frozen from fold-0 training (51 constants, BTCUSDT=0 excluded)
- target_z = α_β / σ_idio, winsorized at ±5σ
- target_bps recoverable at inference: pred_bps = pred_z × σ_idio × 1e4
- 5.73M rows × 50 symbols (BTC dropped)

### Step 2: Feature preparation (`scripts/02_build_features.py`)
- WINNER_17 minus sym_id = 16 numeric features
- Each feature: winsorize at fold-0 [1, 99] pct → z-score with fold-0 mean/std
- sym_id one-hot encoded → 49 dummies (AAVEUSDT as reference)
- Total: 65 features
- Verified: mean ≈ 0, std ≈ 1 on fold-0 training rows

### Step 3: Ridge training (`scripts/03_train_ridge.py`)
- 10-fold walk-forward, 5-seed bootstrap-bagged RidgeCV
- α grid: {0.1, 1, 10, 100, 1000, 10000}
- Result: ALL folds × ALL seeds selected α = 0.1 (smallest in grid)
- Per-cycle IC: +0.0135 overall, 10/10 folds positive (+0.0019 to +0.0300)
- vs LGBM clean-PIT IC: +0.0159 (Ridge gets ~85% of LGBM's IC)

### Step 4: Threshold-bps gated backtest (`scripts/04_backtest.py`)

| threshold (bps) | Sharpe | CI | end-eq | traded% | folds+ |
|---|---|---|---|---|---|
| 0 | −0.61 | [−2.69,+1.51] | $72 | 63% | 5/9 |
| 4.5 | −0.79 | [−3.02,+1.56] | $62 | 38% | 5/9 |
| 9 | −0.75 | [−2.82,+1.62] | $69 | 19% | 5/9 |
| 15 | −1.57 | [−3.52,+0.80] | $46 | 8% | 3/9 |
| 25 | −0.46 | [−2.77,+1.77] | $90 | 2% | 2/9 |

All thresholds negative Sharpe. Higher threshold → smaller baskets → fewer trades but still losses.

### Step 5: Diagnostic — why threshold gating fails

Decile analysis of pred_bps vs realized α_β:

| pred decile | mean α_β (bps) |
|---|---|
| 0 (lowest pred) | **+6.54** |
| 1 | +3.46 |
| 2 | +1.60 |
| 3-8 | +0.4 to +1.6 |
| 9 (highest pred) | +1.23 |

Top-3 picks per cycle: +1.03 bps. **Bottom-3 picks per cycle: +11.39 bps.** Spread is **−10.35 bps** — INVERTED.

The Ridge predictions correlate positively with α_β on average (IC +0.0135) but the relationship inverts at the tails: the symbols Ridge predicts will UNDERPERFORM tend to OUTPERFORM. Threshold-bps gating selects the inverted tails and trades them — hence losses.

Why does Ridge have inverted tails? Two likely contributors:
1. **σ_idio scaling bias**: pred_bps = pred_z × σ_idio. High-vol symbols (PUMP σ=394 bps, PENGU σ=359 bps) dominate the bps tails — even modest z predictions become large bps values. The model's signal on these high-σ symbols is the noisiest part of its predictions.
2. **Linear extrapolation**: at extreme feature values, Ridge produces extrapolated predictions with no tree-based clipping. Tree models naturally limit prediction magnitude through leaf-wise averaging.

### Step 6: Apples-to-apples model comparison (`scripts/06_ridge_via_production_protocol.py`)
- Ridge predictions through the EXACT production protocol (conv_gate + PM_M + filter_refill)
- Same V3.1 6-sleeve aggregation
- Same universe filter (rolling-IC top-15)
- Result: Sharpe **−1.62** vs LGBM clean-PIT +0.68. Gap: −2.30 Sharpe

Per-fold for Ridge production protocol: 4/9 positive. Worst folds: fold 3 (−8.59 Sharpe), fold 9 (−6.92). Ridge has catastrophic fold-specific losses that LGBM doesn't share.

### Step 5/leak: Quantifying β shift

Test: LGBM WINNER_17 with `min_periods=1000` (production warmup):

| β shift | IC | Sharpe | end-eq |
|---|---|---|---|
| shift(1) leaky | +0.0157 | +0.74 | $126.96 |
| shift(49) clean-PIT | +0.0159 | +0.68 | $129.08 |
| Δ | +0.0003 | **−0.06** | +$2.12 |

The 47-bar look-ahead in β estimation is real (verified empirically — return_pct correlates +1.0 with forward 4h return), but quantitatively **−0.06 Sharpe impact**. Production baseline is approximately PIT-clean. Earlier observation of "+2.18 Sharpe lift" was confounded by simultaneously changing `min_warmup` from 1000 to 4032; isolating the shift alone gives the −0.06 result.

## Why Ridge underperforms LGBM here

Per-fold breakdown (Ridge production protocol):
- folds 2, 4, 6, 8: positive (+0.86 to +2.98 Sharpe)
- folds 1, 3, 5, 7, 9: negative (down to **−8.59 fold 3**, **−6.92 fold 9**)
- LGBM in same folds: more stable, no fold worse than −3

Three hypotheses for Ridge's failure:
1. **One-hot sym dummies (49 cols) overfit baseline drift**: Ridge has to spread coefficient capacity across 49 per-symbol intercepts. The 16 numeric features may be regularized into near-zero coefficients.
2. **No interactions**: production WINNER_17 model relies on vol × funding type splits (per Phase H memory). Ridge with raw z-scored features can't learn this.
3. **Linear extrapolation in extreme regimes**: when a feature is at p99 of its training distribution, Ridge linearly extrapolates. LGBM stays within training-leaf bounds. This is most damaging on the few high-vol bars per fold that make up the tail.

## What would actually be needed to make linear competitive

If one wanted to revive the linear path:
1. **Feature engineering for non-linearity**: explicit interactions (funding × vol_regime), polynomial terms on key features, regime-conditional features.
2. **Drop sym_id one-hot**: replace with target-encoded per-symbol historical mean (computed PIT from fold-0 training).
3. **Bagged Ridge across multiple feature sub-blocks**: train one Ridge per feature family, ensemble predictions. Reduces correlation-induced coefficient instability.
4. **Pipeline calibration**: the production conv_gate's percentile thresholds were tuned for LGBM's pred distribution shape. Would need re-calibration for Ridge.

Estimated cost: ~2-3 days of careful engineering, with uncertain payoff. The existing LGBM baseline at +0.74 is hard to beat at this signal level (per-cycle IC ceiling ≈ +0.02). For diagnostic value (interpretable coefficients) the Ridge model could still be informative; for production it's not viable.

## Conclusions

1. **Linear model REJECTED for this problem.** Ridge produces ~2.3-2.4 Sharpe lower than LGBM on identical clean-PIT pipeline. Threshold-bps gating works mechanically but Ridge's prediction tails are inverted, defeating the gate.

2. **Production +0.74 baseline is approximately clean-PIT.** The β shift(1) leak is real but quantitatively only −0.06 Sharpe (not load-bearing). No urgent need to re-validate production.

3. **σ-recovery for bps interpretation works.** pred_bps = pred_z × σ_idio × 1e4 produces interpretable per-symbol expected returns. Useful for diagnostic purposes (e.g., "model expects +30 bps on AVAX") even if not as a trading gate for this specific Ridge model.

4. **Model class is load-bearing.** Confirms Phase RANK's finding from memory: model objective (MSE vs LambdaRank) wasn't the bottleneck, but model FAMILY (linear vs tree) clearly is. Tree models capture interaction value worth ~2 Sharpe units on this problem.

## Files produced

```
linear_model/
├── docs/
│   ├── design.md           (pre-implementation plan)
│   └── RESULTS.md          (this file)
├── scripts/
│   ├── 01_build_target.py
│   ├── 02_build_features.py
│   ├── 03_train_ridge.py
│   ├── 04_backtest.py
│   ├── 05_lgbm_clean_pit_baseline.py
│   └── 06_ridge_via_production_protocol.py
├── data/
│   ├── targets.parquet            5.73M rows
│   ├── features.parquet           65-col X matrix
│   ├── sigma_idio.csv             51 frozen σ per symbol
│   └── beta_pit.parquet           PIT β series
├── models/
│   └── (RidgeCV models discarded — coefficients saved in results/)
└── results/
    ├── predictions.parquet        Ridge OOS preds
    ├── coefficients.csv           per-fold per-seed Ridge coefs
    ├── cv_alphas.csv              RidgeCV alpha selection
    ├── threshold_sweep.csv        per-threshold backtest summary
    ├── v31_thresh_{0,4.5,9,15,25}bps.csv
    ├── ridge_production_protocol_v31.csv
    └── lgbm_shift{1,49}_predictions.parquet
```
