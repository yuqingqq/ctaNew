# Linear-model β-residual pipeline — design doc

Last updated: 2026-05-13
Status: **HISTORICAL INITIAL DESIGN — SUPERSEDED**

> ⚠️ **DEPRECATED.** This document describes the original (Step 1–6) design with absolute-bps target + threshold-bps gating + sym_id one-hot. That design was tested and **rejected** (see `RESULTS.md`).
>
> **The actual winning architecture** (Step 34–41) is different:
> - **z-scored target** (`target_z = α_β / σ_idio`), not absolute bps
> - **NO sym_id one-hot** (Step 9 confirmed sym dummies absorbed 56× more coef mass than numerics without helping)
> - **NO basket features** (R3_BTC discipline — pure BTC-frame)
> - **NaN-safe rank preprocessing** for heavy-tail features + per-symbol rank for funding (Step 34)
> - **IC-signed wrapper** (`pred_B = pred_z × trail_ic`) for per-symbol calibration at inference
> - **V3.1 6-sleeve overlay** + standard `conv_gate` + `PM_M=2` + `filter_refill`
> - Threshold-bps gating is NOT used; the dispersion-based `conv_gate` from production is kept
>
> See `STATUS.md` and `HANDOFF.md` for the current architecture, results, and pending validation gates.

## Why a separate pipeline

The production LGBM stack at WINNER_17 + V3.1 = Sharpe +0.74 has been
extensively tested for feature additions (v3, v4 — all rejected). The
hypothesis here is that the LGBM model class itself may not be load-bearing;
Phase RANK showed model objective is not the bottleneck. A linear (Ridge)
model with proper feature engineering could match or beat LGBM, with the
ancillary benefits of:
- Interpretable coefficients (diagnose which features drive picks)
- Direct prediction in bps units (enable absolute-magnitude gating)
- Faster iteration (~5s/fold vs LGBM ~15s)
- Cleaner theoretical foundation (no tree-noise overfit handles)

## Key design choice: absolute-bps target

Production target is z-scored: `target_β = α_β / σ_idio` per symbol.
That gives unitless predictions which forces dispersion-based gating
(`conv_gate` skips cycles where prediction dispersion < 30th pctile).

Linear pipeline uses absolute target: `target_bps = α_β × 1e4` (basis points).
Predictions in bps directly allow:
- **Threshold gating**: trade only when `|pred_bps| > 9 bps` (covers half-spread + slippage)
- **Position sizing proportional to pred** (optional extension)
- **No conv_gate needed**: model itself decides when its prediction is large enough

This addresses the production stack's "selection-via-noise" failure mode:
when the model has weak signal, threshold gating leaves us flat (no trade);
when signal is strong, we trade and capture the spread.

## Architecture overview

```
1. Compute α_β (rolling 90d OLS β × forward 4h returns)         [same as LGBM pipeline]
2. Target = α_β × 1e4 (bps), winsorize at fold-0 ±X% pctile
3. Features (WINNER_17):
   - 16 numeric: winsorize 1/99 → z-score (fold-0 train stats)
   - sym_id: one-hot encode (50 dummies, drop 1 reference)
4. Walk-forward 10-fold CV, RidgeCV per fold (alpha grid)
5. 5-seed bootstrap bagging ensemble
6. Threshold-gated backtest with V3.1 sleeve overlay
```

## What's reused vs new

| component | reused from production | new for linear model |
|---|---|---|
| Source panel | `outputs/vBTC_features/panel_variants_with_funding.parquet` | — |
| β estimation | rolling 90d OLS via `compute_pit_beta()` | — |
| α_β formula | `return_pct − β × btc_ret_t` | — |
| Target | z-scored (`α_β / σ_idio`) | **absolute bps (`α_β × 1e4`)** |
| Feature standardization | none (LGBM is scale-invariant) | **winsorize + z-score** |
| sym_id handling | raw integer categorical | **one-hot dummies** |
| Model | LGBM 5-seed | **RidgeCV 5-seed bagged** |
| CV | walk-forward 10 folds | walk-forward 10 folds (same splits) |
| Universe filter | rolling-IC top-15 | rolling-IC top-15 |
| Cycle gate | `conv_gate` dispersion-percentile | **threshold-bps** (sweep {0,4.5,9,15,25}) |
| Persistence gate | PM_M2 | PM_M2 (same) |
| Trailing PnL gate | filter_refill 90d | filter_refill 90d (same) |
| Sleeve overlay | V3.1 6-sleeve | V3.1 6-sleeve (same) |
| MTM | β-hedged on α_β | β-hedged on α_β |

## Folder layout

```
linear_model/
├── docs/
│   ├── design.md            (this file)
│   ├── step1_review.md      (step-by-step verification logs)
│   ├── step2_review.md
│   ├── step3_review.md
│   ├── step4_review.md
│   └── RESULTS.md           (final summary)
├── scripts/
│   ├── 01_build_target.py
│   ├── 02_build_features.py
│   ├── 03_train_ridge.py
│   ├── 04_backtest.py
│   └── 05_compare_baseline.py
├── data/
│   ├── targets.parquet      (α_β bps target + meta)
│   ├── features.parquet     (standardized X matrix)
│   └── beta_pit.parquet     (rolling β per symbol)
├── models/
│   └── ridge_fold{0..9}_seed{0..4}.pkl
└── results/
    ├── predictions.parquet
    ├── per_cycle_ic.csv
    ├── threshold_sweep.csv
    └── per_fold_breakdown.csv
```

## Adoption criteria

Same as production:
- Sharpe ≥ +0.94 (i.e. +0.20 over WINNER_17 baseline), or
- Sharpe in [+0.50, +0.94) with **provably better** interpretability and lower fold variance

If the Ridge pipeline lands at +0.50-0.70 Sharpe with similar or smaller per-fold variance, we keep both for diagnostic comparison but don't switch production.

## Open questions

1. **Winsorize target at what threshold?** ±200 bps? ±500 bps? Will inspect fold-0 distribution to decide.
2. **Per-symbol intercept**: one-hot is the cleanest; could also try per-symbol target demean. Start with one-hot.
3. **Bagging method**: bootstrap rows vs feature subsample? Start with bootstrap rows (matches LGBM ensemble convention).
4. **Ridge α grid**: {0.1, 1, 10, 100, 1000}. Will let CV pick per fold.
