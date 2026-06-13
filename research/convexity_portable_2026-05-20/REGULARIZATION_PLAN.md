# Regularization Optimization Plan

## Current state (X6 controlled matrix)

### Ridge
```python
RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])  # 5 values, 4 orders of magnitude
# GCV-selected per fold; α applied uniformly to all features + sym_id one-hot dummies
```

### LGBM (matches V3.1 production)
```python
# Pooled:    LR=0.03, n_estimators=400, num_leaves=31, max_depth=6,
#            min_data_in_leaf=300, feature_fraction=0.85, bagging_fraction=0.85,
#            bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1
# Per-sym:   LR=0.05, n_estimators=200, num_leaves=15, max_depth=4,
#            min_data_in_leaf=30, others same
# NO early stopping
```

## Identified weaknesses (ordered by likely impact)

### Ridge
| # | Issue | Fix | Estimated lift |
|---|---|---|---|
| R1 | α grid coarse (5 values, log spacing of 4 orders) | Expand to 12-value grid: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300 | +0.1–0.2 Sharpe if grid hits ceiling/floor |
| R2 | Don't log α picked per fold | Log `model.alpha_` per fold/cell — diagnostic | 0 (informational) |
| R3 | sym_id dummies penalized equally to standardized features | Demean target per-sym before Ridge → equivalent to free per-sym intercepts | +0.1–0.3 if production V3.1 sym_id effect is large |
| R4 | L2 only, no feature selection | ElasticNet (L1+L2) with l1_ratio ∈ {0.1, 0.5, 0.9} | +0.1 on cells with weak features; might hurt strong cells |
| R5 | GCV (leave-one-out approx) not time-aware | TimeSeriesSplit CV for α selection | +0.05 max |

### LGBM
| # | Issue | Fix | Estimated lift |
|---|---|---|---|
| L1 | No early stopping, fixed n_estimators=400/200 | Add `early_stopping_rounds=30` on last-10% val split | +0.1–0.2 on cells with weak signal that may overfit |
| L2 | `min_data_in_leaf` static (300 pool / 30 per-sym) | Adaptive: `max(20, n_train // 100)` | +0.05 |
| L3 | `reg_alpha=reg_lambda=0.1` token values, structure regularizes | Test {0.0, 0.1, 1.0, 10.0} for leaf-weight regularization | +0.05 |
| L4 | Per-sym `min_data_in_leaf=30` too low for short-history syms | Per-symbol adaptive based on actual n_train | +0.05 |
| L5 | Single LR/n_estim across all cells | Different (LR, n_estim) for pooled vs per-sym (already done) — extend per feature count | minor |

## Optimization plan: 4 phases

### Phase A — Ridge α + sym_id treatment (cheapest, highest impact)
Run on **best cell** (Ridge Pool+symid +aggT = +1.22):
- A1: Baseline (current grid, current sym_id treatment)
- A2: Wider α grid (12 values)
- A3: Wider α grid + free sym_id intercepts (target demean approach)
- A4: ElasticNet l1_ratio=0.1 + wider α grid
- A5: ElasticNet l1_ratio=0.5 + wider α grid
- A6: ElasticNet l1_ratio=0.9 + wider α grid

Log α picked per fold for each.

**Decision gate**: if any variant beats baseline by ≥+0.1 Sharpe with CI overlap, adopt for all Ridge cells.

### Phase B — Ridge regularization applied to all cells (if Phase A wins)
Re-run Ridge × {pool+symid, pool-nosym, per-sym} × {BASE, +aggT, +cohort, +v3, +crossX} with the best regularization scheme from Phase A. 15 cells.

### Phase C — LGBM early stopping + adaptive leaf size
Run on **best LGBM cell** (LGBM Pool+symid +aggT = -0.63):
- C1: Baseline (current)
- C2: With early stopping
- C3: Adaptive min_data_in_leaf
- C4: Tuned reg_alpha/reg_lambda

**Decision gate**: same as Phase A.

### Phase D — LGBM applied to all cells (if Phase C wins)
Re-run all 15 LGBM cells with best config.

## Apples-to-apples discipline

- Within Ridge: SAME regularization scheme across all Ridge cells (or fall back to per-cell-tuned).
- Within LGBM: SAME hyperparams across all LGBM cells.
- Cross-model comparison still valid because each model gets ITS appropriate regularization, just optimized within its class.

## Initial execution

Phase A only (Ridge regularization sweep on best cell). ~25 min compute.
After results, decide whether to proceed to Phase B (full Ridge re-run).
