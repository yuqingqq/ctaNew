# TODO — Comparison Matrix & Regularization Research

**Last updated:** 2026-05-20

---

## 🔁 Workflow discipline (apply for every completed step)

For each completed test/script:

1. **Review** — list key results (Sharpe, IC, folds+, conc per cell)
2. **Diagnose** — explain WHY the results are what they are (mechanism, interactions, etc.)
3. **Update docs** — append to COMPARISON_MATRIX_RESULTS.md and TODO.md
4. **Update task status** (TaskUpdate completed + description with verdict)
5. **Add follow-up tasks** if new questions emerge from results
6. **Dispatch next** from TODO list (highest priority pending)

NEVER let an idle moment between steps. Always have the next job queued.

---

## 📐 Original full comparison plan (the design space)

User-stated requirement: apples-to-apples comparison across model × architecture × features × universe.

### Dimension 1: Model class (2)
- [x] LGBM (non-linear, tree-based, can use categorical sym_id natively)
- [x] Ridge (linear, L2-regularized)
- [ ] ElasticNet (L1+L2) — partial X8b, 3/4 variants done
- [ ] Lasso (pure L1) — pending
- [ ] Stacked Ridge + LGBM ensemble — idea only

### Dimension 2: Architecture (3)
- [x] **Pooled + sym_id** — one model trained on all syms with sym_id as feature (LGBM categorical / Ridge one-hot dummies)
- [x] **Pooled - nosym** — one model trained on all syms WITHOUT sym_id (symbol-agnostic, portable in principle)
- [x] **Per-symbol** — independent model per symbol
- [ ] Hybrid: pooled + per-symbol residual stacking — idea
- [ ] Cluster-wise: k-means on idio features, then per-cluster pooled — idea

### Dimension 3: Feature set (6 tested)
- [x] **BASE** (14) — BTC-frame portable features
- [x] **+aggTrades** (5 added) — aggr_ratio_4h, tfi_4h, buy_count_4h, signed_volume_4h, avg_trade_size_4h
- [x] **+cohort** (3 added) — rvol_7d, ret_3d, btc_rvol_7d broadcast (regime context)
- [x] **+v3-augment** (4 added) — idio_max_abs_12b, idio_skew_1d, idio_kurt_1d, name_idio_share_1d
- [x] **+crossX** (5 added) — Binance vs OKX/Coinbase basis features (cross-exchange)
- [🔄] **+ALL** (31 = base+all extras) — running in X9
- [ ] **WINNER_21** (21, includes basket-frame `*_vs_bk` features) — V3.1's actual features; only tested for LGBM Pool+symid
- [ ] **+orderflow** (Step 95 of_tfi_z1d, of_imb_4h/1d, of_vol_z7d, of_kyle_1d, of_tsz_z1d) — perp aggTrade microstructure on 20-sym subset
- [ ] Polynomial features (return_1d², funding², etc.) — idea
- [ ] Symbol × feature interactions — idea

### Dimension 4: Universe (4 considered)
- [x] **HL-50** — 51-panel minus BTC (50 syms, all HL-tradeable) — PRIMARY universe for X6/X6b/X7/X9
- [ ] **51-panel** — 51 syms incl BTC (V3.1 production universe)
- [ ] **HL-70** — 70 HL-tradeable syms (50 shared + 20 not in 51-panel)
- [ ] **110-panel** — broader 110-sym universe
- [ ] **20-sym aggTrades subset** — Step 95 universe (HL ∩ aggTrades availability)
- [ ] **Universe stress drops** — 40-sym drop-top-vol, vol-tier filter $5M+, etc.

### Dimension 5: Target / preprocessing (2 tested)
- [x] **target_A = per-sym PIT z** (no clip, mild ±50 winsor) — X6+ standard
- [ ] **target_A with ±5 clip** (V3.1 production) — not tested in X6
- [ ] **target_z with per-sym RANK** (not z-score) — idea
- [ ] **alpha_beta raw** (no normalization) — idea
- [ ] **Different rolling window for σ_idio** (e.g. 14d vs 7d) — idea

### Dimension 6: Regularization (partial)
**Ridge / Linear (X8 + X8b done)**
- [x] α grid: {0.01, 0.1, 1, 10, 100} (narrow, X6 baseline)
- [x] α grid: {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300} (wider, X8 A2)
- [x] ElasticNet l1_ratio = 0.1, 0.5, 0.9 (X8b B1-B3 done; L1 hurts)
- [ ] **sym_id dummy normalization** (X8c, HIGH PRIORITY)
- [ ] **Group α** — separate α per feature group (X8c C2/C3)
- [ ] Lasso pure L1 (X8b B4 timed out)
- [ ] TimeSeriesSplit CV for α selection — idea

**LGBM (X8e pending)**
- [ ] Early stopping with 10% val split
- [ ] Adaptive min_data_in_leaf
- [ ] reg_alpha / reg_lambda sweep
- [ ] Monotonicity constraints

---

## 📊 Status snapshot

| Task | State |
|---|---|
| X6 (24-cell matrix, BASE/+aggT/+cohort/+v3 × 6 archs) | ✅ done |
| X6b (6 cohort cells, fix bug) | ✅ done |
| X7 (6 crossX cells) | ✅ done |
| X8 (Ridge reg sweep on best cell, 6 variants) | ✅ done |
| X8b (ElasticNet retry, 3/4 done; B4 timed out) | ✅ partial |
| X9 (6 all-features cells, restarted with 4h timeout) | 🔄 running (cell 1/6) |

---

## 📋 Next up (pending tasks)

### X8c — sym_id normalization sweep (HIGH PRIORITY)
**Hypothesis**: sym_id one-hot dummies (std≈0.14) are over-penalized ~50× vs standardized features (std≈1). Fix should let sym_id intercepts work properly.

Test cell: Ridge Pool+symid +aggT
Compute: ~20 min

- [ ] **C1**: Normalize sym_id dummies (mean+std fold-0) + RidgeCV wider grid
- [ ] **C2**: Group α — α_main from CV, α_sym=0 (free intercept)
- [ ] **C3**: Group α — α_main from CV, α_sym=α_main/50 (compensate 50× scale)
- [ ] **C4**: Drop sym_id entirely + wider α (reference)

### X8d — Cohort collinearity diagnosis (MEDIUM)
- [ ] **D1**: Standardize sym_id + cohort → does multicol resolve?
- [ ] **D2**: Drop sym_id + cohort → reproduce X6 Pool-nosym +cohort
- [ ] **D3**: Per-sym time-varying btc_rvol_7d (vs broadcast) → break collinearity

### X8e — LGBM regularization sweep (MEDIUM)
- [ ] **E1**: early_stopping_rounds=30, 10% val split
- [ ] **E2**: adaptive min_data_in_leaf
- [ ] **E3**: higher reg_alpha (1.0), reg_lambda (1.0)
- [ ] **E4**: E1+E2+E3 combined

### X10 — Apply winning regularization to all cells (HIGH, blocked on X8c)
- [ ] Re-run 12 Ridge cells with winning scheme
- [ ] Re-run 6 cohort cells
- [ ] Re-run 6 crossX cells
- [ ] Re-run 6 all-features cells (when X9 done)

### X11 — Universe stress test (LOW)
- [ ] 40-sym drop-top-vol
- [ ] $5M+ HL vol filter
- [ ] 110-panel broader
- [ ] 51-panel incl BTC (compare to V3.1)

### X12 — Apply best config to V3.1 production stack (LOW)
- [ ] Take best cell config (currently Ridge Per-sym +cohort)
- [ ] Add WINNER_21 basket-frame features back for absolute Sharpe parity
- [ ] Run V3.1 sleeve on full 51-panel incl BTC
- [ ] Compare to production +2.23 / HL-50 +3.00

### Pending from original plan (not yet done)
- [ ] **LGBM Per-sym +ALL** (X9 cell 3/6, running)
- [ ] **Ridge × 3 archs +ALL** (X9 cells 4-6, pending)
- [ ] **LGBM on WINNER_21** (V3.1 absolute Sharpe parity) — X12
- [ ] **Ridge on WINNER_21** (basket-frame compatibility) — X12

---

## 🎯 Current best cells

1. **Ridge Per-sym +cohort = +2.01 Sharpe, 5/9 folds, 48% conc** (BEST overall)
2. Ridge Pool+symid +aggT = +1.22 Sharpe, 5/9 folds, 76% conc
3. Ridge Per-sym +crossX = +1.12 Sharpe, 5/9 folds, **33% conc** (best concentration)

For reference (NOT apples-to-apples — uses WINNER_21 with basket-frame):
- V3.1 production LGBM Pool+symid WINNER_21 on HL-50 = +3.00 Sharpe (production baseline)

---

## 🔑 Key empirical findings

### Model class comparison
1. **Ridge dominates LGBM on portable BTC-frame features** by ~2 Sharpe across the board
2. Best LGBM: -0.17 (Per-sym +crossX); Best Ridge: +2.01 (Per-sym +cohort)
3. LGBM relies HEAVILY on sym_id (Δ -1.11 dropping it); Ridge less so (Δ -0.30)

### Architecture × feature interactions
1. **aggT** helps when sym_id present (+0.81 LGBM Pool+symid, +0.84 Ridge Pool+symid)
2. **cohort** helps when sym_id ABSENT or per-sym (+1.44 LGBM Pool-nosym, +1.67 Ridge Per-sym); HURTS Ridge Pool+symid (-1.10, multicol)
3. **v3 idio_*** substitutes for sym_id (+1.13 LGBM Pool-nosym, +1.11 LGBM Per-sym)
4. **crossX** has cleanest cross-symbol signal: lift +2.06 for LGBM Per-sym (biggest LGBM lift)
5. **+ALL** bimodal: hurts Pool+symid (-0.60 LGBM), helps Pool-nosym (+1.79 LGBM)

### Regularization
6. **ElasticNet/Lasso L1 hurts** (RidgeCV +1.26 > all ElasticNet variants ≤ +0.82)
7. **L1 hurts non-monotonically**: l1=0.5 worst (+0.45), l1=0.9 rebounds (+0.82)
8. **Wider α grid** for Ridge gives +0.04 lift (α=30 sometimes picked, not in narrow grid)
9. **Free per-sym intercept via target demean COLLAPSES** Sharpe (-0.91 vs +1.22) — train means don't generalize
10. **sym_id dummies over-penalized 50×** vs standardized features — open question whether fixing helps

### Cross-exchange features
11. Negative IC -0.013 to -0.018 with t=-3.8 to -5.5 — mean-reversion of cross-venue arb gap
12. Coverage: 53-63% across 44-48 syms
13. Strongest: `bn_cb_basis_z` (Binance perp vs Coinbase spot)
14. Null: `okx_perp_spot_basis_z` (intra-OKX too tight to predict alpha)

---

## 🔬 Ideas / parking lot (lower priority)

- [ ] Polynomial features (x², x³)
- [ ] Per-feature interaction terms (feature × sym_id)
- [ ] Symbol clustering + cluster-wise models
- [ ] Stacking Ridge + LGBM with meta-Ridge on val
- [ ] Sample weights (recent > old)
- [ ] Quantile regression for tail-heavy target
- [ ] Bayesian model averaging across architectures
- [ ] Different sleeve overlay configs (K=2, 4 instead of 3)
- [ ] Longer hold horizon (24h, 48h instead of 4h cycle)

---

## 📁 Output files

- `results/X6_controlled_matrix.csv` — master matrix (36 cells)
- `results/X8_ridge_reg_sweep.csv` — Ridge reg variants (6 cells)
- `results/X8b_elasticnet_results.csv` — pending save (B4 timed out before write)
- `results/COMPARISON_MATRIX_RESULTS.md` — narrative summary
- `data/ml/cache/cross_exchange_features.parquet` — Binance-OKX-Coinbase basis features
- `data/ml/cache/okx_{spot,swap}_<SYM>_1h.parquet` — raw OKX klines
- `data/ml/cache/cb_spot_<SYM>_1h.parquet` — raw Coinbase klines

---

## 📋 Phase 2 update (2026-05-20/21) — all current tasks

### ✅ COMPLETED Phase 2 tasks (X18–X32)

| Task | What | Result |
|---|---|---|
| **X18** | Build panel_v2 (98.3% aggT coverage, all 51 syms) | Done in 50s, 2.0GB parquet |
| X19 | Preprocessing sweep (winsor/rank/robust) | All variants underperformed canonical |
| X20 | Universe N-stress (drifted framework) | Superseded by X23 |
| **X21** | Framework drift bug found (COHORT_EXTRAS in HEAVY_TAIL → -1.85 Sharpe) | Fixed |
| X22 | Re-run +aggT cells with v2 | Per-sym arch +0.17 to +1.30; pooled HURT |
| **X23** | Universe sweep (16 variants) | HL-50 +2.01 global max, HL-45 most robust |
| **X24** | Cluster universes (15 variants) | AI cluster (3 syms) CRITICAL, sector combos collapse |
| X25 | Data-driven clusters | Matches X24 |
| X26 | Cohort + extras combos (OLD panel — partly invalid) | Superseded by X29 |
| **X27** | Per-group α cross-universe validation | DECISIVELY UNIVERSE-OVERFIT (-1.53 / -3.01 lifts) |
| X28 | aggT+crossX diagnostic (OLD panel) | Wrong narrative — superseded by X30 |
| **X29** | Cohort combos with v2 panel | V5 kitchen sink +1.66 (7/9 folds, 26% conc) — robust alternative to V0 +2.01 |
| **X30** | V0 vs V5 diagnostic (proper) | V5 wins via prediction DIVERSITY, not IC |
| **X31** | Build panel_hl70 (50 + 20 new) | 8.6M rows, 678MB, 38s |
| **X32** | HL-70 universe tests | HL70_full = -0.11 (DROP -2.12 vs HL-50), 20 new syms NEGATIVE IC |

### Phase 2 KEY FINDINGS

1. **HL-50 is the global universe optimum**: HL-30 to HL-45 lose, HL-70 catastrophic, 51-panel+BTC collapses
2. **Production cell**: Ridge Per-sym + cohort = +2.01 (canonical), 17 features
3. **Alternative cell**: V5 kitchen sink Ridge Per-sym (31 features) = +1.66, 7/9 folds (robust)
4. **Per-group α universe-overfits**: don't optimize α per universe
5. **AI cluster (3 syms TAO/VIRTUAL/VVV) is critical** — removing crashes Sharpe to ~0
6. **20 new HL syms have NEGATIVE IC** — adding them destroys K=3 selection

### Phase 2 NEW outputs

- `outputs/vBTC_features/panel_variants_with_funding_v2.parquet` (USE for aggT, 98.3% cov)
- `outputs/vBTC_features/panel_hl70.parquet` (HL-70 expansion test)
- `data/ml/test/parquet/aggTrades/<SYM>/` — 51 GB raw aggTrades (51 syms total)
- `research/convexity_portable_2026-05-20/results/X18-X32_*.csv` — per-test results
- `FINAL_SYNTHESIS_PHASE2.md` — current consolidated synthesis

### ❌ Permanently CLOSED directions

- Per-group α optimization (universe-overfit)
- 110-panel/HL-70 universe expansion (NEGATIVE IC noise destroys K=3)
- aggT data collection for missing syms (helped per-sym +0.17-1.30, hurt pooled -0.54)
- LGBM regularization tuning (defaults at local optimum)
- ElasticNet L1 (consistently hurts vs Ridge)
- Cross-feature combinations beyond cohort (BASE+cohort alone = best Per-sym Ridge cell)
- Preprocessing variants (winsor baseline already best)

### 🟡 OPEN ideas (not yet tested)

- **Per-sym α (not per-group)**: even more granular than X27, likely same overfit issue
- **Longer hold horizon** with V5 framework
- **Ensemble V0 + V5** prediction averaging
- **Bootstrap CI on production Sharpe** for V3.1 / V0 / V5 (production-readiness)

### Phase 2 TASK STATUS

All Phase 2 tasks complete. Production candidates fixed:
- V3.1 (LGBM Pool+symid WINNER_21) = +3.00
- V0 (Ridge Per-sym + cohort) = +2.01 (portable, simple)
- V5 (Ridge Per-sym + ALL 31 feats) = +1.66 (robustness-focused)

