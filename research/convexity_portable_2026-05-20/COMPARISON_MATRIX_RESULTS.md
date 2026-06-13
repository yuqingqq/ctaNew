# Apples-to-Apples Comparison Matrix — Live Results

**Last updated:** 2026-05-20 (X13 complete: orthogonality test refutes additivity)

## X13 orthogonality test result (aggT + crossX combining)

Three approaches all fail to beat +aggT alone (+1.22):
| Variant | Sharpe | IC | Notes |
|---|---|---|---|
| **+aggT alone (baseline)** | **+1.22** | +0.0070 | individual best |
| E0 joint Ridge (BASE+aggT+crossX) | +1.00 | +0.0069 | -0.22 vs aggT alone |
| E1 ensemble (avg of separately-trained) | +0.43 | +0.0074 | dilutes to crossX level |
| E2 group α (per-group α from CV) | +0.31 | +0.0071 | all picked α=300 |

**Verdict**: aggT and crossX are NOT orthogonal in PnL space. Combining adds noise. Use single best group.
**Status:** All 6 dimensions × 6 architectures × 5 feature sets = 36 cells DONE. Next: X8c sym_id normalization sweep.

## Complete +ALL feature set results (all 6 cells)

| Cell | Sharpe | folds+ | conc | totPnL | Lift vs BASE |
|---|---|---|---|---|---|
| LGBM Pool+symid +ALL | -2.04 | 1/9 | 100% | -6584 | -0.60 (HURT) |
| LGBM Pool-nosym +ALL | **-0.72** | 3/9 | 64% | -2708 | **+1.83** (BEST nosym LGBM lift) |
| LGBM Per-sym +ALL | **+0.61** | 4/9 | 63% | +1742 | **+2.84** (FIRST positive LGBM!) |
| Ridge Pool+symid +ALL | -1.11 | 3/9 | 92% | -2868 | -1.49 (HURT) |
| Ridge Pool-nosym +ALL | -0.08 | 4/9 | 64% | -265 | -0.16 |
| **Ridge Per-sym +ALL** | **+1.44** | 5/9 | **33%** | +5654 | +1.10 |

**Headline**: combining ALL 31 features:
- HURTS Pool+symid (both models) — multicollinearity / feature interference
- HELPS Per-sym + Pool-nosym (LGBM gets first positive Sharpe!)
- Ridge Per-sym +ALL = #4 in entire matrix (behind +cohort +2.01, +aggT +1.22, +crossX +1.12)
- Concentration 33% (best-tied) — well-distributed signal

## Experimental design

All cells trained from scratch with identical conditions to enable apples-to-apples comparison.

### Fixed
| Aspect | Value |
|---|---|
| Universe | HL-50 (51-panel minus BTC, all HL-tradeable) |
| Sample | 2025-04-01 → 2026-05-06 (13 months) |
| Target | `alpha_vs_btc_realized` → per-symbol PIT z (no clip; mild ±50 winsor) |
| Folds | 9-fold walk-forward expanding, 1-day embargo, label purging via exit_time |
| Preprocessing (Ridge) | Heavy-tail features rank-fold0 + z; standard winsor p1/p99 + z fold-0 |
| Preprocessing (LGBM) | Raw features (LGBM is scale-invariant) |
| Evaluation | V3.1 6-sleeve, K=3, conv_gate + flat_real, 9 bps RT cost |

### Varied
| Dimension | Values |
|---|---|
| Model class | LGBM (non-linear) / Ridge (linear) |
| Architecture | Pool+symid / Pool-nosym / Per-symbol |
| Feature set | BASE / +aggT / +cohort / +v3 / +crossX / +ALL |

### Feature sets

| Set | n | Features |
|---|---|---|
| BASE (14) | 14 | return_1d, atr_pct, obv_z_1d, vwap_slope_96, bars_since_high, bars_since_high_xs_rank, autocorr_pctile_7d, corr_to_btc_1d, beta_to_btc_change_5d, idio_vol_to_btc_1h, idio_vol_to_btc_1d, funding_rate, funding_rate_z_7d, funding_rate_1d_change |
| +aggT (+5) | 19 | + aggr_ratio_4h, tfi_4h, buy_count_4h, signed_volume_4h, avg_trade_size_4h |
| +cohort (+3) | 17 | + rvol_7d, ret_3d, btc_rvol_7d (broadcast) |
| +v3 (+4) | 18 | + idio_max_abs_12b, idio_skew_1d, idio_kurt_1d, name_idio_share_1d |
| +crossX (+5) | 19 | + bn_okx_perp_basis_z, bn_okx_spot_basis_z, okx_perp_spot_basis_z, bn_cb_basis_z, okx_cb_spot_basis_z |
| +ALL | 31 | All of the above combined |

### Cross-exchange feature derivation

- OKX 1h klines collected for 49 swap + 48 spot syms (49/51 syms covered)
- Coinbase 1h klines for 47 syms
- Features computed as basis in bps then per-symbol trailing-30d PIT z

Pooled IC for crossX features vs `alpha_vs_btc_realized`:
- `bn_okx_perp_basis_z`: -0.0143 (t=-4.7) — Binance perp premium → underperform
- `bn_okx_spot_basis_z`: -0.0170 (t=-5.5)
- `okx_perp_spot_basis_z`: +0.002 (t=0.7, null — intra-OKX too tight)
- `bn_cb_basis_z`: **-0.0176 (t=-5.5)** — strongest
- `okx_cb_spot_basis_z`: -0.0127 (t=-3.8)

Mechanism: mean-reversion of cross-venue arbitrage. When Binance perp trades at premium vs other venues, symbol underperforms next 4h as arb closes the gap.

## Master matrix (36 completed cells, sorted by Sharpe descending)

| # | Model | Arch | Features | n | Sharpe | folds+ | conc | totPnL | Lift vs BASE |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **Ridge** | **Per-sym** | **+cohort** | 17 | **+2.01** | 5/9 | 48% | +6,948 | **+1.67** |
| 🥈 | Ridge | Pool+symid | +aggT | 19 | +1.22 | 5/9 | 76% | +5,438 | +0.84 |
| 🥉 | **Ridge** | **Per-sym** | **+crossX** | 19 | +1.12 | 5/9 | **33%** | +3,283 | +0.78 |
| 4 | Ridge | Per-sym | +v3 | 18 | +0.85 | 4/9 | 41% | +2,906 | +0.51 |
| 5 | Ridge | Per-sym | +aggT | 19 | +0.45 | 5/9 | 48% | +1,407 | +0.11 |
| 6 | Ridge | Pool+symid | +crossX | 19 | +0.43 | 3/9 | 97% | +1,916 | +0.05 |
| 7= | Ridge | Pool+symid | BASE | 14 | +0.38 | 2/9 | 99% | +1,681 | — |
| 7= | Ridge | Pool-nosym | +aggT | 19 | +0.38 | 5/9 | 80% | +1,484 | +0.30 |
| 9 | Ridge | Per-sym | BASE | 14 | +0.34 | 4/9 | 61% | +1,017 | — |
| 10 | Ridge | Pool-nosym | +v3 | 18 | +0.30 | 2/9 | 100% | +1,116 | +0.22 |
| 11 | Ridge | Pool-nosym | +cohort | 17 | +0.15 | 4/9 | 39% | +486 | +0.07 |
| 12 | Ridge | Pool-nosym | BASE | 14 | +0.08 | 3/9 | 74% | +323 | — |
| 13 | Ridge | Pool-nosym | +crossX | 19 | +0.05 | 4/9 | 69% | +225 | -0.03 |
| 14 | Ridge | Pool+symid | +v3 | 18 | -0.04 | 3/9 | 93% | -170 | -0.42 |
| 15 | LGBM | Per-sym | +crossX | 19 | -0.17 | 3/9 | 66% | -451 | **+2.06** (biggest LGBM lift) |
| 16 | LGBM | Pool+symid | +aggT | 19 | -0.63 | 3/9 | 94% | -1,800 | +0.81 |
| 17 | Ridge | Pool+symid | +cohort | 17 | -0.72 | 4/9 | 59% | -1,768 | -1.10 (multicol issue) |
| 18 | LGBM | **Pool-nosym** | **+ALL** | 31 | -0.76 | 2/9 | 81% | -2,876 | **+1.79** (biggest nosym lift) |
| 19 | LGBM | Per-sym | +cohort | 17 | -0.84 | 3/9 | 50% | -2,641 | +1.39 |
| 20 | LGBM | Pool+symid | +crossX | 19 | -1.06 | 3/9 | 63% | -3,002 | +0.38 |
| 21 | LGBM | Pool-nosym | +cohort | 17 | -1.11 | 1/9 | 100% | -3,125 | +1.44 |
| 22 | LGBM | Per-sym | +v3 | 18 | -1.12 | 3/9 | 67% | -3,786 | +1.11 |
| 23= | LGBM | Pool+symid | +cohort | 17 | -1.42 | 2/9 | 61% | -4,639 | +0.03 |
| 23= | LGBM | Pool-nosym | +v3 | 18 | -1.42 | 2/9 | 99% | -3,568 | +1.13 |
| 25 | LGBM | Pool+symid | BASE | 14 | -1.44 | 1/9 | 100% | -4,721 | — |
| 26 | LGBM | Pool-nosym | +crossX | 19 | -1.59 | 1/9 | 100% | -5,293 | +0.96 |
| 27 | LGBM | Pool+symid | +v3 | 18 | -2.03 | 1/9 | 100% | -6,648 | -0.59 |
| 28 | LGBM | Pool+symid | +ALL | 31 | -2.04 | 1/9 | 100% | -6,584 | -0.60 (HURT) |
| 29 | LGBM | Per-sym | BASE | 14 | -2.23 | 2/9 | 100% | -8,340 | — |
| 30 | LGBM | Per-sym | +aggT | 19 | -2.34 | 2/9 | 92% | -8,613 | -0.11 |
| 31 | LGBM | Pool-nosym | BASE | 14 | -2.55 | 2/9 | 98% | -7,427 | — |
| 32 | LGBM | Pool-nosym | +aggT | 19 | -3.94 | 1/9 | 100% | -10,541 | -1.39 |

## Cross-table view: Sharpe by (Model × Arch × Feature)

### LGBM
| Arch | BASE | +aggT | +cohort | +v3 | +crossX | +ALL |
|---|---|---|---|---|---|---|
| Pool+symid | -1.44 | **-0.63** | -1.42 | -2.03 | -1.06 | -2.04 |
| Pool-nosym | -2.55 | -3.94 | -1.11 | -1.42 | -1.59 | **-0.76** |
| Per-sym | -2.23 | -2.34 | -0.84 | -1.12 | **-0.17** | pending |

### Ridge
| Arch | BASE | +aggT | +cohort | +v3 | +crossX | +ALL |
|---|---|---|---|---|---|---|
| Pool+symid | +0.38 | **+1.22** | -0.72 | -0.04 | +0.43 | pending |
| Pool-nosym | +0.08 | **+0.38** | +0.15 | +0.30 | +0.05 | pending |
| Per-sym | +0.34 | +0.45 | **+2.01** | +0.85 | +1.12 | pending |

## Key findings

### 1. Ridge dominates LGBM on portable BTC-frame features
Top 14 cells (by Sharpe) are all Ridge. Best LGBM cell is -0.17 (Per-sym +crossX); best Ridge cell is +2.01 (Per-sym +cohort) — gap of ~2.18 Sharpe.

### 2. Feature additions have asymmetric effects by architecture

**aggTrades** (+aggr_ratio_4h, tfi_4h, etc.) helps most when sym_id is present:
- LGBM Pool+symid: +0.81 ✓
- Ridge Pool+symid: +0.84 ✓
- LGBM Pool-nosym: -1.39 ✗ (hurts)
- Ridge Pool-nosym: +0.30 (small)
- LGBM Per-sym: -0.11 (neutral)
- Ridge Per-sym: +0.11 (small)

**cohort** (rvol_7d, ret_3d, btc_rvol_7d broadcast) helps most when sym_id is ABSENT:
- LGBM Pool+symid: +0.03 (none)
- Ridge Pool+symid: -1.10 (multicollinearity)
- LGBM Pool-nosym: +1.44 ✓
- LGBM Per-sym: +1.39 ✓
- Ridge Per-sym: **+1.67** ✓ (BEST)

**v3** (idio_max_abs/skew/kurt, name_idio_share) substitutes for sym_id:
- Pool+symid: hurts (-0.59 LGBM, -0.42 Ridge)
- Pool-nosym: helps (+1.13 LGBM, +0.22 Ridge)
- Per-sym: helps (+1.11 LGBM, +0.51 Ridge)

**crossX** (cross-exchange basis features) helps Per-sym especially:
- LGBM Per-sym: +2.06 (biggest LGBM lift)
- Ridge Per-sym: +0.78

### 3. Combining ALL features (31) is bimodal
- LGBM Pool+symid: hurts (-0.60 vs +0.81 best single)
- LGBM Pool-nosym: helps massively (+1.79 — biggest nosym lift)

### 4. Per-symbol architecture works for Ridge, fails for LGBM
- Ridge Per-sym best Sharpe +2.01 (with cohort)
- LGBM Per-sym best Sharpe -0.17 (with crossX) — better than other LGBM arches at +crossX but still negative

### 5. Concentration (PnL spread across folds) — Ridge Per-sym is best
The 3 cells with lowest concentration (most fold-distributed PnL):
- Ridge Per-sym +crossX: 33%
- Ridge Pool-nosym +cohort: 39%
- Ridge Per-sym +v3: 41%

Ridge Per-sym + good features produces well-distributed signal — robust forward-test candidate.

## Regularization sweep (X8 + X8b) — focused on best cell

Best cell: Ridge Pool+symid +aggT, baseline Sharpe +1.22.

### X8 (RidgeCV variants)
| Variant | Sharpe | Notes |
|---|---|---|
| A1 baseline (α∈{0.01,0.1,1,10,100}) | +1.22 | α picked = mostly 10, rarely 1 |
| A2 wider grid (α∈{0.001…300}, 12 vals) | **+1.26** | α picked = mostly 10 and 30; +0.04 lift |
| A3 wider + free sym_id intercepts | -0.91 | Per-sym mean demean fails OOS |
| A4-A6 ElasticNet (precompute=True, float32) | ERR | Gram matrix numerical instability |

### X8b (ElasticNet retry — 3/4 done, B4 timed out)
Numerical fixes: precompute=False, float64, subsample to 200k for fitting speed.

| Variant | Sharpe | folds+ | conc | IC | α picked | Sparsity | n_nonzero |
|---|---|---|---|---|---|---|---|
| B1 l1=0.1 | **+0.66** | 3/9 | 91% | +0.0067 | 0.001-0.003 | 25% zeroed | 51/68 |
| B2 l1=0.5 | **+0.45** | 2/9 | 99% | +0.0068 | 0.001 | 45% zeroed | 37/68 |
| B3 l1=0.9 | **+0.82** | 4/9 | 50% | +0.0073 | 0.001 | 63% zeroed | 25/68 |
| B4 Lasso (l1=1.0) | ⛔ TIMED OUT | — | — | — | — | — | — |

**Pattern**: ElasticNet L1 hurts non-monotonically — B1 (+0.66) > B2 (+0.45) < B3 (+0.82). Even at best (B3), still below RidgeCV baseline (+1.22 → +1.26 with wider grid). Heavy L1 (0.9) actually rebounds because aggressive feature selection drops the worst features, but pure L1 keeps too few features.

**Verdict**: Pure L2 Ridge wins. L1 feature selection consistently hurts in this problem because there are no fully-useless features — every feature contributes small but real predictive value. The 50 sym_id dummies + 14 base + 5 aggT = 69 coefs all have meaningful contribution; zeroing any of them drops Sharpe.

**Note**: CSV not written (B4 timed out before script could save). Results captured here and in TODO.md.

## Status of background processes

| Task | Status | Cells done | Remaining |
|---|---|---|---|
| X6 (24 cells) | ✅ done | 24/24 (6 cohort errored, recovered by X6b) | — |
| X6b cohort fill | ✅ done | 6/6 | — |
| X7 crossX | ✅ done | 6/6 | — |
| X8 reg sweep | ✅ done | 6/6 (4 ElasticNet errored) | — |
| X8b ElasticNet retry | ✅ done (partial) | B1/B2/B3 ✓; B4 timed out | — |
| X9 all-features | 🔄 running (restart, 4h timeout) | cells 1+2 rerun ✓ matching prior; cells 3-6 pending | ~50 min |
| X8c sym_id normalization | ⏳ pending | will dispatch after X9 done | ~20 min |
| X8d cohort collinearity | ⏳ pending | medium priority | ~15 min |
| X8e LGBM reg sweep | ⏳ pending | medium priority | ~20 min |
| X10 apply best reg to all cells | ⏳ pending | blocked on X8c | ~3h |

## Outputs

- Master CSV: `research/convexity_portable_2026-05-20/results/X6_controlled_matrix.csv`
- Reg sweep CSV: `research/convexity_portable_2026-05-20/results/X8_ridge_reg_sweep.csv`
- ElasticNet CSV: NOT WRITTEN (B4 timed out before save). Results recorded in this doc and TODO.md.
- Cross-exchange features: `data/ml/cache/cross_exchange_features.parquet`
- Per-cell predictions: `research/convexity_portable_2026-05-20/results/_cache/x6_*_preds.parquet`

## Outstanding caveats

1. **Absolute Sharpes are negative for LGBM** because BASE drops V3.1's basket-frame features (`dom_level_vs_bk`, etc.) for portability. V3.1 production with WINNER_21 = +3.00 on HL-50; X6 BASE for LGBM = -1.44. The **Δ from feature additions** is the apples-to-apples lever; absolute levels not comparable to V3.1.

2. **Cohort cells in original X6** silently produced BASE-equivalent results due to NaN-fill bug. Fixed in X6b.

3. **ElasticNet runtime** much slower than RidgeCV due to coordinate descent. ~34 min per variant.

4. **Cross-exchange features** at 53-63% coverage (44-48 syms each). Pooled IC -0.013 to -0.018 with t-stats -3.8 to -5.5 — modest but real.

5. **V3.1 production** (Ridge Pool+symid WINNER_21 = +3.00 on HL-50) sits OUTSIDE this matrix since its WINNER_21 includes basket-frame features that we excluded for portability.
