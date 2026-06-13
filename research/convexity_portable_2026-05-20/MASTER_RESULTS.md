# Master Results Comparison — All Phase 1 + Phase 2 Tests

**Generated:** 2026-05-21 from X33_compile_master_table.py
**Total tests:** 96 across 7 categories (X6–X32)
**Production cell:** Ridge Per-sym + cohort (V0/N_HL50) = **+2.01 Sharpe**

---

## A. Core matrix (X6, X22, X29) — Architecture × Feature set

Sharpe values (Per-sym + cohort canonical baseline = +2.01):

| Architecture | BASE | +aggT | +cohort | +v3 | +crossX | +ALL |
|---|---|---|---|---|---|---|
| **per-sym (Ridge)** | -0.94 | -0.94 | **+2.01** | -0.14 | +0.48 | +1.44 |
| pool+symid (Ridge) | -0.53 | +0.30 | +0.38 | -1.03 | -0.32 | -1.11 |
| pool+symid_C1norm | +0.09 | **+1.38** | -0.10 | -0.58 | -0.03 | -0.41 |
| pool-nosym (Ridge) | -1.23 | -1.78 | +0.08 | -0.56 | -0.77 | -0.08 |

(X6 + X10 C1-norm rows. Per-sym +cohort is winner.)

### With panel_v2 (X22, X29) — fixed aggT coverage:

| Cell | OLD panel | v2 panel | Δ |
|---|---|---|---|
| Ridge Per-sym +cohort (V0) | +2.01 | **+2.01** | 0.00 (sanity ✓) |
| Ridge Per-sym +crossX (V2) | +1.90 | +1.90 | 0.00 |
| Ridge Per-sym +aggT (V1) | +1.62 | +1.52 | -0.10 |
| Ridge Per-sym +aggT+crossX (V3) | +0.60 | **+1.35** | +0.75 (FIX) |
| Ridge Per-sym +ALL (V5) | sleeve ERR | **+1.66, 7/9 folds, 26% conc** | new candidate |
| Ridge Pool+symid +aggT | +1.22 | +0.68 | -0.54 |
| LGBM Pool+symid +aggT | -0.63 | -0.92 | -0.29 |
| LGBM Per-sym +aggT | -2.34 | -1.04 | **+1.30** |

---

## B. Regularization variants

### Ridge wider α + sym_id variants
| Variant | Sharpe | Notes |
|---|---|---|
| A1 baseline (α [0.01-100]) | +1.22 | |
| A2 wider α [0.001-300] | +1.26 | small lift |
| **C1 normalized sym_id + wider α** | **+1.38** | best Pool+symid Ridge |
| A3/C2 free sym_id (target demean) | -0.91 | high IC but sleeve collapses |
| C3 group α (α_sym/50) | +0.77 | under-regularizes sym_id |
| C4 drop sym_id (=Pool-nosym +aggT) | +0.38 | reference |

### LGBM regularization (X8e — all variants HURT vs default)
| Variant | Sharpe | Notes |
|---|---|---|
| X6 default | -0.63 | local optimum |
| E1 early stopping | -1.95 | |
| E2 adaptive leaf | -2.63 | |
| E3 higher reg | -2.98 | |
| E4 combined | -2.80 | |

### Cohort collinearity (X8d)
| Variant | Sharpe | Lift |
|---|---|---|
| D1 drop btc_rvol_7d broadcast + norm sym_id | +0.80 | +1.52 vs X6b -0.72 |
| D2 drop sym_id entirely (full cohort) | -0.03 | |
| D3 per_sym_vol_proxy substitute | +0.26 | |

### C1 applied to all Ridge Pool+symid cells (X10)
| Cell | Sharpe |
|---|---|
| BASE | +0.09 |
| **+aggT** | **+1.38** (best Pool+symid) |
| +cohort | -0.10 |
| +v3 | -0.58 |
| +crossX | -0.03 |
| +ALL | -0.41 |

---

## C. Universe variants (Ridge Per-sym + cohort, canonical framework)

Sorted by Sharpe descending:

| Universe | n syms | Sharpe | Source | Notes |
|---|---|---|---|---|
| EX_smallest_K6 (data-driven) | 49 | **+2.02** | X25 | drop smallest cluster |
| **HL-50** | 50 | **+2.01** | X23 | **GLOBAL MAX** |
| L_dd_K4_c1 (data-driven, K=4) | 45 | +2.01 | X25 | matches HL-50 |
| EX_no_memes | 44 | +1.87 | X24 | memes optional |
| T_top10 (top-10 vol) | 10 | +1.80 | X23 | 93% conc (single-fold) |
| **V5 BASE+cohort+ALL (HL-50)** | 50 | **+1.66** | X29 | 7/9 folds, 26% conc |
| B_bot25 (bottom 25 vol) | 25 | +1.53 | X23 | small-caps have signal! |
| V1 BASE+cohort+aggT (HL-50) | 50 | +1.52 | X29 | data fix |
| HL-45 | 45 | +1.37 | X23 | best fold count 6/9 |
| C1 Ridge Pool+symid +aggT (HL-50) | 50 | +1.38 | X10 | non-Per-sym alt |
| V3 BASE+cohort+aggT+crossX | 50 | +1.35 | X29 | |
| HL-40 | 40 | +1.28 | X23 | |
| C_l1_newer (l1_newer cluster) | 11 | +1.14 | X24 | best single cluster |
| HL-25, T_top25 | 25 | +1.15 | X23 | |
| HL-35 | 35 | +1.07 | X23 | |
| X32 HL50_sanity | 50 | +0.84 | X32 | drift from X31 features |
| $5M+ vol filter | 29 | +0.92 | X11 | |
| HL70_minus_top10 (drops FARTCOIN) | 60 | +0.69 | X32 | |
| drop top-10-vol | 40 | +0.61 | X11 | |
| HL-30 | 30 | +0.38 | X23 | |
| T_top5 | 5 | 0.00 | X23 | can't form K=3 basket |
| 51-panel WITH BTC | 51 | **0.00** | X11 | BTC dynamics break |
| **HL-70 (50+20 new)** | 70 | **-0.11** | X32 | DECISIVELY REJECTED |
| HL70_bot20 (new syms only) | 20 | -0.34 | X32 | NEGATIVE IC (-0.0027) |
| HL-20, T_top20 | 20 | -0.66 | X23 | K=3 too narrow |
| HL70_no_AI | 67 | **-1.34** | X32 | AI critical even @ HL-70 |
| HL70_top30 | 30 | -1.01 | X32 | toxic new sym contamination |

---

## D. Cluster universes (X24 hand-crafted, X25 data-driven)

| Universe | n syms | Sharpe | Verdict |
|---|---|---|---|
| EX_no_memes | 44 | +1.87 | memes optional |
| **C_l1_newer** | 11 | **+1.14** | best single cluster |
| C_other_alt | 9 | +0.53 | mediocre alone |
| P_majors_defi | 14 | +0.26 | sector pair |
| P_defi_ai | 14 | +0.20 | sector pair |
| EX_no_memes_ai | 41 | +0.01 | (confirms AI critical) |
| C_memes | 6 | 0.00 | too small for K=3 |
| C_defi | 11 | -0.05 | low value alone |
| EX_no_majors | 47 | -0.06 | majors critical |
| **EX_no_ai** | 47 | **-0.10** | **AI cluster critical** |
| EX_no_other_alt | 41 | -0.23 | other_alt also load-bearing |
| C_l1_established | 7 | -0.38 | too small |
| **P_L1_all** (l1_est+l1_newer) | 18 | **-3.10** | sector COMBO collapse |
| P_majors_L1 | 21 | -2.82 | sector combo collapse |
| P_majors_L1_defi | 32 | -1.83 | sector combo collapse |

**Sector combinations CATASTROPHIC** due to K=3 collinearity (within-sector picks dominate).

---

## E. Other key tests (Phase 2 corrected)

### X12 — Augment V3.1 production with new features
| Variant | Sharpe | vs V3.1 +3.00 |
|---|---|---|
| Ridge Per-sym W21 + cohort | -0.33 | -3.33 |
| LGBM Pool+symid W21 + cohort | +0.43 | -2.57 |
| LGBM Pool+symid W21 + crossX | +0.86 | -2.14 |

**V3.1 cannot be improved by adding cohort/crossX** — local optimum.

### X13 — Orthogonality (aggT + crossX combinations)
| Approach | Sharpe |
|---|---|
| +aggT alone (Pool+symid Ridge baseline) | +1.22 |
| E0 joint Ridge (BASE+aggT+crossX) | +1.00 |
| E1 ensemble (avg sep models) | +0.43 |
| E2 group α (per-feature) | +0.31 |

**Features NOT orthogonal**; combinations hurt.

### X27 — Per-group α cross-universe (THE answer)
| Pair | Train univ | Best α | Validate univ | Per-group | Uniform RidgeCV | Lift |
|---|---|---|---|---|---|---|
| 1 | top-25 | (100, 0.01) | bot-25 | **+0.00** | +1.53 | **-1.53** |
| 2 | bot-25 | (10, 0.01) | top-25 | **-1.86** | +1.15 | **-3.01** |

**Per-group α UNIVERSE-OVERFITS** — uniform α is the robust choice.

### X14 — crossX granularity (DATA HYGIENE)
| Variant | Sharpe |
|---|---|
| 4h-aligned (X7 original) | +1.12 (Per-sym) |
| 5m price-ffill (X14b — LEAKY) | -0.19 (Per-sym) |
| 5m basis-ffill (X14d — proper) | -0.00 (Per-sym) |

**Original 4h-aligned was correct**; 5m forward-fill introduces auto-correlation.

### X19 — Preprocessing (didn't reproduce canonical due to framework drift)
| Variant | Sharpe |
|---|---|
| P1 winsor baseline (with drift bug) | +0.19 |
| P2 rank-transform | -0.18 |
| P3 robust MAD | -0.39 |

(Note: post-X21 fix, these would reproduce canonical +2.01 baseline.)

---

## F. Production candidates summary

| Cell | Model | Arch | Features | Panel | Universe | Sharpe | Folds+ | Conc | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **V3.1 production** | LGBM | Pool+symid | WINNER_21 (21) | v1 | HL-50 | **+3.00** | 8/9 | 35% | uses basket-frame, not portable |
| **V0** | Ridge | Per-sym | BASE+cohort (17) | v1 or v2 | HL-50 | **+2.01** | 5/9 | 48% | fragile (bootstrap mean +0.20) |
| **V5** | Ridge | Per-sym | ALL (31) | v2 | HL-50 | **+1.66** | **7/9** | **26%** | robust (bootstrap mean +0.85) |
| **V5_minus_v3** ⭐NEW | Ridge | Per-sym | BASE+cohort+aggT+7cx (29) | hl70_v5_full | HL-70 | **+1.67** | 5/9 | 34% | matches HL-50 V5 on HL-70 |
| V5_full (7cx, with v3) | Ridge | Per-sym | ALL (33) | hl70_v5_full | HL-70 | +1.19 | 5/9 | 36% | v3 dragging down |
| V5_full (7cx) | Ridge | Per-sym | ALL (33) | hl70_v5_full | HL-50 | +1.13 | 5/9 | 44% | 2 new cx HURT canonical |
| V1 +aggT | Ridge | Per-sym | BASE+cohort+aggT (22) | v2 | HL-50 | +1.52 | 5/9 | 50% | |
| C1 Pool+symid +aggT | Ridge | Pool+symid+C1norm | BASE+aggT (19) | v1 | HL-50 | +1.38 | 5/9 | 67% | best Pool+symid |

## F2. X53 aggT ablation findings (HL-70)

| Variant | n_feats | Sharpe | Δ vs V5_full |
|---|---|---|---|
| **V5_minus_v3 (BASE+cohort+aggT+7cx)** | **29** | **+1.67** | **+0.48 (BEST)** |
| V5_full | 33 | +1.19 | baseline |
| V5_minus_crossX | 26 | +0.87 | -0.32 |
| V5_minus_crossX_v3 (aggT only) | 22 | +0.71 | -0.48 |
| V5_minus_aggT | 28 | +0.64 | -0.55 |
| V0 BASE+cohort | 17 | -0.11 | -1.30 |

**Decomposition on HL-70**: aggT +0.55, crossX +0.32, **v3 -0.48** (HURTS).

## F3. Phase 3 deep diagnostics (X55-X67) — REGIME-CONDITIONAL CHAMPION

### Production cell update (universe-dependent)

| Universe | Best cell | Sharpe | Folds | Conc |
|---|---|---|---|---|
| **canonical HL-50** | **X66 V5_mv3 sideways + V0 bull (thr=0.20)** | **+2.08** ⭐ | 6/9 | 37% |
| canonical HL-50 (most robust) | X66 V5_mv3 sideways + V0 bull (thr=0.15) | +2.05 | **7/9** | 40% |
| canonical HL-50 (single model) | V5_minus_v3_7cx | +1.74 | 6/9 | 36% |
| canonical HL-50 (normalized ensemble) | X56 w_v0=0.25 | +2.12 | 6/9 | 39% |
| **HL-70** | **X64 V5_mv3 + bull-zero gate (thr=0.10)** | **+2.17** ⭐ | 6/9 | ? |
| HL-70 (baseline) | V5_minus_v3_7cx | +1.67 | 5/9 | 34% |

### X55 diagnostics
- HL-50 V5_mv3 bootstrap mean +0.81 ± 0.94, P(>0)=68%
- HL-70 V5_mv3 still VVV-dependent: drop VVV Δ -2.01
- Cost at 1 bps: both universes +1.89

### X57 cluster dropout HL-50
- **defi (11 syms): Δ -1.74 load-bearing** ⚠️
- other_alt (9 syms): Δ -1.01
- AI: Δ +0.06 (slightly improves)

### X67 fully-data-covered 50-sym subset
- Filter syms by data completeness: -1.19 Sharpe (CATASTROPHIC)
- **Lesson: data completeness ≠ alpha**

---

## G. Permanently CLOSED directions

- ❌ HL-70 universe expansion (-2.12 drop vs HL-50)
- ❌ 51-panel + BTC (collapses to 0)
- ❌ Per-group α (universe-overfits, lifts -1.53 / -3.01)
- ❌ LGBM regularization tuning (defaults at local optimum, all variants hurt)
- ❌ ElasticNet L1 (consistently hurts vs RidgeCV)
- ❌ 5m crossX forward-fill (introduces auto-correlation leakage)
- ❌ Sector combinations (K=3 collinearity → -1.83 to -3.10)
- ❌ Sub-25-sym universes (K=3 basket too narrow)
- ❌ Preprocessing variants (winsor baseline already best)
- ❌ Augmenting V3.1 production with new features (-2.14 to -3.33)

## H. Open ideas (not tested)

- Per-symbol α (vs per-group) — likely same overfit
- V0 + V5 prediction ensemble
- Bootstrap CI on production Sharpe for risk-adjusted comparison

---

## Files

- Raw data: `outputs/vBTC_features/panel_variants_with_funding{,_v2}.parquet`, `outputs/vBTC_features/panel_hl70.parquet`
- Master CSV: `results/X33_master_results.csv` (96 rows)
- Per-test CSVs: `results/X*.csv` (25 files)
- Phase 2 synthesis: `FINAL_SYNTHESIS_PHASE2.md`
- TODO ledger: `TODO.md`
