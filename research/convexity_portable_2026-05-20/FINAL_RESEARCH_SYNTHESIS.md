# Final Research Synthesis — Comparison Matrix Project

**Completed:** 2026-05-20 (all 19 tasks done)
**Universe:** HL-50 (51-panel minus BTC, 50 HL-tradeable syms)
**Sample:** 2025-04-01 → 2026-05-06, 9-fold walk-forward expanding

---

## 🏆 Two best strategies (both validated, distinct designs)

| Strategy | Sharpe | Folds+ | Conc | Notes |
|---|---|---|---|---|
| **V3.1 production** (LGBM Pool+symid WINNER_21) | **+3.00** | 8/9 | 35% | Production reference; relies on basket-frame features |
| **Ridge Per-sym + cohort** (BTC-frame BASE) | **+2.01** | 5/9 | 48% | Best portable alternative; 1-day-vol robust |

These are NOT additive — combining their features INTO each other HURTS both (X12 V2/V3).

---

## 📊 Complete matrix landscape (HL-50, 45+ cells run)

### Core 36-cell matrix (6 architectures × 6 feature sets)

Best per row (feature set), sorted by best Sharpe:

| Feature set | Best cell | Sharpe | folds | notes |
|---|---|---|---|---|
| **+cohort** | Ridge Per-sym | **+2.01** | 5/9 | 48% conc — strongest single addition |
| **+aggT** | Ridge Pool+symid | +1.22 | 5/9 | best with sym_id present |
| **+crossX** | Ridge Per-sym | +1.12 | 5/9 | 33% conc — best concentration |
| **+v3** (idio_*) | Ridge Per-sym | +0.85 | 4/9 | replaces sym_id role when needed |
| **BASE** (14 portable) | Ridge Pool+symid | +0.38 | 2/9 | minimum viable |
| **+ALL** (31 features) | Ridge Per-sym | +1.44 | 5/9 | combining helps Per-sym, hurts Pool+symid |

---

## 🔬 Architectural learnings

### 1. Ridge dominates LGBM on portable (BTC-frame only) features
Top 14 cells in matrix are all Ridge; best LGBM cell is -0.17. **~2 Sharpe gap.** LGBM needs basket-frame features (only present in V3.1's WINNER_21) to achieve +3.00.

### 2. Per-symbol Ridge is the surprise winner
3 of top 3 cells are Ridge Per-sym (+cohort +2.01, +crossX +1.12, +v3 +0.85). LGBM Per-sym fails (-2.23 BASE) — too few samples per symbol for trees.

### 3. Feature-architecture interactions are asymmetric
- **aggT** helps when sym_id present (LGBM/Ridge Pool+symid: +0.81, +0.84 lift)
- **cohort** helps when sym_id absent OR Per-sym (lift +1.39 to +1.67); HURTS Pool+symid Ridge (multicollinearity with sym_id intercepts)
- **v3 idio_*** substitutes for sym_id (helps Pool-nosym/Per-sym, hurts Pool+symid)
- **crossX** most useful for Per-sym (largest LGBM lift +2.06)
- **+ALL** combining: helps Pool-nosym/Per-sym (more info matters), hurts Pool+symid (noise dominates)

### 4. Feature granularity must match evaluation granularity
X14 lesson: forward-filling 4h crossX features to 5m granularity HURT Sharpe (-0.85 to -1.49) despite higher IC (+0.05 to +0.12 vs original +0.007). The V3.1 sleeve evaluates at 4h cadence; densifying features injects auto-correlation that Ridge over-weights. **Original 4h-aligned was correct**.

---

## 🔬 Regularization findings

### Ridge
- **C1 (normalized sym_id dummies + wider α grid)** is feature-set dependent: helps +aggT (+0.16), +cohort (+0.62 — fixes multicol), +ALL (+0.70). Hurts BASE (-0.29), +v3 (-0.54), +crossX (-0.46).
- **Free sym_id intercepts via target demean** COLLAPSES (-0.91) — train means don't generalize OOS
- **ElasticNet L1** consistently hurts — no truly redundant features to zero out
- **Group α** (per-group penalty) offers no advantage when all groups want α=300

### LGBM
- **X6 defaults at local optimum** for LGBM Pool+symid +aggT (-0.63)
- **All 4 regularization variants (early stop, adaptive leaf, higher reg, combined) HURT** by -1.32 to -2.35
- Signal lives in prediction TAILS; regularization erodes extreme predictions → K=3 selects "average" syms → worse

---

## 🔬 Orthogonality / ensemble findings (X13)

**aggT + crossX are NOT orthogonal** in PnL space:
- E0 joint Ridge (BASE+aggT+crossX): +1.00 — worse than +aggT alone +1.22
- E1 ensemble (avg of separate Ridges): +0.43 — dilutes to crossX level
- E2 group α: +0.31 — no improvement

**Conclusion**: use single best feature group, not combinations. crossX coverage (53-63%) introduces NaN-as-zero noise that dilutes aggT's stronger signal.

---

## 🔬 Cohort feature collinearity (X8d)

**btc_rvol_7d broadcast was the +cohort cell problem**:
- D1 (drop broadcast, keep per-sym rvol_7d/ret_3d, normalize sym_id) = +0.80 (lift +1.52 vs X6b)
- Confirmed: broadcast features collinear with sym_id one-hot dummies → Ridge can't fit cleanly

---

## 🔬 Universe stress (X11)

| Universe | n syms | Sharpe | vs HL-50 +2.01 |
|---|---|---|---|
| **HL-50** (baseline) | 50 | **+2.01** | sweet spot |
| Drop top-10-vol | 40 | +0.61 | -1.40 (top-vol syms carry alpha) |
| $5M+ vol filter | 29 | +0.92 | -1.09 |
| **51-panel WITH BTC** | 51 | **+0.00** | **-2.01 collapses** (BTC dynamics ≠ alts) |

**HL-50 is the sweet spot. Adding BTC catastrophic; reducing universe loses top contributors.**

---

## 🔬 V3.1 production augmentation (X12)

Cannot improve V3.1 by ADDING our new features:
- V3.1 baseline (WINNER_21): +3.00
- V3.1 + cohort: +0.43 (multicollinearity)
- V3.1 + crossX: +0.86

**V3.1 (LGBM Pool+symid WINNER_21) is a local optimum.** New features work in different architectures (Ridge Per-sym +cohort = +2.01), not as production extensions.

---

## 🎯 Strategic takeaway

The matrix shows **two distinct alpha pathways**:

1. **V3.1 production pathway**: LGBM Pool+symid with basket-frame features. Sharpe +3.00. Hard to improve.
2. **Portable Ridge pathway**: Ridge Per-sym + cohort. Sharpe +2.01. Uses only BTC-frame features (portable across universes if exchange data updates).

**For deployment**: V3.1 is production; the Ridge Per-sym is a credible portable alternative if V3.1 has issues (universe drift, etc.).

**Cross-exchange data (OKX + Coinbase)**: collected and integrated, but adds modest signal at best (~+0.78 lift for Per-sym Ridge with original 4h-aligned features). NOT a game-changer.

---

## 📁 Outputs

- `results/X6_controlled_matrix.csv` — master 45+ cell matrix
- `results/X8_*_results.csv` — regularization sweeps
- `results/X10_c1_norm_symid_results.csv` — C1 applied to all
- `results/X11_universe_stress.csv` — universe sensitivity
- `results/X12_apply_to_v31.csv` — V3.1 augmentation tests
- `results/X13_ensemble_vs_groupalpha.csv` — orthogonality
- `results/X14b/d_*.csv` — crossX granularity
- `results/X14d_basis_ffill_rerun.csv`
- `data/ml/cache/cross_exchange_features.parquet` — 4h-aligned crossX (USE THIS)
- `data/ml/cache/cross_exchange_features_5m.parquet` — 5m (LEAKAGE, do not use)
- `data/ml/cache/cross_exchange_features_5m_v2.parquet` — 5m basis-ffill (worse, do not use)

---

## ✅ All tasks completed (19/19)

#23 X6 matrix, #24 OKX collect, #25 X7 crossX, #26 X6b cohort fill, #27 Coinbase collect, #28 X8 reg sweep, #29 X9 all features, #30 X8b ElasticNet, #31 X8c sym_id norm, #32 X8d cohort collinearity, #33 X8e LGBM reg, #34 X10 apply C1, #35 X11 universe stress, #36 X12 V3.1 apply, #37 X13 orthogonality, #38 X14 crossX 5m, #39 X14b rerun 5m, #40 X14c basis-ffill, #41 X14d rerun v2
