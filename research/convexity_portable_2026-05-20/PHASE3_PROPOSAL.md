# Phase 3 Proposal — Extensive Tests After System Audit

**Created:** 2026-05-21 (post X34/X35 audit)

## Critical issues identified by audit

### 🚨 V0 (production candidate +2.01) is FRAGILE

X35a/X35b revealed:
- **Drop VVV → -0.49 Sharpe** (single sym carries entire alpha)
- Drop TAO → +0.61 (single sym worth -1.40)
- Folds 1, 3, 5, 8 each independently contribute ≥1.9 Sharpe (drop any → collapse)
- 4 of 9 folds are catastrophic-on-removal

**V0 = Single-sym + multi-fold leveraged bet**. Sharpe number is real but fragility is severe.

### ✅ V5 (kitchen sink +1.66) is MORE robust

- Drop ALL AI cluster → **+1.96** (improves!)
- Drop VVV → +1.24 (vs V0's -0.49)
- Drop fold 3 → +1.80 (helps), drop fold 6 → +1.87 (helps)
- Only 2 catastrophic folds (vs V0's 4)

**V5 = genuine diversification across syms and folds, despite -0.35 Sharpe headline**.

### Sample period concerns
- 405 days (13.5 months) — short for ML
- 9 folds × 45 days OOS each
- Mostly bull regime
- Some syms have <30% sample coverage (ASTER, HYPE in early folds)

## Phase 3 Extensive Tests — Priority Ranked

### TIER 1: Validate V5 as production candidate (HIGH priority)

**X37 — Cost sensitivity** (already prepared, run next)
- V0/V5 Sharpe at cost ∈ {1, 2, 3, 4.5, 6, 9, 12} bps/leg
- Confirms V0 deteriorates faster than V5 with higher cost (more single-sym churn)
- ~10 min

**X41 — V0/V5 ensemble**
- Average V0 + V5 predictions, run sleeve
- If ensemble > both individual → consider as production
- ~5 min

**X42 — V5 cluster dependence**
- For V5: drop each cluster (major, l1, defi, memes, other_alt)
- Confirms V5 doesn't have hidden cluster dependency
- ~10 min

### TIER 2: Reduce sample dependency

**X43 — Time-block bootstrap**
- Resample fold predictions at the cycle level (not fold level)
- Get tighter Sharpe CI from millions of cycles vs 9 folds
- ~5 min

**X44 — Embargo sensitivity**
- Re-run with 3-day, 7-day, 14-day embargo
- Verify 1-day embargo isn't leaking
- ~30 min (full re-train)

**X45 — Half-sample test**
- Train on first 6 months, test on last 7 months
- Single OOS to compare to 9-fold result
- ~10 min

### TIER 3: Production safety nets

**X46 — Sym-dropout simulation**
- For V0 and V5, simulate dropping each sym one at a time
- Plot distribution of Sharpe under sym-loss
- ~30 min

**X47 — V5 weight pruning**
- V5 has 31 features. Identify minimum subset that preserves +1.66 Sharpe
- Likely something like BASE + cohort + 3-5 crossX/aggT features
- ~20 min

### TIER 4: Architectural exploration (LOWER priority)

**X48 — Stacked ensemble (Ridge V0 + Ridge V5 + LGBM Per-sym)**
- Train meta-Ridge on stacked predictions
- ~30 min

**X49 — Per-sym alpha selection (vs uniform RidgeCV)**
- Each sym picks own α independently
- ~20 min

**X50 — Sample weighting (recent > old)**
- Linear or exponential time-decay weights in Ridge
- ~15 min

## Proposed dispatch order

After X35 completes:
1. X37 cost sensitivity (immediate)
2. X41 V0+V5 ensemble (quick win)
3. X42 V5 cluster dropout (verify V5 doesn't hide dependency)
4. X43 time-block bootstrap (better CI)
5. X44 embargo sensitivity (validate no leak)
6. X45 half-sample (independent validation)
7. X46 sym-dropout sim (production risk profile)
8. X47 V5 weight pruning (simplify production)

Total ETA: ~2.5 hours of background compute.
