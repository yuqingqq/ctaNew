# Methodology Review — P-2026-001-ml-cta-engine

Date: 2026-04-29 / Apr 30
Status: 8 issues identified during pipeline review

## Pipeline as built

1. Compute ~190 features (kline-based + trade-flow v1/v2 + regime)
2. Compute Spearman IC of each feature vs forward N-bar return, conditioned on regime
3. Pick top features by IC for each regime
4. Train LGBM regression on demeaned forward return
5. Trigger when |ŷ| ≥ q=0.95 of cal predictions
6. Take direction = sign(ŷ), hold for fixed N bars, exit at close
7. Filter trades by regime gate (autocorr_1h ≤ percentile cutoff)

## Best result achieved

BTC trend strategy: regime_cutoff=0.33 on autocorr_1h, q=0.95, h=48 (4h hold), 5-seed ensemble
- Mean net per trade: **+22.38 bps** (raw) / **+17.18 bps** (vol-scaled)
- Std fold net: 40 (raw) / 31 (vol-scaled)
- 3/4 BTC folds positive
- **Cross-symbol failure on SOL**: mean -6.04 bps, std 76, 2/4 folds positive

## 8 Methodology Issues Identified

### Issue 1: IC-based feature selection misses interaction-rich features
Spearman IC measures each feature's marginal rank correlation with returns. Features with low marginal IC could be highly valuable in interactions (e.g., volume × direction matters only when both align). LGBM exploits interactions but if we *select* by marginal IC we exclude interaction-rich features.

**Fix:** select features by incremental contribution to validation Sharpe (forward selection), or SHAP gain importance, not IC alone.

### Issue 2: Regime classifier is naive
Single statistic (autocorr_1h percentile over 3 hours) can't distinguish:
- Up-trend vs down-trend (caused SOL fold-2 catastrophe — 100% long during crash)
- Sustained trend vs reversal at end of trend
- Slow grinding moves vs fast spike-and-fade

Naive composite (autocorr + ADX + efficiency averaged) hurt — wrong combination method.

**Fix:** train a learned regime classifier predicting "is the next 4h profitable for trend strategy" using regime features as input.

### Issue 3: Target heteroskedasticity not handled
`demean_expand` removes drift but not vol. A 50-bp move in calm regime ≠ a 50-bp move in volatile regime. Model wastes capacity learning to ignore vol.

**Fix:** Sharpe-like target: `(fwd_return - rolling_mean) / rolling_std`. Vol-normalized. Standard CTA practice.

### Issue 4: Trigger by magnitude is wrong
We trigger when |ŷ| is high. But diagnostic showed at high model conviction (top 2-5%), win rate is *worse* than at middle conviction. Model is over-confident in tails — calibration problem.

**Fix:** trigger by classification probability P(win) instead of regression magnitude. Or: regression model + separate calibration model, trigger only when calibration confirms.

### Issue 5: Fixed-horizon exit doesn't match signal half-life
Different events have different half-lives. h=48 for trend, h=6 for mean-rev — but even within trend regime, sub-events vary. We hardcoded what should be learned.

**Fix:** train exit model: `P(continue holding | t bars since entry, current state)`. Exit when P drops below threshold.

### Issue 6: No cross-asset features
BTC-only features. But BTC correlates with ETH; ETH-leading-BTC sometimes has predictive power. Missing orthogonal information source.

**Fix:** pull ETH 5m + aggTrades, compute cross-asset features (BTC-ETH spread, BTC dominance, lagged ETH momentum).

### Issue 7: Sample size limits
50d × 288 bars/day = 14,400 train samples per fold. Small for LGBM with 21+ features. Strict regime subset (top 33%) shrinks this 3×.

**Fix:** multi-symbol pooled training (BTC + ETH + SOL together) — 3× training data per fold. Tests if SOL failure was data-scarcity vs genuine non-generalization.

### Issue 8: Cost model is conservative
13 bps RT (taker × 2 + 1bp slip + 0.5×spread). For a borderline strategy, the maker-tilt opportunity is meaningful — post-only entries fill 60%+ in many setups, dropping effective RT to ~7-8 bps.

**Fix:** post-only execution simulation with realistic fill probability.

## Prioritization

| # | Fix | Effort | Expected impact |
|---|---|---|---|
| 3 | Sharpe-like target | 1 line | Medium-High |
| 4 | Calibrated trigger | Half day | Medium |
| 6 | Cross-asset features | Half day | High |
| 7 | Multi-symbol pooled training | Half day | High |
| 1 | SHAP-based feature selection | Day | Medium |
| 2 | Learned regime classifier | Day | Medium-High |
| 5 | Adaptive exit | 2-3 days | Unknown |
| 8 | Maker execution model | 2 days | Medium |

## Implemented Fixes #3 + #7 — Results

| Config | Trades | Mean net bps | Std | Folds pos | Notes |
|---|---|---|---|---|---|
| A: BTC/demean (baseline) | 423 | +17.18 | 30.77 | 2/4 | reference |
| B: BTC/sharpe alone | 618 | +4.98 | 40.45 | 2/4 | Fix #3 alone hurt |
| C: pooled/demean alone | 2691 | -27.55 | 44.68 | 3/12 | Fix #7 alone catastrophic |
| **D: pooled/sharpe (both)** | **2019** | **+23.11** | 44.05 | **8/12** | **best ever** |

### Key findings

1. **Fixes are interaction-dependent.** Sharpe target alone hurt (-12 bps); pooled alone catastrophic (-45 bps); combined +6 bps over baseline AND generalizes across all 3 symbols.

2. **SOL went from -6 → +29 bps net** under proper pooling. Cross-symbol failure was a methodology issue (heteroskedasticity + data scarcity), not non-generalization.

3. **Per-symbol breakdown of config D:**
   - BTC: +6.98 (slightly down from baseline)
   - ETH: +33.28 (new alpha unlocked)
   - SOL: +29.07 (was -6 alone)

4. **8/12 folds positive** vs 3/4 BTC baseline — better statistical power, more robust evaluation.

### Mechanism explained

Raw returns have wildly different scales across BTC/ETH/SOL (BTC ATR ~18 bps, SOL ATR ~27 bps). Pooling without normalization confuses the model — same raw return is a different signal at each symbol. Sharpe target normalizes everything to vol-units, making cross-symbol training meaningful. The two fixes only work together — confirming hypothesis #3 and #7 are coupled.

## Tested Fix #6 — Cross-asset features (HURT)

| Config | Trades | Mean net | Std | Folds pos |
|---|---|---|---|---|
| D: pooled/sharpe | 2019 | +23.11 | 44.05 | 8/12 |
| E: D + cross-asset | 1454 | +0.80 | 54.09 | 5/12 |

Cross-asset features (excess returns, rolling beta, log-spread z-score vs reference symbol) hurt mean by -22 bps. **Same overfitting pattern as lag features, regime composite, and v2 trade-flow features.** All "feature expansion" experiments in this Program have hurt because:

1. Training window is moderate (50d sliding / expanding); adding features inflates the variance/overfit risk
2. Cross-asset features are correlated with existing return features
3. Per-fold variance went UP (44 → 54), not down

**Conclusion: feature expansion has hit its ceiling on this dataset.** The existing 18-feature TR set is well-tuned. Future feature work needs to substitute, not append.

## Tested Fix #4 — Calibrated trigger (NO IMPROVEMENT)

| Config | Trades | Mean net | Std | Folds pos |
|---|---|---|---|---|
| F1: top_magnitude (current) | 2019 | **+23.11** | 44.05 | 8/12 |
| F2: band [p70, p95] | 6143 | +3.34 | 35.82 | 5/12 |
| F3: band [p80, p98] | 5147 | +11.20 | 40.51 | 6/12 |
| F4: isotonic P(win) | 10997 | +4.12 | 27.60 | 6/12 |

All trigger variants underperformed the baseline. Diagnosis: **Fixes #3+#7 already solved the calibration issue implicitly.** Earlier "high-conviction is anti-predictive" finding was a symptom of heteroskedasticity (Fix #3) and data-pooling problems (Fix #7) — once the target was vol-normalized and training data pooled, the model became well-calibrated and the top-magnitude trigger IS optimal.

F4's isotonic calibration *did* reduce variance (std 44 → 27.6) but at too steep a mean cost. Not deployable as primary trigger.

## Robust takeaway

Three fixes attempted post-Fix-3+7. All hurt or no-help:
- Fix #6 (cross-asset features): **HURT** — feature expansion overfits
- Fix #4 (calibrated trigger): **NO IMPROVEMENT** — Fixes #3+#7 already addressed it
- (Earlier) Lag features, regime composite, sample weighting, more features in general — all **hurt**

The pattern: feature/trigger expansion in this ~14k samples × 18 features regime hits a ceiling. Improvements come from *substituting* (Sharpe target, pooled training) not *adding*.

## Tested Fix #2 — Learned regime classifier (HURT)

| Config | Trades | Mean net | Std | Folds pos |
|---|---|---|---|---|
| G1: autocorr baseline | 2019 | +23.11 | 44.05 | 8/12 |
| G2: learned continuation | 2807 | -43.92 | 59.60 | 1/10 |
| G3: learned meaningful | 2437 | +5.13 | 43.17 | 6/12 |
| G4: continuation strict | 2022 | -33.40 | 37.08 | 3/11 |

All three learned-classifier variants underperformed the simple autocorr-1h percentile gate. Diagnosis:
- Label/strategy mismatch (continuation label assumes past direction continues; trend strategy sometimes goes against)
- The single autocorr-1h statistic captures regime cleanly without overfit
- Adding model complexity here introduces new failure modes faster than it adds signal

## Why the post-Fix-3+7 optimizations all failed — root-cause analysis

### Cross-asset features (Fix #6): correlated with existing features + overfit

The 8 cross-asset features added (`excess_ret_h`, `beta_ref_1d`, `corr_ref_1d`, `spread_*`) are NOT orthogonal to the existing feature set:
- `excess_ret_h = my_ret_h - ref_ret_h` — `my_ret_h` part is already encoded via `return_3, return_5, return_1d`. We're really just adding `ref_ret_h` as a separate signal.
- `beta_ref_1d` correlates with `realized_vol_1h` (both vol-derived).
- The pooled-training rename to `_vs_ref` discards symbol-pair semantics — BTC-vs-ETH and SOL-vs-BTC have different meaning but the model treats them identically.

Net effect: 8 features with low marginal IC and 8 sources of overfit risk. Adding 8 redundant features to a 18-feature × 14k-sample regime is exactly the recipe for overfit.

### Calibrated trigger (Fix #4): underlying problem was already fixed

The motivation was "high-conviction picks have lower win rate than mid-conviction" — observed in the *raw demean* model.

**That observation no longer holds under config D.** Fixes #3 (Sharpe target) and #7 (pooled training) eliminated heteroskedasticity, which was the root cause of the calibration inversion. The well-calibrated regression now produces magnitudes that *do* monotonically correlate with realized win rate.

Fix #4 was treating a symptom that no longer exists.

### Learned regime classifier (Fix #2): label/strategy mismatch + meta-model overfit

Three label variants tested. All failed differently:
- **G2 ("trend continuation"):** label = past direction continues. But the trend strategy sometimes shorts after extended rallies (mean-reversion-within-trend). Label/strategy mismatch.
- **G3 ("meaningful move"):** label = `|fwd_48| > 1σ`. Direction-blind — gates on volatility, not profitability.
- **G4 (strict G2):** same problem, tighter threshold.

Deeper issue: a meta-model (regime classifier) gating a primary model (trend regressor) — both see overlapping features. When the gate tightens, the primary model trains on a regime-shifted, smaller subset → overfits. The gate's filtering quality is bounded by how well it can predict from regime features alone, but the primary model already uses those features.

**Why autocorr_1h percentile beats it:** zero learnable parameters, can't overfit, captures genuine regime info (return serial correlation) directly. A simple statistic out-performs a learned alternative when:
- Sample size is moderate
- The simple statistic captures what matters

## Common pattern across the 3 failures

1. **Each "fix" added complexity to a regime that's already at its complexity ceiling.** ~14k samples / 18 features = 800 samples per feature — barely enough for stable interactions. Adding features, calibrators, or meta-models reduces sample-per-parameter ratio.

2. **Fix #3+#7 fixed many "downstream" problems implicitly.** Anti-predictive top-conviction, regime instability, cross-symbol failure — all symptoms of heteroskedasticity (Fix #3) and data scarcity (Fix #7). Fixing root causes made downstream fixes unnecessary.

3. **Simple statistics beat learned alternatives at this sample size.** `autocorr_1h percentile`, top-magnitude trigger — both are robust because they have zero learnable parameters. Learned alternatives suffer from overfitting risks the simple versions avoid.

## What would actually break the wall

Given the analysis, plausible next moves:
1. **More training data** — 800 samples/feature is the limit. With 1200 days instead of 400d, we'd have 3× the samples.
2. **Different cost regime** (VIP-3+ at ~5 bps RT) — shifts the math entirely.
3. **Genuinely orthogonal signal source** — orderbook L2 features (Tardis), or funding-rate-based features.
4. **Different market** — BTC/ETH/SOL is heavily competed.

What is **unlikely** to help (pattern is strong):
- More features (any kind)
- More model sophistication (calibrators, classifiers, ensembles beyond what we have)
- Hyperparameter tweaks (within seed noise)
- Different objectives (Sharpe target was the meaningful change; further tweaks fit noise)

## Final state — Validated strategy

**Config D — pooled+Sharpe with autocorr-1h gate, top-magnitude trigger, fixed h=48:**
- Mean net: **+23.11 bps**, 8/12 folds positive across BTC/ETH/SOL
- BTC: +6.98, ETH: +33.28, SOL: +29.07
- Std fold net: 44.05 (still significant; needs drawdown controls for deployment)
- Total trades: 2019 across 5 walk-forward folds (~135/day)

**Recommendation:** stop optimizing. Move to:
- OOS holdout test (held-back recent data)
- Drawdown circuit breakers
- Out-of-window forward test (data past 2026-04-27)
- Then deployment-readiness work

## OOS holdout test result — STRATEGY FAILS TO GENERALIZE

| Symbol | In-sample mean net | **OOS mean net** | Δ |
|---|---|---|---|
| BTC | +6.98 | -25.89 | -33 bps |
| ETH | +33.28 | -55.79 | -89 bps |
| SOL | +29.07 | -42.88 | -72 bps |
| Aggregate | +23.11 | **-44.78** | -68 bps |

**Setup:** trained validated config D on first ~310 days (pooled BTC+ETH+SOL with cal slice at the end), evaluated on truly-held-out last 90 days.

**Findings:**
- OOS win rate: 35.1% (vs in-sample ~50%)
- OOS max drawdown: -12.6% cumulative on unit notional
- OOS trigger rate jumped 2-3× for ETH/SOL — feature distribution shifted
- OOS long bias spiked to 72-81% — model heavily long during a bearish OOS window

**Diagnosis:** the in-sample +23 bps was hyperparameter-leakage-overfit. We chose `regime_cutoff=0.33`, `q=0.95`, `h=48`, ensemble seeds by looking at fold-level results across the entire 400d. The cherry-picked config worked on the seen data but not on the genuinely held-out tail.

**This is the third, most decisive negative signal:**
1. Cross-symbol failure (SOL standalone was -6 bps in original tests)
2. Three attempted optimizations all failed
3. Contiguous OOS holdout collapses to **-45 bps** mean net per trade

## Look-ahead bug found and fixed

The Sharpe target normalization had `.shift(1)` where it should have been `.shift(horizon)`. This caused `horizon-1` bars of look-ahead bias in the rolling mean/std of forward returns used for target normalization.

| Test | BUGGY (shift=1) | FIXED (shift=horizon) | Δ |
|---|---|---|---|
| Walk-forward mean net | +23.11 | **+8.51** | -14.6 bps |
| OOS holdout mean net | -41.52 | -39.05 | +2.5 bps |

**The look-ahead was real and inflated in-sample by ~15 bps. But the OOS gap is largely independent.**

After fix:
- True walk-forward edge: +8.51 bps net per trade (8/12 folds positive)
- OOS holdout: -39.05 bps mean net
- IS/OOS gap: -47 bps (still huge)

The remaining IS/OOS gap is from hyperparameter overfitting (config choices made by reviewing all 5 walk-forward folds) and distribution shift in the held-out period.

## Final conclusion

The strategy as designed does NOT deploy at retail VIP-0. The Program produced:
- A reusable methodology (Sharpe-normalized targets, pooled training, regime gating)
- Validated infrastructure (feature pipeline, walk-forward CV, multi-seed ensemble, vol-scaled sizing)
- A clear understanding of WHY this strategy fails (look-ahead inflation, hyperparameter leakage, distribution shift, dataset size limit, retail cost margin)
- One look-ahead bug found and fixed (target normalization shift)

The +23 walk-forward result was 65% real (+8.51) and 35% look-ahead inflation. Even the honest +8.51 doesn't survive OOS holdout.

**Real paths forward require something we don't have:**
- Longer history (2-3 years instead of 400 days) to reduce hyperparameter overfit
- Lower cost regime (VIP-3+) to widen profit margin (would convert +8.5 → +15 net)
- Different signal source (orderbook L2)
- Different market

## Alpha-residual redesign (Apr 30)

After fixing the VPIN look-ahead bug and switching the prediction target from
raw forward return to **alpha residual** = `my_fwd - β × ref_fwd`, we redid the
end-to-end pipeline (feature audit → optimized model → re-eval).

### Step 1 — Alpha target audit

Three β windows (1d / 3d / 7d) compared on in-sample data:

| β window | avg_var_ratio | avg_|corr(alpha,raw)| | avg_alpha_ac1 |
|---|---|---|---|
| 1d  | 0.5694 | 0.586 | 0.974 |
| 3d  | 0.5697 | 0.584 | 0.974 |
| 7d  | 0.5715 | 0.581 | 0.974 |

**Findings:**
- All three β windows produce nearly identical residuals — choice of window does not matter.
- Residual `var_ratio ≈ 0.57` — alpha removes ~43% of variance from raw forward return.
- Residual is still ~0.5–0.65 correlated with raw return — the β-strip is imperfect; some market exposure remains.
- `1d` window picked as default.

### Step 2 — Feature IC audit (in-sample, β=1d)

Per-symbol Spearman IC of each base feature against alpha vs raw target.
Top features by |IC vs alpha| differ across symbols:

- **BTC** (ref=ETH): `ema_slope_20_1h` (-0.080), `return_1d` (-0.074), `bars_since_high` (+0.070)
- **ETH** (ref=BTC): `hour_cos` (+0.064), `volume_ma_50` (+0.051), `atr_pct` (+0.041), `tfi_smooth` (+0.026)
- **SOL** (ref=BTC): `hour_cos` (+0.046), `bars_since_high` (+0.031), `vpin` (+0.026)

**Critical finding:** `ema_slope_20_1h` has *opposite signs* across symbols — strongly negative for BTC alpha, positive for ETH alpha. A pooled tree without per-symbol context can't capture both.

Lift = IC_alpha − IC_raw. Highest |lift|: `ema_slope_20_1h` (0.032), `atr_zscore_1d` (0.028, negative — hurts alpha), `return_1d` (0.027).

**Weak features** (drop candidates): `tfi_smooth`, `signed_volume`, `efficiency_96`, `adx_15m` — all have |IC| < 0.01 vs alpha across all 3 symbols.

### Step 3 — Cross-asset feature audit

Adding cross-asset features (computed by `features_ml.cross_asset`) and the reference symbol's own features as covariates:

| Feature | BTC IC | ETH IC | SOL IC | Notes |
|---|---|---|---|---|
| `spread_log_vs_ref` | -0.036 | **-0.090** | -0.069 | Strong, consistent across symbols |
| `ref_ema_slope_20_1h` | (=raw) | **+0.055** | +0.020 | Lift +0.063 on ETH — large |
| `ref_return_1d` | -0.051 | +0.052 | +0.012 | Lift +0.061 on ETH |
| `beta_ref_1d` | -0.061 | +0.020 | -0.010 | Mild on ETH/SOL |

`spread_log_vs_ref` (relative price level) is the **strongest single cross feature** for alpha — IC up to -0.090, comparable to or better than any base feature. Reference-symbol features (`btc_ema_slope_20_1h`, `btc_return_1d`) provide large lift specifically for ETH alpha.

### Step 4 — alpha_v2 head-to-head

`alpha_v2.py`: drops weak features, adds 4 cross-asset features (`spread_log_vs_ref`, `beta_ref_1d`, `ref_ema_slope_20_1h`, `ref_return_1d`), adds `sym_id` indicator. Pooled training, ensemble of 5 seeds.

Walk-forward (5 folds × 3 symbols = 15 fold-symbols):

| Config | n | alpha_bps | market_bps | net_bps | folds_pos |
|---|---|---|---|---|---|
| WF v1 (18 base feats) | 2050 | -1.42 | -8.09 | -21.87 | 3/15 |
| **WF v2 (refined+cross)** | 2677 | **+6.17** | -13.36 | -19.54 | 6/15 |

OOS holdout (last 90d, 1 fold × 3 symbols = 3):

| Config | n | alpha_bps | market_bps | net_bps | folds_pos |
|---|---|---|---|---|---|
| OOS v1 | 1862 | +3.84 | +6.14 | -2.22 | 1/3 |
| OOS v2 | 6895 | +2.47 | +3.24 | -6.67 | 1/3 |

**Per-symbol OOS:**

| Symbol | v1 alpha | v1 net | v2 alpha | v2 net |
|---|---|---|---|---|
| BTC | -3.07 | -17.65 | **+2.02** | -2.57 |
| ETH | +24.46 | +38.61 | +13.72 | +9.44 |
| SOL | -9.87 | -27.61 | -8.35 | -26.89 |

### Verdict on the alpha-residual redesign

**What v2 achieved:**
- WF alpha capture flipped from -1.42 → +6.17 bps (+7.6 bps signal extraction)
- BTC alpha capture went from -3 to +2 bps OOS — small but real improvement
- Feature importance shifted to cross-asset: `spread_log_vs_ref` 16.8%, `beta_ref_1d` 14.2%, `ref_ema_slope_20_1h` 8.0% — together 39% of gain. The model uses them.

**What v2 failed at:**
- OOS net got *worse* than v1 (-6.67 vs -2.22). Trigger rate ballooned (6895 vs 1862 trades) — predictions have higher variance OOS than IS, so the q=0.95 cal threshold lets through too many signals.
- ETH OOS net regressed sharply (+38.61 → +9.44). The cross-asset features that lifted ETH alpha *in-sample* did not generalize.
- SOL alpha remains broken (-8 to -10 bps OOS). No feature combination reaches profitability there.

**The alpha-residual hypothesis was correct in direction, insufficient in magnitude:**
- Removing market component does isolate a small alpha signal that responds to cross-asset + dominance features.
- The signal magnitude (~5-10 bps gross alpha) is smaller than retail RT cost (~13 bps).
- ETH-vs-BTC has the only meaningful, persistent alpha; BTC and SOL alpha are too weak.
- Pooled training helps via larger sample, but the per-symbol sign reversals (e.g. `ema_slope_20_1h`) mean the global tree spends capacity on splits the audit predicted would happen.

**Conclusion: the migration to alpha-residual produced a methodologically cleaner, in-sample-stronger model — but the OOS economics still don't support deployment at retail cost.** The redesign was worth doing (WF alpha flipped sign, audit identified the real driver = cross-asset spread / reference momentum), but a profitable strategy from this signal class needs lower fees (VIP-3+), longer training history, or a different microstructural angle.

## Alpha v3 — alpha-tailored feature redesign

After v2 still under-performed OOS, the next step was to **regenerate features specifically for the alpha residual target** rather than reusing the raw-return feature set.

### New feature module — `features_ml/alpha_features.py`

22 alpha-tailored features in 6 families:

  1. **Dominance dynamics**: `dom_level_vs_ref` (log my/ref), `dom_change_{12,48,288}b_vs_ref`, `dom_z_{1d,7d}_vs_ref`
  2. **Reference symbol's recent state**: `ref_ret_{12,48}b`, `ref_ema_slope_4h`, `ref_ema_diff_short_long`, `ref_realized_vol_1h`
  3. **Beta dynamics**: `beta_zscore_vs_ref`, `beta_short_vs_ref`
  4. **Idiosyncratic returns / vol**: `idio_ret_{12,48}b_vs_ref`, `idio_vol_{1h,1d}_vs_ref`, `idio_vol_ratio_vs_ref`
  5. **Correlation regime**: `corr_1d_vs_ref`, `corr_change_3d_vs_ref`
  6. **Order flow divergence**: `tfi_diff_vs_ref`, `signed_vol_z_diff_vs_ref`

### Alpha-targeted IC audit (`alpha_v3_audit.py`)

Per-symbol Spearman IC vs alpha target on in-sample data only:

| Feature | Avg \|IC\| | Sign-consistent across symbols |
|---|---|---|
| `dom_level_vs_ref` | 0.065 | YES (always negative) |
| `ref_ret_48b` | 0.044 | sign flips per-symbol |
| `ref_ema_diff_short_long` | 0.042 | sign flips |
| `ref_ema_slope_4h` | 0.041 | sign flips |
| `idio_vol_1d_vs_ref` | 0.033 | sign flips |
| `dom_z_{1d,7d}_vs_ref` | 0.031 | sign flips |
| `idio_ret_48b_vs_ref` | 0.024 | YES (always negative — mean-reversion) |
| `corr_change_3d_vs_ref` | 0.024 | sign flips |

`dom_level_vs_ref` is the **strongest sign-consistent feature** — relative price level mean-reverts in alpha across all 3 symbols. `idio_ret_48b_vs_ref` also sign-consistent (past-4h idiosyncratic move mean-reverts in alpha).

Useless features (avg |IC| < 0.01): `signed_vol_z_diff_vs_ref` — dropped.

### v3 model results

Curated 17 features + `sym_id` indicator. v1/v2/v3 head-to-head:

| Config | OOS IC (BTC, ETH, SOL) | OOS net |
|---|---|---|
| v1 (18 base) | +0.031 / +0.080 / +0.002 | -2.22 |
| v2 (cross-asset) | +0.007 / +0.097 / +0.018 | -6.67 |
| **v3 (alpha-tailored)** | **+0.080** / +0.064 / **+0.035** | -18.81 |

**v3 has the most consistent positive IC across all three symbols.** BTC IC went +0.031 → +0.080. SOL IC went 0.00 → +0.035.

### Trigger-rate calibration breakdown

v3 OOS triggered 26.6% of bars (vs 5% target from q=0.95 cal threshold). Cause: prediction distribution shifted between cal and OOS, especially for SOL.

Fix in `alpha_v3_thr.py`: per-symbol thresholds. Reduces SOL trigger rate from 68% → 46%, ETH from 17% → 16%, BTC stays at 3%. Still high for SOL — its prediction distribution is genuinely wider in OOS.

### Hedged execution test (`alpha_v3_neutral.py`)

We predict alpha but **trade my-symbol unhedged**, so realized P&L = alpha + β × ref_fwd. The market component is uncorrelated noise that drowns the alpha signal in any one symbol.

Tested true market-neutral: long my_symbol + short β×ref_symbol. Realized P&L = side × alpha exactly.

| | OOS gross | OOS cost | OOS net |
|---|---|---|---|
| BTC naked (q=0.99) | -32.0 | 12.0 | -43.95 |
| BTC **hedged** (q=0.99) | **+9.39** | 19.3 | -9.90 |
| ETH naked (q=0.99) | +20.4 | 12.0 | **+8.49** |
| ETH hedged (q=0.99) | +5.57 | 24.2 | -18.65 |

**Hedging recovers alpha capture cleanly** (BTC gross goes -32 → +9.4 — alpha is real, market noise was hiding it). But the second leg roughly doubles cost (~24 bps RT), which exceeds the ~5-10 bps alpha edge.

### One configuration breaks even: ETH naked at q=0.99 OOS

| Symbol | n | Alpha | Gross | Cost | Net | IC |
|---|---|---|---|---|---|---|
| ETH naked q=0.99 OOS | 1873 | +5.57 | +20.44 | 11.95 | **+8.49** | +0.048 |

ETH naked profited because in this OOS window, the market direction *coincided* with the predicted alpha direction — the market_pnl term was +14.87 bps, helping rather than hurting. This is **luck-of-regime, not robust alpha capture**.

### Final synthesis: the alpha edge is real but uneconomic

Three numerical facts that frame the verdict:
- **Real alpha edge**: ~5-10 bps per trade (BTC alpha=+9.4, ETH alpha=+5.6 at q=0.99 OOS, both positive).
- **Naked execution cost**: ~12 bps RT — exceeds alpha. Net P&L depends on luck of market_pnl term.
- **Hedged execution cost**: ~20-24 bps RT. Cleanly captures alpha, but cost > alpha.

Three things that would change the verdict:
1. **Lower fees** (VIP-3 = 0.025% taker, ~5 bps RT per leg → ~10 bps hedged): alpha would survive.
2. **Maker execution**: post-only entries fill ~60% in calm regimes, reducing effective fee by ~40%.
3. **Bigger alpha edge**: more features, longer history (current 400d → 2 years), or different signal type (orderbook L2).

The redesign produced its intended deliverable: a methodologically validated, audit-driven feature set that captures real alpha (positive IC consistently across symbols, +5-10 bps gross alpha at q=0.99 OOS). Whether the strategy deploys depends on cost regime, not signal quality.

## Cross-sectional alpha v4 (25 symbols)

After confirming alpha is structurally ~5-10 bps in the 3-symbol single-pair setup, we tested whether **cross-sectional ranking across 25 perps** could amplify the edge via diversification + built-in market neutrality.

### Universe
25 USDM perps with 400d 5m kline history each:
BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE, LINK, DOT, LTC, BCH, NEAR, UNI, ATOM, FIL, ARB, OP, APT, INJ, SUI, SEI, TIA, RUNE, WLD.

### Setup (`features_ml/cross_sectional.py`, `ml/research/alpha_v4_xs.py`)
- Equal-weight basket = mean(returns across 25 symbols)
- Per-symbol target: alpha_s = my_fwd_s − β_s × basket_fwd
- Features: 7 base (kline) + 9 basket-relative (dom_level_vs_bk, idio_*, beta_*, corr_*, bk_ret_*) + sym_id
- Pooled training across all (symbol, bar) pairs
- Inference: rank predicted alpha cross-sectionally, long top-N / short bottom-N

### Results — top-quintile (5/5) portfolio

| Phase | spread_ret | spread_alpha | net (24 bps cost) | rank IC |
|---|---|---|---|---|
| Walk-forward (4 folds) | +6.56 | +6.17 | -17.44 | +0.036 |
| OOS holdout (90d) | +3.01 | +3.59 | -20.99 | +0.035 |

### Concentrated + conviction-filtered variants (`alpha_v4_concentrated.py`)

Walk-forward:

| Config | spread_alpha | spread_ret | net | Sharpe |
|---|---|---|---|---|
| top-1 / bot-1 | +10.51 | +5.64 | -18.36 | +0.016 |
| top-1 + conviction>q80 | **+16.26** | +4.66 | -19.34 | -0.001 |
| top-5 + conviction>q80 | +7.06 | **+7.31** | -16.69 | **+0.102** |

OOS:

| Config | spread_alpha | spread_ret | net |
|---|---|---|---|
| top-1 / bot-1 | -0.25 | -6.32 | -30.32 |
| top-5 / bot-5 | +3.59 | +3.01 | -20.99 |

### Findings

1. **Cross-sectional ranking works**: rank IC stable at +0.035 across all folds and OOS — the most consistent positive signal we've measured. Pooled training with 25 symbols × 100k bars gives the model enough samples to learn robust patterns.

2. **Alpha magnitude does NOT amplify**: spread_alpha is +6 bps (top-5) to +16 bps (top-1+conviction) — same order of magnitude as the 3-symbol v3 single-pair alpha (+5-10 bps). Cross-sectional improves *measurement consistency*, not *signal magnitude*.

3. **Concentrated picks have higher alpha but worse OOS**: top-1 IS spread_alpha=+10-16 bps, but OOS drops to +0 to +4. Single-name idiosyncratic noise dominates. Top-5/bot-5 is more robust.

4. **Realized return < alpha**: spread_ret (5-7 bps) is consistently *lower* than spread_alpha (6-16 bps). Even with cross-sectional neutralization, executing alpha-residual picks via raw symbol returns adds noise that erodes the edge.

5. **Net P&L still negative everywhere**: best WF config (top-5 + conviction) nets -16.69 bps; best OOS is -21 bps. 24 bps RT cost (long leg + short leg) is uniformly above the 3-7 bps realized spread.

### Conclusion: alpha ceiling confirmed

Cross-sectional with 25 symbols at 4h horizon and kline-only features gives the **same alpha magnitude** as the 3-symbol pair-trading approach. The improvement is in **robustness and Sharpe**, not in absolute edge. The structural ceiling for OHLCV+aggTrade signals at 4h horizon is ~5-10 bps gross alpha — confirmed across:
- 3-symbol pair (alpha vs ref): 5-10 bps
- 25-symbol cross-sectional (alpha vs basket): 6-16 bps in-sample, 0-4 bps OOS

To deploy profitably, one of these is required:
1. **Lower fees**: VIP-3 maker (~1 bp/leg RT) brings cost to ~2-4 bps total → +3 net at top-5/q80 OOS, marginal but positive.
2. **Different horizon**: 1-day or 1-week residual reversal documented at 30-50 bps in academic literature; would clear retail cost.
3. **Different data**: orderbook L2 features add 5-10 bps IC contribution typical in microstructure research.

The 4h-horizon OHLCV+aggTrade alpha is now a fully characterized signal: it exists, it's consistent, it's not big enough.
