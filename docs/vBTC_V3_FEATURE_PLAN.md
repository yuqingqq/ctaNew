# vBTC v3 feature plan — BTC-residual hybrid model with process-type features

Last updated: 2026-05-13
Status: **EXECUTED THROUGH PHASE 3 — REJECTED. Plan retained as audit trail.**

## Executive verdict (2026-05-13)

Two rounds of testing on the same +0.74 baseline panel (`panel_variants_with_funding.parquet`):

### Round 1: v3 features at 1d granularity — REJECTED outright

| variant | #feats | Sharpe | Δ vs +0.74 | IC | folds+ |
|---|---|---|---|---|---|
| V0_WINNER_17 (reproduce) | 17 | **+0.74** | 0 | +0.0157 | 4/9 |
| V1_W17 + v3_aug_8 (1d feats) | 25 | -0.67 | **-1.41** | +0.0139 | 4/9 |
| V2_W17 + v3_aug_4 (1d feats) | 21 | -4.97 | **-5.71** | +0.0116 | 1/9 |
| Pure WINNER_BTC_v3 (1d feats) | 24 | -1.86 | -2.60 | -0.002 | 3/9 |

Root cause: features computed at 1-day rolling windows then forward-filled to 5m bars → constant within day → LGBM has nothing to learn (`best_iter=1`).

### Round 2: rebuild at 5m bar windows + dedupe — TECHNICALLY LIFTS BUT FRAGILE

Rebuilt 17 features at 5m bar windows (β/corr/resid_vol rolled in 8640/25920-bar windows), 5 semantic duplicates of WINNER_17 dropped (`resid_vol_7d`, `funding_mean_7d`, `corr_btc_90d`, `dist_from_90d_high`, `turnover_volatility_30d`). Distributional moments (skew/kurt/jump/trend) computed at 1h cadence for compute tractability, asof-merged to 5m + shift(1).

| variant | #feats | Sharpe | CI | Δ vs +0.74 | IC | folds+ |
|---|---|---|---|---|---|---|
| V0_WINNER_17 | 17 | +0.74 | [-1.35,+2.87] | 0 | +0.0157 | 4/9 |
| V1 W17 + v3_full19 (5m) | 36 | **+2.40** | [+0.04,+4.64] | +1.66 | **0.0000** | 5/9 |
| V2 W17 + v3_top8 (5m) | 25 | +0.48 | [-1.91,+3.03] | -0.26 | +0.0068 | 4/9 |
| V3 W17 + v3_top4 (5m) | 21 | +1.39 | [-0.99,+3.78] | +0.65 | +0.0055 | 5/9 |

**Surface read**: V1 lifts Sharpe by +1.66, passes adoption gate ≥ +0.94, CI [+0.04,+4.64] barely excludes zero.

**Per-fold LOFO attribution exposes fragility**:
- Excluding fold 4 alone → V1 Δ = **+0.04** (vs +1.66 with all folds). Fold 4 ALONE carries V1's lift.
- V3's lift concentrated in folds 1+3+6, fails 5/9 folds.
- V1/V2/V3 non-monotonic (more features ≠ better) — noise-dominated optimization.
- V1 has IC = 0.000 yet Sharpe +2.40 — model is no longer doing the ranking work; V3.1's gates+universe filter happen to surface positive-alpha cycles given V1's prediction distribution → **selection-via-noise**.

This is the **identical fragility pattern as Phase Q WINNER_23 retrain** (memory): fold-6-dependent +0.18 lift that crumbles under LOFO. And Phase K3 cost-aware swap. And Phase L shrinkage IC. The +1.66 looks like a win in aggregate but is statistically a lucky-draw artifact of high-variance per-fold outcomes.

### Decision

Per plan's strict stop criterion at end of Phase 3, NOT adopting v3 features. Production stack stays **WINNER_17 + β-residual + β-hedged + V3.1 6-sleeve = Sharpe +0.74**.

### What was learned

1. **Feature time-resolution matters more than IC.** 1d-granularity features were a fundamental no-go regardless of their per-cycle cross-sectional IC.
2. **Even at correct resolution, adding features to WINNER_17 produces high per-fold variance**, not robust additive signal. The universe-overfit nature of WINNER_17 means perturbing its feature set disrupts equilibrium; some perturbations happen to help specific folds.
3. **IC=0 + Sharpe>0 is a red flag, not a win.** When per-cycle IC drops to zero while strategy Sharpe rises, the gates are filtering noise to favorable cycles by accident.
4. **LOFO is mandatory** for any in-sample lift to trust. Aggregate Sharpe + favorable CI are necessary but not sufficient; fold-level robustness must be checked.

---

## Goal

Build a universe-invariant strategy that captures **process-type** characteristics of each symbol ("trend-follower vs mean-reverter", "high-β unstable vs stable", "crowded vs uncrowded", "liquid vs fragile") via richer features.

Optimization after review: do **not** immediately run the full universal / per-symbol / hybrid stack. First prove that the new feature families improve the universal BTC-residual model under clean model-only diagnostics. Only then spend compute on per-symbol and hybrid residual banks. This keeps failures attributable: feature quality first, architecture second, selector/gate interaction last.

Current baseline reference points to beat:
- **WINNER_17 + β-residual target + β-hedged execution**: Sharpe **+0.74**, end-equity **$126.96** on $100, per-cycle IC +0.0157
- **WINNER_BTC + β-residual + β-hedged (v2 universal)**: Sharpe **−0.49** (variant C in 4-way test), per-cycle IC +0.0164
- **AVAX per-symbol with WINNER_17**: per-cycle IC −0.0152 universal → **+0.0099 per-symbol** (Δ +0.025)
- **AVAX per-symbol with WINNER_BTC**: per-cycle IC +0.0208 universal → **−0.0404 per-symbol** (universal wins)

## Motivating literature

(Sources audited 2026-05-13.)

- **Blitz, Hanauer, Vidojevic 2017 "Idiosyncratic Momentum Anomaly"** — vol-adjusted residual return (`idio_ret / σ(idio)`) outperforms raw idio momentum with ~2× Sharpe + lower crash risk. We currently have numerator and denominator separately but no `idio_sharpe` features.
- **Liu, Tsyvinski, Wu 2022 (JoF)** — C-3 factor model: Market + Size + Momentum captures cross-section of crypto returns.
- **Cakici et al. 2024 (ML on crypto)** — variable importance: **past alpha, illiquidity, momentum** are top-3 predictors. We have past alpha and momentum; missing illiquidity (Amihud).
- **Anastasopoulos & Gradojevic 2025 "Order Flow and Crypto Returns"** — weekly portfolio sorts on signed order flow: Sharpe **+1.93** L-S, alpha 1.72% (t=2.71). We have `signed_volume_4h` in panel but never used it.
- **Ficura & Colak 2023** — small/illiquid coins mean-revert; large/liquid coins trend. Symbol-level process character matters; need features that let model distinguish process types.
- **López de Prado / Easley VPIN** — order flow toxicity predicts short-vol regimes; we have proxies via `tfi_4h`, `aggr_ratio_4h`.

## Feature design philosophy

Current features ("first-order"): how did this symbol move? how vol? how β? how funding? how liquid?

What's missing ("process-type"): is this symbol trend-following or mean-reverting? high-β-stable or unstable? crowded or uncrowded? jump-prone or smooth? funding-driven or volume-driven?

The v3 set adds explicit features for these process-type questions, all universe-invariant.

Design constraint: the final model should be compact. The repo history shows that redundant price-derived additions often fragment LGBM splits and hurt OOS even when individual features look plausible. Treat the 39 features below as a **candidate pool**, not the intended training set.

## v3 candidate feature set (39 features pre-pruning)

### A. Liquidity / tradability (6)

```
log_dollar_volume_7d           # short-window log ADV
log_dollar_volume_30d          # medium-window
volume_stability_30d           = std(daily_dollar_volume_30d) / mean(daily_dollar_volume_30d)
amihud_illiq_30d               = mean(|ret_1d| / dollar_volume_1d) over 30d  # Cakici top-3 predictor
roll_spread_proxy_30d          = 2 * sqrt(-cov(Δp_t, Δp_{t-1})) when negative  # Roll's estimator from close-only
turnover_volatility_30d        = std(daily_turnover) over 30d
```

### B. BTC relationship (7)

```
beta_btc_30d                   # rolling 30d β (faster than current 90d)
beta_btc_90d                   # already exists as beta_btc_pit
beta_btc_180d                  # longer-window structural β
beta_btc_instability           = beta_btc_30d - beta_btc_180d  # symbol's β regime shift
corr_btc_30d
corr_btc_90d
corr_breakdown                 = corr_btc_30d - corr_btc_90d  # correlation regime shift
```

### C. Residual behavior (9)

```
resid_vol_7d                   # multi-horizon idio vol levels
resid_vol_30d
resid_vol_90d
resid_skew_30d                 # tail asymmetry
resid_kurt_30d                 # tail fatness
resid_jump_count_30d           = #{|idio_ret_1d| > 3·rolling_σ} over 30d  # jump-proneness
resid_autocorr_1d              = AR(1) coefficient of idio returns over 30d  # trend vs revert character
resid_reversal_score_7d        = -idio_ret_7d / idio_vol_7d  # Blitz reversal canonical
resid_trend_score_30d          = idio_ret_30d / idio_vol_30d  # Blitz trend canonical
```

### D. Trend / anchoring (5)

```
dist_from_30d_high             = (current_close - max_close_30d) / max_close_30d
dist_from_90d_high
dist_from_365d_high
multi_horizon_trend_score      = mean of trend_score at {7d, 30d, 90d}
volume_confirmed_trend_score   = trend_score_30d * z(volume_30d)  # trend backed by abnormal volume
```

### E. Perp funding crowding (6)

```
funding_mean_7d
funding_mean_30d
funding_z_30d                  = (funding_now - mean_30d) / std_30d  # extreme-funding indicator
funding_persistence_7d         = fraction of past 7d with funding > 0  # perma-contango proxy
funding_abs_30d                = mean(|funding|) over 30d  # symbol's funding-pressure baseline
funding_sign_streak            # current run-length of same-sign funding (already partial in panel)
```

### F. Microstructure (4 — DROPPED from v3 default after Phase 1 inspection)

```
aggr_ratio_4h                  # aggressor (taker buy) ratio
signed_volume_4h_z             = z-score of signed_volume_4h over trailing 7d  # Anastasopoulos Sharpe +1.93
tfi_4h                         # trade flow imbalance
avg_trade_size_4h_z            # z-score of avg trade size (institutional flow proxy)
```

**Phase 1 finding (2026-05-13)**: aggTrade-derived microstructure columns only exist for **25/51 symbols** in the source panel. The 26 missing symbols would force LGBM to handle 51% missing-rate splits — a structural confound, not a feature-quality test. Combined with prior audits already rejecting microstructure under the production stack, drop block F from default v3. Isolated later test possible if v3 succeeds and we collect aggTrades for all 51.

**Effective v3 candidate set: A(6) + B(7) + C(9) + D(5) + E(6) + G(3) = 36 features.**

### G. Process fingerprint (3 — existing in panel, unused)

```
idio_skew_1d                   # daily realized residual skewness
idio_kurt_1d                   # daily realized residual kurtosis
idio_max_abs_12b               # 1h max abs idio return (jump magnitude proxy)
```

## Deferred (need new data sources — v4 if v3 succeeds)

```
open_interest_change_1d        ← Binance OI history endpoint
open_interest_z_30d
oi_price_divergence
long_short_ratio_z             ← Binance L/S endpoint
taker_buy_sell_imbalance       ← tick-level data
liquidation_imbalance          ← liquidation feed
perp_spot_basis                ← spot price history
basis_z_30d
```

## Pruning rules (Phase 2)

After building all 39, validate and prune in three stages: individual quality, block-composite quality, then family-level contribution.

1. **Do not drop solely on weak standalone IC if the feature belongs to a pre-registered process block.** Weak context features can be valuable through combinations/interactions even when direct IC is near zero.
2. **Drop isolated features if |IC| < 0.005** AND they are not in a promoted block or regime-conditioner group.
3. **For each pair with |corr| > 0.85**: drop the more-derived feature, keep the simpler one (e.g., keep raw, drop z-score), unless the pair improves a block composite.
4. **Drop if NaN rate > 30%** on the panel
5. **Drop if per-symbol variance is near-zero for >25% of symbols** (important for per-symbol models)
6. **Do not promote a family just because one member passes IC**. A family must improve either block-composite OOS IC, model-only OOS IC, or tail-pick gross PnL in an additive ablation.

Expected redundancy clusters:
- `log_dollar_volume_7d` vs `_30d` — keep one
- `beta_btc_30d/90d/180d` — likely keep 30d + 180d + instability (drop 90d if redundant)
- `corr_btc_30d/90d` — keep one + breakdown
- `resid_vol_7d/30d/90d` — keep 7d + 90d (mid-window subsumed)
- `dist_from_30d_high` vs `_90d_high` — pick by IC

**Target final set**: WINNER_BTC_v3 with **18-24 features** for universal model.

For per-symbol model: also drop near-constants per-symbol:
- `log_dollar_volume_30d` (slow-moving per symbol)
- `beta_btc_180d` (very slow per symbol)
- `funding_persistence_7d` (low variation within single symbol)
- → WINNER_BTC_v3_PERSYM ≈ **16-22 features**

### Family-level ablation rule

Build feature candidates in blocks and test them in two ways: first as simple composites, then inside the universal model harness.

| block | contents | default status |
|---|---|---|
| Core BTC residual | existing WINNER_BTC residual momentum, beta/corr, funding, stable context | baseline |
| Liquidity / tradability | log dollar volume, stability, Amihud, Roll spread, turnover volatility | candidate |
| BTC relationship | beta/corr multi-window, instability, breakdown | candidate |
| Residual behavior | vol/skew/kurt/jump/autocorr/reversal/trend | candidate |
| Trend / anchoring | distance from highs, multi-horizon trend, volume-confirmed trend | candidate |
| Funding crowding | funding means, z, persistence, abs, streak | candidate |
| Microstructure | signed volume, TFI, aggressor ratio, trade size | optional; high burden |

### Block-composite rule

Before LGBM family ablations, build a simple composite for each block:

```
for each feature in block:
  winsorize at train-window 1/99 pctile
  z-score using train-window mean/std
  orient sign by train-window IC

block_score = equal-weight mean(oriented z-scored features)
```

Also test a ridge composite per block if the equal-weight composite is close but noisy:

```
block_score_ridge = RidgeCV(features_in_block -> target_beta_btc)
```

Strict PIT rule: signs, z-score parameters, and ridge weights are fit only on the training slice for each fold, then applied to cal/test.

A block can survive even if most individual members have weak IC if:
- block composite OOS IC >= +0.010, or
- block composite improves tail-pick gross/cycle by >= +0.20 bps, or
- adding the block to LGBM improves model-only gross/cycle / full V3.1 Sharpe.

This protects interaction/context blocks from being discarded prematurely. Liquidity, beta-instability, and process-fingerprint features may not directly say "long now" but can tell the model which residual process applies to the symbol.

Promotion test:

```
base WINNER_BTC
base + one family
base + promoted families only
```

A family is promoted only if it improves at least one of:
- block-composite OOS IC by >= +0.010,
- model-only per-cycle IC by >= +0.002 without worsening more than 4/9 folds,
- model-only tail-pick gross/cycle by >= +0.25 bps,
- full V3.1 beta-hedged Sharpe by >= +0.10 in a quick run.

Microstructure requires a stricter gate because prior tests were negative:
- full V3.1 Sharpe lift >= +0.20, or
- clear model-only gross/cycle lift plus no turnover increase.

## Execution phases

### Phase 1 — Feature engineering (~30 min)

Single feature-build script that:
- Loads panel + klines + funding
- Computes all 39 features per symbol with PIT discipline (rolling windows shifted by 1 bar)
- Saves to `outputs/vBTC_features_btc_v3/panel_v3.parquet`

Implementation order within script:
1. Multi-horizon β to BTC at 30d/90d/180d (vectorized rolling)
2. Multi-horizon corr to BTC at 30d/90d
3. Multi-horizon resid_vol from idio returns
4. Resid_skew/kurt/jump_count rolling
5. Reversal_score / trend_score (idio_ret / idio_vol ratios)
6. Liquidity: Amihud, Roll's spread, volume stability, turnover vol
7. Trend / anchoring: dist_from_*_high, trend scores
8. Funding extensions
9. Microstructure: aggr_ratio_4h, signed_volume_z, tfi, avg_trade_size_z (from existing columns)
10. Process fingerprint: passthrough from existing columns

**Pass**: all 39 features <30% NaN per panel, non-zero variance per-symbol on full panel.

### Phase 2 — Feature validation + aggressive pruning (~15 min)

- Compute per-feature cross-sectional IC against `alpha_β` (entry-cadence sampled)
- Compute per-feature per-symbol time-series IC distribution
- Build equal-weight and ridge composites for each pre-registered feature block
- Build pairwise correlation matrix on a 100k-row sample
- Compute NaN rate, per-symbol variance, and PIT sanity checks
- Apply pruning rules above
- Final list saved to `outputs/vBTC_features_btc_v3/winner_btc_v3_features.json`

**Pass**: 18-24 features survive; promoted new features show mean |IC| > 0.012, a block composite passes, or a family-level additive test passes. If >24 survive, prune to the lowest-redundancy set before training.

### Phase 3 — Universal feature-family ablation (~30 min)

Train universal models for:
- `base`: WINNER_BTC
- `base + each candidate family`
- `base + promoted families`

Use the same 10 folds x 5 seeds, expanding window, target `target_beta_btc`.

**Output**:
- `outputs/vBTC_audit_panel_v3_family_ablation/summary.csv`
- per-family prediction parquet only for promoted/final sets

**Pass**:
- final universal v3 per-cycle IC >= baseline WINNER_BTC IC + 0.002, and
- tail-pick gross/cycle improves by >= +0.25 bps, and
- no more than 4/9 folds have worse IC than baseline.

If this phase fails, stop. Do not train per-symbol or hybrid banks.

### Phase 4 — Model-only trading diagnostic (~20 min)

Evaluate baseline and final universal v3 without the noisy IC selector/gate stack:

- fixed liquidity/data-quality universe,
- no rolling-IC top-15,
- no `filter_refill`,
- rank directly by prediction,
- beta-hedged MTM on `alpha_β`.

Purpose: isolate whether the model ranks residual alpha better before selector/gate interactions obscure the result.

**Pass**:
- model-only gross/cycle improves vs WINNER_BTC,
- model-only Sharpe is not worse than WINNER_BTC,
- turnover does not rise materially.

If model-only fails but full V3.1 later passes, treat the result as gate/selector mining and require extra placebo validation.

### Phase 5 — Full V3.1 beta-hedged universal test (~20 min)

- 10 folds × 5 seeds, expanding window
- Features: final promoted WINNER_BTC_v3
- Target: `target_beta_btc`
- LGBM hyperparams: same as Phase 1D (num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8)

**Output**: `outputs/vBTC_audit_panel_v3_universal/all_predictions.parquet`

Run full V3.1 beta-hedged stack:
- Rolling-IC universe top-15 (180d window, 90d refresh)
- K=3 long/short with full gates (conv_gate + PM_M2 + filter_refill)
- V3.1 6-sleeve overlay
- MTM on `alpha_β`

**Pass**:
- Sharpe >= +0.94 (`+0.20` over WINNER_17 beta-residual baseline), or
- Sharpe in `[+0.74, +0.94)` with lower drop-5 std and better model-only diagnostics.

If universal v3 fails this phase, stop and diagnose feature family contributions before trying architecture complexity.

### Phase 6 — Train per-symbol model bank, gated by universal pass (~80 min)

- 51 separate LGBMs, each on one symbol's data
- Features: WINNER_BTC_v3_PERSYM
- **Stricter regularization**: min_data_in_leaf=**200** (vs 100), feature_fraction=**0.7** (vs 0.8), lambda_l2=**5.0** (vs 3.0)
- Reason: ~85k rows per symbol → higher overfit risk

**Output**: `outputs/vBTC_audit_panel_v3_persym/all_predictions.parquet` (concatenated)

**Pass**:
- per-symbol time-series IC >= universal time-series IC for >= 30/51 symbols,
- no severe calibration drift: per-symbol prediction means/stds remain comparable after target z-scoring,
- model-only cross-sectional ranking is not worse than universal.

### Phase 7 — Train hybrid residual bank, gated by per-symbol evidence (~80 min)

- 51 LGBMs, features = WINNER_BTC_v3_PERSYM + `universal_pred`
- `universal_pred` must be out-of-fold for every row used to train the residual bank
- Same regularization as Phase 6

**Output**: `outputs/vBTC_audit_panel_v3_hybrid/all_predictions.parquet`

Strict PIT rule:

```
For every training row of R_s, universal_pred must come from a universal model
that did not train on that row.
```

Do not use in-sample universal predictions for residual-model training.

**Pass**:
- hybrid per-cycle IC >= max(universal_v3, per_symbol_v3) IC + 0.001,
- full V3.1 Sharpe >= best single-stage variant + 0.10,
- prediction scale remains cross-sectionally comparable.

### Phase 8 — V3.1 beta-hedged head-to-head + stress (~30 min)

For each variant: WINNER_17 baseline / universal_v3 / per_symbol_v3 / hybrid_v3:
- Rolling-IC universe top-15 (180d window, 90d refresh)
- K=3 long/short with full gates (conv_gate + PM_M2 + filter_refill)
- V3.1 6-sleeve overlay (4h entry, 24h hold)
- MTM on `alpha_β` (β-hedged execution accounting)
- Capital base: $100

For the best v3 variant: drop-5 random symbol stress test (20 draws). Also run the same model-only diagnostic from Phase 4 so the final decision is not based only on the rolling-IC selector.

**Output**: head-to-head table, per-cycle CSVs

**Decision matrix:**

| outcome | interpretation | action |
|---|---|---|
| Any v3 variant >= +0.94 Sharpe and improves model-only gross/cycle | Materially better than baseline (+0.74) | Adopt that variant; validation phase |
| Best v3 variant +0.74 to +0.94 with lower drop-5 std | Marginal lift / stability gain | Investigate feature importance, possibly iterate |
| All v3 variants worse than baseline | v3 features didn't generalize OR architecture wrong | Diagnostic (Phase 9), possibly revert |
| Hybrid > Universal AND > Per-symbol | Architecture matters, both components add value | Adopt hybrid |
| Universal > Hybrid | Per-symbol residual just adds noise | Adopt universal v3 |
| Per-symbol > Universal AND ≥ Hybrid | Per-symbol architecture is the right choice | Adopt per-symbol |

### Phase 9 — Diagnostics regardless of outcome (~30 min)

- Top 10 feature importance (split + gain) per variant
- Family-level lift table: IC, model-only gross/cycle, full V3.1 Sharpe
- Per-symbol IC distribution: where does each architecture do well?
- Identify any new features with NEGATIVE contribution (drop in v4 iteration)
- If lift is marginal: identify which 3-5 features carry the most signal — prune set further

### Phase 10 — Documentation + commit (~30 min)

- Update `docs/vBTC_ALPHA_RESIDUAL_PROGRESS.md` with v3 results
- Update `docs/vBTC_HYBRID_MODEL_DESIGN.md` with chosen architecture
- Save final WINNER_BTC_v3 feature list as production candidate

## Total budget: staged, 1-5 hours

| phase | time |
|---|---|
| 1. Feature engineering | 30 min |
| 2. Validation + pruning | 15 min |
| 3. Universal family ablation | 30 min |
| 4. Model-only diagnostic | 20 min |
| 5. Full V3.1 universal | 20 min |
| Stop point if universal fails | ~2 hours total |
| 6. Per-symbol bank | 80 min |
| 7. Hybrid bank | 80 min |
| 8. Final comparison + stress | 30 min |
| 9. Diagnostics | 30 min |
| 10. Doc update | 30 min |

## Risk register

| risk | likelihood | mitigation |
|---|---|---|
| Feature engineering bug (look-ahead, mis-aligned timestamps) | medium | Spot-check each feature on AVAX rows manually before training |
| Too many redundant derived features fragment LGBM splits | high | Aggressive 18-24 feature target + family-level ablations |
| LGBM overfit on per-symbol (~85k rows) | medium | Stricter regularization (min_data_in_leaf=200, feature_fraction=0.7) |
| Cross-sectional pred miscalibration | medium | model-only ranking diagnostic + prediction distribution checks by symbol |
| Universal_pred PIT leakage into hybrid | medium-low | Use only out-of-fold universal predictions for every residual-training row |
| Compute time overruns | low | Run per-symbol bank in background, monitor with checkpoints |
| Features that work in isolation don't compose | medium | family-level ablation before full architecture tests |
| Universe selector still noise-dominated (independent of model) | high | Known issue; address separately after model improvements |

## What's NOT in this plan

- OI / liquidation / basis features (deferred — needs new data sources)
- Spot market features (deferred — needs spot price history)
- Sentiment / on-chain (out of scope per project)
- Universe expansion to 111-panel (validate on 51 first; expansion is a follow-up)
- IC selector tuning (separate issue from model architecture; model-only diagnostic added only to isolate model quality)

## Resolved before execute (2026-05-13)

### Baseline verification

All reference Sharpes (WINNER_21 +0.57, WINNER_17 +0.74, WINNER_BTC −0.49 / −1.58) trained on `target_β = α_β / σ_idio` with MTM on `α_β` (no basket leakage). Sources confirmed:
- `scripts/diag_alpha_residual_baseline_and_gates.py` (V0–V3 ablations) — `alpha_A` column is `alpha_β` aliased; sleeve aggregator MTM on `alpha_wide`
- `scripts/diag_winner17_beta_residual_51_vs_111.py` — `panel["alpha_beta"] = return_pct − beta_pit × btc_ret_t`
- `scripts/diag_phase1d_rolling_beta_neutral.py` — source of predictions parquet
- `scripts/diag_4_variant_comparison.py` (variants C/D) — same β-residual target

### Per-symbol regularization design (Phase 6)

Scale from universal `_train` (LR=0.03, num_leaves=63, max_depth=8, min_data_in_leaf=100, ff=0.8, bf=0.8, λ_l2=3.0) to per-symbol scale (~50–85k rows/symbol vs ~3.5M pooled, 40–70× less data):

| param | universal | per-symbol | derivation |
|---|---|---|---|
| learning_rate | 0.03 | **0.02** | slower trajectory; more early-stop chances at lower SNR |
| num_leaves | 63 | **31** | capacity ∝ log(n); half capacity for ~6% data |
| max_depth | 8 | **6** | caps tree shape against deep noise splits |
| min_data_in_leaf | 100 | **200** | leaf-mean SE for σ=1 target drops 0.10→0.071 (30% tighter); per-symbol errors no longer decorrelate cross-section |
| feature_fraction | 0.8 | **0.7** | column-bagging diversity at smaller data |
| bagging_fraction | 0.8 | **0.7** | row-bagging variance reduction |
| lambda_l2 | 3.0 | **6.0** | shrinkage ∝ leaf noise (~2×) |
| min_gain_to_split | 0.0 | **0.005** | skip marginal splits below noise floor |
| num_boost_round | 2000 | **3000** | room for slower LR |
| early_stopping | 80 | **100** | more rounds for cal to detect overfit |

**Sweep before commit**: 2-axis cheap grid on 5 anchor symbols before all-51 commit.
- Anchors: AVAX (validated per-symbol +0.025 IC win), ETH (mature/large-data), LINK (mid-cap moderate vol), ADA (top per-symbol IC loser on 111-panel — highest leverage for "did we fix degradation"), ORDI (newer listing, small-n stress)
- Grid: `min_data_in_leaf ∈ {100, 200, 400}` × `lambda_l2 ∈ {3, 6, 12}`
- Cost: 9 configs × 5 anchors × 9 folds × 5 seeds × ~3s ≈ **30 min**
- Lock criteria: (a) max mean OOS time-series IC across anchors; (b) per-symbol pred-std does not collapse below 50% of universal pred-std on same symbol; (c) AVAX retains the +0.025 IC lift seen at baseline reg
- Apply locked (`min_data_in_leaf`, `lambda_l2`) pair to all 51 symbols in Phase 6 main run

### Adoption threshold

`+0.20` Sharpe over baseline (≥ +0.94 vs WINNER_17's +0.74), AND model-only gross/cycle must also improve. Strictest of the three options — avoids re-running into the WINNER_BTC trap where per-cycle IC stayed similar but tail-pick PnL collapsed because gates were calibrated for a different pred distribution.

### OI data

Defer until v3 succeeds. If v3 passes adoption gate, pull Binance OI history during Phase 8 stress test as v4 parallel prep.

## Files produced

```
scripts/
  build_btc_only_features_v3.py         # Phase 1: feature engineering
  diag_validate_prune_v3_features.py    # Phase 2: IC + correlation pruning
  train_v3_universal.py                 # Phase 3 / 5
  diag_v3_family_ablation.py            # Phase 3
  diag_v3_model_only.py                 # Phase 4
  train_v3_persym_bank.py               # Phase 6
  train_v3_hybrid_bank.py               # Phase 7
  diag_3way_v3_comparison.py            # Phase 8
  diag_feature_importance_v3.py         # Phase 9

outputs/
  vBTC_features_btc_v3/
    panel_v3.parquet                    # all 39 candidate features
    winner_btc_v3_features.json         # final pruned list
    feature_ic.csv
    per_symbol_feature_ic.csv
    block_composite_ic.csv
    correlation_matrix.csv
  vBTC_audit_panel_v3_family_ablation/summary.csv
  vBTC_audit_panel_v3_universal/all_predictions.parquet
  vBTC_audit_panel_v3_persym/all_predictions.parquet
  vBTC_audit_panel_v3_hybrid/all_predictions.parquet
  vBTC_3way_v3/
    summary.csv
    v3_universal_v31.csv
    v3_persym_v31.csv
    v3_hybrid_v31.csv
    stress_test_best.csv
```

## Status

- [x] Plan drafted
- [x] Plan reviewed and optimized
- [x] Baselines verified β-hedged (no basket leakage)
- [x] Per-symbol regularization designed + sweep grid defined
- [x] Adoption threshold + OI timing resolved
- [x] Phase 1 — feature engineering (36 features built, microstructure block dropped)
- [x] Phase 2 — validation + pruning (24 WINNER_BTC_v3 features locked)
- [x] Phase 3 — universal family ablation — **REJECTED**
- [ ] Phase 4–7 — SKIPPED per plan's hard stop after Phase 3 failure
- [ ] Phase 8–10 — documentation only, no further training
- [ ] Phase 2 — validation + pruning
- [ ] Phase 3 — universal family ablation
- [ ] Phase 4 — model-only diagnostic
- [ ] Phase 5 — full V3.1 universal
- [ ] Phase 6 — per-symbol bank, only if gated
- [ ] Phase 7 — hybrid bank, only if gated
- [ ] Phase 8 — final comparison + stress
- [ ] Phase 9 — diagnostics
- [ ] Phase 10 — documentation
