# vBTC α-residual to BTC — progress and results

Last updated: 2026-05-13

## Strategy definition (universe-portable variant)

Long/short alpha-residual strategy where the residual is defined against BTC (not against a peer basket as in production), so the strategy is universe-portable.

```
For each symbol s at time t:
  β_PIT(s, t)   = OLS slope of trailing 90d 4h returns vs BTC returns, shifted 1 bar (strict PIT)
  alpha_β(s, t) = return_pct(s, t) − β_PIT(s, t) × BTC_return_pct(t)
  σ_idio(s)     = std of alpha_β over fold-0 training residuals (PIT, locked)
  target_β(s, t) = alpha_β(s, t) / σ_idio(s)              ← z-scored, model trains on this
```

Strategy execution:
1. **Universe** — rolling-IC top-15 (180d window, 90d refresh) selected by per-symbol Spearman correlation of past pred vs realized alpha_β. This is the "predictability" filter — picks symbols where the model has been most rank-correlated with α_β over the past window.
2. **Picks** — K=3 long + K=3 short within the universe, ranked by model pred at each 4h entry.
3. **β-hedged execution** — at the portfolio level, the K=3 long basket has some implicit BTC β, and so does the K=3 short basket. A net BTC short of (β_long − β_short) × position_size neutralizes the implicit β. By construction the hedged portfolio earns the realized α_β spread.
4. **V3.1 6-sleeve overlap** — each 4h entry opens a new sleeve at 1/6 capital weight, held 24h, MTM'd every 4h. Active portfolio is the sum of the 6 most-recent sleeves.
5. **Cost** — V3.1 turnover-based, 2.25 bps per unit absolute weight delta (matches production calibration).

## How it differs from production

| component | production | this version |
|---|---|---|
| residual definition | `alpha_A = return − basket_mean_51` | `alpha_β = return − β_PIT × BTC_return` |
| per-symbol normalization | std of alpha_A | std of alpha_β |
| **target** | **z-scored = alpha_A / per_sym_std** | **z-scored = alpha_β / σ_idio** |
| model | LGBM 5-seed ensemble, WINNER_21 features | identical |
| training procedure | 10 folds × 5 seeds, expanding window, PIT 60d eligibility | identical |
| IC ranker (predictability filter) | per-symbol Spearman, 180d/90d | identical |
| K=3 picks within universe | by pred-rank | by pred-rank |
| V3.1 sleeve overlay | yes | yes |
| **MTM accounting** | **raw return_pct** | **alpha_β (β-hedged execution)** |

The only structural difference is **what "residual" means** and what's hedged at execution. Everything else is identical to production.

Per-symbol residual std comparison (illustrative):

| symbol | prod `alpha_A` std (vs 51-basket) | this `alpha_β` std (vs BTC) | ratio |
|---|---|---|---|
| ADAUSDT | 0.007 | 0.012 | 1.7× |
| LINKUSDT | 0.007 | 0.012 | 1.7× |
| AVAXUSDT | 0.008 | 0.013 | 1.5× |
| ZECUSDT | 0.026 | 0.028 | 1.05× |
| BIOUSDT | 0.029 | 0.032 | 1.1× |

β-residual is uniformly *larger* than basket-residual because the 51-basket captures the common alt factor (alt-beta) that BTC alone does not — basket-residualization is "purer" idio relative to peers, while β-residualization is purer idio relative to BTC only.

## Ceiling tests (run earlier this session)

For context on what's theoretically possible under V3.1's structure with perfect predictions:

| oracle variant | what it picks by | Sharpe | end-equity $100 |
|---|---|---|---|
| Oracle by realized 4h α_β (no gates, full 51) | 4h α_β | +39.7 | $3,031 |
| Oracle by realized 24h α_β (no gates, full 51) | 24h cumulative (matches V3.1 hold) | — | — |
| Phase 1D actual (model preds + gates, un-hedged) | model pred | +0.65 | $124.40 |

The 4h-oracle test confirmed the ceiling is far above what any noisy model achieves. The realistic capture of V3.1's structure under noisy picks is bounded by both (a) prediction quality and (b) the gates filtering noisy preds. The α-residual setup is structurally portable; the question is what the gates do to it.

## Baseline + gate ablations (β-hedged execution)

Setup: rolling-IC top-15 universe, K=3 long/short, β-hedged MTM (α-PnL accounting), V3.1 6-sleeve, $100 capital, 9-month OOS (1620 cycles, folds 1-9 in walk-forward).

Variants in order of increasing gating:
- **V0 baseline** — no gates: every cycle trades
- **V1** — + conv_gate (skip cycles where pred dispersion < 30th pctile of past 252)
- **V2** — + PM_M2 (require symbol to appear in past 2 cycle candidates before entering basket)
- **V3** — + filter_refill (trailing 90d positive-PnL discipline per (sym, side); = full Phase 1D stack with β-hedge)

### Results

| variant | Sharpe | end-equity $100 | PnL % | traded cycles | folds + |
|---|---|---|---|---|---|
| **V0 baseline (no gates)** | **−0.52** | **$82.76** | **−17.2%** | 100% (1620/1620) | 4/9 |
| V1 + conv_gate | −0.49 | $86.30 | −13.7% | 64% (1030/1620) | 3/9 |
| V2 + conv_gate + PM_M2 | +0.21 | $108.35 | +8.3% | 43% (689/1620) | 2/9 |
| **V3 + conv_gate + PM_M2 + filter_refill** | **+0.57** | **$121.18** | **+21.2%** | 44% (709/1620) | 4/9 |

### Marginal contribution of each gate

| gate added | Sharpe lift | mechanism |
|---|---|---|
| conv_gate (skip low-dispersion cycles) | +0.03 | small — pred dispersion already weak signal |
| PM_M2 (persistence) | **+0.70** | huge — filters spurious one-cycle picks |
| filter_refill (trailing PnL discipline) | **+0.36** | significant — picks symbols with persistent positive contribution |
| TOTAL (V0 → V3) | **+1.09** | gates contribute more than the raw signal |

### Per-cycle stats

| variant | gross/cycle bps | cost/cycle bps | turnover/cycle |
|---|---|---|---|
| V0 | +0.07 | 1.13 | 0.503 |
| V1 | −0.10 | 0.75 | 0.333 |
| V2 | +1.09 | 0.58 | 0.257 |
| V3 | +1.90 | 0.60 | 0.265 |

Cost falls sharply as gates filter more cycles; gross/cycle quality rises because each remaining cycle is higher conviction.

## Comparison with un-hedged Phase 1D (production-style execution)

The exact same model + protocol + universe + V3.1 sleeve, but un-hedged execution (MTM on raw return_pct, captures implicit β-carry on top of α):

| | un-hedged (Phase 1D actual) | β-hedged (V3 here) |
|---|---|---|
| Sharpe | +0.65 | +0.57 |
| end-equity $100 | $124.40 | $121.18 |
| PnL % | +24.4% | +21.2% |

The β-hedge gives up ~+0.08 Sharpe / $3 of end-equity. That delta is the BTC-momentum carry the un-hedged strategy accidentally collected. The remaining +0.57 Sharpe is genuine α-extraction.

## Findings

1. **Baseline (no gates) is unprofitable.** With β-hedged execution and the IC-filtered universe, raw K=3 picks every cycle lose money. The model's per-cycle pred has too much noise to act on indiscriminately.

2. **PM_M2 persistence is the most valuable gate.** It accounts for +0.70 of +1.09 total gate lift. The two-cycle persistence requirement filters out spurious one-time picks driven by transient model artifacts.

3. **filter_refill (trailing PnL discipline) adds +0.36.** Symbols whose trailing 90d mean PnL contribution is negative are excluded. This is a "let winners ride, cut losers" symbol-level discipline.

4. **conv_gate (cycle skipping) is marginal.** Only +0.03 Sharpe. Dispersion-based cycle skipping doesn't add much for this β-residual setup — pred dispersion is too noisy a signal.

5. **The full gated stack (V3) recovers most of un-hedged Phase 1D's Sharpe** (+0.57 vs +0.65). The β-hedge costs ~0.08 Sharpe of BTC-carry but yields a portable strategy.

6. **The strategy is dominated by the gates, not the prediction.** Of the +0.57 final Sharpe, +1.09 comes from gates relative to the baseline. The raw prediction at every cycle contributes near-zero (V0 = −0.52). This means the strategy's edge is heavily "noise filtering" rather than "model accuracy on the picks".

## What this implies for next steps

- The strategy as-built is **portable** (β-residual target, no peer basket needed) but **mostly works through gating**, not through prediction accuracy.
- The gates were calibrated for production noisy-prediction characteristics; they should be re-validated for the β-residual setup, particularly conv_gate which adds nothing here.
- The +0.57 Sharpe β-hedged baseline is much lower than the ceiling (Sharpe ~40 with perfect picks under V3.1 structure), so prediction quality is the main lever for improvement, not strategy structure.

## BTC-only feature rebuild (2026-05-13, follow-up)

After identifying that WINNER_21 contains universe-bound features (`_vs_bk` against basket, `xs_rank` against panel, `sym_id` against alphabetical position), built a clean universe-invariant feature set:

### WINNER_BTC (25 features)

| group | features | count |
|---|---|---|
| (1) BTC residual momentum (multiple horizons) | `idio_ret_to_btc_12b/48b/288b` | 3 |
| (2) BTC residual price level | `dom_btc_z_1d`, `dom_btc_change_48b/288b` | 3 |
| (3) BTC β/corr state | `beta_to_btc`, `beta_to_btc_change_5d`, `corr_to_btc_1d`, `corr_to_btc_change_3d` | 4 |
| (4) BTC residual risk | `idio_vol_to_btc_1h/1d`, `idio_vol_ratio_to_btc` | 3 |
| (5) BTC market regime (same for all syms at t) | `btc_ret_48b`, `btc_realized_vol_1d/30d` | 3 |
| (6) Single-name flow/funding | `atr_pct`, `obv_z_1d`, `vwap_slope_96`, `funding_rate`, `funding_rate_z_7d`, `funding_rate_1d_change` | 6 |
| (7) Stable per-symbol context (replaces sym_id) | `listing_age_days`, `log_quote_volume_90d`, `residual_vol_90d_own_pctile` | 3 |

Properties: every feature is **universe-invariant** — its value for symbol s at time t does not depend on what else is in the panel. No basket references, no cross-sectional ranks, no alphabetical encodings.

### 4-variant A/B/C/D comparison (β-hedged, all gates ON)

| variant | features | universe | Sharpe | end-equity $100 | drop-5 std |
|---|---|---|---|---|---|
| **A** | WINNER_21 (old, sym_id, _vs_bk, xs_rank) | rolling-IC top-15 | **+0.57** | **$121.18** | **0.955** |
| B | WINNER_21 | liquidity top-30 | −1.38 | $74.13 | 0.622 |
| C | WINNER_BTC (new, universe-invariant) | rolling-IC top-15 | −0.49 | $85.61 | **0.305** |
| D | WINNER_BTC | liquidity top-30 | −1.58 | $70.05 | 0.369 |

### Findings

**Portability hypothesis: CONFIRMED**

The drop-5 random-symbol stress test (20 draws, same seed across variants) shows:
- A: std = 0.955 (highly universe-sensitive)
- C: std = 0.305 (**3× more stable** than A)
- D: std = 0.369 (2.6× more stable than A)

Universe-invariant features deliver structural portability as designed.

**Absolute Sharpe: REJECTED**

Variants C and D are both negative-Sharpe in baseline. The structural fix gives up the in-sample alpha:
- A − C delta in absolute Sharpe = **+1.06** (the cost of portability)
- A − C delta in drop-5 std = **−0.65** (the benefit of portability)

Per-cycle IC barely shifts (+0.0149 → +0.0164), but tail-pick PnL collapses. Three mechanisms:

1. **sym_id removal** costs ~+2.6 Sharpe (measured directly earlier). Stable context features partially compensate but don't recover the per-symbol identity signal.
2. **`_vs_bk` → `_vs_btc`** substitution changes feature semantics. LGBM has to relearn what each feature means; the new features individually have similar |IC| but the model's combined alpha-extraction is weaker.
3. **Production gates calibrated for WINNER_21's pred distribution** — different feature set produces different pred distribution → gates fire differently, often suboptimally.

**Liquidity universe is worse than IC universe**

Both A vs B and C vs D show IC top-15 beats liquidity top-30 in absolute Sharpe. Mechanism: liquidity ranking picks BIG names (BTC-followers, blue chips). The IC ranker, despite being noise-dominated, accidentally surfaces alpha-rich mid-cap names (VVVUSDT, AXSUSDT, ORDIUSDT etc.) that carry the actual cross-sectional alpha.

### Decision branch outcome

Per the user's pre-stated criteria: "If D beats A or is at least more stable under drop-symbol tests, then the BTC-only rebuild is working."

D does NOT beat A in absolute Sharpe (−1.58 vs +0.57).
D IS more stable (std 0.369 vs 0.955, **2.6× improvement**).

The rebuild "works" in the portability sense but at substantial Sharpe cost. **The strategy's current +0.57 Sharpe is substantially universe-overfit alpha** that cannot be captured under a portable feature/universe specification.

### Implications

1. **Free-data 4h β-residual α-extraction is structurally limited.** The universe-portable ceiling for this strategy is probably +0.5 to +1.0 Sharpe given our oracle test (theoretical max +40, but the realistic capture given 4h horizon, alt-perp noise, sleeve overlap, and gates is much lower).

2. **The current +0.57 production-style Sharpe is mostly universe-specific alpha** generated by per-symbol identity (sym_id) and basket-residual dynamics. Removing those layers reveals the underlying portable component is near-zero.

3. **Three forward directions emerge:**
   - **(a) Accept non-portability**: freeze the 51-symbol universe, retrain annually, kill-switch on drawdown. Take the +0.57 with universe-noise risk.
   - **(b) Improve portable features**: try richer per-symbol context features, sophisticated ML architectures, or rank-aware loss functions. Diminishing returns expected.
   - **(c) Recognize the ceiling**: stop chasing +2 Sharpe portable α. Either deploy +0.57 with risk-acknowledgment, or move to different signal sources (richer data, longer horizons, different strategies).

### Scripts and outputs

- `scripts/build_btc_only_features.py` — initial feature engineering (created _x/_y duplicates due to existing panel having some features pre-computed)
- `scripts/build_btc_only_features_v2.py` — clean panel resolution, defines WINNER_BTC, IC sanity check
- `scripts/train_btc_only_model.py` — train model on β-residual target with WINNER_BTC features
- `scripts/diag_4_variant_comparison.py` — 4-variant baseline Sharpe comparison
- `scripts/diag_4_variant_drop_stress.py` — drop-5 random symbol stress test
- Output dirs: `outputs/vBTC_features_btc_only/`, `outputs/vBTC_audit_panel_btc_only/`, `outputs/vBTC_4variant_comparison/`, `outputs/vBTC_4variant_stress/`

## WINNER_17 retrain + 111-panel diagnostic (2026-05-13, round 2)

After the BTC-only feature rebuild test (variant C) gave universe-portable but unprofitable Sharpe, came back to a smaller targeted change: keep WINNER_21 architecture but drop only the 4 dead-weight features identified in the audit (`mfi`, `price_volume_corr_20`, `idio_ret_48b_vs_bk`, `funding_streak_pos`, each <0.5% LGBM gain). WINNER_17 = WINNER_21 minus those 4 = 17 features. Still keeps `sym_id`, basket features, xs_rank features.

### Test 1: WINNER_17 + β-residual on 51-panel

| stack | per-cycle IC | Sharpe | end-equity $100 |
|---|---|---|---|
| WINNER_21 + β-residual + β-hedged (baseline) | +0.0149 | +0.57 | $121.18 |
| **WINNER_17 + β-residual + β-hedged** | **+0.0157** | **+0.74** | **$126.96** |
| **Δ from dropping 4 dead features** | +0.0008 | **+0.17** | **+$5.78** |

Confirms audit's prediction: dead-weight features dilute `feature_fraction=0.8` sampling at each tree, removing them tightens the model. **+0.17 Sharpe is a clean gain.**

### Test 2: WINNER_17 + β-residual on 111-panel (full retrain)

| panel | per-cycle IC | Sharpe | end-equity $100 | folds + |
|---|---|---|---|---|
| 51-panel WINNER_17 | +0.0157 | +0.74 | $126.96 | 4/9 |
| **111-panel WINNER_17** | **−0.0087** | **−0.50** | **$69.11** | 4/9 |

**Per-cycle IC went NEGATIVE on the 111-panel.** Predictions are slightly anti-correlated with realized α_β. Sharpe collapses to −0.50 (vs +0.74 on 51).

### Why 111-panel still fails despite β-residual target

Per-symbol IC diagnostic (`scripts/diag_why_111_fails.py`):

| symbol group on 111-panel | mean per-cycle IC | positive |
|---|---|---|
| Original 51 symbols | **−0.0138** | 10/51 |
| New 60 symbols | −0.0098 | 15/60 |

Cross-panel comparison (same 51 symbols, 51-model vs 111-model):
- **36/51 of original symbols LOST IC** under the 111-trained model
- Only 14/51 improved
- Mean Δ IC (111 − 51) = **−0.0080**

Top 10 IC drops on the SAME original symbols:

| symbol | 51-model IC | 111-model IC | Δ |
|---|---|---|---|
| ADAUSDT | +0.028 | **−0.025** | **−0.053** |
| LINKUSDT | +0.016 | −0.018 | −0.034 |
| ONDOUSDT | +0.016 | −0.018 | −0.034 |
| DOTUSDT | −0.003 | −0.034 | −0.032 |
| ETHUSDT | +0.011 | −0.020 | −0.031 |
| TONUSDT | +0.002 | −0.028 | −0.031 |
| LDOUSDT | +0.018 | −0.006 | −0.024 |
| BCHUSDT | +0.029 | +0.008 | −0.022 |
| AAVEUSDT | +0.012 | −0.010 | −0.022 |

**The 111-trained model produces worse-than-random predictions on the same symbols the 51-model handled well.**

### Mechanism: training-set contamination

β-residual target IS universe-portable (independent of basket composition). So why does the 111-trained model fail on the original 51?

The 60 new symbols added in the 111-panel have:
- Short listing histories (mostly Q4 2024 / Q1 2025 listings)
- Meme-coin behavior (FARTCOIN, MELANIA, BROCCOLI, JELLYJELLY, PUMP, ZEREBRO, etc.)
- Noisy data with thin early volume

When LGBM trains on 111 pooled rows:
- Tree splits "average" across mature alts and memes
- Patterns that work for AVAX in low-vol regimes get fit against FARTCOIN in meme spikes
- Feature importance shifts toward discriminating memes
- Model becomes "median symbol" predictor — degrades on **everything**, including the original 51

**This is a training-set composition issue, not a target/feature design issue.** β-residual target was the right conceptual fix; it's just not sufficient on its own.

### The three things needed for genuine universe expansion

| layer | status |
|---|---|
| Universe-portable target (β-residual against fixed BTC) | ✅ done in Phase 1D |
| Universe-portable features (no `_vs_bk`, no `xs_rank`, no `sym_id`) | partial — WINNER_17 still has 5 basket-referenced features (`bk_ema_slope_4h`, `dom_change_288b_vs_bk`, `dom_level_vs_bk`, `corr_change_3d_vs_bk`, `idio_vol_1d_vs_bk_xs_rank`) |
| Training-set quality filter (no short-history / noisy symbols pollute training) | ❌ NOT done |

The missing piece is (3). The 60 new symbols pollute training even when the target is portable.

### Architecture alternative: per-symbol vs universal model

Per-cycle IC went negative on the 111-panel because the **universal LGBM model** must average patterns across 111 heterogeneous symbols. ADA's prediction is influenced by FARTCOIN's data through shared tree splits.

Three viable architectures going forward:

| approach | data per fit | cross-symbol signal | new-symbol cost |
|---|---|---|---|
| A. Per-symbol (51-111 models) | ~85k rows each | none | train 1 new model |
| B. Per-cluster (5-7 models, e.g., Phase G's K=6 clustering) | ~850k-1.3M each | within-cluster | classify into existing cluster |
| C. Universal + rich context block (8-10 stable per-symbol features replacing sym_id) | full ~4.3M | full | retrain universal |

Current setup is C with thin context (just sym_id). Prior tests Phase SEG (2-way symbol split) and Phase CAL (per-symbol calibration of universal pred) both failed but didn't try **true per-symbol model training**.

### Open question for next investigation

Does per-symbol model training fix the 111-panel issue? Test plan:
1. Cheap diagnostic: train AVAXUSDT-only model on 51-panel data → compare per-cycle IC to universal model's AVAX prediction (~10 min)
2. If positive: scale to 51-symbol per-symbol bank, run V3.1 β-hedged (~80 min)
3. Also test: universal + 8-10 stable context features (~5 min)

### Scripts and outputs (round 2)

- `scripts/diag_winner17_beta_residual_51_vs_111.py` — WINNER_17 retrain on both panels
- `scripts/diag_why_111_fails.py` — per-symbol IC decomposition diagnostic
- Outputs: `outputs/vBTC_winner17_b_residual/`

## v3 feature re-engineering REJECTED (2026-05-13, round 3)

After identifying universe-overfit ceiling from round 2, drafted a 39-candidate
v3 feature plan (`docs/vBTC_V3_FEATURE_PLAN.md`) with 7 process-type families:
liquidity, β multi-window, residual behavior, anchoring, funding crowding,
microstructure, process fingerprint. Goal: build universe-invariant feature
set that captures process-type characteristics ("trend-follower vs
mean-reverter", "high-β-stable vs unstable") and beats WINNER_17 +0.74.

### Phase 1: built 36-feature panel (microstructure block F dropped — only 25/51 symbols had aggTrade data, structural 51% missing-rate confound).

### Phase 2: pruned to 24-feature WINNER_BTC_v3 via:
- Per-feature cross-sectional IC vs alpha_β (strongest: idio_max_abs_12b -0.040, resid_vol_30d -0.040)
- Per-symbol time-series IC distribution
- Block composites (B_btc_relationship +0.0321, G_process_fp +0.0312, D_trend_anchor +0.0238, C_resid_behavior +0.0222, A_liquidity +0.0136, E_funding +0.0083 — all blocks pass +0.005 gate)
- Pairwise correlation pruning (dropped 3 redundant pairs)
- Trim-to-24 by absolute IC

### Phase 3: REJECTED — all v3 variants hurt Sharpe

Apples-to-apples comparison on `panel_variants_with_funding.parquet` with the
same β-residual computation as the +0.74 baseline (90d × 288 bar β on 4h
forward returns):

| variant | #feats | Sharpe | Δ vs WINNER_17 | per-cycle IC | folds+ |
|---|---|---|---|---|---|
| V0 WINNER_17 (reproduce) | 17 | **+0.74** | — | +0.0157 | 4/9 |
| V1 W17 + v3_aug_8 (top-8 unique) | 25 | -0.67 | **-1.41** | +0.0139 | 4/9 |
| V2 W17 + v3_aug_4 (top-4 unique) | 21 | -4.97 | **-5.71** | +0.0116 | 1/9 |

Pure WINNER_BTC_v3 (24 features alone, no WINNER_17 base) failed at Sharpe
-1.86 with per-cycle IC -0.002. Diagnosed cause: all v3 features computed at
1d granularity then forward-filled to 5m, so they're constant within each day.
LGBM has no within-day variation to learn from (`best_iteration=1` in many
folds → model gives up).

### Why augmenting WINNER_17 also fails

When v3 features are added to WINNER_17's 5m-resolution base, LGBM CAN learn
from them (best_iter goes up). Per-cycle IC barely shifts (-0.002 to -0.004).
But tail-pick PnL collapses: -0.54 to -9.16 bps/cycle vs WINNER_17's +2.33 bps.

Mechanism: same "IC vs Sharpe disconnect" pattern from Phase G (sector
features), Phase H (WINNER_16 pruning), Phase Q (WINNER_23 retrain). Adding
features to a tightly-fit universe-specific WINNER_17 model fragments splits
and degrades tail picks. The model's split structure was implicitly optimized
for the 17-feature distribution; perturbing it disrupts the rank discrimination
at the head/tail of the prediction distribution where K=3 picks come from.

### Per the plan's strict Phase 3 stop criterion

"If this phase fails, stop. Do not train per-symbol or hybrid banks." — User
designed this gate exactly for this case. v3 is the 44th direction tested in
vBTC research and is also rejected.

### Files produced

- `scripts/build_btc_only_features_v3.py` — Phase 1 feature builder (39 nominal, 36 effective)
- `scripts/diag_validate_prune_v3_features.py` — Phase 2 IC + composite + pruning
- `scripts/train_v3_universal.py` — Phase 3 pure-v3 trainer (failed -1.86)
- `scripts/train_v3_augment.py` — Phase 3 W17+v3 on v3-panel (mismatched β — invalid)
- `scripts/train_v3_augment_v2.py` — Phase 3 W17+v3 on original panel (apples-to-apples, valid result)
- `outputs/vBTC_features_btc_v3/panel_v3.parquet` (5.85M rows × 139 cols)
- `outputs/vBTC_features_btc_v3/winner_btc_v3_features.json` (24-feature list)
- `outputs/vBTC_features_btc_v3/feature_ic.csv`
- `outputs/vBTC_features_btc_v3/per_symbol_feature_ic.csv`
- `outputs/vBTC_features_btc_v3/block_composite_ic.csv`
- `outputs/vBTC_features_btc_v3/correlation_matrix.csv`
- `outputs/vBTC_audit_panel_v3_universal/` (pure-v3 predictions + V3.1 csv)
- `outputs/vBTC_audit_panel_v3_augment_v2/` (apples-to-apples comparison)

### Forward implications

The free-data Binance perp 4h-horizon β-residual signal extraction is
**closed**. The current production stack (WINNER_17 + β-hedged + V3.1 sleeve)
at Sharpe +0.74 is the local optimum. Forward Sharpe expectation should
include universe-overfit risk; mitigations are operational (annual retrain,
kill-switch on DD), not signal-side.

Possible directions if appetite remains (none likely to succeed given pattern):
- (a) Move beyond free Binance perp data (Glassnode, on-chain, options)
- (b) Different horizon (1h or 1d instead of 4h)
- (c) Different target (e.g. directional 4h sign instead of residual magnitude)
- (d) Different model class (LambdaRank or GBM ensemble — already rejected in Phase RANK)

## Scripts and outputs (round 1 — initial β-hedged ablations)

- Strategy implementation: `scripts/diag_alpha_residual_baseline_and_gates.py`
- Per-cycle CSVs: `outputs/vBTC_alpha_residual_gates/V0_baseline_no_gates.csv`, `V1_…csv`, `V2_…csv`, `V3_…csv`
- Summary: `outputs/vBTC_alpha_residual_gates/summary.csv`
- Predictions used: `outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet` (Phase 1D's β-residual retrain)
- Strategy parameters (matching production):
  - N_universe = 15, K = 3, sleeve_n = 6, hold = 24h, entry = 4h
  - GATE_PCTILE = 0.30, GATE_LOOKBACK = 252 cycles
  - PM_M = 2 (persistence)
  - filter_refill window = 90 days
  - COST_PER_LEG = 4.5 bps, COST_PER_UNIT_ABS_DELTA = 2.25 bps
  - β_PIT window = 90 days, σ_idio from fold-0 training only
