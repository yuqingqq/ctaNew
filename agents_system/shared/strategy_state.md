# Canonical strategy state

The single source of truth for the current strategy stack. Evaluation/Orchestrator update this
when a change is adopted. Agents read this to know what they're modifying.

## Pipeline (top to bottom)
1. **Data**: 5m klines (Binance Vision, free) + funding; HL70 (70 syms) production, 44-sym robustness.
   Panels: `outputs/vBTC_features/panel_3yr_v0.parquet` (44-sym 2023-26),
   `panel_ext2021_v0.parquet` (23-sym 2021-26), HL70 preds cached (see baseline.md).
2. **Features (V0)**: BASE (14: returns, atr, obv, vwap_slope, bars_since_high, autocorr, corr/beta-to-btc,
   idio-vol, funding×3) + cohort (rvol_7d, ret_3d, btc_rvol_7d). All point-in-time (trailing/shifted).
3. **Target**: `alpha_vs_btc_realized` = 4h-fwd ret − β·BTC_fwd_ret; per-symbol z-scored (`target_z`).
4. **Model**: per-symbol Ridge, walk-forward (9 expanding folds, 1-day embargo, label purge by exit_time).
5. **Regime gate (PIT)**: BTC trailing-30d return. bull >+10% / bear <−10% / else sideways.
6. **Construction (held-book)**:
   - bull → rank by mom_30d (momentum), long top-K / short bottom-K
   - sideways → rank by pred (mean-reversion), beta-neutral leg sizing
   - bear → FLAT
   - K=5 each side, 24h hold = 6 overlapping sleeves, equal weight.
7. **Cost**: 4.5 bps/leg (conservative); evaluate also @1bp (HL maker), @3bp (HL taker).

## Current parameters
| param | value | notes |
|---|---|---|
| K (legs/side) | 5 | discrete architecture choice (untuned) |
| hold | 24h (6 sleeves) | |
| regime thresholds | ±10% BTC 30d | |
| TOP_N universe | full universe (no IC pre-filter) | IC-selector found value-negative |
| sign-flip layer | OFF | rejected (universe-specific, unforecastable) |
| GATE/conv_gate | off in held-book (was in V3.1 sleeve) | |

## Open problems / optimization targets (priority order)
1. **Drawdown** (~−57% maxDD) — the central target. Low %positive (39.9%), fat-tail losses.
2. Robustness to universe composition (44-sym 2026 decay vs HL70 health).
3. Tail-risk control without killing Sharpe (prior attempts: invvol HURTS, vol-target-as-written HURTS).

## Things proven NOT to work (see baseline.md "hard truths")
invvol sizing, vol-target lever-up, sector features, cost-margin swap, decay sleeves,
sign-flip (lag & predictor), feature pruning, V5 features, 111/universe expansion via retrain-with-clip.


## DEPLOY UNIVERSE (iter-031 decision)
Trade the WIDEST tradable set — breadth = dispersion = the cross-sectional edge. ALL HL USDT perps ≥6mo history; exclude stables/wrapped/non-crypto-beta (PAXG-gold); liquidity FLOOR for EXECUTION only (~$3-5M/day per capital), NEVER rank/truncate/prune by liquidity or past-IC (both proven value-negative). N≈69. Refresh quarterly keeping breadth maximal. Forward Sharpe ~+1.0 to +2.0 (mean ~+1.5). Kill: rolling-90d Sharpe→0 or maxDD breach while iter-012 stop engaged.
