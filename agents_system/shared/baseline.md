# Baseline (frozen)

The fixed reference all iterations are measured against. **Do not edit** except to correct errors;
the evolving champion lives in `current_best.md`.

## Strategy
Cross-sectional alpha-residual held-book with regime hybrid:
- **Universe (production): HL70** (70 Hyperliquid-tradable perps). Robustness universe: 44-sym Binance set.
- Model: V0 (Ridge per-symbol, BASE+cohort features) — HL70 uses the cached V5_mv3+7crossX preds.
- Target: 4h-forward alpha-residual (`alpha_vs_btc_realized` = ret − β·BTC_ret), per-symbol z-scored.
- Construction: **regime hybrid** — momentum (mom_30d) in bull, mean-reversion (pred) in sideways
  with beta-neutral leg sizing, FLAT in bear. Regime = BTC trailing-30d return (±10% thresholds).
- Held-book: K=5 long / K=5 short, 24h hold = **6 overlapping sleeves**, equal weight.
- Cost: 4.5 bps/leg baseline (conservative); HL maker ~1bp, taker ~3bp.

## Baseline metrics (HL70, from X117, 2025-03→2026-05, 402d, cost 4.5bps)
| metric | value |
|---|---|
| Sharpe (ann) | **+1.93** |
| total PnL | +10,472 bps |
| **maxDD** | **−5,674 bps (−57%)** ← prime reduction target |
| Calmar | +1.68 |
| %positive cycles | 39.9% |
| 2025 (Apr+) | Sharpe +1.42 (maxDD lives here) |
| 2026 | Sharpe +3.44, maxDD −1,704 bps |
| Sharpe @1bp / @3bp | +2.19 / +2.04 |

## 44-sym robustness reference (X97/X116)
Base K=5 Sharpe +1.84, maxDD −4,170 bps; note 2026 decays to −1.19 on this universe
(a composition artifact — HL70 2026 is +3.44).

## Known hard truths (do not re-litigate without new data)
- Construction-layer tweaks, feature additions (V5≈V0), sector features, cost-margin swaps,
  decay-weighted sleeves, and the per-symbol sign-flip predictor have all been tried and
  **failed honest OOS**. See `memory/project_convexity_portable_phase2.md` (X1–X116) and
  `project_vBTC_status.md`. Don't repropose these unchanged.
- The deep drawdown (~−57%) is the strategy's nature (low %positive, fat tails) and appears on
  every universe — it is the central unsolved problem and the main optimization target.
- The 2026 "alpha decay" is specific to the 44-sym composition; HL70 is healthy.

## Scripts that define the baseline backtest
- `research/convexity_portable_2026-05-20/scripts/X117_hl70_pnl_cost.py` (HL70 base PnL/DD/cost)
- `research/convexity_portable_2026-05-20/scripts/X116_hl70_lagging_flip.py` (held-book engine + regime hybrid)
- preds: `research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet`
