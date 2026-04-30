# P-2026-001-ml-cta-engine — RESULT

**Status:** STOPPED at Phase 0 gate (i) — FAIL.
**Date:** 2026-04-29.

## Verdict

The pre-registered gate (i) acceptance criteria failed on every falsifiable
probe defined in the plan. Real signal was demonstrated (LGBM 5m kline-only
produced +3.70 bps gross expected return per triggered trade with 47.7% win
rate vs ~41.6% random baseline). The signal is too small to clear the 13 bps
round-trip cost floor at retail (Binance VIP-0) levels.

Per the plan, this triggers a Program stop *before* any Tardis spend or
Track-B (orderbook) work.

## Gate (i) acceptance summary

Plan required **≥4 of 5 walk-forward folds** to satisfy ALL of:
- mean net-of-cost return per triggered trade ≥ 1.5× round-trip cost floor (≥15 bps)
- mean net return positive under stress costs (2× slippage)
- fold-level annualized net Sharpe ≥ 1.0
- ≥150 triggered trades per fold

| Probe | Best gross bps | Best net bps | Folds passing | Verdict |
|---|---|---|---|---|
| 5m linear, k+t, q=0.95 | +0.67 | -12.31 | 0/5 | FAIL |
| 5m linear, label sweep (10 configs) | +0.67 (baseline best) | -12.31 | 0/5 | FAIL |
| 5m LGBM, kline-only, q=0.98 | **+3.70** | **-9.21** | 0/5 | FAIL |
| 1m linear, k+t, q=0.90 | -0.99 | -13.87 | 0/5 | FAIL |
| 1m LGBM, kline-only, q=0.95 | +0.01 | -12.83 | 0/5 | FAIL |

Costs: round-trip 0.05% × 2 fees + 1 bp × 2 slippage + 0.5 × bar-Roll-spread
(median 0.65 bps) ≈ **13 bps**. Stress: ≈ 15 bps with 2× slippage.

## What we actually learned

1. **Stationary feature IC at 400d is real but small.** Top features (`bb_position_10`,
   `tfi_smooth`, `dist_ema_5`, `return_3`, `rsi_5`) sit at mean |IC| ≈ 0.025–0.029,
   block-bootstrap CIs at fwd_6 horizon all exclude zero. This is the typical band
   where ML can find marginal edge — not where rule-based strategies live.

2. **Trade-flow features carry incremental signal over klines, but only barely.**
   `tfi_smooth` ranks 2nd among stationary features. At 5m linear q=0.95,
   kline+tape beats kline-only by +1.5 bps gross consistently across all 5 folds.
   Real but not transformative. Justifies trade-tape collection; does NOT yet
   justify orderbook (Tardis) spend.

3. **LGBM beats linear by ~5×** on signal extraction — 0.67 → 3.70 bps gross.
   The non-linearity is real: linear models map signal onto the wrong region
   (extreme tails behave differently from the middle, e.g., trend continuation
   beats mean reversion in the tails).

4. **Microstructure half-life on SOL is NOT sub-5m.** 1m and 5m comparison shows
   strictly worse gross at 1m. The right decision cadence for SOL CTA appears to
   be 5m or longer — exactly where the rule-based Alpha30 already operates.

5. **Costs are the binding constraint, not the model.** The cost model came in
   tighter than expected (Roll spread median 0.65 bps), but ~13 bps round-trip
   is still 3.5× the best gross we found. Binance VIP-3+ levels (~5 bps RT)
   would change the calculus dramatically — but require sustained volume that
   isn't accessible at retail entry.

## Why this is consistent with the original hypothesis being wrong

The Program's hypothesis was: "microstructure features (orderbook imbalance,
trade flow, queue dynamics) carry predictive content not present in 5-minute
klines, and a learned model can compose features in ways the rules cannot."

What we found:
- Predictive content exists, but it's small and mean-reversion-flavored.
- Rules can in principle capture mean reversion (Alpha30 already does, partly).
- The "learned model composing features" hypothesis was tested via LGBM
  non-linear interactions — gave a real 5× lift, but starting from too low.
- Going to orderbook (gate (ii)) was supposed to lift signal further. The
  IC analysis already showed only one trade-flow feature (`tfi_smooth`) was
  competitive with the best kline features. There's no strong prior that L2
  features (depth/imbalance/microprice from snapshots) would 4× the signal —
  which is roughly what would be needed to clear costs.

## Reusable artifacts

This Program produced infrastructure that future Programs can use:

| Artifact | Purpose | Reusable for |
|---|---|---|
| `data_collectors/binance_vision_loader.py` | Free daily kline + aggTrade dumps from Binance Vision | Any symbol, any cadence |
| `features_ml/klines.py` | Wraps existing `HFFeatureEngine` into ML schema | Any kline-based ML work |
| `features_ml/trade_flow.py` | Streaming bar-aggregated trade-flow features | Any aggTrade-based features |
| `features_ml/labels.py` | Triple-barrier labeler (López de Prado), close-only + intrabar | Any classification target |
| `ml/cv.py` | Walk-forward CV with embargo + label purging | Any time-series ML |
| `ml/cost_model.py` | Roll-spread estimator + cost computation | Any backtest |
| `ml/research/{quick_ic_check, ic_stationary, gate_i_probe, gate_i_lgbm, label_sweep, cadence_probe}.py` | Documented research scripts | Reference for next Program |

400 days of SOL klines + aggTrades (2.6 GB parquet) remains in `data/ml/test/`.

## Recommended follow-up Programs (if pursued)

These are *not* in scope for this Program but are honest options:

1. **Different symbol** — repeat the same probe on BTC or ETH. Different
   microstructure, different liquidity, may carry stronger signal. ~1 day of
   work using existing infrastructure.

2. **Higher VIP cost regime** — only pursue if user has access to VIP-3+
   funded accounts. At ~5 bps RT cost the LGBM 5m result (+3.70 bps gross)
   becomes net positive. Out of scope without that access.

3. **Different decision cadence (longer)** — 15m or 1h. Mean-reversion
   strategies often work better at horizons where signal half-life matches
   trade duration. This Program tested 1m and 5m; longer is unexplored.

4. **Regression target instead of classification** — predict forward return
   directly. The triple-barrier framing may be discarding gradient information
   the model could use.

5. **Pivot to ML augmentation of Alpha30 only** — drop the standalone-ML
   ambition, use ML as a meta-filter on Alpha30 signals (the (a) head of the
   original plan). Smaller scope, lower upside, but builds on what exists.

None of these are recommended without further user input on which direction
matters for their goals.

## Closing notes

The Program was correctly designed: pre-registered, falsifiable, gated. The
gate did its job — caught a strategy that wouldn't work *before* committing
to Tardis spend or orderbook engineering. Total cost of the Program: ~1 day
of research time, no external spend, no production changes. Cleanest possible
negative result.
