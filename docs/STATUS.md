# Status — 2026-04-30

## Program

P-2026-001: ML CTA engine for crypto perpetuals. Goal was a deployable signal
layer extracting alpha from kline + aggTrade data on Binance USDM perps.

## Current state

**Phase: research complete, deployment blocked on cost economics.**

The strategy class (LGBM regression on alpha-residual targets, with
cross-sectional ranking across 25 symbols) is fully characterized:

- **Signal exists**: rank IC consistently +0.035 across folds and OOS.
- **Signal is real**: alpha capture of +5–10 bps per trade verified in
  hedged execution (which strips market noise).
- **Signal is too small**: 5–10 bps alpha vs 12 bps naked / 24 bps hedged
  cost at retail VIP-0.

## What works

| Component | Status |
|---|---|
| Binance Vision data loader (klines + aggTrades) | ✅ |
| Feature pipeline (160+ kline + 22 alpha-tailored + cross-asset + cross-sectional) | ✅ |
| Walk-forward CV with embargo + label purging | ✅ |
| Pooled multi-symbol training | ✅ |
| Cross-sectional ranking and portfolio P&L | ✅ |
| Cost model (fee + slip + Roll-spread; per-leg hedged) | ✅ |
| Look-ahead bug detection (Sharpe target shift, VPIN bucket) | ✅ |

## Known issues / debts

1. **Trigger-rate calibration breaks under regime shift** — q=0.95 on cal
   doesn't translate to OOS when prediction distributions widen. SOL
   especially: 5% calibrated → 68% OOS trigger rate. Per-symbol thresholds
   help BTC/ETH but not SOL. Workaround: use rank-based selection (top-K
   per bar) instead of magnitude threshold — built into v4.

2. **`sym_id` underused by LGBM** (0.04% importance in v3) despite per-symbol
   IC sign reversals. Suggests trees don't naturally partition on a
   low-cardinality categorical. Workaround tried: per-symbol heads. Did not
   significantly improve at current sample size.

3. **AggTrades are 16 GB for 3 symbols × 400d** — cross-sectional v4 uses
   kline-only features for the 25-symbol universe. Adding aggTrade features
   (TFI, VPIN, Kyle's λ) for all 25 would be ~130 GB; exceeds local disk.

4. **Hyperparameter selection bias** — LGBM params (num_leaves=63,
   min_data_in_leaf=50, lambda_l2=3.0) and trigger config (q=0.95, h=48)
   were chosen by reviewing all walk-forward folds. Some selection bias
   bakes into the WF results.

## Reproducibility

All results in `docs/METHODOLOGY_REVIEW.md` reproducible from this repo:
1. `python3 -m scripts.pull_xs_klines` (~20 min)
2. `python3 -m ml.research.alpha_v4_xs` (~15 min)
3. `python3 -m ml.research.alpha_v4_concentrated` (~30 min)

Caches build to `data/ml/cache/` on first run; subsequent runs are fast.

## Compute footprint

- Disk: ~700 MB for 25-symbol klines, ~16 GB if pulling BTC/ETH/SOL aggTrades
- RAM: peak ~8 GB during cross-sectional panel assembly
- CPU: training a 5-seed LGBM ensemble on 700K rows × 17 features takes
  ~2-5 minutes
