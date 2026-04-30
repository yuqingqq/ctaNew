# Status — 2026-04-30

## Program

P-2026-001: ML CTA engine for crypto perpetuals. Goal was a deployable signal
layer extracting alpha from kline + aggTrade data on Binance USDM perps.

## Current state

**Phase: research complete, deployment plausible at h=288 with VIP-3 + maker
execution. Earlier "blocked on cost" conclusion was inflated by a per-bar
cost-accounting bug.**

The strategy class (LGBM regression on alpha-residual targets, with
cross-sectional ranking across 25 symbols) is fully characterized:

- **Signal exists**: rank IC consistently +0.035 across folds and OOS at h=48,
  +0.038 at h=288.
- **Signal is real**: alpha capture of +2.5 bps (h=48) to +7.4 bps (h=288)
  per rebalance verified in β-neutral execution (which strips market noise).
- **Honest cost picture**: with turnover-aware non-overlapping label
  evaluation, OOS net per cycle is **-7.5 bps** (h=48) / **-8.7 bps**
  (h=288) at retail VIP-0. Was reported as -21 under naive per-bar 24-bps
  accounting (over-charged ~2.5×).
- **Signal-quality plan (Apr 30) lifted Sharpe ~3× over the corrected v4
  baseline** at the deployment-relevant tier. Best config under multi-OOS
  validation: **v6 + K=5 + β-neutral**, Sharpe +1.20 at VIP-3+maker, 95% CI
  [-0.78, +3.30] over 270 OOS cycles (9 expanding-WF folds).
- The single-OOS Sharpe of +3.94 (90 cycles, 2026-01-28 to 2026-04-28) was
  partly regime luck — multi-OOS across 9 windows is the more reliable estimate.
- Phase 3 (aggTrades microstructure features): pulled 10 symbols × 402 days,
  audited 19 features. Only `avg_trade_size` passed gates (OOS |IC| 0.035,
  weak). True microstructure (TFI/VPIN/Kyle's λ) doesn't carry signal at
  h=288 — it's a shorter-horizon phenomenon. v8 not pursued.
- v6 = 32 features: v4 base + cross-asset + 7 kline-flow (obv_z_1d, vwap_*, mfi)
  + 8 cross-sectional pctile-rank features.
- Funding-rate features (Phase 4.1) had strongest single-feature OOS IC (up
  to 0.084) but did NOT improve portfolio Sharpe — already captured implicitly
  by v6's basket-relative features.
- **Leakage audit on v6 passed all 4 tests** (forward-peek shift, sanity
  control, xs_rank PIT, embargo).
- See `docs/METHODOLOGY_REVIEW.md` "Apr 30 — Signal-quality plan execution"
  for the full progression and per-phase details.

Caveats:
- Non-overlapping label evaluation, not a full equity-curve backtest.
- Single 90-day OOS window — CIs remain wide despite point-estimate gain.
- Deployment-grade still needs funding accrual, real maker fill modelling,
  drawdown limits, queue-position economics, more OOS data.

See `docs/METHODOLOGY_REVIEW.md` "Apr 30 follow-up" section for full tables.

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

1. `python3 -m scripts.pull_xs_klines` (~20 min) — kline data for 25 symbols
2. `FEATURE_SET=v6 python3 -m ml.research.alpha_v4_xs_1d` (~10 min) — best config
3. `FEATURE_SET=v6 TRIM_UNIVERSE=1 python3 -m ml.research.alpha_v4_xs_1d` — adds IS-trim
4. `FEATURE_SET=v6 python3 -m ml.research.alpha_v4_edge_diagnostic` — diagnostic sections A–G
5. `python3 -m ml.research.alpha_v6_leakage_check` (~3 min) — leakage verification
6. `python3 -m ml.research.alpha_v4_flow_audit` (~1 min) — flow feature IC audit
7. `python3 -m ml.research.alpha_v7_funding_audit` (~2 min) — funding feature IC audit

Other feature sets via `FEATURE_SET=v4|v5|v5_lean|v6|v7|v7_lean`. Defaults to v4.

Funding-rate data is auto-downloaded by `data_collectors/funding_rate_loader.py`
on first import (caches per-symbol parquet to `data/ml/cache/funding_*.parquet`).

Caches build to `data/ml/cache/` on first run; subsequent runs are fast.

## Compute footprint

- Disk: ~700 MB for 25-symbol klines, ~16 GB if pulling BTC/ETH/SOL aggTrades
- RAM: peak ~8 GB during cross-sectional panel assembly
- CPU: training a 5-seed LGBM ensemble on 700K rows × 17 features takes
  ~2-5 minutes
