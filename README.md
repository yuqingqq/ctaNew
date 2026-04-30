# ctaNew — Alpha-Residual ML Research

Cross-sectional and pair-trading alpha-residual prediction on USDM perpetual
futures. ML pipeline for predicting symbol-specific alpha (return after
stripping market beta), trained on Binance Vision public archives.

This repo contains the research code, methodology document, and reproducible
probes from the P-2026-001 program. **It is not a production trading system** —
no exchange adapters, no execution server, no live bot.

## What this is

Four iterative attempts at extracting tradeable alpha from 5-minute crypto
perp data, ending in a cross-sectional ranking model across 25 symbols:

| Version | Approach | Outcome |
|---|---|---|
| v1 | LGBM on raw forward return, 18 features | OOS net -2 bps; ETH-driven, fragile |
| v2 | + cross-asset (spread, beta, ref features) | WF alpha flipped to +6, OOS still negative |
| v3 | Audit-driven 17 alpha-tailored features + sym_id | Consistent OOS IC across symbols (+0.06–0.08), but threshold calibration breaks down OOS |
| v4 | Cross-sectional ranking across 25 symbols | Stable rank IC +0.035, spread alpha +6 bps WF / +3.6 OOS, robust but below cost line |

**Bottom line**: structural alpha in OHLCV+aggTrade signals at 4h horizon is
~5–10 bps gross — confirmed three ways. Retail VIP-0 cost (~12 bps naked,
~24 bps hedged) sits above the alpha line. See `docs/METHODOLOGY_REVIEW.md`
for the full audit trail.

## Repo layout

```
hf_features.py                 # 160+ kline-based features (technical indicators)
features_ml/
  klines.py                    # wrapper over hf_features
  regime_features.py           # ATR z-scores, distance from highs, etc.
  trade_flow.py                # bar-level aggTrade aggregation (TFI, VPIN, Kyle's λ)
  cross_asset.py               # pairwise: spread, beta, correlation
  alpha_features.py            # alpha-tailored: dom_level, idio_vol, etc.
  cross_sectional.py           # 25-symbol basket + cross-sectional alpha
  labels.py                    # triple-barrier and Sharpe-normalized labels
ml/
  cv.py                        # walk-forward folds with embargo + label purging
  cost_model.py                # fee/slippage/Roll-spread cost model
  research/
    alpha_v2.py                # v2 head-to-head probe
    alpha_v3.py                # v3 alpha-tailored model
    alpha_v3_*.py              # v3 variants (per-symbol thr, hedged, BTC+ETH-only)
    alpha_v4_xs.py             # v4 cross-sectional ranking
    alpha_v4_concentrated.py   # v4 with top-1/2/5 + conviction filter
    alpha_*_audit.py           # IC audits per feature × target × symbol
    alpha_review.py            # PnL decomposition (alpha vs market vs cost)
    trend_pooled_v2.py         # symbol-feature builder with disk caching
data_collectors/
  binance_vision_loader.py     # download daily kline + aggTrade archives
scripts/
  pull_xs_klines.py            # pull 25-symbol kline universe
docs/
  METHODOLOGY_REVIEW.md        # full audit trail + findings
  STATUS.md                    # current state
  HANDOFF.md                   # what's next, where to pick up
orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/
  PROGRAM.md                   # original program plan (phases, gates, scope)
  STATUS.yml                   # task tracking + results
  SCOPE.yml                    # write-scope rules (legacy, ctaBot-rooted)
  workspace/
    RESULT.md                  # Phase 0 gate (i) verdict — STOPPED
    METHODOLOGY_REVIEW.md      # mirror of docs/ (program-relative path)
    alpha_*_audit.csv          # IC audit outputs
```

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Pull data (one-time, ~20 min)
Klines for 25 symbols (~700 MB):
```bash
python3 -m scripts.pull_xs_klines
```

For BTC/ETH/SOL with aggTrades (heavier, ~16 GB):
```bash
python3 -m data_collectors.binance_vision_loader \
  --symbol BTCUSDT --dataset aggTrades \
  --start 2025-03-23 --end 2026-04-28
# repeat for ETHUSDT, SOLUSDT
```

### 3. Reproduce the headline results
```bash
# Cross-sectional v4 (25-symbol portfolio)
python3 -m ml.research.alpha_v4_xs

# Concentrated + conviction-filtered variants
python3 -m ml.research.alpha_v4_concentrated

# Alpha-tailored v3 (3-symbol pair trading)
python3 -m ml.research.alpha_v3

# Alpha target / feature audits
python3 -m ml.research.alpha_feature_audit
python3 -m ml.research.alpha_v3_audit
```

## Key findings (TL;DR)

1. **Alpha exists but is small** — 5–10 bps gross per trade across all setups.
2. **Cross-sectional improves robustness, not magnitude** — rank IC stable
   +0.035 across all folds, but raw alpha doesn't multiply.
3. **Look-ahead bugs are subtle and matter** — fixing target normalization
   shift removed +15 bps of in-sample inflation. VPIN window was leaking
   full-dataset volume.
4. **Per-symbol direction reversals require sym_id** — same feature has
   opposite IC sign on different symbols; pooled models without symbol
   indicator average to noise.
5. **Hedged execution doubles cost** — hedging cleanly captures alpha but
   2× cost ≈ 24 bps RT, exceeding the 5–10 bps alpha edge at retail VIP-0.

## Path forward

The structural alpha ceiling at this horizon and feature set is now
characterized. To produce a deployable strategy, three independent levers
exist:

1. **Lower fees** (VIP-3 maker → ~2–4 bps RT): top-5/q80 OOS turns +3 bps net.
2. **Different horizon** (1d / 1w residual reversal): documented 30–50 bps
   in academic literature; would clear retail cost.
3. **Different data** (orderbook L2): typical +5–10 bps IC contribution.

See `docs/HANDOFF.md` for specific next-step plans.

## License

Research code, no warranty. Use at your own risk for trading.
