# Agent Guide for ctaNew

Fast-path orientation for AI assistants and human collaborators working in
this repo. Read this once, then read `docs/HANDOFF.md` for the current
research state and where to pick up.

## What this repo is

**Research code, not production.** Alpha-residual ML prediction on Binance
USDM perpetual futures, using free public data (Binance Vision daily archives).
The strategy class is fully characterized — alpha exists at ~5–10 bps gross
per trade, below retail VIP-0 cost. Three structural pivots are documented as
next steps in `docs/HANDOFF.md`.

There is **no live trading code, no exchange adapters, no execution server**
in this repo. Don't add them — keep this clean as a research artifact.

## Where to start reading

1. `README.md` — high-level overview + reproduction commands
2. `docs/METHODOLOGY_REVIEW.md` — full audit trail, numbered issues, fix
   attempts, all results. The most important doc.
3. `docs/STATUS.md` — current state, known issues, compute footprint
4. `docs/HANDOFF.md` — three ranked next-step plans
5. `orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/PROGRAM.md` — original
   program plan with phase gates (gate (i) failed → program stopped)

## Core code paths

```
features_ml/cross_sectional.py   # 25-symbol basket pipeline (v4)
features_ml/alpha_features.py    # alpha-tailored features (v3)
features_ml/cross_asset.py       # pairwise cross-asset features (v2)
features_ml/trade_flow.py        # aggTrade-derived features (TFI/VPIN/Kyle's λ)
features_ml/klines.py            # kline feature wrapper
features_ml/regime_features.py   # regime z-scores, distance-from-high
ml/cv.py                         # walk-forward folds + label purging
ml/cost_model.py                 # fee/slip/Roll-spread cost
ml/research/alpha_v4_xs.py       # cross-sectional v4 (final)
ml/research/alpha_v3.py          # alpha-tailored v3 single-pair
ml/research/alpha_*_audit.py     # IC audits per feature × target × symbol
data_collectors/binance_vision_loader.py  # daily archive downloader
hf_features.py                   # 160+ kline indicators (legacy, monolithic)
```

## How to run

```bash
pip install -r requirements.txt

# 1. Pull data (one-time, ~20 min, ~700 MB)
python3 -m scripts.pull_xs_klines

# 2. Headline cross-sectional probe (~15 min)
python3 -m ml.research.alpha_v4_xs

# 3. Concentrated + conviction-filtered variants (~30 min)
python3 -m ml.research.alpha_v4_concentrated

# 4. 3-symbol pair-trading (needs aggTrades — heavy)
python3 -m data_collectors.binance_vision_loader \
  --symbol BTCUSDT --dataset aggTrades --start 2025-03-23 --end 2026-04-28
# repeat for ETHUSDT, SOLUSDT, then:
python3 -m ml.research.alpha_v3
```

Caches build to `data/ml/cache/`. Subsequent runs reuse them — fast.

## Conventions

- All features are **point-in-time**: rolling stats use trailing windows;
  beta estimates and z-scores are `.shift(1)` to avoid using current bar.
- Walk-forward CV uses fold layouts in `ml/cv.py` with embargo (1 day default)
  and label purging via `exit_time` column on label DataFrames.
- Pooled training: stack per-symbol panels, train one model. `sym_id`
  indicator column lets the model partition by symbol when feature signs
  reverse across symbols.
- LGBM hyperparameters are pinned in `_train()` of each probe — same params
  across v1/v2/v3/v4 for fair comparison. Don't tune per-version.
- Cost model in `ml/cost_model.py` — fee_taker=0.0005, slip=1 bps flat,
  spread from `effective_spread_roll`. Don't change these without explicit
  reason; results are stated relative to retail VIP-0.

## Look-ahead bugs to watch for

This codebase had two real look-ahead bugs found during the research:

1. **Sharpe target normalization shift** — `rolling.shift(1)` should be
   `.shift(horizon)` because forward returns at horizon h require prices h
   bars ahead. Fixed in `_make_alpha_label` of all `alpha_v*.py`. If you
   add a new label, propagate this pattern.

2. **VPIN bucket sizing** — used `total_vol.iloc[-1]` (full dataset) to
   size buckets; now uses trailing 7d window per bar. Fixed in
   `features_ml/trade_flow.py::_vpin`.

When adding a new feature, sanity-check by computing IC vs `fwd_ret` *one
bar shifted forward*. Suspicious +0.10 IC vs forward return often hides a
lookback that uses the current bar's close in the feature.

## Things NOT to do

- **Don't add live-trading code, exchange adapters, execution servers.**
  This repo is research only. If you need production hardening, start a
  separate repo.
- **Don't tune hyperparameters per probe** — keep the pinned LGBM params
  for fair v1↔v4 comparison.
- **Don't change the cost model without flagging it loudly** — most
  conclusions are framed against VIP-0 retail cost (~12 bps RT naked).
- **Don't delete `cache/` files casually** — `xs_feats_*.parquet` take
  ~7 minutes to rebuild from klines.
- **Don't push large data files (parquets, CSVs > 1 MB) into git.** The
  `.gitignore` excludes `data/`. If you need to publish a dataset, use
  HuggingFace or a separate data release.

## Per-symbol idiosyncrasies

- BTC alpha (vs ETH) has highest OOS IC (+0.08 in v3) but worst trade P&L
  unhedged because BTC's market noise dominates.
- ETH alpha (vs BTC) is the only consistently profitable single-pair OOS,
  but only naked at q=0.99 — and partly luck-of-regime, not robust.
- SOL alpha is broken at 4h horizon. Don't waste time trying to make SOL
  work alone; it dragged the cross-sectional results too.

## Open questions worth chasing

See `docs/HANDOFF.md` for full plans. Quick list:
- Does longer horizon (1d / 1w) deliver 30+ bps alpha?
- Does adding funding-rate features help? (free public data, untried)
- Does maker-tilt simulation produce realistic fill rates? (needs L2 data)
