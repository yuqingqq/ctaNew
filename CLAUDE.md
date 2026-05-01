# CLAUDE.md

Project instructions for Claude Code working in this repo. See `AGENTS.md`
for the full agent guide; this is the Claude Code-specific summary.

## Project type

**Research code, not production.** ML pipeline for predicting alpha-residual
on Binance USDM perpetual futures using free public data. No live trading,
no exchange integrations, no execution server.

## Build & test

```bash
pip install -r requirements.txt
# No formal tests in this repo; reproduction is via the probe scripts in
# ml/research/. See README.md "Quick start".
```

## Running probes

```bash
# Cross-sectional v4 (25-symbol portfolio) — ~15 min
python3 -m ml.research.alpha_v4_xs

# v3 alpha-tailored (3-symbol pair) — needs aggTrades
python3 -m ml.research.alpha_v3

# Audits
python3 -m ml.research.alpha_feature_audit
python3 -m ml.research.alpha_v3_audit
```

Caches to `data/ml/cache/`; subsequent runs reuse them.

## Architecture

```
features_ml/  Feature pipeline (klines, regime, trade_flow, cross_asset,
              alpha_features, cross_sectional, labels)
ml/           CV, cost model, research probes (alpha_v2/v3/v4 + audits)
data_collectors/  Binance Vision daily archive loader
hf_features.py    Legacy 160+ kline indicators (used by features_ml/klines.py)
docs/         METHODOLOGY_REVIEW (audit trail), STATUS, HANDOFF
orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/  Original program plan + results
```

## Key conventions

- **Point-in-time features only.** Rolling stats use trailing windows; beta
  and z-score features `.shift(1)` to avoid using current bar.
- **Pooled training across symbols** with `sym_id` indicator. The model
  trains on stacked per-symbol panels.
- **Walk-forward CV** with embargo (1 day) and label purging via `exit_time`.
  Use the `_expanding_train` helper pattern in each `alpha_v*.py`.
- **LGBM hyperparameters pinned** across v1→v4 for fair comparison. Don't
  retune per-version without flagging it.
- **Cost model is retail VIP-0** (~12 bps RT naked, ~24 bps hedged). Most
  conclusions stated relative to this.

## Look-ahead pitfalls (real bugs found during research)

1. **Target normalization** — `rolling.shift(1)` is wrong for h-bar forward
   returns; must be `.shift(horizon)`. Fixed in all `alpha_v*.py`.
2. **VPIN buckets** — was sized using full-dataset volume; now trailing 7d
   per bar. Fixed in `features_ml/trade_flow.py::_vpin`.

When adding a new feature, sanity-check IC against forward return shifted
by +1 bar. Anything >+0.10 IC is suspicious and probably has hidden look-ahead.

## What's in/out of scope

**In scope** (free to edit):
- `features_ml/`, `ml/`, `data_collectors/`, `scripts/`, `docs/`
- `hf_features.py` (legacy but in-repo)
- `orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/` (research record)
- `live/` (paper-trading harness, added 2026-05-01) — multi-symbol
  orchestrator, basis-risk diagnostics, Binance-train / Hyperliquid-execute
  pipeline. Not production-grade; this is the forward-test layer that
  validates v6_clean predictions transport from backtest data to real-time.

**Out of scope** (don't add):
- Live trading code, exchange adapters, execution servers
- Production deployment infra (Docker, CI for trading, etc.)
- Large data files in git — use `.gitignore` and external storage

## Where to read first

1. `README.md` — overview + quick start
2. `docs/METHODOLOGY_REVIEW.md` — full audit trail (most important)
3. `docs/STATUS.md` — current state, known issues
4. `docs/HANDOFF.md` — three ranked next-step plans

## Tone

Brief, technical, no hype. Match the existing docs — short headers, tables
where relevant, numbers with units (bps, IC, days). The methodology review
sets the tone for new docs.
