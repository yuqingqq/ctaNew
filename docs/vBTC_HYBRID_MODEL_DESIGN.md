# vBTC hybrid model architecture design

Last updated: 2026-05-13

## Motivation

The 111-panel diagnostic showed that the universal LGBM model trained on 111 pooled symbols **degrades predictions on the same original 51 symbols** (36/51 of originals lose IC, mean Δ −0.0080). The cause is training-set contamination: 60 new short-history meme symbols add tree splits that don't generalize to mature alts.

Two architectural fixes are viable:

| approach | mechanism | tradeoff |
|---|---|---|
| **Per-symbol** | 51-111 separate models | clean isolation; less data per fit; lose cross-symbol regime signal |
| **Per-class** | cluster symbols, model per cluster | middle ground; cluster choice introduces a hyperparameter |
| **Hybrid (universal + context)** | one universal + per-symbol/class residual | keeps cross-symbol regime AND per-symbol context |

AVAXUSDT diagnostic (2026-05-13) showed per-symbol model produces +0.025 IC improvement over universal on the same panel. This validates the per-symbol direction has signal.

The hybrid architecture combines the strengths: universal model captures cross-symbol patterns, residual model adds per-symbol or per-cluster context that the universal averaged away.

## Architecture

### Two-stage training

```
                              ┌─────────────────────┐
                              │  Stage 1            │
   Pooled training data ────► │  Universal Model U  │
   (all N symbols)            │  (LGBM, WINNER_BTC) │
                              └────────┬────────────┘
                                       │
                              U(s, t) ─┴── predictions per row
                                       │
                                       ▼
                              ┌─────────────────────┐
                              │  residual = y − U   │  per row
                              └────────┬────────────┘
                                       │
                                       ▼
                            ┌──────────────────────────┐
                            │   Stage 2                │
                            │   K residual models      │
                            │   (per-symbol or         │
                            │    per-cluster)          │
                            │   R_k(features)          │
                            └──────────┬───────────────┘
                                       │
                                       ▼
                              Final pred(s, t) =
                                U(s, t) + R_k(s, t)
                              where k = symbol s OR
                                   k = cluster of s
```

### Stage 1: Universal model

- **Features**: universe-invariant only (WINNER_BTC: residual-to-BTC features, BTC regime features, single-name flow, stable context)
- **Target**: target_β = β-residual / σ_idio (z-scored)
- **Training**: pooled across all symbols, expanding-window CV, 10 folds × 5 seeds
- **Output**: U(s, t) — universal prediction per (symbol, time)

The universal model captures cross-symbol regime signal: "when BTC vol is high AND funding is positive, all alts under-perform". These patterns generalize.

### Stage 2: Residual model

Two variants — choose based on data quantity and stability:

**Variant 2a — Per-symbol residual (R_s)**

For each symbol s:
1. Compute residual = target_β(s, t) − U(s, t) on training data
2. Train LGBM model R_s on (features, residual) using only s's rows
3. Save R_s as one of N model files

At inference: pred(s, t) = U(s, t) + R_s(features(s, t))

- **Pros**: cleanest isolation, captures all per-symbol patterns universal averaged away
- **Cons**: ~85k rows per model (LGBM still works but with more variance); N models to maintain; new symbols need new R_s (initialize to zero or fast-fit from new data)

**Variant 2b — Per-cluster residual (R_c)**

1. Cluster symbols by behavior (Phase G's K=6 clustering on 4h return correlation OR re-derive from stable per-symbol attributes)
2. For each cluster c:
   - Compute residual on training data for symbols in c
   - Train R_c on (features, residual) using c's pooled rows
3. Save 5-7 models

At inference: pred(s, t) = U(s, t) + R_c(features(s, t)) where c = cluster_of(s)

- **Pros**: more data per residual model (~850k-1.3M rows for K=6); fewer models to maintain; new symbols just need cluster assignment
- **Cons**: cluster definition adds a hyperparameter; cluster boundaries may not align with true behavioral splits

### Decision criteria for 2a vs 2b

- If per-symbol AVAX diagnostic shows clear +0.025 IC win (it did) AND per-symbol scaling validates on 10+ symbols → **2a** is the right choice
- If per-symbol overfits on smaller-data symbols (some symbols only have 50k rows) → **2b** is safer
- If per-cluster works comparably to per-symbol → prefer **2b** for operational simplicity

## Implementation steps

### Phase 1: Build universal model on universe-invariant features (already done)

Use `WINNER_BTC` (25 features, no `_vs_bk`, no `xs_rank`, no `sym_id`). Train on β-residual target. Save model + predictions.

Output: `outputs/vBTC_audit_panel_btc_only/all_predictions.parquet` — universal predictions per (symbol, time)

### Phase 2: Train per-symbol residual models

```python
universal_preds = load("outputs/vBTC_audit_panel_btc_only/all_predictions.parquet")
panel = load("outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet")

for symbol in panel.symbol.unique():
    rows = panel[panel.symbol == symbol]
    rows["residual"] = rows["target_beta_btc"] - universal_preds.set_index(["symbol","open_time"]).loc[(symbol, rows.open_time)]["pred"]
    # Train LGBM on WINNER_BTC features → residual
    R_s = train_lgbm(rows, features=WINNER_BTC, target="residual",
                     folds=_multi_oos_splits(panel))
    save(R_s, f"models/per_symbol/R_{symbol}.lgbm")

# At inference:
pred(s, t) = universal_preds[(s, t)] + R_s.predict(features(s, t))
```

### Phase 3: Build inference pipeline

```python
def predict_hybrid(symbol, time, features_row):
    u_pred = universal_model.predict(features_row)
    if symbol in PER_SYMBOL_MODELS:
        r_pred = PER_SYMBOL_MODELS[symbol].predict(features_row)
    elif symbol in cluster_assignments:
        cluster = cluster_assignments[symbol]
        r_pred = PER_CLUSTER_MODELS[cluster].predict(features_row)
    else:
        # New symbol with no model trained yet
        r_pred = 0.0
    return u_pred + r_pred
```

### Phase 4: Run V3.1 β-hedged with hybrid predictions

Use the same V3.1 sleeve + β-hedged execution machinery. Just plug in hybrid predictions instead of universal predictions. Compare Sharpe to baseline.

## Operational rules

### Adding a new symbol to the universe

1. **Run universal model on new symbol's rows** — predictions are immediate, no retrain needed (universal model is universe-invariant by construction)
2. **Trade with U-only predictions** until ≥ 30 days of new symbol's data accrues
3. **Train R_s on new symbol's data** — small dedicated LGBM, ~5-10k rows minimum
4. **Plug R_s into inference pipeline** — pred = U + R_s

This is genuinely additive: existing models untouched, only new model added.

### Periodic retraining

- **Universal model**: retrain quarterly with all symbols' data. Universe-invariant target/features means no per-symbol contamination.
- **Per-symbol residual models**: retrain quarterly OR when symbol's recent performance drifts. Each symbol's model is independent so retrain order doesn't matter.

### When a symbol delists

1. Remove R_s from inference pipeline
2. Universal model can keep its training data (delisted symbol's history still informs the universal patterns)
3. No global retrain needed

## Risks and limits

### What this architecture doesn't solve

1. **Universe overfit at the trading layer** — the IC ranker's noise problem (rank-15/16 cutoff in noise) is independent of model architecture. Need separate selector improvements.
2. **Feature engineering quality** — if WINNER_BTC features are sub-optimal, neither universal nor residual model can fix it.
3. **Strategy structure** — K=3 long-short, 24h hold, V3.1 sleeve overlay are unchanged. The hybrid model only improves predictions, not how we trade them.

### Failure modes to watch

1. **R_s overfit on noisy symbols** — if a symbol has erratic history, R_s may memorize noise. Add LGBM regularization (higher min_data_in_leaf, lower learning_rate).
2. **U + R_s scale mismatch** — residual model's output should be small compared to U. If R_s dominates, U isn't doing its job.
3. **Per-symbol predictions become uncalibrated cross-sectionally** — at decision time, K=3 picks compare preds across symbols. If R_s shifts each symbol's pred differently, ranks may be distorted. Mitigation: z-score per-symbol preds before ranking, OR confirm target_β being z-scored already keeps things calibrated.

## Test plan

1. **Validate per-symbol on 51-panel** — train per-symbol residual models for all 51 symbols. Compare hybrid (U + R_s) Sharpe to universal-only Sharpe.
2. **Compare per-symbol vs per-cluster** — same data, different residual granularity. Choose simpler one if comparable.
3. **Test universe expansion** — retrain universal on 111 symbols (universe-invariant features). Add 60 new R_s models. Compare hybrid 111 Sharpe to hybrid 51 Sharpe; should be comparable if architecture works.
4. **Production deployment** — if (1-3) pass, replace WINNER_21+sym_id model with hybrid model + WINNER_BTC features. Backtest, paper-trade, then deploy.

## Open questions

1. **What's the right R_s LGBM size?** Smaller (fewer trees, more regularization) is safer for ~85k rows but may underfit. Need to tune.
2. **Cluster definition for 2b** — use Phase G's K=6 (Ward on 4h-return correlation) or derive new clusters from stable per-symbol features? Memory's Phase G showed K=6 has reasonable separation but the dominant cluster contains 47/111 names — uneven.
3. **Calibration**: should we calibrate hybrid predictions on a held-out fold before deploying? Or trust the LGBM RMSE training?
4. **Cross-validation structure**: with per-symbol models, can each model use its OWN fold structure (some symbols have longer history → more folds available)? Or stick to global folds?

## Files (when implementation begins)

Anticipated structure:

```
models/
  universal_btc_residual.lgbm           # Stage 1 universal model
  per_symbol/
    R_AAVEUSDT.lgbm
    R_ADAUSDT.lgbm
    ...                                 # one per symbol
  per_cluster/                          # if 2b chosen
    R_cluster_0.lgbm
    ...
  cluster_assignments.json              # symbol → cluster mapping

scripts/
  train_universal_btc.py                # Stage 1 trainer
  train_residual_models.py              # Stage 2 trainer (per-symbol or per-cluster)
  run_hybrid_inference.py               # combined inference

outputs/vBTC_hybrid/
  predictions.parquet                   # hybrid predictions per (sym, time)
  v31_sharpe.csv                        # comparison vs baseline
```
