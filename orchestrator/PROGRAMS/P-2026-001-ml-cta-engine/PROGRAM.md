# P-2026-001: ML-Driven CTA Signal Engine

## Goal

Replace the rule-based Alpha30 entry signal — and only the entry signal — with a learned model that consumes orderbook, trade-tape, and kline features and produces a calibrated trade probability + size. The existing execution stack (sizing infra, executor, circuit breaker, kill switch) is reused unchanged. Alpha30 keeps running side-by-side until ML beats it on ≥90 days of live paper.

## Background

The Alpha30 rule engine encodes a fixed view of breakout/mean-reversion regimes. It works in steady regimes but cannot adapt when correlations or microstructure shift. The hypothesis behind this Program is that microstructure features (orderbook imbalance, trade flow, queue dynamics) carry predictive content not present in 5-minute klines, and that a learned model can compose features in ways the rules cannot.

That hypothesis is **not assumed** — it is tested at the Phase 0 gate. If a linear probe on microstructure features cannot beat a kline-only baseline by a pre-registered margin, the Program stops.

## Scope decisions (defaults; redirect to override)

| Decision | Default | Rationale |
|---|---|---|
| Trade-tape & kline backfill | **Binance Vision** daily CSV dumps (free, ~5y history) | No need to scrape via REST; faster, deterministic, no rate limits. |
| Orderbook history | **Tardis.dev** purchased archive — `book_snapshot_25` (or equivalent) on Binance Futures SOLUSDT for ≥1 year | Eliminates the calendar-time wait for L2 accumulation. One-time cost ~$200–600. |
| Orderbook live (paper/prod) | **REST snapshot collector** — `/fapi/v1/depth?limit=20` every 5s, single collector with retry-on-fail | Decision cadence is ≥30s; 5s book snapshots are ample. Storage near-zero. Drops the WebSocket diff-stream + dual-collector complexity. |
| First symbol | SOL only through Phase 5 | Active workstream branch, smaller dataset = faster iteration |
| Replace vs augment | Augment — ML signal runs side-by-side with Alpha30 in paper, never overrides until ≥90 days live paper proves edge | Reduces risk of regressing existing live PnL |
| Location | `ctaBot/orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/` | In-place, single repo |
| Decision cadence | TBD by Phase 0 cadence experiment (30s vs 1m vs 5m) — do not assume 5m | 5m may average away the alpha that motivates orderbook collection |

## Pipeline structure

Single pipeline, single feature library, **two ablation gates in Phase 0**. The gates do the diagnostic work that the old "Track A / Track B" framing did, without duplicating engineering.

```
Phase 0 →  Gate (i):  kline+tape probe       — is there any alpha?
           Gate (ii): full-feature probe     — does orderbook add ≥1 bps net?
                ↓ both must pass
Phase 1 (features) → 2 (labels) → 3 (models) → 4 (meta+sizing) → 5 (A/B) → 6 (paper) → 7 (prod)
```

**Why two gates, not one combined:**
- Gate (i) failing tells us there is no signal in our feature universe — stop, don't engineer orderbook features further. Halts Tardis spend before we buy more history.
- Gate (ii) failing while (i) passes tells us orderbook adds no incremental lift — ship the kline+tape model and skip the orderbook stack in production (lower complexity, no Tardis ongoing dependency).
- Combining into one gate on the full feature set creates ambiguous failure: we wouldn't know if no-alpha or orderbook-noise killed it.

Calendar-realistic timeline: **first model lands at ~10–12 weeks** (no calendar-time wait, since Tardis dissolved it). Phase 0 takes ~3 weeks for backfill + collector validation + both probes.

## Phases

### Phase 0 — Data infrastructure & go/no-go gates
**Data setup (in parallel from week 1):**
- aggTrade + kline backfill for SOL (**400 days** to support 5-fold walk-forward; see Phase 2) via **Binance Vision daily CSV dumps**; parquet schema; data-quality audit (gap detection, dedup).
- Purchase **Tardis.dev** archive of Binance Futures SOLUSDT `book_snapshot_25` (or `incremental_book_l2` if snapshot-25 is insufficient). **Default: staged purchase** — buy **90 days first** (~$50–150), enough for a 1-fold gate (ii) proof-of-concept, then top up to full **400 days** (~$150–450 incremental) only if gate (ii) passes on the 90d slice. Trade-off: ~1–2 weeks added if (ii) passes; saves the full Tardis spend if (ii) fails outright. The 90d slice is sufficient for gate (ii) because the gate is a *relative* comparison (full-feature vs kline+tape on the same window), not an absolute production-grade evaluation. Override to bulk 400d purchase only if engineering speed dominates cost.
- Stand up the **5-second REST snapshot collector** (`/fapi/v1/depth?limit=20`) for live data going forward, writing to parquet. Single collector with retry-on-fail; verify continuous operation over 7 days before relying on it for paper/prod.
- **Train/serve skew check (two-part):**
  - **(a) Distribution match** — compute identical microstructure features (OBI, spread, microprice, depth-imbalance) on Tardis historical snapshots vs live REST snapshots over a 24h overlap window. Distributions must match within 5% per feature; flag any feature that fails for redefinition or exclusion.
  - **(b) Timestamp alignment** — regress live REST snapshot receive-time against Tardis exchange-timestamp over the same 24h overlap. Require **median offset < 1s** and **99th-pct < 3s**. A systematic offset is fine if stable; **shift live snapshots back by the median offset** before computing labels at sub-5m cadence so they align with how the model was trained. The 5% distribution check alone won't catch a 100–300ms systematic shift that would misalign features with labels at 30s/1m cadence — this regression is the only thing that does.

**Cadence experiment:** for each candidate cadence (30s, 1m, 5m), fit a leak-free linear probe (forward-return target), then evaluate the **net-of-cost Sharpe of the implied long/short strategy** (probe sign + threshold). Pick the cadence with the highest fold-stable net Sharpe — *not* raw R², which is noisy near zero on returns.

**Two gates (both falsifiable, both pre-registered):**

Both gates use the same probe class (ℓ²-logistic), same cost model, same walk-forward folds, same labels. They differ only in feature input.

- **Gate (i) — kline+tape probe vs kline-only baseline.** Asks: is there any alpha at all? If this fails, stop the Program; orderbook is unlikely to rescue it. Halts Tardis spend before further history is purchased.
- **Gate (ii) — full-feature probe (kline + tape + orderbook snapshots) vs the gate-(i) winner.** Asks: does orderbook add incremental lift? Must beat gate (i)'s **mean net-of-cost return per triggered decision by ≥1.0 bps absolute** (same units as the 1.5× cost-floor criterion below — bps of notional per trade) AND show non-inferior fold-level annualized Sharpe (within 0.1 of gate (i) at worst, ≥ on majority of folds). If (i) passes but (ii) fails, ship the kline+tape model in Phase 5+ and drop the orderbook stack from production scope (Tardis archive is retained for reproducibility, but no live snapshot collector dependency in prod).

**Common acceptance — each gate must satisfy ALL of:**
- **Per triggered decision** (not per bar), expected net-of-cost return ≥ **1.5× round-trip cost floor** at the chosen cadence under the **primary cost model** (taker baseline: ≥15 bps; maker baseline: ≥6 bps), AND **net-positive under the stress cost model** (50th-pct funding, 2× slippage). The stress check is here in the gate — not deferred to Phase 3 — so that a probe surviving primary costs but dying under stress fails fast before Phase 1.
- **Fold-level annualized Sharpe ≥ 1.0 net of costs** on **≥4 of 5** walk-forward folds (4/5, not 5/5, to tolerate noise; the failing fold must not be net-negative).
- Minimum **2,000 labeled samples AND ≥150 triggered trades per fold**, achieved at a **pre-registered probability threshold** chosen on the calibration slice. **Re-tuning the threshold to hit the trade count is forbidden** (data-snooping). If a fold cannot reach 150 triggers within an extended test window at the pre-registered threshold, the gate **fails** — do not silently lower the threshold.
- The probe's expected trigger rate (≈ trades / bars) must be stated up-front. At 5m cadence, 150 triggers / 20d test slice ≈ **2.6% trigger rate**; at 1m cadence ≈ **0.5%**; at 30s cadence ≈ **0.25%**. If the chosen cadence + threshold can't produce these rates on the calibration slice, the probe is structurally underpowered — abort, don't lower thresholds.
- Evaluated on the **standalone** primary universe (see Phase 2 — universe (b)), not on the Alpha30-meta universe.

**Outcome matrix:**
- Gate (i) fails → stop Program.
- Gate (i) passes, gate (ii) fails → continue with kline+tape feature set; drop orderbook from Phases 1+; sequence-model option in Phase 4 is removed.
- Both pass → continue with full feature set; orderbook features and sequence-model option remain in scope.

### Phase 1 — Feature library
Single feature library; the gate (ii) outcome from Phase 0 determines whether `microstructure.py` is wired into the production model.

- `features_ml/microstructure.py` — **snapshot-based** features only: OBI, microprice, depth-imbalance at L1/L5/L20, spread, book slope, depth-weighted mid, depth ratios. **Queue-depletion / fast-cancel features are out of scope** — they require sub-second L2 streaming we are not collecting (and Tardis snapshot-25 doesn't reconstruct them either). If Phase 3 plateaus, revisit by purchasing Tardis `incremental_book_l2` and adding queue features in a follow-up Program. Excluded from production model if Phase 0 gate (ii) failed.
- `features_ml/trade_flow.py` — signed volume, TFI, VPIN, Kyle-λ, aggressor ratio, large-trade detection.
- `features_ml/klines.py` — wraps existing `hf_features.py` outputs into the ML feature schema.
- All features computed point-in-time, feature-by-feature leak audit (shift-by-1 sanity), unit tests on synthetic + recorded data.
- **Snapshot-cadence ablation** (only if gate (ii) passed): train identical orderbook features at 1s / 5s / 30s snapshot resolution on Tardis historical and pick the lowest cadence that doesn't cost ≥0.2 fold-Sharpe vs the densest. Default to 5s if no cadence wins clearly.

### Phase 2 — Labels and research harness
- **Labeler:** triple-barrier with TP/SL set at `k·ATR(14)`. Asymmetric barriers allowed. `k` is gridsearched on the **calibration slice** of each fold (see fold structure below) — never on the test slice, never reused across folds.
- **Two primary signal universes (both must be evaluated):**
  - **(a) Alpha30-meta universe** — meta-classifier predicts `P(win | Alpha30 fires)`. Interpretable, low-risk, but capped at "Alpha30 minus its losers".
  - **(b) Standalone primary universe** — independent classifier with its own entry trigger (e.g. probability above a calibrated threshold), free to find trades Alpha30 misses. **This is the universe gated in Phase 0 and graded in Phase 3.**
  - The two heads are trained separately, evaluated separately, and may ship independently. (b) is the answer to "does ML beat rules"; (a) is a fallback if (b) is borderline.
- **Walk-forward CV:** 5 folds, **train 50d / calibration 10d / test 20d** within each fold (total fold span 80d; 5 folds need ≥400d of data — covered by the Phase 0 backfill spec).
- **Embargo:** `max(label_horizon, 1×ATR-time)` calendar days between train and test, applied to both ends of the test slice. Concretely: triple-barrier vertical-barrier defaults to 24h, so embargo ≥ 1 day; longer if `k·ATR` widens the typical hit time. **Purging** removes any train/calibration label whose label window overlaps the test slice.
- **Cost model (pre-registered):**
  - Fees: maker 0.02%, taker 0.05% (Binance VIP-0).
  - **Spread cost:** 0.5 × bid-ask spread per side, taken from Tardis `book_snapshot_25` historical (available from project start) or from live REST snapshots at decision time. AggTrade tick-bounce is only used as a fallback for timestamps with no L2 snapshot available (e.g. collector gaps).
  - **Slippage / impact:** `λ · sqrt(size / top-of-book depth)`, λ calibrated against any available execution log; floor at 1 tick. **Do not** assume "1 tick" flat — at SOL's tick size that's < 1 bps and unrealistic for taker.
  - **Maker fill probability < 1:** model fill probability as `f(queue_position, time_in_force, adverse_volatility)`; unfilled maker orders pay opportunity cost of the missed move. If we can't model this in Phase 2, default to **taker-only** in the cost stack to be conservative.
  - **Funding:** realized historical perp funding rate per timestamp; **shorts pay/receive separately from longs** (asymmetric).
  - **Sensitivity:** all acceptance metrics also reported under stress (50th-percentile worst rolling-30d funding, 2× modeled slippage) — required to remain net-positive though the bar may be lower.

### Phase 3 — Baseline models
- **Baseline-of-baseline:** ℓ²-logistic regression on a curated 15–25 feature set. If this fails, stop.
- **LightGBM:** with monotonic constraints where economically meaningful, max-depth ≤ 6, leaves ≤ 64, early stopping on validation log-loss.
- **Hyperopt budget:** ≤ 50 trials; report **deflated Sharpe ratio (DSR)** with Bonferroni correction over the trial count, not raw Sharpe.
- **Calibration:** Brier score + reliability diagram on test folds; isotonic recalibration if Brier-vs-uncalibrated improves on validation.
- **Feature analysis:** permutation importance + SHAP on the held-out test fold of fold 5.

**Acceptance (pre-registered, all required) — evaluated on the standalone (b) universe:**
1. **DSR ≥ 0.95** OR (**DSR ≥ 0.7 AND fold-level annualized net Sharpe ≥ 1.0**) — per fold, not pooled. Trial count for Bonferroni adjustment is the full hyperopt budget.
2. **≥ 200 trades per fold** at the **pre-registered threshold from the calibration slice** (consistent with the Phase 0 ≥150 trigger floor — Phase 3 raises the bar to 200 because the production model is more selective and lower-variance than the linear probe). If a fold falls short at the pre-registered threshold, **extend that fold's test window**; **never re-tune the threshold to hit the count** and **never drop a fold** (both are data-snooping).
3. **Net PF ≥ 1.4** at the primary cost model AND **net PF ≥ 1.15** under the stress cost model (50th-pct funding, 2× slippage), per fold.
4. **MaxDD < 1.5×** Alpha30's MaxDD on the same fold.
5. **Beats Alpha30 on ≥3/5 folds** on net Sharpe **with paired block-bootstrap p < 0.10** on per-trade PnL difference. **Block size is standardized across the Program as `max(avg trade duration in bars, label horizon in bars) × 2`** — the `× 2` covers serial correlation from overlapping triple-barrier label windows that extends past trade duration. Phase 6 uses the same formula.

If any of (1)–(4) fail, the model fails. (5) is the comparative bar.

**(a) Alpha30-meta head** is graded with the same criteria but evaluated only on Alpha30's trade universe; failure of (a) does not kill the Program if (b) passes.

### Phase 4 — Meta-labeling, sizing, and (conditional) sequence models
- Secondary classifier (the (a) Alpha30-meta head) is finalized in this phase.
- **Sequence-model option (only if Phase 0 gate (ii) passed):** once tabular orderbook features pass Phase 3, allow a 1D-CNN or shallow GRU on the last N L2 snapshots (`N=32–64`) as an additional feature extractor or stand-alone head. Orderbook state is fundamentally Markovian — tabular snapshots discard transition signal. **Transformer / large sequence models remain out of scope.** Skipped entirely if gate (ii) failed.
- **Position sizing:**
  - `KellyFraction = (p̂_cal · b - q̂_cal) / b` where `p̂_cal` is the **calibrated** probability (isotonic from Phase 3) and `b = TP/SL ratio`.
  - **Half-Kelly** (multiply by 0.5) is the default; `p̂_cal` is **clipped to [0.5, 0.7]** before being plugged in to bound exposure under calibration error.
  - Position size = `max(0, min(0.5 · KellyFraction(clip(p̂_cal)), risk_cap))`, where `risk_cap = current Alpha30 per-trade risk fraction`. The outer `max(0, ·)` guards against negative-Kelly cases (e.g. clipped `p̂=0.5` with `b<1`) producing an inverted position.
- **Acceptance:** OOS Sharpe ≥ Phase 3 baseline at lower MaxDD on ≥3/5 folds, with the sizing rule applied.

### Phase 5 — Backtest A/B
- Run ML signal through `hf_backtest.py` with **identical execution rules and cost model** as the Alpha30 backtest (close-only, legacy trailing, no partial TP, no scale-in to match live config).
- Generate side-by-side report: PF, Sharpe, DSR, MaxDD, trade count, win rate, regime breakdown (trend/range/shock).
- **Acceptance:** ML matches/exceeds Alpha30 on **net Sharpe and MaxDD on both 90d and 180d windows**, and is not worse on any single regime by more than 20%.

### Phase 6 — Paper trading + ML safety layer

**Note on conditional scope:** if Phase 0 gate (ii) failed, the production model uses kline+tape features only — the live REST snapshot collector is not required for inference and may be decommissioned (or kept running for research-only logging). The drift monitor's "top-10 features" list is drawn from whichever feature set the production model actually uses.

- **In-process inference** (no model server). Model artifact loaded at bot startup; signal computed inline next to feature computation. Adds < 5 ms/bar at 5m cadence.
- **realtime_paper_bot.py** gains an ML mode toggle that runs ML signal alongside Alpha30, logs both, executes both as separate dual-paper books. **This is dual-paper, not a true A/B** — the two signals share data feed and drift, so the test is comparative-not-causal.
- **Statistical bar (replaces "tracks within ±15%"):** **paired block-bootstrap on per-trade PnL difference**, block size = `max(avg trade duration, label horizon) × 2` bars (same formula as Phase 3 acceptance #5), null `H₀: ML ≤ Alpha30`, reject at p < 0.05. **Power calc must show ≥80% power to detect a 0.5 Sharpe-unit difference** at the expected per-symbol trade rate; if power is insufficient at 90 days, extend paper or include BTC/ETH paper-runs to gain sample size.
- **Drift monitor (revised hierarchy):**
  - **Primary trigger:** PSI on the **prediction distribution** (model output), daily. PSI > 0.25 → alert; > 0.4 → fail-closed disable ML, revert to Alpha30.
  - **Diagnostic (no auto-action):** per-feature PSI on top-10 features — logged, surfaced in Grafana, but does not trigger automated disable (microstructure features are inherently regime-driven and would alert constantly).
- **ML kill-switch:** consecutive 5-loss streak OR daily drawdown > 2× Alpha30's worst day in last 90d → auto-disable ML for 24h. Manual re-enable required.
- **Model versioning:** every artifact tagged `model_v{semver}_{train_hash}`; bot logs the active version each bar; rollback = swap the symlink and restart.
- **Acceptance:** 90 calendar days paper, paired bootstrap p < 0.05 vs Alpha30, prediction-PSI never sustained > 0.25 for > 3 consecutive days, **at most 1 kill-switch activation in the 90-day window AND a written post-mortem confirming no model defect** (this avoids a circular block where one tripped switch in a chop regime resets a 30-day clock indefinitely).

### Phase 7 — Production rollout
- Standard MANUAL_START_GUIDE phasing: testnet → 0.01x → 0.1x → 1.0x.
- **Retrain cadence:** weekly on rolling 180d window. **Promotion gate (all required to avoid model-churn ratchet):**
  - Beats current production model on the most recent **30d held-out** window (paired bootstrap p < 0.10 on per-trade PnL).
  - **Non-inferior** to current production on the **full 180d window** (Sharpe within 0.2 of current).
  - **Minimum production lifetime 14 days** before any model can be replaced (prevents weekly thrash).
- Full Grafana panels: prediction-distribution PSI (primary), per-feature PSI (diagnostic), prediction histogram, per-trade PnL attribution, kill-switch state, active model version.

## Files (in scope)

**New (always required):**
- `data_collectors/binance_vision_loader.py` — downloads + parses Binance Vision daily CSV dumps for aggTrades + klines
- `data_collectors/tardis_loader.py` — downloads + parses Tardis.dev archives for historical L2 snapshots (used for gate (ii) and for spread cost in the cost model regardless of outcome)
- `features_ml/{trade_flow,klines,labels}.py`
- `ml/research/` — notebooks for EDA, gates, hyperopt, calibration
- `ml/models/{logreg_baseline,lgbm}.py` — training entry points
- `ml/inference.py` — in-process artifact loader + predict
- `ml/drift.py` — PSI computation, kill-switch logic
- `tests/ml/`

**New (conditional on Phase 0 gate (ii) passing):**
- `data_collectors/snapshot_collector.py` — REST `/fapi/v1/depth?limit=20` every 5s → parquet (single collector, retry-on-fail). Required for live inference only if gate (ii) passes; otherwise stop after Phase 0 train/serve skew validation.
- `features_ml/microstructure.py` — orderbook snapshot features. Wired into the production model only if gate (ii) passes.

**Modified at Phase 5/6 (will require SCOPE update):**
- `signals.py` — add `MLSignalEngine` alongside `Alpha30SignalEngine`
- `realtime_paper_bot.py` — ML mode toggle
- `realtime_live_bot_hybrid.py` — ML mode toggle (Phase 7 only)
- `hf_backtest.py` — accept signal-engine interface for A/B

## Storage plan

- **Orderbook historical (Tardis):** SOL Binance Futures `book_snapshot_25` for 400 days ≈ ~5–15 GB compressed (Tardis daily archives, line-delimited JSON or CSV). One-time download.
- **Orderbook live (REST 5s snapshots):** 17,280 snapshots/day × ~3 KB/row parquet ≈ **~50 MB/day** ≈ ~1.5 GB/month. Negligible. Local disk through Phase 7. **Required only if Phase 0 gate (ii) passes**; otherwise the live collector may be stopped after the Phase 0 train/serve skew check completes.
- **AggTrades + klines:** Binance Vision daily CSV dumps for SOL, 400 days, ~10 GB parquet after compression. One-time download.
- **Features:** materialized to parquet per cadence; rebuild on schema change is cheap.
- **Collector resilience:** single REST snapshot collector is sufficient — REST is stateless and trivially restartable. Cron-style supervisor (systemd unit) with auto-restart. No dual-collector / dual-region complexity needed at this cadence.

## Risks and explicit non-goals

**Risks:**
1. Phase 0 gate fails — high prior probability for retail-frequency CTA. Mitigation: gate is the early-exit, not a checkpoint.
2. **Tardis historical snapshots vs our live REST snapshots have different sampling artifacts** (Tardis is exchange-side, ours is REST polling with network jitter). Mitigation: explicit train/serve skew check in Phase 0; any feature failing the 5% distribution match is excluded or redefined.
3. **Tardis license** prohibits redistribution; OK for internal research/trading. Mitigation: store in private buckets, do not commit raw archives to git.
4. REST snapshot collector has gaps from network or rate-limit issues. Mitigation: systemd auto-restart; gap detection at parquet-merge time; if gaps > 1% of expected snapshots in any 24h, alert and investigate.
5. Microstructure features have multi-second half-life that disappears at 5m. Mitigation: cadence experiment in Phase 0 explicitly tests this; snapshot-cadence ablation in Phase 1 confirms 5s is dense enough.
6. Overfitting hyperopt on 5 folds × small trade counts. Mitigation: DSR with Bonferroni, hard 50-trial cap.
7. Funding regime shifts mid-backtest invalidating cost model. Mitigation: use realized historical funding, not flat assumption.
8. **Queue dynamics features (out of scope) turn out to matter.** If Phase 3 plateaus on snapshot features, expand scope by purchasing Tardis `incremental_book_l2` and adding queue features — explicitly a follow-up Program, not in-scope here.
9. **REST `/fapi/v1/depth` weight ceiling.** Single-symbol 5s polling on SOL is well within Binance Futures' 2400/min IP-weight cap (depth limit=20 = weight 2 → ~24 weight/min). **Any future expansion to additional Binance symbols requires re-budgeting** against the cap; document the budget in `snapshot_collector.py` and assert at startup. Hyperliquid expansion (Open Q #2) uses a separate venue with its own limits, so doesn't compete.

**Non-goals:**
- Multi-symbol joint model (defer until single-symbol works on SOL).
- Reinforcement learning, transformer, or large sequence models (transformers explicitly deferred). **Note:** shallow 1D-CNN/GRU on L2 snapshots IS in scope at Phase 4 conditional on gate (ii) passing — orderbook state is Markovian and tabular snapshots discard transition signal that motivated collecting L2.
- **Sub-second / queue-dynamics features** — out of scope by collection design. Snapshots at 5s are intentionally too coarse for queue depletion or fast-cancel detection. Adding these is a follow-up Program (requires Tardis `incremental_book_l2` purchase and a streaming live collector).
- Replacing Alpha30 (this Program only adds ML as an augmentation through Phase 7; replacement is a separate future Program).
- High-frequency / sub-second execution (decision cadence ≥30s by design).

## Open questions

1. **Tardis archive scope** — purchase `book_snapshot_25` only (cheaper, sufficient for snapshot features), or also `incremental_book_l2` to keep queue features as an option without a second Tardis spend? Tentative: snapshot-25 only at start; only buy incremental if Phase 3 plateaus.
2. Should we also collect Hyperliquid orderbook (5s REST snapshots) in parallel to enable cross-venue features later? Tentative: log only, no features yet.
3. Funding rate in the cost model — realized perp funding for primary, 50th-pct rolling 30d funding for stress? Tentative: yes (already encoded in Phase 2 cost model and Phase 0 stress gate).

## Acceptance criteria (Program-level)

- [ ] Phase 0 gate (i) passes (kline+tape probe beats kline-only baseline). Else stop.
- [ ] Phase 0 gate (ii) evaluated: orderbook adds ≥1 bps incremental, OR explicitly recorded as failed → orderbook stack dropped from production scope.
- [ ] Phase 3 acceptance criteria all met on SOL with the feature set determined by Phase 0.
- [ ] Phase 5 A/B shows ML ≥ Alpha30 on Sharpe and MaxDD across 90d and 180d.
- [ ] Phase 6 paper passes 90d with drift and kill-switch criteria met.
- [ ] Phase 7 ML in production at 1.0x with weekly retrain pipeline running.
