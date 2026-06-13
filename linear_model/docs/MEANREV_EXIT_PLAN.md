# Mean-Reversion-Exit / Ranked-Pool Strategy — Test Plan

Status: planned 2026-05-15. New strategy class (event-driven, NOT the sleeve replay).
Motivation: blue-chips lack explosive meme tails (clean-108's edge, dead on HL);
hypothesis = their β-residual has a tradeable *mean-reverting* component that a
fixed-24h-hold strategy discards. Hold until convergence, recycle capital via a
ranked pool.

## Locked design (user, 2026-05-15)

| fork | choice |
|---|---|
| Exit trigger | **Hybrid** — exit on first of: signal decay, convergence target, max-hold time, adverse stop |
| Universe | **HL-executable & daily vol ≥ $2M = 44 symbols** (on_hl from live HL API) |
| Entry/pool | **Market-neutral L/S**, rank by `pred_B`, long top-N / short bottom-N, refill freed slots each 4h, dynamic BTC hedge on residual net beta |

## Reused (validated machinery — do not reinvent)

- Target `α_β = ret − β_pit·btc_ret` (PnL on α_β is **already BTC-hedged at PIT β**;
  the explicit BTC leg matters only for execution sizing/cost, not backtest PnL —
  modelled as optional turnover-cost realism, v2).
- Full-PIT 110-panel features, V2 Ridge, 9-fold walk-forward, frozen-fold-0 σ_idio.
- Causal aggregator + funding (proven immaterial, included), analytic cost
  (`COST_PER_UNIT_ABS_DELTA` = 2.25 bps/unit |Δw|), block-bootstrap CI, P1/P2.

## Build order

1. **Step 62 — predictions producer.** Step 56 machinery with `HL_FILTER=vol2m`
   (on_hl & hl_day_vol_usd ≥ 2e6 → 44 syms). Full retrain V2 Ridge, save
   `predictions.parquet` (symbol, open_time, pred_z, trail_ic, pred_B, alpha_beta,
   fold, exit_time, return_pct, beta_pit).
2. **Step 63 — event-driven backtest.** 4h decision cadence. State = open
   long/short books with per-position {entry_t, entry_pred, cum_realized_α}.
   Per cycle t:
   - mark: accrue `alpha_beta[sym,t]` (forward 4h residual, causal) to each open
     position's cumulative realized α.
   - **hybrid exit** (first-to-fire) per open position:
     - decay: `|pred_B[sym,t]| < decay_frac·|pred_B_entry|` OR sign-flip vs entry side
     - target: `cum_realized_α ≥ tgt_bps`
     - time: `held ≥ max_hold`
     - stop: `cum_realized_α ≤ −stop_bps`
   - close exited (charge cost); refill empty slots from ranked `pred_B`
     (PIT-eligible, not already open); charge entry cost.
   - weights equal per slot; net residual beta `Σ wᵢ·β_pit,ᵢ` reported (implicit
     BTC hedge in α_β; explicit BTC turnover cost optional v2).
   - record per-cycle gross/funding/cost/net, hold-duration, open-count.
3. **Step 64 — validation.**

## Parameter rigor — THE critical point (K2/K3/V3.3 lesson)

Hybrid exit has tunable params {decay_frac, tgt_bps, max_hold, stop_bps, N}.
The entire prior vBTC/linear history shows **untuned discrete architecture
generalizes; tuned continuous params fail honest OOS**. Therefore:

- Pre-register a SMALL discrete grid (≤ ~16 combos), e.g.
  decay_frac ∈ {0.3, 0.5}, tgt_bps ∈ {50, 100}, max_hold ∈ {24h, 48h},
  stop_bps ∈ {80}, N ∈ {3, 5}.
- Report in-sample best **only as a ceiling, never the headline.**
- **Headline = nested-OOS**: for fold k, pick the grid point best on folds < k,
  apply to fold k; chain k=1..9. Honest, no look-ahead.

## Success criteria (must pass ALL)

1. Nested-OOS Sharpe CI strictly > 0.
2. Beats the **same-44-universe fixed-24h-hold baseline** (isolates: does the
   adaptive exit add value over just holding?).
3. Beats a **random-exit placebo** (same entries, hold durations randomly drawn
   from the realized hold distribution) at p95 — isolates whether the
   *mean-reversion* trigger specifically is the alpha, not any exit.
4. Beats P1/P2 universe placebos at p95.
5. Per-fold ≥ 6/9 positive; per-cycle PnL NOT tail-concentrated
   (top-5%-of-cycles share ≪ clean-108's 48%).

Fail any → the mean-reversion-exit overlay does not add executable alpha; document
and close. Pass all → first executable candidate; proceed to forward-test design.

## Sequencing

Step 61 (iterative meme removal, ~2 hr) is running and CPU-heavy. Build Step 62-64
code now (no contention); defer heavy retrain/backtest execution until Step 61
finishes to avoid resource contention and keep numbers clean.
