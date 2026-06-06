# Feature-vector diff protocol — settle generation-box vs live-box (June, swapped names)

Goal: determine whether the June pred divergence is **panel vintage** (resolves once data settles) or a
**real pipeline difference** between the generation box and the live box.

## My side (committed here)
`MY_PANEL_june_features_swapped.csv` — generation-panel feature vectors for the swapped names
(XLM/SEI/POLYX/ZRO + HYPE/SKR), 6/1→6/4 16:00, 23 cycles. Columns = the 19 model inputs + pred_long/pred_short:

- **Model inputs (17 V0 + 2 resid_rev):** return_1d, atr_pct, obv_z_1d, vwap_slope_96, bars_since_high,
  bars_since_high_xs_rank, autocorr_pctile_7d, corr_to_btc_1d, beta_to_btc_change_5d, idio_vol_to_btc_1h,
  idio_vol_to_btc_1d, **funding_rate, funding_rate_z_7d, funding_rate_1d_change**, rvol_7d, ret_3d,
  btc_rvol_7d, resid_rev_2, resid_rev_3
- `pred_long` = resid_rev model (V0+resid_rev), `pred_short` = base model (V0). Deploy models md5 long
  `7d320599…` / short `a2ea46de…`.

## Live box should emit (same schema)
Per (symbol, open_time) over June for the same names: `symbol, open_time, <19 features>, pred_long, pred_short`,
**plus** a per-symbol `last_settled_bar` (the latest kline timestamp the live feed had finalized) so we can
detect decision-time freshness. Use the live box's **current settled** values (and, if cheap, also what it
used at decision time — that's the freshness audit).

## Expected diff outcome (this is the discriminator)
- **funding_rate / funding_rate_z_7d / funding_rate_1d_change** — WILL differ. My side is forward-filled from
  5/31 (Vision June monthly archive unpublished); the live feed has the real values. Expected, not a bug.
  → patch my panel with the live funding, regenerate, re-check pred convergence.
- **The other 14 price features + resid_rev_2/3** — should MATCH within fp tolerance if same vintage.
  - If they match → divergence was 100% panel vintage (funding + the already-fixed stale-preds freeze). Done.
  - If they DON'T match → a real generation-vs-live pipeline/computation difference → chase that feature.

## ⚠️ float32 vs float64 when re-scoring (verified 2026-06-05)
The deploy pipeline runs in **float32** (panel dtype) and `apply_preproc` rank-transforms via `searchsorted`
(a step function). Re-scoring the **CSV's float64-widened** feature values flips rank buckets at boundaries →
spurious pred Δ up to 0.16 (mean ~0.05). The preds themselves are scored with the frozen pickle's
`apply_preproc` (NOT fresh-fit) and reproduce `base.parquet`/`long.parquet` to ~3e-8.

**To diff correctly:** use the `.parquet` (preserves float32), OR `.astype('float32')` before re-scoring, OR
compare ranks/picks (unaffected). The float32 re-score reproduces the dumped preds to 5.7e-08 — no real skew.

## Exact 41/41 reconstruction — my funding panel (added 2026-06-05)
`MY_funding_panel_94syms.{csv,parquet}` — the gen-side funding inputs for ALL 94 low-vol symbols, 5/15→6/4
(incl. 7d trailing context for the z-window). Columns: `symbol, open_time, funding_rate, funding_rate_z_7d,
funding_rate_1d_change`.

Substitute these into your panel for the 94 symbols and you reproduce `recent_cycles_v2.csv` **to the cycle
(41/41)** — funding is the only differing input (price features are settled / same vintage; the validated
mechanism already gets you 36/41, the remaining ~6% is funding-driven pred shifts).

⚠️ This is my **stale-funding vintage** (June = forward-filled 5/31, distinct-values/symbol = 1). It reproduces
MY records exactly; it is NOT settled truth. Settled funding comes from your live feed / the July monthly
archive. Use float32 when re-scoring (see the dtype note above).

## Funding vintage — precise characterization (2026-06-06)
Re-examined after a "mixed stale/fresh?" question. It is **uniformly stale, not a mix**:
- **0/94 symbols have real June funding** — all forward-filled from their last May settlement.
- **93/94 caches end 2026-05-31** (16:00–20:00, the last May archive settlement). The apparent 5/15→6/1
  "freeze-point" spread is **flat funding values** (e.g. TNSR/YGG/PROVE pinned at 5e-05), not data staleness —
  a change-detector artifact, not a vintage mix.
- **1 genuine outlier: VINEUSDT** cache ends **2026-04-28** (~5wk stale, likely delisted/illiquid). **NOT traded**
  in the records → zero impact.

So `MY_funding_panel_94syms` is the single 5/31 vintage end-to-end; substituting it reproduces the records
deterministically. Real June funding for all symbols comes only from the live feed / July monthly archive.

## The last cycle: klines-vintage in the cross-section (2026-06-06)
With funding matched, the residual 40/41→41/41 gap is **klines vintage in the 94-symbol cross-section**.
`bars_since_high_xs_rank` is **cross-sectional**, so even ONE symbol's klines differing (a missing recent bar
or a different rolling high) shifts the rank of others → can flip a borderline pick on a single cycle.

`MY_PANEL_features_ALL94_0529_0604.{parquet,csv}` — full V0+resid_rev feature vectors for **all 94 low-vol
symbols**, 5/29→6/4 (3,854 rows), incl. the kline-derived `bars_since_high, bars_since_high_xs_rank, atr_pct,
vwap_slope_96` + `pred_long/pred_short`.

**To close to 41/41:** diff `bars_since_high` (raw count, vintage-robust — no float issue) per symbol against
yours → the ~8-9 symbols that differ are the klines-vintage set. Reconcile those klines and the cross-sectional
rank matches → 41/41. The one miss is this residual, **not a logic error**. (Use parquet/float32 for the
continuous feats per the dtype note; `bars_since_high` itself is an integer count so it diffs cleanly.)
