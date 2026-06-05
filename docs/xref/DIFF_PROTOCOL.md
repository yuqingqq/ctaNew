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
