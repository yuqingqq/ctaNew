# Recent cycles — production v1 (for cross-checking)

`recent_cycles_v1.csv` = the **last 15 cycles** of the **production v1 config**, for verifying the live forward system
reproduces the intended trades/PnL.

## Config that produced these
Deployed v1 (NOT the v2 candidate — all v2 env flags OFF):
- Universe: book-B (low-vol, exclude top-80 rvol), maturity≥180d + liquidity + hygiene.
- Signal: dual-pred — **long** leg ranked by resid_rev model, **short** leg by base model.
- Construction: K=3 L/S, **beta-neutral** in side, **mom** in bull, **flat** in bear (BEAR_MODE=flat), 24h/6-sleeve.
- Risk: equity DD-stop (k=2, g_floor=0.40) active.
- Cost: 4.5 bps/leg.

## Source & caveat
- These are from the **walk-forward backtest replay** on data **through 2026-05-30** (the last bar in the local panel).
  The **live forward cycles past 2026-05-30 are on the exec server** — not reproducible here without that feed.
- So use this to cross-check: (a) the *logic* (which symbols get picked long/short given the preds), (b) the PnL/cost
  computation, (c) any overlapping dates with the live system. It is NOT the live feed itself.
- Deterministic: model@fit_cut 2026-05-29 + monthly universe re-rank → these cycles reproduce exactly on the same data.

## Columns
`top_k_long` / `bot_k_short` = the 3 long / 3 short symbols selected that cycle · `regime` bull/side/bear ·
`gross_target` pre-stop gross · `gross_after_stop` post DD-stop · `stop_engaged` · `turnover` · `cost_bps` ·
`long_ret_bps`/`short_ret_bps` per-leg realized · `gross_pnl_bps` pre-cost · `pnl_bps` net · `equity_post`.

Note: all 15 recent cycles are **side** regime (BTC-30d ≈ −6% as of 2026-05-30, above the −10% bear threshold).
