# Live run cycles export (convexity_v2)

`convexity_v2_cycles_through_2026-06-21.csv` — the live `convexity_v2` **modeled
per-cycle track** (the forward-test baseline), exported 2026-06-23, truncated at the last
cycle of 06-21 (`2026-06-21 20:00`). This is the live system's own `cycles.csv`; use it as
the baseline to compare backtest reproductions / the liquidity-filter & concentration tests
against.

## Window
- **573 cycles**, `2026-03-03 00:00` → `2026-06-21 20:00` (4h grid).
- In-sample / backtest before the 5.29 training cutoff; **OOS-live forward = 144 cycles since 5.29**.

## Key columns (28 total)
`open_time, regime, pnl_bps` (per-cycle modeled return, perfect-fill net of a cost assumption),
`gross_pnl_bps, cost_bps, turnover`, `top_k_long / bot_k_short` (the new sleeve's picks),
`long_ret_bps / short_ret_bps` and `long_alpha_bps / short_alpha_bps` (per-leg attribution;
alpha = BTC-beta-residualized), `btc_ret_30d, pred_disp, gross_after_stop, stop_engaged, n_trades`.

## Caveat — 27 NaN-alpha rows
Cycles `2026-06-05 12:00 → 2026-06-09 20:00` have **NaN** `long/short_(ret|alpha)_bps` — a
logging gap in the settle/catch-up path (fixed 06-10, commit `4d55ecc`). **`pnl_bps` is correct
for every cycle**; only the per-leg decomposition is missing for those 27. Recompute from the
panel if the attribution is needed there.
