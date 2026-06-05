# Recent cycles — v2 candidate (5.29 model, for cross-reference)

`recent_cycles_v2.csv` = post-cutoff cycles (2026-05-29 → 2026-06-04) from the **FROZEN 5.29 deploy model**
(`convexity_v1_{short,long}_model.pkl`) run with the **v2 construction**: equal-weight (notional-neutral) all regimes
+ DD-stop OFF in bear + **bear K=2** + inv-vol sizing. Deterministic — the live v2 box reproduces this exactly.

## vs production v1
- Production v1 (deployed) = **flat in bear** (BEAR_MODE=flat) → no trades on the June bear days.
- v2 (this) = **trades the bear at K=2** → the 6/2+ rows show 2 longs / 2 shorts.

## What the live bear shows (honest)
- The market is in a deep bear (btc_30d ≈ −20% by 6/4). v2 trades it and is **volatile**: big swings incl. −362/−377 bps
  on 6/4 as BTC fell hard. June net is small/choppy — this is the genuine forward test of the bearK2 fix, and the recent
  days are drawdown-y. Treat the bear edge as unproven live until more data accrues.

## Caveats
- June **funding is stale (5/31)** — Vision's June monthly archive isn't published yet, so the model's funding features
  are forward-filled for 6/1–6/4 (affects the v2 bear picks; v1 is flat so unaffected).
- Klines are Vision daily (through 6/4); the OOS walk-forward validation (separate) put v2 at +4.32 Sharpe / −2,793 maxDD.

## Columns
cycle_id, open_time, regime, btc_ret_30d, top_k_long, bot_k_short, gross_target, gross_after_stop, stop_engaged,
long_ret_bps, short_ret_bps, pnl_bps.
