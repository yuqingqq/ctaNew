# Recent cycles — v2 candidate (5.29 model, for cross-reference)

`recent_cycles_v2.csv` = post-cutoff cycles (2026-05-29 → 2026-06-04) from the **FROZEN 5.29 deploy model**
(`convexity_v1_{short,long}_model.pkl`, md5 long `7d320599…` / short `a2ea46de…`) run with the **v2 construction**:
equal-weight (notional-neutral) all regimes + DD-stop OFF + **bear K=2** + inv-vol sizing.

## ⚠️ REGENERATED 2026-06-05 — stale-preds bug fixed (XLM cross-check)
The first version of this file was generated from **stale stored preds**. `predict_twobook_incremental.py`
was **append-only**: it computed a pred for each new bar once (from whatever Vision pull was current) and
**never recomputed it**. Binance Vision daily archives publish ~1–2 days late and the most-recent days arrive
**incomplete**, so bars appended from a partial pull froze stale preds. Concretely: **XLM @ 5/29 00:00** had
already ripped **+25%** (trailing, PIT-correct), but its pred was frozen at **−0.13** (rank 88/94, dropped)
from a pre-rip partial pull; recomputing from the completed panel gives **+1.28 (rank #1 long)**. The bug
affected the **whole window** (39/41 cycles' picks changed on regen), not just XLM.

**Fix:** `predict_twobook_incremental` now **recomputes a trailing 10-day window** each run
(`PREDICT_RECOMPUTE_DAYS`), overwriting stale rows once the panel completes (historical seed untouched —
recomputing it with the frozen model would be look-ahead). After regen, 5/29 00:00 ties out **exactly** to the
v1-deploy / live-box picks: **longs HYPE/SEI/XLM, shorts BCH/ME/XMR**. Window totPnL +283 (stale) → +776 (correct).

**Lesson:** the **live box was fresher than this backtest** — its real-time feed saw XLM's move and longed it;
the Vision-archive incremental preds were stale. The cross-reference now reconciles.

## vs production v1
- Production v1 (deployed) = **flat in bear** (BEAR_MODE=flat) → no trades on the June bear days.
- v2 (this) = **trades the bear at K=2** → the 6/2+ rows show 2 longs / 2 shorts.

## Caveats
- June **funding is still stale (5/31)** — Vision's June monthly archive isn't published yet, so the model's
  funding features are forward-filled for 6/1–6/4. (Separate from the preds bug above; affects funding features
  only. Real fix needs the exec-server live feed or July's monthly archive.)
- Klines are Vision daily (through 6/4). The most-recent 1–2 days can still be partially incomplete at any pull;
  the trailing-recompute fix corrects them on the next run once Vision finalizes.

## Columns
cycle_id, open_time, regime, btc_ret_30d, top_k_long, bot_k_short, gross_target, gross_after_stop, stop_engaged,
long_ret_bps, short_ret_bps, pnl_bps.
