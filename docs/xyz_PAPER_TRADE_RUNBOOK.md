# xyz paper-trade runbook

How to run the v7 xyz alpha-residual shadow trade and read the results.

## What this is

A daily shadow-trade harness that:
- Loads the v7 xyz model artifact (15-LGBM ensemble, frozen at training time)
- Refreshes yfinance daily prices for full S&P 100
- Predicts cross-sectional residual returns for tomorrow
- Filters to the chosen execution universe (default: 11 Tier A+B names)
- Applies hysteresis + dispersion gate
- Logs target weights and (with L2 fills) realistic shadow P&L cycle-by-cycle

**No real orders. No real money. Logs only.**

## Spec at a glance

| | |
|---|---|
| Model artifact | `models/v7_xyz_ensemble.pkl` (15 models, 18 features) |
| Training universe | full S&P 100 daily, 2013-2026 |
| Default execution preset | `tier_ab` (11 names) |
| Position rule | top-K=4 long, K=4 short, hysteresis exit at K+M=5 |
| Cadence | daily (one decision per US trading day, 1d hold) |
| Gate | dispersion ≥ 60-pctile of trailing 252d (PIT) |
| Cost in shadow accounting | per-cycle L2 fill simulation + 0.8 bps/side taker fee (configurable) |
| Notional per leg | $10,000 default (configurable via `--notional-usd`) |
| Backtest active Sharpe | +3.25 [+1.93, +4.62] @ 1.5 bps/side, +3.29 @ 0.8 bps |

Other presets: `tier_a` (8 names, K=3/M=1), `full15` (all 15, K=5/M=2).

## What you need

- Python 3.10+ with `requirements.txt` installed
- ~10 MB disk (state + cycles log)
- Internet access to `query2.finance.yahoo.com` (yfinance) and `api.hyperliquid.xyz`

No exchange API keys — shadow only, no orders.

## One-time setup

```bash
cd /path/to/ctaNew

# 1. Train artifact (~15 minutes)
python -m live.train_v7_xyz_artifact

# 2. Verify end-to-end with cached data first
python -m live.xyz_paper_bot --no-refresh
# Should print: artifact loaded, panel built, predictions ranked, target weights
# selected, xyz L2 books fetched. State saved to live/state/xyz/positions.json.

# 3. Verify with fresh refresh (~15 sec yfinance pacing)
python -m live.xyz_paper_bot
# Note `decision_ts` should match the most recent US trading day.
```

## Cron schedule

Two cron jobs:

1. **Daily rebalance** — runs `xyz_paper_bot.py` at 21:30 UTC on weekdays,
   produces a Telegram rebalance summary
2. **Hourly snapshot** — runs `xyz_hourly_monitor.py` at minute :05 every
   hour, marks open positions to xyz mids + sends Telegram snapshot

Both jobs use `live/run_xyz.sh` as a wrapper that sources `.env` (for
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) and uses the venv python.

```cron
# v7 xyz paper-trade — daily rebalance after US close (Mon-Fri only)
30 21 * * 1-5 /home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_paper_bot >> /home/yuqing/ctaNew/live/state/xyz/cron.log 2>&1

# v7 xyz hourly portfolio snapshot
5 * * * * /home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_hourly_monitor >> /home/yuqing/ctaNew/live/state/xyz/cron.log 2>&1
```

Notes:
- 21:30 UTC = US RTH close +30 min (yfinance lag is ~15 min after close in EDT).
- Doesn't filter US holidays — on a holiday yfinance returns no new bar, the
  same-day guard kicks in and leaves positions/P&L state unchanged.
- Hourly monitor runs 24/7 (xyz perps trade 24/7 even when cash is closed),
  so it picks up off-hours drift too.

### Telegram setup

`live/run_xyz.sh` automatically loads `.env` from the repo root. Add or
verify these two keys in `<repo>/.env` (gitignored — never commit):

```bash
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=-100...
```

If env vars are missing, the bot still runs (CSV logs still update);
Telegram is silently skipped. To verify Telegram works:

```bash
/home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_paper_bot --no-refresh
# Look for "telegram rebalance: sent (NNN chars)" in the output
```

## Run modes

```bash
python -m live.xyz_paper_bot                                 # default cycle
python -m live.xyz_paper_bot --universe tier_a               # 8 names, conservative
python -m live.xyz_paper_bot --universe tier_ab              # 11 names, default
python -m live.xyz_paper_bot --universe full15               # all 15
python -m live.xyz_paper_bot --notional-usd 25000            # $25k/leg
python -m live.xyz_paper_bot --taker-fee-bps 1.0             # override fee
python -m live.xyz_paper_bot --no-refresh                    # skip yfinance
python -m live.xyz_paper_bot --check-state                   # print state + cycles tail
```

Universes:

| preset | names | K | M | backtest active Sh |
|---|---|---|---|---|
| `tier_a` | 8 (cleanest basis) | 3 | 1 | +2.78 |
| `tier_ab` | 11 (drops Tier C) ⭐ | 4 | 1 | +3.25 |
| `full15` | 15 (full backtest panel) | 5 | 2 | +3.01 |

Recommendation: stay on `tier_ab`. Mixing presets across cycles makes the
realized Sharpe series inconsistent.

## State files

```
live/state/xyz/
  positions.json        Current target weights, entry mids/fills, last_marked_mids (for hourly drift)
  cycles.csv            Append-only: one row per closed cycle with realized P&L
  predictions.csv       Append-only: every cycle's full per-symbol predictions (forensics)
  hourly_pnl.csv        Append-only: one row per hourly tick with portfolio MtM
  hourly_last_tick.json Last hourly tick timestamp (for funding window)
  cron.log              Stdout/stderr from both cron jobs (rotate as needed)
```

### `positions.json` schema

| field | meaning |
|---|---|
| `decision_ts` | Latest panel ts the model decided on (US date) |
| `decision_at_utc` | Wall-clock when this cycle ran |
| `long`, `short` | Lists of symbols in each leg |
| `entry_mids` | Rolling mark references: entry VWAP for new names, current mid for held names |
| `top_k`, `exit_buffer` | Active K/M params |
| `preset`, `universe` | Active preset name + universe list |
| `gate_open` | Whether the dispersion gate let this cycle trade |
| `gate_disp`, `gate_thresh` | Dispersion value and PIT threshold |
| `missing_mids` | Names where xyz mid couldn't be fetched; current bot refuses to mutate state when required books are missing |

### `cycles.csv` columns

| col | meaning |
|---|---|
| `decision_ts` | When the cycle's decision was made |
| `close_ts` | When the cycle was marked-out and closed |
| `K` | Top-K used |
| `n_long`, `n_short` | Position counts |
| `n_missing_mid` | Legacy field; current bot aborts before writing a cycle if required books are unavailable |
| `long_alpha_bps` | Mean log-return of long leg, bps. **Includes embedded slippage on entries/exits**; held names use mid-to-mid (no slip re-paid). |
| `short_alpha_bps` | Mean log-return of short leg, bps (negative = shorts profit) |
| `spread_bps` | `long_alpha - short_alpha`. The realized P&L per cycle, with realistic taker slippage embedded. |
| `avg_entry_slip_bps` | Diagnostic: avg entry slippage over names that ROTATED IN this cycle (positive = adverse). 0 if no rotation. |
| `avg_exit_slip_bps` | Diagnostic: avg exit slippage over names that ROTATED OUT this cycle. 0 if no rotation. |
| `long_chg`, `short_chg` | Symdiff vs prev sets (rotation count) |
| `turnover` | `(long_chg + short_chg) / (2K)` |
| `fee_bps` | `(long_chg + short_chg) / K * taker_fee_bps`. Explicit fee on rotation events only. |
| `net_bps` | `spread_bps - fee_bps`. Slippage is in spread; fee is separate. |
| `notional_usd`, `taker_fee_bps` | Cycle config snapshot |
| `gate_open_now`, `disp_now`, `thresh_now` | Current-cycle gate state |

### How fills work in shadow

Each cycle:
1. Fetch L2 orderbook for every name in the universe + all prev/new positions.
2. For each entry trade (newly added to a leg): walk the book taker-style,
   compute volume-weighted average price (`vwap`) until `notional_usd / K` filled.
   If the book cannot fully fill the simulated order, the cycle aborts before
   mutating state.
3. For each exit trade (rotated out of a leg): same on the opposite side.
4. Held names don't simulate a trade — mid-to-mid mark only.

Slippage is the signed bps gap between `vwap` and mid (positive = paid more
than mid, adverse for the taker). It's already embedded in `spread_bps`
because we use `vwap` as the entry/exit reference. The `avg_entry_slip_bps`
and `avg_exit_slip_bps` columns are *diagnostic only* — they let you
compare realized live slippage to expectations.

Fees are charged separately as `fee_bps` (only on rotated names; held
names don't pay).

Cumulative shadow P&L over many cycles ≈ sum of `net_bps`. After ~30
cycles, run the quick-check script below to compare to backtest.

## Reading the cycles log

Per-day expectations from the backtest (tier_ab, K=4, M=1, 1.5 bps cost):
- Active Sharpe: +3.25, CI [+1.93, +4.62]
- Net per cycle: ~+5 bps median, with realized stdev ~50 bps
- Hit rate: ~60% positive cycles
- Turnover: low (hysteresis keeps positions, expect 0.0-0.3 typical)
- Active Sharpe @ N=30 cycles will have wide CI; N=60 narrows substantially

Realistic-cost adjustment: shadow harness uses L2-walked vwap (typical
slippage +2-5 bps/side observed at $10k notional/leg) plus 0.8 bps taker
fee, so realized live cost will be ~3-6 bps RT per rotation vs backtest's
3 bps RT assumption. Hysteresis means most cycles have low turnover, so
this drag is mild. Selection-bias haircut still applies (~25-30%) so
expect realized live Sharpe in **+2.0 to +2.5** range.

Quick check:

```bash
python -c "
import pandas as pd, numpy as np
d = pd.read_csv('live/state/xyz/cycles.csv')
print(f'cycles: {len(d)}')
print(f'mean net bps: {d.net_bps.mean():+.2f}')
print(f'std net bps:  {d.net_bps.std():.2f}')
print(f'hit rate:     {(d.net_bps > 0).mean():.0%}')
print(f'cumulative net bps: {d.net_bps.sum():+.1f}')
print(f'mean turnover: {d.turnover.mean():.2f}')
print(f'mean fee bps:  {d.fee_bps.mean():+.2f}')
print(f'mean slip in:  {d.avg_entry_slip_bps.mean():+.2f}')
print(f'mean slip out: {d.avg_exit_slip_bps.mean():+.2f}')
sh = d.net_bps.mean() / d.net_bps.std() * np.sqrt(252) if d.net_bps.std() > 0 else 0
print(f'annualized Sharpe (rough): {sh:+.2f}')
"
```

After **N≥30 cycles** (~6 weeks of trading days), compare realized
annualized Sharpe to the backtest +3.25-3.29. If it lands inside the
backtest CI [+1.93, +4.67], the strategy is transporting; if below or
outside, debug or recalibrate.

## Annual retrain

The v7 spec mandates **annual retraining** (memory: more frequent hurts;
less frequent decays). Set a calendar reminder. To retrain:

```bash
python -m live.train_v7_xyz_artifact
# Re-runs full panel build + 15-LGBM ensemble fit. Overwrites artifact.
```

Sanity-check after retrain by inspecting the per-bar XS IC printed at the
end of training — should be order +0.10 to +0.20 vs `fwd_resid_1d` on
recent in-sample data. Anything below +0.05 means something's broken;
investigate before trusting.

## Failure modes and what to do

| symptom | likely cause | action |
|---|---|---|
| `decision_ts` lags wall-clock by 1 day | yfinance hasn't updated | wait 30 min or re-run; if persistent, check yfinance status |
| `gate=CLOSED` consistently | dispersion below 60-pctile (low-vol regime) | normal during quiet periods; strategy holds flat |
| same-day guard fires repeatedly | cron running multiple times per day | debug cron; harness leaves state/P&L unchanged |
| big negative `spread_bps` | bad cycle, real risk | normal — single cycles can be -50bps+; only worry at multi-cycle drawdown |
| `net_bps < spread_bps` mismatch | turnover/fee calc surprise | `fee_bps = (long_chg + short_chg) / K * taker_fee_bps`; verify in code |
| bot aborts on missing book or partial fill | xyz API/depth gap for a required position | leave state untouched; rerun later or reduce notional/drop symbol if persistent |

## Where the strategy could break

(See `project_xyz_alpha_residual_v7.md` for full discussion.)

- **xyz fee regime change** — strategy assumes ~1 bps taker. If growth-mode rebate ends, fees jump 10×; revisit. Use cycles.csv to estimate sensitivity quickly.
- **Basis quality drift** — `xyz_data_quality.py` was the snapshot at deploy. Re-run quarterly to confirm Tier B/C names still behave.
- **2021-style meme-bubble year** — strategy expects mean-reversion; gets momentum chasing. Backtest 2021 was -2.11 active Sh.
- **15-name execution universe is concentrated** — 11 Tier A+B is even more so. One name's idiosyncratic blow-up could exceed daily strategy P&L.

## Promoting from shadow to live

Out of scope for this runbook. Minimum gates before considering it:
1. ≥30 shadow cycles logged
2. Realized Sharpe within backtest CI lower bound
3. No missing-book or partial-fill aborts for ≥10 consecutive cycles
4. xyz fee regime confirmed favorable
5. Real executor written + tested independently
6. Drawdown brake / kill-switch implemented
7. Position sizing and risk overlay decided

None of these are done as of the runbook's writing.
