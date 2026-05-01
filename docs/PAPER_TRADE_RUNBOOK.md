# Paper-trade runbook

How to run the v6_clean live paper trade and check daily PnL.

## What you need

- A machine that's always on (laptop, VPS, server) — bot runs once per day
- Python 3.10+ with `requirements.txt` installed
- ~50 MB disk for caches, ~1 GB RAM during cycle runs
- Internet access to `api.hyperliquid.xyz` (no Binance fapi needed if using `--source hl`)

## One-time setup

```bash
cd /path/to/ctaNew

# 1. Install deps if not already
pip install -r requirements.txt

# 2. Train the model artifact (~1 minute)
python -m live.train_v6_clean_artifact
# Writes models/v6_clean_ensemble.pkl + meta.json

# 3. Verify the bot can run end-to-end
python -m live.paper_bot --source hl
# Should fetch klines, fetch L2 books, log a cycle decision

# 4. Verify state was saved
python -m live.paper_bot --check-state
# Should show 10 open positions
```

## Daily cron + hourly monitor

The bot rebalances daily (h=288 = 1d). The hourly monitor marks open
positions to current HL mids + funding accrual + sends Telegram.

**Cron's environment is stripped** (no `~/.bashrc`, no shell init), so
`.env` won't be auto-loaded. We use a small wrapper `live/run_with_env.sh`
that sources `.env` with auto-export then execs the venv python.

Set Telegram tokens in `<repo>/.env` (gitignored — never committed):

```bash
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=-100...
```

Make wrapper executable:
```bash
chmod +x /path/to/ctaNew/live/run_with_env.sh
```

Install crontab (`crontab -e`, paste this — adjust paths):

```cron
# v6_clean paper-trade

# Daily decision at 00:01 UTC (just after the 23:55 5-min bar closes)
1 0 * * *  /path/to/ctaNew/live/run_with_env.sh -m live.paper_bot --source hl >> /path/to/ctaNew/live/state/run.log 2>&1

# Hourly portfolio + Telegram snapshot at minute :05
5 * * * *  /path/to/ctaNew/live/run_with_env.sh -m live.hourly_monitor >> /path/to/ctaNew/live/state/hourly.log 2>&1

# Weekly model retrain — Mondays 00:30 UTC
30 0 * * 1 /path/to/ctaNew/live/run_with_env.sh -m live.train_v6_clean_artifact >> /path/to/ctaNew/live/state/train.log 2>&1
```

To test the wrapper before installing cron (simulates cron's stripped env):
```bash
env -i HOME=$HOME PATH=/usr/bin:/bin /path/to/ctaNew/live/run_with_env.sh \
    -c "from live.telegram import notify_telegram; print(notify_telegram('test'))"
# Should print: True (and message arrives in your chat)
```

### Telegram messages you'll receive

- **Daily** decision summary (after `paper_bot` runs): cycle PnL breakdown
  (gross MtM / fees / slippage / funding / net), new target portfolio,
  trade count + notional.
- **Hourly** portfolio snapshot (after `hourly_monitor` runs): hourly +
  cumulative MtM PnL, hourly funding cost, per-leg cumulative %, etc.

Without `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` set, the bot logs locally
and skips Telegram silently.

## Daily monitoring

After the bot has run a few cycles, check progress:

```bash
# Latest open positions + last 5 cycles
python -m live.paper_bot --check-state

# Full forward-test summary: cumulative PnL, Sharpe, vs backtest
python -m live.cycle_summary

# Tail the run log for errors
tail -50 live/state/run.log

# Show the cycle log directly
cat live/state/cycles.csv | column -ts,
```

The `cycle_summary` output prints both cost models (close-all-reopen-all,
turnover-aware), per-cycle stats, rolling 7d/30d Sharpe, and a head-to-head
vs the backtest expectation (+2.95 Sharpe, +26.7 bps/cycle net at K=5).

After **N≥30 cycles** (~1 month of running), the summary includes a bootstrap
95% CI on Sharpe — that's the empirical answer to "does v6_clean predict
real-time edge."

## What the cycle log records

`live/state/cycles.csv` columns:

| column | what it means |
|---|---|
| `decision_time_utc` | bar timestamp the decision was made on |
| `long_symbols` / `short_symbols` | top-K / bot-K names selected |
| `scale_L` / `scale_S` | β-neutral scaling per leg (clipped to [0.5, 1.5]) |
| `prior_spread_ret_bps` | gross PnL of the position held over the prior cycle (mid-to-mid) |
| `prior_entry_slip_bps_mean` | average L2 slippage paid at entry (signed: + = adverse) |
| `prior_exit_slip_bps_mean` | average L2 slippage paid at exit |
| `prior_fees_bps` | taker fees on close-all + reopen-all (conservative) |
| `tt_long_turnover` / `tt_short_turnover` | actual turnover (if we'd traded only the delta) |
| `tt_fees_bps` | taker fees under turnover-aware accounting |
| `net_bps` | spread − fees, close-all + reopen-all model |
| `tt_net_bps` | spread − fees, turnover-aware (matches backtest) |
| `new_entry_slip_bps_mean` | slippage paid on this cycle's entries (will be exits next cycle) |

## Reading the dual cost models

We log two cost variants because the L2 simulator naively closes-all and
reopens-all every cycle, which over-charges fees on names that carry over
between cycles. The backtest assumes you only trade the delta. So:

- `net_bps` (close-all + reopen-all) = pessimistic upper bound on cost
- `tt_net_bps` (turnover-aware) = realistic cost matching backtest assumptions

For comparing forward Sharpe to the +2.95 backtest expectation, **use
`tt_net_bps`**.

## Troubleshooting

**Bot exits with `insufficient kline coverage: 0/25`**
→ HL info API was unreachable. Check internet, retry. If geo-blocked
on a particular VPS region, try a different region.

**`Model artifact missing`**
→ Run `python -m live.train_v6_clean_artifact` first.

**Bot runs but predictions look identical day-over-day**
→ HL only has 15 days of 5min history. After 14 days of running the bot
gradually loses warmup. Re-train weekly to refresh.

**Slippage is much larger than +2 bps**
→ Reduce `INITIAL_EQUITY_USD` in `live/paper_bot.py` (default $10K). Smaller
notional = thinner book consumption. For research-grade paper trading,
$1K-$5K equity is fine; the strategy isn't sensitive to absolute size.

**Run log shows L2 fetch errors for some coins**
→ HL info API can rate-limit. The bot retries per-coin; if a few fail,
those legs are skipped (logged as warnings). Run continues. Re-running
the cycle on next cron tick recovers.

## Stopping / restarting

The bot is **stateless between runs** (state is in `live/state/`). To stop
forward test, just remove the cron entry. To restart with a clean slate:

```bash
rm -f live/state/positions.json live/state/cycles.csv
# (keep live/state/binance_5m/ klines cache; it's just kline data)
```

To deploy a new model (e.g., after fixing a feature):

```bash
# Re-train
python -m live.train_v6_clean_artifact

# Optional: clear positions to start fresh under new model
rm live/state/positions.json
```

## When to evaluate

After **30 cycles** (~1 month), `cycle_summary` will output a bootstrap CI
on Sharpe. Compare to backtest +2.95 [+0.85, +4.54]:

- **Forward Sharpe within [+0.85, +4.54]**: strategy transports cleanly. Consider
  scaling up live capital.
- **Forward Sharpe in [-1, +0.85]**: marginal. Run another 30 cycles before
  deciding. May need to adjust K, or add maker fills.
- **Forward Sharpe < -1 or CI excludes 0 negatively**: regime shift or
  modeling bug. Stop and investigate.

After 60-90 cycles the CI tightens enough to make a confident deployment
decision.
