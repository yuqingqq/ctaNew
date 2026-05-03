# Health checklist for v6_clean h=48 K=7 ORIG25 deployment

After following `live/MIGRATION_FAPI.md` and starting the bot on a fresh
server, run through these tiered checks to confirm everything is wired
up correctly and the bot is performing as expected.

Run on the deploy server. Each command is copy-paste ready.

---

## Tier 1 — Did the first cycle complete cleanly?

```bash
cd /home/yuqing/ctaNew
tail -60 live/state/run.log
```

Look for these markers in order:
- `Loaded model artifact: 25 symbols, 28 features, trained <ts>` → horizon-suffixed artifact loaded
- `regime-active symbols at target_time: <N>/25` → regime gate working (N typically 12-25)
- `execute_cycle_turnover_aware: <N> trades` → first cycle = 14 trades; later cycles = 0-14
- No `Traceback` / `ERROR` lines

If you see a `Traceback`, capture the last ~30 lines of `run.log`.

---

## Tier 2 — Are the right env vars active?

```bash
# 1. Confirm .env content
grep -E "HORIZON_BARS|TOP_K|UNIVERSE" /home/yuqing/ctaNew/.env
# Expect:
#   HORIZON_BARS=48
#   TOP_K=7
#   UNIVERSE=ORIG25

# 2. Confirm the LOADED ARTIFACT is the h=48 one (not the legacy fallback)
HORIZON_BARS=48 python3 -c "
import sys; sys.path.insert(0, '.')
import live.paper_bot as pb
m, meta = pb.load_model_artifact()
print(f'horizon_bars: {meta[\"horizon_bars\"]}')
print(f'universe_mode: {meta.get(\"universe_mode\", \"n/a\")}')
print(f'n_symbols: {len(meta[\"sym_to_id\"])}')
print(f'TOP_K env-resolved: {pb.TOP_K}')
print(f'HORIZON_BARS env-resolved: {pb.HORIZON_BARS}')
"
# Expect: horizon_bars: 48, universe_mode: ORIG25, n_symbols: 25, TOP_K: 7, HORIZON_BARS: 48
```

If `n_symbols=39` or `horizon_bars=288`, the env vars aren't being
picked up — the bot is running the wrong config. Most common cause:
`.env` was edited but the cron is using a stale shell. Test the
wrapper end-to-end:

```bash
env -i HOME=$HOME PATH=/usr/bin:/bin /home/yuqing/ctaNew/live/run_with_env.sh \
    -c "import os; print(os.environ.get('HORIZON_BARS'))"
# Expect: 48
```

---

## Tier 3 — Is state correct after the first cycle?

```bash
# 1. Position count: should be 14 (7 long + 7 short)
python3 -c "
import json
with open('live/state/positions.json') as f:
    p = json.load(f)
print(f'Open positions: {len(p)}')
longs = [x for x in p if x['weight'] > 0]
shorts = [x for x in p if x['weight'] < 0]
print(f'  Long ({len(longs)}): {[x[\"symbol\"] for x in longs]}')
print(f'  Short ({len(shorts)}): {[x[\"symbol\"] for x in shorts]}')
"
# Expect: 14 positions, 7 long + 7 short

# 2. Cycle log: at least one row
column -ts, -W$(tput cols) live/state/cycles.csv | head -3
# Expect: header + at least one data row.
#   First cycle:    had_prev_positions=0, gross_pnl_bps=0 (no prior portfolio to mark)
#   Subsequent:     had_prev_positions=1, gross_pnl_bps populated
```

---

## Tier 4 — Is cron actually scheduled and firing?

```bash
# 1. Crontab installed?
crontab -l
# Expect: 3 lines (paper_bot 4-hourly, hourly_monitor, weekly retrain)

# 2. When did each cron last fire?
ls -la live/state/run.log live/state/hourly.log
# Expect: run.log mtime within the last 4h (paper_bot)
#         hourly.log mtime within the last 1h (hourly_monitor)

# 3. What does cron's syslog say?
grep -i "cron\|crontab" /var/log/syslog 2>/dev/null | tail -20 || \
  journalctl -u cron -n 20 --no-pager 2>/dev/null || \
  echo "(cron logs unreadable — try: sudo journalctl -u cron)"
```

---

## Tier 5 — Does cycle_summary report the right horizon?

```bash
HORIZON_BARS=48 python3 -m live.cycle_summary
```

Expected header:
```
vs backtest expectation (h=48 K=7 ORIG25 multi-OOS @ 4.5 bps/leg taker)
  backtest Sharpe:           +3.63
  backtest net/cycle:        +4.33 bps
  backtest spread/cycle:     +7.90 bps
```

At N<30 cycles you'll see `(need N>=30 for meaningful comparison)` — that's normal.

If the header says **h=288**, the env var isn't reaching cycle_summary
— either fix `.env` or always invoke with `HORIZON_BARS=48` prefix.

---

## Tier 6 — After 24h (~6 cycles), does it look healthy?

```bash
# Forward-test summary
HORIZON_BARS=48 python3 -m live.cycle_summary

# Look for:
#   - N=5-6 realized cycles
#   - mean net/cycle in [-15, +25] bps (single-cycle variance is high; OK either direction)
#   - hit rate variable; only meaningful at N≥30
#   - no negative funding > +5 bps drag (would suggest funding is eating PnL)

# Run log error check
grep -iE "error|traceback|warning" live/state/run.log | tail -10
# Expect: zero errors. Warnings about funding fetch retries are OK.

# Hourly monitor check
tail -20 live/state/hourly_pnl.csv | column -ts,
# Expect: rows accumulating ~hourly. funding_usd should be small (cents/dollar scale).
```

---

## Tier 7 — After N≥30 cycles (~5 days at h=48)

```bash
HORIZON_BARS=48 python3 -m live.cycle_summary
```

Look for the "vs backtest expectation" section at the end:

```
forward Sharpe: <X.XX>  (Δ <X.XX> vs backtest)
```

Verdict thresholds:

| Δ vs backtest Sharpe (+3.63) | Verdict |
|---|---|
| within ±1.5 | ✓ consistent with backtest |
| in [-3.0, -1.5] | ~ within wide CI; keep running |
| < -3.0 | ⚠️ materially below; investigate |

At N=30, forward CI is still wide; `~ within wide CI` is normal for
the first month. By N=60-90 the point estimate stabilizes.

---

## Operational alarm thresholds

| Metric | Target (multi-OOS) | Investigate if |
|---|---|---|
| net/cycle (mean) | +4.33 bps | rolling-30 mean < -5 bps |
| Sharpe (rolling 30) | +3.63 | rolling-30 < -1 OR > +10 (CI overflow) |
| Hit rate (rolling 30) | ~54% | rolling-30 < 40% |
| Fees / cycle | ~3.6 bps | > 8 bps (turnover spike) |
| Funding / cycle | ±1 bps typical | > 5 bps drag on net |

---

## Daily one-liner

After it's been running a few days, this is the single command for a
quick health check:

```bash
HORIZON_BARS=48 python3 -m live.cycle_summary | head -50
```

Tells you: cycles run, mean PnL, Sharpe, hit rate, recent cycles
table, and head-to-head vs backtest expectation.

---

## What to capture if something looks off

If any tier fails:
1. The exact failing output
2. `tail -100 live/state/run.log`
3. `crontab -l`
4. `cat /home/yuqing/ctaNew/.env` (redact `TELEGRAM_*` tokens)

That's enough to diagnose 95% of issues.
