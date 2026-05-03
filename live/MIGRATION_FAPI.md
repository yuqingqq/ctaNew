# Deploy v6_clean h=48 K=7 ORIG25 on a FAPI-accessible host

This is the single end-to-end runbook for setting up the recommended
production config (h=48 K=7 ORIG25, Sharpe +3.63 multi-OOS) on a fresh
server with Binance FAPI access. Follow top-to-bottom; every command is
copy-paste ready.

If you want the legacy h=288 K=5 cadence instead, see "Appendix:
legacy h=288 deployment" at the bottom.

---

## Pre-flight (verify the new server before doing anything)

```bash
# 1. FAPI must return HTTP 200
curl -s -o /dev/null -w "Binance FAPI: HTTP %{http_code}\n" \
  "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=1"
# Expect: HTTP 200

# 2. HL info API must be reachable (used for execution venue)
curl -s -o /dev/null -w "HL info:     HTTP %{http_code}\n" \
  -X POST -H "Content-Type: application/json" \
  -d '{"type":"meta"}' "https://api.hyperliquid.xyz/info"
# Expect: HTTP 200

# 3. Python 3.10+ available
python3 --version
# Expect: Python 3.10.x or higher
```

If any of those fail, stop and fix that first.

---

## One-time setup

### 1. Clone the repo and install dependencies

```bash
cd ~
git clone git@github.com:yuqingqq/ctaNew.git
cd ctaNew
pip install -r requirements.txt
```

### 2. Create `.env` with the h=48 deployment env vars

The cron wrapper sources `.env` and auto-exports every key, so this is
where the deployment config lives. Both `paper_bot` and the weekly
`train_v6_clean_artifact` cron pick these up automatically.

```bash
cat > /home/yuqing/ctaNew/.env <<'EOF'
# ---- v6_clean h=48 K=7 ORIG25 deployment config ----
HORIZON_BARS=48
TOP_K=7
UNIVERSE=ORIG25

# ---- optional: Telegram notifications ----
# Get a bot token from @BotFather; chat ID from @userinfobot
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ---- optional: override Binance FAPI URL (only if proxying) ----
# BINANCE_FAPI_URL=https://your-proxy.example.com
EOF
chmod 600 /home/yuqing/ctaNew/.env
```

If you skip Telegram, the bot logs locally and silently — fine for
solo monitoring via `cycle_summary`.

### 3. Make the cron wrapper executable

```bash
chmod +x /home/yuqing/ctaNew/live/run_with_env.sh
```

---

## Get the model artifact (pick ONE of the two paths)

### Path A — Copy the pre-trained artifact from the dev server (recommended, ~1 min)

You already have a verified `v6_clean_h48_*` artifact (sanity IC +0.0688
matches multi-OOS expectation). Skip the bulk data pull and retrain
this way:

```bash
# From the dev server, push the artifact to the new server:
scp -r /home/yuqing/ctaNew/models/v6_clean_h48_* \
  newserver:/home/yuqing/ctaNew/models/

# Verify on the new server:
ls -la /home/yuqing/ctaNew/models/v6_clean_h48_*
# Expect: v6_clean_h48_ensemble.pkl (~65 KB) + v6_clean_h48_meta.json
```

The artifact's training data ends 2026-04-08. Slightly stale, but the
weekly Monday cron will refresh it on the new server's data. Good for
first cycle.

### Path B — Pull data and retrain from scratch (~25 min)

Use this if you want the artifact built on the new server's full data
(slightly more recent training window).

```bash
cd /home/yuqing/ctaNew

# 1. Pull Binance Vision daily klines for ORIG25 (~700 MB, ~20 min)
python3 -m scripts.pull_xs_klines

# 2. Train the h=48 ORIG25 artifact (~1-2 min)
HORIZON_BARS=48 UNIVERSE=ORIG25 python3 -m live.train_v6_clean_artifact
# → writes models/v6_clean_h48_ensemble.pkl + v6_clean_h48_meta.json
# → final log line: "sanity: mean per-bar XS IC over last ~30d: +0.06xx"
#   (expect +0.05 to +0.08; matches multi-OOS expectation +0.0627)
```

---

## Smoke test (mandatory — run before installing cron)

```bash
cd /home/yuqing/ctaNew

# 1. One manual paper_bot cycle. Reads .env via the wrapper.
./live/run_with_env.sh -m live.paper_bot --source binance

# Expect:
#   - "Loaded model artifact: 25 symbols, 28 features, trained <ts>"
#   - "regime-active symbols at target_time: ~12-25/25"
#   - Cycle decision logged
#   - First-cycle row written to live/state/cycles.csv (had_prev_positions=0)
#   - 14 positions opened (7 long + 7 short)

# 2. Confirm state was written
./live/run_with_env.sh -m live.paper_bot --check-state
# Expect: 14 open positions (7L + 7S, β-neutral)

# 3. Confirm cycle_summary picks up h=48
./live/run_with_env.sh -m live.cycle_summary
# Expect header: "vs backtest expectation (h=48 K=7 ORIG25 multi-OOS @ 4.5 bps/leg taker)"
# Expect "backtest Sharpe: +3.63"
# (N<30 cycles → "(need N>=30 for meaningful comparison)" — that's normal)

# 4. Run the cycle isolation regression test
./live/run_with_env.sh -m live.test_cycle_isolation
# Expect: ✓ PASS
```

If any of those fail, fix before installing cron. The most common
failures and what they mean:

| Symptom | Cause | Fix |
|---|---|---|
| `Model artifact missing` | Artifact not in `models/` | Re-do "Get the model artifact" |
| `insufficient kline coverage` | klines didn't pull / FAPI rate-limited | Wait 60s, retry |
| `cycle_summary` shows h=288 numbers | `.env` not sourced or wrong vars | Check `.env` content; run with `HORIZON_BARS=48` explicit |
| `regime-active = 0/25` | `autocorr_pctile_7d` feature not present | Bug — file an issue, the panel build is broken |

---

## Install cron

```bash
crontab -e
# Paste:
```

```cron
# v6_clean paper-trade — h=48 K=7 ORIG25 deployment
# Env vars (HORIZON_BARS, TOP_K, UNIVERSE, TELEGRAM_*) loaded from
# /home/yuqing/ctaNew/.env via run_with_env.sh wrapper.

# Cycle decision every 4 hours at minute :01 (UTC bars at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
1 */4 * * *  /home/yuqing/ctaNew/live/run_with_env.sh -m live.paper_bot --source binance >> /home/yuqing/ctaNew/live/state/run.log 2>&1

# Hourly portfolio mark-to-market + Telegram snapshot at minute :05
5 * * * *  /home/yuqing/ctaNew/live/run_with_env.sh -m live.hourly_monitor >> /home/yuqing/ctaNew/live/state/hourly.log 2>&1

# Weekly model retrain — Mondays 00:30 UTC. Uses HORIZON_BARS + UNIVERSE
# from .env, so it produces v6_clean_h48_ensemble.pkl with ORIG25.
30 0 * * 1 /home/yuqing/ctaNew/live/run_with_env.sh -m live.train_v6_clean_artifact >> /home/yuqing/ctaNew/live/state/train.log 2>&1
```

Save and exit. Verify install:

```bash
crontab -l
# Expect: 3 active lines (paper_bot 4-hourly, hourly_monitor, weekly retrain)
```

---

## Operational verification (first 24h)

```bash
# After ~1 cron tick (4h or less), check the cycle log
tail -20 /home/yuqing/ctaNew/live/state/run.log

# After ~6 cycles (~24h), check the summary
HORIZON_BARS=48 ./live/run_with_env.sh -m live.cycle_summary
# Expect:
#   - N=5-6 realized cycles
#   - mean net/cycle in [-10, +20] bps (single-cycle variance is high)
#   - Sharpe: variable; only meaningful at N≥30 (~5 days of cycles)

# Hourly monitor logs:
tail -20 /home/yuqing/ctaNew/live/state/hourly.log
```

Forward Sharpe stabilizes around N=30-60 cycles (~5-10 days at h=48).
Compare to multi-OOS expectation +3.63 [+1.31, +6.14] only after N≥30.

---

## Operational targets and alarm thresholds

| Metric | Target (multi-OOS) | Investigate if |
|---|---|---|
| net/cycle (mean) | +4.33 bps | rolling-30 mean < -5 bps |
| Sharpe (rolling 30) | +3.63 | rolling-30 < -1 OR > +10 (CI overflow) |
| Hit rate | ~54% | rolling-30 < 40% |
| Fees / cycle | ~3.6 bps | > 8 bps (turnover spike) |
| Funding / cycle | depends on book; usually ±1 bps | > 5 bps drag on net |

When in doubt, sanity-check with:

```bash
HORIZON_BARS=48 ./live/run_with_env.sh -m live.cycle_summary
```

It auto-detects horizon, reports vs backtest expectation, and flags
material divergence at N≥30.

---

## What does NOT change vs the dev-server bot

- **Execution venue: still Hyperliquid.** L2 books, funding, mids
  fetched from `api.hyperliquid.xyz`.
- **Fee model: HL VIP-0 taker, 4.5 bps one-way.** No HYPE staking
  assumed. Stake 100+ HYPE for Bronze (-10% taker discount, +0.2
  Sharpe estimate) only after live performance confirms expectation.
- **Cycle bookkeeping: `last_cycle_mid` separate from
  `last_marked_mid`.** Per-cycle gross MtM is true cycle-to-cycle delta,
  not affected by hourly_monitor's intermediate marks.
- **Universe: ORIG25** (the original 25 BNF perps; excludes the 14
  newer entrants that hurt multi-OOS).
- **Feature set: v6_clean (28 cols) unchanged.** All h=48 reselection
  attempts hurt — see `docs/STATUS.md` for the negative-result table.

---

## Appendix: legacy h=288 K=5 deployment

If you want to run the daily-cycle config instead of h=48:

1. **`.env`**: omit `HORIZON_BARS`, `TOP_K`, `UNIVERSE` (or set
   `HORIZON_BARS=288 TOP_K=5`). Universe defaults to FULL 39 symbols
   for h=288.
2. **Artifact**: copy `models/v6_clean_ensemble.pkl` (legacy
   unsuffixed) instead of `v6_clean_h48_*`. Or train fresh with default
   env: `python3 -m live.train_v6_clean_artifact`.
3. **Cron**: replace `1 */4 * * *` with `1 0 * * *` (1×/day at 00:01 UTC).

Multi-OOS Sharpe at h=288 K=5: **+3.30** [+1.11, +5.42], 1620 cycles —
slightly lower than h=48 K=7 (+3.63) and forward CI tightens 6× slower
(30 cycles/month vs 180 cycles/month).

---

## Appendix: rollback

If h=48 misbehaves and you want to revert to the dev-server bot's last
config:

```bash
# 1. Stop the cron
crontab -r

# 2. Restore the dev server's crontab backup
scp devserver:/home/yuqing/ctaNew/live/state/crontab.backup.20260503.txt /tmp/
crontab /tmp/crontab.backup.20260503.txt

# 3. Switch back to legacy artifact
unset HORIZON_BARS TOP_K UNIVERSE
# (or remove those lines from .env)
```

Both artifacts (`v6_clean_ensemble.pkl` legacy and `v6_clean_h48_*`)
can coexist on the same server — paper_bot loads whichever matches the
active `HORIZON_BARS` env var, falling back to the legacy unsuffixed
file when no horizon-suffixed file is present.
