# Migrating live bot to a FAPI-accessible host

The model is trained on Binance USDM data (`data/ml/test/parquet/klines/`).
The cron currently runs `--source hl` because Binance FAPI returns HTTP 451
from this host. Hyperliquid 5min klines are 5-30× smaller than Binance for
the same coin (real venue liquidity gap), so feeding HL volume to a
Binance-trained model creates ~30-40% selection drift (`drift_v2` measured
Spearman = 0.64 between HL-fed and Binance-fed predictions).

Moving the bot to a host where Binance FAPI is reachable lets us run
`--source fapi` and feed the model its training-distribution data. The
1.4 bps/cycle basis cost of trading on HL with Binance-derived predictions
is documented in `outputs/binance_hl_basis_1h.log`.

## Pre-flight on the new host

```bash
# 1. Confirm FAPI is reachable
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=1"
# Expect: HTTP 200
```

## Files to copy from the current host

```
ctaNew/
├── live/state/positions.json          # current open positions
├── live/state/cycles.csv              # cycle history
├── live/state/hourly_pnl.csv          # hourly MtM history
├── live/state/hourly_last_tick.json   # hourly monitor cursor
├── models/v6_clean_ensemble.pkl       # model artifact
├── models/v6_clean_meta.json          # model metadata
├── data/ml/test/parquet/klines/       # training data (large; needed only
│                                      # if you also want to retrain on
│                                      # the new host — otherwise skip)
└── .env                               # TELEGRAM_BOT_TOKEN, etc.
```

Skip `live/state/binance_5m/` — the bot will refetch it cleanly on first
run. Don't bring the HL parquets over.

## Cron on the new host

Replace `--source hl` with `--source fapi`. Recommended:

```cron
# v6_clean paper-trade — Binance for predictions, HL for execution
1 0 * * *  /path/to/ctaNew/live/run_with_env.sh -m live.paper_bot --source fapi >> /path/to/ctaNew/live/state/run.log 2>&1
5 * * * *  /path/to/ctaNew/live/run_with_env.sh -m live.hourly_monitor   >> /path/to/ctaNew/live/state/hourly.log 2>&1
30 0 * * 1 /path/to/ctaNew/live/run_with_env.sh -m live.train_v6_clean_artifact >> /path/to/ctaNew/live/state/train.log 2>&1
```

If FAPI ever flaps, set `--source auto` to fall back to Vision (1-day-lag
archive) automatically — but recognize that with Vision your decisions
are made on yesterday's data, so it should be a temporary fallback only.

## Optional: route FAPI through a relay/proxy

`BINANCE_FAPI` is now env-configurable. To point at a proxy (e.g. an SG
VPS that proxies fapi.binance.com), add to `.env`:

```bash
BINANCE_FAPI_URL=https://your-proxy.example.com
```

The proxy must mirror Binance's `/fapi/v1/klines` interface verbatim.

## First-run sanity checks

```bash
cd /path/to/ctaNew

# 1. Manual cycle (won't actually trade — paper mode regardless)
PYTHONPATH=. live/run_with_env.sh -m live.paper_bot --source fapi

# 2. Confirm cycle row written and matches expectations
tail -1 live/state/cycles.csv

# 3. Validate that pipeline behaves like backtest
PYTHONPATH=. live/run_with_env.sh -m live.replay_paper_bot
# Expect: ✓ Replay matches backtest within tolerance

# 4. Confirm the regression test still passes
PYTHONPATH=. live/run_with_env.sh -m live.test_cycle_isolation
# Expect: ✓ PASS
```

## What does NOT change

- Execution venue: still Hyperliquid (HL L2 books, HL funding, HL mids).
- Fee/slippage model: still HL VIP-0 taker (HL_TAKER_FEE_BPS = 4.5 one-way).
- Cycle bookkeeping: last_cycle_mid is read from HL execution mids,
  unchanged. Predictions only affect WHICH symbols are picked; PnL accounting
  is venue-anchored to HL.
- Model: same `models/v6_clean_ensemble.pkl`. No retrain required.

## What you should expect to see change

- The cycle row's `long_symbols` / `short_symbols` should diverge from
  what `--source hl` would have picked (drift_v2 estimated ~30-40%
  selection turnover). This is the intended effect — predictions now
  match the training distribution.
- `gross_pnl_bps` should track backtest expectations more closely
  (mean +30.7 bps/cycle gross at K=5 β-neutral per multi-OOS).
