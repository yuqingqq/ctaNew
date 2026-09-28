# movewatch — multi-exchange sudden-move alerts (P-2026-004)

**Status: live** (user systemd `movewatch.service` since 2026-09-28). Watches ~5,900 markets on 12 venue feeds in real time, detects sudden moves
(pumps and dumps) per **asset** across venues, and sends Telegram alerts that
escalate as a move extends and flag the reversal. Motivating case: QNT,
$66.87 → $374.50 (5.6×) over 2026-09-22..27 — see `docs/QNT_2026-09_CASE_STUDY.md`.

State of the project: **`STATUS.yml`** (tasks, flags) and **`HANDOFF.md`**
(done / in-progress / next / watch-outs). Read both first.

## What it watches

| Venue | Markets | Stream |
|---|---|---|
| Binance USDM perps | USDT/USDC perps | `kline_1m` (fstream `/market/` path) |
| Binance spot | USDT/USDC | `kline_1m` (data-stream.binance.vision) |
| Bybit linear / spot | USDT | `kline.1` (discovery by subscription — REST is geo-blocked here) |
| OKX swap / spot | crypto only (`instCategory 1`) | `candle1m` |
| Coinbase | USD/USDT/USDC | `matches` |
| Upbit, Bithumb | KRW | `trade` (also sets the live KRW/USD rate) |
| Bitget perps / spot | USDT/USDC, 24h vol ≥ $250k | `candle1m` (+ ~500-minute backfill on connect) |
| Hyperliquid | perps | `candle` 1m |

Plus a REST watcher (every 60 s): new markets on Upbit / Bithumb / Coinbase /
Binance spot / OKX, and Upbit caution flags (price fluctuation, volume soaring,
deposit soaring, global price gap, warning).

## How a move is detected

Per asset, every second, at horizons 1m / 5m / 15m / 1h / 4h / 24h
(`movewatch/signals.py`, thresholds in `config/movewatch.yaml`):

- **move net of the market** `e = r − median(r of liquid coins)` ≥ a floor (e.g. 10% at 1h)
- **unusual for this coin**: `|e| / (σ_1m·√h)` ≥ z-min (σ from the prior 24h, excluding the last hour)
- **volume-backed**: USD volume over h ≥ v-min × its normal pace
- **liquid**: ≥ $2M 24h volume summed across venues

Venue returns are combined 24h-volume-weighted; the same ticker trading at a
different price on another venue is a **different asset** (price clusters,
e.g. `META` vs `META@upbit`), never averaged in.

Messages (`movewatch/alerts.py`): **NEW** → **ESCALATE** at +10/20/30/50/75/100/150/200/300 %…
from the pre-move price (≥ 3 min apart) → **REVERSAL** when it gives back
clamp(¼ of the gain, 8 %, 20 %), which opens the opposite episode anchored at the
peak. Plus **MARKET** (market-wide median move), **LISTING**, **FLAG**.

Example (format; numbers from the 2026-09-28 dry run):

```
🟢 HBAR +17.1% in 4h
$0.1108 · from $0.0946 (05:38Z) · z 4.2 · vol ×14
15m -3.0% · 1h +6.9% · 4h +17.1%
2/2 venues agree · lead bitget_fut +17.1% (4h) · 1h vol $24.2M
mkt 1h -0.3%
```

## Presets (noise vs earliness)

Thresholds are calibrated on August 2026 (`docs/CALIBRATION.md`). The live default is the
**strict** preset — the declared budget pick (Binance-USDM replay: ~10 NEW / ~38 messages
per day). A looser preset alerts earlier for ~2× the messages; on QNT the ≤80 preset flagged
the final run at $237 (1.5 h before the $373 top) instead of $309:

```bash
python3 -m research.apply_pick data/replay/cal_v4/summary.json --budget 80   # or 40 / 60 / 120
systemctl --user restart movewatch
```

## Run

```bash
cd ~/ctaNew/movewatch
python3 -m pytest -q tests/                          # 18 unit tests (each with a falsifier)
python3 -m movewatch status                          # live health + last alerts
python3 -m movewatch run --dry-run --duration 300    # 5-min smoke, log + JSONL only
python3 -m movewatch test-telegram                   # one labelled test message
python3 -m movewatch run                             # live (Telegram if creds are set)
```

Credentials: `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (from `ctaNew/.env`) or the
`MOVEWATCH_TELEGRAM_*` overrides to send to a different chat. Without them every
alert still goes to `data/alerts/alerts-YYYYMMDD.jsonl` and the log.

Service (user systemd):

```bash
cp deploy/movewatch.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now movewatch
tail -f data/movewatch.log            # HEALTH line every minute; data/state/health.json
```

## Calibrate / replay

Memory rule: run at the default slot cap (4 GiB), never raise it; the replay aborts itself
above 2 GiB RSS (measured peak 0.8 GiB).

```bash
python3 -m research.download_usdm_1m                    # Binance USDM 1m, Aug + Sep (Vision)
MALLOC_ARENA_MAX=2 ../.worktrees/research_slot.sh 3 -- python3 -m research.replay --grid --tag cal_vN \
    --grid-floor 1.5,2,2.5,3 --grid-z 1.0,1.25,1.5 --grid-v 1.0,1.5 --grid-liq 2e6,5e6,1e7
python3 -m research.apply_pick data/replay/cal_vN/summary.json         # write the pick into the config
python3 -m research.qnt_case_study                      # QNT numbers for the case study
```

The replay runs the **same** `signals` + `alerts` code as the live engine on
minute closes (so replay detection times are an upper bound; live is up to 59 s
earlier). Results: `docs/CALIBRATION.md`.

## Layout

```
movewatch/        engine.py (1 s loop) · store.py (rings) · signals.py (trigger math)
                  alerts.py (episodes) · notify.py (Telegram/JSONL) · watchers.py (listings/flags)
                  instruments.py (normalization, FX) · venues/ (12 feeds)
research/         fetch_1m.py (multi-venue history) · download_usdm_1m.py · replay.py · qnt_case_study.py
config/           movewatch.yaml
deploy/           movewatch.service
docs/             QNT_2026-09_CASE_STUDY.md · CALIBRATION.md
data/             (gitignored) hist/ replay/ state/ alerts/ movewatch.log
```

Scope: market data and alerts only. No order placement, no exchange accounts.
