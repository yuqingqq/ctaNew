# HANDOFF — P-2026-004 movewatch

Updated: 2026-09-28 10:40Z (end of session 1). Single-seat project: this file and
`STATUS.yml` are the state; update both after every completed step.

## State in one paragraph

**Live.** `movewatch.service` (user systemd, enabled) streams ~5,900 markets on 12 venue
feeds and sends Telegram alerts using the **strict preset** (logic v4, `cal_v4` pick:
floor ×3.0, z ×1.25, v ×1.0, liquidity ≥ $5M → replay 9.7 NEW / 38.4 messages per day on
Binance USDM in August; live volume will be higher — more venues and assets). Memory is
capped and measured (replay 0.815 GiB peak at the default 4 GiB slot cap; service 332 MB
peak vs `MemoryMax=1536M`). The clean test of the final logic is the forward live log.

## Done (session 1)

- **QNT case study** — `docs/QNT_2026-09_CASE_STUDY.md` (every number from
  `research/qnt_case_study.py`): $66.87 → $374.50 (5.6×), peak 09-27 22:56 UTC; catalyst
  09-24 (The Clearing House + UK Finance); perps 88% of volume; all venues in lockstep at
  1m; squeeze signature (OI $4.0M → $56.6M, funding negative on each leg).
- **Live system** — 12 feeds, 1 s engine, shared `signals.py` / `alerts.py` (same code in
  the replay), Telegram + JSONL notifier, burst digest, listing + Upbit-flag watcher,
  Binance funding in messages, state snapshot + graceful SIGTERM, `status` command.
- **Calibration** — `docs/CALIBRATION.md`: protocol declared before results; grid 1
  failed the budget (best 39.9 NEW/day) → pre-declared grid 2; logic iterated v2 → v4 on
  the QNT failure modes (each change recorded with what had been seen). Frontier (2d)
  and QNT-per-preset (2e) tables for choosing noise vs earliness.
- **Tests** — `tests/test_core.py`, 18/18 pass, each with a falsifier.
- **Deploy** — Telegram test message delivered; service installed 10:37:06Z.

## Next (ranked)

1. **Watch the first live day.** `python3 -m movewatch status`; `tail -f data/movewatch.log`
   (HEALTH every minute: `rss_mb`, feeds up/shards, reconnects). Expect ~40–70 messages/day.
2. **Preset choice is the user's.** Strict (default) caught QNT's run first at $177.55 and
   the blow-off at $308.82; the ≤80 preset at $147.67 and $237.04 (1.5 h before the top)
   for ~2× the messages. Switch: `python3 -m research.apply_pick data/replay/cal_v4/summary.json
   --budget 80 && systemctl --user restart movewatch`.
3. **7-day forward review (due 2026-10-05, STATUS M8):** messages by kind/day, misses vs
   ex-post big moves (same event definition as the replay), false alarms, memory trend.
   Any threshold change is picked on those days and validated on LATER days.
4. **Wake path for calm names (M9, measured, NOT deployed):** QNT 09-24 was +8.3% / 1h at
   z 8.8, volume 20× (raw) — blocked only by the size floor (strict 1h = 30%). Added as an
   OR path `1h ≥5%, z ≥6, vol ≥8×` it fires 09-24 14:09 at $80.18 (in-sample: designed on
   QNT) but NEW/day 9.7 → 35.0 (Aug), 7.7 → 34.7 (09-01..19); `research/wake_path.py`.
   Next: a targeted version (calm-for-days + both 15m and 1h unusual) on a fresh window.
5. **Post-signal behaviour (bug-fixed v2, `research/post_signal.py`):** the typical pump fades
   (24h median −11.2% Aug / −7.2% Sep 1–19; 40% higher after 1h) but the mean is NOT reliably
   negative (a few explode) — v1's significant average fade was a log-averaging artifact. Pump REVERSAL alerts preceded further falls in 2 of 3 windows
   (not significant in 09-01..19). Any trade rule (e.g. short after REVERSAL) must be declared
   first, costed (fees + slippage + funding), tested vs a matched null, then run forward —
   and prior repo work (pump→dump thread) found that pattern not monetizable on free data.
6. **Signal metrics (done, `docs/SIGNAL_FEATURES.md`):** OI, positioning, premium, breadth, detection
   latency etc. — none replicates; weak consistent tendencies only (broad-market moves, retail
   crowding, OI build-up and extended moves fade more; liquid coins fade less). Use them as context
   lines in alerts, log them, and test the squeeze-vs-flush interaction on forward data.
7. **Persistence study (`docs/PERSISTENCE_STUDY.md`) — reviewed, bugs fixed, rerun (v3):** nothing
   validates under the declared protocol; heavy post-alert volume is the best candidate (holdout
   −7.9% [−12.6, −2.9] heavy vs quiet third; discovery means opposite sign). Market breadth passes
   the 4h study's declared test narrowly (`docs/SIGNAL_FEATURES.md`). Both need forward data. Next: a +4 h follow-up message carrying this bucket's historical base rate; the April–
   July holdout is now CONSUMED — any new idea needs forward data.
8. Roadmap: OI lines (Bybit tickers / HL ctx), Upbit/Bithumb announcement feeds (listing
   notices precede the market), MEXC/Gate/KuCoin feeds (early small-cap listings),
   sub-second lead-lag from exchange timestamps, periodic re-discovery instead of the
   daily restart.

## Watch out for

- **Same ticker ≠ same asset.** OKX lists equity perps/spot under crypto tickers
  (`QNT-USDT-SWAP` ≈ $48); Binance/Bitget `METAUSDT` is a stock perp; Upbit/Bithumb
  `KRW-META` is Metadium; `KRW-EDGE` ≠ global EDGE. Handled by `instCategory` filtering +
  runtime price clusters (`store.refresh_consistency`, names like `META@upbit`).
- **Geo-blocks from this box:** api/fapi.binance.com → 451, api.bybit.com → 403. Use
  data-api / data-stream .binance.vision, fstream `/market/` path (kline / miniTicker /
  markPrice are silent on bare `/stream`), Bybit WS with one topic per subscribe.
- **Consumed windows:** 2026-09-20..09-27 (QNT) was seen in development; 09-01..09-19 has
  been read three times (v2 clean, v3/v4 second/third looks). Neither is out-of-sample
  for logic v4 — only forward data is.
- **Memory budgets (user directive 2026-09-28: never break the limit, keep a buffer).**
  - Replay: run at the DEFAULT slot cap (4 GiB address space), never raise
    `RESEARCH_MEM_GIB`; launch with `MALLOC_ARENA_MAX=2`; `research/replay.py` keeps the
    (n × T) panel in disk memmaps, logs `[mem]` per stage and aborts above 2.0 GiB RSS.
  - Service: `MemoryHigh=1G` / `MemoryMax=1536M`, one Telegram warning above 768 MB; float32
    rings, 25% growth, chunked baselines, WS queues ≤ 1,024 messages.
  - Check `free -g` before starting a second heavy job; the box runs tape collectors.
- **Cold start:** Bitget backfills ~500 minutes, so the first evaluation can alert on a move
  already in progress; 24h volumes are normalized by observed minutes until a day accrues.
- **Replay vs live:** replay evaluates minute closes → detection times are an upper bound.
- `pgrep -f`/`pkill -f` with a pattern that appears in your own command line matches your
  own shell; anchor with `^python3` or bracket a letter (`[m]ovewatch`).

## Resume

```bash
cd ~/ctaNew/movewatch && cat STATUS.yml HANDOFF.md
python3 -m pytest -q tests/                 # 18 tests
python3 -m movewatch status                 # live health + last alerts
systemctl --user status movewatch           # memory current / high / max
```
