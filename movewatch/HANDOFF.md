# HANDOFF — P-2026-004 movewatch

Updated: 2026-10-02 14:45Z (session 1, continued). Single-seat project: this file and
`STATUS.yml` are the state; update both after every completed step.

## State in one paragraph

**Live, and a forward test is running.** `movewatch.service` (user systemd, enabled) streams
~4,500–5,900 markets on 12 venues and sends Telegram alerts with the **strict preset** (logic v4,
`cal_v4`: floor ×3.0, z ×1.25, v ×1.0, liquidity ≥ $5M). The replay rate on Binance USDM was
9.7 NEW / 38.4 messages per day in August; live counts will differ because it covers more venues
and assets. Since **15:54Z the service runs the frozen commit `1536a86`**, which adds:
- a breadth line on NEW alerts;
- a +4 h volume check message for pumps;
- stock/ETF/commodity exclusion;
- silent +4/+24/+72 h follow-up records for the forward test declared in
  `docs/FORWARD_TEST.md`.

Memory is measured: service 308 MB RSS, 356 MB peak vs `MemoryHigh=1G` / `MemoryMax=1536M`,
swap 0.

## Done (session 1)

- **QNT case study**: `docs/QNT_2026-09_CASE_STUDY.md`. $66.87 → $374.50 (5.6×), peak 09-27
  22:56 UTC; catalyst 09-24; perps were 88% of volume; squeeze signature.
- **Live system**: 12 feeds, 1 s engine, shared `signals.py` / `alerts.py` (the replay runs
  the same code), Telegram + JSONL, burst digest, listing + Upbit-flag watcher, funding line,
  state snapshot, `status` command.
- **Calibration**: `docs/CALIBRATION.md`. The protocol was declared first; noise-vs-earliness
  frontier in 2d, QNT per preset in 2e.
- **Research, reviewed and bug-fixed** (commit `bf94a99`):
  - Typical pumps fade (24 h median −11.2% Aug / −7.2% Sep), but the mean is not reliably
    negative.
  - Signals predict size, not direction.
  - Market breadth passes its declared 4 h test narrowly.
  - Nothing validates for 72 h persistence. Heavy post-alert volume is the best candidate
    (holdout heavy − quiet −7.9% [−12.6, −2.9]; discovery means have the opposite sign).
- **Forward test** (commit `1536a86`, the freeze): declaration, recorder (`movewatch/followup.py`
  and engine), evaluator `research/forward_eval.py`.
  - Market adjustment uses price snapshots of the liquid universe: the replays' `mkt_median`,
    and restarts don't affect it.
  - Every start logs a code fingerprint. The evaluator admits only runs of the frozen code and
    gives every excluded row a status per hypothesis.
- **TradFi exclusion** (`movewatch/tradfi.py`): 228 stock/ETF/commodity clusters excluded,
  matched by price against labelled Bitget RWA and OKX stock/commodity instruments, plus Bitget
  tokenized "r" stocks. `bitget_spot` minimum volume is now $1M.
- **Restart path fixed**: `store.load` decompressed the full npz arrays once per instrument,
  ~20 min at 100% CPU, and detection is dead during load. The 10:37 start had no state file, so
  it never showed. Every daily restart would have hit it. Now 0.24 s; the regression test fails
  on the old code.
- **Tests**: 46 pass. Every falsifier was mutation-checked (a deliberately broken instrument
  must fail its test).

## Live check 2026-10-02 14:16Z (4.0 days into the forward test; counts only, no outcome read)

- Healthy: 4 admitted runs (daily restarts, all fingerprint 65c7c892), restore ~0.2 s each,
  0 ERROR lines, max RSS 320–363 MB/day, swap 104 MB (cap 256M), 12/12 feeds up.
- 33 NEW (17 pumps / 16 dumps) = 8.4/day vs replay 7.4–9.7. NEW+ESC+REV 51 (09-30), 47 (10-01)
  vs replay 31–38; Upbit flags add 11–26/day.
- Test counts: H2 30/300, H1 2/200, B1 2/200, complete weeks 0/5. 30/30 4h checks on time
  (max 5 s), 0 gap minutes, 0 price-source changes; 1 row `72H_NO_PRICE` (AIOT).
- **BUG — Binance USDM discovery** (`venues/binance.py` `BinanceUsdm.discover`): it reads
  `!miniTicker@arr` for only 6 s, and that stream carries only symbols that ticked that second.
  Coverage per restart: 574 / 470 / 435 / 347 instruments. Perps that miss the window are
  invisible on Binance for the whole day. AIOT (still trading: $2.2M on 09-30, $1.0M on 10-01)
  dropped at the 09-29 restart, so its 24h/72h records have no price. Fixing it changes code
  and ends the test: **user decision**.
- **Same token, two tickers**: `龙虾` (binance_usdm, bitget_fut) and `LONGXIA` (bybit_linear)
  are one token. 6 NEW between them, one exact duplicate pair (09-30 20:36). This duplicates
  rows (rule 2). An evaluator-side de-dup can be declared before any outcome is read; the
  alert side needs a code change.
- Minor: OKX `ZEC-USD` candle subscribe fails (24× per day); ESCALATE text shows
  "0/6 venues agree" (cosmetic).

## Forward test v2 (2026-10-02 14:40Z, commit 55ae98d)

User approved fixing the 10-02 findings, which ended v1 unread. Fixed: Binance USDM universe now
comes from markPrice (786 instruments vs 347; settled contracts dropped), config aliases (one
token, several tickers), 71 Binance spot tokenized stocks excluded, venue line on NEW only, BTC
leg + `FU_ENTRY` recorded, fingerprint excludes watcher/notify. Declaration
`docs/FORWARD_TEST_v2.md`; admitted 1/1. Next: BTC-hedge study (`docs/BTC_HEDGE_STUDY.md`,
declared) on the consumed replay windows; any candidate goes in as v2 amendment 1.

## Next (ranked)

1. **Let the forward test run; read counts only.** `python3 -m research.forward_eval` prints
   counts and statuses. It is allowed any time. Statistics come once, with `--final`, at the
   minimum: H1/B1 ≥ 200 complete pumps, H2 ≥ 300, ≥ 5 complete ISO weeks (earliest
   2026-11-09; likely late Nov–Dec at the replay rate).
2. **7-day live review (due 2026-10-05, STATUS M8)**: messages by kind/day, misses vs ex-post
   big moves, false alarms, memory trend, follow-up statuses (share `VOL_GAP` / `LATE`).
   **Read-only.** Any code or config change ends the forward test, so weigh that before
   touching thresholds.
3. **Preset choice is the user's**, but switching now ends the forward test. Strict caught
   QNT at $177.55 and the blow-off at $308.82; the ≤80 preset caught $147.67 and $237.04, for
   ~2× the messages.
4. **Wake path for calm names (M9, measured, NOT deployed)**: catches QNT 09-24 at $80.18
   (in-sample) but NEW/day 9.7 → 35.0. It needs a fresh window, and deploying it ends the test.
5. Roadmap, after the test or as a declared new test:
   - OI lines;
   - Upbit/Bithumb announcement feeds;
   - MEXC/Gate/KuCoin feeds;
   - sub-second lead-lag;
   - periodic re-discovery instead of the daily restart.

## Watch out for

- **Frozen code.** The forward test ends at the first service start whose code/config
  fingerprint differs from `1536a86`'s (`movewatch/**/*.py` + `config/movewatch.yaml`). Docs,
  research and tests can change freely.
- **Never run a second engine on `data/`.** Smoke tests and rehearsals use `--data-dir <scratch>`
  (or a scratch `data_dir`), otherwise they write into the live state and follow-up files.
  Dry runs are ignored by the evaluator, but they would still corrupt state.
- **Restart rehearsal recipe** (it found the restore bug): production config, dry run, a
  scratch data dir seeded with a copy of `data/state/store.npz`, ~150 s live. Then run
  synthetic alerts through `_schedule_followup` / `_process_followups`. The script from
  2026-09-28 lived in the session scratchpad; recreate it from this description.
- **SIGTERM during startup** only sets the stop flag; the process stops once discovery ends.
  systemd's `TimeoutStopSec=30` bounds it.
- **Same ticker ≠ same asset.** OKX lists equity instruments under crypto tickers (QNT ≈ $48
  stock); `METAUSDT` perps are the Meta stock; Upbit `KRW-META` is Metadium; `ON@bybit` is ON
  Semiconductor. Handled by price clusters plus the TradFi price match.
- **Geo-blocks from this box**: api/fapi.binance.com → 451, api.bybit.com → 403. Use
  `*.binance.vision` and fstream `/market/`, and Bybit WS with one topic per subscribe.
- **Consumed windows**: 09-20..27 (QNT) and 09-01..19 (three looks), plus the Apr–Jul holdout
  (persistence). Only forward data is out-of-sample.
- **Memory** (user directive: never break the limit, keep a buffer):
  - Research: `.worktrees/research_slot.sh` at the default 4 GiB cap, `MALLOC_ARENA_MAX=2`, and
    the 2 GiB self-guard.
  - Service: systemd limits above.
  - Check `free -g` before a second heavy job.
- **Cold start**: Bitget backfills ~500 min. A complete volume window needs ≥ 7 h of history,
  which the restore provides after restarts.
- `pgrep -f` / `pkill -f` patterns match your own shell. Anchor with `^python3` or bracket a
  letter.

## Resume

```bash
cd ~/ctaNew/movewatch && cat STATUS.yml HANDOFF.md
python3 -m pytest -q tests/                 # 46 tests
python3 -m movewatch status                 # live health + last alerts
python3 -m research.forward_eval            # forward-test counts (reading rule: counts only)
systemctl --user status movewatch           # memory current / high / max
```
