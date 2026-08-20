# Polymarket collector audit — 2026-08-20

Snapshot time: approximately 14:30 UTC. Scope: the running processes,
`collect_pm.py`, `collect_pm_prices.py`, current logs and all locally collected
`data/pm_5min` files. This is a read-only audit; no collector was restarted or
changed.

> **Operational update, 15:12 UTC:** commits `50dd889` and `2deb8e8` repair
> the findings below and are deployed. The historical tape and the original
> evidence remain degraded; the new CLOB era is **REPAIRED / MONITORING**, not
> yet accepted as clean. See the deployment addendum at the end.

## Verdict

| lane | verdict | consequence |
|---|---|---|
| market discovery and resolution | **CLEAN CURRENTLY** | all seven market grids are complete; 1,963 final resolutions, no give-ups |
| S30/S60 Route-A tape | **FILTERED USE / V2 MONITORING** | sufficient row flow; v2 shortens and records outages, but only admissible windows may enter and missingness still needs a selection audit |
| CLOB book/trade tape | **REPAIRED / NOT YET ACCEPTED** | v2 passed a 19m23s high-load smoke test; the pre-v2 tape remains degraded and the required full busy day has not elapsed |
| compute and storage | **SUFFICIENT** | no capacity blocker for the ten-day collection horizon |

The collectors should keep running: stopping would destroy unreplayable data.
But “running” must not be read as “clean”. Route A may accumulate through its
frozen filters. CLOB-derived inference remains blocked.

## Live process and capacity check

- exactly one `collect_pm_prices.py` process, up about 22.9 hours;
- exactly one `collect_pm.py` process, up about 3.5 hours from its 10:55 restart;
- both logs and active files were fresh at the audit time;
- CLOB collector: about 5% CPU and 416 MB RSS; price collector: 0.2% CPU and
  44 MB RSS;
- `data/pm_5min` occupied 6.1 GB, with 1.2 TB filesystem space free;
- the observed rate projects to roughly 60–70 GB for ten more days, well inside
  capacity.

## Route-A input lane

### What is clean

- The discovered five-minute grid has **zero missing windows** within each
  coin's collection span: BTC/ETH/SOL/XRP have 290/290; BNB/DOGE/HYPE have
  277/277.
- Resolution capture has 1,963 final slugs, zero give-ups and no current final
  duplicates. Eight known non-final rows from the first-hour bug remain and are
  correctly filtered.
- Both TWAP topics have 24 consecutive hourly files: 23 immutable gzip files
  plus exactly one active hour. There are no duplicate hours and no gzip errors.
- Parsing found zero malformed S30/S60 payloads. Every knowledge lag is
  non-negative. S60 duplicate event timestamps are small (28–29 per symbol),
  and `route_a_v1` deterministically keeps earliest knowledge time in both
  streams.
- Settlement direction has already reproduced 1,560/1,560 admissible outcomes
  in the frozen Route-A run.

### What is not clean

The price WebSocket has no replay. Its log contains 28 global 30-second silent
reconnects and one connection error over 22.9 hours. The S60 BTC series (the
other symbols move synchronously) contains 92 gaps over five seconds:

- 51 gaps of 6–10 seconds;
- one gap of 21–30 seconds;
- 40 gaps over 30 seconds, maximum 44 seconds.

Since the 10:55 CLOB restart, S60 has seven gaps over 30 seconds and six gaps of
6–10 seconds. Under the stricter joint check—both streams, fresh target
boundaries and all six predictor horizons—224/273 resolved windows (82.1%) are
fully admissible. Across the tape the UTC-day rates are 566/766 (73.9%) for the
partial first day and 987/1,197 (82.5%) for the partial second day.

This does **not** contaminate retained values: the frozen Route-A builder reads
at knowledge time and excludes sparse or stale rows. It does create a selection
question. Global feed outages are not proven missing-at-random with respect to
volatility/regime, so a ten-day gate must also report accepted-versus-excluded
activity and volatility. More rows alone cannot answer that.

The tape volume is sufficient. At the recent retention rate a full day should
still contribute roughly 230 usable rows per symbol/horizon, far above the
30-row cell minimum after ten test days. The missing quantity is independent
UTC days: with 2026-08-19 training-only, the first possible ten-test-day run is
after the 2026-08-29 UTC day has closed, rotated and resolved.

## CLOB book/trade lane

The 10:55 process loaded the hot-loop repair committed as `15d8fc2`; its file
mtime predates process start. The repair is therefore genuinely under test, not
merely present in the worktree.

It did not close the failure:

- 20 WebSocket retries after restart, all BTC;
- 13 confirmed `1013 slow consumer` closes;
- 11 unique BTC windows with a confirmed slow close;
- only 41 recent completed BTC raw windows were available, so at least 26.8%
  have a known load-linked loss;
- four additional ping timeouts and three no-close failures.

The loss remains concentrated in the busiest symbol and is therefore MNAR for
flow, trade and queue inference. Reconnection can restore a current book
snapshot; it cannot replay missed trades, event order or queue transitions.
Windows carrying these gaps must not be used for markout, fill, flow or
model-vs-book conclusions.

The raw archive itself is structurally intact: 1,991 slugs across 2,084 files,
75 multi-shard slugs (maximum three shards), as expected from prior restarts.
Consumers still need message-identity deduplication and explicit shard
concatenation. Silence in a raw file cannot currently prove completeness because
the channel exposes no collector-side sequence ledger.

## Implementation findings

1. **The slow-drop repair is incomplete.** `collect_pm.py:270` still executes an
   `asyncio.wait` allocation for every message, and `:279` awaits each threaded
   disk write from the receive loop. These are candidate bottlenecks, not proven
   root causes, but extended live evidence rejects the zero-drop claim.
2. **Health output hides the failure.** `slow_drops` is incremented at
   `collect_pm.py:287-288`, but the heartbeat at `:350-353` does not print it or
   expose per-symbol rates/gaps.
3. **The price watchdog is global and slow.** `collect_pm_prices.py:104-107`
   waits 30 seconds for total socket silence. One TWAP topic can stop while
   another continues without triggering it, and a global outage necessarily
   loses at least the watchdog interval because there is no replay.
4. **No durable gap ledger exists.** Logs are not a stable, joined input keyed
   to every affected Route-A window and topic. The analysis has to reconstruct
   causes after the fact.
5. **Malformed price messages are silently dropped.** The parser returns at
   `collect_pm_prices.py:59-62` without a counter. Current files contain none,
   but the live health signal is not fail-visible.
6. **Hourly compression is not atomic.** The gzip target is written directly at
   `collect_pm_prices.py:85-88`; a crash can leave a partial `.gz`. Current files
   are readable and there are no logged gzip errors.
7. **A two-hour resolution give-up is permanent.** `collect_pm.py:333-337`
   writes `gave_up`, while `is_final()` at `:112` treats it as resolved on every
   restart. No current market has hit this path, but a delayed resolution would
   never be backfilled.

## Acceptance boundary

### Route A

Continue collection and rerun `route_a_v1` unchanged after the tenth OOS test
day. A pricing verdict still requires all 84 frozen gates. In addition, report
coverage and accepted/excluded activity by UTC day; do not assume the 18% recent
joint exclusion is benign.

### CLOB-derived work

Keep the current tape but label it degraded. Before using it:

1. land a second receive-path repair;
2. persist connection/gap events keyed by slug, token and knowledge time;
3. expose per-symbol message rates, last-message ages and drop counters;
4. demonstrate at least a full busy-day with zero `1013` losses, or pre-register
   a cause-aware exclusion rule and show enough complete day clusters remain;
5. never pool pre-repair and post-repair observations unpaired.

Until those conditions hold, the data is sufficient for filtered Route-A
measurement accumulation, but not sufficient for clean microstructure or
market-making inference.

## Remediation deployment addendum — 15:12 UTC

Two reviewed commits are pushed on `mm-research` and live:

- `50dd889` (`clob_v2`, `prices_v2`) removes per-message timer allocation and
  disk waits from the CLOB receive hot loop, decouples ordered writes through a
  bounded queue, atomically publishes gzip shards, persists disconnect/gap causes, and
  makes resolution give-up retryable. The price lane now has independent 8 s
  global/topic watchdogs, durable per-topic gap open/close records, preserved
  malformed payloads, stale-hour recovery and atomic gzip rotation.
- `2deb8e8` (`clob_v2_1`) adds exact active/unseen socket counts, per-coin
  message rates and oldest active-socket ages. Every market task is strongly
  referenced and drained on shutdown.

Live evidence is deliberately split by collector version:

- `clob_v2` ran from `14:50:21` to `15:09:44` UTC (**19m23s**). Its last
  heartbeat reported **908,843** messages including **552,166 BTC**, and it
  crossed four discovery generations with `retries=0`, `slow=0`,
  `writer_wait=0`, queue high-water
  `1`. Shutdown drained the two receivers still blocked after the other
  sockets exited cooperatively.
- The first `clob_v2_1` minute processed **84,328** messages, including
  **53,835 BTC**, with the same zero retry/drop/backpressure counters. It
  reported 21 overlapping sockets (three per coin), zero unseen sockets, BTC
  rate **896.6 msg/s** and oldest BTC receive age **1.61 s**. Less-active coins
  correctly remained fail-visible at 17–24 s rather than being hidden by a
  global freshness number.
- `prices_v2` caught one genuine global silence. It reconnected and closed the
  S60/S30 gaps at **11.50/11.57 s**, versus waiting at least 30 s under the old
  watchdog. Both topics recovered to sub-second freshness; the event is joined
  durably in `prices/collector_gaps.jsonl` rather than inferred from prose logs.
- Exactly one process per collector is live. Twenty-eight gzip shards closed
  in the final ten-minute check all pass integrity validation; no temporary
  archive remains.

This is enough to call the implementation repair successful, not enough to set
`clob_capture_clean=true`. Preserve the preregistered acceptance boundary: one
full busy day with zero `1013`, or a cause-aware exclusion rule with enough
complete independent days. Never pool pre-v2 and post-v2 observations without
an explicit repair-era field derived from the versioned start/stop ledger.
