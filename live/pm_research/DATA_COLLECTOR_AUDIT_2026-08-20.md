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

## v3 remediation — 16:31 UTC, and a correction to the v2 addendum

**The v2 addendum's "enough to call the implementation repair successful" was
premature and is withdrawn.** It was written at 15:12 citing a 19m23s clean
smoke test. The collector's own ledger recorded a `SLOW_CONSUMER_1013` on BTC at
**15:16:13, 5.8 minutes after the `clob_v2_1` deployment**, and `clob_v2_1`
finished its 80-minute run at **`retries=5 slow=5`, all BTC — one slow-consumer
drop every ~16 minutes** over 4,822,835 messages. Withholding
`clob_capture_clean=true` was correct; the narrative around it was not.

### What v2 did fix, verified rather than assumed

- **The write path is provably not the bottleneck**: `writer_wait=0`, `q_hi=1`
  across 4.8M messages. The queue-decoupling repair works — it was simply not
  the binding constraint.
- The gap ledger pairs `disconnect` ↔ `gap_closed` with cause and slug keys.
- The price lane caught a real global silence and closed it at **11.5 s**
  against ≥30 s under the old watchdog.
- Atomic gzip and numbered shards survived a real restart: all 14 in-flight
  market tasks flushed and archived at the v2 stop, and v2_1 resumed the same
  windows into `.1.gz`.

### The defect v2 missed, measured

`gzip_atomic` ran **synchronously on the event loop**. Measured on real BTC
shards: **1,818–1,915 ms** to compress ~180 MB at level 6. That is a full
event-loop stall, every five minutes per coin, during which **no socket is
drained** and the server's send buffer to us fills. At BTC's 500–1,000 msg/s
that is 900–1,900 messages the venue must hold for us.

A second, latent one: `to_thread` (writers) and four `run_in_executor(None, …)`
HTTP calls (`urlopen`, 15–25 s timeouts) shared the **default 20-worker pool**.
If HTTP saturates it, writer writes queue behind them, `write_q` fills, and
`flush_buf` falls through to `await write_q.put(...)` **inside the receive
loop** — producing the same 1013 by a different path. It had not fired
(`writer_wait=0`), but offloading gzip onto that same pool would have made it
more likely, not less.

### `clob_v3` changes

1. **gzip off the loop** onto a dedicated 2-worker disk pool; per-shard cost
   recorded as `gzip_ms_max`.
2. **Disk and HTTP executors split** (`DISK_WORKERS=2`, `HTTP_WORKERS=8`), so a
   stalled `urlopen` can never delay a raw-shard write.
3. **Event-loop lag probe.** A 1013 has two candidate causes with *opposite*
   fixes — the loop stalled (offload work) or the socket rate genuinely exceeded
   capacity (shard connections). Nothing previously distinguished them. The
   probe measures the overshoot of a fixed 100 ms sleep, reports `lag_ms_max`
   and `lag_stalls` per heartbeat, writes a `loop_stall` ledger event at
   ≥250 ms, and **stamps the measured lag into every `disconnect` record** so
   the next 1013 is attributable rather than argued about.
4. **`active_markets_drained` → `markets_force_cancelled`**, plus
   `markets_completed_cooperatively`. The old name read as "markets drained"
   (14 at the v2 stop) when it counted only those needing a forced cancel (2); a
   later auditor would reasonably have inferred 12 lost shards.
5. **Every gap now has an explicit end.** An outage running to window end
   previously left a `disconnect` with no matching `gap_closed`, which a
   consumer cannot distinguish from a lost close record. `gap_open_at_exit`
   closes it — the Route-A selection audit needs this to be unambiguous.
6. **Narrow chunk-loss window closed.** `flush_buf` cleared `buf` before the
   chunk was safely queued, so a cancellation landing on the backpressure
   `await` lost it. The chunk now survives in `pending` for the finally to
   re-flush.

Selftest is 12 checks including a **control**: the same gzip run inline must
stall the loop by ≥100 ms and ≥20× the off-loop figure, or the off-loop test
proves nothing. Measured 211 ms on-loop versus 0.5 ms off-loop, **393×**.

### Measured confirmation that the loop no longer stalls

The decisive pairing is gzip work and loop lag **in the same interval**:

| heartbeat | msgs | retries / slow | `gzip_ms_max` | `lag_ms_max` |
|---|---:|---:|---:|---:|
| 16:32:26 | 48,551 | 0 / 0 | 1 | 204 (startup) |
| 16:33:26 | 108,499 | 0 / 0 | 1 | 2 |
| 16:34:26 | 200,200 | 0 / 0 | 1 | 2 |
| 16:35:26 | 242,491 | 0 / 0 | 1 | 2 |
| 16:36:26 | 313,512 | 0 / 0 | 1 | 6 |
| **16:37:26** | **387,229** | **0 / 0** | **845** | **2** |

At 16:36:30 a BTC shard was published; the next heartbeat records **845 ms of
compression work against a 2 ms worst-case loop stall**. Under `clob_v2` that
same 845 ms was 845 ms of full event-loop block. BTC also touched **953 msg/s**
at 16:34:26 with `lag_ms_max=2` and no disconnect — `clob_v2_1` took its
15:16:13 drop at 999.6 msg/s.

The mechanism is therefore proven fixed. That is a different claim from the tape
being clean.

### Acceptance — unchanged, and not yet met

The pre-registered boundary stands: one full busy day with zero `1013`, or a
cause-aware exclusion rule with enough complete independent days. `clob_v3`
started at 16:31:26 UTC. The honest comparison is against the `clob_v2_1`
baseline of **one slow-drop per ~16 minutes**; a few clean minutes prove
nothing, which is exactly the error the v2 addendum made. Do not set
`clob_capture_clean=true` on anything less than the pre-registered evidence, and
never pool v2 and v3 observations without the version field the ledger records.

## CORRECTION 17:39:56 UTC — the v3 root-cause claim is FALSIFIED

**A `SLOW_CONSUMER_1013` occurred under `clob_v3`**, on
`btc-updown-5m-1787247300`, 68 minutes after deployment. The v3 commit
(`07bede1`) and the section above assert "ROOT CAUSE, MEASURED NOT
HYPOTHESISED". That assertion is withdrawn.

What was actually measured, and still stands:
- `gzip_atomic` blocked the event loop for 1,818–1,915 ms per BTC shard. Real,
  timed, reproduced.
- After the fix it does not: 1,916 ms of compression work against 2 ms of loop
  lag, with a control proving the test would catch a regression.

What was **inferred and wrongly stated as measured**: that this stall *caused*
the 1013s. Eliminating the stall did not eliminate the 1013.

### What the instrumentation rules out

At the moment of the drop the disconnect record carries
`lag_ms_max_interval = 1.8 ms`. This is exactly the discrimination the probe was
built for, and it points away from the branch v3 fixed:

| candidate | evidence | verdict |
|---|---|---|
| event-loop stall | `lag_ms_max_interval` 1.8 ms; heartbeat max 13 ms | **ruled out** |
| gzip in flight | nearest window closes 17:36:30 / 17:41:30, none at 17:39:56 | **ruled out** |
| write-queue backpressure | `writer_wait=0`, `q_hi=1` since start | **ruled out** |
| memory / queue ballooning | RSS 260 MB, stable (v2 ran 416 MB) | **ruled out** |
| peak message rate | dropped at **580 msg/s**; sustained **984.7 msg/s** cleanly at 16:38 | **not a simple rate ceiling** |
| OS descheduling the process | would appear in the sleep-overshoot probe; it does not | **ruled out** |

### What remains open

All four v3 disconnects are BTC, and three of the four were on a single slug in a
four-minute burst before this one. Loop lag, write path, memory and raw rate are
all clean at the time of each. The cause is therefore *not* anything currently
instrumented. Remaining candidates, none yet tested:

1. per-connection reader throughput inside the `websockets` frame parser;
2. venue-side per-connection or per-IP limits, independent of our behaviour;
3. an interaction with concurrent socket count (3 BTC sockets were active);
4. `DISK_WORKERS=2` — v3 narrowed writers from the 20-worker default pool to two
   shared with gzip. `writer_wait=0` argues against it but does not exclude a
   slow writer that never fills a 32-deep queue.

### Consequence for acceptance

The pre-registered boundary — one full busy UTC day with zero
`SLOW_CONSUMER_1013` — **has failed at 68 minutes**. `clob_capture_clean` stays
`false`, and the honest read of v3 so far is: total disconnects 3.7/h → 3.7/h
(five in 80 min v2_1; four in 68 min v3), with the *composition* changed. The
gzip stall was a genuine defect worth removing on its own merits. It was not the
answer.

The next step is a discriminating measurement, not another fix. Changing
`DISK_WORKERS` or `ping_timeout` now would create a fourth era boundary while
the cause is still unidentified — which is how the v2 addendum went wrong.

## RESOLVED 17:46:41 UTC — the 1013 is VENUE-SIDE, not ours

`clob_v3_1`'s discriminator answered on its first 1013, two minutes after
deployment:

```
cause               SLOW_CONSUMER_1013
ws_queue_depth_max  133        <- out of a 65,536-frame pause threshold (0.2%)
ws_ever_paused      False      <- the library NEVER stopped reading the transport
lag_ms_max_interval 1.8 ms
gap                 1.40 s
```

`websockets` pauses reading from the socket once its inbound backlog passes the
high-water mark, and a paused transport is what fills a server's send buffer.
**It never paused.** Our backlog peaked at 133 frames out of 65,536 while the
venue was telling us its send buffer was full.

### Every client-side cause is now excluded by measurement

| candidate | measurement | verdict |
|---|---|---|
| event-loop stall | `lag_ms_max_interval` 1.8 ms | ruled out |
| gzip on the loop | 1,916 ms of work vs 2 ms lag; none in flight at the drop | ruled out |
| write-queue backpressure | `writer_wait=0`, `q_hi=1` throughout | ruled out |
| memory / queue growth | RSS 260 MB stable (v2 ran 416 MB) | ruled out |
| **our consumption rate** | **`ws_ever_paused=False`, depth 133/65,536** | **ruled out** |
| **network throughput** | **11.7 Mbps sustained; one BTC socket is 0.24 MB/s** | **ruled out** |

The `1013` label is the venue's, and it does not describe our behaviour. This
also explains, retrospectively, why two successive client-side repairs failed to
remove these: neither the write path nor the gzip stall was ever the cause. Both
were real defects worth fixing on their own merits, and neither was the answer.

### What this changes

**The acceptance boundary as written is probably unachievable.** "One full busy
UTC day with zero `SLOW_CONSUMER_1013`" tests something we do not control. The
pre-registered *alternative* — "a cause-aware exclusion rule with enough
complete independent days" — is now the operative path. That branch was written
into the boundary before any of this was known, and it is what makes this
finding actionable rather than a dead end.

**The exclusion rule already has its input.** Every disconnect is recorded with
cause, slug, coin, knowledge time and an explicit end, so per-window loss is
computable now:

| window | lost | of 390 s |
|---|---|---|
| `btc-updown-5m-1787247000` | 21.46 s | 5.5 % |
| `btc-updown-5m-1787247900` | 1.40 s | 0.4 % |
| four earlier BTC windows | ~1.3 s each | ~0.3 % |

30.8 s total across 8 windows, and **every disconnect in every era has been
BTC** — never the other six coins. The loss is concentrated in the single
busiest symbol, which is the same MNAR shape as the original incident: it is not
random with respect to activity, so gap-touched BTC windows must be excluded
from queue, flow and fill inference rather than averaged in.

### Recommended posture

1. Stop trying to fix the 1013 client-side. It is measured as not ours.
2. Adopt the cause-aware exclusion rule for CLOB-derived work, keyed on the gap
   ledger. Report accepted-versus-excluded activity by UTC day.
3. Keep `clob_capture_clean: false` — the tape still has holes, and the reason
   it has holes is now known rather than assumed.
4. Leave `DISK_WORKERS` and `ping_timeout` alone. There is no longer a
   client-side hypothesis to test with them.
