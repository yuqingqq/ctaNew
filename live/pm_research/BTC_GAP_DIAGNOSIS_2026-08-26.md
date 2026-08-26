# BTC-only CLOB gap diagnosis — 2026-08-26 ~18:55Z (read-only; nothing touched)

## (a) Mechanism — with evidence chain

**Call: a remote per-connection throughput bottleneck on the Polymarket WS edge
(or the path to it). The venue cannot sustain the btc market channel's byte
rate to us; the client is exonerated by its own instrumentation on every single
event. Confidence: HIGH that the bottleneck is remote (not this process, not
this host); MEDIUM on venue-infra vs network-path attribution (indistinguishable
from one vantage point without active probes).**

Evidence chain (ledger = `data/pm_5min/collector_gaps.jsonl`, 2,702 rows; code =
`live/pm_research/collect_pm.py`, clob_v3_1):

1. **Architecture rules out cross-coin coupling.** One websocket **per market**
   (coin × 5-min window) — btc and eth never share a connection. The receive
   loop does no JSON parsing, no disk I/O (batched 2,000-line flushes to a
   bounded queue, writes on a dedicated 2-thread disk pool, gzip off-loop on the
   same pool). No code change since 08-20 17:44 (git: `350364e` last); the
   degradation began 08-24/25 with the collector running continuously from
   08-21 01:45 to the 08-26 03:56 crash — **no restart, no deploy at onset**.
2. **Client-side slow-consumer is refuted by the v3_1 discriminator.** Across
   all 1,106 btc disconnects 08-24..26: `ws_ever_paused=False` in **every**
   event (the websockets Assembler never once paused reading the transport);
   `ws_queue_depth_max` p50 ≈ 80–100, max 346 vs the 65,536 cap; event-loop lag
   at disconnect ≤ 79 ms (one 2.4 s stall in 3 days, at the 04:38 boot); box
   load now 0.69, collector at 2.1% CPU. The 08-26 04:38 reboot did **not**
   clear the condition — not process state.
3. **Three close causes, one condition.** btc disconnects by cause,
   silence-before-close (`recv_ns − last_message_recv_ns`):
   - `PING_TIMEOUT` (349 of 665 on 08-25; 205 of 366 on 08-26): silence p50
     **exactly 10.0 s** (= ping_timeout), p10 1.7–2.5 s. The pipe went
     **completely silent** — no data AND no pong for the full window. Error:
     `sent 1011 keepalive ping timeout; no close frame received`.
   - `SLOW_CONSUMER_1013` (112 / 57): silence ≈ 0.0 s — killed mid-flow with
     the server's own words: **"1013 slow consumer: send buffer full"** — the
     *server's* send buffer to us, while our side was provably draining (point 2).
   - `NO_CLOSE_FRAME` (204 / 106): silence ≈ 0.0 s — TCP died unclean mid-flow.
   All three are one condition seen at three stages: the far end can't push the
   btc stream — it stalls ≥10 s (we kill), declares its buffer full (it kills),
   or the TCP session dies (path kills).
4. **Time-of-day is flat, not busy-hours.** 08-25 btc disconnects by UTC hour:
   28–45 **every** hour, 00:00–23:00 uniform. Not correlated with US/EU market
   hours, not with our cron/pipeline activity. Argues venue-side capacity/config
   change over client load.
5. **Onset shape.** 08-23: 37 btc disconnects → 08-24: 76 (btc raw volume that
   day was the period's **peak**, 11.24 GB gz, +16% vs 08-23 — consistent with
   first crossing a threshold) → 08-25: **665** while volume fell **back to
   baseline** (9.54 GB ≈ 08-23's 9.67). Nine-fold worse at unchanged load =
   the *threshold moved* (remote change), not our traffic.
6. **Host NIC exonerated.** The Binance HF collector (same box, same NIC, 16
   symbols sub-second) logged ~32 reconnect-ish lines total across 08-24..26 vs
   665/day on PM-btc alone. General egress/ingress is fine; only the Polymarket
   WS lane suffers. (Path-specific degradation to their Cloudflare edge remains
   possible — see "not verified".)

Numbers reconcile with DA: 366 btc disconnects over 19.0 h on 08-26 =
**19.3 gaps/hr** (DA: 19.26); eth 08-24/25/26 = 5/2/**0** disconnects (DA:
1.4%/0.7%/0.0% windows). PING_TIMEOUT 08-25 = 349 vs prior daily max 25 = 14x
(DA's measure). Damage mode on 08-25: 665 gaps, total 3,645 s (61 min) of btc
blackout in ~5.5 s median slices sprinkled over 80.2% of windows.

## (b) Why btc and not eth

Pure rate asymmetry — btc is the only feed above the choke threshold.
Measured on the same 18:25Z window today: btc 115,182 msgs / 81.9 MB raw per
300 s = **384 msg/s, ~2.2 Mbit/s**; eth 21,594 / 15.3 MB = **72 msg/s,
~0.4 Mbit/s** (5.3x). Daily gz volume: btc 9.5–11.2 GB ≈ 3.4x eth ≈ 7–20x the
rest. Current heartbeat rates: btc 344.6 msg/s, eth 143.2, sol 70.1, others
≤46. Every coin at ≤143 msg/s shows 0–3 disconnects/day through the entire
episode; btc at ~350–460 shows hundreds. The choke sits somewhere between
eth's ~51 KB/s and btc's ~273 KB/s sustained per-connection.

## (c) Minimal candidate fix and blast radius

**Recommended: btc-only token-shard — two websockets per btc market, one per
asset_id, feeding the SAME buffer/file.** The subscribe payload already takes a
list (`{"assets_ids": toks}`); send `[tok]` on each of two connections inside
`_market()` when `coin == "btc"` (~25 lines, one file:
`live/pm_research/collect_pm.py`). Why this addresses the mechanism:

- `price_change` messages bundle **both** tokens' mirrored changes (verified on
  raw tape: `price_changes` list carries UP and DOWN entries, each with its own
  `best_bid/best_ask`); `book` and `last_trade_price` are per-asset. A
  single-asset subscription should carry ~**half the bytes** per connection
  (~136 KB/s — between eth's proven-safe 51 and btc's failing 273).
- **Flow diversity is the guaranteed win**: two independent TCP flows; a 10 s
  stall on one leaves the other token's stream recording, and since UP ≈ 1−DOWN
  with per-change `best_bid/best_ask`, *half the tape still carries the full
  quote picture*. Full-market blackout then needs a simultaneous double stall
  (≈ p² for independent stalls).
- Raw schema unchanged (`recv_ns\traw`, one file per slug; two tasks appending
  to one shared `buf` is atomic under asyncio; consumers already order by
  `recv_ns` and already concat numbered shards). Streams are disjoint by asset
  → **no duplicate messages** (a naive "second redundant full subscription"
  WOULD duplicate trades — do not do that one).

Blast radius: `collect_pm.py` only; other 6 coins untouched; gap-ledger schema
unchanged (optionally add a `conn` tag — additive). **Requires one collector
restart** (`systemctl --user restart pm-collector-clob.service`): SIGTERM
triggers the built-in drain (flush + atomic gzip, ~3–10 s; TimeoutStopSec=180),
startup resumes in-flight windows from `markets.jsonl` and numbered shards
prevent clobbering — expected coverage gap **~10–30 s across all coins, once**
(same machinery has done this cleanly at the 08-20/21 restarts and today's
04:38 boot). Best moment: a few seconds after a :x0/:x5 window boundary.

Calibrated expectations: the byte-halving may or may not fully clear the
(unknown, remote) threshold; the redundancy term helps regardless. Note the
gap **metric** counts per slug — a one-sided stall still logs a disconnect, so
DA's %-windows number improves fully only for double-stalls until the verifier
learns half-gap semantics (later, not tonight). The raw tape improves
immediately.

Cheap adjunct (optional, same restart): drop `ping_interval/ping_timeout` from
10/10 to 5/5 for btc — cuts the dominant stall-gap from ~11.3 s p50 (10 s dead
pipe + 1.3 s reconnect) to ~6.3 s. Reduces gap seconds, not gap counts.

**Anti-fixes** (explicitly): raising `max_queue`, buffer sizes, or writer
capacity — client buffers are measured nowhere near limits; that lever was
already maxed on 08-20 and the discriminator proves it isn't the constraint.
Doing nothing is also a poor option: the condition has held ≥2 days at ~flat
intensity (still ~17–28 gaps/hr this evening), so tonight's btc day would be
quarantine-grade again.

## The +1 missing-window drip (08-26, 05:52Z)

Crash debris, not the gap mechanism. The box died ~03:56Z (btc 03:55 window
truncated at 21 MB ≈ 1 min of data; 04:00 window file created 0-byte; 03:50
window complete, 73.8 MB, but never gzipped) and booted 04:38. **All seven
coins are missing exactly the same 6 windows (04:05–04:30)**; as of 18:40Z
every coin has 220 present / 6 missing — no ongoing file drip. The verifier
(`da_forward_day_verify.py:137`, `short = expect − counts`) sees btc one
shorter than eth wherever a 0-byte or never-gzipped shard fails its count —
that is the "+1". The three damaged btc shards sit uncompressed in
`data/pm_5min/raw/20260826/` if salvage is wanted (03:50 is fully salvageable).

## (d) Not verified

- **Single-asset subscription semantics at the venue** (assumed standard;
  no test connection made — read-only mandate). Must be smoke-tested at deploy:
  confirm both single-token connections deliver, and that per-connection bytes
  actually drop ~2x.
- **permessage-deflate negotiation** on these connections (websockets 15.0.1
  offers it by default; server acceptance unknown → true on-wire rate unknown).
- **Venue-infra vs network-path attribution**: no traceroute/ping/status-page
  checks were run (no active probes, no web access). Binance-lane health clears
  the host NIC, not the specific route to ws-subscriptions-clob.polymarket.com.
- **Why PING_TIMEOUT silence pins at exactly 10.0 s** (p90 ≤ 10.1): under
  random stall phase vs a 10 s ping cycle one expects 10–20 s spread. The
  pinning suggests interaction between our ping and the stall onset (or a
  keepalive-geometry detail of websockets 15.0.1 I did not confirm). Does not
  change the diagnosis (the pipe is provably undelivering for 10 s with our
  loop healthy), but a packet capture would settle it.
- **DA's exact "10 vs 9" arithmetic** at 05:52Z (their expected-count formula
  was not re-executed for that timestamp; the crash-debris account above is the
  consistent explanation).
- Journal evidence for 08-24/25 is gone (volatile journald; retention starts
  ~13:42Z today) — the collector's own ledger/log supplied the timeline instead.
- `collect_pm_prices.py` lane: healthy per lane-health; not examined further.
