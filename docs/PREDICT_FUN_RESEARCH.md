# Predict.fun strategy transfer and data collection

As of 2026-10-21. **The Polymarket strategy is a plausible research candidate
for Predict.fun BTC five-minute Up/Down markets. Profitability is untested.**
Collect market data first, establish what fills can actually be reconstructed,
then compare a venue-specific candidate against simple baselines on untouched
days. The Predict-only collector is now installed, enabled and running as
`predict-fun-collector.service` on this host, with `--no-binance` so research
reuses the existing external tapes. It started at 2026-10-21T15:27:01Z.
The testing token is configured in an owner-readable environment file outside
the repository. Collection remains exploratory; profitability is untested.

## Strategy being transferred

The current implementation is **D1**, rather than the older dense-value LGBM
candidate in the September strategy overview. Inspected Rust repo HEAD:
`1b61e8c`. Its contract is [D1_SPEC.md](../pmmmrust/docs/D1_SPEC.md), implemented
in [research.rs](../pmmmrust/crates/strategy/src/research.rs),
[btc.rs](../pmmmrust/crates/signals/src/btc.rs) and
[tick.rs](../pmmmrust/crates/signals/src/tick.rs).

D1 quotes around the depth-weighted microprice, preserves queue position when
an inside quote merely changes label, withholds quotes behind the touch,
limits inventory exposure, stops quoting at 240 seconds, avoids touch levels
with a gap behind them, and withholds the side exposed to an adverse Binance
move. It inherits the shorter Binance return and probability-change guards.
The probability guard uses a Polymarket Chainlink reference and timing formula.
Those inputs are part of the strategy and cannot be replaced by ticker names.

The current Rust README and D1 spec describe shadow operation and an
uncalibrated fill model. The latest consumed-window comparison is
[REEVALUATION_RESULTS_20261021.md](../pmmmrust/docs/REEVALUATION_RESULTS_20261021.md).
It does not establish live profitability or validation on Predict.fun. The
nested Python checkout is at `952f4a1`; its handoff is older than the D1 port.
The Rust spec pins the later research sources separately.

## Venue compatibility

| Component | Predict.fun finding | Transfer consequence |
| --- | --- | --- |
| Instrument | The public site lists BTC five-minute Up/Down markets, including the [inspected contract](https://predict.fun/market/btc-updown-5m-1792594200). | Start with BTC 5m, the closest existing instrument. Treat other assets and horizons as separate candidates. |
| Settlement | That contract names Chainlink's Binance BTC/USDT top-of-book DataLink stream, endpoint price comparison, the closing five-minute candle for the final price, and a 50–50 payout on equality. | Archive every contract's actual rule and final outcome. Rebuild the reference and terminal probability; do not import Polymarket's S60 reference, USD basis or `tau = 240 - t + 20`. Binance futures mid is a signal, not the settlement label. |
| Book | The [book guide](https://dev.predict.fun/doc-685654) expresses levels in YES price space. NO bid is `1 - YES ask`; NO ask is `1 - YES bid`, at venue precision. | Microprice and inventory concepts transfer. These are complementary views of one aggregated book; counting both as independent liquidity doubles depth. Confirm Up maps to YES/indexSet 1 for each market. |
| Data | [Public WS topics](https://dev.predict.fun/subscription-topics-1915507m0) provide aggregated book snapshots, trading status and market status. The documented book payload includes pending settlement levels and the last settled order, but not a complete trade stream or per-order queue. | Capture raw snapshots. A shrinking level is not a demonstrated fill. Neither lastOrderSettled nor snapshot differences reconstruct every execution. Pending settlement is not final inventory. |
| Trades | [GET /v1/orders/matches](https://dev.predict.fun/get-order-match-events-25663812e0) provides paginated matches, maker/taker details, transaction hash and executedAt, newest first. | Archive overlapping pages and investigate timestamp meaning, precision, indexing delay, late arrivals and stable identities. REST receipt time is when the collector learned the row. Do not use later learned rows in a live signal. |
| Orders | The [create-order schema](https://dev.predict.fun/create-an-order-32534694e0) exposes isPostOnly. | This supports a maker-only design in principle. Actual cancel effectiveness, partial fills, priority, minimum size, tick rules and replacement semantics still need venue measurements. A LIMIT order alone does not establish maker status. |
| Fees | The [fee guide](https://predict.fun/learn/how-prediction-market-fees-work) says makers pay no trading fee. A [September announcement](https://predict.fun/news/30-in-maker-rebates-just-landed) advertises 25% of taker fees for eligible Up/Down fills. | Report gross, costs and realized rebates separately. A 25% rebate share is not 25% of notional or a fixed cents/share payment. The expanded promotion is provisional for a month from September 22; recheck eligibility and expiry before modeling it as income. |
| Access and location | [Developer FAQ](https://dev.predict.fun/) requires a mainnet API key, lists a default REST budget of 240 requests/minute, and places primary infrastructure in ap-northeast-1. | Request a market-data key in the developer console. Measure an eligible Tokyo host as well as any existing collector location before adopting latency assumptions. REST polling cannot replace the WS book stream. |

The API's schema example still shows `priceFeedProvider: PYTH`, whereas the
inspected BTC contract names Chainlink. Testnet examples also identify
Chainlink. **Neither the generic example nor testnet certifies every mainnet
market's oracle.** Read the actual category/market rule and provider ID.

The `shareThreshold` and `spreadThreshold` fields describe reward eligibility;
do not use them as proof of a minimum order size or price tick. Fetch venue
constraints separately. Keep the existing Binance VIP-0 model unchanged;
Predict.fun needs its own explicit accounting model.

## What is ready locally

[predict_fun_recorder.py](../data_collectors/predict_fun_recorder.py) records:

- Rolling BTC 5m categories by default, including all ended windows still
  within retention after a restart or discovery outage, plus current/upcoming windows, with strict
  market, time-window and Up/Down token mapping checks. ETH and 15m are optional.
- Full Predict WS frames for orderbook, market status, trading status and
  metadata-change notifications. It echoes application heartbeats, checks
  subscription acknowledgements, reconnects and records connection boundaries.
- Category metadata, initial book snapshots, reconnect resnapshots and
  paginated REST matches. Ended markets remain under observation for 30 minutes
  by default; a later unresolved outcome requires a follow-up read.
- Binance BTCUSDT spot and USDM futures bookTicker and raw trades, with both
  feeds on the same collector's clock. Spot is an observable reference proxy;
  the licensed Chainlink report stream is **not collected** by this version.
- Receive wall-clock and monotonic nanoseconds, source payloads, connection
  IDs and receipt sequence numbers. HTTP rows include request timing. The run
  manifest pins recorder source bytes and settings; tokens are not recorded.

Files are exclusive to each run under
`data/predict_fun/runs/<run_id>/<source>/<UTC_hour>.jsonl.gz`, plus manifest,
events, a source snapshot and final summary. Compression runs on a separate writer thread.
Disk failure or queue overflow aborts the run. A process heartbeat certifies
neither market-data freshness nor continuity. Flushes do not guarantee fsync
durability; an abrupt host failure can truncate a gzip tail, which later QC
must mark as excluded. A missing summary also requires a failed-run status.

This initial recorder preserves evidence; it does not certify day quality,
infer fills, or run the strategy. Binance spot bookTicker can lack a source
event timestamp: never invent one to compute source-to-receive latency.
Wall-clock steps require explicit QC despite the companion monotonic clock.

The shared REST limiter defaults to 120 requests/minute. Discovery and match
polling share that budget, so the configured five-second poll pause is not a
per-market freshness guarantee. Expanding assets/intervals or retaining many
ended markets increases the actual cycle time. Raw polling times reveal it.
New runs start match traversal at run start minus 60 seconds; this does not
backfill a complete earlier window. Every completed traversal retains a
60-second overlap. The page cap is 40; exhausted traversals remain explicitly
incomplete. Live/upcoming windows take priority and ended markets switch to
60-second polls after their first minute, while metadata remains under
observation for the configured retention period.
`MATCH_RANGE_TRAVERSED` reports pagination traversal only; delayed indexing,
pagination mutation and trades arriving beyond the overlap remain possible.

## Configure and run

### Reuse the existing collectors on this host

Checked at 2026-10-21T15:17:23Z: these three user services are running from
`/home/yuqing/ctaNew` and their current output files have recent receive stamps.
This verifies current writes, not complete-day continuity.

| Existing service | Data | Output |
| --- | --- | --- |
| `pm-collector-clob.service` | Polymarket books, trades, metadata and resolutions | `data/pm_5min/raw/`, `markets.jsonl`, `resolutions.jsonl` |
| `pm-collector-prices.service` | Polymarket RTDS Binance spot-price mirror and Chainlink price topics | `data/pm_5min/prices/crypto_prices/` and the TWAP topic directories |
| `collect-hf.service` | Binance USDM futures bookTicker, depth20 and trades | `data/mm_hf/raw/{bookTicker,depth20,trade}/BTCUSDT/` |

The current spot feed is a price mirror through Polymarket, not a direct spot
order-book/trade tape. The last 30 inspected BTC source timestamps have a
1,000 ms median positive step. Reuse it for price-level research and the HF
futures tape for the existing fast guards; it does not supply a sub-second
spot order book or Predict's licensed oracle reports. These limits do not
require duplicating Binance collectors for the initial study.

**Launch Predict with `--no-binance` on this host.** It records only Predict
data. Later research joins the existing external tapes by their receipt
timestamps, respecting feed gaps and collector eras. This flag does not
automatically merge the files or certify the joined dataset. Leave the
existing services and their files untouched.

### Why the spot mirror has one-second prices

A simultaneous, unthrottled 25-second diagnostic completed at
2026-10-21T15:29:25.120103Z observed:

| Feed | Observed BTC messages | Source timing |
| --- | ---: | --- |
| Polymarket RTDS `crypto_prices` | 24 | All 23 positive source-timestamp steps were exactly 1,000 ms |
| Direct Binance spot bookTicker | 6,065 | Multiple messages per second; no source event timestamp in this payload |
| Direct Binance spot trades | 2,025 | Sub-second transaction timestamp differences observed |

The raw feed probe, source code and timing receipt are stored under
`data/predict_fun/price_cadence_probe/runs/20261021T152900Z_a6e1a621e3/`.
These are message counts in a short engineering sample, not independent market
outcomes or guaranteed update rates. The temporary diagnostic sockets closed
after the declared interval; no persistent Binance collector was added.

The existing `collect_pm_prices.py` reads each WebSocket message immediately
and appends its original timestamp and value. A synthetic control delivered
three updates within the same second and all three were preserved; a malformed
frame went to its named raw status instead of a price row. The five-second
flush interval affects disk visibility, not subscription or sampling cadence.
The independent live subscription still received one-second source steps, so
this behavior originates upstream along the RTDS relay path. The probe does
not identify the relay's internal sampling algorithm or input channel.

One-second prices are usable for display, reference-level diagnostics and
one-second volatility bars. They cannot resolve a 250/300 ms spot-price move
or reliably measure sub-second lead/lag. Keep the existing D1 fast guards on
their recorded Binance futures bookTicker input. If a Predict-specific design
requires a fast spot guard aligned with its Binance-top-of-book oracle,
collect the direct spot feed as an explicit new research input and test its
basis, timing and incremental value; do not substitute the slow mirror or
silently equate futures with spot. Predict's book WS remains sub-second.

### Launch only Predict data collection

Install the small recorder dependency set from the repository root:

```bash
python3 -m pip install -r data_collectors/predict_fun_requirements.txt
python3 -m pytest -q tests/test_predict_fun_recorder.py
```

Configure `PREDICT_FUN_API_KEY` through the host's secret environment, then
run a bounded REST probe and a bounded mainnet smoke:

```bash
python3 -m data_collectors.predict_fun_recorder --no-binance --probe
python3 -m data_collectors.predict_fun_recorder --no-binance --duration-s 900
```

After inspecting those runs, continuous BTC collection uses:

```bash
python3 -m data_collectors.predict_fun_recorder --coins btc --intervals 5 --no-binance
```

An existing process supervisor can run that command from this repository with
a synced clock and a writable data disk. Graceful SIGTERM flushes queued rows.
Choose host/disk limits from the smoke's measured traffic before adding ETH
with `--coins btc,eth`. No wallet private key, JWT, balance or trading approval
is needed for this collector.

For this host, [predict-fun-collector.service](../data_collectors/systemd/predict-fun-collector.service)
is a prepared user-service template. It uses the same Python environment as
the existing collectors (aiohttp 3.10.11 verified), the existing
`collectors.slice`, a separate Predict output root, and `--no-binance`.
It is installed and enabled, with Predict frames and successful REST responses
observed after startup. The initial run directory was
`data/predict_fun/runs/20261021T152701Z_9dad09ee0c`; the reviewed deployment
restarted at 2026-10-21T15:46:20Z into
`data/predict_fun/runs/20261021T154620Z_e45bf1bb0d` with schema
`predict_research_raw_v2`. At 2026-10-21T15:29:01Z the initial
archive contained 2,512 Predict WS frames and 126 REST response
records, with no Binance output directories and no recorded WS failure, REST
cycle failure or incomplete-match events in that startup sample. This is a
liveness observation, not a complete-day quality verdict.

To configure it, create `~/.config/predict-fun/collector.env`, readable only
by the owner, containing `PREDICT_FUN_API_KEY=<token>`. Keep the real token
out of shell command history. Then install and launch from the repository root:

```bash
install -m 644 data_collectors/systemd/predict-fun-collector.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now predict-fun-collector.service
systemctl --user status predict-fun-collector.service --no-pager
journalctl --user -u predict-fun-collector.service -n 30 --no-pager
```

Stop only this new collector with
`systemctl --user stop predict-fun-collector.service`. Every restart makes a
separate run directory; its manifest and connection boundaries must be read
when assembling a day. The service is data collection only, with no wallet
access or order submission.

Inspect a finalized run with the checked engineering reader:

```bash
python3 -m data_collectors.predict_fun_inspect data/predict_fun/runs/<run_id>
```

The reader requires a source snapshot and exact sequence/count reconciliation,
and refuses corrupt, truncated or inconsistent runs. Its receipt reports
observed feed counts and failure statuses, never strategy admission.

The keyless testnet REST schema exercise is:

```bash
python3 -m data_collectors.predict_fun_recorder \
  --testnet --rest-only --probe --root data/predict_fun/schema_smoke
```

Testnet WS access is unverified and refused by this tool; Binance feeds are
disabled in testnet mode. Testnet runs carry `TESTNET_SCHEMA_ONLY` and must
never enter an economic test.

## Testing sequence

1. **Capture and quality assessment.** Begin with a proposed 24–48-hour
   engineering soak. Measure subscription success, book update cadence,
   content stalls, clock steps, reconnect spans, rollover initialization,
   missing sides, crossed books, REST lag/rate limits, resolution coverage
   and resource use. Declare acceptance thresholds before inspecting any
   economic results. Missing snapshots and unresolved windows get statuses.
   Establish authorized access to the exact Chainlink feed if the probability
   guard needs it; metadata updates learned later cannot be backdated.
2. **Decide whether fills are identifiable.** Reconcile paginated executions
   against snapshots, settlement states and stable trade identities. The
   public feed lacks order-level queue events. If execution timing/coverage
   cannot support the intended latency, stop at signal research or obtain a
   better permitted feed. Public snapshots alone do not support a credible
   market-making profit backtest. Calibrating actual own-order fills would
   require a separately authorized, funded experiment later.
3. **Develop a Predict-specific policy.** Compare joining touch, microprice
   quoting, and a guarded candidate using the same sizes, inventory limits,
   eligibility and accounting. Measure signal value incremental to the
   Predict book mid/microprice and a settlement-aligned spot baseline.
   Test the spot/futures distinction. Refit the r60 risk lookup locally;
   Polymarket's 1.86 limit, two-tick gap threshold, minimum size and last-minute
   cutoff are hypotheses, not transferable calibrations. Keep the candidate
   family small and record every variant tried.
4. **Replay only with supported execution assumptions.** Use each event's
   real knowledge time, a venue-specific fill model and separately varied
   market-data, post, cancel and reporting delays. Simulate partial fills,
   stale quotes, failed settlement and continuous inventory/capital across
   windows. A touch does not imply a fill; a book deletion does not imply a
   cancellation or a trade. Use declared queue bounds if exact priority is
   unavailable, and label the result conditional. Report terminal inventory,
   paired spread capture, adverse selection, cost, rebate and net separately.
5. **Freeze and validate once.** A reasonable initial allocation is seven
   complete development days followed by eight later untouched complete UTC
   days; this is a proposed design, not an existing freeze. Commit the full
   pipeline and candidate before opening validation outcomes. Eight day-level
   paired sign flips allow all 256 assignments, exceeding the repo's minimum
   200; five complete days remain the minimum for any day-cluster interval.
   Withholding controls must match action count, side and hour. Evaluate one
   cancellable generation per action on the neutral reference path, not a
   population selected by fills. Record exclusions, n, as-of, all variants and
   family multiplicity. New choices consume the development data; corrections
   supersede receipts in-band.

Passing those stages would justify an execution-calibration proposal. It
would not by itself demonstrate a funded bot can attain the simulated latency
or fills. Venue execution work belongs in a separate execution project, in
accordance with this repository's research-only instructions.

## Evidence and current checks

The public BTC contract, official docs, fee announcement and HTTP responses
are archived locally with retrieval times and SHA-256 hashes in
`data/predict_fun/recon/20261021T145502Z/manifest.json` (gitignored). Mainnet
`GET /v1/markets?first=1` returned 401 without credentials at
2026-10-21T14:55:03Z. This is an access check, with no order submission.
The live testnet REST probe found three mapped BTC windows, retrieved books
and traversed match pages; that exercises schemas only.

### Authenticated mainnet smoke on October 21

The authenticated REST probe succeeded. The first mainnet smoke recorded
Predict and Binance futures, but Binance spot returned HTTP 451 and busy
markets exhausted the initial ten-page traversal cap. That run and its source
snapshot remain under `data/predict_fun/mainnet_smoke/` as diagnostic provenance.

The collector now uses Binance's documented
[public market-data endpoint](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md)
for spot, prioritizes live market polls and bounds initial backfill to the
run-start overlap. The corrected bounded run covers
**2026-10-21T15:09:48.477524Z to 15:12:48.484252Z**, with the following observed
population as of **2026-10-21T15:13:05.806806Z**:

| Observation | Count |
| --- | ---: |
| Raw records reconciled to the finalized writer summary | 421,755 |
| Mapped BTC windows | 4 |
| Predict WS frames, including status/ack/heartbeat frames | 3,809 |
| Successful WS request acknowledgements | 16 of 16 |
| Binance spot frames | 100,848 |
| Binance futures frames | 316,819 |
| REST responses, all HTTP 200 | 196 |
| Recorded WS failures, REST cycle failures or incomplete match traversals | 0 |

This engineering smoke crosses a five-minute rollover and records a newly
discovered upcoming market. It does not cover a complete trading day or
certify feed continuity. Exact counts, timing, code identity and event statuses
are in
`data/predict_fun/mainnet_smoke_v2/runs/20261021T150948Z_e5cbc888f1/engineering_inspection_v1.json`.
The saved source bytes identify the recorder used for that smoke. Both bounded runs stopped;
the testing token was passed through hidden input into transient process
environments, without saving a credential file or placing an order.

**Timing limitation:** all 15,382 archived REST match rows in the corrected run
have whole-second `executedAt` timestamps, with zero unreadable timestamps.
These are overlapping-page rows, not independent fills. They contain 2,347
distinct market/transaction pairs; a transaction pair is not asserted to be a
unique action. The observed timestamps cannot establish within-second match
order, and their relationship to the matching-engine clock is still unknown.
The present public data therefore cannot validate an exact sub-second FIFO
fill replay. Snapshot-based signal research remains possible; execution tests
need defensible timing/queue bounds or a separately authorized calibration.

The local protocol suite exercises HTTP GET-only access, raw frame persistence,
heartbeat echo, disconnect recording, pagination refusal, outcome/window
identity refusal, missing-key refusal, disk-failure detection, Binance endpoint
success/refusal, external-feed mode and inspector corruption refusal. **92 tests pass after the review loop below.** Long-run
resource behavior, actual queue semantics, licensed oracle access and strategy
economics remain unmeasured.

### Collector review and corrected deployment

The 2026-10-21 review found three defects with failing synthetic reproductions:

- Match pagination stopped when its oldest row equalled the overlap cutoff,
  potentially skipping the remaining rows at that whole-second timestamp.
  It now continues until a strictly older timestamp or the final page.
- A failed initial REST book snapshot did not remain queued for retry. Initial
  and reconnect snapshots now stay pending until an HTTP-successful read.
- A category transport/schema failure could abort discovery of other windows.
  Category and per-market match failures are now isolated and recorded with
  identity; authentication failures still stop the process.

New `health.json` telemetry distinguishes the latest WS frame from each
market's latest book frame. Heartbeats cannot refresh a book's receipt age.
Expired/one-sided books remain explicit observations rather than inferred
feed failures. The journal prints queue depth and connection state every
30 seconds. `content_liveness` remains `NOT_CERTIFIED`: a heartbeat or quiet
book cannot prove lossless delivery.

All 44 tests pass in the actual service Python environment, including forced
WS disconnect/resubscription, snapshot retry, poisoned-category/match
isolation, timestamp-boundary traversal, malformed-depth refusals and
heartbeat-versus-book freshness controls. The installed systemd unit verifies.

The initial run shut down cleanly at the planned restart: 18,746 enqueued and
persisted records reconcile exactly. Its finalized inspection, as of
2026-10-21T15:42:21.778984Z, checked 17,381 book frames with zero malformed,
locked or crossed books; one-sided observations are retained. All 923 REST
responses were HTTP 200 and all 24 subscription requests were acknowledged.
Rollover metadata and resolved outcomes were observed. Receipt:
`data/predict_fun/runs/20261021T152701Z_9dad09ee0c/engineering_review_20261021.json`.
No claim is made that the synthetic bugs caused a measured historical loss.

The v2 startup observation at 2026-10-21T15:43:09Z contains 2,198 WS frames,
106 HTTP-200 REST responses, 12/12 subscription acknowledgements, queue depth
zero at the latest heartbeat, and no recorded failure/refusal events in the
observed prefix. Its deployed source SHA-256 matches the manifest and snapshot.
The planned transition is append-recorded in
`data/predict_fun/collector_transitions.jsonl`; treat its book interval as an
unobserved restart boundary. The existing PM/Binance services were untouched.
This verifies current operation over a short span, not day admission, trade
completeness or calibrated fill timing.

The v2 code also passed a live rollover at 15:45Z: the new upcoming market was
discovered and subscribed, with book frames observed, queue depth zero and no
snapshot retry left pending. Its finalized intermediate run reconciles 5,974
records, checks 5,541 book frames with zero malformed/locked/crossed frames,
and has no recorded failures. Nested null/list schema cases were then given
explicit refusals so malformed responses cannot escape the request isolation
through an AttributeError; these additional controls bring the suite to 44.

Final reviewed deployment: `20261021T154620Z_e45bf1bb0d`, recorder SHA-256
`801c4d30533704917f7050ae12df2dceb91e76fae811c44e819e368553865218`.
At 2026-10-21T15:47:22Z its observed prefix contained 1,245 Predict WS frames,
64 HTTP-200 REST responses, 12/12 subscription acknowledgements and no recorded
errors. The service was running with zero crash restarts and approximately
30 MiB cgroup memory; the latest heartbeat showed queue depth zero and a fresh
current-market book. The final source bytes match the run's manifest and saved
snapshot. `post_review_observation_20261021.json` in that run records the prefix
and health sample. Both planned restart boundaries are in the transition log;
neither is silently treated as continuously observed book data.

### Second collector review — October 21

This review supersedes the earlier broad claim about nested schema refusal:
the first review's null/list controls did not cover every nested value. New
synthetic controls reproduced numeric category dates escaping as AttributeError,
missing/naive dates being accepted, nested outcome indices escaping as TypeError,
and duplicate tokens/market IDs being accepted. Discovery now requires explicit
offset-bearing times, typed outcome indices and distinct identities. An existing
market ID cannot silently acquire a different window.

Two data-health defects also reproduced: an HTTP-200 malformed snapshot cleared
its retry flag, and an identity-only WS book refreshed health without depth or
a source timestamp. REST snapshot acceptance, WS freshness and offline depth
inspection now share one validator. Invalid raw payloads remain archived;
snapshots stay due and refused books do not refresh health. Legitimate empty
and one-sided books remain valid observations. Inspector receipts also identify
the validator source hash.

WS envelopes now receive explicit type/topic/request-ID validation. A malformed
JSON value records a connection failure and reconnects rather than escaping as
AttributeError and terminating the other readers. Local WS tests force both
malformed-envelope and transport disconnects, verify resubscription and queue
resnapshots. Offline inspection counts retained malformed envelopes. Match
timestamps require an explicit timezone, and malformed cursors cannot advance
the traversal cutoff. All **68 tests pass** in the service environment; the
installed unit matches its template and passes systemd verification.

The previous run finalized cleanly at 2026-10-21T16:03:48Z. As of
2026-10-21T16:03:54.689044Z, its hashed inspection reconciles **23,976 records**,
checks **22,193 book frames** with zero malformed/locked/crossed books, records
**1,271 REST responses, all HTTP 200**, and matches **24/24 subscription
acknowledgements** across six discovered windows. It crosses the 16:00 UTC
hourly file rotation and reports no recorded failures or malformed envelopes.
Receipt: `data/predict_fun/runs/20261021T154620Z_e45bf1bb0d/second_review_finalized_20261021.json`.
This does not establish that the synthetic defects caused a historical loss.

The updated service started at 2026-10-21T16:03:48Z in run
`20261021T160348Z_034130a585`, recorder SHA-256
`e5a3100c53b833835b09ad3d4c6b74ed02ff6f5cc981c14a186aba2d5e168090`.
The planned restart and unobserved book interval are append-recorded in
`collector_transitions.jsonl`. Existing PM/Binance collectors remain separate.

At **2026-10-21T16:05:53.888454Z**, the updated run's inspected prefix contains
2,766 WS frames, 140 REST responses (all HTTP 200), 16/16 acknowledged
subscriptions and no recorded failures/refusals. All 2,728 inspected book
frames pass the shared validator. The 16:05 rollover added the next window;
latest health observes all four mapped markets with no snapshot retries,
connection active and queue depth zero. Systemd reports zero crash restarts
and 31,956,992 bytes of cgroup memory. The same PM CLOB, PM price and Binance
HF service PIDs remain running. Receipt:
`data/predict_fun/runs/20261021T160348Z_034130a585/second_review_live_observation_20261021.json`.
Active gzip tails are unfinalized, and source prefixes were read sequentially;
these counts are not a coherent whole-run completeness check.

Remaining limits: traversal is bounded by `--match-pages` and records
`MATCH_POLL_INCOMPLETE` on exhaustion; repeated exhaustion can leave historical
matches uncollected. Late indexing beyond the overlap is not ruled out.
After a restart discovery starts with previous/current/next windows, so the
prior process's older retained windows may require explicit resolution reads.
A healthy process does not establish lossless books, complete trades, a full
UTC day of usable data, or a calibrated fill model.

### Autonomous review loop — October 21

The user requested a review/fix/verify loop. It started at 16:13:41Z with the
stopping condition declared in advance: two consecutive clean engineering
verification cycles, including a live rollover on the final deployed source.
Its issue ledger, test gates, source hashes and exclusive cycle receipts are
under `data/predict_fun/review_loop/20261021T161341Z/`.

Four reproduced findings were fixed:

| Finding | Corrected behavior | Regression control |
| --- | --- | --- |
| Reaching the match page cap repeatedly fetched the same newest pages. | A bounded batch retains its cursor and original cutoff, then resumes on a later poll. The cutoff advances only after traversal completes. | `test_page_limit_resumes_backlog_instead_of_repeating_head` |
| Restart/discovery outages forgot older windows still within retention. | Each discovery recomputes eligible windows, prioritizing upcoming/current windows; expired windows are not reintroduced. | `test_cold_start_rediscovers_all_retained_windows_first_live`, `test_retained_discovery_never_reintroduces_expired_windows` |
| Malformed retained HTTP-200 responses could crash inspection. | Inspection reports malformed REST envelopes/pages/rows and invalid REST books explicitly, while retaining raw row counts and archive reconciliation. | `test_inspector_preserves_malformed_rest_as_diagnostics`, REST book/match positive and refusal controls |
| An empty match page with a continuation cursor was treated as final. | Pagination follows the cursor and retains loop/budget checks. | `test_empty_match_page_with_cursor_is_not_the_end_of_traversal` |

Additional controls cover transport interruption, refused/looping cursors,
changed ordering, partially malformed pages, time-budget yielding and backlog
retirement. A refused or inconsistent cursor restarts from the head without
advancing the cutoff; transport failure preserves valid continuation state.
Discovery can run between batches. `--match-batch-s` defaults to five seconds
and yields **between** responses; an individual request can still consume its
20-second timeout. The existing REST rate limit remains in force.

The final gate, `cycle_04.json`, records **92 passing tests**, their output,
and exact source hashes. The deployed run is
`20261021T162823Z_0dd45f2d5a`, started at 16:28:23Z, source SHA-256
`a3241af770e44e15836273ba1f6e06ea13e0164162b2f6bd2615a814c192008e`.
Both planned deploy boundaries are in the collector transition log. Previous
runs finalized with 26,239 and 7,960 enqueued/persisted records respectively;
their finalized inspections are retained beside the cycle receipts.

A provisional cycle-2 flag caught a sampled traversal in progress. Subsequent
completion events demonstrated normal progress. The review observer now uses
completion evidence after the health sample to resolve that provisional flag;
the original receipt remains. Cycle 3 passed but was superseded by the later
empty-page fix; it does not count toward final-deployment verification.
The final source's first clean observation is cycle 5 at
16:29:04.648378Z: 916 WS frames, 61 HTTP-200 REST responses, 32/32 acknowledged
requests, and no findings. Active prefixes are sequential observations, not
finalized whole-run populations.

The loop completed at 16:31:24Z. Final-source cycles **5 and 6** are consecutive
clean observations; cycle 6 verifies the **16:30 rollover**, discovery of the
next window and retirement of an expired one. As of
**2026-10-21T16:31:06.273396Z**, its inspected prefix contains **3,582 WS frames**,
**232 REST responses, all HTTP 200**, and **40/40 acknowledged requests**, with
no recorded failures. The latest health sample has eight mapped windows,
queue depth zero, connection active and no snapshot retry pending. An
in-progress match batch in that sample has a later completion event in the
same observation; it is not classified as a stalled backlog. Systemd reports
zero crash restarts and 35,012,608 bytes of cgroup memory. The three existing
PM/Binance service PIDs remain unchanged. `completion.json` and `state.json`
record four closed findings, their regression controls and the final source
identities. The bounded review process finished; the Predict collector remains
enabled and running.

The loop fixes supersede the page-cap livelock and forgotten-retention
limitations described for the preceding build. Cursor state is in-process;
after restart match collection starts from the bounded run-start overlap.
Missed book intervals, trades beyond that overlap, late indexing, outcomes
unresolved after retention, and exact sub-second fill/queue timing still need
separate treatment. No day-quality or strategy-profitability claim follows
from a clean engineering cycle.

### Overnight data audit — October 22

At **2026-10-22T02:09:02Z**, the same reviewed process remains active after
about **9 hours 41 minutes**, with zero crash restarts. Its latest health
sample has queue depth zero, no pending snapshot retry, and a current-market
book receipt age of 0.039 seconds. Existing PM price and Binance futures
collectors are also running and writing recent rows. The audit made no
collector-code changes, raw-data edits or service restarts.

The full data audit fixes its population at local sequence numbers
**0 through 731,238**, as of **2026-10-22T01:57:25.098797Z**. It reads each raw
stream and reconciles exactly **731,239 records**, with no missing/duplicate
local sequences and no malformed payloads. All 27 sealed hourly gzip files
pass footer checks; three active files are bounded by the sequence cutoff.
Local sequence reconciliation cannot establish venue-message completeness.
The saved auditor includes identity, missing-population, truncated-gzip and
book-schema positive/refusal controls.
`auditor_controls.json` additionally records synthetic whole-auditor runs:
a known 11-second gap and locked book are detected, missing records and a
clock reversal are flagged, and duplicate/wrong-run records are refused.

| Observation in the fixed population | Count |
| --- | ---: |
| Predict WS frames | 672,027 |
| Book snapshots checked | 667,879 |
| REST responses, all HTTP 200 | 44,535 |
| Acknowledged subscription/unsubscription requests | 944 / 944 |
| Full BTC 5m windows since 16:30Z with live book observations | 113 / 113 |
| Those complete windows with captured resolutions | 113 / 113 |
| WS disconnects or source-book timestamp regressions observed | 0 |

Two kinds of source-data flags remain relevant to replay:

- **Locked books:** 168 live snapshots have equal best bid and ask, across
  47 episodes in 25 markets. No crossed book was found. The longest interval
  from the first locked receipt to the next normal/one-sided observation is
  1.357 seconds. These are preserved raw venue observations; downstream book
  features/fill simulations must handle them explicitly.
- **Book freshness gaps:** market 3621430 has a 9.536-second interval at
  **2026-10-21T21:52:36.919Z–21:52:46.456Z**; market 3624468 has an
  11.410-second interval at **23:27:48.465Z–23:27:59.875Z**. The alert threshold
  of five seconds was recorded before scanning. During these intervals the
  collector records other WS traffic and respectively 18 and 12 successful
  REST responses. Source-book timestamps advance by approximately the same
  durations. The cause and any missing venue updates remain unproven; keep
  explicit freshness flags rather than interpolating book states through them.

There are 28 recorded batch-time-limit yields and one retirement with an
unfinished overlap sweep. That retired market, **3620847**, had already
completed earlier sweeps. A separately archived, read-only API traversal at
**2026-10-22T02:02:14Z** covers 40 pages and **3,967 settlement/transaction
objects**. Its object set and recorded immutable projections exactly match
the archive: zero probe-only, archive-only or changed objects. This resolves
the warning's observed missing-object concern for that market; those objects
are not asserted to be independent fills, and other markets' trade
completeness is not certified. Event checks after the full-audit cutoff through
the final process check report no new warning events.

All sources, fixed-cutoff metadata, compressed-input hashes, detailed checks
and follow-up raw responses are under
`data/predict_fun/audits/20261022_overnight_v1/`.
`summary.json` indexes the evidence; `replay_quality_flags.json` preserves the
47 locked-book observation runs and two freshness intervals as review statuses.
Raw observations remain intact. Collection spans parts of two UTC days and
does not yet supply a complete UTC day or certify an exact-fill replay.

### Cause investigation — October 22

The follow-up localizes the observations without identifying a proven venue
backend bug. Evidence and source traces are in
`data/predict_fun/audits/20261022_overnight_v1/cause_investigation/cause_findings.json`.
This is the same fixed overnight population through 01:57:25Z; it is not a
claim about every later observation. The investigation leaves the collector
source and running process unchanged.

**Locked books originate in the published snapshot.** Exact decimal parsing
of all 168 raw message prices confirms 168 equal bid/ask pairs and zero locks
introduced by binary floating-point conversion. The checker includes a
counterexample where two unequal decimal prices collapse under ordinary float
parsing. Both sides are present together in a single raw source message.
The longest locked episode contains 23 source book updates over 1.357 seconds;
this is not an unchanged locally cached row being printed repeatedly.

Pending settlement values occur at the locked price in **150/168 snapshots**.
Predict's [topic documentation](https://dev.predict.fun/subscription-topics-1915507m0)
identifies these as settlement-in-flight values, separate from book quantities.
Asynchronous matching/display/settlement state is a plausible explanation,
but concurrency does not prove cause. Eighteen locks have no pending value
at that price, and one of the three detailed longest episodes clears without
a change in `lastOrderSettled`. Provider-side precision, order constraints or
matching behavior remain unverified. Do not subtract pending `value` directly
from book share quantities or assume both locked sides are executable.

**The local process remains responsive during both book pauses.** The server's
heartbeat timestamps have receive-minus-server epoch differences of **75.6 ms**
and **73.3 ms** inside the two gaps. These are uncalibrated clock differences,
not network-latency estimates. Predict [documents heartbeats as current server
epoch timestamps](https://dev.predict.fun/heartbeats-1915508m0). During the same
intervals 18/12 REST requests finish in ranges **190–223 ms / 195–402 ms**,
and the independent spot collector records 9/11 BTC ticks. Source book update
timestamps jump 9.401/11.391 seconds; the next book has a current source time,
rather than a nine-second-old message buffered locally. These observations
argue against a sustained local event-loop, disk or receive-queue stall.
The recorder has no WS resampling; its one-second receive timeout is an idle
wait, not a one-message-per-second limit.

The gaps still have different causal evidence:

- **Market 3621430, 21:52:36–46Z:** scanning all 346 retained match pages finds
  84 API execution objects timestamped inside the pause. All are learned after
  the book resumes. Delayed venue book/cache publication is a supported
  inference; API indexing delay and uncertified matching-clock semantics stop
  this from establishing the exact publisher/matcher failure.
- **Market 3624468, 23:27:48–59Z:** all 464 retained match pages contain zero
  objects timestamped inside the pause. Two head polls during it report the
  same latest `executedAt`, 23:27:44Z. Quiet trading, paused matching and shared
  cached-state delay remain distinguishable hypotheses, not established causes.
  The received-update pause is not itself proof of missing data.

The initial spot cross-check searched only uncompressed files; closed hours
are gzip archives. `independent_spot_during_gaps_v2.json` explicitly supersedes
that incomplete extraction and records the 9/11 observed ticks. No zero from
the first extraction is used as evidence of a spot outage.

There is no historical REST orderbook probe inside either gap, so the records
cannot conclusively separate WS publication from shared cached-book state.
A future paired REST-book/independent-WS capture during a pause, or Predict's
publisher and matcher logs, would resolve that boundary. Existing raw data and
replay review flags stay intact; no interpolation or invented book repair is
applied.

## Cross venue information for market making — October 22

**Both directions are plausible research candidates. Neither direction has a
measured incremental edge yet.** Start by testing whether the other venue
improves adverse-selection protection beyond the destination book and the
existing Binance guards. A foreign venue's price is an additional signal,
not a replacement fair value. This assessment changes no running policy and
does not open or amend D1's existing forward test.

### What each direction could contribute

| Direction | First signal to test | Potential market-making use | Main limitation |
| --- | --- | --- | --- |
| Predict to PM | Recent UP mid changes, displayed depth imbalance, microprice-minus-mid, spread and book age | Withhold or cancel the PM side exposed to an adverse move that is visible in Predict before PM reprices | Predict locks, one-sided states and delayed updates must disable the input. REST executions are not a fast trade stream. |
| PM to Predict | The same quote features; separately assess PM public trade pressure using its actual receipt time | Protect Predict quotes when PM moves first; later investigate a bounded quote skew | PM prices a different settlement event. Predict queue position, cancel effectiveness and execution reporting are uncalibrated. |

**Implementation priority is a quote-only guard study in both directions.**
PM-to-Predict additionally has a path to testing fresh public trade pressure;
the Predict public subscription does not supply an equivalent complete trade
stream. This is a data-capability difference, not evidence that PM leads.
Larger volume alone does not identify the faster or more informative venue.
The documented feeds are [Predict's aggregated snapshots](https://dev.predict.fun/subscription-topics-1915507m0)
and [Polymarket's public market stream](https://docs.polymarket.com/market-data/realtime-data).

D1 is already a microprice strategy with Binance adverse-side protection,
inventory controls and a stop at 240 seconds. The first PM candidate should
retain those rules and add one separately evaluated foreign-book guard.
Changing the anchor, quoting thresholds, size and guard together would obscure
what the new source contributed. D1 remains an unvalidated research candidate;
its historical results do not supply a prior guarantee for either venue.

### The contracts cannot be priced as interchangeable claims

Verified the actual archived Predict category/market payload and the matching
PM market metadata, rather than treating equal slugs as equivalent payoffs:

| Property | PM BTC 5m | Predict BTC 5m |
| --- | --- | --- |
| Reference named by the contract | Chainlink BTC/USD TWAP 60s stream | Chainlink BTC/USDT top-of-book DataLink stream, based on Binance top of book |
| Rule text | Up if the specified TWAP comparison is greater than or equal to the starting reference | Up on a strictly higher closing reference, Down on a lower one; equality pays 50–50 |
| Consequence | Smoothed USD reference | Top-of-book USDT reference and different tie treatment |

The PM wording alone is not a new adjudication of its exact terminal
statistic; preserve its actual feed and resolution evidence. The different
feeds, starting reference levels, smoothing and tie rules are sufficient to
refuse payoff equivalence. In particular, do not interpret a raw 10-cent
cross-venue price difference as 10 cents of mispricing. Differences can become
larger near expiry without creating an exploitable error. Buying UP on one
venue and DOWN on the other is not a guaranteed payout hedge: their outcomes
can disagree and both purchased claims can lose.

Use recent foreign-venue changes conditional on the destination state and
underlying move. Any later level-based residual needs separate settlement
models for both venues and the actual reference values known at decision
time. Predict's licensed Chainlink reports remain uncollected. The existing
one-second spot mirror cannot substitute for a fast spot order book; reuse
the existing sub-second Binance futures tape as the already available fast
baseline and retain its spot/futures basis limitation.

### Direct volume checks

Declared three contracts before requesting their statistics: the first,
middle and last of the prior 113-window engineering cohort. As of
**2026-10-22T09:17:54.914089Z, n=3 matched contracts**, the official API
reported the following lifetime figures:

| UTC window start | Predict reported USD | PM reported USDC |
| --- | ---: | ---: |
| October 21 16:30 | 75,034.29 | Field absent |
| October 21 21:10 | 35,994.07 | Field absent |
| October 22 01:50 | 57,454.56 | 31,160.75 |

These are contract-lifetime reported totals, including pre-open trading,
not independently reconciled within-window notional or venue-wide totals.
USD and USDC are nominally compared; accounting conventions, duplicate
counting and artificial activity have not been audited. Missing PM fields
are unknown, never zero. This small sample supports investigating Predict's
liquidity but supplies no information-leadership conclusion.

Predict statistics came from the documented
[market statistics endpoint](https://dev.predict.fun/get-market-statistics-25326906e0).
The PM `/markets?slug=` reads returned empty arrays; `/events?slug=` returned
the matching markets. All three returned PM condition IDs and token IDs were
reconciled to the local ledger. Do not interpret an empty endpoint response
as an absent contract or zero volume.

Raw responses, request/receipt clocks and the pre-read selection are under
`data/predict_fun/cross_venue/20261022_feasibility_v1/`.
`volume_summary_v2.json` is operative; it supersedes the initial summary's
as-of to include the later PM response times. Numerical volumes are unchanged.
There were three read-only Predict API requests and six PM API requests.

### Book availability preflight

The offline [preflight](../data_collectors/predict_pm_preflight.py) reads the
existing tapes and the audited Predict sequence prefix ending at
**2026-10-22T01:57:25.098797Z**. Before reading the sampled books it declares
every tenth window in the ordered 113-window cohort, plus the two previously
flagged gap windows: **12 regular + 2 stress windows = 14 distinct windows**.
The selection spans October 21 16:30 through October 22 01:40 UTC starts.
It opens no settlement labels or trading outcomes.

As of **2026-10-22T09:21:50.630693Z**, n=14 windows and 4,200 fixed diagnostic
grid points, with zero omitted grid points:

| Region | Grid points | Both books pass the declared availability checks | Other statuses |
| --- | ---: | ---: | ---: |
| First 240 seconds | 3,360 | 2,937 | 423 |
| Final 60 seconds | 840 | 207 | 633 |
| Whole window | 4,200 | 3,144 | 1,056 |

Each grid point uses only the most recent received state at or before that
time. The diagnostic limits are receipt age at most one second and source
age at most five seconds, with a strictly two-sided, unlocked book. Missing,
one-sided, locked, stale and clock-ahead states remain counted statuses.
An invalid latest state never carries forward an earlier valid book.
PM's empty-side BBO sentinels 0/1 are treated as absent sides.

These are engineering presence/freshness counts on a one-second grid,
not orders, independent samples, sub-second lead estimates, or strategy
eligibility. The stress selection is intentional and not representative.
No full UTC day is certified, no confidence interval or null is run, and no
predictive-value conclusion is drawn. Low late-window availability is a
reason to keep the final minute separate; it does not diagnose an outage.
The metadata ledger also contains one unidentifiable `{}` record; it is
counted separately and cannot be assigned to this cohort or its time range.

The operative receipt is
`data/predict_fun/cross_venue/20261022_feasibility_v1/books_v3/receipt.json`.
Its per-window and phase statuses reconcile to the complete declared grid,
and its design, selected metadata, collector identity and input hashes are
recorded. V3 explicitly supersedes V2's handling of empty-side sentinels and
adds phase counts; the failed initial attempt remains as a design artifact.
The instrument has nine passing tests, including future-book refusal, stale
and locked positive controls, receipt regression refusal, reversed outcome
mapping refusal and empty-side sentinel controls. Existing raw files and
collectors were unchanged.

Reproduce into a new output directory:

```bash
python3 -m pytest -q tests/test_predict_pm_preflight.py
python3 -m data_collectors.predict_pm_preflight \
  --audit data/predict_fun/audits/20261022_overnight_v1 \
  --out data/predict_fun/cross_venue/<new-run>/books
```

### Proposed incremental signal and policy test

This is a development design, **not a committed freeze or an armed test**.
It does not change the existing D1 forward race.

1. **Build both directions on receipt time.** Join by asset, exact UTC start,
   duration and outcome mapping, while recording distinct contract/reference
   identities. Use backward as-of joins, actual metadata knowledge times,
   connection/collector eras and named exclusions. A Predict execution row
   becomes usable only when first learned through REST, never retroactively
   at `executedAt`. Receipt-time leadership is specific to the measured host
   and network paths; do not order events using incomparable source clocks.
2. **Test added information before a guard.** Start with two fixed,
   quote-only feature probes, one per destination. The baseline includes its
   own mid/microprice, spread, imbalance, recent returns, time remaining and
   available reference features, plus the existing fast Binance input. Add
   foreign-book changes, imbalance, microprice-minus-mid, spread and age.
   Use a one-second destination markout as the proposed primary signal
   target; 250 ms and three seconds are sensitivity diagnostics, not a way
   to choose the best horizon afterward. No inferred Predict trade flow.
   Score strictly incremental to the full baseline, so a redundant Binance
   reaction or a persistent contract basis is not reported as new alpha.
3. **Keep the terminal region separate.** The first policy study is the
   first 240 seconds to match D1. Last-minute research requires its own
   reference/payoff model and population; do not blend its easier terminal
   forecasts into early-window market-making skill.
4. **Evaluate real policy actions.** For PM, build cancellable generations
   on the fixed D1 reference path without the new guard, then test one
   added guard against D1 alone. For Predict, first establish a venue-specific
   touch/microprice reference and whether execution/queue evidence supports
   replay; until then report only signal/markout diagnostics, not MM P&L.
   Value only exposure surviving command latency, and measure plausible
   placement/cancel/report delays for each destination. The PM port's 50 ms
   replay assumption is not a measured Predict latency.
5. **Price participation and execution.** Compare same-count withholding
   controls matched on side, UTC hour and window phase. Report missed
   favorable fills as well as avoided adverse fills, continuous inventory,
   spread capture, fees and settlement/basis costs. Use current per-market
   fee parameters; rebates remain zero in the base case unless realized and
   verified. Public depth disappearance does not establish a Predict fill.
   The [Predict fee guide](https://predict.fun/learn/how-prediction-market-fees-work)
   and [PM fee documentation](https://docs.polymarket.com/trading/fees) both
   distinguish maker and taker economics; do not import the Binance cost model.
6. **Freeze before opening validation.** A proposed allocation is seven
   complete development days followed by eight later untouched complete
   UTC days. Commit the full builder, candidate rules, data eligibility,
   primary metric, null and both-direction multiplicity before validation.
   Eight paired day units permit all 256 sign assignments for the declared
   policy contrast. Day-cluster inference needs at least five complete days;
   lower counts remain exploratory. Missing/gap/unvalued actions stay in the
   population with statuses. Preserve the existing forward-test boundaries.

The result needed to justify a change is a repeatable improvement beyond the
destination book **and** Binance after executable delay and participation
cost. A raw cross-venue correlation, larger reported volume or a visually
earlier price move is not that result.

## Exact Chainlink reference collection — October 22

The Predict book collector does **not** record the continuous Chainlink
reference stream. It records contract metadata containing reference/feed
identities and later start/end reference values. Those metadata observations
are not a substitute for the live oracle tape or signed reports.

A separate read-only recorder is prepared in
[predict_chainlink_recorder.py](../data_collectors/predict_chainlink_recorder.py).
**It is not running: Chainlink credentials and exact-feed entitlement are not
configured on this host.** No paid subscription was created. The Predict API
token authenticates to Predict, not Chainlink. Existing collectors remain
unchanged, and no Binance collector is added.

### Identity and access checks

The archived Predict contract names Chainlink feed ID:

```text
0x0003ef42b4d20a19409a187c39670a6f5be7ce17fc78c290d25eb45bbf769d37
```

The provider is `CHAINLINK`, symbol `BTCUSDT`. A fresh read of
`btc-updown-5m-1792673100` on October 22 confirmed the same ID. The recorder
derives its subscription from archived category metadata, validates Up/Down
token mapping and the provider/symbol, and records the metadata and source
hashes with each run. It refuses an unsupported schema or a mismatching
envelope/embedded feed ID. Do not replace this ID with a generic BTC/USD,
TWAP, testnet, or newer-schema top-of-book feed.

The official [report WebSocket](https://docs.chain.link/data-streams/reference/data-streams-api/interface-ws)
requires separate Chainlink authentication and a subscription to the requested
feed. The [authentication specification](https://docs.chain.link/data-streams/reference/data-streams-api/authentication)
uses an API key and shared secret to sign the exact request path/query.
On this host an unauthenticated latest-report request returned HTTP 400 for
missing authentication headers; an unauthenticated candlestick streaming
request returned HTTP 401. These are access checks, not evidence the feed
does or does not exist. Public discovery returned an empty BTC/USDT list;
that does not establish entitlement or certify that the metadata-pinned feed
has been decommissioned. Authenticated exact-feed access remains unverified.

The documented Predict public WS topics include books and market/trading
status, but no continuous underlying-oracle price subscription was found in
that topic list. This does not establish that no other permitted relay exists.
The current implementation uses Chainlink's documented report service.

Probe responses, the reference contract, fresh identity check and independent
decoder comparison are saved under `data/predict_fun/oracle_recon/`.

### What the recorder preserves

- Raw text/binary WS frames before JSON decoding, local wall/monotonic
  receive clocks, connection boundaries, local sequence numbers and health
  events. One initial latest-report REST read is explicitly distinguished
  from live WS reports; reconnect gaps are recorded, not interpolated.
- Raw `fullReport` bytes containing the report context and signatures,
  plus v3 `validFromTimestamp`, `observationsTimestamp`, `expiresAt` and exact
  integer price fields. Every report is bound to the requested feed ID.
  Source timestamps retain their published seconds precision.
- An independent run manifest, collector and storage-engine source snapshots,
  copied reference metadata and hourly gzip files under
  `data/predict_chainlink/runs/`. Compression uses the existing reviewed
  bounded writer in a separate thread. Overflow/disk failure fails visibly;
  abrupt host loss can still truncate an unsealed gzip tail.

Decoding is **not cryptographic verification of OCR signatures**. The raw
signatures are retained and decoded rows explicitly say `NOT_PERFORMED`.
Fixed-point integers stay unscaled until decimals for this exact feed are
verified; the recorder does not assume 18 decimals or relabel generic v3
bid/ask fields as raw Binance prices. The actual Predict contract controls
the price interpretation. Successful live access, cadence, report continuity,
timestamp semantics and settlement reconciliation still require measurement.

There are **13 recorder tests plus the existing 92 storage/collector tests,
105 passing**. Controls include an official Chainlink SDK v3 example decoded
independently with `eth-abi==5.2.0`, an HMAC vector checked with OpenSSL,
feed-identity and truncated/malformed ABI refusals, and a local fake-WS
transport that preserves an invalid raw frame while refusing its normalized
report. A denied-access control proves failure and prevents provider error
text from leaking account identity. This is offline engineering validation,
not a successful live subscription. The independent ABI dependency was used
only in a temporary directory, not installed into the collector environment.

### Configure and start when access is available

The prepared unit is
[predict-chainlink-collector.service](../data_collectors/systemd/predict-chainlink-collector.service).
As of 2026-10-22T13:18:44Z it is installed in the user systemd directory,
but is inactive and not enabled. A local environment-file template is
[predict-chainlink.env.example](../data_collectors/systemd/predict-chainlink.env.example).
Configure these variables in an owner-readable file at
`~/.config/predict-fun/chainlink.env`:

```text
CHAINLINK_DATA_STREAMS_API_KEY=<configure locally>
CHAINLINK_DATA_STREAMS_API_SECRET=<configure locally>
```

The account must be entitled to the exact feed above. Supply only the file
location or variable names in chat. Keep actual credentials outside the repo.
Configure the file with mode 0600, load its variables in a local shell, and
first run a bounded probe:

```bash
python3 -m pytest -q tests/test_predict_chainlink_recorder.py
python3 -m data_collectors.predict_chainlink_recorder \
  --reference-category data/predict_fun/oracle_recon/predict_category_reference.json \
  --duration-s 60
```

Inspect the resulting run for live WS reports, exact feed/schema matches,
raw/decoded timestamp agreement and named failures before enabling the unit.
A REST result alone does not prove WS delivery. The unit refuses to restart
repeatedly on missing configuration or denied access. Once live access is
verified, install and enable the separate service; it requires no restart of
the Predict book, PM, or Binance collectors.

Launch was attempted at 2026-10-22T13:18:44Z with the bounded recorder command.
It refused before creating a run, exit code 2, because neither Chainlink
credential is configured. An empty owner-readable template now exists at
`~/.config/predict-fun/chainlink.env` (mode 0600). The launch receipt is
`data/predict_fun/oracle_recon/launch_attempt_20261022T131844Z.json`.
No live oracle reports have been captured; the original Predict book service
remains active with the same PID and zero restarts.

## Predict price relay alternative and deployment — October 22

**Chainlink credentials are required for direct signed reports, but are not
required for Predict's own reference-price relay.** The earlier launch block
applies to the signed-report collector only. Predict's website exposes an
additional price topic absent from the public API topic list. It was found
by inspecting the website's public client assets, then verified live using
the already configured Predict API token. No third-party credentials or
additional account were used.

The website client constructs `chainlinkAssetPriceUpdate/<priceFeedId>` for
its Chainlink price display. For the inspected BTC contract, its GraphQL/page
metadata maps `priceFeedId: "1"`, symbol `BTCUSDT`, provider `CHAINLINK`, and
market ID 3637527. The official Predict category API independently identifies
that same market and maps it to provider feed
`0x0003ef42b4d20a19409a187c39670a6f5be7ce17fc78c290d25eb45bbf769d37`.
The website's numeric feed ID and Chainlink's provider feed ID are distinct
namespaces. The collector pins and archives that checked mapping; it does
not infer identity merely from the number 1.

Primary client evidence:

- [Website topic construction](https://predict.fun/_next/static/chunks/24vt0g3eefvdg.js)
- [Website Chainlink price hook](https://predict.fun/_next/static/chunks/034xaf87r3m6l.js)
- [Inspected market](https://predict.fun/market/btc-updown-5m-1792674900)

Captured assets, hashes, API/reference mapping, initial live probe and later
receipts are under `data/predict_fun/oracle_alternatives/`. The API schema
called `OracleUpdate` was also checked: it describes textual resolution
clarifications, not reference-price updates, and is not used for this feed.

### Running collector

[predict_reference_recorder.py](../data_collectors/predict_reference_recorder.py)
records the relay in a separate service,
[predict-reference-collector.service](../data_collectors/systemd/predict-reference-collector.service).
It was installed, enabled and started at **2026-10-22T13:29:01Z**, initial PID
1356092, run `data/predict_reference/runs/20261022T132901Z_8c043d17b6`.
Its source hash is
`72dac8412819acd080422f1eca6a2ee6a6ffdc14832013c3e1e8b99a918ef6b5`.
It uses the existing Predict environment file. The direct signed-report
service remains inactive; existing Predict books, PM and Binance collectors
were not changed or restarted.

Raw frames are recorded before parsing with wall and monotonic receipt
timestamps, connection IDs and local sequences. Normalized rows preserve
the exact decimal price text, `publishTime`, relay `timestamp`, both feed
identities and receipt clocks. Every row explicitly identifies an
`UNSIGNED_PREDICT_CHAINLINK_RELAY`. Duplicates and source-time regressions
are flagged without deleting raw observations. Subscription acknowledgements
and application heartbeats are handled. Missing valid price messages cause
a reconnect after 45 seconds even when heartbeat frames continue. Health
output includes receipt age, source epoch difference, refusals and queue
depth; it does not certify content completeness.

The first 30-second recorder run caught a reserved-field collision in the
new normalization code: raw frames persisted, but normalized rows did not.
That run remains failed and excluded. The field was renamed before deployment,
and programming `TypeError`s now fail rather than being retried indefinitely.
A local WS control reproduces the full raw-to-normalized path with a bounded
timeout, so this failure would fail the test instead of hanging it.

The corrected bounded run ended successfully. As of
**2026-10-22T13:28:54.639265Z**, its population was **131 records, including
61 normalized price rows and 31 distinct source publish seconds**. There
were 26 consecutive duplicate payloads, zero source timestamp regressions,
zero price refusals and zero WS failures. All 30 positive distinct-source
timestamp steps were one second. Local sequences and every decoded row's
raw price/timestamp/receipt pair were reconciled, and sealed gzip files read
through their footers. This is one short engineering probe, not a cadence
guarantee or complete-day quality result. The initial cached snapshot and
duplicate messages are not independent observations.

There are **13 relay tests plus 92 existing collector/storage tests, 105
passing**. They cover captured live input, exact decimal preservation,
wrong topic/feed, bad source times/prices, metadata mapping refusal, raw-first
capture, duplicate preservation, heartbeat echo and fatal subscription denial.
The persistent unit passed `systemd-analyze --user verify` before enabling.

```bash
systemctl --user status predict-reference-collector.service --no-pager
journalctl --user -u predict-reference-collector.service -n 20 --no-pager
```

### What the alternative does and does not provide

This supplies the reference price **as published by Predict's relay** without
a separate Chainlink subscription. It is appropriate to collect alongside
the two venues' books for reference and timing research. It does not deliver
the signed report, prove identical delivery/completeness to Chainlink's direct
service, or establish exact settlement sampling by itself. Source times have
seconds precision; several received messages per second do not create
sub-second oracle timestamps. Receipt minus source epoch is an uncalibrated
combined clock/publication/network difference, not measured pure latency.
The topic is a website implementation interface and may change.

Direct Binance top-of-book midpoint is another public observable proxy, but
it is not a substitute for recording the venue's published reference. The
existing PM TWAP reference is a different feed. Keep each source separately
identified; do not silently relabel either proxy as Predict's settlement
oracle. Direct signed Chainlink access remains an optional further source
for report-level verification, not a prerequisite for collecting this relay.

## Collection completeness review — October 22

The core tapes for an initial **BTC five-minute cross-venue signal study**
are being recorded: Predict books/status/match pages/category metadata,
Predict's reference relay, PM books/trades/metadata/resolutions, PM S60/S30
reference relays and Binance futures bookTicker/depth/trades. The existing
spot-price mirror is also active. This is a scope inventory and a bounded
engineering check, not full-day continuity certification or MM validation.

At the 2026-10-22T15:34:07Z service snapshot, all five relevant services were
active; both new Predict services retained zero restarts and their original
PIDs. BTC rows were present in the latest spot, S60 and futures output tails.
The host reported NTP synchronized. That confirms neither historical clock
calibration nor every feed's completeness. Available storage was about 557 GB.
No service was changed or restarted during this review.

### Concrete missing fields and remaining limits

| Item | Current collection | Remaining work |
| --- | --- | --- |
| Official settlement reference | Raw REST category data preserves start/end prices, resolution, contract rules and feed ID. | Running collectors do not continuously archive `startPricePublishTime` and `endPricePublishTime`. They are available from the website GraphQL `category(id: <slug>)` → `CryptoUpDownCategory.marketData`. Add periodic capture alongside market/token/feed identity and actual HTTP receipt time. |
| Reference-feed identity over time | Relay ID 1 is pinned to BTCUSDT/CHAINLINK and the provider's full feed ID using an archived website/API comparison. | The running relay does not automatically recheck that mapping every window. The existing category tape enables retrospective provider-ID checks; live mapping-change detection remains missing. |
| Predict trade and execution detail | Overlapping paginated match records, raw receipt clocks, whole-second `executedAt`, book aggregates and pending settlement states. | Precise match-time/indexing semantics, complete late-arrival coverage, per-order queue position and own-order placement/cancel/fill/report measurements are not established. A disappearing book level is not proof of a fill. This limits credible MM P&L replay. |
| Fast underlying spot data | One-second PM Binance spot-price mirror plus separate sub-second Binance **futures** data. | No direct sub-second Binance spot L2/trade tape. Optional for the first cross-venue book study; needed if a later candidate explicitly uses fast spot microstructure. Do not relabel futures or the relay as that tape. |
| Late outcomes and matches | Ended Predict markets remain under observation for 30 minutes. | After that, unresolved outcomes or late indexing require explicit follow-up; there is no permanent late-resolution repair job. Retention does not certify historical completeness. |
| Joined dataset and historical scope | Receipt clocks, raw records, local sequence/connection evidence and engineering checks exist. | A continuous, per-window cross-venue quality ledger is not running. It must preserve missing/one-sided/locked/stale/clock/rollover statuses, validate initialization and source mapping, and account for every expected window. |

The Predict reference tape starts **2026-10-22T13:29:01Z**. Earlier Predict
books have no simultaneously captured reference receipts. Historical prices
could support reference-level backfill, but cannot reconstruct when the live
collector would have learned them. Predict scope is BTC 5m; this review does
not imply coverage of other Predict assets or horizons.

Signed Chainlink reports are not being recorded. They are an optional source
for independent report verification; they are not required for the currently
running venue relay. No additional credentials or collector were requested
or deployed during this review.

### Bounded settlement-reference check

Before examining the comparisons, selected five consecutive completed windows
starting October 22 at 13:35, 13:40, 13:45, 13:50 and 13:55 UTC. Primary
comparison was exact Decimal equality at the stated boundary second; adjacent
seconds were declared diagnostics only. The checker proves an exact-match
control, a mismatch control, ambiguous-second classification, missing-tick
status and invalid/nonfinite/boolean refusal before reading results.

As of **2026-10-22T15:35:13.464067Z**, **n=5 completed windows / 10 boundary
fields** were all present in the existing raw metadata, with all five official
resolutions. Eight boundary fields matched the relay exactly. The remaining
two are the shared end/start boundary at 13:50:00 UTC: the official reference
is `81588.445`, while Predict's raw WS JSON says `81588.44499999999`, a Decimal
difference of `1E-11`. The latter spelling is already on the wire; the
collector did not create it. No rounding tolerance was introduced to relabel
those cases as exact matches.

An unauthenticated read-only website GraphQL query subsequently retrieved
all five markets' explicit reference publish times. All **10/10 boundary
timestamp fields** equal their respective nominal UTC boundaries in this
sample, and market IDs, provider, symbol, relay ID and reference values
reconcile to the archived REST records. This confirms those samples, not a
universal settlement-selection rule. Website category IDs are slugs, whereas
the REST category ID is numeric; an initial numeric-ID GraphQL query returned
null and is not used as evidence of missing data.

Continue storing raw decimal spellings and official final reference values;
do not calculate settlement labels from an arbitrarily rounded relay price.
The first collection improvement is ongoing archival of the explicit
reference publish timestamps, followed by mapping and joined-window QC.
Public price/book collection alone does not fill the execution/queue gap.

All raw evidence, declared scope, checker controls, comparisons and follow-up
queries are under `data/predict_fun/audits/20261022_collection_scope_v1/`;
`summary.json` binds their hashes. This audit reads reference labels for
engineering on October 22 only. It runs no predictive score, economic
markout, P&L, significance test or candidate selection and changes no existing
forward-test artifacts.

### Five-minute collection monitor — October 22

The contract windows remain the nominal UTC five-minute intervals. The extra
`startPricePublishTime` / `endPricePublishTime` fields describe the actual
reference observation chosen for those boundaries; they do not redefine the
window. Continuous archival of those website fields remains a separate task.
Execution and cancellation latency measurements are deferred to Dublin.

`data_collectors/predict_window_check.py` is a read-only observation monitor,
with no API calls or additional price subscriptions. Its explicit collection
scope starts at `2026-10-22T13:29:01Z`; therefore the first full window is
13:30–13:35 UTC. Earlier windows are outside scope, not labeled missing.
No historical reconstruction is attempted. Expected windows come from the
clock, independently of the files that happen to exist, and become checkable
two minutes after close.

Each expected BTC window must have observations in six named tapes: Predict
book, Predict unsigned reference relay, PM UP book, PM 60-second TWAP, Binance
futures bookTicker, and the existing PM Binance spot-price mirror. Reports
preserve missing-feed, missing-file and missing-metadata statuses, one-sided
and locked/crossed book diagnostics, and the largest receipt gap including
window edges. A gap over five seconds is a diagnostic, not proof of packet
loss on a quiet feed. No additional spot collector is introduced.

The `predict-window-check.timer` runs at UTC minute 02, 07, 12, …, 57, giving
the most recently closed window the two-minute grace period. A lock prevents
overlapping checks. Results are written to
`data/predict_fun/window_checks/latest.json` and immutable per-run JSON
receipts under `receipts/`. Cached window rows identify their original
receipt; missing windows are retried, and `--recheck-all` rereads all windows.
Failures refuse a successful replacement and write `last_failure.json`;
consumers must check receipt freshness and service status, rather than treat
an older success as current. No external alerts are sent.

This is observation monitoring, not a complete joined quality ledger or
day certification. It does not establish queue initialization, packet
completeness, source latency, fills, or profitability. Tests exercise the
actual CLI with complete tapes, an absent entire feed, a missing window,
missing metadata and a wrong run identity; separate controls refuse sealed
gzip truncation and an incomplete expected-feed set. Test fixtures also
confirm that nonadjacent local sequence numbers are valid across interleaved
collector sources.

Initial scan as of **2026-10-22T15:54:37.467062Z**, **n=28 expected BTC
windows** (13:30–15:50 UTC), found no window missing observations from any
of the six feeds. All 28 PM mappings were known before open. Maximum
within-window receipt gaps including edges were 2.810 seconds for Predict
books and 3.938 seconds for the Predict relay; neither had a gap over five
seconds. PM TWAP had gaps over five seconds in 7/28 windows (maximum
12.738 seconds), and the spot mirror in 3/28 (maximum 12.325 seconds).
These rounded maxima are rounded upward from the receipt; they describe
observed delivery gaps without identifying an upstream cause.

The receipt retains one unidentifiable PM metadata ledger row as an explicit
diagnostic, as well as empty/one-sided and locked/crossed book observations.
It does not promote those observations to usable two-sided books. The initial
scan read three still-open gzip prefixes; this is explicitly recorded in
input provenance. The 29 focused monitor/parser tests passed. The timer is
enabled, and the installed oneshot service completed successfully at
15:56:01 UTC. Existing stream collectors were neither modified nor restarted.

The first automatic timer invocation at **15:57:00.443051 UTC** added the
15:50–15:55 window: **n=29 expected windows**, still zero windows without
observations from some required feed. It reread one new window and completed
successfully at 15:57:08, confirming the periodic incremental path.
