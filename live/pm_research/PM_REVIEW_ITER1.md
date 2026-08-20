# PM_REVIEW_ITER1 — data-completeness review, iteration 1 (PM-E0.5)

Date: 2026-08-19 ~15:15–15:40 UTC. Reviewed: `PM_MM_PLAN.md` (§2–§5),
`collect_pm.py`, `collect_pm_prices.py`, live data under `data/pm_5min/` +
`data/mm_hf/raw/`. All findings empirical unless marked (code-read).

---

## A. L3 identity test — VERDICT: Binance SPOT mirror, not the settlement stream

Overlap tested: 15:12:50–15:26:14 UTC (804 s, btcusdt, 1 s cadence, n=804).

| comparison | result |
|---|---|
| PM `crypto_prices` value at payload-ts t vs **Binance spot 1s-kline close of t−1** | **803/803 = 100.0% EXACT 8-decimal string match** |
| vs spot open@t / close@t | 54.9% / 20.8% (partial, as expected for adjacent samples) |
| vs Binance futures mid (local bookTicker) | mean **+4.60 bps**, std 0.76 bps — distinct series (spot/perp basis) |
| 1s-diff correlation vs futures mid | 0.92 @ lag 0, ≤0.16 at ±1–2 s — synchronous, not lagged |
| 60s-averaging signature | none (it is raw last-price; see TWAP test below) |

The feed is a **1 Hz snapshot relay of Binance SPOT last trade price**
(value = spot kline close of the previous second, Binance 8-decimal string
format passed through verbatim; recv−payload latency mean 471 ms). It is a
chart mirror — **not** the Chainlink aggregate and **not** the TWAP stream.
Plan §4 L3 row should be re-labeled accordingly (role: "what retail sees" +
free spot series; NOT a settlement input).

**The real stream is separately available on the SAME WebSocket** — see B.

TWAP smoothing test (85 s probe of the real stream, btc): `crypto_prices_twap_sixty`
vs trailing 60s mean of the spot mirror: residual std **0.62–0.84 bps**
(smoothing signature confirmed) with mean offset **−5.4 to −5.8 bps** =
Chainlink-aggregate-vs-Binance-spot venue basis. Unsmoothed comparison std ~10 bps.
Consequence: a Binance-synthetic X̂ carries a ~5 bps drifting bias — larger than
many tie zones — so recording the actual stream (B) is the difference between
"calibrated proxy" and "truth."

## B. L4 settlement-truth recovery

**(1) On-chain (Polygon): DEAD for prices — outcome-only.** Verified on our
resolved window `btc-updown-5m-1787149500` (conditionId `0x8cf1a33f…7761`):

- `ConditionResolution` log on CTF `0x4D97DCd9…6045`, tx
  `0x91df43400cd5f129b6cfe6ab16060e5897e692da4adb99c22eea1ce2f9314005`,
  block 92296572 @ 14:31:25 (window end 14:30:00, +85 s). Oracle adapter =
  `0x58e1745bedda7312c4cddb72618923da1b90efde`.
- Resolution tx is an ERC-4337 bundle → `execute` on the adapter, selector
  `0xc49298ac(questionID, uint256[] payouts)` with payouts **[1,0] only — no
  Chainlink report, no price anywhere in calldata or receipt logs** (6 logs, all
  decoded). Adapter emits exactly ONE event type (87 logs scanned over ~30 min,
  all `0xc50439cd…` questionID+payouts). The resolver bot reads Data Streams
  off-chain and posts only the outcome. (K, X_T) are NOT on-chain.
- Still useful: on-chain logs give an independent resolution record +
  settlement timestamp (+85 s) for any conditionId, forever. Recipe (verified,
  public RPC `polygon-bor-rpc.publicnode.com`): eth_getLogs(address=CTF,
  topics=[`0xb44d84d3…5894`, conditionId]).

**(2) data.chain.link: BLOCKED** — Vercel Security Checkpoint (JS challenge),
both curl-with-browser-UA (429/403 + challenge page) and WebFetch (403). No
JSON API reachable headlessly. Chainlink DataEngine REST/WS
(`api.dataengine.chain.link`) requires commercial credentials.

**(3) SOLVED — Polymarket RTDS relays the settlement stream publicly.**
Per docs.polymarket.com/market-data/chainlink-twap, the same
`wss://ws-live-data.polymarket.com` we already record serves topics
**`crypto_prices_twap_sixty`** and **`crypto_prices_twap_thirty`** — the
Chainlink TWAP stream updates, no credentials. **Probe-verified 15:23–15:25 UTC**:
676 msgs/85 s; 7 symbols (`btc/eth/sol/xrp/bnb/doge/hype + "/usd"` naming);
payload `{symbol, value, full_accuracy_value (1e18-scaled), timestamp,
window_s: 60}`; 1 s median cadence (max observed gap 8 s). This is X_t.
K = X_t at window_start, X_T = X_t at window_end, directly recorded.
Caveats: RTDS has **no snapshot/history/replay** — a disconnect at a window
boundary loses that window's K or X_T irrecoverably (free tier); stream gaps
up to 8 s exist, so boundary reads need a nearest-tick-within-ε rule and
gap-windows must be flagged, not interpolated.

**(4) Historical X_t backfill exists (paid, third-party):** pmdata.dev serves
daily Parquet of the exact streams (`streams_twap60s`/`30s`, 7 coins, since
2026-08-01, fields incl. observationsTimestamp/price(1e18)/validFrom/expiresAt)
at `api.pmdata.dev/chainlink/{SYM}USD/streams_twap60s/…parquet`, API-key gated.
Covers the whole post-rule-change era (change 2026-08-07). Decision for the
orchestrator: forward-collect free (from today) vs buy the ~18-day patch.

**Fallback assessment:** Binance-synthetic X̂ + outcome calibration is
adequate for σ̂/μ̂ and coarse p̂, but its −5.4 bps drifting venue basis (A) makes
near-tie classification unreliable; with (3) available there is no reason to
lean on it for truth.

## C. Traceability matrix (plan §2–§3 symbol → collected series)

| symbol | source / file / field | status |
|---|---|---|
| F_t (fair value) | L2 futures `data/mm_hf/raw/bookTicker/<SYM>/<hr>.csv` (recv_ns,E,T,u,bid,bidqty,ask,askqty); depth20/trade same tree | OK — continuity verified (hour boundary 3 ms; rotation is flush-based, hour files start ≈hh:06, no gap) |
| σ̂, μ̂, burst flag | derived from L2 trade + bookTicker | OK |
| X_t stream (settlement) | `crypto_prices_twap_sixty` topic — **not subscribed by collect_pm_prices.py** | **MISSING → MUST-FIX 2** |
| K = X_0 | X_t at t=window_start (stream lookback embeds pre-window [−60,0]; no extra series needed) | **MISSING until fix 2**; boundary-gap rule needed |
| X_T | X_t at t=window_end | **MISSING until fix 2** |
| S_t (Chainlink aggregate) | not directly available; nearest: twap topics (smoothed) + spot mirror (Binance venue only) | PARTIAL — acceptable; not required once X_t recorded |
| book state, both tokens | `raw/<day>/<slug>.jsonl.gz` `book` + `price_change` (asset_id ↔ markets.jsonl clobTokenIds verified; `price_change.size` = **new level total**, 99.89% snapshot-consistent; best_bid/ask on every delta) | OK (minus damaged windows, D) |
| trades / fills / markout labels | `last_trade_price` events: price, size(shares), side, fee_rate_bps, transaction_hash | OK forward; Data-API backfill not yet built (plan build item 1) |
| fee_rate | `fee_rate_bps` recorded — **all 19,141 trades = 0**; docs say crypto taker feeRate 0.07, CLOB market says maker/taker_base_fee 1000; CLOB fee fields NOT captured at discovery | PARTIAL — conflicting sources, E1 must reconcile → SHOULD-FIX 3/4 |
| rewards params (band) | markets.jsonl rewardsMinSize=50 / rewardsMaxSpread=4.5 / orderPriceMinTickSize / orderMinSize — present 48/48 | OK |
| rewards RATES ($/market) | CLOB market `rewards` object — not captured | MISSING → SHOULD-FIX 3 (needed for G3b rewards PnL line) |
| resolution winners | resolutions.jsonl `source:"clob"` rows (32 clean; 23 Up/9 Down; 8 documented garbage rows need `is_final` filter) | OK |
| q, sim fills | replay-internal over books+trades; queue bracket supported by level-total semantics; no MBO (bracket only, as planned) | OK |
| tick size | orderPriceMinTickSize + `tick_size_change` events (40 observed, 0.01→0.001) | OK — plan §2.4 text needs the 0.001-regime caveat |
| τ, window bounds | markets.jsonl window_start/window_end/endDate | OK |
| on-chain resolution ts (aux) | Polygon CTF logs, recipe in B(1) | OK (optional) |

## D. Collector audit on real data

1. **MUST-FIX 1 — restart loses up to 2 windows/coin (verified data loss).**
   `_load_state` marks all markets.jsonl slugs `known` but never re-spawns
   `_market` tasks for windows still in flight; discovery then skips them.
   At the 15:12:49 restart: `btc-updown-5m-1787152200` (15:10–15:15) recording
   stops 15:12:45 (last 2m15s of the window lost); `*-updown-5m-1787152500`
   (15:15–15:20) contains **495 msgs, all pre-window (15:10:22–15:12:46) — the
   entire in-window book is missing for all 4 coins**; heartbeats show
   `msgs=0` for 3 min. Fix: on resume, re-spawn `_market` for any known window
   with now < end+grace. Also: gzipped partials carry **no truncation marker** —
   consumers must gate on recorded span (last recv_ns ≥ end+grace−ε), and the
   8 damaged files from today must be excluded from any fill/flow statistics
   (they look like valid, merely "quiet" windows otherwise).
2. **markets.jsonl fields: verified.** 48/48 rows carry full description
   (~400 chars, names the exact stream URL), resolutionSource, rewards params,
   tick/min-size, endDate. Window sequences gapless 14:25→15:25, 4 coins.
3. **Resolutions: flowing and clean.** 32 CLOB rows, winners consistent
   ({Up,Down} complementary; spot-checked btc-14:25 vs on-chain payouts [1,0] ✓).
   Garbage rows (8) are pre-fix history, `is_final` filter documented in
   HANDOFF. Give-up path untriggered so far; note `is_final` treats
   `gave_up` as terminal — a later-resolving market is never re-polled
   (SHOULD-FIX 7, cheap: re-poll gave-up slugs on restart).
4. **Raw WS files parse clean.** btc 15:05 window: 88,564 lines, 1 blank, 0
   unparseable; types book 3,894 / price_change 82,838 / last_trade_price
   1,828 / tick_size_change 4; recording spans t−280 s → end+90 s. Message
   `market` field == conditionId, asset_ids == clobTokenIds (verified).
   Trade side labels are heavily skewed (BUY 16,147 / SELL 2,994) — likely
   mechanical from the unified book; E1 item, not a bug.
5. **Prices CSV**: schema `recv_ns,t_ms,symbol,value` OK; 6 symbols × 1.00 s
   median cadence, max gap 1.98 s in 804 s; recv−payload latency 281–750 ms
   (mean 471 ms). Rotation gzip works on hour change (code-read: an
   hour-boundary with zero traffic leaves the previous hour un-gzipped —
   cosmetic only).
6. **Disk burn**: 255 MB (gz) for ~1 h / 4 coins ≈ **6 GB/day** — matches the
   HANDOFF estimate; fine short-term, revisit before multi-week collection
   (book snapshots are ~40% of volume and re-derivable from deltas).
7. NOTED: restart `nohup … >` **truncated collector.log** (first 3 h of logs
   gone); use `>>`. Heartbeat/counter behavior otherwise correct.

## E. Design gaps

1. **Uncollected markets exist**: `doge/bnb/hype-updown-5m-*` are live on
   Gamma (probed 15:30 UTC) — exactly the RTDS 7-coin set — plus
   `btc-updown-15m-*`. COINS=(btc,eth,sol,xrp) was chosen before this was
   known. Decision needed: extend (same infra, +75% disk) or re-affirm scope
   in the plan; silence would look like an oversight later.
2. **Matching mechanics**: public docs + secondary sources agree on
   **price-time priority, hybrid off-chain matching / on-chain settlement,
   unified book** (a Up-bid at 0.60 can match a Down-bid at 0.40 via mint;
   complement merge on the way out). This validates §2.3 pair-harvest and the
   E3 queue bracket (join-back vs pro-rata). No order-lifecycle (MBO) feed
   exists on the public market channel — queue POSITION is unobservable;
   bracket approach stands, user-channel data would only come with auth'd
   trading (PM-E4).
3. **Settlement convention residual ambiguity**: market description reads
   "TWAP … of the time range specified in the title ≥ the price at the
   beginning of that range" — literally a 300 s TWAP vs an opening price,
   while plan §2 models stream-reading X_T ≥ X_0 (both 60 s-smoothed). The
   60s-stream resolutionSource supports the plan's reading, but no public doc
   states the exact boundary convention (which report at T, observations vs
   validFrom timestamp). E1 must disambiguate empirically (rule variants vs
   ≥50 recorded winners, near-ties weighted) before p̂ is trusted at |d| small.
   Until then treat the rule as hypothesis, not fact.
4. **Fee truth**: three sources conflict (docs crypto taker 0.07 formula
   `C·feeRate·p(1−p)`, 15–25% maker rebate; CLOB market
   maker/taker_base_fee=1000; observed fee_rate_bps=0 on every trade).
   E1's fee empirics cannot rely on the WS field alone; snapshot the docs +
   capture CLOB fee fields per market + cross-check Data-API trades.
5. **Docs snapshot still missing** (`data/pm_5min/docs/` — plan build item 3):
   fees, rewards, TWAP/RTDS pages are versioned and known to change (H-PM4);
   snapshot before E1 freezes gates.
6. **Units audit**: book/trade `size` = shares (outcome tokens), `price` =
   USDC/share ∈ (0,1); notional = price×size; rewardsMinSize is in shares;
   TWAP full_accuracy_value scaled 1e18 (USD); spot mirror = raw USDT price.
   No inconsistencies found.

## Triage

| # | severity | finding | owner action |
|---|---|---|---|
| 1 | **MUST-FIX** | collect_pm.py resume never re-spawns in-flight windows → every restart silently loses up to 2 windows × 4 coins (verified: 15:15–15:20 window empty, 15:10–15:15 truncated). Partial files carry no marker. | Re-spawn on resume for now<end+grace; add recorded-span gate to consumers; blacklist today's 8 damaged files |
| 2 | **MUST-FIX** | Settlement stream NOT being recorded: K=X_0 and X_T have no source in the current data layer. `crypto_prices` is a Binance spot mirror (A); the true stream is publicly available as `crypto_prices_twap_sixty`/`_thirty` on the same WS (B, probe-verified). | Subscribe both TWAP topics in collect_pm_prices.py (schema incl. window_s); start TODAY — no replay exists; decide on pmdata paid backfill for Aug-01→now |
| 3 | SHOULD-FIX | CLOB per-market metadata not captured (maker/taker_base_fee, rewards rates object, seconds_delay, is_50_50_outcome) — G3b rewards line and fee model need them | one CLOB GET per market at discovery, store full JSON |
| 4 | SHOULD-FIX | Fee sources conflict (docs 7% / CLOB 1000 / observed 0 on 19,141 trades) | E1: reconcile empirically + docs snapshot; never hardcode |
| 5 | SHOULD-FIX | Settlement boundary convention unverified (E.3): plan's X_T≥X_0 vs literal 300s-TWAP reading; tie/timestamp convention unknown | E1 rule-variant test vs recorded winners incl. near-ties; plan §2 marked provisional until then |
| 6 | SHOULD-FIX | doge/bnb/hype 5-min (and btc 15-min) markets exist uncollected | explicit scope decision in plan; extending COINS is a 1-line change + disk |
| 7 | SHOULD-FIX | `gave_up` resolutions never re-polled; docs snapshot dir missing; collector.log truncated by restart (`>` → `>>`) | small collector/process hygiene batch |
| 8 | NOTED | price_change = level totals (99.89%); price-time priority + unified book documented → E3 queue bracket sound | — |
| 9 | NOTED | TWAP stream: 1 s cadence, gaps to 8 s, no replay → boundary-gap windows must be flagged unquotable/excluded, never interpolated | E1 rule |
| 10 | NOTED | Chainlink-aggregate vs Binance-spot basis ≈ −5.4 bps (std 0.6–0.8 bps, 85 s probe) — synthetic X̂ unfit for near-tie truth; fine for σ̂/μ̂ | basis series measurable once TWAP recorded |
| 11 | NOTED | On-chain = outcome-only (payouts [1,0]), resolution at end+85 s; recipe kept as independent audit path | — |
| 12 | NOTED | Disk 6 GB/day; L2 hour files rotate at flush (≈hh:06) with 3 ms continuity; PM feed latency ~471 ms | — |

Verdict: **data layer NOT yet complete** — two MUST-FIX items, both cheap and
both time-critical (no-replay stream + ongoing restart exposure). After fixes
land and ≥1 day of TWAP topics + repaired-restart data accumulate, iteration 2
re-audits on real data per charter.
