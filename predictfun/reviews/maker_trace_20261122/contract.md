# MAKER-TRACE-RECONSTRUCTION-v1 — build contract

2026-11-22, session aea5bfb9 (development).

**Governing documents:** [plan](../automatic/20261122_maker_trace_reconstruction/plan.md) and its [coverage amendment](../automatic/20261122_maker_trace_reconstruction/coverage_amendment.md) (8b39f9e8, 9ff1e197).

**USER request:** "we figure out how top makers make money first".

**Code anchor:** `predictfun/research/pf_mt_common.py`. It holds the day guard, paths, schemas, statuses, classes and scenarios. Every module imports it, and nobody redefines them.

## 1. Rules every module obeys

**Dates and hours:**
- Only consumed days: Oct 23–27 (whole days) and Nov 8 hours 00–22.
- Use `pf_mt_common.hour_path` / `day_hours`, which refuse before any filesystem access.
- Never `pf_io.hours_of` padding.
- Never type, print or glob a path for any other date.

**Running:**
- Every Python run goes through `predictfun/research/run_light.sh` with the assigned `PF_SLOT` and absolute script paths.
- Log to `predictfun/reviews/maker_trace_20261122/logs/<tag>.log`; tags start `mt_`.
- Memory: a self-guard of `GUARD_MIB = 1450` on VmHWM. Caps are never raised; process per hour or per market.

**Outputs:**
- Outputs are write-once under `OUT` (refuse to overwrite).
- An exclusion is a **status**, never a silent drop. Every table carries `status`, and counts are reported.
- Compute predicates; never print conclusions. A claim such as "exhausted" or "front" is a column computed by code.

**Tests and repository:**
- Selftests use synthetic fixtures with exact expected values, plus known-bad inputs that must be refused, with the reason asserted.
- No commits, and no edits to tracked files or to FINDINGS / HANDOFF / STATUS. Run `git ls-files` before creating a file.

## 2. Populations (declared before any outcome is read)

| Name | Definition | Use |
|---|---|---|
| ID | every BTC 5m market of Oct 23 (REST legs only) | ex-ante maker identities: top 10 addresses by maker shares, and top 10 by fill P&L to resolution |
| FILL | every BTC 5m market whose window starts on Oct 24–27 | ledger, fill economics and markouts (top of book) for every maker |
| BOOK | the 19 complete markets (`reviews/ladder_async_fast_20261122/manifest.json` `complete`) | full-depth queue and lifecycle reconstruction |
| GT | Nov 8 live windows in `/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/scope.json` | our own orders, whose submit / size / cancel are known: the reconstruction's real-data positive control, scored blind |

A market is assigned to the UTC day of its window start `ws`, read from `step1/<day>/registry.parquet` (`mid, slug, ws, we, outcome`). REST rows are filed by receipt hour.

**Boundary status:** a FILL market whose records could be received in a forbidden hour gets `boundary_censored`. The margin is the measured p99.9 receipt lag, rounded up to whole minutes. Oct 23 uses Oct 24 00Z, which is allowed; Oct 27 never reads Oct 28.

**Same-day top-K:** descriptive only, and labelled as such.

## 3. Raw semantics (inventory 2026-11-22 + venue report; DOC / OBS / ASSUME)

**Match record** (`/v1/orders/matches`, `data[]`). One record is one settlement:
- `taker` and `makers[]`. Each leg has `hash`, `signer`, `quoteType`, `amount`, `price`, `outcome.indexSet` and `fee{amount,type}`.
- Amounts and prices are 1e18-scaled strings.
- Checked exactly in integers: `taker.amount == amountFilled == Σ makers.amount`. `taker.price == priceExecuted` is the taker's AVERAGE price, not its limit.
- Maker fee is 0 (OBS). The taker fee is ≈ 0.02·min(p, 1−p) per share, in SHARES or COLLATERAL.

**Complementary matches:** a YES bid can match a NO bid (mint), and a YES ask a NO ask (merge). Everything is mapped to **YES space** by `yes_space` (Up = indexSet 1 = YES).

**Book frame** (`predictOrderbook/<mid>`): a full aggregated YES-space snapshot (DOC), with no per-order identity.
- Fields: `updateTimestampMs` (venue ms), `orderCount` (semantics UNK), and `settlementsPending` (USD, at the taker's limit).
- `version` is constant at 1 and unused.
- Publication is about every 50 ms, and frames arrive p50 53 ms after `updateTimestampMs`.

**Clocks — never mix:**

| Field | Clock | Meaning |
|---|---|---|
| `upd_ms` | venue book | frame time |
| `recv_ns` | collector receipt | when we received it |
| `settle_ms` | UUIDv7 of `settlementId` | id creation: bounds the match from ABOVE only (C23) |
| `exec_ms` | `executedAt` | whole-second Confirmed time, 0 / 1 / 10 s after Matched: not a match time |
| `first_recv_ns` | REST receipt | first receipt of a record |

## 4. Modules, owners, outputs (`OUT/…`, schemas in `pf_mt_common.SCHEMAS`)

| Module | Output |
|---|---|
| `pf_mt_extract.py` | `records_<day>.parquet`, `legs_<day>.parquet`, `tob_<day>.parquet`, `depth/<day>_<mid>_{frames,levels}.parquet` (BOOK + GT markets), `inventory_<day>.json`, `extract_<day>.json` |
| `pf_mt_lifecycle.py` | `life/<day>_<mid>_{fills,orders}.parquet` (orders have one row per hash per scenario), `life/<day>_<mid>.json` |
| `pf_mt_econ.py` | `econ/maker_market.parquet`, `econ/maker_day.parquet`, `econ/fills_valued.parquet`, `econ/segments.parquet`, `econ/identities.json` |
| `pf_mt_truth.py` | `truth/own_orders.parquet`, `truth/scores.json` (GT) |
| `mt_checks.py` (receipt dir) | an independent checker and falsifier |
| `pf_mt_report.py` (integration) | coverage tables, timelines, report |

**De-duplication:** a record's key is `settlementId`. Canonical content excludes volatile fields: the `market` object, `outcome.status`, `bestBid` / `bestAsk`, `onChainId` and receipt fields. A re-poll with identical canonical content raises `n_recv`. Different content under one key gives `status=conflict` (first kept, both counted).

## 5. Lifecycle rules (`pf_mt_lifecycle`)

### Fill window `[lo_ms, hi_ms]` per maker leg

**Upper end:** `hi_ms = min(settle_ms, first_recv_ns // 1e6)`.

**Lower end:** from the book, using the reviewed method of `pf_step1_timing.py` / `pf_step1_booktime.py`:
- prefer the first frame whose `settlementsPending` increment matches the record; otherwise use the drop at the maker's YES price;
- `lo_ms` = the venue time of the frame before that evidence, and `t_method` names the method;
- with no evidence, `lo_ms = -1` (unbounded) with status `fill_lo_unbounded`;
- never a fitted constant.

### Hard constraints (all scenarios)

| | Constraint | Violation |
|---|---|---|
| H1 | one hash maps to one (signer, YES side, YES price) | status `identity_conflict` |
| H2 | a resting order never grows, and a removed order never returns: proven presence spans [first fill, last fill] | — |
| H3 | at a frame `f` where a set of orders is proven resting at side/price p, the shown size ≥ Σ of their fills strictly after f | a contradiction status, and the order is not fitted away (the plan's "impossible overfill" control) |

**H3 also bounds posting:** the latest frame before an order's first fill at which shown(p) < (its fills after f) gives a posting lower bound (`post_method = capacity`).

**Not in the book:** a price absent from a full snapshot has shown size 0. A frame gap > 1 s gives an unknown span, not 0.

### Scenarios (`pf_mt_common.SCENARIOS`)

| Scenario | Rule |
|---|---|
| S_FIFO | price-time priority within a YES price, and `makers[]` lists the fill order. A maker at index i < last whose price is no worse than a later maker's was fully consumed: exhausted at that fill, `size_exact = cum_shares`. |
| S_FIFO_LO | S_FIFO with fills placed at their window's lower end (or `hi − MAX_LAG_MS`) |
| S_NOPRIO | no priority inference: only H1–H3 |

**Priority is UNKNOWN at the venue.** The lifecycle receipt therefore TESTS it:
1. **Re-appearance:** the share of S_FIFO-"exhausted" hashes that fill again later must be ~0. Report the count and rate.
2. **Ordering:** within a record, `makers[]` YES prices are monotone from the taker's best.

### Classes (per field)

- **observed:** read directly. Fills, hash persistence between fills, and the size lower bound are observed.
- **constrained:** implied by H1–H3 and identical across all three scenarios.
- **assumption_sensitive:** differs across scenarios. This includes everything that needs S_FIFO.
- **unresolved:** no bound beyond trivial.

The coverage tables report the classes separately for posting, persistence / queue and removal, over three denominators: orders, executions and shares.

### Queue at each fill (S_FIFO)
- `ahead_lo` = the shares of earlier same-price makers in the same record.
- `ahead_hi` = shown(p) at `f_before` − this order's fills after `f_before`. It is clipped at 0 only with status, never silently.

### Removal set
Keep every possible explanation from `REMOVAL`, with a window `[removal_lo, removal_hi]`:
- `exhausted` requires S_FIFO non-last evidence, or shown(p) after the fill = 0 with no other proven resting order;
- `end_censored` applies when the order could still rest at the last supported frame;
- `cancelled_or_expired` applies when shown(p) drops below its possible remaining without a print at p.

Never a single invented time.

### Linking
A *refill* is the same signer, a new hash, the same side and price, first fill after the previous hash's last fill. A *reprice* is the same at a different price. Both are labelled descriptive patterns: unfilled orders are invisible.

## 6. Economics (`pf_mt_econ`)

**Value of a fill.** Every fill, maker or taker leg of an address, is valued in YES space. With s = +1 for a YES buy and −1 for a sell, at price p, and V = 1 if Up won:
- P&L to resolution = Σ s·q·(V − p) − fees;
- spread at the fill = s·q·(m₋ − p), where m₋ is the YES mid of the last frame with `upd_ms` ≤ the fill's `lo_ms` (else `hi_ms`, with a status);
- markouts s·q·(m_{t+h} − p) at h = 1 s, 10 s, 60 s and END. END is the last frame ≤ we − 2 s.

**Decomposition:**
- FIFO lots per address × market split the P&L into a matched round-trip profit and a residual inventory value;
- the residual is again split into a drift and a resolution part.

**Segments:**
- market time (pre / 0–60 / … / 240–300 / post);
- price region (|p − 0.5| buckets; ≥ 0.9 / ≤ 0.1);
- leg role (maker / taker; mint / merge / regular);
- lifecycle class on BOOK (repeated / single fill, S_FIFO exhausted / partial, front (`ahead_hi` = 0) / behind).

**Identities:** top 10 by Oct 23 maker shares and top 10 by Oct 23 P&L (ex-ante), compared with the field and with the same-day top 10 (descriptive).

**Days:** per-day point estimates for Oct 24–27. Four days is fewer than 5, so there are no intervals (rule 8), and the report says so.

**Rewards:** `market.rewards` / `spreadThreshold` / `shareThreshold` are inventoried. Per-maker rewards are unobservable. Report only a demonstrated-eligible-presence lower bound (proven presence of orders with `size_lb` ≥ the share threshold within the spread threshold of the mid).

**Not a strategy:** no imitation, fit, tradeable claim or profitability verdict for us.

## 7. Ground truth (`pf_mt_truth`, GT)

1. Run extract and lifecycle on the GT markets blind, exactly as on BOOK.
2. Join our own orders by hash to `kernel_orders_*` / `core_orders_*` (export `live_vs_replay/`), which give true submit / activation / size / cancel / final state.

**Score:**
- coverage of our filled orders;
- whether the true activation lies in `[post_lo, post_hi]`;
- whether the true size ≥ `size_lb`, and equals `size_exact` when stated;
- whether S_FIFO "exhausted" is truly FILLED, and how a removal set's true member is distributed;
- each per class: a "constrained" claim that fails is a defect, not noise.

Only the export's derived files are read. The capture directory is not opened.

## 8. Independent checker (`mt_checks.py`)

**Independence:** written from this contract, not from the builders' code.

**Raw re-parse:** re-parse ONE BOOK market and ONE GT market from raw (only the needed hours), then recompute:
- unique records, legs, and integer amount reconciliation;
- per-hash cumulative fills and the H1–H3 predicates.

**Checks on the builder outputs:**
- schemas;
- statuses counted, nothing silently dropped;
- class consistency (no observed / constrained removal without evidence; size_exact ≥ size_lb; removal_lo ≥ the last fill's lo);
- coverage arithmetic, and the GT scores recomputed.

**Falsifier:** planted defects that must each be refused, with the reason asserted:
- an overfill;
- a fill after removal;
- an identity conflict;
- a forbidden-hour path;
- a dropped row;
- an "observed" cancel;
- a time inversion;
- a duplicate-key conflict merged silently.

## 9. Resources

The run is projected from measurement: one REST hour (86 MB) costs 12.65 CPU s.

Projection:
- REST for Oct 23–27 plus Nov 8: ≈ 0.45 h;
- WS: ≈ 0.10 h;
- lifecycle, econ, truth, report and checks: ≈ 0.15 h;
- selftests: ≈ 0.03 h.

It needs a USER-approved cap before any non-synthetic run beyond the schema probes. The probes so far cost 13.5 CPU s, declared.
