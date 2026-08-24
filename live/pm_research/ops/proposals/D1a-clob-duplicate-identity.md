# D-1a handover — the CLOB duplicate-identity conflict is TWO classes, not one

**From OPS to DA (`tier1_pipeline.py::normalize_clob` is DA-locked; OPS does not
apply anything).** OPS measured this while triaging the outage, before the
re-cut moved the repair to DA. Handed over so DA does not repeat the scan.
Characterisation only — **no admissibility rule is proposed here**; that is
R-ADMISS and the coordinator's to ratify.

## Where it fires

`tier1_pipeline.py:1059`,
`ValueError: duplicate identity has conflicting payload for btc-updown-5m-1787184000`.
Two raw records share `_raw_message_key` but differ in canonical payload.

## Scope measured

60 btc shards of `raw/20260820/`, 7,953,968 records: **4 conflicts across 4
shards**, two per class. Rare, and totally blocking — the parser aborts on the
first, which is why only the 00:00 UTC window is ever named. Not yet extended to
all coins/days; that sweep is DA's under D-2 ("how many `(day, coin, slug)` keys
are affected"), and the harness OPS used is at
`ops/proposals/scan_clob_conflicts.py` if it is useful.

## Class A — `book`. Same book, different decoration. Benign.

Key `(event_type, asset_id, hash, timestamp)` matches; `bids`/`asks` match; the
venue's own book `hash` matches. The **only** differences:

```
A: last_trade_price "0.820"   tick_size "0.01"
B: last_trade_price  null     tick_size  null
key: ('book', '70343184730245917298…130', '88dedbc08d479e0f54a230d87812c946cd0f3e3d', '1787184063582')
```

The venue emits the same book snapshot with and without the optional
decorations. Genuinely one observation.

## Class B — `price_change`. Same delta, DIFFERENT top of book. NOT benign.

Same `timestamp`, and every per-row `(asset_id, hash, price, size, side)`
matches — so `_raw_message_key` collides — but `best_bid`/`best_ask` differ:

```
A: row1 best_bid 0.21 best_ask 0.23  | row2 best_bid 0.77 best_ask 0.79
B: row1 best_bid 0.22 best_ask 0.23  | row2 best_bid 0.77 best_ask 0.78
key: ('price_change', '1787188741166', (('853045302…268','47ff9dca…','0.22','0','BUY'), …))
```

These are **two different top-of-book observations**, not two copies of one. The
key is too narrow: it omits exactly the fields that differ. Collapsing them as
duplicates silently discards a distinct book state — and `best_bid`/`best_ask`
is the field the programme reads book state from, `book` snapshots being p90
6.2 s stale.

So the guard is right to refuse, and **the two classes cannot take the same
treatment.** A single "dedup harder" or "compare fewer fields" change would
either keep the lane wedged on class A or silently drop class B state.

## What OPS is explicitly not doing

Not proposing which copy is authoritative, not proposing an exclusion, not
touching the guard. `CLOB_ADMISSIBILITY_PROTOCOL.md` governs *window*
admissibility and is silent on raw-record identity, so this needs its own
ratified rule rather than an extension read into that one.

## Not yet checked, and DA's D-2 asks for it

Whether either class is the known duplicate-collector overlap
(`exp_e0_data_audit.py`, `DATA_COLLECTOR_AUDIT_2026-08-20.md`). Note the
programme's own rule that `recv_ns` differs per process so exact-line dedup does
not catch a duplicate collector: class B is consistent with two *venue*
messages, but the collector-era/PID provenance of each copy is unverified.
Supervision-side, OPS confirms exactly one `collect_pm.py` and one
`collect_pm_prices.py` process, both under systemd since 2026-08-21 01:43.
