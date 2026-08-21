# CLOB cause-aware admissibility — protocol `clob_adm_v1`

Status: **FROZEN BEFORE USE**. Version `clob_adm_v1`.

This is the pre-registered alternative to the unachievable acceptance boundary
"one full busy UTC day with zero `SLOW_CONSUMER_1013`". The `1013` is
**venue-side** (`ws_ever_paused` False, queue depth 0.2 % of the pause
threshold), so that boundary tests something we do not control. This rule
decides instead **which windows may carry flow/fill inference**, and it is a
shared input to G-FF2, G-FF3 and G-FF4 — built once, keyed on cause, never on
seconds-lost alone.

## Design-data disclosure

The ledger and window inventory were inspected to design this rule: cause
counts, per-coin concentration, era boundaries, shard prevalence and ledger
coverage. Those are **structural facts about the collector**, not outcomes of
any flow statistic. No `α_trade`, ζ or queue quantity was computed before this
freeze.

## The finding that shapes the rule

**The gap ledger begins 2026-08-20 14:50:21 UTC. Windows begin 2026-08-19.**
Of 3,076 distinct windows, only **1,057 (34.4 %)** lie inside ledger coverage.

For the other **2,019 (65.6 %) the absence of a gap record is not evidence of a
gap-free window** — the collector was not recording gaps yet. Treating "no gap
row" as "clean" across the whole tape would silently admit 2,019 windows of
unknown quality. That is the same error as reading `open_gaps=[]` on the prices
lane as "clean" while it logged 58 gaps in 11 hours.

Uncovered windows are therefore **`NO_LEDGER_COVERAGE`, excluded** — not
"probably fine".

## Cause classification — pre-registered, keyed on mechanism

| class | causes | evidence | handling |
|---|---|---|---|
| **MNAR** | `SLOW_CONSUMER_1013`, `PING_TIMEOUT` | coin-concentrated: 1013 is **31/32 BTC**, PING_TIMEOUT **8/9 BTC**. Activity-correlated, lands on the busiest windows | exclude, and **report the excluded set beside the retained one** |
| **MAR** | `CONNECTIONCLOSEDOK`, `NO_CLOSE_FRAME` | spread across coins (1001 over 5 coins, NO_CLOSE_FRAME over 4). Server cycling, activity-independent | exclude; the remainder stays representative |

`PING_TIMEOUT` is classified **MNAR**, against the earlier grouping in
`3b9bddc`. At n=9 it is 8 BTC + 1 ETH — it behaves like the 1013 class by coin
concentration, not like the across-coins cycling class. It is also the **largest
single loss contributor** (99.2 s of 190.4 s total) despite being fourth by
count, because its gaps are long (median 11.3 s) where 1013 gaps are short
(median 1.33 s).

**A seconds-lost-only threshold would inverentirely miss this**: it would rank
1013 as the mildest cause when it is the most selective one.

## Admissibility predicate

A window (one `slug`) is `ADMISSIBLE` iff **all** hold:

1. **`IMMUTABLE`** — every shard is `.gz`. An open `.jsonl` is still being
   written.
2. **`LEDGER_COVERED`** — `[window_start, window_start + 300 s]` lies inside a
   single collector era interval, derived from `collector_start` /
   `collector_stop` rows. This subsumes era-boundary exclusion: a window
   spanning a restart is not inside any one era.
3. **`UNSHARDED`** — exactly one shard. 152 windows carry 2–3 shards, which
   means the collector restarted or rotated mid-window; loss between shards is
   not recorded by the gap ledger and cannot be bounded.
4. **`GAP_FREE`** — no `gap_closed` row for this `slug` whose
   `[gap_start_ns, gap_end_ns]` overlaps the measurement interval.

Every exclusion is recorded with its reason and, for gap exclusions, its
**cause class**. Counts alone are not sufficient: window identity is retained
so the excluded set can be characterised later.

## Era discipline

Any analysis consuming this rule **must declare a single `collector_version`**
and may not pool across eras unpaired. Covered windows by era:

```
clob_v2      14
clob_v2_1   105
clob_v3      91
clob_v3_1   805   <- primary
```

`clob_v3_1` is the primary era: it is the current code, it is the largest, and
it is the only era with enough windows for per-coin statements.

## Mandatory reporting contract

Any consumer of this rule **must** report, beside its result:

1. retained and excluded counts, by reason and by cause class;
2. the excluded set characterised **on the statistic the consumer estimates** —
   not on a convenient proxy. The route_a lesson is explicit: a displacement
   statistic (range) answered a variance question and reversed sign once
   quadratic variation replaced it. For a flow statistic the comparison must be
   on flow intensity, not on realised return.

A consumer that reports only a retained-set number is non-compliant with this
protocol regardless of its verdict.

## What this rule does NOT establish

It does not make the tape complete. It selects windows where **recorded** loss
is absent within a covered, single-era, unsharded interval. Undetected loss
remains possible; the ledger records what the collector noticed. Nor does it
license pooling `clob_v3_1` with earlier eras.
