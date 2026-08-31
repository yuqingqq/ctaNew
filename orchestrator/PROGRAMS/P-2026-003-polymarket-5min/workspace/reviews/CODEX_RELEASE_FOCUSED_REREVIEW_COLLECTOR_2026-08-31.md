# Codex release-focused re-review — collector — 2026-08-31

**Exact review tip:** `157c6b23b49bff4320d56b3f30e4d4d9bb343f22`

**Scope:** `21dbb85..157c6b2`: HJ-R1 repair and the newly reported 08-31
all-coin low-message interval. This review deliberately does not reopen the
historical BTC root-cause search. The release question is whether a controlled
next collector era can start safely enough to learn what happens.

**Live-mutation statement:** production was inspected read-only. Nothing was
armed, restarted, stamped, or written to a production ledger/tape. Lightweight
checks ran inside `research.slice`.

## Decision

### HJ-R1 — RELEASED

The repaired host-load join now produces the exact governed population:

```text
143 cells · 665 disconnects · 663 joined · 2 UNEVALUABLE
as-of 2026-08-25T23:50:15Z · Pearson r = +0.0332696
```

The first interval is anchored at UTC day start, actual `event=disconnect`
rows are evaluated at integer `recv_ns`, the unsampled right edge is explicit,
and the selftest grows from 6 to 10 checks with join-boundary controls. The
real result matches the independent value in the prior Codex review. This
finding is closed and needs no further round.

### Controlled next-era experiment — HOLD, with one engineering blocker

**COL-R3 is the only current code/deploy blocker.** HEAD's 10/10 control-Ping
behavior and the resident 3/3 behavior both identify as `clob_v4`. Restarting
would therefore create an unidentifiable measurement regime that the era walk
cannot express (`clob_v4 -> clob_v4`). No distinct 10/10 identity or boundary
package exists at this tip.

COL-R2 remains worthwhile operational hygiene, but it is not a release gate
for this experiment. The retained telemetry refutes host-wide load as the
dominant historical break mechanism, and the live collector already runs in
`collectors.slice`. Track the drop-in and improve research-slice enforcement,
but do not hold the learning experiment for another root-cause cycle.

**Minimum release closure:** give 10/10 a distinct collector identity, wire
that identity through `collector_start`, the era consumer, emitter, and
rollback/runbook, and submit that narrow seam for review. Reuse the already
green collector/deploy machinery; do not restart the full historical audit.

After that narrow closure, proceed. The first complete future era is the test;
historical certainty about why 08-25 happened is not a prerequisite.

## R-368 — anomaly confirmed; “collector lost 4.5 hours” is not established

There is a sharp all-coin message-rate collapse on 08-31. The live log changes
from ordinary rates to nearly zero at about 06:32Z, remains there until about
10:41Z, and recovers at 10:42Z without a process restart. The tape-density
detector reports 352 low-content coin-windows across all seven coins. Selected
gzip decompressions reproduce the reported row counts (for example the 08:30Z
BTC window has 210 rows and ETH has 168, while 13:30Z has 95,321 and 53,551).

But the failure level is upstream of the writer, not demonstrated collector
data loss:

| exact interval | observation |
|---|---:|
| 06:32:56Z collector `msgs` | 42,654,493 |
| 10:41:07Z collector `msgs` | 42,705,210 |
| collector receive-counter delta | 50,717 |
| timestamped raw rows in the same interval | 50,599 |
| raw/counter reconciliation | 99.77% |
| retries during interval | 11 across all concurrent sockets |
| process restart | none |

The 118-row difference is consistent with the minute-log endpoints and file
selection edges; the raw files contain 402 additional blank physical lines,
not malformed timestamped messages. Per-coin counter deltas and raw counts are
also close. New five-minute subscriptions continued to receive book snapshots,
and `writer_wait`, queue high-water, and event-loop lag stayed clean.

Therefore the collector wrote essentially everything its receive loops saw.
What remains unresolved is **content liveness**: either the venue/book makers
genuinely produced almost no updates, or Polymarket's market-event publisher
silently degraded while connections and control-frame keepalives stayed alive.
Row density alone cannot distinguish those states. Calling the interval
“4.5 hours lost” or permanently voiding the day asserts the answer the current
instrument cannot observe. `pm_tape_density.py` itself correctly declares that
its post-selected threshold REPORTS and does not VETO.

For result integrity, keep 08-31 in an explicit
`CONTENT_LIVENESS_UNRESOLVED` status rather than counting it as a clean
validation day. That is a conservative status, not a root-cause finding.

## Learn prospectively, not by another retrospective loop

Before or with the next era, run a lightweight independent shadow subscription
that records per-coin message counts, last-message age, and snapshot/event
identity without writing into the production tape. Then the next collapse has
a direct discriminator:

- production and shadow collapse together: upstream venue/market regime;
- production alone collapses: collector/process/path defect;
- counters rise but tape does not: writer/storage defect.

The current log already supplies the production-side counters. The missing
piece is the independent observation and a prospective content-liveness status
in the day verifier. Freeze that status rule before judging the next untouched
day. This monitoring work need not delay starting the new era if it is active
at the boundary; it must be present before the era is interpreted.

## Executed checks at `157c6b2`

| surface | result |
|---|---:|
| host-load join selftest | 10/10 PASS |
| exact repaired 08-25 join | 143 / 665 / 663 / 2, r=+0.0332696 |
| tape-density selftest | 7/7 PASS |
| 08-31 density report, as of review | 184 windows/coin; 352 invisible-thin coin-windows |
| exact 06:32:56–10:41:07 counter/tape reconciliation | 50,717 vs 50,599 (99.77%) |
| selected raw gzip decompression | row-count anomaly reproduced |
| collector source SHA vs prior reviewed `c03c6c2` | identical `e257967d...dd36d5` |
| live service | active/running, PID 3687786, `NRestarts=0` |
| `git diff --check` | PASS |

Because `collect_pm.py` is byte-identical to the prior reviewed tip, repeating
the full eight-gate package would add no evidence; those exact bytes already
passed it. This round executes the changed instrument and the new critical
claim instead.

## Final release path

1. Close COL-R3 with a distinct 10/10 identity and narrow era/runbook seam.
2. Start the independent shadow observation at the same boundary.
3. Deploy the controlled era; do not wait for another historical-cause theory.
4. Mark the next day PASS/FAIL/UNRESOLVED using gap quality plus the
   prospectively frozen content-liveness evidence.

No further mechanism story is required before step 3. One narrow re-review of
the identity/boundary closure should be the final pre-deploy review.
