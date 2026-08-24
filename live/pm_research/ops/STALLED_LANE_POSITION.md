# Does a stalled derivation lane belong in the acceptance boundary?

**OPS position for dispatch D-1b. A position, not a rule** — R-ADMISS says the
selection decision is the coordinator's. Written before the lane is repaired, so
it is not fitted to what the repaired lane happens to show.

## Short answer

**No. A stalled derivation lane is an OPS incident, not a data-admissibility
event, and no window should be excluded for having been derived late.** But it
must be *disclosed*, because a stall silently truncates the evidence base of any
claim made while it is running.

## Why they are not the same thing

| | collector gap | stalled derivation lane |
|---|---|---|
| what is missing | the tape for that interval | derived artifacts |
| recoverable? | **never** — nobody recorded it | **yes** — raw tape intact, rebuild is deterministic |
| is it a fact about the data? | yes | no; a fact about our machines |
| detectable how? | gap ledger row | only as absence |

The whole reason `CLOB_ADMISSIBILITY_PROTOCOL.md` excludes on cause rather than
on seconds-lost is that a gap is **selective**: `SLOW_CONSUMER_1013` is 31/32
BTC and lands on the busiest windows, so the loss is correlated with the thing
being measured. That is MNAR and it is why exclusion is the right move.

A stall has no such property. The pipeline is content-addressed, resumable and
commit-last; re-running it over the same raw shards reproduces the same
artifacts. Nothing about a window's content makes it more or less likely to be
caught by an outage — the selection is on **wall-clock**, not on market state.
Excluding those windows would discard good data for an operational accident, and
would itself introduce a time-correlated hole where none existed.

**This outage is the clean case in point.** 26 h of failure lost exactly zero
data: the raw tape holds 5 UTC days / 20 G and the blocker is a comparison bug
plus an unbound control. The correct repair is to rebuild, not to exclude.

## The one way they genuinely are alike, and it is not about admissibility

**Both are invisible as absence.** Neither announces itself; both are found only
by checking for something that is not there. The collector gap problem was
solved by *recording* gaps, and reading `open_gaps=[]` as "clean" was already
identified as a mistake — the prices lane logged 58 gaps in 11 h while every
spot check read empty. A stalled lane is the same shape of error one level up:
`tier1/` looks like a directory of results whether or not anything committed
today.

That is a **monitoring** conclusion, and it is discharged by `pm_lane_health.py`
`LANE_PROGRESS`, which reads committed receipts rather than partitions.

## The hazard that IS real, and what to do about it

A stall creates a time-correlated hole in *derived* coverage. It biases nothing
if analysis reads committed receipts, and biases a lot if analysis reads "what
is on disk" — the newest days are exactly the ones a stall withholds, so a naive
read is a **recency-truncated** sample that still looks complete. The coordinator
has already ruled the operative half of this ("do not read a day count or a
coverage fact off `tier1/`"). OPS proposes one addition:

> Any result should state its **derivation lag** — last committed day vs last
> closed UTC day — beside its day count. A claim made at lag 0 and the same
> claim made at lag 3 rest on different evidence bases, and the difference is
> invisible in the day count alone.

`pm_lane_health.py` already computes both sides of that (`newest_committed`,
`eligible_uncommitted`), so it is a read, not new machinery.

## Where recoverable would become irrecoverable

The claim "a stall loses nothing" holds only while the raw tape survives it.
Measured 2026-08-23: `data/pm_5min` is 20 G over 5 UTC days ≈ **4 G/day**, with
**1.2 T free (41 % used)**. At that rate the disk is ~300 days from pressure, so
recoverability is not currently time-bounded, and no retention rule is needed
yet. **This stops being true if raw retention is ever capped** — at that point a
stall longer than the retention window converts to permanent loss and *would*
become an admissibility matter. OPS will re-raise if retention changes.

## What OPS asks the coordinator to settle

1. Confirm a stall is **not** an exclusion cause (OPS's position above).
2. Rule on whether derivation lag becomes a **mandatory reporting field** under
   the existing reporting contract, or stays advisory.
3. The `LANE_PROGRESS` grace window (currently 3 h) is an **ops bar** chosen from
   the hourly timer cadence. **Ruling R-6 now answers what this question was
   asking:** it is **Class A — configuration**, freely changeable, *for as long
   as nothing gates on it*. The moment it is read as an admissibility threshold
   it becomes **Class D** and may only be set before the measurement it gates.
   OPS does not intend it as an admissibility bar; this records the trigger that
   would change its character. See `plans/OP_PLANE_PLAN.md` §8a.
