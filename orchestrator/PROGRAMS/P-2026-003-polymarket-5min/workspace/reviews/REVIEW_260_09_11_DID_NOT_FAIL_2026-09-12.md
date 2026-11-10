# REVIEW 260 — 09-11 did not fail; it was never evaluated. And the monotone signal is real, in a different field.

REV round 224. Filed 2026-09-12T01:11:59Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE ANSWER, IN ONE LINE

**09-11 did not fail. It is an unevaluated day, and my 90.9% was a miscount.**
The corrected rate is **10 of 10**. But the more important finding is the one
you asked me to look for and I did not expect to find: **there IS something
monotone, and it is in a field no gate reads.**

## 1. Why 09-11 reads as a failure, and why that reading is wrong

Every day's verdict is written **twice**, and I read the wrong one.

| day | as_of_utc | hours_elapsed | day_closed | race_eligible |
|---|---|---|---|---|
| 2026-09-09 | 09-09T00:06:38 | 0.111 | False | False | ← placeholder |
| 2026-09-09 | **09-10T00:06:02** | **24.0** | **True** | **True** | ← real, supersedes |
| 2026-09-10 | 09-10T00:06:35 | 0.11 | False | False | ← placeholder |
| 2026-09-10 | **09-11T00:06:02** | **24.0** | **True** | **True** | ← real |
| **2026-09-11** | **09-11T00:06:36** | **0.11** | **False** | **False** | ← **placeholder, and the ONLY one** |

`da-midnight-verify` writes a placeholder for day D at 00:06 **on** day D — 6.6
minutes in, `day_closed: false`, all bars trivially false — and supersedes it at
00:06 **on day D+1** with the real 24-hour verdict. Files matching
`da_dayverdict_20260911*`: **1**. Positive control, same query for 09-10: **2**.

**The 09-11 placeholder itself records zero damage**: `lost_seconds: 0`,
`coin_level_gap_intervals: 0`, `hours_with_a_gap: 0`,
`gap_affected_pct_COIN_LEVEL: 0.0`, on both coins. Nothing went wrong in the
6.6 minutes it measured. It fails P1/P2/P3 because there was no day yet.

## 2. Why the real verdict is missing — determinate, not inferred

`da-midnight-verify.service` **ran at 2026-09-12T00:06:00 and refused**:

    Active: failed (Result: exit-code) ... status=7
    DEPLOY_DRIFT (REFUSE_TIER_DRIFT), rc 7: the bytes this unit would execute
    are NOT the bytes it was deployed at. Nothing ran.
    sha256sum: WARNING: 1 computed checksum did NOT match
    FIX: re-run live/pm_research/da_deploy_midnight.sh.
         This night is recovered by days_needing_verdict

So the missing 09-11 verdict — and the missing 09-12 placeholder, from the same
run — are one event: **a deployment-drift guard refusing to execute code that
changed after it was deployed.** Tonight's editing of the lane moved a file the
unit's checksum covers. **The guard did its job**, it names its own fix, and
the night is recoverable by design.

**So the cause is neither of your two branches.** It is not transient-random
and it is not monotone-degrading. It is a one-off, self-inflicted by our own
work tonight, correctly refused, and reversible. It carries **no information
about the collector at all** — and the collector is demonstrably alive:
`collector_health.jsonl` newest sample 2026-09-12T01:02:37Z on a 60-second
cadence.

## 3. The arithmetic, corrected — and my 0.6% was wrong twice

| | REVIEW 259 §B.4 | corrected |
|---|---|---|
| eligible days | 10 of 11 = **90.9%** | **10 of 10 = 100%** (09-01..09-10; 09-11 pending) |
| P(`INSUFFICIENT_EVIDENCE`) at that rate | 0.0062 | 0.0000 |

That is the pessimistic-reading half of your question answered: **the rate was
understated.** But the honest statement is *not* "the margin is better than
2.73 days", because a point estimate from ten clean days is not precise:

    0 failures in 10 observations
    95% one-sided LOWER bound on the per-day pass rate p = 0.741
    at p = 0.741, P(INSUFFICIENT_EVIDENCE) = 0.285

**So my 0.0062 was wrong twice: wrong `p`, and quoted as though a rate from
eleven days pinned the risk.** The defensible statement today is: *no day-quality
failure has been observed since 09-01, and the evidence does not exclude a
per-day failure rate as high as ~26%, at which the band fails more than a
quarter of the time.* The ETH-book hazard from REVIEW 259 §B.4 multiplies into
that, unchanged.

## 4. **THE MONOTONE SIGNAL IS REAL — and no gate reads it**

You asked whether anything is becoming more common. Counted from the gap
ledger by UTC day:

| day | disconnects | **loop_stalls** | **cumulative stalled time** |
|---|---|---|---|
| 09-06 | 15 | 15 | 23.7 s |
| 09-07 | 49 | 55 | 182.7 s |
| 09-08 | 69 | 57 | 134.0 s |
| 09-09 | 69 | **42** | **32.7 s** |
| 09-10 | 66 | **121** | **157.5 s** |
| 09-11 | 82 | **208** | **336.6 s** |
| 09-12 (1.2 h) | 0 | 7 | 4.9 s |

**`loop_stall` count and cumulative stalled time both rise monotonically across
09-09 → 09-10 → 09-11: 42 → 121 → 208, and 32.7 s → 157.5 s → 336.6 s, a
tenfold rise in two days.** Disconnects are flat to mildly rising (49 → 82) and
show no such trend.

Severity per stall is **not** rising: p50 lag 543 → 604 → 997 ms, and the worst
single stall was on 09-07 (79,480 ms), not on 09-11 (12,289 ms). So this is
**more frequent stalls of similar size**, not worse ones. In absolute terms it
is still small — 336.6 s is 0.39% of a day.

**And it is invisible to every day-quality predicate.** `loop_stall` carries
`recv_ns`, `collector_version`, `event`, `lag_ms` — and **no `window_start`**,
which is the field the gap accounting keys on. That is the same property that
made it the wrong liveness signal in REVIEW 259 §A, now showing its other face:
**a class of degradation that accumulates without ever touching a gate.** The
day verdicts read 10 of 10 clean while this rose tenfold underneath them.

**I am not inferring a mechanism.** I do not know what `loop_stall` means
operationally, whether the rise is load, market count, host contention or
something else, and nothing I read determines it. 09-12 is 1.2 hours old with 7
stalls — on a linear extrapolation that lands between 09-10 and 09-11, so there
is no sign of reversion yet, but one partial day is not a trend point.

**What I would do with it:** measure it before it matters. `lag_ms` is already
recorded, so a per-day stall-count and total-stalled-seconds series is free,
and it should be a reported diagnostic beside the day verdict — not a gate,
since nobody knows the threshold, but a number that is looked at. If the trend
continues through the validation band it will be the first thing anyone asks
about, and having the series start before the band is worth more than
reconstructing it after.

## 5. What I excluded

Read: `da_dayverdict_*.json` under `data/pm_5min/derived/`; the
`da-midnight-verify` unit state and journal; `collector_gaps.jsonl`;
`collector_health.jsonl`. **Not read**: the collector's source, archive lag,
per-day market counts, host metrics, or any P-2026-002 artifact. So I can say
the stall series is rising and that no gate reads it; **I cannot attribute it**,
and I am not going to.

## 6. Owed

- REVIEW 259's open items stand: `score_is_evidence_permitted` still `None`;
  the `PLAN_ENUMERATION_UNPARSEABLE` refusal still ambiguous; the freeze
  checker's falsifier still red (REVIEW 258 §4).
- **New and operational:** `da_deploy_midnight.sh` needs re-running, or
  tonight's 09-11 verdict stays missing and 09-12's placeholder never lands.
  Not mine to run; flagged.
- The full-day 576-file Identity run; `limits[2]`; REVIEW 247 Part 1's
  resolver; 246 and 245 items.
- Standing: 09-11's real verdict, when the deploy is repaired and it exists.
