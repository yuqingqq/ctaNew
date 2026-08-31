# Codex re-review — collector root cause and readiness — 2026-08-31

**Exact review tip:** `c03c6c2d1e8a6d961b19626a77dd92e76b839775`

**Delta reviewed:** `b04ff13..c03c6c2` (R-366, the new
`pm_host_load_join.py`, and the corrected `collect_pm.py` diagnosis).

**Live-mutation statement:** the production unit, drop-in, process, era
ledger, and tapes were inspected read-only. Nothing was armed, restarted, or
stamped. Test executions were placed in `research.slice`.

## Verdict — NOT GOOD TO GO

R-366 is the right correction. It explicitly withdraws R-365, independently
reproduces the retained sysstat evidence, and restores the honest status:
**the dominant 2026-08-25 BTC disconnect cause is unknown**. The revised
source comment also closes COL-R4: 10/10 is now described as removal of a
measured amplifier, not as a root-cause repair.

That correction does not make a collector restart ready. R-366 explicitly
leaves COL-R2 and COL-R3 open, and execution confirms they remain open:

- the research-agent scopes still run under `app.slice`, outside the aggregate
  `research.slice` fence;
- the collector's `Slice=collectors.slice` remains only in an installed,
  untracked drop-in rather than the repository unit;
- HEAD's 10/10 control-Ping behavior still declares `clob_v4`, while the live
  3/3 process also declares `clob_v4`; there is no event-distinguishable 10/10
  era or reviewed boundary package.

Do not restart the live collector from HEAD. Keep app-v5 held. The 10/10
rollback can be a sensible next controlled era without first discovering the
historical cause, but it needs a new identity and complete boundary/rollback
evidence before deployment.

## Findings

### COL-R1 closure — R-365 is correctly withdrawn

The new tool reproduces the important negative evidence from the retained
artifact:

| check | independent result |
|---|---:|
| `pm_host_load_join.py --selftest` | 6/6 PASS |
| 08-25 00:00–06:00 BTC gap starts | 210 |
| corresponding unweighted sysstat idle mean | 89.80% |
| 08-25 BTC gap starts in first hour | 43 |
| 08-25 busy-vs-gap Pearson value printed by tool | +0.039 |
| 08-30 mean busy / joined events / Pearson value | 2.73% / 751 / -0.134 |

The timing also remains dispositive against the filed R-365 story: the break
is present from the first hour, whereas the cited heavy-run record is roughly
twenty hours later. R-149, not an OOM-killer inference, remains the authority
for the later I5 deaths.

This evidence does not prove that sub-ten-minute scheduler pressure never
matters. It does establish that R-365's claimed dominant cause was not
supported and that **root cause unknown** is the correct status.

### COL-R4 closure — the source no longer claims a false root cause

`connect_keepalive_kwargs()` now separates three statements correctly:

1. the 3/3 O1a change amplified total post-break gaps;
2. 10/10 rolls back that known-bad amplifier;
3. neither observation explains the underlying 08-25 BTC-only break.

The old “THE MECHANISM IS PING COUNT, NOT LOAD” authority is gone. This
finding is closed.

### COL-R2 remains open — the resource controls are not comprehensive or durable

All six running Claude/Codex `run-r*.scope` units sampled in this review report
`Slice=app.slice`, not `research.slice`. Commands enter the aggregate fence
only when the caller explicitly uses `systemd-run --slice=research.slice`.

The live collector benefits from the parent `collectors.slice` values
(`CPUWeight=500`, `MemoryLow=2 GiB`), but its only `Slice=collectors.slice`
directive is still:

```text
~/.config/systemd/user/pm-collector-clob.service.d/slice.conf
```

The tracked `live/pm_research/ops/pm-collector-clob.service` has no `Slice=`.
A repository-based reinstall can therefore discard the protection. R-366
acknowledges this finding but does not close it.

### COL-R3 remains open — 10/10 is not an era-safe candidate

The live service remains PID `3687786`, started
`2026-08-30T05:30:01Z`, with `NRestarts=0` and no heartbeat-mode argument. It
therefore continues the pre-HEAD 3/3 control-Ping code resident since that
start. HEAD returns 10/10 for `control-v4`, but still declares:

```python
COLLECTOR_VERSION = "clob_v4"
```

A restart would change the measurement regime without changing its event
identity, and the existing era machinery cannot validly express
`clob_v4 -> clob_v4`. No 10/10-specific deploy/runbook/rollback package was
added in the reviewed delta.

A read-only tail inspection also found repeated live BTC `PING_TIMEOUT`
disconnect/reconnect pairs while the unit itself stayed active. That is a
spot observation, not a complete-day result, but it confirms that process
liveness (`NRestarts=0`) is not feed health.

### HJ-R1 — the new join's conclusion survives, but its evaluated population is inexact

`pm_host_load_join.py` parses 143 endpoint rows from `sa25`, then constructs
only 142 intervals by starting at the second row. The first row describes the
00:00–00:10 interval, but the tool drops that interval rather than anchoring it
at UTC day start. At the reviewed ledger bytes:

```text
sar rows:              143
evaluated cells:       142
BTC gap starts total:  665
joined gap starts:     654
before first endpoint:   9
after last endpoint:     2
```

Thus R-366's “143 samples” label is not the correlation population actually
used, and the decisive first ten minutes are absent from the correlation.
The final unsampled interval is legitimately unevaluable, but it must be
reported as such rather than silently disappearing.

The function named `disconnects()` also selects `gap_closed` records and bins
their `gap_start_ns` (last market-message time), not `event=disconnect` and its
`recv_ns`. On 08-25 the 665 records pair one-for-one and none cross a ten-minute
bin, so this does not reverse the reported sign. It is nevertheless the wrong
declared decision timestamp and can change bins on another day. The six-check
selftest exercises Pearson arithmetic and archive absence, but contains no
positive/known-bad control for the central interval join.

Required closure: define the population as actual disconnect rows at exact
integer `recv_ns`; attach the first sar endpoint to `[day_start, first_end)`;
report the right-edge unobserved population and as-of endpoint explicitly;
refuse malformed governed rows; and add a synthetic join control that catches
an event shifted to the wrong interval. Recompute the table and sample labels.

This is an instrument-correctness finding, not evidence for the withdrawn
load diagnosis. The current real-data result remains strongly inconsistent
with host-wide average load as the dominant break mechanism.

## Regression execution at `c03c6c2`

| surface | result |
|---|---:|
| consolidated v5 deploy gates | all 8 PASS |
| consolidated runner falsifier | injected ninth gate FAILED by name; falsifier fired |
| collector selftest | 17 checks PASS |
| v5 heartbeat behavior | 27/27 PASS |
| deadline mutation falsifier | wrong deadline killed by 2 deadline checks |
| preflight selftest | 230/230 PASS |
| DA day-verifier selftest | 164/164 PASS |
| chain equivalence | 38/38 agree |
| differential fuzz | 1,288 ledgers, 0 disagreements |
| preflight mutation audit | 0 survivors |
| host-load instrument selftest | 6/6 PASS, subject to HJ-R1 |
| `git diff --check` | PASS |

The new commit changes no collector execution logic beyond comments and adds
the standalone evidence tool, so the existing gate results are stable. They
do not close COL-R2, COL-R3, or establish post-change data quality.

## Minimum next step

1. Repair HJ-R1 so the retained negative evidence is exact and auditable.
2. Track the collector slice/drop-in and enforce or audit the research-slice
   boundary for heavy children.
3. Give the 10/10 rollback a new collector identity and submit its complete
   deploy/rollback package at a future ruled boundary.
4. Add prospective process/socket/scheduler evidence for that era. Judge feed
   health only after a complete admissible day; do not promote a root-cause
   story from partial or proxy telemetry.
