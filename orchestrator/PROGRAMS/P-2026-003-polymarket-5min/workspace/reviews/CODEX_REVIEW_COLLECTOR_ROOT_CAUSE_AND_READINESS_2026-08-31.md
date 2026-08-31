# Codex review — collector root cause and readiness — 2026-08-31

**Exact review tip:** `e7437cab3b380519afea4afba4789982e4c124c2`

**Scope:** commits after the round-6 review (`2d34fac..e7437ca`), the live
collector and installed resource controls, the retained gap/log evidence, and
the host's retained 2026-08-25 sysstat archive.

**Live-mutation statement:** inspection and test execution were read-only with
respect to the production service. Tests were run inside `research.slice`.
Nothing was armed, restarted, stamped, or written to production ledgers/tapes.

## Verdict — NOT GOOD TO GO

The collector test surface is materially better and all submitted gates pass.
The round-6 deadline false coverage is genuinely closed. The 10/10 control-Ping
revert is directionally safer than retaining 3/3.

The claimed dominant root cause, however, is **not found**. R-365 says retained
load telemetry does not exist and attributes the 08-25 break to research
compute contention. `/var/log/sysstat/sa25` does exist, and joining it to the
BTC disconnects contradicts that attribution at its available ten-minute
resolution. The operational isolation also remains bypassable, and the 10/10
change has no distinct collector era or deploy package.

Do not restart the live collector from HEAD yet. Keep v5 held.

## Critical findings

### COL-R1 — R-365's “cause found” conclusion fails against retained host telemetry

R-365 explicitly says there is no load time-series. The host retains
`/var/log/sysstat/sa25` (664 KiB, complete through 23:50 UTC), including CPU,
memory, run-queue, paging, I/O-wait, and context-switch samples every ten
minutes.

I joined all 143 CPU samples to the 663 BTC disconnects whose `recv_ns` falls
inside those sample intervals. Pearson correlations between disconnect count
and the claimed resource-pressure variables are near zero:

| ten-minute host metric | correlation with BTC disconnect count |
|---|---:|
| aggregate busy CPU | +0.033 |
| nice CPU | +0.145 |
| load average (1 min) | +0.059 |
| available memory | -0.025 |
| kernel page scans | +0.026 |
| context switches | -0.051 |

Concrete controls are more decisive than the aggregate correlations:

- 00:00–00:10 had **9 BTC disconnects** while the host was **98.36% idle**,
  had **22.18 GiB available**, load average **0.25**, and zero blocked tasks.
- 02:00–02:10 had **11 disconnects** while the host was **98.31% idle**,
  had **22.18 GiB available**, load average **0.13**, and zero blocked tasks.
- The highest busy intervals were 01:40–02:00, dominated by I/O wait. They
  carried 9 and then 2 disconnects, not a corresponding maximum.

The break was already present at 00:00. The contemporaneous first-hand record
R-365 cites for heavy I5 runs is at roughly 19:40/20:20, many hours later.
More importantly, R-149 already records the confirmed explanation for those
“externally killed” I5 controls: the inherited `.bashrc` virtual-memory/file
ulimits. Calling those kills an OOM-killer signature in R-365 contradicts the
programme's own earlier correction. The separately proven box-wide memory
exhaustion occurred on 08-26 around 03:55, not at the start of the 08-25 BTC
regime.

This does not prove compute pressure can never hurt the collector; sub-ten-
minute bursts and per-process scheduling are not fully identified by sysstat.
It does prove that the filed evidence cannot support “dominant cause found.”
The correct current status is **root cause unknown; compute contention remains
one hypothesis**. A controlled prospective test or process-level scheduler/
socket telemetry is required.

### COL-R2 — the resource mitigation does not contain all research work

`research.slice` is effective only for commands explicitly launched into it.
Every currently running Claude/Codex `run-*.scope` inspected in this review is
under `app.slice`, not `research.slice`. Heavy commands spawned from those
sessions can therefore bypass the 12-core/60%-memory aggregate fence that the
root-cause claim relies on.

The live collector is in `collectors.slice` with CPUWeight 500 and MemoryLow
2 GiB, but its `Slice=collectors.slice` setting lives only in the installed
untracked `~/.config/systemd/.../slice.conf`. The tracked
`live/pm_research/ops/pm-collector-clob.service` contains no Slice directive.
A reinstall from the repository removes the protection.

Thus even if compute contention were later confirmed, the repository does not
yet make the remedy durable or comprehensive. Track the collector/research
slice units and drop-ins, and enforce or audit that every heavy research child
actually enters `research.slice` (or move research off the collector host).

### COL-R3 — the 10/10 source change is not an era-safe deployable candidate

The live unit is still the 3/3 process started at
`2026-08-30T05:30:01Z` (PID `3687786`, `NRestarts=0`). HEAD would use 10/10
after a restart, but still declares `COLLECTOR_VERSION = "clob_v4"`.

That restart would change the heartbeat and gap measurement distribution while
emitting the same event-level version as the running process. The era consumers
also refuse a self-superseding `clob_v4 -> clob_v4` row, so there is currently
no valid receipt that can distinguish the two regimes. No 10/10-specific
boundary instrument or runbook is committed.

The expected quality is not established either: the latest in-band correction
estimates roughly **123 lost seconds/hour versus the 120 bar**, with only two of
five post-break 10/10 days passing. That makes 10/10 a sensible rollback of the
3/3 amplification, not a demonstrated fix.

Before restart, give the candidate a new collector identity (for example a
ruled clob_v4 successor), wire that identity through the collector-start rows,
era admissibility, boundary emitter and runbook, then review the actual
deployment package.

### COL-R4 — HEAD embeds a diagnosis already superseded by the latest filing

`collect_pm.py` lines 198–225 state “THE MECHANISM IS PING COUNT, NOT LOAD” and
describe 10/10 as the fix. R-364 says ping count was the smaller amplifier, and
R-365 says the dominant cause was load. These cannot all be the authority at
HEAD. The production source must carry the corrected, narrower statement:
10/10 removes a measured amplification but does not establish or repair the
dominant 08-25 break.

## What did pass

| Surface | Executed result at `e7437ca` |
|---|---:|
| consolidated v5 gate runner | all 8 gates PASS |
| runner falsifier | injected ninth gate reported FAIL; falsifier fired |
| collector selftest | PASS (17 checks) |
| v5 heartbeat behavior | 27/27 PASS |
| deadline falsifier | hard-coded wrong deadline killed with exact attribution |
| preflight selftest | 230/230 PASS |
| DA day-verifier selftest | 164/164 PASS |
| chain equivalence | 38/38 agree |
| differential fuzz | 1,288 ledgers, 0 disagreements |
| preflight mutation audit | 0 survivors |
| v4 behavior, including 10/10 kwargs | 10/10 PASS |
| source identity / `git diff --check` | exact at HEAD / PASS |

These results establish local code and checker behavior. They do not establish
the historical root cause, host isolation, an era-safe restart, or post-change
data quality.

## Minimum path to readiness

1. Correct R-365 in-band to **hypothesis, not finding**, and retain a script or
   receipt that joins gap events to `sa25`; the existing calculations currently
   live only in prose.
2. Track and verify the resource-control units/drop-ins, and fail a preflight if
   heavy research processes are outside `research.slice` while collection is
   accruing.
3. Add process-level evidence for the next prospective interval: scheduler
   delay/CPU pressure, socket/reader backlog, per-coin disconnects and tape
   density. Prefer a quiet-host or dedicated-core control against an otherwise
   matched interval.
4. Package 10/10 as a new, event-distinguishable collector era with a ruled
   future boundary and complete deploy/rollback receipts.
5. Only after the isolated 10/10 era produces a complete day should P1/P2/P3
   decide whether the collector is actually healthy. Do not infer that result
   from the historical average sitting at the bar.
