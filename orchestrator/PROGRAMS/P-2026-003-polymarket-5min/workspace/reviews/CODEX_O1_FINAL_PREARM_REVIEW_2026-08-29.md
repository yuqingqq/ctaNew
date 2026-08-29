# Codex O1 final pre-arm review — 2026-08-29

**Exact reviewed tip:** `edf4da11161637ba9909fd8c24730d93aff64460`

**Operational surface:**
`live/pm_research/plans/O1_DEPLOY_RUNBOOK_2026-08-29.md`, latest commit
`6c59b75` (original runbook `cb85ebd`).

**Review time:** `2026-08-29T22:49:59Z`, before the scheduled 23:55Z arming
for the postponed `2026-08-30T00:00:00Z` boundary.

## Verdict

**ADVERSE O1 FINDING — HOLD ARMING. DO NOT EXECUTE THE CURRENT RUNBOOK.**

The collector package remains the reviewed O1 package and the live hold is
correct. The operational instructions are not: the authoritative runbook still
hard-codes the superseded Aug 29 boundary in four decision-bearing places, and
its stamp sequence requires a new process PID before that process is started.
Following it literally would write false era provenance or force the operator
to improvise at the boundary.

This is a runbook/provenance blocker, not a new collector-code finding. Do not
restore or restart the collector until a corrected, executable runbook is
committed and re-reviewed. The standing pre-arm adverse-finding rule governs
whether closure before 23:55Z can retain tonight's boundary or whether the
deployment postpones; the reviewer does not waive that ruling.

## Finding O1-RB1 — the postponed date never reached the runbook

R-251 postponed deployment to `2026-08-30T00:00:00Z`; R-276 re-armed that
exact boundary and R-305 explicitly kept it green. The runbook still says:

```text
line 1   O1 boundary deploy — 2026-08-29T00:00:00Z
line 22  "boundary_utc": "2026-08-29T00:00:00Z"
line 34  restart gap is inside 08-29's P1 budget
line 48  08-29 is judged under DAY_BAR_V2
```

Those are not historical prose. Line 22 is the immutable era-stamp value the
operator is told to append; lines 34 and 48 decide which day owns the restart
gap and the first post-O1 quality read. Executing them on Aug 30 would:

- label the new collector era as beginning a full day early;
- place pre-deploy Aug 29 rows inside the declared v4 era;
- attribute the actual Aug 30 restart gap to the wrong day;
- point structural/day-bar verification at a day produced entirely by v3_1.

`data/pm_5min/collector_runs.jsonl` is currently absent, so the first row will
define the PM era ledger. There is no prior row that could make a wrong initial
boundary harmless.

## Finding O1-RB2 — the stamp asks for a PID that cannot exist

The runbook orders:

```text
2b append era stamp containing "pid": <new pid>
2c systemctl --user restart pm-collector-clob.service
```

The new process does not exist until 2c. The operator therefore cannot execute
2b as written. Guessing the PID, retaining the old PID, omitting it, or appending
the row later without saying that the sequence changed all break the artifact's
claimed provenance.

This also weakens the abort semantics. The current design writes a success-like
era row first and supersedes it if restart fails. If the executable sequence is
changed to restart-before-append, the attempted/aborted transition must still
be recorded explicitly; silently writing nothing after an attempted code
transition would lose the event.

## Controls that passed

The deployment inputs themselves are correctly staged:

| Check | Observed |
|---|---|
| live working-tree collector | `clob_v3_1`, SHA-256 `c0a52d3337022db3ad6686ae95a242b0f4800d067c919c6aadf74d1735d62203` |
| `6786a02^` control | same `c0a52d33…` bytes |
| committed O1 collector | `clob_v4`, SHA-256 `5b718a15501549c5c39c1a11d7dc9f8c22f755eef64ffc866d0a285831953409` |
| `6786a02` O1 bytes | same `5b718a15…` bytes |
| collector unit | active, PID 1048, running from `/home/yuqing/ctaNew` |
| pre-arm timers | scheduled at 23:55:00Z and 23:55:05Z |

Thus the deliberate v3_1 hold has not drifted, HEAD still carries the reviewed
v4 package, and the timer exists. These positive controls isolate the blocker
to the instructions that join them at the boundary.

## Required closure

Before arming:

1. Make the target `2026-08-30T00:00:00Z` explicit everywhere the runbook
   carries a boundary, era, restart-gap day, or first day-bar target. Rename the
   file or clearly supersede the old dated file so an operator cannot choose
   between two apparent authorities.
2. Specify an executable stamp/restart order. At minimum it must capture the
   declared boundary/restart instant, restore the exact v4 bytes, restart,
   obtain and validate the **new** PID, assert the live process is v4, and only
   then commit a row whose PID and timing semantics are truthful.
3. State the failure path for each point: restore failure, restart failure,
   unchanged PID, non-v4 heartbeat, and stamp-append failure. An attempted
   transition must remain visible through an immutable aborted/superseding row;
   it must not disappear because the success row moved later.
4. Add a preflight that refuses the old boundary, old/same PID, inactive unit,
   wrong collector version, or an already-existing conflicting era row. Keep a
   positive control for the exact Aug 30/new-PID/v4 shape.
5. Re-run the existing O1 behavioral and real producer→day-bar seam controls;
   this finding changes no collector code, but the deployment must still point
   at the already-cleared bytes and consumer contract.

No model, strategy, threshold, or queue assumption is implicated. Until this
closes, Aug 30 cannot safely become the first untouched post-O1 forward day,
and the five-day validation clock must not start from a falsely stamped era.
