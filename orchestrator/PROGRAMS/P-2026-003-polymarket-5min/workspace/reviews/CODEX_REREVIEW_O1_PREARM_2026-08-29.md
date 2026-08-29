# Codex O1 pre-arm re-review — 2026-08-29

**Review tip:** `727130e837bfb6c74f17ff6593d4b7db8fb1a163`  
**Closure commit:** `71097a57a426f21148beff505cc376900376f815`  
**Prior finding:** `CODEX_O1_FINAL_PREARM_REVIEW_2026-08-29.md`

## Decision

**HOLD MAINTAINED — DO NOT ARM OR EXECUTE O1 YET.**

`71097a5` correctly repairs the two defects in the repository runbook: the
operating date is now 2026-08-30, the old runbook is visibly superseded, and
the successful path restarts and verifies the new process before emitting and
appending its era stamp. The collector package itself remains green.

Two executed pre-arm findings remain, however. One is in the installed
operating trigger, outside the committed runbook. The other is an identity
hole in the new postflight checker. Either can recreate the class this review
is intended to prevent, so the boundary is not yet safe to arm.

## Executed evidence

At 2026-08-29T22:57Z, against the exact review tip:

| Check | Result |
|---|---|
| `o1_boundary_preflight.py --selftest` | **16/16 pass** |
| real `--pre-arm` | **PASS**, held v3_1 SHA `c0a52d33...`, reviewed HEAD v4 SHA `5b718a15...`, unit active at PID 1048, no conflicting era row |
| v4 behavior suite at `O1_REF=6786a02` | **10/10 pass** |
| real O1 producer -> DAY_BAR_V2 seam at `O1_REF=6786a02` | **7/7 pass** |
| real `--post-restart 1048` before boundary | nonzero refusal, correctly says it is before the boundary |
| v3_1 abort restore ref `6786a02^` | SHA `c0a52d33...`, exactly the live held bytes |

The green checks establish that the intended O1 behavior and the ordinary
producer/consumer seam did not regress. They do not exercise either remaining
failure shape below.

## O1-RB3 — the installed 23:55 wake still carries the superseded order

The live transient unit `co-prep-wake.service` still contains this executable
operator instruction:

> `checkout collect_pm.py from HEAD, era stamp, restart pm-collector-clob`

That is the old stamp-before-restart order rejected by O1-RB2. At 23:55Z this
unit is scheduled to inject that instruction into `pm-co`, while the new
runbook says restart first and stamp last. A corrected file in git does not
supersede an already-installed operating trigger. Following the wake text can
write an era row for a PID that does not yet exist without ever invoking the
new checker.

**Required closure:** cancel or replace the installed wake before it fires.
The installed replacement must name the 2026-08-30 runbook and require
`--selftest`, `--pre-arm`, restore/restart, `--post-restart OLD_PID`, and only
then append the emitted JSON. Re-read the installed unit/timer payload as the
verification artifact. Removing the stale wake without replacement is also
safe if the coordinator executes the reviewed runbook directly; leaving the
old payload installed is not.

## O1-RB4 — postflight matches vocabulary, not event identity

`observe_collector_start()` selects a line with:

```python
if '"collector_start"' not in ln:
    continue
```

and `check_post_restart()` never asserts
`start_row["event"] == "collector_start"`. It also relies on the observer,
rather than the checker that emits the stamp, for the lower timestamp bound.
This violates the repository rule to verify identity rather than vocabulary.

The following known-bads were executed against the committed checker:

| supplied row | current result |
|---|---|
| correct pid/version/time, `event` absent | **ACCEPTED** |
| correct pid/version/time, `event="heartbeat"`, `note="collector_start"` | **ACCEPTED end-to-end by observer + checker** |
| `event="collector_start"` but `recv_ns` five seconds before the boundary, passed to the checker | **ACCEPTED** |

The live collector currently places `pid` only on its real `collector_start`
row, so this is a structural fail-open rather than evidence that today's
ledger already contains a false row. That distinction does not make the
checker certifying: its job is to refuse a wrong-shape ledger, including after
partial writes, corruption, or future schema additions.

**Required closure:** parse rows and require exact
`row.get("event") == "collector_start"`; in `check_post_restart()` independently
require an integer `recv_ns >= BOUNDARY_EPOCH * 1e9` (and retain the exact
PID/version checks). Add known-bad falsifiers for missing event, wrong event
whose other field contains the vocabulary, and a pre-boundary start row. Each
must refuse; the existing exact positive row must still pass.

## Scope and admission safeguard

This hold is on O1 arming only. It does not reopen the already-reviewed v4
collector behavior, DAY_BAR_V2 producer seam, fragment diagnostic, model
freeze, or strategy baselines. `QR_CANCEL_HOLD_X_SKEW` remains the required
queue-realistic strategy baseline and `QR_SKEW_ONLY` remains the comparator.

The frozen DAY_BAR_V2 document still contains its historical 08-29 O1 date.
The R-251/R-276 ruling and the new operating runbook supersede that operational
date: **08-29 is entirely v3_1 and must not count as a post-O1 forward day even
if its day-quality predicate passes.** The first possible complete post-O1
day remains 08-30, conditional on a successful stamped deployment and the
close-time day gate. This should be carried in-band before the five-day
admission decision so a generic `race_accrual_eligible` field cannot admit
08-29 by itself.

## Release condition

Re-review can be narrow. At a committed tip:

1. prove O1-RB3 closed by reading the installed timer/service payload;
2. execute the three O1-RB4 known-bads and the exact positive control;
3. rerun the full preflight selftest and the real `--pre-arm` check;
4. confirm the live tree remains v3_1, committed collector remains the exact
   reviewed v4 bytes, and the unit remains active until the boundary.

Only an explicit subsequent **HOLD RELEASED** filing clears the deployment.
