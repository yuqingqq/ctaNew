# Codex pre-arm review — clob_v5 07:00Z boundary — 2026-08-31

**Review tip:** `8702171ae151f2e6b0b3ef776625df4dd49bfb58`  
**Origin tip at execution:** same  
**Collector candidate:** `7aa952058385f06672e5c1008414a7a837dc053c`  
**Collector SHA-256:** `1c5291aa6d66ceef0c4a724ea7a1e9fa5128d65d1b69034df5638c0136e98ad5`  
**Scheduled boundary reviewed:** `2026-08-31T07:00:00Z` (`1788159600`)  
**Scoped inputs:** `4e1133c` (DA era admission), `a1080bd` (v5 deploy
instrument/runbook), `f4a7aa1` (R-340 instant), and coordinator register
`8702171`.

## Decision

**LIVE DEPLOYMENT HOLD MAINTAINED — REFUSE the scheduled 07:00Z restart.**

Do not install the v5 drop-in, arm the unit, or restart at this boundary.  The
code/test release of the inert clob_v5 collector remains intact; this refusal
is for the live deploy instrument, rollback accounting, and boundary receipt.

The current state is safe and unchanged: `pm-collector-clob.service` is active
on PID `3687786`, started `2026-08-30T05:30:01Z`, with `NRestarts=0`; installed
`ExecStart` has no v5 flag and no v5 override.  The `co-v5-wake0648.timer` is
only a tmux wake/instruction at 06:48Z, not the restart itself.  I did not arm
or restart the collector.

The package's own positive tests pass, but six independently executed
known-bad states are accepted.  Four affect the success path directly and the
era defect makes the declared rollback path unsafe.

## Blocking findings

### V5-0700-R1 — exact boundary is not enforced

`check_boundary_current()` refuses a pre-arm run only at
`boundary_epoch + 3600` (`v5_boundary_preflight.py:137-140`).  Therefore both
`--pre-arm` and `--armed` accept from 07:00:00Z through 07:59:59Z.  Postflight
only requires `collector_start.recv_ns >= boundary`; a restart substantially
after 07:00Z can still emit a stamp claiming the 07:00Z boundary.

Executed known-bad:

```
check_pre_arm(good_observation_at_BOUNDARY_plus_1_second, False)
=> ACCEPTED
```

This defeats the runbook's explicit “boundary amends; no improvisation” rule
if the wake, arm, or restart is delayed.  Pre-arm/armed must refuse at or after
the boundary.  Postflight must also bind the declaring start to the ruled
execution window (the runbook already gives a two-minute maximum), otherwise
the stamped boundary is not the observed transition time.  Add positive and
late known-bad controls at both seams.

### V5-0700-R2 — post-deploy heartbeat evidence is neither post-boundary nor process-bound

`observe_heartbeat_line(since_epoch)` never uses `since_epoch`
(`v5_boundary_preflight.py:110-125`).  It searches the last 400 KB for any
text matching the two counters.  The log line carries only an `HH:MM:SSZ`
clock and the observer does not parse even that.  It records no date, byte
offset, PID, collector version, or start-row identity.

Executed known-bad:

```
temporary log containing only:
2026-08-30T00:00:00Z heartbeat app_ping=9 app_pong=9
observe_heartbeat_line(2026-08-31T07:00:00Z)
=> ACCEPTED {'app_ping': 9, 'app_pong': 9}
```

`check_counters()` then requires only that both totals are positive
(`v5_boundary_preflight.py:249-256`).  It does not show that PONGs are keeping
up with PINGs or advancing after the new process starts:

```
check_counters({'app_ping': 100, 'app_pong': 1})
=> ACCEPTED
```

The `--verify-counters` code path also does not perform the runbook's adjacent
claims that the unit is active, fresh gap rows declare `clob_v5`, and market
rows advance (`V5_DEPLOY_RUNBOOK_2026-08-31.md:64-72`).

Use structured, full-time evidence bound to the verified new PID/version (or
a pre-arm log offset plus structured post-start record).  Validate progress
over an interval and a declared reconciliation invariant that rejects a
material unresolved-PING population, while allowing only explicitly bounded
in-flight heartbeats.  Wire the unit/start-row/market-advance seams into the
instrument or state clearly which separate executable checker owns them.

### V5-0700-R3 — installed command identity is a substring check

Both pre-arm and postflight define installed mode as
`"--heartbeat-mode app-v5" in ExecStart` (`v5_boundary_preflight.py:172` and
`:190`).  That is vocabulary matching, not the exact argv identity required
to show which mode systemd will execute.

Executed known-bad:

```
ExecStart = "python3 collect_pm.py --note=--heartbeat-mode app-v5"
check_pre_arm(..., expect_flag=True)
=> ACCEPTED
```

Parse systemd's installed command and require the exact reviewed interpreter,
script, and one exact `--heartbeat-mode app-v5` argument, with no conflicting
or malformed heartbeat-mode arguments.  Add embedded-token, duplicate-mode,
wrong-script, and extra-command known-bads and use the same check post-restart.

### V5-0700-R4 — rollback receipts corrupt the DA era state

The DA guard is correct for the current successful-transition ledger: it adds
era admission as the third eligibility conjunct, applies it to the whole day
and every coin, refuses an unknown era, and correctly makes 08-29/08-30
ineligible in the superseding receipts.

It is not correct for the failure rows required by this runbook.
`era_spans()` treats every `collector_runs.jsonl` row as an effective
transition and does not inspect `aborted` (`da_forward_day_verify.py:348-370`).
The runbook tells restart-failed and postflight-refused paths to append an
aborted clob_v5 row at the ruled boundary (`V5_DEPLOY_RUNBOOK...:78-83`).

Executed known-bad with the current v4 row followed by the runbook's
`aborted:true, stage:restart_failed` v5 row:

```
day_era_admission('20260901', ledger)
=> eras_touched=['clob_v5'], era_pure=true,
   race_admissible_by_era=true
```

That falsely turns a failed/no-transition attempt into an admissible v5 era.
The later failure path is also incomplete: if the v5 stamp succeeds and the
counter check then refuses, restoring v4 and appending an aborted v5 row does
not close the already non-aborted v5 row and does not stamp the verified v4
restoration.  DA would continue to call later days v5, and a future preflight
would continue to see a live conflicting v5 row.

Define an executable transition/rollback receipt state machine.  An attempt
that never transitioned must not enter `era_spans`; an attempt that did run v5
and then rolled back must preserve both real boundaries and a verified v4
restoration.  DA must consume only effective transitions, refuse ambiguous
attempt state, and carry positive/known-bad tests for restart-failed,
postflight-refused, post-stamp counter failure, verified rollback, and retry.

### V5-0700-R5 — the runbook mutation control checks the wrong step

`check_runbook_consistency()` looks for a line starting `2. **At `, while the
real restart instruction is step `3. **At 07:00:00Z`
(`v5_boundary_preflight.py:271`; runbook line 48).  A fixture with the correct
instant repeated elsewhere but a stale real step-3 time is accepted:

```
3. **At 06:00:00Z (restart FIRST, stamp LAST):**
=> ACCEPTED
```

Bind the exact operative step (preferably the reviewed runbook bytes or a
parsed single authoritative instant), and add a mutation of the actual step
3.  This is not merely cosmetic because the operational instruction is what
the coordinator executes.

## Executed evidence

| Surface | Result at review tip |
|---|---:|
| HEAD equals `origin/mm-research`; initial tree clean | PASS |
| collector SHA on disk and at HEAD | exact `1c5291aa...` |
| live unit identity | active v4, PID 3687786, no v5 flag |
| installed wake timer/service payload | 06:48Z, tmux instruction only |
| v5 preflight bundled selftest | 25/25 PASS |
| live `--pre-arm` observation | PASS, 12,769 s before boundary |
| DA forward-day selftest | 91/91 PASS |
| collector selftest | 17/17 PASS |
| v5 fake-socket behavior | 12/12 PASS |
| legacy v4 behavior at candidate | 10/10 PASS |
| real producer to DAY_BAR_V2 seam | 7/7 PASS |
| compile / `git diff --check` | PASS |
| late pre-arm known-bad | **ACCEPTED — BLOCKER** |
| embedded/malformed flag known-bad | **ACCEPTED — BLOCKER** |
| stale counter-line known-bad | **ACCEPTED — BLOCKER** |
| 100 PING / 1 PONG known-bad | **ACCEPTED — BLOCKER** |
| stale operative runbook step known-bad | **ACCEPTED — BLOCKER** |
| aborted restart becomes v5 era known-bad | **ACCEPTED — BLOCKER** |

The green bundled suites establish that the collector candidate and the
normal-shape preflight still work.  They do not override the executed false
accepts, because those states are absent from the bundled falsifier set.

## Minimum re-review package

1. Close V5-0700-R1 through R5 with red-first controls for the exact six
   accepted known-bads above.
2. Execute the repaired pre-arm against the still-live v4 service before any
   drop-in is installed.
3. Demonstrate the armed checker on a non-live systemd fixture or reversible
   test unit; do not use the production restart as the first seam test.
4. Execute a full fake transition and each rollback stage through the same
   receipt consumer DA uses, proving later-day era admission.
5. Present a new ruled future boundary.  The refused 07:00Z instant must not be
   reused after this hold.

Until those closures are committed and re-executed, **HOLD MAINTAINED**.
