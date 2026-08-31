# Codex round-4 re-review — clob_v5 live-deploy package — 2026-08-31

**Reviewed package tip:** `93b8f27aa53ba0c4d124080b792dcd665174604d`
**Filing base:** `8fdbc4b` (R-354 records the clean lapse of the 07:00Z
instant; it changes only `COORDINATION.md`)
**Collector candidate:** `7aa952058385f06672e5c1008414a7a837dc053c`,
SHA-256 `1c5291aa6d66ceef0c4a724ea7a1e9fa5128d65d1b69034df5638c0136e98ad5`
**Prior filing superseded for this decision:**
`CODEX_REREVIEW_V5_BOUNDARY_2026-08-31.md`

## Decision

**clob_v5 application-heartbeat CODE/TEST RELEASE remains intact.**

**v5 LIVE-DEPLOY PACKAGE HOLD MAINTAINED at `93b8f27`.** Do not arm or
restart from this package. The submitted round-3 defects are materially
improved and all submitted headline suites are independently green, but fresh
execution found six release-blocking paths and one retry-operability defect.
Most importantly, the recovery emitter can recreate the exact cross-consumer
divergence the round was meant to close: the preflight walk accepts a duplicated,
out-of-order recovery chain that DA refuses.

The ruled `2026-08-31T07:00:00Z` instant has also lapsed. At `08:26:31Z`, the
real `--pre-arm` refused by name at `+5,191 s`. Nothing was armed, restarted, or
stamped. The live service remained clob_v4, PID `3687786`, `NRestarts=0`, exact
v4 argv, with only `slice.conf` installed. A later release requires a new
USER-ruled future instant and a constants/runbook re-point followed by a narrow
boundary check.

This review targets **v5**. The v4 behavioral suite is used only as the required
no-regression check and because v4 is the rollback state; it is not a proposal
to deploy or continue optimizing v4.

## Blocking findings

### V5-R4-1 — recovery retry reopens cross-consumer disagreement

`check_post_recovery()` has no idempotent already-landed path. I emitted one
valid recovered-v5 + rollback bundle, placed it in the ledger, then invoked the
same checker with the same process declarations. It emitted the same two rows
again.

Appending that second bundle produces this result:

```
current_era_and_open_v5(rows) => ACCEPT ('clob_v4', None)
day_era_admission(..., rows)  => REFUSE: ledger is OUT OF ORDER
                                (07:00 transition after 07:03 rollback)
```

Thus the claimed shared chain semantics are not shared on a real emitter output.
The committed 17-case equivalence set omits retrying a complete recovery bundle,
and the mutation audit does not test sequence composition.

Require chronological ordering in the preflight consumer to match DA. An exact
already-landed two-row recovery bundle must return idempotent success with **no
stdout**; a conflicting or half-landed bundle must refuse. Put this exact
emitter-produced retry through both consumers in the shared seam test.

### V5-R4-2 — both idempotent fast paths still bypass safety legs

The submitted critical says the retry branch no longer returns before every
safety leg. That is true only for some system legs.

Executed post-restart fixture: an exact open ledger receipt plus a supplied
declaration with `event=heartbeat`, `collector_version=clob_v4`, and foreign PID
`999`, sharing only the recorded `recv_ns`, returned `already_stamped=True`.
The return at lines 594–598 precedes the event/version/PID checks at lines
607–636.

Executed rollback fixture: an exact prior rollback receipt returned
`already_stamped=True` while the installed command was the **app-v5** vector.
The return at lines 776–780 precedes the control-v4 command, changed-PID, and
restored-process checks at lines 785–811.

Idempotency must suppress a duplicate append, not suppress current-system and
declaration validation. Move both returns behind the same legs as a first
emission and require exact event, version, PID, strict-int `recv_ns`, current
installed mode, and live-unit identity.

### V5-R4-3 — PID-aware observers exist but are not wired into the real CLI

`observe_collector_start()` now accepts `unit_pid` and correctly filters foreign
collectors. The production paths do not pass it:

- `--post-restart` calls `observe_collector_start(BOUNDARY_EPOCH)`;
- `--post-rollback` does the same.

I placed the valid unit row first and a later foreign row second. The exact CLI
form returned the foreign row; passing `4242` returned the real unit row. This is
not hypothetical: R-351 produced foreign collector rows today. The current
checker then refuses a healthy unit and can route a live v5 process into the
wrong recovery/abort branch.

Wire `obs["main_pid"]` into both calls. The recovery observer is weaker still:
it chooses the first post-boundary row merely declaring clob_v5 and accepts any
positive PID. Bind it to the recorded `V5_PID` as the runbook requires, rather
than reconstructing a production era from any collector process that wrote the
shared gap ledger.

### V5-R4-4 — `--abort-row` can assert “nothing ran” while unstamped v5 is live

`make_abort_row()` consumes only the era ledger and stage. With the era ledger
still on v4 but observations describing an active app-v5 process, it emitted an
`aborted:true` row. That is exactly the stamp-unwritable / live-v5 state in which
the runbook says an abort is untrue and the recovery path is mandatory.

The CLI must prove the post-failure system is restored v4 before emitting an
abort: exact control-v4 installed mode, active new PID, and that PID's own
post-boundary clob_v4 `collector_start`. Ledger silence cannot prove a process
never ran—the recovery path exists precisely because a real v5 transition can
be absent from the ledger.

### V5-R4-5 — an absent `WorkingDirectory` passes a relative-script command

`check_system_safe()` rejects a wrong nonempty `WorkingDirectory` but accepts
the empty value because the condition is `if obs.get("working_dir") and ...`.
The argv names `live/pm_research/collect_pm.py` relatively, so an absent working
directory does not identify the reviewed file and may crash-loop or execute a
different path.

Executed: `working_dir=""` passed. Require exact equality to the repo path;
missing and empty must refuse. The currently installed unit is exact, but the
checker still has a false-accepting identity predicate.

### V5-R4-6 — the recovery runbook promises a path that expires after 600 s

The runbook says that after an unwritable stamp, restore v4 and emit the recovery
bundle “once the ledger is writable.” `check_post_recovery()` calls the common
post checker, which refuses at `boundary + 600 s`. Executed at `+601 s`, a
fully evidenced v5 start and later live-v4 restoration refused before recovery.

If the append target remains unavailable past ten minutes, the package leaves a
real v5 span permanently unstamped with no executable repair path. Recovery is a
historical reconstruction and needs its own timing rule: never before the ruled
instant, strictly bound to the recorded start/restoration declarations, but not
silently governed by the success-stamp deadline. If a late cap is intended, the
runbook must name the terminal remediation that preserves the era.

## Additional retry-operability defect

The armed-time `LOG_OFFSET` is captured before the boundary, while
`observe_heartbeat_lines()` reads every matching line after that offset and
`check_counters()` refuses every pre-boundary sample. On the current first
attempt the still-live pre-v5 binary does not print app counters, so those old
lines do not match. After any rollback, however, control-v4 runs the candidate
binary and **does** print `app_ping=0 app_pong=0`; a later ruled retry will ingest
ordinary pre-boundary v4 heartbeat lines and refuse even if the new v5 process
is healthy.

Anchor the evidence at the verified new process/restart, or filter structured
full-date records to the new process. Do not make a documented future retry
depend on the accidental log vocabulary of the currently resident old binary.

## Submitted closures and regressions re-executed

All commands below ran in a detached Git worktree at exact `93b8f27`; an archive
without `.git` correctly refused candidate-commit resolution and was not counted.

| Surface | Independent result |
|---|---:|
| v5 preflight selftest | `144/144` PASS |
| DA forward-day selftest | `136/136` PASS |
| shared chain equivalence | `17/17` PASS on its declared fixtures |
| preflight mutation audit controls A/B/C/D | PASS |
| preflight mutation sites | 95 sites: 85 assertion-killed, 10 crash-killed, 0 survivors |
| reusable DA mutation harness selftest | `9/9` PASS |
| collector selftest | `17/17` PASS |
| v5 fake-socket behavior | `12/12` PASS |
| v4 rollback/no-regression behavior | `10/10` PASS |
| O1 producer → day-bar seam | `7/7` PASS |
| contamination-record generator | `7/7` PASS |
| committed contamination record shape | 21 files / 3 windows / 302,941 estimated lost lines / unrecoverable |
| compile / whitespace | PASS / PASS |
| real late `--pre-arm` | REFUSED as designed; 07:00Z lapsed cleanly |

The zero-survivor result is real for the declared raise sites, but it is not a
proof of branch ordering, observer-to-checker wiring, idempotent sequence
composition, or cross-consumer completeness. The executed failures above are
examples of each blind spot.

## Minimum next re-review

1. Close V5-R4-1 through R4-6 with the exact executed fixtures above and add the
   retry-offset case.
2. Extend the one-fixture/two-consumer seam with duplicate complete recovery,
   half recovery, and out-of-order transition cases.
3. Demonstrate the **real CLI wiring**: unit-PID selection for post-restart and
   rollback, recorded-V5-PID selection for recovery, and actual-process proof
   before abort emission.
4. Re-run the 144/136 suites, mutation audit, chain equivalence, collector
   17/12/10, and 7-check day-bar seam.
5. After code release only: obtain a new USER-ruled future instant, re-point the
   constants/runbook/falsifiers, run a narrow final boundary check, and only
   then install a drop-in.

Until those closures are committed and independently executed, **HOLD
MAINTAINED**.
