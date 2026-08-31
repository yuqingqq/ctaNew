# Codex re-review — clob_v5 boundary package — 2026-08-31

**Review tip:** `038a1b201ff231b8520f3fe27f61934710f3d075`  
**Closure commits:** `bc854d3`, `b6d6f96`, `e519416`  
**Prior review:** `CODEX_PREARM_REVIEW_V5_0700_2026-08-31.md`
(`edc81f5`)  
**Collector candidate:** `7aa952058385f06672e5c1008414a7a837dc053c`,
SHA-256 `1c5291aa6d66ceef0c4a724ea7a1e9fa5128d65d1b69034df5638c0136e98ad5`

## Decision

**LIVE DEPLOYMENT HOLD MAINTAINED. No restart boundary is released by this
filing.**

The R4 transition/abort/rollback repair is now mechanically sound on its
declared paths, and the six narrow false-accepts from the first filing now
close. The complete requirements stated in that filing are not yet met:
four independently executed false-accepts remain in the committed package.

The live service remains safe and untouched on clob_v4, PID `3687786`, with
`NRestarts=0`, no app-v5 drop-in, and no v5 timer armed. Coordinator register
R-343 also records the deployment instant as pending USER and the previously
refused 07:00Z instant as not reusable. This review did not arm or restart the
collector.

## Closures verified

### R1 narrow closure

Pre-arm and armed checks now refuse at `boundary + 1 second`. The exact
executed known-bad that previously passed now names the late-arm cause.

### R2 narrow closure

Counter evidence is anchored to the byte offset printed by `--armed`; two
post-boundary lines are required; PONG and market-message totals must advance;
unit-active and the clob_v5 gap-ledger tail are consumed by the same checker;
and `100 PING / 4 PONG` refuses. The old pre-boundary/stale-line route no longer
supplies success.

### R3 narrow closure

`app-v5x` and separated/non-adjacent flag tokens refuse. The checker now parses
the `argv[]` segment rather than performing the original raw substring test.

### R4 transition state machine

This is materially improved and passes end to end through the real DA
consumer:

- an aborted pre-stamp attempt contributes no era and later days remain v4;
- the success emitter declares `transitioned:true` and DA admits a later
  complete v5 day;
- the rollback emitter derives its boundary from the restored v4 process's
  own start row, closes the named v5 transition, preserves both real
  boundaries on the mixed day, and returns later days to inadmissible v4;
- missing, double, stale, zero-width, and undeclared states in the committed
  DA suite refuse;
- retry after an aborted attempt or verified rollback is representable.

### R5 narrow closure

The runbook checker no longer keys on the wrong step number. A stale
`3. **At 06:00:00Z**` instruction and a runbook with no operative `At` step
both refuse.

## Remaining blockers

### V5-RR1 — postflight still accepts a transition one hour late

The first filing required both sides of the boundary:

> Postflight must also bind the declaring start to the ruled execution window
> (the runbook already gives a two-minute maximum).

The repair added the pre-arm upper bound but left postflight with only
`recv_ns >= BOUNDARY_EPOCH`. Executed at this tip:

```
now_epoch = boundary + 3600
collector_start.recv_ns = boundary + 3600
check_post_restart(...)
=> ACCEPTED; emitted boundary_utc = 2026-08-31T07:00:00Z
```

That receipt claims an era for the preceding hour during which the observed
v5 process did not exist. Require the declaring start inside the runbook's
predeclared execution window (currently two minutes) and bind postflight time
to the same deadline. Add controls immediately inside and outside the limit.

### V5-RR2 — installed command identity still does not identify the command

`exec_start_has_flag()` proves only that the adjacent option/value pair occurs
somewhere in `argv[]`. It does not require the reviewed interpreter, collector
script, unique mode option, or absence of a conflicting later mode.

Executed known-bad:

```
argv[]=/x/python3 /tmp/not_collector.py --heartbeat-mode app-v5
check_pre_arm(..., expect_flag=True)
=> ACCEPTED
```

This is the exact-command portion of prior R3, not a new scope expansion. Parse
and compare the complete expected argv vector. Refuse a wrong interpreter,
wrong script, duplicate/conflicting mode, extra command, and embedded token;
use the same exact identity for armed, post-restart, and rollback verification.

### V5-RR3 — impossible counter histories pass the health gate

The checker enforces only the upper side of
`unresolved = app_ping - app_pong`. It does not require non-negative
unresolved totals or monotone cumulative counters.

Executed known-bad over an advancing timestamp/message interval:

```
first: ping=3, pong=3, msgs=1000
last:  ping=1, pong=8, msgs=5000
check_counters(...)
=> ACCEPTED
```

For one process's cumulative counters, PING cannot decrease and PONG cannot
exceed PING. Require integer/non-negative values, monotone PING and PONG,
PING advancement over the interval, and `0 <= ping - pong <= declared_bound`
at the evaluated endpoint. The bound of two should also be justified against
the production collector's multiple concurrent market-heartbeat tasks; it is
currently asserted as “one in-flight” despite the counters being aggregated
across connections.

### V5-RR4 — a broken transition chain can mint an admissible era

The DA state machine validates role flags and rollback linkage, but an ordinary
`transitioned:true` row is not required to supersede the currently open era.

Executed ledger:

```
clob_v4 supersedes clob_v3_1 @ 2026-08-30T05:30:00Z
clob_v5 supersedes clob_v3_1 @ 2026-08-31T07:00:00Z, transitioned=true
```

The second row breaks the chain—it should supersede clob_v4—yet
`day_era_admission('20260901')` returns clob_v5, pure, and admissible.

Require every effective transition's `supersedes` to equal the currently open
era, with the rollback checks obeying the same single chain. The preflight and
DA consumers should share or equivalently enforce this contract; otherwise
pre-arm can accept a ledger that the nightly consumer interprets differently.
Add missing/wrong `supersedes`, out-of-order transition, and exact legacy-row
identity controls.

## Executed evidence

| Surface | Result |
|---|---:|
| HEAD/origin identity and clean tree at pin | `038a1b2`, PASS |
| live service | active clob_v4, PID 3687786, unarmed |
| v5 boundary selftest | 47/47 PASS |
| DA forward-day selftest | 108/108 PASS |
| live `--pre-arm` | PASS while safely unarmed |
| collector selftest | 17/17 PASS |
| v5 fake-socket behavior | 12/12 PASS |
| legacy v4 behavior | 10/10 PASS |
| producer to DAY_BAR_V2 seam | 7/7 PASS |
| compile / whitespace | PASS / PASS |
| original six false-accept classes | CLOSED |
| success emitter through DA | PASS |
| aborted attempt through DA | PASS |
| rollback emitter through DA | PASS |
| post-restart at boundary + 1 hour | **ACCEPTED — BLOCKER** |
| wrong script with exact flag pair | **ACCEPTED — BLOCKER** |
| decreasing PING / PONG greater than PING | **ACCEPTED — BLOCKER** |
| wrong transition supersession chain | **ACCEPTED — BLOCKER** |

## Minimum next re-review

1. Close V5-RR1 through V5-RR4 with the exact four executed known-bads above,
   plus adjacent positive controls.
2. Re-run the 47/108 suites and the real emitter→DA success, abort, rollback,
   and retry seams.
3. Run `--pre-arm` against the still-live, unarmed v4 service.
4. Only after code review releases: record a new USER-ruled future boundary,
   re-point constants/runbook/falsifiers, and request the final boundary check
   before installing any drop-in.

Until then, **HOLD MAINTAINED**.
