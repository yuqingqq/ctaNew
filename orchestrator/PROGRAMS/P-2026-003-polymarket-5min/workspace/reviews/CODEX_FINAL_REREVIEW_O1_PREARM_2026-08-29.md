# Codex final O1 pre-arm re-review — 2026-08-29

**Review tip:** `9ac0bd187e5dc42afed792c781d84ae0239aee70`  
**Requested closure:** `71097a5`, `727130e`, and `9ac0bd1`  
**Prior filing:** `CODEX_REREVIEW_O1_PREARM_2026-08-29.md` at `97e0010`

## Decision

**HOLD MAINTAINED — DO NOT ARM YET.**

The two findings filed at `97e0010` are substantively closed:

- the obsolete `co-prep-wake` timers are absent, and the installed
  `co-prep-wake30.service` payload names the 2026-08-30 runbook, preflight,
  restart-first order, and verifier-emitted stamp last;
- exact `event == "collector_start"` identity is now enforced in both the
  observer and checker, and the checker independently enforces the boundary
  lower bound.

One narrow structural defect remains in that new timestamp check. It misses
the explicit integer-type part of the prior release condition and admits a
malformed producer row end-to-end. It is a small repair, but this is the
boundary's certifying instrument, so the release remains gated on it.

## Executed evidence at `9ac0bd1`

| Check | Result |
|---|---|
| installed old timers | **absent** |
| installed `co-prep-wake30` payload | **correct**, 23:55:00/05Z, Aug-30 runbook, restart first, emitted stamp last |
| preflight selftest | **19/19 pass** |
| real `--pre-arm` | **PASS**, v3_1 hold intact, reviewed v4 HEAD, unit PID 1048 active, no conflicting era row |
| heartbeat row with `note="collector_start"` | **REFUSES** |
| missing `event` | **REFUSES** |
| pre-boundary `collector_start` | **REFUSES in the checker** |
| finite floating `recv_ns` after boundary | **ACCEPTED end-to-end**, emitted as JSON float |

The exact executed malformed row was otherwise a correct post-boundary start
declaration (event, pid, and version), with:

```json
{"recv_ns": 1.788048005e+18,
 "collector_version": "clob_v4",
 "pid": 4242,
 "event": "collector_start"}
```

`observe_collector_start()` accepts the float comparison. Then
`check_post_restart()` uses:

```python
int(start_row.get("recv_ns") or 0) < BOUNDARY_EPOCH * 10**9
```

which checks a coercible magnitude rather than the producer field's type. The
returned stamp carries the original float in `collector_start_recv_ns`. Thus
the era record no longer preserves the exact nanosecond integer identity it
claims to bind. This is the same distinction as value versus typed identity;
coercion is not validation.

## Final closure required

Before arming:

1. in the checker, require `type(start_row.get("recv_ns")) is int` before the
   boundary comparison (thereby rejecting booleans as well as floats/strings);
2. add and execute a finite post-boundary float known-bad that must refuse;
3. retain the exact integer positive control;
4. rerun the full selftest and real `--pre-arm`, and confirm the installed
   wake payload and live v3_1 hold remain unchanged.

No collector behavior or day-bar rerun is required for this checker-only
repair if the committed collector bytes remain `5b718a15...`. Re-review is
narrow and can release immediately once the committed closure is available.

The first admissible post-O1 complete forward day remains 2026-08-30 only
after a successful, stamped boundary deploy and its close-time gate. The
strategy baselines and frozen model are unchanged.
