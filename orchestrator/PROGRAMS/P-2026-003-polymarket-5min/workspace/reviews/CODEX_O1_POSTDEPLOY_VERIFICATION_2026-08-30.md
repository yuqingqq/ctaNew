# Codex O1 post-deploy verification — 2026-08-30

**Deployment register tip:** `bda1c968451739d8e1f161aa18b6a8c2660e68d5`  
**Released closure:** `2150ebab99007d01738e0a406e2604e3ae73b481`  
**Release filing:** `bd8b965ae433797437dbc3118fce56c4044a68c0`  
**Ruled boundary:** `2026-08-30T05:30:00Z`

## Decision

**OPERATIONAL TRANSITION ACCEPTED. NO IMMEDIATE O1-RELEVANT ADVERSE
FINDING.**

The one released v3_1-to-v4 transition executed in the required order and its
runtime artifacts reconcile. This is an operational/deployment verification,
not evidence that O1 improves disconnect duration, not a day-bar result, and
not permission to admit mixed-era 2026-08-30 into any model or strategy
evaluation.

## Executed evidence

Verified directly from the service, committed/live bytes, collector audit
ledger, era ledger, journal, and advancing output files at
2026-08-30T05:30–05:31Z:

| Requirement | Direct evidence | Verdict |
|---|---|---|
| exact package live | working-tree collector and HEAD collector both SHA-256 `5b718a15501549c5c39c1a11d7dc9f8c22f755eef64ffc866d0a285831953409` | **PASS** |
| old process stopped cleanly | audit row: `event=collector_stop`, `collector_version=clob_v3_1`, PID 1048, zero force-cancelled markets | **PASS** |
| new process identity | audit row at `recv_ns=1788067802114726542`: exact `event=collector_start`, `collector_version=clob_v4`, PID 3687786 | **PASS** |
| systemd identity/health | active/running unit, MainPID 3687786, process start 05:30:01Z, command is `live/pm_research/collect_pm.py` | **PASS** |
| emitted stamp identity | sole era row names boundary 05:30, PID 3687786, and the exact collector-start integer above | **PASS** |
| restart-first/stamp-last | `stamp_written_ns=1788067815886120604`, 13.771394062 s after the verified start row; stamp states the ordering | **PASS** |
| no fork/abort | era ledger has exactly one row and it is non-aborted; no second live-v4 stamp | **PASS** |
| nominal-to-actual-start interval | no market or resolution rows have `recv_ns` in `[1788067800000000000, 1788067802114726542)`; the only ledger event there is the truthful v3_1 stop row | **PASS** |
| immediate data continuity | `markets.jsonl` advanced after restart; collector log reports v4, all seven coins active, and 26,005 messages in its first status minute | **PASS** |
| unrelated collector | prices collector was not restarted and continued writing | **PASS** |

The actual per-event change anchor is the stamp's
`collector_start_recv_ns=1788067802114726542`; the nominal UTC boundary remains
the operating instant. This distinction creates no hidden market-row overlap in
the observed transition interval, and 2026-08-30 is excluded in full anyway.

## What remains unproven

- O1a–O1d structural effects require within-cause observations over the coming
  hours. Absence of an immediate disconnect is not a positive result.
- DAY_BAR_V2 cannot judge a post-O1 complete day before 2026-08-31 closes.
- Neither 2026-08-29 nor mixed-era 2026-08-30 may be admitted by a generic
  eligibility field.
- No model, threshold, strategy arm, multiplicity count, or forward-race
  status changes because the collector was deployed.

The post-boundary research sequence therefore remains: keep Iteration 011 dark
through its identity and non-fit reviews; run the two frozen arms once only
after explicit release; keep integrated economics separate and retain
`QR_CANCEL_HOLD_X_SKEW` as the queue-realistic baseline with `QR_SKEW_ONLY` as
the required comparator.
