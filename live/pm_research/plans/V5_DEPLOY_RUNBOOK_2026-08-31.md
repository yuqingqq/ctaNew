# clob_v5 deploy runbook — 2026-08-31T07:00:00Z

**Authority:** USER ruling R-340 (mid-day boundary, recorded before
execution); code/test release `CODEX_REVIEW_V5_HEARTBEAT_REPAIR_2026-08-31.md`
(`df424de`, candidate `7aa9520`, collector sha `1c5291aa…`). Live deployment
additionally gated on the Codex pre-arm review of THIS runbook + instrument +
DA's era-admission guard.

**What deploys:** a keepalive CONTRACT change only — application text
`PING`/`PONG` at 10 s replacing RFC control-Pong liveness (98.22% of v4's
disconnects were local PING_TIMEOUT against the wrong contract). No
working-tree swap: the candidate is INERT on disk; activation is the
installed unit command gaining `--heartbeat-mode app-v5`. No row-stamping
change; no model, threshold, or policy moves.

**Every decision below is checked by `v5_boundary_preflight.py`** (selftest:
positive controls + refusals for stale instant, non-candidate bytes, missing
or premature flag, missing/duplicate era rows, unchanged PID, wrong-MODE
declaration, wrong event identity, float/pre-boundary timestamps, silent or
one-sided heartbeat counters, and a stale runbook body). The era stamp is
EMITTED by the instrument from verified observations, never hand-written.

## Sequence (coordinator, at the boundary)

1. **~06:50Z prep:**
   ```
   python3 live/pm_research/v5_boundary_preflight.py --selftest   # must pass
   python3 live/pm_research/v5_boundary_preflight.py --pre-arm    # must pass
   ```
   Record `OLD_PID`. Any refusal → do NOT proceed; register it; the boundary
   amends (no improvisation at 07:00:00Z).

2. **~06:55Z arm the mode (inert until restart):** write the drop-in
   `~/.config/systemd/user/pm-collector-clob.service.d/override.conf`:
   ```
   [Service]
   ExecStart=
   ExecStart=/home/yuqing/pricer-sol/venv/bin/python3 live/pm_research/collect_pm.py --heartbeat-mode app-v5
   ```
   then `systemctl --user daemon-reload`, then:
   ```
   python3 live/pm_research/v5_boundary_preflight.py --armed      # must pass
   ```
   Record `LOG_OFFSET` from the output — it anchors the counter evidence to
   the post-arming log region (V5-0700-R2).
   `--armed` re-runs every pre-arm check AND requires the flag in the
   ExecStart **read back from systemd** — never trusted from what was
   written. (The running v4 process is untouched by the reload.)

3. **At 07:00:00Z (restart FIRST, stamp LAST):**
   a. `systemctl --user restart pm-collector-clob.service`.
   b. Wait for the new process to declare itself (collector_start row,
      typically <15 s), then:
      ```
      python3 live/pm_research/v5_boundary_preflight.py --post-restart OLD_PID \
        >> data/pm_5min/collector_runs.jsonl
      ```
      REFUSES on: unchanged PID, non-candidate bytes, flag lost, a
      `clob_v4`-declaring process (the flag not taking effect), wrong event
      identity, float or pre-boundary `recv_ns`, foreign pid. On success the
      emitted stamp carries `boundary_utc 2026-08-31T07:00:00Z`, the verified
      pid, `clob_v5 supersedes clob_v4`, and the R-340 authority line.
   c. `tail -1` the ledger; confirm the live MainPID and
      `boundary_utc` = `2026-08-31T07:00:00Z` on one row.

4. **07:01–07:05Z verify:**
   ```
   python3 live/pm_research/v5_boundary_preflight.py --verify-counters \
     --log-offset LOG_OFFSET
   ```
   Requires TWO heartbeat lines after the armed-time offset (~2 min), each
   post-boundary on its OWN stamp; pongs and market rows must ADVANCE over
   the interval; unresolved PINGs (ping−pong) bounded at 2 — one answered
   ping followed by silence is the v4 failure shape one layer up and
   REFUSES. The same invocation checks unit-active, the newest gap-ledger
   row declaring `"collector_version":"clob_v5"`, and market advance — the
   seams live IN the instrument. The prices collector is NOT touched.

## Failure paths (each leaves the attempt VISIBLE — nothing silent)

| Failure | Action |
|---|---|
| 2 armed-check refuses | Nothing restarted; remove the drop-in, daemon-reload, verify `--pre-arm` passes again; register; the boundary amends. No era row (no transition attempted). |
| 3a restart fails / unit inactive | Append `{"collector_schema_version":"clob_v5","boundary_utc":"2026-08-31T07:00:00Z","aborted":true,"stage":"restart_failed","stamp_written_ns":<now>}`; remove the drop-in, daemon-reload, start the unit (boots v4 default), verify a `clob_v4` collector_start row; register the abort. |
| 3b postflight REFUSES (unchanged PID / v4-declaring / flag lost / no row in 2 min) | Same abort path: aborted-row append (stage = the refusal's first clause), drop-in removed, v4 restarted and verified, refusal text registered verbatim. |
| 3c stamp append fails | v5 LIVE but unstamped — may not persist. Retry the append (safe: a FAILED append left no row, so re-emission does not duplicate; on an EXACT already-landed row the emitter returns idempotent already-stamped success with no output). If unwritable within 5 min: restore v4 (drop-in removed, daemon-reload, restart), then once the ledger is writable append the RECOVERY BUNDLE (DA `8bfcc9b` shape, in order): row 1 the reconstructed v5 transition `{clob_v5, transitioned:true, supersedes:clob_v4, boundary_utc:<ruled instant>, recovered:true, stage:"stamp_unwritable_recovery", collector_start_recv_ns:<the v5 process's own>}`; row 2 the standard rollback receipt closing it. NEVER a bare aborted row (it cannot truthfully encode the v5 span that ran) and NEVER a rollback alone (a rollback of a row that does not exist). A half-written bundle fails LOUD: both consumers refuse an unclosed recovered transition. Days resting on a recovered boundary are NON-ACCRUING (DA conservative default; USER ruling pending). Register. |
| 4 counters refuse (post-stamp) | The contract is still wrong: remove the drop-in, daemon-reload, restart (boots v4), then `--post-rollback V5_PID --stage counters_refused >> collector_runs.jsonl` — the emitted ROLLBACK RECEIPT (clob_v4, supersedes clob_v5, rollback:true) CLOSES the live v5 row after verifying the restoration from the restored process's own collector_start (V5-0700-R4: a bare aborted row after a real transition leaves the v5 era open forever). Register; candidate returns to review. |
| Retry after an aborted row | Permitted: the preflight accepts `aborted:true` rows and refuses only a LIVE conflicting `clob_v5` row (supersession, rule 13). Pre-stamp failures append `aborted:true` rows (no transition happened — they must NOT enter era spans); post-stamp failures use the ROLLBACK RECEIPT (a transition happened — both real boundaries preserved). DA's era consumer distinguishes the two. |

## Era and admission consequences

- 08-30 (v3_1→v4) and **08-31 (v4→v5) are both MIXED-ERA and never
  admissible as forward days**; no generic eligibility field may admit them
  (DA's era-admission guard enforces this in code — a deploy blocker).
- **08-31's close-time verdict is still computed and PRESERVED** — honest
  v4-storm evidence, not suppressed by the boundary (Codex's guidance
  honoured under the USER's mid-day ruling).
- **Day one of the five-day validation clock: 2026-09-01 if it completes.**
- Structural v5 verification over the coming hours, within-cause (R-182):
  PING_TIMEOUT/`APP_HEARTBEAT_TIMEOUT` disconnect-rate collapse vs the
  measured v4 baseline (~43/hour BTC), counters advancing, no queue pauses.
