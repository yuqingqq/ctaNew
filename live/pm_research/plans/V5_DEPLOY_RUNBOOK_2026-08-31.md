# clob_v5 deploy runbook — 2026-08-31T07:00:00Z

> **THIS INSTANT HAS LAPSED (audit C12).** 07:00:00Z passed unexecuted while
> the reviewer was quota-blocked; `--pre-arm` now refuses by name and every
> emitting mode refuses with it. **Nothing below is executable until a NEW
> instant is USER-ruled and the constants + this body are re-pointed.** The
> times in the sequence are placeholders relative to the ruled instant T:
> prep at T−10 min, arm at T−5 min, deploy at T. Do not read them as clock
> times for a lapsed boundary.

**Authority:** USER ruling R-340 (mid-day boundary, recorded before
execution); code/test release `CODEX_REVIEW_V5_HEARTBEAT_REPAIR_2026-08-31.md`
(`df424de`, candidate `7aa9520`, collector sha `1c5291aa…`). Live deployment
additionally gated on the Codex pre-arm review of THIS runbook + instrument +
DA's era-admission guard.

**What deploys:** a keepalive CONTRACT change only — application text
application heartbeat at 3 s with a 3 s answer deadline, replacing
RFC control-Pong liveness (98.22% of v4's
disconnects were local PING_TIMEOUT against the wrong contract). The cadence
was lowered from the venue-documented 10 s by USER ruling 2026-08-31 so that
worst-case dead-socket blindness (interval + timeout) returns to the ~6 s the
v4 keepalive was tuned to by O1a — detection lag is stamped inside recorded
gap durations, which are the quantity the day bars are denominated in. It is
an EMPIRICALLY TESTED deviation, not a documented one; the residual is
recorded in the candidate source and in DAY_BAR_V2_PREREGISTRATION.md. No
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

**Run every command from `/home/yuqing/ctaNew`.** All paths below are
relative and the shell opens `>>` targets before exec, so a wrong cwd both
fails to find the script and can create a stray file (audit S4).

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
   c. **Record `V5_PID`** (also required by `--post-recovery`) — `systemctl --user show pm-collector-clob.service
      -p MainPID --value` — BEFORE any later step restarts the unit; after a
      rollback restart that value is gone and rows 3c/4 need it.
   d. `tail -1` the ledger; confirm the live MainPID and
      `boundary_utc` = `2026-08-31T07:00:00Z` on one row.

4. **T+3 min to T+6 min verify** (the first heartbeat line appears ~60 s
   after start and the check needs TWO at a 60 s cadence, so anything before
   ~T+2:05 refuses by construction; T+3 gives margin — audit S1):
   ```
   python3 live/pm_research/v5_boundary_preflight.py --verify-counters --log-offset LOG_OFFSET
   # LOG_OFFSET is a SCAN HINT only (any value at or before the restart is
   # safe): counter samples are floored at the VERIFIED NEW PROCESS start,
   # so a pre-restart line refuses on its own stamp rather than being
   # silently included (audit C6 — the code and this runbook previously
   # disagreed three ways about where the offset comes from)
   ```
   Requires TWO heartbeat lines after the armed-time offset (~2 min), each
   post-boundary on its OWN stamp; pongs and market rows must ADVANCE over
   the interval; unresolved PINGs (ping−pong) bounded at 2 — one answered
   ping followed by silence is the v4 failure shape one layer up and
   REFUSES. The same invocation checks unit-active, the newest gap-ledger
   row declaring `"collector_version":"clob_v5"`, and market advance — the
   seams live IN the instrument. The prices collector is NOT touched.

## Stated residual — the armed branch cannot be rehearsed on an inert fixture

`installed_mode()` requires the exact full argv vector AND systemd's `path=`
(the binary actually executed). A fixture unit that does not run the real
collector therefore CANNOT pass `--armed` — verified: an inert `/bin/sleep`
unit refuses on `path=`. So the app-v5 branch of the armed check is first
exercised for real at step 2 of an actual deploy. **This is a deliberate
trade: the property that makes a fixture rehearsal impossible is the same one
that stops a foreign binary from passing as the collector, and it is what made
the earlier fixture attempt dangerous (R-351).**

Why the residual is acceptable rather than merely unavoidable:
* `--armed` runs BEFORE any restart, so a defect there costs an ABORTED
  deploy, never a wrong one — the failure direction is safe.
* The property read and the parser are exercised against the PRODUCTION unit
  on every `--pre-arm` run (currently returning `control-v4`), so only the
  v5-direction return value is unexercised in production, not the plumbing.
* The vector logic itself is pinned by known-bads for a wrong interpreter,
  a foreign script, non-adjacent tokens, an `app-v5x` superset, a trailing
  conflicting `control-v4`, and non-ASCII whitespace.

## Failure paths (each leaves the attempt VISIBLE — nothing silent)

| Failure | Action |
|---|---|
| 2 armed-check refuses | Nothing restarted; remove the drop-in, `daemon-reload`. **If still BEFORE the instant**, re-run `python3 live/pm_research/v5_boundary_preflight.py --pre-arm` and retry the arming. **If the instant has passed**, the deploy is missed: register it and rule a NEW boundary — `--pre-arm` refuses at/after the instant BY DESIGN, so "verify it passes again" is not executable then (runbook-audit finding 7). No era row: no transition was attempted. |
| 3a restart fails / unit inactive | A transition was ATTEMPTED but none ran. Remove the drop-in, `daemon-reload`, start the unit (boots v4 default), confirm a `clob_v4` collector_start row, then `python3 live/pm_research/v5_boundary_preflight.py --abort-row --stage restart_failed >> data/pm_5min/collector_runs.jsonl`. **Never hand-write the row** — the emitter supplies the timestamp and refuses if a transition actually ran (finding 3). |
| 3b postflight REFUSES **and v5 is NOT live** (flag lost, v4-declaring process) | Same as 3a with `--stage postflight_refused`; put the refusal text in the register verbatim (it is now a single `REFUSED:` line, exit 2 — audit C13). |
| 3b(ii) postflight REFUSES with **unchanged PID** | The restart never happened, so the unit is STILL ACTIVE and `systemctl start` is a no-op that produces no new process and no post-boundary `collector_start` — `--abort-row` would then refuse for want of a restoration declaration. Use `systemctl --user restart` (not `start`), confirm a fresh post-boundary `clob_v4` collector_start, then `--abort-row --stage postflight_refused` (audit C5). |
| 3b' postflight REFUSES **while v5 IS live** — **FIRST record `V5_PID` = the current MainPID, BEFORE the restore restart destroys it** (audit C8; if already lost, it is the `pid` field of the post-boundary `clob_v5` `collector_start` row in `data/pm_5min/collector_gaps.jsonl`) (no collector_start within 2 min, start later than boundary+120 s, or emission later than boundary+600 s) | **NOT an abort — v5 is running.** The late-START refusal fires after the new-pid, flag and unit-active legs have passed, so a v5 process is provably live. The late-EMISSION refusal (>600 s) fires FIRST, before any of those legs, so it establishes nothing about liveness on its own — check the unit before choosing this row (audit C4). Restore v4 (drop-in removed, `daemon-reload`, restart), then use the **recovery bundle** in 3c. An `aborted` row here would be untrue and the emitter refuses to write one. |
| 3b(iii) the stamp APPEND itself fails (the `>>` redirect in step 3b) | v5 LIVE but unstamped. Retry the append: a FAILED append left no row, and on an EXACT already-landed row the emitter returns idempotent success with no output. If unwritable within 5 min: restore v4, then once the ledger is writable run `python3 live/pm_research/v5_boundary_preflight.py --post-recovery --v5-pid V5_PID --stage stamp_unwritable_recovery >> data/pm_5min/collector_runs.jsonl` (the recorded `V5_PID` is REQUIRED — reconstructing an era from any collector process that wrote the shared gap ledger is how a foreign row becomes history), which EMITS the two-row bundle (reconstructed v5 transition + the rollback closing it) with both boundaries read from the processes' OWN collector_start rows. Never hand-compose those rows; never a rollback alone (a rollback of a row that does not exist). A half-written bundle fails LOUD: both consumers refuse an unclosed recovered transition. Recovery is a historical reconstruction, so it is NOT governed by the success-stamp deadline: it is permitted from the instant up to +24 h (never earlier), which is why an append target unavailable past ten minutes no longer leaves the era permanently unstampable. Days resting on a recovered boundary are NON-ACCRUING (DA default; USER ruling pending). |
| 4 counters refuse (post-stamp) | The contract is still wrong. Remove the drop-in, `daemon-reload`, restart (boots v4), then `python3 live/pm_research/v5_boundary_preflight.py --post-rollback V5_PID --stage counters_refused >> data/pm_5min/collector_runs.jsonl` — the emitted ROLLBACK RECEIPT closes the live v5 row after verifying the restoration from the restored process's own collector_start. Register; candidate returns to review. |
| A ledger row is PARTIALLY written (short write, disk full mid-append) | The next read refuses by name: `era ledger line N is NOT VALID JSON … repair it before any deploy step`. This is the ONE case where the append-only rule yields: the truncated fragment was never a row. Remove exactly that fragment line (nothing else), re-read to confirm the refusal clears, and register the repair with the removed bytes quoted. Do not re-run any emitter until the read is clean (audit S3). |
| Retry after an aborted row | The ledger PREDICATE permits it (an `aborted` row never enters the era line). **Operationally it is only executable while `now < the ruled instant`** — every real abort happens at/after it, so in practice a retry means ruling a NEW boundary (finding 8). |

## Era and admission consequences

- 08-30 (v3_1→v4) and **08-31 (v4→v5) are both MIXED-ERA and never
  admissible as forward days**; no generic eligibility field may admit them
  (DA's era-admission guard enforces this in code — a deploy blocker).
- **08-31's close-time verdict is still computed and PRESERVED** — honest
  v4-storm evidence, not suppressed by the boundary (Codex's guidance
  honoured under the USER's mid-day ruling).
- **Day one of the five-day validation clock: 2026-09-01 if it completes —
  and ONLY on the success path.** Every failure row above leaves `clob_v4`
  in force, which the era guard rules inadmissible, so 09-01 and every later
  day stay inadmissible until a NEW deploy lands. A recovered boundary is
  additionally non-accruing. The clock does not start by waiting; it starts
  by a stamped, verified transition (runbook-audit finding 10).
- Structural v5 verification over the coming hours, within-cause (R-182):
  PING_TIMEOUT/`APP_HEARTBEAT_TIMEOUT` disconnect-rate collapse vs the
  measured v4 baseline (~43/hour BTC), counters advancing, no queue pauses.
