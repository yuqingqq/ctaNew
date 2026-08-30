# O1 boundary deploy runbook — 2026-08-30 (boundary amended to 05:30:00Z)

> **RULED AMENDMENT 2 — 2026-08-30T02:05Z (USER, via the coordinator):**
> boundary moved 02:30:00Z → **05:30:00Z** because the reviewing seat
> (pm-codex) is quota-blocked and the amended instant may not arm without
> its narrow re-review clearing first (register R-334). Nothing else changes.
>
> **RULED AMENDMENT 1 — 2026-08-30T02:00Z (USER, option 2, via the coordinator):**
> the 00:00:00Z instant passed UNEXECUTED (the 23:55Z wake's two coalesced
> unordered timers left the instruction unsubmitted — register R-333). The
> boundary moved to a mid-day instant. Consequences, ruled with it:
> **2026-08-30 is MIXED-ERA and inadmissible as a post-O1 day under any
> field; the earliest complete post-O1 day is 2026-08-31.** Every other
> element below — sequence, failure paths, verification, abort semantics —
> is UNCHANGED; `o1_boundary_preflight.py` carries the amended instant as
> its compile-time constant and remains the authority an operator executes.

**SUPERSEDES `O1_DEPLOY_RUNBOOK_2026-08-29.md` in full** (that file carried the
pre-postponement boundary in four decision-bearing places and an inexecutable
stamp order — CODEX_O1_FINAL_PREARM_REVIEW_2026-08-29.md findings O1-RB1/RB2).
This file is the ONLY operating authority; the old file carries a SUPERSEDED
banner and must not be executed.

**Authority:** user ruling R-232 (O1 package a–d approved). Boundary postponed
to 2026-08-30 by R-251/R-276/R-305; instant amended to
**2026-08-30T05:30:00Z** by R-333 (option-2 mid-day ruling) + R-334 (+3h).
Patch committed UNDEPLOYED at `6786a02` (v4 sha `5b718a15…`); working tree
deliberately held at `clob_v3_1` (sha `c0a52d33…`) because
`pm-collector-clob.service` has `Restart=always` and runs from the live tree.
**Do not clean or restore `collect_pm.py` before the boundary.**

**Every decision below is checked by `o1_boundary_preflight.py`** (its
selftest: positive controls + refusals for stale boundary, drifted hold, wrong
HEAD bytes, inactive unit, conflicting era row, unchanged PID, non-v4
declaration, foreign PID, early execution). The era stamp is EMITTED by the
instrument from verified observations, never hand-written.

## Sequence (coordinator, at the boundary)

1. **~05:25Z prep (immediately pre-boundary):**
   ```
   python3 live/pm_research/o1_boundary_preflight.py --selftest   # must pass
   python3 live/pm_research/o1_boundary_preflight.py --pre-arm    # must pass
   ```
   Record `OLD_PID` from the pre-arm output. Any refusal → do NOT proceed;
   register the refusal; the boundary postpones (no improvisation at 05:30:00Z).

2. **At 05:30:00Z (order closes O1-RB2 — restart FIRST, stamp LAST):**
   a. `git checkout -- live/pm_research/collect_pm.py` (restores reviewed v4
      from HEAD).
   b. `systemctl --user restart pm-collector-clob.service`.
   c. Wait for the new process to declare itself (collector_start audit row,
      typically <15 s), then:
      ```
      python3 live/pm_research/o1_boundary_preflight.py --post-restart OLD_PID
      ```
      This REFUSES on: unchanged PID, tree not v4, unit inactive, no
      post-boundary `collector_start` row, a row declaring any version but
      `clob_v4`, or a declaring pid ≠ the unit MainPID. On success it PRINTS
      the era-stamp JSON — pid and `collector_start_recv_ns` taken from the
      VERIFIED new process, `stamp_written_ns` at emission, and an explicit
      `stamp_order` field stating the stamp was written after the restart it
      describes.
   d. Append that emitted line VERBATIM:
      ```
      python3 live/pm_research/o1_boundary_preflight.py --post-restart OLD_PID \
        >> data/pm_5min/collector_runs.jsonl
      ```
      (Run c and d as one invocation or re-run within the same minute; the
      instrument re-verifies at each call.)
   e. Re-read the appended row (`tail -1`), confirm `pid` equals the live
      MainPID and `boundary_utc` is `2026-08-30T05:30:00Z`.

3. **05:31–05:35Z verify:** unit active; fresh gap-ledger rows carry
   `"collector_version":"clob_v4"`; the ~seconds restart gap at 05:30 lands
   inside MIXED-ERA 08-30 (inadmissible as a post-O1 day regardless — no P1
   consequence rides on it). The prices collector is NOT
   touched — out of package scope.

## Failure paths (each leaves the attempt VISIBLE — nothing silent)

| Failure | Action |
|---|---|
| 2a checkout fails | Nothing restarted, v3_1 still live. Do NOT restart. Register the failure; boundary postpones. No era row (no code transition was attempted). |
| 2b restart fails / unit inactive | A transition WAS attempted: append `{"collector_schema_version":"clob_v4","boundary_utc":"2026-08-30T05:30:00Z","aborted":true,"stage":"restart_failed","stamp_written_ns":<now>}`, restore the hold (`git show 6786a02^:live/pm_research/collect_pm.py > live/pm_research/collect_pm.py`), start the unit, verify a `clob_v3_1` collector_start row, register the abort. |
| 2c postflight REFUSES (unchanged PID / non-v4 declaration / foreign pid / no row within 2 min) | Same as restart-failure path: aborted-row append (stage = the refusal's first clause), restore v3_1, restart, verify v3_1 live, register. The refusal text goes in the register verbatim. |
| 2d stamp append fails (write error) | v4 is LIVE but unstamped — the one state that may not persist. Retry the append (the instrument re-verifies). If it cannot be written within 5 min, restore v3_1 per the abort path (an unstampable era must not run) and register. |
| Retry after an aborted row | Permitted: the preflight accepts `aborted:true` rows and refuses only a LIVE conflicting `clob_v4` row (supersession, rule 13 — never edit the earlier row). |

## Structural verification (post-deploy; R-182 — never a throughput A/B)

- **O1a:** PING_TIMEOUT duration distribution collapses toward
  ~ping_interval + 1.3 s (median gap 11.3 s → ~4.3 s), within-cause.
- **O1b:** retry spacing under persistent faults shows the exponential ladder
  and jitter (`retry in Xs (fail #N)` log lines).
- **O1c:** SUBSCRIBE_UNCONFIRMED appears as its own cause when it fires; its
  absence is legitimate only alongside normal first-message latencies.
- **O1d:** never-connected gaps record from scope start (longer, honest
  durations for that class; cross-boundary comparisons treat pre-boundary as
  understated).
- **Day bar: 2026-08-31 is the first day judged post-O1** (08-30 is
  MIXED-ERA and never admissible, R-333), under DAY_BAR_V2
  (P1/P2/P3, `dfa0977`) as amended 06:04Z by the doc's §3 single-band reading
  (`368345b`): expect **~55–80 s/hr** (both models); below ~45 = O1b's
  reconnect residual also fell (unmodelled); above ~120 (P1 FAIL) = the
  detection-lag diagnosis was wrong, NOT the fix underperforming. The restart
  gap at 05:30:00Z lands inside mixed-era 08-30. **Neither 08-29 (pure v3_1)
  nor 08-30 (mixed-era) is ever a post-O1 day; no generic eligibility field
  may admit either.** The five-day validation clock starts at the first
  complete post-O1 UTC day: **08-31 is day one if it completes.**
