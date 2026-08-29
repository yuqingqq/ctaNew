# O1 boundary deploy runbook — 2026-08-29T00:00:00Z

> **SUPERSEDED — DO NOT EXECUTE.** This file predates the R-251/R-276
> postponement and hard-codes the WRONG boundary (08-29) in its era stamp,
> restart-gap attribution, and day-bar target, and its stamp order asks for a
> PID that does not exist until after the restart
> (CODEX_O1_FINAL_PREARM_REVIEW_2026-08-29.md, O1-RB1/RB2). The sole operating
> authority is **`O1_DEPLOY_RUNBOOK_2026-08-30.md`** with
> `o1_boundary_preflight.py`. This file is retained as provenance only.

**Authority:** user ruling R-232 (O1 package a–d approved). Patch committed
UNDEPLOYED at `6786a02`; working tree deliberately held at `clob_v3_1`
(`collect_pm.py` modified vs HEAD) because `pm-collector-clob.service` has
`Restart=always` and runs from the live tree — an auto-restart before the
boundary would otherwise load v4 UNSTAMPED mid-day. **Do not clean or restore
that file before the boundary; it is a safety hold, not a stray edit.**

## Sequence (coordinator, at the boundary)

1. **~23:56Z prep:** confirm tree still holds v3_1 (`grep COLLECTOR_VERSION`),
   confirm unit active, arm the boundary script (background bash sleeping to
   the epoch second).
2. **At 00:00:00Z exactly:**
   a. `git checkout -- live/pm_research/collect_pm.py` (restores v4 from HEAD).
   b. Append the era stamp to `data/pm_5min/collector_runs.jsonl` (creates the
      pm-side era ledger; schema mirrors `data/mm_hf/collector_runs.jsonl`):
      ```json
      {"collector_schema_version": "clob_v4",
       "supersedes": "clob_v3_1",
       "boundary_utc": "2026-08-29T00:00:00Z",
       "package": ["O1a ping 10/10->3/3",
                    "O1b cause-aware jittered backoff",
                    "O1c subscribe-confirmation (SUBSCRIBE_UNCONFIRMED cause)",
                    "O1d gap_start at last coverage for never-connected sockets"],
       "commit": "6786a02", "authority": "R-232 user ruling",
       "era_semantics": "distributional only; NO row-stamping change; pre-boundary never-connected gap durations are understated (O1d)",
       "started_at_ns": <time.time_ns() at restart>, "pid": <new pid>}
      ```
   c. `systemctl --user restart pm-collector-clob.service`.
3. **00:01–00:05Z verify:** unit active; heartbeat/audit rows carry
   `clob_v4`; the ~seconds-long restart gap at 00:00 recorded normally (it is
   inside 08-29's P1 budget). The prices collector (`collect_pm_prices.py`)
   is NOT touched — out of package scope.

## Structural verification (post-deploy; R-182 — never a throughput A/B)

- **O1a:** PING_TIMEOUT duration distribution collapses toward
  ~ping_interval + 1.3 s (median gap 11.3 s → ~4.3 s), within-cause.
- **O1b:** retry spacing under persistent faults shows the exponential ladder
  and jitter (log lines carry `retry in Xs (fail #N)`).
- **O1c:** SUBSCRIBE_UNCONFIRMED appears as its own cause when it fires; its
  absence is legitimate only alongside normal first-message latencies.
- **O1d:** never-connected gaps record from scope start (longer, honest
  durations for that class; cross-boundary comparisons treat pre-boundary as
  understated).
- **Day bar:** 08-29 is judged under DAY_BAR_V2 (P1/P2/P3, `dfa0977`).
  SUPERSEDED 06:04Z by the doc's §3 amendment (`368345b` — MEM caught the
  stale citation here): the reading is now ONE band — expect **~55–80 s/hr**
  (both models); **below ~45** = O1b's reconnect residual also fell
  (unmodelled); **above ~120 (P1 FAIL)** = the detection-lag diagnosis was
  wrong, NOT the fix underperforming. *(Original two-model text "~30 vs ~79
  vs >120" superseded; the pre-registration doc is authoritative.)*

## Abort conditions

Unit fails to start, or v4 rows do not appear within 2 minutes: restore the
hold (`git show 6786a02^:... > collect_pm.py`), restart (back on v3_1), record
the abort in the register with the failure, and the era stamp entry gets a
superseding `aborted: true` row (rule 13 — never edit the earlier stamp).
