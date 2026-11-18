# A sealed-day read in the readiness loop, and its fix — 2026-11-18

**Standing rule** (predictfun/HANDOFF.md): "never the sealed 10-28..30". The readiness loop broke it once. The loop found the read itself
when it checked the five-day run's receipts.

## What was read

| | |
|---|---|
| Job | `st_day_20261027`: `pf_settlement_times.py day 20261027` (v1), run_light tag `st_day_20261027`, session aea5bfb9 |
| When | 2026-11-18 19:08:44Z to ~19:15:44Z (wall 419 s), inside the driver's `all` phase |
| What | The REST match pages of the sealed hours **10-28 00Z and 01Z**: 336,709 + 333,745 match records parsed into memory |
| Why | v1 padded every day by 1 h before and 2 h after (`pf_io.hours_of(day, extra_before=1, extra_after=2)`) with no day rule. REST pages arrive after their matches, so the padding finds settlement ids of late-recorded 10-27 legs |
| What was kept | Only legs of 10-27's own markets (the module drops every other market's legs). No 10-28 market's leg entered any output. 10-28 information reached outputs only as the settlement times of 10-27 legs whose pages were recorded after midnight |
| Not read | The replay loader skips the sealed books hours before opening them (`SKIP_FOREIGN_PARTS`, part_allowed). The 10-27 step-1 inputs the ledger and replay read (pend_inc, legs_timed2) hold no row at or after 10-28 00Z (counted) |

**Cause.**
- The module was written in this session without a day rule.
- I launched the `all` phase without checking every module in the chain for hour padding, although my notes already held this exact
  pitfall (FINDINGS P4 / P11 / P35).
- The same session's earlier days (10-23..26) pad only into consumed days.

## Fix (`pf_settlement_times.py` v2)

1. The page reader refuses any sealed (the replay loader's SEAL_MS: 10-28..30) or forward (>= FWD_MS, 11-04) hour **before calling its
   reader**.
2. `day_run` drops such hours from the padded list and records them as `hours_excluded` in the receipt.
3. A leg absent from the pages read whose executedAt lies within the padding of an excluded hour gets status 5
   (`absent_boundary_hour_excluded`) instead of 1.

**Falsifiers.**
- The reader is never called for 10-28 00Z.
- A padded list keeps only the allowed hour.
- A leg 30 min before the boundary gets status 5, and one 3 h before stays 1.
- The forward boundary is checked.

**Mutation suite: 15 / 15 caught by checks.** The harness also counted crashes as kills; v2 requires the selftest's own failure message.

**v2 output:** `settlement_times_v2_20261118`, all five days. v1 (`settlement_times_20261118`) stays as provenance; its 20261027 output
is VOID.

## Void outputs (set aside or left in place, never deleted)

| Output | Why |
|---|---|
| `data/predict_fun/research/settlement_times_20261118/20261027` (left in place inside v1, which stays as provenance) | VOID: built from the sealed hours |
| `data/predict_fun/research/keep_pull_exec_windows_v5_20261118_VOID_sealed_read` | The five-day ledger built on it |
| `data/predict_fun/research/keep_pull_scenarios_v6_20261118_VOID_sealed_read` | The five-day scenario replay built on it |

**Deleting them is the USER's choice.**

## Reruns (declared before they read data)

- [compare_settlement.py](compare_settlement.py):
  - S1: 10-23..26 identical to v1 row by row.
  - S2: 10-27 excludes exactly 10-28 00Z / 01Z; every other leg is identical to v1; every status-5 leg lies within 2 h before the
    boundary.
  - S3: every day's controls pass.
- The ledger and the scenario replay are rerun on v2 for all five days by the driver's `all` phase. The scenario replay's Oct 23
  reproduction (`compare.py`, R5a-d) is rechecked.

## Results

**[compare_settlement.txt](compare_settlement.txt): ALL true.**
- **S1:** 10-23..26 are identical to v1 row by row (802,157 / 681,400 / 689,852 / 818,939 legs).
- **S2:** 10-27 excludes exactly 10-28 00Z and 01Z and reads v1's other 25 hours. 18 of its 824,671 legs are now status 5. All 18 were
  mapped in v1 through the sealed pages, and all lie within 2 h before the boundary. Every other leg is identical to v1.
- **S3:** every day's controls pass.
- Status counts by day: [settlement_v2_status_by_day.csv](settlement_v2_status_by_day.csv).

**The five-day rerun on v2.**
- Every step returned 0; the ledger's V0 CONTROLS read YES and the scenario replay's Z0 CONTROLS read YES.
- The scenario replay's Oct 23 reproduction of v5 is ALL true again.
- The values are reported only from this run ([scenario_checklist_v6_20261118](../scenario_checklist_v6_20261118/report.md)); the VOID
  run's numbers are not quoted anywhere.

**Suites.** The settlement selftest passes, with 15 / 15 mutants caught by checks ([logs/](logs/)). The final suite of record over every
module is in the readiness plan.

## Decisions for the USER

1. **Does 10-28 stay sealed?** Two hours (00Z-01Z) of its REST match records were read and filtered to 10-27's markets. No 10-28 market's
   data entered any output. Under rule 34a a read consumes; the date is recorded here and in HANDOFF either way.
2. **Delete the three VOID outputs?** They are set aside under VOID names above.
