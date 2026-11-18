# Scenario replay v6 — FINDINGS checklist C12, C2 and C1 (B1a), 2026-11-18

**USER:** "My goal is to make our simulator ready for test runs and generate correct results, can you launch a loop to achieve this"
(readiness loop, [plan items 3 and 5](../simulator_readiness_20261118/plan.md)).

## What v6 adds

`predictfun/research/pf_keep_pull_scenarios.py` v6. Scenario values are unchanged from v5; v6 adds columns, readings and a control.

| Item | Before | v6 |
|---|---|---|
| C12: the age split by trade-anchor source | missing | Every re-simulated fill carries the source of the trades at its instant (WS / E2 / imputed, or mixed when several), assigned by the committed run's rule (`KP.trade_index` + `KP.window`). C1's age table is split by it. **Control (j):** at the baseline, every fill's source equals the committed fill's, order by order |
| C2: say whether the claim holds at M250 | M0 / M50 / M250 values only | Readings `C2 E AT M250 vs M50` and `C2 KEEP + SCORE AT M250 vs M50`: HOLDS-AT-M250 iff the sign at M250 equals the baseline's |
| C1: the headline B1a-weighted beside it | v5 printed it without fills < 100 ms only | E weighted by age bin with FV3's B1a precision weights (`pf_fv3_check.B1A`, M 0 and M 50), per scenario. Reading `C1 B1a-WEIGHTED E AT B` |

**Falsifiers.**
- A committed fill's trade source changed under a matching receipt must be refused by control (j). It is: no output is written.
- The age-by-source table must partition the fills.
- The B1a weighting is unit-tested on planted bins.
- The M250 and B1a readings are tested on planted signs.

**Mutation suite:** 44 / 44 caught by checks, with four mutants added for C12 / C2 and three for B1a ([logs/scenarios_mutants.log](logs/scenarios_mutants.log)).

## Reproduction (declared before v6 read any data: [compare.py](compare.py))

Two runs were checked:

- the pre-B1a v6 source on Oct 23 ([compare_20261023.txt](compare_20261023.txt); its output was moved to `keep_pull_scenarios_v6pre_20261118`);
- the final v6 source on settlement times v2, Oct 23 of the five-day run ([compare_20261023_final.txt](compare_20261023_final.txt)).

Both read ALL true:

| Predicate | Result |
|---|---|
| R5a: every v5 value (E, score, null, KEEP, PULL-ALL, fills, shares, eligible, fills' value, age bins, no-young, cancel-pending) and every v5 reading identical | **true** (19 scenarios, 0 differences) |
| R5b: the source split partitions every age bin (fills exactly, E within 1e-6) | **true** |
| R5c: control (j) ran on every committed order | **true** (30,088 of 30,088) |
| R5d: the C2 readings equal the predicate recomputed from the reported values | **true** |

## Oct 23 readings (the B1a values are in the five-day section)

**C2.** E changes sign at M250, while KEEP + score keeps it.

| | E (c) | KEEP + score (c) |
|---|---:|---:|
| M50 | +34,948 | -174,961 |
| M250 | -13,342 | -297,803 |
| Reading | SIGN-DIFFERS-AT-M250 | HOLDS-AT-M250 |

**C12, the baseline's E by source (c, with fills):** most of E sits on WS-timed fills.

| | WS | E2 | imputed | mixed |
|---|---:|---:|---:|---:|
| M50 | +20,957 (11,311) | +3,158 (4,312) | -156 (44) | +10,989 (2,003) |
| M250 | -1,478 (13,449) | -4,973 (4,544) | -256 (49) | -6,635 (1,998) |

At M250, E is negative on every source.

## Five days (Oct 23-27) under the final source

**Run.** The driver's `all` phase on the final sources, on settlement times v2 ([sealed_read_fix_20261118](../sealed_read_fix_20261118/report.md):
the first five-day run is VOID because v1 read sealed hours).
- Settlement times, the five-day ledger (V0 CONTROLS YES), 149,174 states re-simulated per scenario (143,134 eligible at B) and the pooled
  report.
- Every step returned 0; peak memory 1.9 GiB. Logs: [logs/all_phase/](logs/all_phase/).
- **Z0 CONTROLS YES:** B's E and score improvement equal the committed per-fill values.

**Values (c, pooled over 5 days; [scenarios_5days.txt](scenarios_5days.txt), [scenarios_5days.csv](scenarios_5days.csv), by day in
[by_day_5days.csv](by_day_5days.csv)).** Columns:
- **E**: the declared score's improvement over its matched null.
- **w/o < 100 ms**: the headline without fills on orders < 100 ms old.
- **B1a M50**: weighted with FV3's M 50 precision.
- **Days E > 0**: of 5.
- **KEEP + score**: the rule's own value, not P&L.

| Scenario | E | w/o < 100 ms | B1a M50 | Days E > 0 | KEEP + score |
|---|---:|---:|---:|---:|---:|
| B (M50, A58, L37, N100) | +234,279 | +230,615 | +94,810 | 5 | -776,492 |
| M0 | +243,587 | +241,226 | +99,469 | 5 | -646,474 |
| **M250** | **-27,124** | -25,561 | -15,065 | **1** | -1,393,791 |
| S1lo / S1mid / S1hi | -23,167 / +170,868 / +224,690 | | | 1 / 5 / 5 | |
| S2lo / S2hi | +329 / +32,467 | | | 2 / 4 | |
| S3lo / S3hi | +19,702 / +232,299 | | | 4 / 5 | |
| SThi / EVhi | +283,895 / +18,962 | | | 5 / 4 | |
| L0 / L100 / slow mixture | +241,635 / +152,191 / +200,385 | | | 5 / 5 / 5 | |
| A15 / A100, N160 / N660 | +231,732 / +238,012, +233,543 / +234,060 | | | all 5 | |

The per-scenario C1 / C3 tables are in [scenarios_5days.txt](scenarios_5days.txt) and [days_positive_5days.csv](days_positive_5days.csv).

**Readings (declared in the module).**

| Reading | Value |
|---|---|
| C1 B1a-WEIGHTED E AT B | HOLDS-SIGN (+114,705 at M 0 weights, +94,810 at M 50) |
| C2 E AT M250 vs M50 | **SIGN-DIFFERS-AT-M250** |
| C2 KEEP + SCORE AT M250 vs M50 | HOLDS-AT-M250 (negative at both) |
| Z1 / Z2 / Z3, Z1-NY / Z3-NY | NOT-STABLE-IN-TESTED-SCENARIOS |
| TIMING EVIDENCE | LOWER-MATCH-BOUND-UNIDENTIFIED |

**C3, cancel-pending fills at B:** 1,258 fills, -63,310 c, 2.5% of the fills' gross. At L100 they are 7.5%; under the slow mixture 12.4%.

**C12, B's E by source, c (fills):**

| Source | E (c) | Fills |
|---|---:|---:|
| WS | +130,296 | 56,763 |
| E2 | +21,380 | 20,314 |
| imputed | -428 | 249 |
| mixed | +83,032 | 9,578 |

Young fills (< 100 ms) are 5.1% of B's fills.

**What these numbers support (protocol section 4).**
- The declared score improves on its matched null at the baseline on every day, and under every activation / latency / notice variant.
- The improvement does not survive the slow trade clock (M250, 1 of 5 days) or the early end of the S1 / S2 timelines.
- Its sign therefore depends on the unidentified match timing: "stable in tested scenarios" does NOT hold.
- KEEP + score is negative in 17 of 19 scenarios.
- None of this is P&L, order-level validated (C10), or robust to the queue rule (C5).
