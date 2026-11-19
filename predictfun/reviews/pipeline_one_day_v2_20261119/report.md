# The corrected pipeline's keep / pull test (prediction model + value mapping + rule), Oct 26 and Oct 27 — 2026-11-19

**USER:** "We run the whole pipeline for one day test first", then, after review 89caa74c, "Run the checks first".

This supersedes the first-day run, which is **void**: its POST inputs read a future arrival, refused forecasts acted, and its gate was blind to
the value's sign (review 89caa74c, all three held).

**What is tested** is a prediction → keep / pull path with decisions fixed across the 19 scenarios:
- the export ([v2](../pipeline_export_20261119/report.md#v2--review-89caa74c-p1-the-pre-activation-post-state));
- `pf_pipeline_bench.py` v2: the joint two-part model and the contract's mapping;
- `pf_pipeline_policy.py` v2: "pull at the first live step whose contract-coherent predicted keep value is below 0", through the simulator's
  bank.

It is not the full controller: there is no learned POST / WAIT, re-entry or inventory.

**Days and declarations.**
- The model is fitted on Oct 23-25.
- **Oct 26 is the economic gate's day of record**, declared before v2 ran; no output had been evaluated on it.
- Oct 27 was seen in the first run, so it is reported **post hoc**.
- Values are local marks at END (+5 s), in cents, not P&L.

## Checks (all before the results; USER: "Run the checks first")

| Module | Selftest | Mutation suite | Key controls |
|---|---|---|---|
| Export v2 (P1 fix) | pass, with the reviewer's counterexample (next update at +30 vs +80 ms → identical POST state; v1 differs) | **30 / 30** | X1-X6, and **X5b**: on ~300 orders a day, the complete POST input vector is unchanged when later frames arrive at other times with other values and changes on a same-time change |
| Benchmark v2 (P2 coherence, P3 economic gate) | pass, with the reviewer's shifted-value case (the rank diagnostic passes, the economic gate fails) | **24 / 24** | B1-B5; each forecast carries the contract's own validity check (`J_ok`) |
| Trading test v2 (P2 accepted path) | pass, with a mixed valid / refused batch and a flag that disagrees with the contract | **15 / 15** | P0a-P0g; parity **2,000 / 2,000** on both days, 1,000 of them refused forecasts kept as the contract's fallback |

**Export v2 changes only the POST queue inputs** ([compare_v1_v2.txt](../pipeline_export_20261119/v2/compare_v1_v2.txt)):
- every RETAIN row is identical to v1 in every column;
- POST rows changed only in the five inputs the review named (`fp_queue`, `o_q_rel`, `o_q0_rel`, `o_q_frac`, `o_qunk`), on 130,714 rows.

One mutation escape on the way: "the parity ignores the contract verdict". It is fixed with the disagreeing-flag falsifier; the log is in
[logs/](logs/).

## The prediction model ([bench.txt](bench.txt))

**KEEP / PULL — economic gate (Oct 26): PASS.**
- **What the rule would pull** (predicted value < 0): realizes -9.2 c (bid) and -10.1 c (ask) per decision.
- **What the rule would keep:** realizes +0.75 c and +0.37 c.
- **Value level:** 0.93 / 0.95.

Post hoc on Oct 27, the kept decisions realize -0.07 and -0.92 c (E2 FAIL there).

Rank / quantity diagnostic:
- value rank +0.114 / +0.115 against the fair edge's +0.062 / +0.044 on Oct 26;
- amounts within 1-4%.

**POST — economic gate (Oct 26): FAIL** (E2). The orders the model would post (predicted value > 0, 7-8%) still lose: -4.0 c (bid) and
-8.5 c (ask) on Oct 26; on both days the predicted-positive set realizes about -5.9 c, against -14 to -15 c for all.

The diagnostic passes for both kinds. Incoherent forecasts (refused by the contract's check) are 0.08-0.15% of RETAIN decisions.

## The keep / pull rule through the bank ([policy_20261026.txt](policy_20261026.txt), [policy_20261027.txt](policy_20261027.txt))

Counts computed by [counts.py](counts.py) ([counts.txt](counts.txt)):

| Value at B (c) | Oct 26 (gate day) | Oct 27 (post hoc) |
|---|---:|---:|
| KEEP | -416,661 | -443,038 |
| A+0 (fair-edge rule) | -152,453 | -181,823 |
| **MODEL-J (learned rule)** | **-21,781** | **-23,916** |
| PULL-ALL (every order at its first live decision) | -23,731 | -23,346 |
| MODEL-J - PULL-ALL | **+1,950** | **-570** |
| Scenarios with MODEL-J above PULL-ALL / A+0 / KEEP | 15 / 17 / 17 of 19 | 3 / 17 / 17 of 19 |
| Scenarios with MODEL-J's E > 0 (its improvement over its own matched null) | 19 of 19 (STABLE-POSITIVE) | 18 of 19 |
| MODEL-J - PULL-ALL summed over the 19 scenarios | +29,943 | -23,877 |
| Orders pulled (at the first live step) | 99.0% (96.3%) | 99.1% (97.0%) |

**Readings (declared):**
- Oct 26: P1 E **STABLE-POSITIVE-IN-TESTED-SCENARIOS**; P3 / P4 / P5 / D1 / D2 NOT-STABLE; C2 at M250 HOLDS; C1 B1a HOLDS-SIGN.
- Oct 27: P1 NOT-STABLE (and the rest as Oct 26).

## What this supports (protocol section 4)

- **The corrected pipeline runs end to end with every check passing.** On the gate day its keep-value forecasts are calibrated in level, and
  rank better than the fair edge. Its keeps realize a small gain (+0.4 to +0.8 c per decision) and its pulls avoid losses.
- **The learned rule is PULL-ALL plus about 1% of kept orders.** On Oct 26 those kept orders add +1,950 c over PULL-ALL at B (15 / 19
  scenarios); on Oct 27 they cost -570 c (3 / 19). Two consumed days with opposite signs: **no consistent edge over pulling everything is
  shown.** Against A+0, which keeps many losing orders, the rule is far ahead on both days (17 / 19 scenarios).
- **Entry is the loss.** No POST selection the model makes avoids losing on these days. For the 30-sh JOIN order at the best, posting loses
  on average in this simulator; the decision that loses money is entering, not when to cancel.

None of this is P&L, validated on live orders (C10), robust to the queue rule (C5), or more than two consumed days (G = 2, point values).
The model is one of the benchmark's 8 candidates.

**Limit.** The model's own-fill inputs (known remaining) are the baseline timing's in every scenario.

## Files

- **Outputs:** [bench.txt](bench.txt) / [bench.json](bench.json) (readings, both gates, models' hashes), and the policy text / JSON / CSV
  for both days.
- **Logs:** [logs/](logs/). **Sources:** [sources.sha256](sources.sha256). Every recorded source, output, model and input hash matches the
  committed files.
- **Data:** `data/predict_fun/research/pipeline_{export_v2,bench_v2,policy_v2}_20261119`.
- **The void first-day outputs** (`pipeline_bench_20261119`, `pipeline_policy_20261119`, `pipeline_policy_20261119_run1`, and the untracked
  local receipt `predictfun/reviews/pipeline_one_day_20261119`) are kept, not deleted: deleting them is the USER's call.
