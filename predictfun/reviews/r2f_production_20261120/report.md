# R2-F: the full r2_label_sample_v2 production build — 3,072 worlds on consumed Oct 24–25 — 2026-11-20 to 2026-11-21

**USER:**
- "... continue the implementing and next priority loop";
- "As long as the results are reliable, we can accelerate the process";
- "I have let the other session to include as many issues as possible in each round, so we don't waste time to fix and validate one by
  one";
- "We need to make the run more efficient".

**Review answered:** 38febc0b's next priority: "complete the declared production label sample and its support/readiness receipt", as
one development round (items 1–4, decisions 2–3, parity, the full sample, one receipt).

**Scope:**
- **Data:** consumed Oct 24–25 only, under the frozen pi0_50_v1 teacher; no refit, no candidate fit.
- **Labels:** gross cents with zero costs; END 5,000 ms; the 50 ms grid; 58 / 37 / 100 ms.
- **What this is:** a label-only support and readiness receipt. **It measures target support, variation and build reliability, not
  prediction skill.** No intervals: two consumed days are below the five-day rule.

**Commits:**
- `d3d024fd2c5ccb1ca7e4d33e9b1162b91b50abd5`: the freeze. Code, harnesses, the parity and production plans and the declared checks,
  committed before any parity or production pair.
- The result commit: this receipt, the corrected v2 selftest and harness, the declared check corrections (v1.1, v1.2) and the state hunks.

## Verdict

- **Declared checks A–H: all_true = true** (8 of 8 true).
- **Labels (both days, both heads):** 18,249 ok pairs; **17,706 exactly 0 c (97.0 %)**; 203 positive and 340 negative (3.0 % nonzero).
- The nonzero labels come from 493 of the 6,084 originals with an ok pair (8.1 %) and 468 of 3,072 realized worlds (15.2 %). The 5 largest |Y| carry 11.4 % of all |Y|; the top 1 % carry 83.5 %.
- Y range -720 to 940.64 c; mean -0.410 c; median 0 c.
- **Label-only:** this receipt measures target support, variation and build reliability. It has not measured model skill.

## Review 38febc0b closure matrix (producing module `pf_r2_label_sample.py` `4cdeff0c`, unchanged since the freeze)

| Item | What was done | Closure evidence |
|---|---|---|
| **1** exclusion statuses, END, mark, fill times (P2) | **Status against result** (`validate` :599–602): a picked row's status must equal its result's: `pair_not_ok` when the pair failed, `feature_unavailable` when the fork inputs failed, else `ok`. **Failed pairs** (:603–606): only `missing_mark` (`PAIR_DATA_STATUSES` :116) is an exclusion; any other failed pair is refused as a failure. **Feature exclusions** (:607, `_pair_bad` :518–523) pass every pair invariant, without the inputs, and need an unavailable fork status. **Per pair** (`_pair_bad` :477): END = the declared 5,000 ms in the result and its identity (:485–487); every arm's mark is the common finite mark (:488–489); every fill time is within [0, END] (:490–491) | **Selftest** block 4: **50 corruptions refused**. They include the review's three (a `feature_unavailable` row with a false reconciliation, an intact pair relabelled `pair_not_ok`, END 4,000), plus another common mark, a consistent fill after END, a failed pair dressed as an exclusion, a feature exclusion with fork status `error`, a NaN arm value and a zero fill at 4,999 ms. Each corruption is isolated: it keeps every other identity, so only the check under test can refuse it. A legitimate `missing_mark` exclusion passes. **On real rows** (check G): the three refusals and the four of review 4393ad19, on every production partition |
| **2** zero-callback final tick (P2) | **Required callbacks recomputed** from each arm's decision log (`policy_calls` :453). Counted: every decision after the fork, plus the other side's at the fork tick when it acts after the forced one. Less the engine's no-book fallbacks. **Reconciled** (`_arm_calls_bad` :466): the teacher's calls = its binding checks = that count, and no binding failed. Zero only when none was required. The logs are stored and hash-bound (`arm_ledgers` :308; `decisions_sha256`). **Audit denominators** (:513–517): every arm's causal checks = ceil(calls / audit_every) | **Selftest**, the final-tick boundary on the real engine with a stub continuation. A KEEP fork at k = 99 has zero required callbacks and is valid (**Y = +5 c** from a trade at 4,990 ms). k = 98 and the other side at k = 99 have callbacks and are valid. A skipped required callback, an edited decision log and a count-neutral decision-log edit are each refused. A hand-built log (pre-fork, forced, the other side before / after it, a fallback) counts 3 / 4. **On real rows:** 71 arms of ok pairs made zero teacher callbacks. The revalidation of every part with the executing sources refused nothing (check F true), so each of them reconciled with its hash-bound decision log; 4,435,006 binding checks, 0 failed |
| **3** four interpretations (P3) | The readings below: the **union** of nonzero originals and their worlds; **path-only** against **economic** fill differences; **decision_ms** (elapsed since the original) against the **remaining horizon** (5,000 − decision_ms); the cost is the measured total, with no per-pair or per-world coefficient claimed | `y_summary.csv`, `paths.csv`, `coverage.csv`, `cost.csv` |
| **4** mutant record identity (P3) | `pf_mutant_summary.judge` (:29) requires each record's own index and name | Its selftest: 8 cases, including a copied record (another index) and a renamed record; both fail ([log](logs/summary_selftest.log)) |
| **Decision 2** audit every 35th call | `AUDIT_EVERY = 35` (:115), recorded in each plan (`audit_every`) and passed to every arm teacher (`world_rows` :353). Full binding checks on every callback, the fork hidden / rebuild checks and all-record accounting are unchanged | **Parity** (below): 288 / 288 pilot pairs are byte-identical, with 2,202 audit checks against attempt 4's 9,715. **Production:** 145,045 causal audit checks with 0 mismatches and 138,385 parity checks with 0 mismatches over the ok pairs' arms (check F true) |
| **Decision 3** per-day input cache | **Skipped**, as the review allowed ("optional"). Loading is 52.3 s of a 587 s partition on average (8.9 %) | — |

## Parity: the 288 accepted pilot pairs ([r2f_parity.py](r2f_parity.py) → [parity.json](parity.json))

The pilot run set was rerun under a new plan namespace (`parity_pilot_e35`) with the production code, audit every 35. Every row and pool
line was compared in canonical JSON (sorted keys). Only these fields are excluded, as declared before the run:
- each row's `causal` (the per-arm audit counters: the declared rate change);
- each arm's `decision_list` / `fallback_list` (the new logs, which the old rows do not carry). Their `decisions_sha256` is compared.

| Item | Value |
|---|---|
| rows identical | **288 / 288** (0 differing) |
| pool lines identical | **192 / 192** (0 differing) |
| same sample, same partitions | true / true |
| causal audit checks | 2,202 (every 35th call) against 9,715 (every 7th) |
| pair seconds (mean) | **2.40** against 5.0 in attempt 4 |
| parity CPU | 0.30 core-hours (6 partitions) |
| high-water | 1,158.9 MiB |

Finalize accepted the parity output (288 rows, rc 0, 3.56 s at 239 MiB; [log](logs/r2f_finalize_parity.log)).

## The R2-E pilot readings, superseded (review 38febc0b item 3)

The review kept the R2-E receipt and asked for its readings to be superseded here. The corrected values are computed by r2f_checks v1.2 on the parity output, whose 288 rows equal the pilot's 288 byte for byte outside the declared audit fields (so the same Y, fills, cash and positions): [y_summary](checks_equivalence/v12/y_summary.csv), [paths](checks_equivalence/v12/paths.csv). They are compared with the review's own count ([labels.json](../automatic/20261120_714ce759/labels.json)).

| Reading | R2-E receipt | Now (v1.2, the pilot's rows) | Review 38febc0b | Equal |
|---|---|---|---|---|
| Nonzero originals | 11 (7 with a positive + 4 with a negative label, summed) | 10 | 10 | true |
| Worlds with a nonzero label | — | 9 | 9 | true |
| Pairs whose arms' fill lists differ | 12 ("only when it changes fills") | 21 | 21 | true |
| … path-only: Y = 0, equal cash and inventory | — | 9 | 9 | true |
| … economic: Y ≠ 0 | 12 | 12 | 12 | true |
| … other: Y = 0, cash or inventory differ | — | 0 | 0 | true |

- **The horizon bins:** R2-E's `horizon_ms` binned `decision_ms`, the time elapsed since the original. The coverage below reports `decision_ms` and the remaining horizon (5,000 − `decision_ms`) as separate dimensions.
- **The throughput fit:** pairs = 6 × worlds in all 13 pilot runs, so the fit's per-pair and per-world coefficients are not separately identifiable (rank 2 of 3 columns). The aggregate projection, 26.97 core-hours with the probe at every 7th call, and the 46.1 core-hour slow case were projections, and "about 7 h on the four slots" was not a guarantee. This round's measured production cost (probe every 35th call) is under Cost below.

## Suites

- **v2 run 1** (the freeze's selftest `09416000`, 114 mutants): **107 / 114** ([summary](logs/mutants_v2_run1_summary.log); records kept in `logs/mutants_v2_run1/`). 7 escapes: the selftest passed with the check removed.

  | # | Mutant (check removed) | Why it escaped | Fix (selftest only) |
  |---|---|---|---|
  | 84 | finite values | the NaN corruption was on the mark, so the common-mark check refused it first | a NaN arm value: only the finite check sees it |
  | 89 | the fills hash / count | the added zero fill was at 1e9 ms, so the new fill-time check refused it first | the zero fill at 4,999 ms, inside the episode |
  | 92 | the pre-fork fills | the pre-fork fill was on a fork at 0 ms, so it fell before the episode (fill-time check) | the fill on a fork after 0 ms |
  | 101 | any failed pair as an exclusion | the failed-pair corruption lacked the fork fingerprint, so the fingerprint check refused it first | the fingerprint kept; only the data-status rule tells the outcomes apart |
  | 108 | the decision-log hash | the edit popped a decision, so the callback count refused it first | a count-neutral edit (an action renamed): only the hash sees it |
  | 113 | a feature exclusion's reason | no corruption exercised it | a feature exclusion with fork status `error` |
  | 100 | the status against its result | **equivalent:** every status / result mismatch is also refused by its status branch (an `ok` row by the pair or fork-input checks, a `feature_unavailable` row by its reason rule or the pair checks, a `pair_not_ok` row by the data-status rule) | removed from the harness; documented in its header |

- **v2 run 2** (selftest `60414b3f`, harness `ae015731`, module `4cdeff0c` = the freeze; **113 mutants**): **113 / 113 caught by a check**, the control passes ([summary](logs/mutants_v2_summary.log); 85.3 CPU min; one source set, equal to the files on disk).
- **Unchanged since their passing runs** (sources equal to the freeze; check H; counts from the R2-E receipt):
  - v1 (`pf_r2_labels.py` `f56f1ae5`), 35 / 35;
  - runner v5 (`5778c1a6`), 30 / 30.
- **The summary's selftest** passes 8 cases ([log](logs/summary_selftest.log)).

## The declared checks ([r2f_checks.py](r2f_checks.py) → [checks.json](checks.json))

The checks were declared before any parity or production output existed ([declaration](logs/declaration.txt)). Two instrument corrections
followed, each declared before any label was read:
- **v1.1** ([declaration](logs/declaration_v1_1.txt)): H's mutant count 114 → 113, for the equivalent mutant.
- **v1.2** ([declaration](logs/declaration_v1_2.txt)): memory only.
  - **The problem:** v1.1 decoded all 96 partitions' rows at once. On the parity output, decoded rows take 5.32 MB per MB of JSON.
    Production rows are ~3.6 MB of JSON a partition, so all 96 would need ~1.9 GB plus the module, over the 2 GiB rule.
  - **The fix:** v1.2 holds one partition at a time and reduces each row to the scalars the readings use.
  - **Acceptance:** **passed.** On the parity output (288 ok pairs, 6 partitions), v1.1 and v1.2 wrote identical outputs: all 7 CSVs byte for byte, and checks.json apart from its asof ([compare](logs/r2f_checks_eq_compare.log); outputs in [checks_equivalence/](checks_equivalence/)). Peaks there: v1.1 264 MiB, v1.2 230 MiB. The comparison's falsifiers ([log](logs/r2f_checks_eq_falsify.log)): a neutral csv / json rewrite compares identical; one changed CSV count and one flipped control are each flagged.

### Checks

| check | value |
|---|---|
| A_plan | true |
| B_runs | true |
| C_parity | true |
| D_acceptance | true |
| E_resources | true |
| F_labels | true |
| G_controls | true |
| H_suites | true |
| all_true | true |

accepted: dataset_sha256 `2c62fc9baf696e6cd8913c5c294dea2c9513640714ae34b413dc37ef8c0acb2e`; rows 18,372; pairs {'20261024': 9126, '20261025': 9125}; audit_every 35; plan_sha256 `0de151d1b980661d739e12c880f71eb1c81cca680ef6cef7ea97fcb36ce6c7fc`

### Audit and binding totals

| binding_bad | binding_checked | checked | fork_delivered_changed | mismatch | parity_checked | parity_mismatch | zero_callback_arms |
|---|---|---|---|---|---|---|---|
| 0 | 4,435,006 | 145,045 | {"20261024": 6037, "20261025": 6010} | 0 | 138,385 | 0 | 71 |

### Controls on real rows (every partition)

| control | refused on |
|---|---|
| END 4,000 | 96 / 96 |
| Y + 1 | 96 / 96 |
| another fork | 96 / 96 |
| binding_bad 1 | 96 / 96 |
| feature_unavailable, reconciled false | 96 / 96 |
| reconciled false | 96 / 96 |
| relabelled pair_not_ok | 96 / 96 |
| the intact part valid | 96 / 96 |

## Readings (descriptive; the full sample's own populations; no intervals)

### Population

| item | 20261024 | 20261025 | both |
|---|---|---|---|
| worlds_planned | 1,536 | 1,536 | 3,072 |
| worlds_realized | 1,536 | 1,536 | 3,072 |
| markets | 288 | 285 | 573 |
| utc_hours | 24 | 24 | 48 |
| originals | 3,072 | 3,072 | 6,144 |
| picks | 9,186 | 9,186 | 18,372 |
| pairs_run | 9,126 | 9,125 | 18,251 |
| ok | 9,125 | 9,124 | 18,249 |

Statuses by head (both sides, every source):

| head | status | 20261024 | 20261025 | both |
|---|---|---|---|---|
| keep | empty_pool | 0 | 1 | 1 |
| keep | feature_unavailable | 1 | 1 | 2 |
| keep | ok | 3,041 | 3,040 | 6,081 |
| keep | world:no_path:missing_endpoint | 30 | 24 | 54 |
| keep | world:no_path:stale_future_book | 0 | 6 | 6 |
| post | ok | 6,084 | 6,084 | 12,168 |
| post | world:no_path:missing_endpoint | 30 | 24 | 54 |
| post | world:no_path:stale_future_book | 0 | 6 | 6 |

ok rows by head and source:

| head | source | 20261024 | 20261025 | both |
|---|---|---|---|---|
| keep | delayed | 2,391 | 2,384 | 4,775 |
| keep | joint_chase | 202 | 233 | 435 |
| keep | joint_post_keep | 250 | 224 | 474 |
| keep | neutral | 175 | 186 | 361 |
| keep | pi0 | 23 | 13 | 36 |
| post | delayed | 1,539 | 1,571 | 3,110 |
| post | joint_chase | 50 | 52 | 102 |
| post | joint_post_keep | 39 | 27 | 66 |
| post | neutral | 3,042 | 3,042 | 6,084 |
| post | pi0 | 1,414 | 1,392 | 2,806 |

### Y (cents; ok pairs)

Y = R_a − R_b per ok pair, gross cents with zero costs. "nonzero originals (union)" counts the distinct originals with a positive or a negative label (review 38febc0b item 3); "nonzero worlds" the distinct worlds.

| day | head | n_ok | zero | > 0 | < 0 | nonzero originals (union) | nonzero worlds | originals | mean | median | p01 | p05 | p95 | p99 | min | max | top-5 share | top-1 % share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | post | 6,084 | 5,886 | 73 | 125 | 186 | 175 | 3,042 | -0.39 | 0 | -21.63 | 0 | 0 | 3.73 | -720 | 675 | 0.239 | 0.802 |
| 20261024 | keep | 3,041 | 2,951 | 36 | 54 | 90 | 88 | 3,041 | -0.43 | 0 | -15 | 0 | 0 | 4.52 | -510 | 165 | 0.363 | 0.815 |
| 20261024 | both | 9,125 | 8,837 | 109 | 179 | 260 | 246 | 3,042 | -0.40 | 0 | -16.72 | 0 | 0 | 4.53 | -720 | 675 | 0.191 | 0.809 |
| 20261025 | post | 6,084 | 5,911 | 60 | 113 | 162 | 155 | 3,042 | -0.45 | 0 | -30.72 | 0 | 0 | 0 | -542.64 | 940.64 | 0.233 | 0.829 |
| 20261025 | keep | 3,040 | 2,958 | 34 | 48 | 82 | 81 | 3,040 | -0.36 | 0 | -7.89 | 0 | 0 | 1.06 | -330 | 525 | 0.430 | 0.888 |
| 20261025 | both | 9,124 | 8,869 | 94 | 161 | 233 | 222 | 3,042 | -0.42 | 0 | -15 | 0 | 0 | 0.95 | -542.64 | 940.64 | 0.192 | 0.860 |
| both | post | 12,168 | 11,797 | 133 | 238 | 348 | 330 | 6,084 | -0.42 | 0 | -27.03 | 0 | 0 | 2.16 | -720 | 940.64 | 0.150 | 0.821 |
| both | keep | 6,081 | 5,909 | 70 | 102 | 172 | 169 | 6,081 | -0.40 | 0 | -11.20 | 0 | 0 | 2.16 | -510 | 525 | 0.267 | 0.858 |
| both | both | 18,249 | 17,706 | 203 | 340 | 493 | 468 | 6,084 | -0.41 | 0 | -15.10 | 0 | 0 | 2.18 | -720 | 940.64 | 0.114 | 0.835 |

### Fill paths (ok pairs)

"identical fills": the two arms' fill lists are equal. "path-only": the lists differ, but Y = 0 with equal cash and inventory. "economic": Y ≠ 0. "other": Y = 0 with cash or inventory differing.

| day | head | economic (Y != 0) | identical fills | other (Y = 0, cash or inventory differ) | path-only (Y = 0, equal cash and inventory) |
|---|---|---|---|---|---|
| 20261024 | keep | 90 | 2,935 | 1 | 15 |
| 20261024 | post | 198 | 5,402 | 2 | 482 |
| 20261025 | keep | 82 | 2,941 | 0 | 17 |
| 20261025 | post | 173 | 5,498 | 1 | 412 |

### Arms (sums over ok pairs)

| day | head | arm | rows | value_sum | cash_sum | position_sum | inventory_nonzero_rows | fills | filled_qty | generations | open_at_end_rows | teacher_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | keep | KEEP | 3,041 | -10,540.42 | 58,275.57 | -267.65 | 617 | 1,783 | 15,280.43 | 9,438 | 796 | 237,187 |
| 20261024 | keep | PULL | 3,041 | -9,221.63 | 50,677.08 | -218.40 | 581 | 1,683 | 14,360.62 | 9,638 | 717 | 236,435 |
| 20261024 | post | POST | 6,084 | -12,122.67 | 86,719.16 | 551.51 | 861 | 2,084 | 19,827.84 | 24,510 | 1,496 | 862,525 |
| 20261024 | post | WAIT | 6,084 | -9,757.35 | 60,565.02 | 841.51 | 744 | 1,843 | 17,840.74 | 19,222 | 1,493 | 877,961 |
| 20261025 | keep | KEEP | 3,040 | -6,964.54 | 12,615.38 | 456.26 | 571 | 1,777 | 14,852.49 | 8,947 | 649 | 241,795 |
| 20261025 | keep | PULL | 3,040 | -5,874.27 | 10,787.53 | 386.51 | 535 | 1,645 | 13,777.40 | 9,163 | 571 | 241,027 |
| 20261025 | post | POST | 6,084 | -6,167.28 | 69,536.17 | 924.69 | 707 | 1,832 | 16,385.19 | 22,221 | 1,174 | 861,022 |
| 20261025 | post | WAIT | 6,084 | -3,454.60 | 25,327.92 | 1,416.91 | 609 | 1,622 | 14,541.70 | 16,701 | 1,151 | 877,054 |

Teacher actions and statuses (calls):

| day | head | arm | action KEEP | action POST | action PULL | action WAIT | status cap_blocked | status incoherent_forecast | status model | status near_end | status outside_training_range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | keep | KEEP | 10,997 | 5,129 | 7,011 | 214,050 | 19,912 | 676 | 212,760 | 3,839 | 0 |
| 20261024 | keep | PULL | 9,382 | 5,329 | 4,365 | 217,359 | 19,032 | 620 | 212,854 | 3,929 | 0 |
| 20261024 | post | POST | 32,262 | 16,401 | 19,289 | 794,573 | 35,843 | 3,127 | 815,293 | 8,218 | 44 |
| 20261024 | post | WAIT | 31,238 | 17,197 | 14,253 | 815,273 | 26,458 | 3,087 | 839,998 | 8,381 | 37 |
| 20261025 | keep | KEEP | 10,106 | 4,599 | 6,647 | 220,443 | 18,277 | 1,725 | 217,616 | 4,177 | 0 |
| 20261025 | keep | PULL | 8,197 | 4,815 | 3,979 | 224,036 | 16,685 | 1,369 | 218,706 | 4,267 | 0 |
| 20261025 | post | POST | 28,123 | 14,242 | 17,415 | 801,242 | 28,638 | 5,447 | 817,894 | 8,967 | 76 |
| 20261025 | post | WAIT | 25,867 | 14,806 | 12,150 | 824,231 | 21,192 | 4,628 | 841,996 | 9,165 | 73 |

### Coverage of the ok rows (absent categories are explicit zeros)

decision_ms is the fork's time since the original; remaining_ms = 5,000 − decision_ms, the horizon left to the label's end.

source:

| day | head | neutral | delayed | joint_post_keep | joint_chase | pi0 |
|---|---|---|---|---|---|---|
| 20261024 | post | 3,042 | 1,539 | 39 | 50 | 1,414 |
| 20261024 | keep | 175 | 2,391 | 250 | 202 | 23 |
| 20261025 | post | 3,042 | 1,571 | 27 | 52 | 1,392 |
| 20261025 | keep | 186 | 2,384 | 224 | 233 | 13 |

decision_ms:

| day | head | 0 | 50-500 | 550-1000 | 1050-2000 | 2050-4950 |
|---|---|---|---|---|---|---|
| 20261024 | post | 3,149 | 291 | 349 | 606 | 1,689 |
| 20261024 | keep | 0 | 158 | 204 | 547 | 2,132 |
| 20261025 | post | 3,151 | 282 | 305 | 604 | 1,742 |
| 20261025 | keep | 0 | 190 | 209 | 523 | 2,118 |

remaining_ms:

| day | head | 5000 | 2050-4950 | 1050-2000 | 550-1000 | <=500 |
|---|---|---|---|---|---|---|
| 20261024 | post | 3,149 | 1,812 | 600 | 287 | 236 |
| 20261024 | keep | 0 | 1,498 | 744 | 399 | 400 |
| 20261025 | post | 3,151 | 1,764 | 641 | 279 | 249 |
| 20261025 | keep | 0 | 1,537 | 703 | 400 | 400 |

generation:

| day | head | 1 | 2 | 3+ | none |
|---|---|---|---|---|---|
| 20261024 | post | 5,733 | 169 | 182 | 0 |
| 20261024 | keep | 2,817 | 105 | 119 | 0 |
| 20261025 | post | 5,748 | 156 | 180 | 0 |
| 20261025 | keep | 2,821 | 99 | 120 | 0 |

partial_queue:

| day | head | known_filled>0 | none |
|---|---|---|---|
| 20261024 | post | 0 | 6,084 |
| 20261024 | keep | 101 | 2,940 |
| 20261025 | post | 0 | 6,084 |
| 20261025 | keep | 101 | 2,939 |

support:

| day | head | outside_training_range | supported |
|---|---|---|---|
| 20261024 | post | 0 | 6,084 |
| 20261024 | keep | 0 | 3,041 |
| 20261025 | post | 2 | 6,082 |
| 20261025 | keep | 0 | 3,040 |

utc_hour:

| day | head | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | post | 248 | 248 | 252 | 256 | 256 | 256 | 256 | 252 | 256 | 256 | 248 | 256 | 256 | 252 | 256 | 252 | 252 | 248 | 256 | 252 | 256 | 252 | 256 | 256 |
| 20261024 | keep | 124 | 123 | 126 | 128 | 128 | 128 | 128 | 126 | 128 | 128 | 124 | 128 | 128 | 126 | 128 | 126 | 126 | 124 | 128 | 126 | 128 | 126 | 128 | 128 |
| 20261025 | post | 256 | 252 | 256 | 248 | 256 | 256 | 252 | 252 | 256 | 256 | 252 | 252 | 256 | 256 | 248 | 248 | 256 | 256 | 256 | 256 | 256 | 248 | 256 | 248 |
| 20261025 | keep | 128 | 126 | 128 | 124 | 128 | 128 | 126 | 126 | 128 | 128 | 126 | 126 | 128 | 128 | 123 | 124 | 128 | 127 | 128 | 128 | 128 | 124 | 128 | 124 |

side:

| day | head | 0 | 1 |
|---|---|---|---|
| 20261024 | post | 3,042 | 3,042 |
| 20261024 | keep | 1,521 | 1,520 |
| 20261025 | post | 3,042 | 3,042 |
| 20261025 | keep | 1,521 | 1,519 |

### Weights

w = 1 / (pi · N · q); ESS = (Σw)² / Σw². The weighted means are descriptive: two consumed days carry no interval.

| day | head | picked | raw weight total | ok | ok weight total | weighted mean Y (c) | ESS | max weight share | top-10 % weight share | raw mean Y (c) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261024 | post | 6,084 | 30,406.7 | 6,084 | 30,406.7 | -0.503 | 3,195.3 | 0.0003 | 0.206 | -0.389 |
| 20261024 | keep | 3,042 | 30,406.7 | 3,041 | 30,396.6 | -0.426 | 3,035.4 | 0.0004 | 0.106 | -0.434 |
| 20261025 | post | 6,084 | 29,709.2 | 6,084 | 29,709.2 | -0.450 | 3,191.2 | 0.0003 | 0.206 | -0.446 |
| 20261025 | keep | 3,041 | 29,700.0 | 3,040 | 29,689.8 | -0.344 | 3,031.7 | 0.0003 | 0.106 | -0.359 |
| both | post | 12,168 | 60,115.8 | 12,168 | 60,115.8 | -0.476 | 6,385.7 | 0.0002 | 0.207 | -0.417 |
| both | keep | 6,083 | 60,106.7 | 6,081 | 60,086.4 | -0.386 | 6,066.4 | 0.0002 | 0.107 | -0.396 |

## Cost and efficiency

| item | value |
|---|---|
| partition_jobs | 96 |
| partition_cpu_core_hours | 15.458 |
| partition_wall_hours_sum | 15.657 |
| parity_cpu_core_hours | 0.296 |
| pairs | 18,251 |
| pair_s_mean | 2.515 |
| pair_s_max | 4.270 |
| load_s_mean | 52.344 |
| census_s_mean | 1.723 |
| peak_mib_jobs | 1,271 |
| peak_mib_records | 1,181.273 |
| ceiling_core_hours | 40 |

stops: 0; records as of 2026-11-21T00:21:10.154532+00:00; checks as of 2026-11-21T00:26:53.305358+00:00

- **Production:** 96 partition jobs on 4 slots, 15.46 core-hours of CPU; wall 19:48Z to 00:21Z (4.55 h). R2-E projected 26.97 core-hours with the probe at every 7th call (the review's recomputation; the R2-E receipt rounded it to 27.0): actual / projection = 0.57.
- **Pairs:** 2.52 s mean, 4.3 s max (the R2-E pilot: 5.0 s mean). Loading 52.3 s of a 587 s partition on average (8.9 %); census 1.72 s a world.
- **Peaks:** partition jobs ≤ 1,271 MiB; records' high-water ≤ 1,181 MiB; finalize 280 MiB (47 s); the declared checks v1.2 283 MiB (4.7 min).
- **Parity:** 0.30 core-hours (6 partitions). **Suites:** v2 run 2 85.3 CPU min (113 mutants, one per job; 113 caught).
- No per-pair or per-world coefficient is claimed (review 38febc0b item 3); these are measured totals and means.

## Deferred (nonblocking; recorded per the batching rule)

- **Decision 3, the per-day input cache:** skipped, as the review allowed. Loading is 52.3 s of a 587 s partition on average (8.9 %).
- **Mutant #100, the status against its result:** equivalent; removed and documented in the harness header. The header says the first R2-F run listed it "and it passed": its selftest passed with the check removed, i.e. it escaped. The wording is left as it is because the 113 / 113 run used that file (`ae015731`).
- **R2-E's deferrals stand:** v1 pairs' between-day release is not sentinel-tested (one synthetic day); v1's resume / finalize stay the historical path; runner v5's log has no per-mutant durations; attempt-3 and profile-only outputs stay in `data/` (the USER deletes).
- **The pool design's absent strata** (R2-E: pi0 KEEP and joint_chase POST never drawn in the pilot). In the full sample's ok rows: KEEP from pi0 20261024 23, 20261025 13 (0.6 % of the KEEP ok rows); POST from joint_chase 20261024 50, 20261025 52 (0.8 % of the POST ok rows). Both strata are drawn on both days, so neither is absent; a candidate that needs more of either needs a declared stratification amendment.
- **The 96 parts stay in `data/`** (32 MB); the receipt carries `accepted.json` and a sha256 manifest of every part, pool and record ([checkpoint/](checkpoint/)).

## Next

The combined review of this round (one request), which names the next priority. The question put to it: the candidate stage on consumed days (plan v2 R4: one declared candidate and its evaluation design). The target that stage would inherit: 543 nonzero of 18,249 ok labels from 493 originals in 468 worlds, on two consumed days (below the five-day interval rule).
