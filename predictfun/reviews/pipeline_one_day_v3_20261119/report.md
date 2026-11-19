# Step 0 (G0) of plan 0c3d2523: the coherence repair, checks on every row, the corrected attribution — 2026-11-19

**USER:** "Can you check the commit and proceed the next priority" (review 1adf6a33), "Check the new commit about the testing plan" (plan 0c3d2523),
then "Build the new model and get the numbers" (step 1, the no-fit five-policy baseline, skipped by the USER). This receipt is the plan's step 0.
The new model (steps 2–3) is [pipeline_paired_20261119](../pipeline_paired_20261119/report.md).

**This supersedes two statements of the [v2 report](../pipeline_one_day_v2_20261119/report.md)**, both review 1adf6a33 P2:
- **Lines 79–81** attributed the whole MODEL-J − PULL-ALL difference to the ~1% of orders never pulled. The corrected attribution is below.
- **Lines 82–83** concluded that entry is the loss. The POST label values a post as if it were never cancelled; it cannot settle entry
  followed by cancellation. The defensible statement is narrower: the POST predictor fails its declared neutral-continuation target.

Every v2 value stands: v3.1 reproduces all of them (below).

## Review 1adf6a33, each finding at the code

| Finding | Holds? | Fix |
|---|---|---|
| P2: zero expected quantity accepted with a nonzero E[q · move] | Holds: `pf_pipeline_bench.py` set the implied move to 0 when E[q] = 0, so `ok` stayed True; the parity rebuilt a zero-value forecast and compared actions on a 2,000-row sample | Benchmark v3: a forecast is coherent only if the two-atom forecast rebuilt from its moments gives back E[q], E[q · move] and the mapped value (1e-9 tolerance, relative above 1). Keep/pull test v3: the parity runs on **every** RETAIN row and compares the contract's verdict, the round trip and the action |
| P2: the attribution vs PULL-ALL | Holds: the report's sentence was not computed | Computed by [counts.py](counts.py) from the keep/pull run's own fills (table below) |
| P2: the entry conclusion | Holds: the POST label's continuation is not the cancellation rule's | The statement is narrowed here; entry under a frozen cancellation is valued in the new model's step |

## Checks (all before any result)

| Check | Result |
|---|---|
| Benchmark v3: selftest; mutation suite | pass; **25 / 25** (one new mutant: the v2 flag without the round trip) |
| Benchmark v3 refit vs v2 ([compare_bench_v2_v3.py](compare_bench_v2_v3.py), declared before v3 ran) | the 44 model files are **byte-identical**; every shared prediction column is identical except the coherence flags; `J_ok` changes only where E[q] = 0 with a nonzero E[q · move]: **23 RETAIN rows / 9 orders on Oct 26 (max 36.5011 c), 15 / 6 on Oct 27 (max 5.2075 c), no POST row**, exactly the review's counts |
| Keep/pull test v3.1: selftest; mutation suite | pass; **19 / 19** |
| [g0_coherence_cases.py](g0_coherence_cases.py): the flag = the expected value = the contract's verdict | **12 / 12**: a valid forecast, an impossible endpoint, E[q] = 0 with E[q · move] −100 / +100 / 0 c, P = 0, P = 1, E[q \| fill] at the remaining-size cap, NaN, inf, 5e-10 c (inside the tolerance) and 5e-9 c (beyond it) |

**One defect found by checking every row** ([logs/pp3_day_20261026_run1_P0f_451_clip_rows.log](logs/pp3_day_20261026_run1_P0f_451_clip_rows.log)).
The first v3 run refused on Oct 26: on 451 of 1,269,738 rows the contract rejected a forecast the benchmark accepted.
- **Cause:** all 451 have the conditional quantity capped at the remaining size (30 shares). Rebuilding it as E[q] / P rounds one step
  above the cap (30 + 3.6e-15), which the contract refuses.
- **The benchmark's flag was right;** the parity's reconstruction was wrong ([diag_parity_clip.py](diag_parity_clip.py) →
  [logs/diag_parity_clip.log](logs/diag_parity_clip.log)).
- **v3.1** sets the rebuilt quantity back to the remaining size within the same declared tolerance. A falsifier covers the boundary, and a
  mutant removing the fix is caught.

The 2,000-row sample of v2 could not have seen this; the plan's "validate every forecast" requirement found it.

## Results: nothing changes except the refused-forecast count

- **Every value is unchanged.** Policy v3.1 equals v2 in all 190 rule × scenario × field cells on each day, with the same pulls
  (27,982 / 27,626).
- **The contract agrees on every row:** 1,269,738 / 1,269,738 and 1,404,540 / 1,404,540.
- **The only reading that differs is the refused-forecast count:** 1,062 → 1,085 on Oct 26 and 2,146 → 2,161 on Oct 27. These are exactly
  the zero-quantity rows.

This is the plan's expected result: zero changed first triggers and zero value change.

## The corrected attribution (MODEL-J − PULL-ALL at B, c; [counts.py](counts.py) → [logs/pc3_counts.log](logs/pc3_counts.log))

The group totals reconcile with the committed policy v3.1 values on both days. The table reproduces the review's follow-up exactly.

| Group | Oct 26 orders | Oct 26 difference | Oct 27 orders | Oct 27 difference |
|---|---:|---:|---:|---:|
| Pulled at the first live step (+100 ms) | 26,942 | 0.00 | 26,795 | 0.00 |
| Pulled later | 1,040 | **+954.33** | 831 | **−821.85** |
| Never pulled, with RETAIN predictions | 292 | **+995.36** | 255 | **+251.74** |
| No RETAIN prediction | 168 | 0.00 | 157 | 0.00 |
| **Total** | 28,442 | **+1,949.69** | 28,038 | **−570.11** |

By side, Oct 26: later pulls +134.79 (bid) / +819.53 (ask); never-pulled −337.79 / +1,333.15. Oct 27: later −680.37 / −141.48; never-pulled
+197.52 / +54.22.

**So on Oct 27 the orders the rule never pulls gain relative to PULL-ALL; the later pulls lose.** The v2 report's sentence was wrong for
both days.

## Files

- **Sources:** `pf_pipeline_bench.py` v3, `pf_pipeline_policy.py` v3.1 and their mutation harnesses.
- **Receipt scripts:** [compare_bench_v2_v3.py](compare_bench_v2_v3.py), [g0_coherence_cases.py](g0_coherence_cases.py),
  [diag_parity_clip.py](diag_parity_clip.py), [counts.py](counts.py).
- **Logs:** [logs/](logs/); hashes: [sources.sha256](sources.sha256).
- **Data:** `data/predict_fun/research/pipeline_{bench_v3,policy_v3}_20261119`. The v2 outputs stay as provenance; nothing is tabulated
  against them except the equality predicate above.
