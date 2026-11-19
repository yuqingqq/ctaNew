# The new model: paired POST / WAIT and KEEP / PULL advantages under a frozen continuation — 2026-11-19

**USER:** "Build the new model and get the numbers" (plan 0c3d2523 steps 2–3; step 1, the no-fit baseline, skipped by the USER). The plan's
step 0 (the coherence repair and the corrected attribution) is [pipeline_one_day_v3_20261119](../pipeline_one_day_v3_20261119/report.md).

## What was built

**pi0, the frozen follow-on policy.** It is the plan's initial reference, the E1 rule:
- post at an original opportunity iff the coherent J POST forecast's value is above 0;
- while live, pull at the first 50 ms step (from +100 ms) whose coherent J RETAIN forecast is below 0;
- no re-entry before the common END (+5 s).

Its J model is fitted on **Oct 23 only** (the plan's warm-up), frozen, and predicts Oct 24–27 point in time.

**The paired targets**, at every neutral decision:
- **Delta_keep(j)** = value of KEEP through the next 50 ms tick, then pi0, minus PULL now. This is the value of the fills between this
  pull's landing and the landing of pi0's next pull.
- **Delta_post(t)** = value of POST now, then pi0, minus WAIT one tick, then pi0. The first term is the order's value under pi0's
  cancellation.

Under pi0 nothing can be posted after a WAIT before END (its entries are on the 5 s opportunity grid), so the WAIT branch carries no
exposure. This is pi0's definition, not a shortcut. A pi0 with entries at any 50 ms tick needs a new-order simulator, which is not built.

**Candidates** (pinned LightGBM, 200 rounds, separate bid / ask × POST / KEEP heads, zero action boundaries; declared before any label was
read; multiplicity 12 candidate heads + 32 control heads):

| Name | What | Role |
|---|---|---|
| **A** | direct regression on the paired advantage, the 91 rich inputs | primary candidate |
| AF | the same heads on fair edge + time to END | diagnostic control |
| AM | the same heads on the 23 mechanical inputs | diagnostic control |
| PI0 | the Oct 23 J model, E1 rule | the label continuation |
| NEU | the old neutral-target J model fitted on Oct 24–25 (the candidate's history), E1 rule | same-history control |
| E1 | the benchmark v3 J model (Oct 23–25), E1 rule | the current rule of record |

The chronology is pi0 on Oct 23, labels on Oct 24–25, candidates fitted on Oct 24–25, evaluated on Oct 26–27. All are consumed days, so this
is development evidence, not validation.

**One property of this simulator matters for reading the results.** All fills of an order are valued at the same END mark. So an order's
paired label always has the same sign as its old neutral label. They differ in size and in which orders carry value. The plan's
counterexample (the neutral hold loses while keep-one-tick-then-pull gains) can only arise across orders; for one order it is tested on the
label arithmetic itself.

## Checks (all before any result)

| Module | Selftest | Mutation suite | Key controls |
|---|---|---|---|
| `pf_pipeline_paired.py` | pass: every label recomputed by a per-order loop from the fills; the counterexample and its converse; the planted advantage recovered | **20 / 20** (first run: 4 selftest gaps, fixed — [logs/](logs/)) | K1 chronology, K2 no label among inputs, K3 export receipts, **K4 every label reconciled with the keep / pull run's own fills, pi0's steps by a second path**, K5 every eligible row labelled, K6 saved models reload |
| `pf_pipeline_paired_eval.py` | pass: B values by a per-fill loop; readings recomputed from the CSV | see below | R0a coverage, R0b bank = per-order path for every policy and scenario, **R0c the paired POST labels sum to the bank's value under pi0**, R0d branch identity, R0e refusal accounting, R0f decisions by a second path |
| `pf_pipeline_entry.py` (the per-order path the evaluation uses) | pass | see below | |

## The labels (the new target, [logs/pq_run.log](logs/pq_run.log))

| Day | POST: mean paired / neutral (c per opportunity) | RETAIN: mean paired / neutral (c per decision) | pi0 pulls |
|---|---:|---:|---:|
| Oct 24 | −0.645 / −11.046 | −0.311 / −6.049 | 27,817 orders |
| Oct 25 | −0.594 / −11.330 | −0.171 / −6.170 | 27,305 |
| Oct 26 | −0.835 / −13.967 | −0.351 / −8.807 | 27,774 |
| Oct 27 | −0.887 / −15.101 | −0.237 / −8.686 | 27,344 |

**Measured against what actually happens after posting, entering costs about 0.6–0.9 c per opportunity, not 11–15 c.** pi0 still pulls
almost every order at its first review.

## The prediction model (G3; Oct 26–27)

**PREDICTION GATE (A): FAIL on both days and both sides.** In every selected group the predicted and realized totals differ in size, and
usually in sign:

| A, selected group | Oct 26 bid | Oct 26 ask | Oct 27 bid | Oct 27 ask |
|---|---:|---:|---:|---:|
| POST (pred > 0): n / predicted / realized c | 6,319 / +18,360 / −399 | 7,807 / +15,952 / +96 | 5,866 / +15,358 / −6,680 | 7,453 / +12,610 / −1,787 |
| KEEP (pred > 0, reached rows) | 39,085 / +41,663 / −8,314 | 59,731 / +36,149 / +940 | 42,098 / +38,895 / +1,501 | 75,159 / +33,591 / +3,790 |
| PULL (pred < 0, the pull step) | 13,613 / −17,602 / −12,137 | 13,634 / −16,381 / −9,255 | 13,424 / −14,818 / −7,467 | 13,379 / −13,265 / −17,297 |

- **Ranking:** A ranks a little better than its controls (POST +0.022 to +0.060; KEEP +0.044 to +0.059), against AF about 0 and AM up to
  +0.04. All three are weak.
- **Calibration:** in the plan's fixed bins, A's predictions spread from −10 to +10 c while realized values stay near 0 or below in almost
  every bin. The direct regression on these heavy-tailed labels (mostly 0, a few large) is dominated by noise, and "predicted > 0" selects
  that noise. The two controls fail the same way.

## Through the scenario bank (single-order episodes; decisions fixed across the 19 timing scenarios)

All numbers come from [counts.py](counts.py) → [logs/pq_counts.log](logs/pq_counts.log): value at baseline timing B, in cents marked at END
(not P&L). The checks passed first:
- the evaluation suite caught **17 / 17** planted errors, after a harness defect and 5 selftest gaps were fixed ([logs/](logs/));
- the entry module's per-order path, which the evaluation uses, caught **32 / 32**;
- every control R0a–R0f passed on both days. R0c, the paired labels against the bank's value under pi0, held at B.

| Policy | Oct 26 | Oct 27 | Above 0 in scenarios (Oct 26 / 27) |
|---|---:|---:|---:|
| N0: never post | 0 | 0 | — |
| PA: post every opportunity, pull at once | −23,731 | −23,346 | 0 / 1 of 19 |
| **A: the new candidate** | **−2,032** | **−11,865** | 4 / 3 of 19 |
| A-PA: A's admissions, pull at once | −1,053 | −6,944 | 1 / 1 |
| AF (fair edge + time) / AM (mechanical) | −18,951 / −17,772 | −18,588 / −16,601 | 2 / 2, 2 / 2 |
| PI0: Oct 23 J, E1 rule | +478 | −2,608 | 9 / 3 |
| NEU: neutral-target J on Oct 24–25, E1 rule | **+3,540** | −2,827 | 14 / 1 |
| E1: neutral-target J on Oct 23–25 (rule of record) | +277 | **−824** | 14 / 3 |
| A's matched null (random selection of the same count per window / side / minute, A's cancellation) | −11,026 | −13,432 | |

E1's Oct 26 values (+276.93, and −745.56 with pull at once) reproduce the review's anchors.

**Readings, computed:**
- **Economic gate: FAIL on both days** (A > N0, A > A-PA and A > PI0 all false for both sides summed). Per side the only true predicate is
  Oct 26 bid, A > A-PA.
- **Matched control: PASS on both days.** A's admissions beat a random selection of the same size by about 9,000 and 1,600 c.
- **Practical gate: FAIL on both days**, with the same predicates on each. Against its own admissions pulled at once, A holds both sides at
  the best longer (Oct 26: 0.28% of the clock against 0.19%), with fewer first-review reversals and fewer cancellations per filled share.
  But its median lifetime is the same (79 ms) and the economic gate fails.
- **Prediction gate (G3): FAIL**, carried from the fit.
- Across scenarios, A − N0, A − A-PA, A − PI0, A − NEU and A − E1 are all **NOT-STABLE**.

**What the candidate does** (at B, both sides):

| | A, Oct 26 / 27 | PI0, Oct 26 / 27 | PULL-ALL of every opportunity |
|---|---:|---:|---:|
| Posts (of 29,832 / 29,338 opportunities with a path) | 14,126 / 13,319 | 3,364 / 2,978 | all |
| Pulled at the first review (of orders live there) | 57% / 57% | 74% / 73% | 100% |
| Resting-quote time (of the valid clock) | 2.5% / 2.9% | 0.65% / 0.67% | 1.5% |
| Lifetime p90 | 679 / 779 ms | 579 / 779 ms | 79 ms |

**The candidate posts about half of all opportunities and keeps orders it should pull.** On the rows it reaches before its pull, it predicts
+0.35 / +0.31 c, but those decisions realize −0.23 / −0.14 c (the neutral rows: predicted −0.34 / −0.28, realized −0.35 / −0.24).

## What this supports

- **The new target is right in kind.** Valued against what follows a post (pi0's cancellation), posting costs 0.6–0.9 c per opportunity,
  not the 11–15 c of the "never cancelled" label. The candidate's own entry selection beats random selection of the same size on both days.
- **The new candidate does not work as declared.** Its direct-regression heads on these heavy-tailed paired labels are noise-dominated and
  overconfident. It loses money on both days, falls below pi0, and fails the prediction, economic and practical gates.
- **No policy beats never posting on both days.** The neutral-target rules come closest: NEU +3,540 / −2,827, E1 +277 / −824, PI0 +478 /
  −2,608. These are two consumed days, one result per day, no interval.

**Limits:**
- single-order episodes, not a joint bid / ask replay with inventory;
- pi0 has no re-entry at grid times;
- decisions are frozen across scenarios;
- queue rule M only (C5); no live validation (C10);
- markouts at +100 ms / +1 s are not computed;
- the Oct 27 replay peaked at 2,075 MiB, above the 2 GiB guard level, which is checked only at fixed points. A memory reduction of its
  metrics section is a follow-up.

## Files

- **Sources:** `pf_pipeline_paired.py`, `pf_pipeline_paired_eval.py`, `pf_pipeline_entry.py` (the per-order path, extended to the plan's
  metrics) and their mutation harnesses.
- **Outputs (copies):** [paired.json](paired.json) / [paired.txt](paired.txt); eval_<day>.json / .txt / .csv.
- **Logs:** [logs/](logs/), including each first run that found the gaps; hashes: [sources.sha256](sources.sha256).
- **Data:** `data/predict_fun/research/pipeline_paired_20261119` (labels_<day>, pred_<day>, models/) and `pipeline_paired_eval_20261119`.
