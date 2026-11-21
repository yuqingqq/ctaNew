# PI0-MOVEMENT-SHRINK-v1 — declaration

2026-11-21 12:39Z, session aea5bfb9. This is review 7ad03681's one next priority:
- One declared scalar movement-calibration challenger, fitted on the consumed Oct 26 admitted states and measured on the consumed Oct 27
  admitted states.
- The prediction test and the local action-value test are kept separate.
- The review states that no further USER approval is needed for this bounded experiment.

What it is not:
- It is a new adaptive experiment after the movement audit (`f311caae`), not a retroactive pass of that audit's
  `representation_or_data_needed` flag. That flag stays as recorded: a declared research heuristic, not a comparison of calibrated and
  uncalibrated predictions on another day.
- The audit's nearly unbiased forced result is aggregate calibration on that cohort. It is not proof of conditional calibration, or of
  calibration of the admission population.
- Both dates have already informed this research. This is chronological consumed-day development, not unseen validation.

The specification is the header of `predictfun/research/pf_movement_shrink.py`. Where the two differ, the header governs.

## What is frozen, and in what order

| Step | Commit content | Gate |
|---|---|---|
| Freeze | the module, selftest, 56-mutant harness, this declaration, `manifest.json` (TRAIN / EVAL / descriptive keys, input hashes, rules, multiplicity), `shrink_checks.py` (A-J, falsifier, selftest), `launch_shrink.sh` | nothing fitted, no Oct 27 score |
| Fit | `fit.json` (four groups: status, n, fills, weights, mu, lambda, lambda_raw, den, num, binding share) | `fit` refuses unless the freeze is in HEAD; the fit reads TRAIN rows only and refuses any other day |
| Results | Oct 27 scores, checks, receipt | `score` refuses unless `fit.json` is committed in HEAD and matches its sources and manifest |

Mutation results:
- 56 / 56 mutants are caught on the final sources (`logs/mutants_shr`).
- A first run caught 55 / 56 and is kept in `logs/mutants_shr_run1`. Its escape was a fixture gap: MSE and |bias| agreed on every
  fixture. A case where they disagree was added. No module code changed.

## Multiplicity

- One new candidate, in 4 side × head groups, with at most 8 scalars (one mu and one lambda per group).
- Earlier declared candidates on these consumed days: R3 ADV2-RICH-v1 (12 candidate heads) and R4E ADV2-ENTRY-SMOOTH-v1 (one primary, two heads).
- No variant is tuned in this round.

## Populations (consumed Oct 26 / Oct 27; the admitted cohort only; never the forced cohort)

| Set | Head | Rows (eligible) | Day |
|---|---|---:|---|
| TRAIN | entry (POST) | 166 | Oct 26 |
| TRAIN | first review (RETAIN) | 151 | Oct 26 |
| EVAL | entry | 163 | Oct 27 |
| EVAL | first review | 148 | Oct 27 |
| descriptive | first pull (first-review parameters) | 134 | Oct 27 |

Inputs and binding:
- The rows are PI0-ADMISSION-EXPOSURE-v1's 12 partitions, read through the frozen `pf_movement_audit.adm_rows`.
- Each partition record is byte-equal to its committed receipt copy (review 7ad03681 item 4), and both cohort manifests are committed.
- Weights, cells and parts are bound by exact key.
- Statuses and identities come from the frozen `pf_movement_audit.tables_of`.
- Every eligible row is reconciled with the audit's committed opportunities table.

There is no raw tape and no mid lookup. Every other row is counted with its status.

## The fit, the arms and the scores

These are in the module header:

    mu = sum(w M) / sum(w z)
    a = clip(P mu, E[q] L, E[q] U)     (L, U): physical bounds from the stage's own decision mid
    lambda = clip(sum(w d (M - b)) / sum(w d^2), 0, 1)

Support is ≥ 30 rows and ≥ 10 fills per group, or `insufficient_fit_support`. A zero denominator gives `zero_denominator`.

The arms are evaluated on identical rows: original (λ = 0), prior (λ = 1, a fixed control), and candidate (the fitted λ).

The baselines are zero and edge × E[q]. Reported for each:
- weighted MSE and bias, with MAE secondary;
- conditional errors on fills, with both weighted and unweighted fill denominators (review 7ad03681 item 3);
- Brier, which is unchanged across arms;
- calibration by each arm's own E[qm] and V bins;
- the old horizon / queue / ordinal bins, as diagnostics only;
- the projection's binding rate.

No trimming, winsorizing, bin fitting, threshold sweep, or any choice made on Oct 27.

## Local actions (Oct 27)

| Stage | Action | Value |
|---|---|---|
| Entry | POST iff V > 0; a tie WAITs | the neutral hold from activation to END; WAIT = 0 |
| First review | KEEP iff V ≥ 0 | the hold after the cancel effect; PULL = 0 |

Rules compared:
- none;
- all;
- original V;
- prior V;
- candidate V.

Measures:
- J = Σ(w·chosen·value) / Σ w over the eligible rows (c per opportunity).
- The mean value among the chosen.
- Counts, weights, worlds, Kish n.
- Severe value (< −30 c) totals.
- Weighted and unweighted top-5 generation / world shares.
- Per side.

Stages are never summed. Entry is a filter of PI0's own admissions; the review is scored on PI0's reached live states. This round
scores local choices on neutral labels and simulates no new strategy.

The null:
- 500 draws, seed 20261121.
- Masks are drawn uniformly within decision day × side × decision-UTC-hour, with each stratum's chosen count preserved.
- Every draw's chosen weight must equal the candidate's.
- Free and degenerate strata are reported.

## Readings (computed by `predicates()`, per head)

**Prediction** (pooled and per side). These are three separate booleans, with no combined verdict:
- `movement_mse_improves`
- `value_mse_improves`
- `movement_bias_reduced`

**Decision advantage:**

| Answer | Rule |
|---|---|
| INCONCLUSIVE | the head has no group with fit support, the candidate's mask equals the original's or the prior's, free strata < 5, free rows < 30, or a null weight mismatch |
| PASS | chosen weight > 0, J > 0, J > J_original, J > J_prior and J > the null p95 |
| FAIL | otherwise |

If the challenger only reduces posting, or fixes the mean without selecting positive exposure, that is reported as it stands, and no
other variant is tuned in this round. A favorable result would still need a declared controller / reached-state replay before
independent validation.

## Budget

- One aggregate core-hour, on the existing slots.
- 1,536 MiB checkpoint and 2,048 MiB hard bound; stop with statuses if either is exceeded.
- One freeze / fit / results package, with no intermediate minor-fix review.
- Resource totals in the receipt carry a named as-of time (review 7ad03681 item 3).

## Review 7ad03681 items 1-4

1. **Inference wording.** Covered by the "What it is not" list above. The audit's flag and thresholds stay as recorded.
2. **Checker coverage.** `shrink_checks.py` recomputes every aggregate that drives this receipt:
   - fitted parameters;
   - per-row predictions;
   - every score, calibration cell and action row;
   - all 500 null draws;
   - the readings.

   The falsifier corrupts each class of value (conditional mean, MSE, weighted denominator, selection value, chosen weight, null draw,
   p95, decision).
3. **Denominators.** Weighted and unweighted fill denominators are stored and labelled. Resource totals carry an as-of time. The audit's
   per-share table, which put 2,014.7 unweighted shares beside weighted ratios (weighted denominator 4,429.48), is not reused; this is
   noted in HANDOFF.
4. **Record binding.** Every reused admission record is bound to its committed copy. The audit's opportunities table is bound by its
   committed hash in the manifest. No new mid extraction.
