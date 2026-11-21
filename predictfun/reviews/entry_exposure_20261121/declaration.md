# ENTRY-EXPOSURE-v1 — declaration (frozen before any branch runs)

Review `3d8bf5ad` (one next priority) on target `37300cc9`. Development session Claude aea5bfb9, 2026-11-21. This is a declared diagnostic
continuation variant on consumed dates. It is not a fitted candidate, not a production minimum-hold timer and not a profitable-policy claim.
Multiplicity: **one continuation contrast** (POST_KEEP_INITIAL vs POST_PI0) enters the experiment ledger.

## Code and inputs

| Part | File | Role |
|---|---|---|
| Runner | `predictfun/research/pf_entry_exposure.py` | manifest, three branches, readings, declared predicates |
| Selftest | `predictfun/research/pf_entry_exposure_selftest.py` | crafted tapes, fork_package end to end on the synthetic world, every writer |
| Mutants | `predictfun/research/pf_entry_exposure_mutants.py` | 42 planted errors; every one caught by a check (logs/mutants_exp) |
| Checks | `exposure_checks.py` (this directory) | checks A–G, falsifier, its own synthetic selftest |
| Manifest | `manifest.json` (this directory) | byte-equal to `data/predict_fun/research/entry_exposure_v1_20261121/manifest.json`; `run_part` refuses otherwise |
| Launcher | `launch_exposure.sh` (this directory) | pilot, rest, summarize, checks; never edited while it runs |

Frozen inputs, read and never edited:
- the label machinery (`pf_r2_labels.fork_for` / `pair_bound` / `teacher_factory`, `pf_r2_label_sample.arm_ledgers`, `pf_branch_engine`);
- the frozen PI0 teacher (`L.teacher_assets`);
- R2-F's accepted plan and dataset `2c62fc9b` (Oct 24–25, `real_files`);
- R3's accepted evaluation plan `8619eec5` and dataset `902526da` (Oct 26–27, day-guarded files).

## Population (the manifest)

- Per day and UTC hour, the plan's selected worlds are ranked by sha256(`entry_exposure_v1|` + canonical JSON of {day, m, t_ms}), with
  ties broken by that JSON. The four lowest are taken: 384 worlds and 768 side forks. Every hour holds 64 plan worlds, so the inclusion is 1/16.
- Each fork is the saved original-time neutral mandatory POST row (`is_mandatory`, pick `mandatory`, status ok).
- Exclusions are statuses that stay in every table: `world:no_path:missing_endpoint` (4 forks Oct 24–25, 8 forks Oct 26–27),
  `no_mandatory_row`, `mandatory_<status>` and `mandatory_row_not_unique`. The manifest holds 380 + 376 = 756 evaluable forks.
- Partitions: 12 of 32 worlds. Part 0 (Oct 24) is the cost pilot.
- No outcome, prediction or first-pull result enters the selection.

## Branches (from each rebuilt fork; its fingerprint must equal the saved one)

| Branch | At the fork | Continuation |
|---|---|---|
| WAIT_PI0 | WAIT | frozen PI0 on both sides (`pair_bound`) |
| POST_PI0 | POST | frozen PI0 on both sides (`pair_bound`); reproduction control |
| POST_KEEP_INITIAL | the identical POST | frozen PI0, except that a PULL of the live focal initial generation is replaced by KEEP |

- **Initial generation.** It is identified from the executed fork POST in the KEEP branch itself. It stays subject to post-only refusal,
  fills, the client structural rule, pending and acknowledgement timing, caps and END (size 30, cap 30, 58/37/100 ms).
- **After it closes.** PI0 governs re-entry and later generations. The opposite side is always PI0.
- **Binding and probes.** A fresh teacher is bound to each branch, with causal and binding probes every 35 callbacks. The shadow PI0 PULL is
  logged with each override; an override is never logged as a PI0 decision.
- **Statuses that are not ok** (none is skipped):
  - `fork_mismatch`, `pair_not_ok`, `missing_mark`;
  - `reproduction_failed`: the fills, decisions or requests hashes, the values, Y or the pair reconciliation differ;
  - `keep_baseline_mismatch`: with no override, KEEP must equal POST_PI0 exactly;
  - `prefix_mismatch`: with an override, the decisions before the first override and the fills up to its tick must be POST_PI0's, and the
    first difference must be that override;
  - `not_reconciled`, `initial_gid_mismatch`, `components_mismatch`.

## Values (common END mark, gross)

- `Y_PI0 = R(POST_PI0) − R(WAIT_PI0)`, which must equal the stored Y.
- `Y_KEEP_INITIAL = R(POST_KEEP_INITIAL) − R(WAIT_PI0)`.
- **`D_EXPOSURE = R(POST_KEEP_INITIAL) − R(POST_PI0)`** is the primary paired comparison; `Y_KEEP_INITIAL = Y_PI0 + D_EXPOSURE` is verified.
- Each POST arm's END value splits by fills: the initial generation (V_initial), the focal side's other generations, and the opposite side.
  So `D_EXPOSURE = D_initial + D_focal_rest + D_opposite`, and `D_rest = D_EXPOSURE − D_initial`.

## Readings (descriptive; per day and side; equal-world means over the common evaluable forks; denominators and every status shown)

**Per day and side:**
- statuses and overrides;
- the means and sums of the three contrasts and the components;
- positive, negative and exact counts, and meaningful counts (|value| > 1e-6 c);
- worst and best values, and |value| > 30 c counts.

**Per branch:**
- the initial fill probability, filled quantity and active time;
- the time to first fill (mean, median), the request and effect times, and the termination reasons;
- V_initial, the unfilled initial order and open inventory at END;
- per-fill execution edge and 50 / 100 / 1,000 ms venue-mid markouts inside the episode, with fill and quantity denominators;
- queue-ahead at submission and at the end (venue state; a diagnostic, never a policy input).

**Additional-fill forks** (KEEP's initial generation filled more than PI0's): their count, mean D and mean D_initial.

No interval, no significance claim and no favorable side. More fills alone is not success.

## Declared predicates (`predicates()`; days Oct 24, 25, 26, 27; both sides pooled per day)

- **Day sign:** the sign of mean D_EXPOSURE; |mean| ≤ 1e-9 counts as 0.
- **`diagnosis`:**
  - `incomplete` when a day has no evaluable fork;
  - else `not_identified_thin` when the pooled additional-fill forks are fewer than 30;
  - else `keep_better` when every day is positive, `pi0_better` when every day is negative;
  - else `not_identified_days_disagree`.
- **`more_execution`:** the pooled fill probability KEEP − PI0 is at least 0.05.
- **`initial_carries`:** |Σ D_initial| ≥ |Σ D_rest|.
- **`extra_fills_harmful`:** the mean D_initial over the additional-fill forks is below 0.

**Reading map.** The report quotes the sentence its computed predicate selects:

| diagnosis | sentence |
|---|---|
| keep_better | Keeping the initial order adds value over PI0's early pull on every consumed day under PI0's continuation: PI0 removes the initial order before useful execution develops. A next target must name this continuation and be paired with a controller that implements it (a new declared candidate with its own costs); this is not a profitable-policy claim. |
| pi0_better | PI0's early pull avoids value-losing exposure on every consumed day: keeping the initial order costs value, so later work focuses on adverse-selection discrimination, not longer resting, larger size or unconditional posting. |
| not_identified_days_disagree | Not identified: the consumed days disagree on the sign of the exposure effect, so no target or controller change follows from this comparison. |
| not_identified_thin | Not identified: fewer than 30 forks gain initial-generation fills by keeping the order, so the exposure effect is too thin to read. Changing Y to V_initial alone would retain the observed sparsity under PI0. |
| incomplete | Incomplete: a declared day has no evaluable fork; no reading. |

| predicate | true | false |
|---|---|---|
| more_execution | Keeping the order raises the pooled initial fill probability by at least 0.05. | Keeping the order raises the pooled initial fill probability by less than 0.05: resting longer rarely produces execution. |
| extra_fills_harmful | The additional fills lose value at the END mark on average. | The additional fills do not lose value at the END mark on average. |
| initial_carries | The initial generation's own fills carry most of the difference. | Most of the difference comes from the rest of the episode (re-entry, capacity, the opposite side). |

## Checks (`exposure_checks.py`, after the run)

| Check | Content |
|---|---|
| A, identity | committed manifest; partition hashes, members and sources; the 768 forks once each; exclusions carried |
| B, reproduction | every evaluable fork ok, every flag true, Y_PI0 equal to the stored Y |
| C, raw reconciliation | every ok fork from its own raw arms (values, fill windows, forced-POST gid, components, contrasts, no PULL of gid0, shadow log, binding and causal probes; the delivered-copy control fired) |
| D, summaries | recomputed from forks.csv |
| E, predicates | recomputed |
| F, resources | recursive over logs/: peak below 2,048 MiB, partition high-water ≤ 1,536 MiB, ≤ 2 core-hours |
| G, suites | the mutation suite judged against the final sources |

The falsifier corrupts one real input per check: a dropped fork; a flag; compensating components; a wrong but valid gid; a fill price; a
summary mean; the diagnosis; an over-budget job; a copied mutant record. A check that cannot run is false.

## Budget and stop rules

- 2 aggregate core-hours over the four slots claude / claude2 / claude3 / claude4, one process each. RSS 1,536 MiB checkpoint and 2,048 MiB
  hard. 120 s per fork package; per partition, 600 + 360 s × 32 worlds.
- **Pilot first (part 0).**
  - The projection is the pilot's CPU × 12 plus the round's CPU so far.
  - If the projection exceeds 2 core-hours, the rest is not run. The receipt reports `incomplete` with the projection.
  - A `STOP_RESOURCE` partition stops the round with its status. Nothing is shortened, resampled or relaxed after a result.
- No fit, no label expansion, no new date and no END extension. Consumed days only.

## Review 3d8bf5ad's nonblocking items (folded into this round's receipt)

1. **Wording.** The audit's "rare execution only" recommendation is superseded by "changing Y to V_initial alone would retain the observed
   sparsity under PI0". This is not a market-wide scarcity claim. The audit receipt stays as provenance.
2. **The audit's `cancellation` flag.** It is an aggregate attenuation screen (Σw|Y| against Σw|V_initial|), not an order-cancellation
   measure. The receipt shows the review's rowwise offset measure, recomputed from the audit's saved rows, together with this experiment's
   within-fork offset of D_initial by D_rest.
3. **Checker controls.** The audit checker is not reused. This round's checker carries source-to-output reconciliation, a
   compensating-component corruption and a wrong-but-valid gid control (check C, falsifier, checker selftest).
4. **Costs and labels.** The receipt's core-hours use check F's recursive aggregation. The audit's top-5 concentration is labelled
   "unweighted rows" wherever it is quoted.
