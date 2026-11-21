# PI0-ADMISSION-EXPOSURE-v1 — declaration (frozen before any branch runs)

Review `99fb165d` (one next priority) on target `a44f7dd8`. Development session Claude aea5bfb9, 2026-11-21. This is one diagnostic with
**zero fitted candidates**, on consumed Oct 26–27 only. It covers:
- the exposure comparison on orders PI0 itself chose to post;
- the corrected mean-forecast metrics;
- the declared exposure × admission cross-tab;
- the superseding ENTRY-RETENTION reading.

## Code and inputs

| Part | File | Role |
|---|---|---|
| Module | `predictfun/research/pf_pi0_admission.py` | census, selection, factual and KEEP arms, stages, targets, measurements, predicates (its header is the full specification) |
| Selftest | `predictfun/research/pf_pi0_admission_selftest.py` | census/select, the factual loop against `pf_branch_engine.step`, `world_rows` on real-engine synthetic generations with refusals, the measurements, the writers |
| Mutants | `predictfun/research/pf_pi0_admission_mutants.py` | 29 planted errors, every one caught by a check |
| Checks | `admission_checks.py` (this directory) | checks A–H, the falsifier, its own synthetic selftest |
| Manifest | `manifest.json` (this directory) | byte-equal to the data copy; `run_part` refuses otherwise |
| Launcher | `launch_admission.sh` (this directory) | pilot, rest, measure, checks |

**Frozen inputs, read and never edited:**
- the R4E replay traces (`adv2_entry_smooth_v1_20261121/replay_v1`), each partition byte-equal to its committed record;
- R3's evaluation plan and the day-guarded files;
- the frozen PI0 (`pf_adv2.Teacher0`) and the predictor;
- `hold_from` / `target`;
- ENTRY-EXPOSURE's v2 outputs and ENTRY-RETENTION's outputs, used for the cross-tab and RETENTION v2.

## Population (the manifest)

- **Generations.** There are 689 PI0 POST generations across the 380 ok R4E worlds; 4 world exclusions are kept as statuses. A generation
  is an absent-state POST that created a gid, whatever followed it: refusal, no fill, structural removal, partial fill or no review.
- **Cells.** A cell is the decision's UTC day × side × UTC hour, using t_ms + decision time.
- **Selection.** Each cell keeps its four lowest sha256(`pi0_admission_exposure_v1|` + canonical JSON of {day, m, t_ms, side, gid,
  decision_ms}), with ties broken by that JSON. That gives 329 generations in 93 cells and 155 worlds.
- **Weights.** Each selected generation carries N_cell / min(4, N_cell); the weights sum to 689.
- No execution, outcome, review action or score enters the selection.

## Arms, stages, targets

| | Definition |
|---|---|
| factual | PI0's episode rerun from the world start, which must equal the saved R4E trace in hashes, value and teacher calls |
| POST_KEEP_SELECTED | from the snapshot just before the selected decision: PI0's own POST, then PI0 except that a PULL of the live selected gid is replaced by KEEP |

In POST_KEEP_SELECTED, other sides and generations follow PI0. Overrides are counted from the log, never in `decide`.

**Values (END mark, gross).** D_EXPOSURE = R(KEEP) − R(factual) = D_selected + D_focal_rest + D_opposite.

**Stages.** The stages are entry (PI0's own POST decision), first_review and first_pull of the selected gid in the factual run. Each records the
107 inputs and their hash, the PF receipt time, the venue state, the frozen moments (reproduced) and the hidden-notice check.

**Targets.** They come from `hold_from`, which must be equal to the KEEP arm's selected-gid fills. The first-pull reconciliation is against
D_selected and the factual selected fills.

**Statuses.** None is dropped: `factual_not_reproduced`, `horizon_mismatch`, `not_admitted`, `gid_mismatch`, `keep_not_admitted`,
`prefix_mismatch`, `not_reconciled`, `target_not_equivalent`, `components_mismatch`, `causality_failed`, `moments_not_reproduced`,
`binding_failed`, `pull_not_reconciled`, `world:*`.

## Measurements (declared; weighted by inclusion and unweighted; no interval, no significance)

**Economic**, per day × side and pooled:
- D_EXPOSURE and its components: means, sums, gains, losses and zeros (floor 1e-6 c);
- severe counts (< −30 c), the top-5 |D| share over unweighted rows, and worlds;
- the selected generation's lifetime, fill probability, filled shares and END value in each arm.

**Forecasts**, per stage, with MAE as a secondary, median-type diagnostic only:
- **MSE** of V, of zero and of edge × E[q];
- mean error;
- Brier of P, and of the pooled rate;
- conditional-on-fill q and qm residuals;
- probability bins;
- per side and day.

**First-review selection.** PULL is compared against 500 matched draws (seed 20261121, counts matched within day × side × UTC hour, weights
carried). The receipt reports the strata and rows that can change assignment. A thin stage (below 30 selected or unselected) or one with
degenerate strata is **not identified**, not "no signal".

**Entry → first review**, on forks modeled at both stages:
- input-group bit changes;
- new observations per source: pf_age, bn_age and bnd_age are in ms, and a source has a new observation when its age grew by less than the
  elapsed time minus 1 ms; the PF receipt time is also recorded;
- dV with and without any new observation.

**Cross-tab** (adaptive accounting, distinct from the new sample):
- the 768 ENTRY-EXPOSURE forks by ENTRY-RETENTION's entry admission: would_POST, would_WAIT (model), fallback (not model), excluded;
- n, worlds, D_EXPOSURE, D_initial and D_rest means and sums, and concentration.

**RETENTION v2** (supersedes the MAE-gated reading in band):
- per stage × side × day: MSE of V, of zero and of edge × E[q], the mean error and Brier;
- flips on transitions where both stages are model-valid;
- MAE kept as secondary.

## Declared predicates and reading map (`predicates()`)

- **Per day:** the sign of the weighted mean D_EXPOSURE, with |mean| ≤ 1e-9 counted as 0.
- **Thin:** fewer than 30 selected generations gain selected-gid fills by keeping.

| reading | when | sentence |
|---|---|---|
| admitted_longer_beneficial | both days positive, not thin | PI0's own admitted orders gain from longer exposure on both consumed days: the next declared candidate should target a consistent conditional retention policy with a matching controller. |
| admitted_longer_harmful | both days negative, not thin | Longer exposure of PI0's own admitted orders loses on both consumed days: later work should focus on admission and remaining-value discrimination, not unconditional duration. |
| not_identified | otherwise | Not identified on this sample: the consumed days disagree or fewer than 30 selected generations gain fills by keeping; no target is chosen from it. |

| predicate | true | false |
|---|---|---|
| review_discriminates | PI0's first-review pulls of its own admissions select worse remaining value than matched random draws. | No first-review selection advantage is demonstrated on this sample (not identified when thin or degenerate). |
| mse_beats_zero (entry) | At PI0's own admissions, V's squared error is below zero's. | At PI0's own admissions, V's squared error is not below zero's. |
| retention_v2 entry_mse_beats_zero | At the forced entries, V's squared error is below zero's: the earlier MAE-based failure reading does not hold for the mean forecast. | At the forced entries, V's squared error is not below zero's. |
| retention_v2 review_mse_beats_zero | At the forced first reviews, V's squared error is below zero's. | At the forced first reviews, V's squared error is above zero's: retention calibration remains a problem. |

## Checks (`admission_checks.py`, after the run)

| Check | Content |
|---|---|
| A, identity | committed manifest; the selection recomputed from the committed R4E records; partition hashes, sources and R4E records; every selected generation |
| B, arms | independent of the rerun, against the saved R4E trace: the POST of gid; END and mark; the KEEP prefix to its first override; both ledgers from fills; values and components from fills; binding and causal |
| C, targets | the held fills equal the KEEP fills; stage targets from the held fills; the unavoidable value; the first-pull reconciliation |
| D, forecasts | input hashes; moments recomputed, bit-equal |
| E, measurements | every output recomputed |
| F, predicates | recomputed |
| G, resources | ≤ 1 core-hour (recursive); peak < 2,048 MiB; partition high-water ≤ 1,536 MiB |
| H, suites | judged |

The falsifier corrupts one real input per check. A check that cannot run is false.

## Budget and stop rules

- **Limits:** 1 aggregate core-hour; the four slots, one process each; RSS 1,536 / 2,048 MiB; a 120 s guard per selected generation, with each
  partition bounded at 600 + 360 s per world.
- **Pilot first (part 0).** The projection is the pilot's CPU × (the selected generations / the pilot's) plus the round's CPU so far. The
  projection log is **write-once** (review 99fb165d item 5). If the projection exceeds 1 core-hour, the rest is not run and the receipt
  reports the limitation.
- **Not allowed:** outcome-dependent thinning, any resource increase, fit, label build, new date, END extension or minimum-hold timer.

## Review 99fb165d's findings, folded into this round

1. **The MAE gate.** It is superseded by RETENTION v2 (MSE, calibration and side/stage detail); MAE is kept as secondary. The perfect-mean
   mixture is a selftest control. The ENTRY-RETENTION receipt and its gate stay as provenance.
2. **Model-valid masks.** Transitions use both-modeled forks (RETENTION v2 flips; this round by construction). The cross-tab separates
   model-valid WAIT from fallback.
3. **New observations.** These are classified per source from ages (ms) and the PF receipt time, so a group-change indicator is not
   read as new information.
4. **Wording.** "Not demonstrated on this forced sample". This round measures actual admissions.
5. **The pilot record.** It is write-once.
