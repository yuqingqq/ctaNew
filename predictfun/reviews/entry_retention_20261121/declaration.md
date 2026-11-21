# ENTRY-RETENTION-DISCRIMINATION-v1 — declaration (frozen before any measurement)

Review `67a0d9f7` (one next priority) on target `73f6cfa6`. Development session Claude aea5bfb9, 2026-11-21. This is one bounded diagnostic
with **zero new fitted candidates**, on consumed dates only (Oct 24–27). It enters the experiment ledger as one diagnostic.

## Code and inputs

| Part | File | Role |
|---|---|---|
| Module | `predictfun/research/pf_entry_retention.py` | stages, prefix, neutral-hold targets, measurements, predicates (its header is the full specification) |
| Selftest | `predictfun/research/pf_entry_retention_selftest.py` | crafted tapes, `fork_stages` end to end on the synthetic world, the measurements, every writer |
| Mutants | `predictfun/research/pf_entry_retention_mutants.py` | 32 planted errors, every one caught by a check (`logs/mutants_ret`) |
| Checks | `retention_checks.py` (this directory) | checks A–H, the falsifier, and its own synthetic selftest |
| Launcher | `launch_retention.sh` (this directory) | pilot, rest, measure, checks; never edited while it runs |

**Frozen inputs, read and never edited:**
- ENTRY-EXPOSURE-v1's committed manifest and its 12 saved partitions, hash-bound;
- the label machinery (`fork_for`, the `pair_bound` arm loop, `hold_from`, `target`, `hidden_copy`);
- the frozen PI0 teacher and models through `pf_r2_teacher.predict_rows` / `pf_pipeline_bench.predict`, unchanged;
- the support adapter and the plans' day identities.

`run_part` refuses unless the exposure manifest, this module and this declaration are committed in HEAD.

## Population and stages

**Population.** Exactly the 768 manifest forks, with the 12 `world:no_path:missing_endpoint` statuses carried. The other 756 forks are
evaluable when every check passes; any other status is kept as its own status.

**Stages.** Each fork yields at most one row per stage, and stages are evaluated separately:

| Stage | Definition |
|---|---|
| entry | the shadow unforced PI0 decision on a copy of the absent fork state (would_POST / would_WAIT, plus its status) |
| first_review | the first actual live PI0 callback on the forced initial gid in POST_PI0's prefix |
| first_pull | its first discretionary PULL (`same_as_review` when the first review pulls) |

**The prefix.** It runs as `pair_bound` runs the POST arm and stops after the first pull's tick, the initial generation's closure, or END.
Up to its stop it must equal the saved arm in decisions, requests and fills. Its callback count must be the true one, and its probe count
must be ceil(callbacks / 35).

## Targets (matched to the heads' training targets, `pf_r2_shadow.target`, END mark, gross)

**Reference path.** The neutral hold of the forced initial generation (`hold_from`). Its gid0 fills must equal the saved KEEP_INITIAL gid0
fills (`target_not_equivalent` otherwise).

| Stage | Fills counted |
|---|---|
| entry | at or after activation |
| first_review / first_pull | at or after the decision's modeled cancellation effect (the engine's request rule) |

**Definitions.** `qm` is measured against the stage's own pf_mid. The unavoidable value is the entry value minus the stage value.

**First-pull reconciliation.** At the first pull, the held value and quantity after the effect must equal D_initial and the KEEP − PI0
initial quantity. The POST_PI0 initial fills must equal the held fills before the effect.

## Measurements (declared; descriptive, adaptive development days, no interval, no significance)

Every table covers all eligible rows (fork and stage ok, a model decision) and shows every exclusion count.

**Calibration**, per day × side × stage and pooled:
- fill rate and mean P;
- Brier of P, and of the stage's pooled rate;
- E[q] against q, with its MAE;
- the MAE of V, of zero and of the mechanical edge × E[q], each against the realized value;
- the q | fill and qm | fill errors on filled rows.

**Probability bins:** 0, .01, .02, .05, .1, .2, .4, 1.

**Decision tables**, per stage × V sign × shadow entry admission:
- benefit, loss and zero counts and sums (floor 1e-6 c);
- severe counts (< −30 c);
- the top-5 |value| share, over unweighted rows;
- worlds.

**Selection** (entry: would_WAIT; first_review: PULL):
- 500 matched random draws with seed 20261121 + stage index, drawn within day × side × UTC hour from the same eligible rows;
- the selected mean value's percentile is reported; lower means the protective action selects worse exposure;
- degenerate strata are counted;
- a stage is thin below 30 selected or 30 unselected rows.

**Changes from entry to first_review:**
- changed-input counts (float32 bits) per input group;
- dP, dE[q], dE[qm], d edge and dV;
- the V flip, overall and on forks whose market_flow and fair inputs are unchanged.

**Input groups**, a name rule:

| Group | Inputs |
|---|---|
| time | t_rel, act_rel, t_to_end, fp_phase, o_age, o_horizon, ctx_generation_age_ms |
| fair | fair_edge, dfv250, dfv1000, gap_adv, sens, pbn_dist, sigma |
| order_state | o_*, fp_*, ctx_* not listed under time |
| market_flow | every other input |

**Fill-time execution edge** of the held initial fills: sign × (venue mid at the fill − price) × q, with availability and quantity
denominators. The END-mark value is named `end_value`.

## Declared predicates and reading map (`predicates()`)

- **`discriminates[stage]`:** the stage is not thin and its percentile is at most 0.05.
- **`value_fails[stage]`:** the pooled MAE(V) is at least MAE(0).
- **`flip_share`:** among forks with both stages modeled, the share where V_entry > 0 and V_review < 0.

The report quotes the sentence of each reading that holds:

| reading | when | sentence |
|---|---|---|
| entry_identifiable | entry discriminates | Negative exposure was already identifiable at entry: PI0's shadow admission separates the forced entries' value, so entry admission and its label should be aligned with the intended resting policy. |
| review_protective | first_review discriminates | Updated forecasts discriminate harmful remaining exposure at the first review: those protective cancellations should be kept, and selective longer resting investigated. |
| forecasts_fail | value_fails at entry or first_review, or flip_share ≥ 0.5 without review_protective | Actual-state value forecasts fail (no better than zero, or a mechanical flip without economic separation): the target / representation / model mismatch should be fixed before another conditional-retention fit. |
| not_distinguishable | none of the above | This sample cannot distinguish these explanations; no target is chosen from it. |

## Checks (`retention_checks.py`, after the run)

| Check | Content |
|---|---|
| A, identity | committed manifest; partition hashes, sources, exposure partition hashes and members; three stage rows per fork; exclusions carried |
| B, prefix | independent of the rerun, from the saved POST_PI0 ledger: END 5,000, one mark and settings across arms (review 73f6cfa6 item 3); first review and first pull in time and action; the effect rule; the true callbacks and the probe count |
| C, targets | the held fills equal the KEEP gid0 fills; every stage target recomputed from the held fills; the unavoidable value; the first-pull reconciliation |
| D, forecasts | input bit hashes; P, E[q], E[qm], V recomputed by the frozen predictor, bit-equal; V = edge × E[q] + E[qm] |
| E, measurements | every output recomputed from the partitions; the pooled Brier and MAEs recomputed from stages.csv |
| F, predicates | recomputed |
| G, resources | recursive over logs/: peak below 2,048 MiB, partition high-water ≤ 1,536 MiB, ≤ 1 core-hour |
| H, suites | the mutation suite judged against the final sources |

The falsifier corrupts one real input per check, including a missing callback, a wrong END, compensating target and unavoidable values, and a
wrong but valid gid. A check that cannot run is false.

## Budget and stop rules

- **Limits:** 1 aggregate core-hour; the four slots claude / claude2 / claude3 / claude4, one process each; RSS 1,536 MiB checkpoint and
  2,048 MiB hard bound; a 120 s per-world work guard, with each partition bounded at 600 + 120 s × 32 worlds.
- **Pilot first (part 0).** The projection is the pilot's CPU × 12 plus the round's CPU so far. If the projection exceeds 1 core-hour, the
  rest is not run, and the receipt reports the cost limitation and an `incomplete` status. A `STOP_RESOURCE` partition stops the round with
  its status.
- **Not allowed after a result:** thinning, resampling or relaxing anything.
- **Out of scope:** any fit, new label, new date, END extension, residual-queue bonus or new cost assumption. No held-market sensitivity
  rescoring is run; it is optional and not declared here. The RICH / SMOOTH heads are not scored.

## Review 73f6cfa6's nonblocking items, folded into this round

1. **Wording.** The exposure result is restated: suppressing **every** discretionary PULL of the forced initial generation (the entire
   initial-generation retention policy) loses on these forced forks. It does not show that the first pull was necessary. The "86 ms" is
   the mean removal-effect delay after activation among the 735 orders with a request. Shadow entry admission, first-review status and
   first-pull time are reported separately here.
2. **Forecast components.** This measurement compares the forecast components against matched full-opportunity targets, with full
   107-input vectors. POST and RETAIN scores are not compared as one forecast, because their targets differ.
3. **Horizon and callback checks.** END, mark, settings and callback/probe denominators are enforced in check B, with the wrong-END and
   missing-callback falsifiers.
4. **The override counter.** Stage capture happens in the teacher's `fn` (actual callbacks only), and `decide` stays side-effect free. No
   counter of actions is kept outside the log.
5. **Naming.** A fill's END-mark value is named `end_value`. The fill-time venue-mid execution edge is added, with its denominators.
