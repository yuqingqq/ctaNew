# PI0-MOVEMENT-CALIBRATION-AUDIT-v1 — declaration

2026-11-21 11:55Z, session aea5bfb9. Review 5a8ee6f5's one next priority: a bounded diagnostic of PI0's frozen movement forecast on the
existing admitted and forced cohorts, with that review's four reporting / validator amendments in the same round. The review states that no
further USER approval is needed for this consumed-day package. Frozen before any new table is computed: no fill-time mid has been looked up,
and no moments, strata, pre / post split, concentration or evidence table of this package exists.

The specification is the header of `predictfun/research/pf_movement_audit.py`. This file restates the populations, the reading map and
the budget. Where the two differ, the header governs.

## What is frozen

| Item | File |
|---|---|
| Module | `predictfun/research/pf_movement_audit.py` |
| Selftest | `predictfun/research/pf_movement_audit_selftest.py` |
| Mutation harness | `predictfun/research/pf_movement_audit_mutants.py` |
| Checks, falsifier, checker selftest | `movement_checks.py` (A-I) |
| Launcher | `launch_movement.sh` (pilot, project, rest, measure, checks) |
| Write-once pilot rule | `tools/pilot_projection_mov.py` |

Mutation results:
- 64 / 64 mutants are caught on the final sources (`logs/mutants_mov`, `logs/mutants_mov_summary.log`).
- A first run caught 63 / 64 and is kept in `logs/mutants_mov_run1`. The escape was the measure-time membership refusal. Its fixture
  failed on the partition hash before membership was checked, so the guard itself was never tested. The fixture now binds a complete-hash,
  incomplete-membership cohort.

Nothing else is re-simulated. The teacher, features, latency, size, cap, structural rules, terminal mark and original END are those of the
saved rows.

## Populations (consumed days only; never pooled)

| Cohort | Rows | Days | Weights | Source (hash-bound) |
|---|---:|---|---|---|
| Admitted | 329 generations (all stage statuses kept) | Oct 26-27 | manifest inclusion weights, summing to 689 | `data/predict_fun/research/pi0_admission_v1_20261121/parts` (12) ← `predictfun/reviews/pi0_admission_20261121/manifest.json` sha256 `3c5bf097…1696c` |
| Forced | 768 forks (384 worlds × 2 sides; 756 evaluable) | Oct 24-27 | none (unweighted) | `data/predict_fun/research/entry_retention_v1_20261121/parts` (12) ← `predictfun/reviews/entry_exposure_20261121/manifest.json` sha256 `6cf2c624…b839c` |

Binding and membership:
- Admitted rows are bound to their manifest rows by exact key: weight, n_cell, cell and part must all match, or the row's status is
  `weight_mismatch`.
- Admitted membership must equal the manifest's selected keys. Forced membership must equal the manifest's world × side forks.
- Incomplete membership refuses the measurement.

How the cohorts are shown:
- The forced cohort is shown for Oct 24-27, per day, and for the Oct 26-27 overlap.
- It is also split by shadow entry admission: would_POST, would_WAIT, or fallback (a non-model entry decision).
- Entry, first review and first pull are separate estimands. A first pull that is also its first review is counted (`same_as_review`), and
  is the same observation.

## Measurements

The units, statuses and formulas are in the module header.

**Accounting.** These rowwise identities are verified, with explicit missing and zero-P statuses:
- `V − value = edge (E[q] − q) + (E[qm] − M)`
- `E[qm] − M = (P − z) mhat + z (mhat − M)`
- `E[q] − q = (P − z) qhat + z (qhat − q)`

**Moments.** Reported for each:
- mean and bias, MSE, and MAE (secondary);
- Brier against the pooled fill rate;
- zero and mechanical (edge × E[q]) baselines;
- weighted and unweighted n, unique worlds and Kish n;
- severe losses (value < −30 c) with their signed sums.

**Strata.** Marginal views only, with no Cartesian product:
- Decision-state strata:
  - remaining horizon (0,100], (100,500], (500,2000], (2000,5000] ms;
  - queue ratio fp_queue / fp_remaining: 0, (0,1], (1,3], >3, or missing / invalid;
  - ordinal: 1, 2, >=3.
- Forecast bins: P, E[qm], E[q] and V.
- Outcome stratum: the first counted fill's delay class, [0,100], (100,500], (500,2000], >2000 ms, or no fill.
- Explanatory stratum: the venue queue, at reviews only. It is never a decision input and is never read by a predicate.
- A cell is thin below 30 opportunities, or below 10 fills for a conditional statistic.

**Pre / post-fill split.** For every counted fill, `sign q (END_mark − decision_mid) = sign q (fill_mid − decision_mid) + sign q (END_mark − fill_mid)`.
- Admitted mids are looked up on the tapes of the selected worlds that have held fills (56 worlds in 12 partitions). The lookup uses the
  frozen identity-checked loader, the `pf_adv2_replay.mid_at` convention, and the frame's venue time, receipt time and age. The world's
  canonical mark must equal the row's.
- Forced mids come from ENTRY-RETENTION's retained fill-time edges. Their frame time was not retained.
- Missing references are statuses (`reference_missing`, `reference_unavailable:<why>`, `reference_out_of_time`). They are never dropped,
  never moved to another clock, and never moved past END.

**Other tables:**
- Concentration: weighted and unweighted top-5 generation and world shares of |D_EXPOSURE|, and of the admitted movement error per stage.
- Signed day × side contributions.
- Sources (entry → first review):
  - a new PF frame, by pf_age;
  - a new Binance **bookTicker** update, by bn_age;
  - a new Binance depth snapshot, by bnd_age;
  - a new PF receipt;
  - the decision-mid change, with its direction and magnitude.
- The retained 500-draw first-review control. Each review's UTC hour is derived from its own decision time.

## Reading map (computed by `predicates()`; the declared reading is the admitted entry stage, weighted)

Notation: u = E[qm] − M per opportunity; c = mhat − M on fills.

| Question | Answer | Rule |
|---|---|---|
| (i) tail or broad | broad | u's weighted mean > 0 in ≥ 3 of the 4 day × side cells, and the weighted top-5 world share of \|Σ w u\| < 0.5 |
| | concentrated | that share ≥ 0.5, or u > 0 in ≤ 1 cell |
| | inconclusive | otherwise |
| (ii) decision state or fill timing | decision_state | D ≥ 1.5 T, or T unavailable |
| | fill_timing | T ≥ 1.5 D, or D unavailable |
| | both | otherwise |
| | inconclusive | both D and T unavailable |
| (iii) neutral vs admitted | material | c_admitted (weighted) and c_forced (Oct 26-27, unweighted) differ in sign, or by ≥ 0.5 × the larger magnitude |
| | similar | otherwise |
| | inconclusive | fewer than 10 fills in either cohort |
| (iv) populated | populated | every decision-state stratification has ≥ 2 cells with ≥ 30 opportunities and ≥ 10 fills |
| | thin | otherwise |

For (ii):
- D is the largest range of the weighted mean c across one decision-state stratification.
- T is the range across the delay classes.
- Both use only cells with ≥ 10 fills, and each needs at least 2 such cells.

**Recommendation:**
- `calibration_candidate_defensible`: (i) is broad, (iv) is populated, and (ii) is decision_state or both.
- `representation_or_data_needed`: (i) is concentrated, (iv) is thin, or (ii) is fill_timing.
- `inconclusive`: otherwise.

The first review and first pull are computed by the same function and are secondary.

What the readings cannot say:
- There is no interval, no validation and no profitability claim. The generations are related, within two consumed days.
- The forced cohort is a differently selected reference.
- Fill-conditional displays are explanatory, not a training population.
- Delay classes are realized outcomes, never features.
- An oracle corrected with the same outcomes is not a model score.
- Nothing is fitted, and no threshold is chosen. A `calibration_candidate_defensible` reading only says that a small chronologically
  fitted candidate could be declared next.
- New observations alone do not attribute the POST-to-PULL reversal to changed danger.

## Budget and pilot (fixed before the run)

Limits:
- One aggregate core-hour.
- At most four single-core workers on slots claude to claude4.
- A per-process high-water checkpoint of 1,536 MiB and a hard bound of 2,048 MiB (the frozen loader's Guard and Sampler; run_light's caps
  unchanged).

The pilot is partition 0 alone. `tools/pilot_projection_mov.py` writes the projection once:

    max(pilot CPU × partitions with fills, pilot CPU × worlds with fills / pilot worlds) + the round's CPU so far

Partitions 1-11 run only if all of these hold:
- the pilot record exists;
- its job returned 0;
- its peak and high-water are below 1,536 MiB;
- the projection is at most 1 core-hour.

Otherwise the run stops and the reason is reported. A partition that stops (`STOP_RESOURCE`) leaves its fills `reference_unavailable`
(counted). No thinning depends on outcomes.

Order: measure → checks and falsifier → tables / report → state.

## Review 5a8ee6f5 findings 1-4 (bundled)

1. The bookTicker label replaces `new_bn_age`, and the module header documents the three clocks:
   - pf_age: ms since the latest known PF frame;
   - bn_age: ms since the latest Binance bookTicker update;
   - bnd_age: ms since the latest known Binance depth snapshot.

   Trades are not counted.
2. Wording and units:
   - Use "useful probability discrimination with calibration error".
   - qm is quantity-weighted decision-to-END movement, pre- and post-fill.
   - Units are c per opportunity (unconditional), c per filled opportunity (conditional), and c per share (pre / post).
3. Weighted and unweighted concentration, and signed day × side contributions.
4. Rows are bound to the manifest by exact key in `adm_rows`, independently in check B, and as a falsifier case (B_weight). The control's
   hour is derived from each review's own decision time, and the count equal to the admission cell is recorded.

## Controls

Selftest:
- Positive controls:
  - the exact calibrated mixture;
  - independent probability versus conditional-head bias;
  - an equal total movement split into pre- and post-fill;
  - a quote-only update;
  - a clock-only transition.
- Refusals:
  - wrong row weights, cells or parts;
  - a V in other units, and a changed value;
  - wrong ask-side signs;
  - a fill moved before the effect;
  - a moved END mark;
  - fills of another gid;
  - late, missing or unavailable references;
  - a substituted partition;
  - incomplete membership;
  - the venue queue in a decision-time stratum;
  - an overwritten measurement;
  - a run before the freeze.

The checker's falsifier corrupts real copies with one case each:
- A substituted partition.
- A doubled weight (rehashed).
- A dropped row.
- V in other units.
- An edge sign.
- A venue queue.
- A moved fill time.
- A moved END mark.
- A moments count.
- A reading.
- A reproduced number.
- An over-budget job.
- A copied mutant record.
