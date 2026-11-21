# R4E ADV2-ENTRY-SMOOTH-v1 — the freeze: one declared entry candidate, its evaluation on the existing labels and the bounded own-state comparison, before any fit

Review **371aab568277dea8ed15c0e386b8c07e7bacd340** accepted R3 (`76be4960`) as a reliable negative result and named one next priority: test
whether lower entry-model complexity improves useful selection on the existing labels, with PI0 retention held fixed. This is a **new
declared candidate**, not a repair of the R3 freeze. Its multiplicity is one more primary candidate with two new fitted heads.

Oct 26–27 informed this design (R3), so every result here is **adaptive consumed-day development**, despite the new freeze.

| Module | Role | Selftest / mutation check |
|---|---|---|
| [pf_adv2_entry.py](../../research/pf_adv2_entry.py) | the declared delta, two POST fits on R3's verified Oct 24–25 rows, the four policies | `pf_adv2_entry_selftest.py` / `pf_adv2_entry_mutants.py` |
| [pf_adv2_entry_eval.py](../../research/pf_adv2_entry_eval.py) | the offline entry diagnostic on R3's accepted Oct 26–27 POST rows, reused unchanged | `pf_adv2_entry_eval_selftest.py` / `pf_adv2_entry_eval_mutants.py` |
| [pf_adv2_entry_replay.py](../../research/pf_adv2_entry_replay.py) | the 384-world own-state comparison and at most 768 reached entry pairs | `pf_adv2_entry_replay_selftest.py` / `pf_adv2_entry_replay_mutants.py` |
| [r4e_checks.py](r4e_checks.py) | the declared predicates A–I and their falsifier | the falsifier runs with the checks |

## The candidate (review items 1–2)

- **Inputs and rows:**
  - two squared-error POST regressors, bid and ask, of the existing paired label Y = R(POST) − R(WAIT) under frozen pi0;
  - RICH's 107 inputs with native NaN;
  - every POST row of R3's Oct 24–25 extraction, with zeros and tails kept. Its files must equal the committed receipt manifest.
- **Weights:** the stored w, with one constant n / sum(w) per head.
- **Parameters:** pf_adv2.PARAMS with exactly one change, `min_data_in_leaf` 200 → 1,000, and **50 rounds** instead of 200. The hash is
  `ce5260158d2abced`… (`PARAMS_SHA`). This is one smoothing package, not a search.
- **Controller:**
  - absent: POST iff the supported prediction is > 0, otherwise WAIT. An exact tie WAITs.
  - live: the unchanged frozen PI0-50.
  - support: R3's POST support, unchanged. No new cap, eligibility rule, cooldown, minimum lifetime or score threshold.
- **Policies:** SMOOTH, ENTRY_ONLY (the primary comparator: R3's RICH POST entry with the same PI0 retention), PI0 and N0.

## Evaluation on the existing labels (item 4)

- **Rows:** R3's accepted dataset `902526da`…, POST rows only. Before any reading, the rows' keys, Y and w, and R3's RICH predictions
  recomputed here, must equal R3's saved `predictions.csv.gz` exactly.
- **Readings and screens:** per day and side, as in R3, for SMOOTH against RICH on identical rows.
  - Readings: MSE / MAE against zero and the training mean, local value / regret, admitted and missed values, and |Y| > 30 c.
  - Screens: plan v2 with 200 originals / 20 meaningful originals.
  - Control: 500 draws, seed 20261121, scopes (day, side) and (day).
- **Calibration:** each bin carries `w_sum`, so pooling by weights is exact (checklist item 5).
- **Strata (counts, no sparse-cell claim):** generation 1 / 2 / ≥ 3, source, horizon and the other side's state.
- **Predicates, reported separately per day / side and candidate:**
  - `admissions`;
  - `post_value_positive`: the selected POST group's weighted mean label > 0;
  - `screen_pass`;
  - `above_draw_mean` / `above_draw_p95`.

  Lower MSE alone is not success. **Zero admissions means no demonstrated entry skill**, never a successful selector. No side or
  threshold is chosen afterwards.

## The bounded own-state comparison (items 5–6) and retained evidence (checklist item 4)

- **Policies and population:** R3's identical nested 384-world manifest; each partition's members must equal R3's. SMOOTH, ENTRY_ONLY, PI0
  and N0 each run on their own state.
- **Controls:** ENTRY_ONLY, PI0 and N0 are **re-run**, not reused, and each episode must equal R3's by value and decision / fill hashes.
  This also serves as a reproduction control of the new module.
- **Retained evidence:** every episode keeps:
  - its fills, sizes, decisions and fallbacks;
  - its ledger, mark, END and settings;
  - its calls / binding and probe counters.

  The checker recomputes the ledger and the exact required callbacks with the frozen label validator's functions.
- **Reached entry states:** per world and side, SMOOTH's eligible entry decision of minimum sha256 under
  `adv2_entry_smooth_v1_reached_20261121`. This gives at most 768 pairs, each with its inclusion count or `no_eligible_decision`.
  - The fork is rebuilt with SMOOTH as the roll-in and fingerprint-checked.
  - Its 107 inputs are rebuilt by `fork_inputs` and must equal the decision's own.
  - The POST − WAIT label comes from `pair_bound` under frozen pi0, audit every 35, with arm ledgers attached.
  - Each row has the label-row form, so `pf_r2_label_sample._pair_bad` validates it.
  - Both the new and the old POST heads predict on the identical inputs: the common-state comparison that R3's "3 of 4" lacked.

## Checklist items carried (review 371aab56)

- Item 2: the receipt names the POST advantage at the post and the KEEP advantage at the review as different targets. Hybrid losses are
  not summed into a decomposition.
- Item 3: the strata above, and reached-state comparisons on common states.
- Item 4: the retained traces and their recomputation.
- Item 5: calibration `w_sum`, plus R3's pooled table recomputed by weights in the receipt.
- Operational:
  - `run_part` accepts an existing record only with this round's members, sources, fit, plan and bytes;
  - the summary checks complete membership;
  - this round has its own launcher, and no launcher is edited while it runs.

## Order and budget

1. This freeze, with selftests and mutation suites.
2. The two fits, then the model-hash commit.
3. The evaluation and the replay. Both refuse until HEAD holds `fit/fit.json`.

Partition 0 (32 worlds) is the cost pilot. The round's ceiling is **8 core-hours** (check I). If the projection exceeds it, explicit stop
statuses are kept, and no smaller population is chosen after outcomes are seen.
