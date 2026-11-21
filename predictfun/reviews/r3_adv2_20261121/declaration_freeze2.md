# R3 ADV2-RICH v1 — freeze 2: the candidate, the evaluator, the replay and the declared checks, before any fit or Oct 26–27 label read

Review 315979d2's contract, implemented. Freeze 1 (`551185f2`) holds the evaluation sample. This commit holds every module that reads a
training row, fits, predicts, evaluates or replays, with its selftest and mutation check, plus the round's declared checks
([r3_checks.py](r3_checks.py)).

| Module | Role | Selftest / mutation check |
|---|---|---|
| [pf_adv2.py](../../research/pf_adv2.py) | inputs, training extraction, weights, 12 fits, candidate support, controller, the 8 policies | `pf_adv2_selftest.py` / `pf_adv2_mutants.py` |
| [pf_adv2_eval.py](../../research/pf_adv2_eval.py) | offline prediction diagnostic on the Oct 26–27 sample; screens; the 500-draw local control | `pf_adv2_eval_selftest.py` / `pf_adv2_eval_mutants.py` |
| [pf_adv2_replay.py](../../research/pf_adv2_replay.py) | the bounded own-state R4 diagnostic (≤ 384 worlds, 8 policies, ≤ 1,536 reached pairs) | `pf_adv2_replay_selftest.py` / `pf_adv2_replay_mutants.py` |
| [r3_checks.py](r3_checks.py) | the round's declared predicates A–I and their falsifier (E, G, H on clean and corrupted copies of the real inputs) | the falsifier runs with the checks |

The receipt formatter `r3_tables.py` computes no predicate: it copies output values, sums, ratios and counts into markdown. It is committed
with the results.

## Implementation choices fixed here (beyond the review's text)

- **Ties:** an exact zero prediction keeps the state: WAIT when absent, KEEP when live. For the offline diagnostic, d = 1 means POST or KEEP.
  An unsupported row takes the fallback (WAIT or KEEP), lies outside the four screen groups, and is reported as `fallback`.
- **Candidate support statuses:** checked in this order:
  1. `unsupported_horizon` (the horizon bin was not observed in training);
  2. `insufficient_candidate_support` (fewer than 200 distinct training originals in the side/head);
  3. `unsupported_state_category`;
  4. `outside_training_range` (`pf_r2_shadow.NUMERIC` ± 1e-9; NaN is outside);
  5. `nonfinite_prediction`.

  Structural statuses are the teacher's: `cap_blocked`, `near_end`, `feature_unavailable`.
- **Weights:** the screens and readings use the stored sampling weights `w = 1/(pi N q)`. The fits use `w · n / sum(w)` per side/head.
  These are the same relative weights in both.
- **Local control anchors:** the `mandatory` POST pick (only if ok) and the `keep` pick. Eligible means the candidate decided by its
  model; fallback anchors are counted apart.
  - Strata: day, side, head, source, instrument and minute of t0.
  - One generator per scope (seed 20261121), strata in sorted key order, 500 draws.
  - Scopes: (day, head) and (day).
- **Reached states (RICH):** eligible means a model decision in an absent or live state. The minimum sha256 rank under
  `adv2_rich_v1_reached_20261121` is chosen per world/side/head. The fork is rebuilt by `pf_r2_labels.bound_fork` with RICH as the
  roll-in and refused unless its fingerprint equals the one recorded at the decision. The arms come from `pf_r2_labels.pair_bound` under
  the frozen pi0, audit every 35.
  - Every world records exactly one entry per side and head. It is either the pair with its inclusion count (the number of eligible
    decisions it was chosen from) or the status `no_eligible_decision` with count 0.
- **Quote time:**
  - from activation to the earliest of removal effect, full fill and END;
  - refused posts excluded; cancel-pending stays exposed;
  - "at best" uses the side's best in the latest venue frame at or before each moment;
  - either is the union of the sides, both is their intersection.
- **Coverage (review item 1):** the evaluation writes `coverage.csv`. It covers each day, side, head and group: all, RICH's selected
  action, and fallback. For the source, the horizon bin and each of the six support dimensions of `pf_r2_teacher.categories` (remaining size,
  queue ahead, generation, inventory, the other side's state, price) it gives rows, distinct originals, exact- and meaningful-nonzero
  originals and weight. `coverage_train.csv` gives the same for the Oct 24–25 training rows. These are counts, not calibration.
- **Exact versus meaningful nonzero (review item 2):** the screens' 20-original floor counts `|Y| > 1e-6 c` originals. The exact-nonzero
  count is reported beside it, and the `exact_count_support` column shows the floor under the exact count. Raw labels, weights and
  targets are unchanged.
- **Own-state exposure and churn (items 1 and 14):**
  - RICH's replay callbacks are counted by state and status, and its decisions with inputs by support dimension. The tables set these
    against the training rows of the same head.
  - Every policy records each reviewed generation's POST and first live review: times, the deciding policy's value, the action, and the
    recorded input values at the post with their changes to the review. These are recorded state only; no cause is inferred.
- **Cap validity:** each episode's `cap_ok` requires its peak absolute engine position to be within 30 shares. The engine's reservation and
  position checks run at every notice and command, and a violation stops the partition.
- **Mutation identity (review item 3):** check H runs `pf_mutant_summary.judge` directly on every record file. It uses the harness's own
  names and the sources on disk now, and requires one record line per file and no other record file. It does not read summary text.
- **Markouts:**
  - sign × (venue mid − price) × shares, at the fill, +100 ms and +1 s;
  - an endpoint after END is missing (no frame beyond the episode);
  - the END markout uses the END mark, and equals the episode value by identity.
- **Pilot:** replay partition 0 (32 Oct 26 worlds of the nested population) runs first and measures cost; the rest follow without a
  pause, within the round's 40 core-hours.
- **Order, enforced in code:**
  1. extract and fit;
  2. commit the receipt copy of `fit.json`;
  3. evaluate and replay.

  The selftests and the mutation suites of the three new modules ran before this freeze, on the files committed here. Their logs and
  summaries are in `logs/`, and check H re-verifies every record against these sources.

  `pf_adv2_eval` and `pf_adv2_replay` refuse unless HEAD holds that `fit.json` byte for byte. `r3_checks.py` C checks that its commit
  precedes the evaluation's asof.

## Limits stated now

- The Spearman correlation is unweighted and secondary. No probability calibration is claimed: the heads output mean advantages.
- Two consumed days: no interval. A failed screen allows the diagnostic replay; it does not allow promotion or a refit.
- The replay is a diagnostic, not the R4 gate. Same-history NEU2 and the full economic/quoting qualification stay deferred.
