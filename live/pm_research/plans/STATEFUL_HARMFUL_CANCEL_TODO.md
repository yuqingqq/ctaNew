# Stateful harmful-flow cancel x skew — TODO plan

**Role:** subordinate implementation worksheet only. Overall project progress
and ordering are governed by `HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` §10; do not
use this file's checkbox count as the total project completion figure.

**Recorded:** 2026-08-26T02:43:12Z, after the I5 lead control completed  
**Status:** ACTIVE / PHASE-2 DEVELOPMENT RECEIPT COMPLETE / NOT FROZEN
**Parent plan:** `HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md`  
**Working comparators:** `QR_SKEW_ONLY`, `QR_CANCEL_HOLD_X_SKEW` and the
Phase-2 v2.2 linear/state/LightGBM arms on the unchanged neutral shadow
trajectory

## 0. 2026-08-28 progress and superseding direction

This section supersedes the old progress snapshot below without deleting its
provenance.

- [x] The decision/exposure dataset, `PRED_STATE_V1`, embargo and action-unit
  reconciliation are implemented and repeatedly reproduced.
- [x] The research-only cancel/hold/repost state machine and its synthetic
  parity/lifecycle battery are implemented; real-data integrated wiring remains
  outstanding.
- [x] The Phase-2 four-arm development receipt v2.2 is independently
  recomputed 15/15 and bit-exact across the latest replication chain.
- [x] Fill hazard has incremental development evidence for BTC at the 5% and
  10% budgets under the declared optimistic window-level null.
- [x] The receipt classifies that evidence as development, not validation: the
  scored top-up is one 14.4-hour consumed span and `G=0` complete UTC days.
- [ ] Conditional harmful-sign/value discrimination remains the principal
  modelling question. `PLUS_PRED_STATE_V1` survives nowhere; ETH has no
  promotable increment.
- [ ] The latest predictor has not yet passed a complete cancel x skew x fair
  price lifecycle comparison.
- [ ] No timestamped PM fair-price object currently enters `PRED_STATE_V1`.
  `Identity` remains the supported fair-price baseline.

The next work therefore separates four lanes: conditional value, fair price,
frozen skew and integrated replay. Build and preregistration may proceed in
parallel; result-based selection may not reuse consumed days.

### 0.1 Correctness blockers before downstream integration

- [x] Require both conditional LightGBM value-model artifacts in the expected
  manifest hash set; absence or byte mismatch must refuse scoring.
  — done, `5d8d58b`: the FIT records `val_models.json` inside the hash lattice
  and the expected set is derived from it (`phase2_arms.py:1151`); exercised in
  production at `fd1e949` (14 `file_hashes`, both val models required).
- [x] Extend fit-side pre-load/write-time drift detection to every bound
  frozen/top-up/receipt input used by the scorer.
  — done, `5d8d58b`: `identity_drift` compares every captured key with
  `IDENTITY_DRIFT_EXEMPT = ()` (`phase2_arms.py:975`), wired at both fit
  (`:1617`) and score (`:2071`).
- [x] Enforce the repository root for every result-bearing imported module,
  including the harmful exposure/valuation and policy modules.
  — done, `5d8d58b`: all 12 `CODE_IDENTITY_FILES` imported at entry with
  `__file__` resolved under `_ROOT` (`phase2_arms.py:118`), replacing the
  four-module check.
- [x] Make population role, reach (`G`, complete-day count and as-of) and
  development-not-validation caveats generator-owned receipt fields.
  — done, `5d8d58b`: `is_a_validation` is COMPUTED from the declared population
  label and the complete-UTC-day count (`phase2_arms.py:1695`); first emitted by
  the generator at `fd1e949`.
- [ ] Preserve peer audit annotations across regeneration by mechanism rather
  than manual reattachment.
- [x] Regenerate the increment-null receipt against the same Phase-2 v2.2
  provenance chain.
  — done, `163bd36` under R-234/R-235, re-bound to the v2.3 chain that
  supersedes the v2.2 named above. The re-binding surfaced a determinism defect
  (unpinned `PYTHONHASHSEED` made every draw independent); repaired canonically
  with sight-unseen pre-committed acceptance, survivors unchanged.
- [x] Every new guard gets a real positive control and a known-bad behavioural
  refusal; source-text checks do not satisfy this item.
  — done as standing practice, verified at R-231 (11 red-then-green known-bads
  that cycle; BE battery 489/0, seams 184/0 on the coordinator's own run).

## 1. Current state and decision boundary

The v3.4 exposure rebuild is accepted for research: 471 windows, 1,125,289
`OK` rows, and zero reconciliation, unhooked-state, wrong-generation,
boundary-time or consume-clock failures.

The consumed 2026-08-24/25 development fragment supports one working model:

- **primary:** PM state plus reduced Binance L1 imbalance and 10–250 ms mid
  movement;
- **held, not adopted:** OFI plus big-print, whose gain is confined mainly to
  the 5% cancellation budget;
- **rejected/null:** depth20 and PM-thinning;
- **not adopted:** BTC-to-ETH lead. Its true-lead increment is
  `+229/+292/+161c` at 5/10/15%, while the T-5s control is
  `+264/-871/-662c` relative to reduced fine. The 5% increment survives stale
  BTC data, so the candidate fails the declared timeliness control.

Five candidate specifications have consumed the old development fragment.
Do not run another feature scoreboard on it. Reproduction, correctness checks
and cost/latency sensitivity of an unchanged score are allowed; selecting a
new feature family from those same outcomes is not.

The last consumed BTC/ETH window slug is
`{btc,eth}-updown-5m-1787650200` (2026-08-25 09:30 UTC). Data after that point
has not been used by the v2/v3/v4/v5 comparison receipts. If a one-time
development top-up is used, its exact start/end, hashes and role as
**development, never forward validation**, must be recorded before scoring.

## 2. Architecture: prediction state is not policy state

Keep two explicit layers.

### 2.1 Harm predictor

The predictor estimates:

```text
E[latency-preventable cancel value | market state, queue state]
```

It may consume only point-in-time, candidate-independent state that also
exists on the no-cancel shadow trajectory. Inventory and cancel lifecycle do
not define whether external flow is harmful and must not be allowed to turn
the predictor into an implicit decision policy.

### 2.2 Stateful action policy

The policy consumes the harm score plus strategy state:

```text
cancel when expected avoided harm
            > sacrificed fill value + queue-reset/repost cost
```

Inventory determines the value and admissibility of an action. Skew continues
to determine placement. The harm score may override any live generation on
the threatened side, including a skewed order, but inventory-reducing
protection must remain an explicit rule and an explicit ablation.

## 3. Phase 0 — repair the freeze and receipt surface

These are blockers before any candidate is called frozen.

- [x] Pin the admitted high-frequency era boundary in the candidate manifest;
  do not derive it from the latest collector restart at runtime.
- [x] Record artifact `as_of`, exact train/development bounds, selected slugs,
  row counts by coin/day/status and the 60-second split embargo.
- [x] Record SHA-256 hashes for the exposure dataset, raw-source ledger,
  feature schema, builder files and dependency lock.
- [x] Save the exact feature names, normalization means/scales, hazard weights,
  conditional-value weights, target latency and action thresholds.
- [x] Correct the historical hardcoded split field and refuse receipts whose
  declared split disagrees with their row timestamps.
- [x] Write candidate and score-dump artifacts atomically.
- [x] Reproduce the reduced-fine receipt to the cent from committed code before
  extending it. A reproduction may not change features, labels or thresholds.

**Phase-0 gate:** a fresh process can load one manifest and reproduce the named
development scores without fitting or consulting growing raw data.

## 4. Phase 1 — one combined predictor-state extension

Declare a single family, `PRED_STATE_V1`, before reading its result. It adds:

- [x] `time_remaining_s` and a terminal-window indicator;
- [x] true order-generation age `t_start - gen_t0` (not the current market
  quote freshness field);
- [x] remaining-size fraction and normalized queue-ahead;
- [x] exact-level depletion/replenishment velocity at 50/250/1,000 ms;
- [x] recent same-side and opposite-side fill shares at 50/250/1,000 ms;
- [x] time since last same-side PM touch move;
- [x] PM and Binance feed freshness/event age at the decision cutoff;
- [ ] point-in-time PM microprice/fair-price disagreement, only if the existing
  fair-price object supplies a timestamped value without re-derivation here;
- [x] explicit missing/stale flags instead of silent zero imputation.

Do **not** put these policy variables into the harm predictor:

- inventory `net` or current skew tier;
- last cancel, cooldown, cancel-pending or repost state;
- action-rate budget or queue-reset-cost assumption.

Those variables are absent or policy-induced on the no-cancel training
trajectory. They belong in the deterministic decision layer. Feeding them to
the predictor would mix toxicity with action preference and create off-policy
state that the training population does not identify.

### Phase-1 correctness tests

- [x] Every feature carries `feature_asof <= decision_time - source_cutoff`.
- [x] A synthetic event after the cutoff cannot change any feature.
- [x] Exact-level depletion tests distinguish cancellations from executions.
- [x] Generation age resets only on a generation change.
- [x] Feed-staleness tests fire on real and synthetic gaps.
- [x] Duplicate decision states are collapsed or explicitly weighted.
- [x] Adding the feature builder drops zero rows silently: every exclusion has
  a counted status.

## 5. Phase 2 — limited model comparison

No hyperparameter search.

- [x] Keep `PM_PLUS_FINE` linear hazard x conditional-value as the reference.
- [x] Test `PM_PLUS_FINE + PRED_STATE_V1` as one combined state candidate.
- [x] Test exactly one fixed-capacity nonlinear candidate using the same
  reduced/state feature schema. Pin LightGBM capacity, regularization, seed and
  early-stopping rule before scoring.
- [x] Report fill-hazard discrimination, harmful-fill sign discrimination and
  conditional-value error separately; do not let expected-value ranking hide
  a failed head.
- [x] Use generation-native first-crossing evaluation and at least 200 matched
  random controls.
- [x] Increment multiplicity for every scored candidate, including a nonlinear
  candidate that fails.

If new state/nonlinear candidates are scored on a one-time top-up, freeze the
top-up receipt before the first number is read. Otherwise carry them as
separate forward candidates and price the increased multiplicity honestly.

**Phase-2 gate:** a candidate must improve gross cancel value and harmful-tail
selection over reduced fine on identical rows without concentrating its gain
in one hour or a handful of fills. AUC alone cannot pass the gate.

### 5.1 Phase 2A — conditional signed-value lane

Freeze one `COND_VALUE_V1` specification before reading a new result.

- [ ] Define `V_cancel` from each fill/tranche's own event timestamp, resting
  level, shares and five-second markout; never proxy a fill timestamp with a
  nearby quote or resync event.
- [ ] Keep no-fill hazard, harmful sign and signed magnitude as separately
  observable heads. No-fill rows train hazard; only latency-preventable fills
  enter the conditional heads.
- [ ] Fit `P(harm | fill, x)` plus separate harmful and favourable magnitude
  heads, and combine them as:

  ```text
  p_fill * (p_harm * m_harm - (1 - p_harm) * m_good).
  ```

- [ ] Compare exactly three conditional-value specifications: existing linear
  reference, one fixed-capacity sign-plus-magnitude model and one fixed-capacity
  direct nonlinear-value model. No model sweep.
- [ ] Calibrate out of fold with the declared embargo and preserve one action
  per cancellable generation in scoring.
- [ ] Report hazard AUC/PR, harmful-sign PR/lift, signed-value rank correlation,
  tail value captured, favourable-fill sacrifice and calibration separately.
- [ ] Compare hazard-only, harmful-sign-only and full-value actions on identical
  rows and matched cancellation budgets.
- [ ] Report BTC and ETH independently. A BTC pass does not carry ETH; ETH may
  stop if its conditional increment remains negative or cost-fragile.
- [ ] Add synthetic positive controls with known harmful/favourable fills and a
  known-bad refusal for outcome-selected or duplicate-action populations.

**Phase-2A gate:** full conditional value must improve the decision metric over
hazard-only and matched random at a material retention level. Better fill AUC
without better harm/value selection does not pass.

### 5.2 Phase 2B — timestamped fair-price lane (parallel)

Do not edit or implement the refuted `BE_BELIEF_PLAN.md`. Write a short
successor contract that carries only its surviving ownership rules.

- [ ] Define a typed point-in-time output containing coin/window, side or
  outcome convention, fair value, source-event time, local-knowledge time,
  freshness and book-admissibility status.
- [ ] Use executable-book `Identity` as the mandatory baseline and fallback.
- [ ] Predeclare at most two challengers: PM microprice and one cross-venue
  forecast. Do not reuse fill outcomes to construct either forecast.
- [ ] Score each challenger incrementally to `Identity` using a proper forecast
  score, point-in-time parity and day-level reporting.
- [ ] Add future-event invariance, stale-feed, crossed/one-sided/insufficient-
  depth and convention/complement selftests.
- [ ] Join the artifact into predictor rows strictly as-of and expose explicit
  missing/stale flags; never rederive fair price inside the harmful feature
  builder.
- [ ] Define toxicity labels as fill-conditional residuals relative to the
  unconditional fair-price output, avoiding adverse-selection double-counting.
- [ ] Run `Identity` versus each passing challenger as an explicit integration
  ablation. If none passes, keep `Identity`; fair price does not block the
  cancellation experiment.

**Phase-2B gate:** a challenger must improve on `Identity` out of sample without
failing timestamp/admissibility checks. Similarity to settlement or a base-rate
comparison is not an incremental fair-price result.

### 5.3 Phase 2C — frozen skew lane (parallel)

Treat skew as inventory/risk control, not as a substitute alpha model.

- [ ] Freeze `QR_SKEW_ONLY` placement, size, queue and inventory semantics as
  the neutral integration reference.
- [ ] Do not select another band, hysteresis or skew threshold on 2026-08-20
  through 2026-08-25.
- [ ] Define the policy interface through desired exposure, allowed placement,
  marginal inventory-risk value and inventory-increasing/reducing status.
- [ ] Keep inventory, skew tier, cooldown and cancel lifecycle out of the harm
  predictor; consume them only in the action-value policy.
- [ ] Run explicit reducing-side protection and all-orders override ablations.
- [ ] Require bit-identical `QR_SKEW_ONLY` replay when predictor output is
  disabled or the cancel threshold is infinite.
- [ ] Report whether skew improves inventory even when it does not improve P&L;
  do not force one module to pass both roles.

**Phase-2C gate:** skew is admissible when it respects the frozen queue model
and declared inventory/traffic bounds. It is not promoted as alpha merely for
reducing inventory.

## 6. Phase 3 — stateful cancel x skew replay

Implement a research-only state machine per `(slug, side, generation)`:

```text
LIVE -> CANCEL_PENDING -> HELD -> REPOST_ELIGIBLE -> LIVE
```

- [ ] `LIVE`: ordinary inventory-skewed placement remains active.
- [ ] First score crossing above `theta_cancel` sends one simulated cancel for
  the generation; later crossings cannot double-count it.
- [ ] Optional `theta_reduce < theta_cancel` reduces displayed size before a
  full cancel. Keep this as an explicit ablation, not an implicit behavior.
- [ ] `CANCEL_PENDING`: fills inside assumed latency remain stale/unprevented.
- [ ] `HELD`: no repost until score is below `theta_repost` for a declared
  dwell time. Require `theta_repost < theta_cancel`.
- [ ] `REPOST_ELIGIBLE`: ordinary skew rules choose side, level and size again;
  the predictor does not choose placement.
- [ ] A fresh generation receives a fresh queue position and incurs the
  declared queue-reset cost.
- [ ] Inventory-increasing and inventory-reducing orders are reported
  separately. Run both explicit reducing-side protection and all-orders
  override cells.
- [ ] Rate limits count requested, effective and suppressed cancellations.

### Required parity tests

- [ ] Disabled predictor is bit-identical to `QR_SKEW_ONLY`.
- [ ] Infinite cancel threshold is bit-identical to `QR_SKEW_ONLY`.
- [ ] Zero repost threshold with permanent hold matches cancel-and-hold.
- [ ] One generation can be cancelled at most once.
- [ ] Cancelled skewed orders cannot fill after simulated effectiveness.
- [ ] Pre-effectiveness fills remain charged as stale.
- [ ] No policy-generated trajectory is reused as its own training population.

### Required integrated arms

Run these on the same neutral opportunities and independent event clocks:

- [ ] `QR_SKEW_ONLY`.
- [ ] `QR_CANCEL_HOLD_X_SKEW`.
- [ ] Fill-hazard-only cancel with neutral placement.
- [ ] Conditional-value cancel with neutral placement.
- [ ] Conditional-value cancel x frozen skew.
- [ ] Conditional-value cancel x frozen skew x fair-price residual.
- [ ] Random cancel matched on action count, side, hour and budget.

For the full arm, the policy must compute rather than print:

```text
delta_EV(cancel vs keep)
    = avoided conditional harm
    - sacrificed favourable fill value
    - lost spread capture
    - queue-reset/repost cost
    + marginal inventory-risk benefit
    - action/traffic cost.
```

The predictor emits estimates, never the final cancel boolean. Skew owns desired
placement; the state machine owns cancel, hold and repost lifecycle.

## 7. Phase 4 — economic and latency gates

Evaluate the unchanged score at assumed cancel-effective latencies:

`5, 10, 20, 30, 50, 75, 100, 150, 250 ms`.

Use a declared queue-reset/repost-cost grid that brackets the measured gross
break-even cost. The current reduced-fine point estimates imply only about
`0.29/0.38/0.32c` per BTC cancellation and
`0.011/0.068/0.072c` per ETH cancellation at 5/10/15% budgets.

For every latency x cost x budget cell report:

- [ ] gross avoided harm, favorable-fill sacrifice and cost-adjusted value;
- [ ] cancellations per minute and per generation;
- [ ] effective, stale, unresolved and zero-value cancellations;
- [ ] hold duration, repost count and queue resets;
- [ ] fill/share retention and spread capture;
- [ ] retained-book adverse-cost/spread-capture ratio;
- [ ] terminal inventory, peak inventory and reducing/increasing-side split;
- [ ] complete maker P&L, post-fill markout and inventory loss, rather than
  treating `net_cancel_cents` as strategy P&L;
- [ ] comparison with `QR_SKEW_ONLY`, `QR_CANCEL_HOLD_X_SKEW` and matched
  random cancellation on identical opportunities.
- [ ] marginal module deltas: hazard -> conditional value, cancel -> cancel x
  skew and `Identity` -> fair-price challenger.

Do not label `net_cents` as strategy profit unless queue-reset/repost costs and
the complete lifecycle are included.

**Phase-4 gate:** positive cost-adjusted value at a material retention level,
with inventory and traffic no worse than their declared limits, on both the
point estimate and matched-random comparison. ETH may be rejected separately
if its very small per-cancel margin cannot survive the cost grid.

## 8. Phase 5 — freeze and forward validation

- [ ] Choose the forward candidate set only after the conditional-value,
  fair-price and integrated replay gates. The current BTC LightGBM hazard arm
  is a research seed, not a frozen full-policy candidate; no ETH arm is
  currently promotable.
- [ ] Refit each chosen candidate once on all declared development/training
  data and write a complete immutable manifest.
- [ ] Stamp the freeze commit and UTC instant before admitting any forward
  window.
- [ ] Admit only complete UTC days whose earliest required receipt is after the
  freeze. UTC day is the cluster unit.
- [ ] Score candidates without refitting for at least five complete untouched
  UTC days.
- [ ] Report candidate multiplicity and day-level results; no window-level
  interval may masquerade as independent evidence.
- [ ] Promote to strategy integration only if the cost-adjusted stateful replay
  beats the skew/cancel baseline without worsening inventory or traffic.

Public data can establish latency sensitivity, not causal cancel
preventability. Real cancel-send, acknowledgement and owned-fill timestamps
remain a separate measurement outside this research-only repository if they
require an exchange adapter.

## 9. Ordered deliverables

1. Phase-2 seam closure and same-chain increment-null receipt.
2. `conditional_value_v1` protocol, builder/models and paired receipt.
3. Fair-price successor contract, timestamped `Identity` artifact and
   predeclared challenger receipt.
4. Frozen skew policy interface and parity receipt.
5. Common action-value interface plus seven-arm offline replay.
6. Integrated latency x cost x budget receipt with complete strategy metrics.
7. Immutable candidate/freeze manifest with module hashes and multiplicity.
8. Unchanged forward scorer for at least five later complete UTC days.

The filenames are proposed interfaces, not authorization for live execution.
All code remains offline replay/research code.

## 10. Parallel execution order

```text
correctness seams ─> Phase 2A conditional value ─┐
                  └> Phase 2B fair price ────────┼─> integrated replay
                  └> Phase 2C frozen skew ───────┘        |
                                                        freeze
                                                          |
                                              >=5 later complete UTC days
```

- [ ] Start Phase 2A and Phase 2B only after their protocols and multiplicity
  are recorded.
- [ ] Build Phase 2C parity and the replay harness concurrently using typed
  stubs; do not score a synthetic stub as a candidate.
- [ ] Integrate immutable outputs only after their standalone gates are
  computed.
- [ ] A candidate changed after seeing a day cannot score that day; restart its
  forward clock after the new committed freeze.
- [ ] No full-policy promotion until standalone ablations and the complete
  lifecycle both pass.
