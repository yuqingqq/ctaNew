# Stateful harmful-flow cancel x skew — TODO plan

**Recorded:** 2026-08-26T02:43:12Z, after the I5 lead control completed  
**Status:** TODO / offline research only / no live-trading implementation  
**Parent plan:** `HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md`  
**Working baseline:** reduced-fine linear model (`PM_PLUS_FINE`) on the
unchanged `QR_SKEW_ONLY` shadow trajectory

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

- [ ] Pin the admitted high-frequency era boundary in the candidate manifest;
  do not derive it from the latest collector restart at runtime.
- [ ] Record artifact `as_of`, exact train/development bounds, selected slugs,
  row counts by coin/day/status and the 60-second split embargo.
- [ ] Record SHA-256 hashes for the exposure dataset, raw-source ledger,
  feature schema, builder files and dependency lock.
- [ ] Save the exact feature names, normalization means/scales, hazard weights,
  conditional-value weights, target latency and action thresholds.
- [ ] Correct the historical hardcoded split field and refuse receipts whose
  declared split disagrees with their row timestamps.
- [ ] Write candidate and score-dump artifacts atomically.
- [ ] Reproduce the reduced-fine receipt to the cent from committed code before
  extending it. A reproduction may not change features, labels or thresholds.

**Phase-0 gate:** a fresh process can load one manifest and reproduce the named
development scores without fitting or consulting growing raw data.

## 4. Phase 1 — one combined predictor-state extension

Declare a single family, `PRED_STATE_V1`, before reading its result. It adds:

- [ ] `time_remaining_s` and a terminal-window indicator;
- [ ] true order-generation age `t_start - gen_t0` (not the current market
  quote freshness field);
- [ ] remaining-size fraction and normalized queue-ahead;
- [ ] exact-level depletion/replenishment velocity at 50/250/1,000 ms;
- [ ] recent same-side and opposite-side fill shares at 50/250/1,000 ms;
- [ ] time since last same-side PM touch move;
- [ ] PM and Binance feed freshness/event age at the decision cutoff;
- [ ] point-in-time PM microprice/fair-price disagreement, only if the existing
  fair-price object supplies a timestamped value without re-derivation here;
- [ ] explicit missing/stale flags instead of silent zero imputation.

Do **not** put these policy variables into the harm predictor:

- inventory `net` or current skew tier;
- last cancel, cooldown, cancel-pending or repost state;
- action-rate budget or queue-reset-cost assumption.

Those variables are absent or policy-induced on the no-cancel training
trajectory. They belong in the deterministic decision layer. Feeding them to
the predictor would mix toxicity with action preference and create off-policy
state that the training population does not identify.

### Phase-1 correctness tests

- [ ] Every feature carries `feature_asof <= decision_time - source_cutoff`.
- [ ] A synthetic event after the cutoff cannot change any feature.
- [ ] Exact-level depletion tests distinguish cancellations from executions.
- [ ] Generation age resets only on a generation change.
- [ ] Feed-staleness tests fire on real and synthetic gaps.
- [ ] Duplicate decision states are collapsed or explicitly weighted.
- [ ] Adding the feature builder drops zero rows silently: every exclusion has
  a counted status.

## 5. Phase 2 — limited model comparison

No hyperparameter search.

- [ ] Keep `PM_PLUS_FINE` linear hazard x conditional-value as the reference.
- [ ] Test `PM_PLUS_FINE + PRED_STATE_V1` as one combined state candidate.
- [ ] Test exactly one fixed-capacity nonlinear candidate using the same
  reduced/state feature schema. Pin LightGBM capacity, regularization, seed and
  early-stopping rule before scoring.
- [ ] Report fill-hazard discrimination, harmful-fill sign discrimination and
  conditional-value error separately; do not let expected-value ranking hide
  a failed head.
- [ ] Use generation-native first-crossing evaluation and at least 200 matched
  random controls.
- [ ] Increment multiplicity for every scored candidate, including a nonlinear
  candidate that fails.

If new state/nonlinear candidates are scored on a one-time top-up, freeze the
top-up receipt before the first number is read. Otherwise carry them as
separate forward candidates and price the increased multiplicity honestly.

**Phase-2 gate:** a candidate must improve gross cancel value and harmful-tail
selection over reduced fine on identical rows without concentrating its gain
in one hour or a handful of fills. AUC alone cannot pass the gate.

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
- [ ] comparison with `QR_SKEW_ONLY`, `QR_CANCEL_HOLD_X_SKEW` and matched
  random cancellation on identical opportunities.

Do not label `net_cents` as strategy profit unless queue-reset/repost costs and
the complete lifecycle are included.

**Phase-4 gate:** positive cost-adjusted value at a material retention level,
with inventory and traffic no worse than their declared limits, on both the
point estimate and matched-random comparison. ETH may be rejected separately
if its very small per-cancel margin cannot survive the cost grid.

## 8. Phase 5 — freeze and forward validation

- [ ] User chooses the forward candidate set. Working recommendation:
  reduced-fine primary; extended remains held and is not silently adopted;
  BTC lead stays rejected.
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

1. `harmful_candidate_manifest_v1.json` — pinned data/code/model provenance.
2. `harmful_state_features.py` — point-in-time `PRED_STATE_V1` builder and
   selftests.
3. `harmful_state_model_comparison_v1.json` — linear/state/nonlinear paired
   research receipt.
4. `harmful_stateful_policy.py` — offline cancel/hold/repost state machine.
5. `harmful_stateful_comparison_v1.json` — latency x cost x budget results.
6. Freeze receipt and forward scorer — only after the preceding gates pass.

The filenames are proposed interfaces, not authorization for live execution.
All code remains offline replay/research code.
