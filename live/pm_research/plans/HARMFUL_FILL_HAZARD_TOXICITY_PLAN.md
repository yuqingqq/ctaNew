# Harmful-fill hazard × toxicity optimization plan

**Status:** ACTIVE / BTC HAZARD SEED FROZEN-UNVALIDATED / FULL ROUTE NOT FROZEN
**Scope:** offline research only; no venue adapter, live cancellation path, or
execution server  
**Incentives:** maker rebates and liquidity rewards excluded from the primary
economics, by current user direction

**Role:** this is the governing project TODO. The stateful cancel x skew TODO is
a subordinate implementation worksheet, not the overall progress tracker.

**2026-09-01T03:49:46Z progress map (repository HEAD `1aaac18`):**

| §10 item | Current state |
|---|---|
| 1. Receipt/runtime seams | Substantially built; final model-path wiring and identity closure remain before another result-bearing 011 run. |
| 2. Dataset and `PRED_STATE_V1` | Complete and repeatedly reproduced. |
| 3. Conditional-value lane | Iteration 011 preregistration and implementation exist; the review hold remains and nothing has been fitted or scored. |
| 4. Fair-price lane | Typed `Identity` and build-only machinery exist; the challenger protocol is not freeze-ready and no challenger has been scored. |
| 5. Skew lane | `QR_SKEW_ONLY` freeze draft exists; it is not user-frozen. |
| 6. Common action-value/seven-arm replay | Contracts, parity stubs and inert trajectories exist; real queue-integrated replay is not complete. |
| 7. Passing-module integration | Not started. |
| 8. Stateful replay and latency/cost grids | Not started for the integrated candidate. |
| 9. Freeze and forward validation | The separate BTC hazard seed is frozen but unvalidated; no integrated candidate is frozen; `G=0/5` qualifying complete UTC days. |
| 10. Real execution latency | Not measured; remains outside this research-only repository if it requires a venue adapter. |

The live data dependency is now `clob_v4_1`, running since the ruled
2026-08-31T22:00Z boundary. The 2026-08-31 day is mixed-era and BTC failed its
quality bar. 2026-09-01 is the first era-pure admissible v4.1 day, but it is
incomplete and cannot accrue. Do not call collector operation a forward-test
day until the closed-day verifier admits it.

**2026-08-28 update:** Phases 0--2 have produced a reproducible development
receipt and a BTC fill-hazard research seed, but no complete-UTC-day forward
validation. The main unresolved estimand is now conditional signed fill value,
not whether an exposed order is likely to fill. Fair price, conditional harm,
inventory skew and lifecycle cost will be developed as separate modules and
combined only through the action-value interface in this plan.

## 1. Decision and motivation

The next optimization should not be another generic adverse-price forecast.
It should estimate the expected economic harm of leaving each currently
resting side exposed.

The corrected five-second concentration result motivates a selective policy:

| population | BTC fills required to carry the operative drift share | ETH fills required |
|---|---:|---:|
| real-maker tape, cash-ranked | 1.60% | 4.38% |
| fixed-five-share `JOIN_BBO` replay | **7.47%** | **13.46%** |

The replay population is the strategy-relevant one. Its worst 10% of fills
carry 53.2% of BTC and 52.9% of ETH total negative five-second drift. The
required tail is stable across the three sampled days: BTC
7.24%/7.66%/7.58%; ETH 13.71%/13.99%/12.81%.

These are perfect-foresight ceilings, not model results. They establish that
the harm is selective enough for a gate to be possible. They do not establish
that the tail is predictable, that every tail event is an informed sweep, or
that an implementable cancellation can arrive in time.

## 2. Target architecture

At decision time `t`, for each live order generation and each resting side,
estimate two separate objects:

1. **Fill hazard:** the probability that the same order generation receives a
   latency-preventable fill during the declared action horizon.
2. **Conditional toxicity:** the signed five-second economic value of
   cancelling, conditional on such a fill.

For side `s`, assumed cancel-effective latency `L`, and fill horizon `H`:

```text
p_fill(t,s,L,H) = P(same-generation fill in [t+L, t+H] | state at t)

V_cancel(fill)  = - maker five-second markout from the resting order level
                   × filled shares

tox(t,s)        = E[V_cancel | preventable fill, state at t]

expected_cancel_value(t,s) = p_fill(t,s,L,H) × tox(t,s)
```

Positive `V_cancel` means cancellation avoids a harmful fill; negative
`V_cancel` means cancellation sacrifices a profitable fill. This formulation
therefore prices both sides of the decision instead of labelling every fill as
something to prevent.

The prediction unit is a **decision-time quote-exposure row**, not a completed
fill. Training only on completed fills would condition on the event the policy
is trying to change and would not teach the model when an exposed order will
remain unfilled.

### 2.1 Conditional-value decomposition

The next model revision must not rely on one unconditional regression through a
zero-heavy, signed target. Estimate the components explicitly:

```text
p_harm(x) = P(V_cancel > 0 | preventable fill, x)
m_harm(x) = E[V_cancel | V_cancel > 0, preventable fill, x]
m_good(x) = E[-V_cancel | V_cancel < 0, preventable fill, x]

conditional_cancel_value(x)
    = p_harm(x) * m_harm(x) - (1 - p_harm(x)) * m_good(x)

expected_cancel_value(x)
    = p_fill(x) * conditional_cancel_value(x)
```

The empirical comparison must isolate four questions: fill arrival, harmful
sign, harmful/favourable magnitude and their combined expected value. A strong
hazard head does not establish useful toxicity discrimination.

### 2.2 Module ownership and no double-counting

The fair-price module estimates the unconditional object `E[Y | state]`. The
toxicity module estimates a fill-conditional residual relative to that anchor,
for example `E[fill value - fair value | fill, state]`. It must never absorb an
`E[Y | state, FILLED]` fair price; that would put adverse selection in both the
fair-price and toxicity terms.

Inventory and lifecycle state remain policy inputs. They price whether a
cancel, size reduction or repost is desirable; they do not become predictor
features merely because they affect the action decision.

## 3. Dataset contract

Build rows from the unchanged queue-realistic `QR_SKEW_ONLY` reference path.
Each row represents `(decision time, order generation, side)` and records:

- whether an order is live and its remaining size;
- queue ahead, filled fraction, order age and quote level;
- inventory and whether the side increases or reduces inventory;
- whether a same-generation fill would occur after `t+L` and by `t+H`;
- filled shares and five-second markout when the horizon is observable;
- explicit exclusions for gaps, missing future mid, generation replacement,
  truncated horizon and unproven timing;
- point-in-time feature timestamps and source-receipt provenance.

No candidate policy may generate its own training population. The no-cancel
reference trajectory defines the shadow counterfactual for every candidate so
models are compared on identical opportunities.

Rows with fills before `t+L` are economically stale, not preventable. Rows
whose cancellation timing cannot be established remain unresolved and cannot
be promoted as proven preventions. A latency grid is a sensitivity analysis,
not a substitute for venue cancel acknowledgements and owned-fill timing.

## 4. Feature plan

### 4.1 PM-native event state — primary

Use event-time, side-signed features at approximately
25/50/100/250/500/1,000 ms horizons:

- aggressive buy/sell quantity and notional;
- same-direction trade run length and flow purity;
- distinct levels consumed and multi-level sweep progression;
- bid/ask depth depletion and replenishment rates;
- imbalance, imbalance change, microprice and microprice change;
- distance from the resting order to touch and recent trade-through pressure;
- spread state, spread transition and quote age;
- order-generation queue state and partial-fill state.

Changes and acceleration matter more than static book levels for a fast pull
decision. Features must be available at local knowledge time and must not use
the event that causes the labelled fill.

### 4.2 Fair-price and external-flow context — secondary

Add, without replacing PM-native flow:

- PM microprice minus the point-in-time fair-price estimate;
- fair-price change over the same multiscale horizons;
- Binance spot/perpetual signed flow, microprice displacement and burst state;
- cross-venue disagreement and convergence speed;
- remaining window time, distance from strike and settlement-risk state.

The fair-price module describes where price should move. PM flow and queue
state describe whether this particular order is about to be selected. The two
roles must remain identifiable in ablations.

The currently supported fair-price output is `Identity`: the executable PM
top-of-book unchanged. The refuted `BE_BELIEF_PLAN.md` remains provenance and
must not be revived as an implementation plan. Create a small successor
contract before adding a fair-price feature to this dataset. It must emit a
point-in-time value, source timestamp, local-knowledge timestamp, freshness and
book-admissibility status. `Identity` is the mandatory baseline; PM microprice
and at most one cross-venue forecast may be predeclared challengers. A failed
challenger does not block integration: the full policy runs with `Identity`.

### 4.3 Frequency requirement

One-second bars that exclude the current second can be roughly 250–1,250 ms
old and are not sufficient as the only fast-cancellation representation.
Build sub-second event windows from receipt-time data. The decision clock may
run faster than the feature state changes, but duplicate-state decisions must
be collapsed or weighted so they do not create fictitious sample size.

## 5. Model plan

Use a limited, predeclared model family rather than a broad hyperparameter
sweep:

1. calibrated logistic/ridge models as the linear reference;
2. fixed-capacity LightGBM classifiers for fill hazard and harmful-fill sign;
3. a fixed-capacity nonlinear conditional-value or magnitude head.

The nonlinear candidate is justified by interactions such as sweep intensity
× side × queue position × fair-price disagreement. Hyperparameters must be
pinned before forward scoring and shared across comparable cells.

The conditional-toxicity component should report both:

- `P(V_cancel > 0 | fill, x)` for tail discrimination; and
- an economically weighted magnitude estimate or value-weighted sign score.

The combined action score is evaluated in expected cents, but ranking quality
in the harmful tail is reported separately so a noisy magnitude head cannot
hide useful discrimination.

The predeclared conditional-value comparison is limited to:

1. the existing linear conditional-value reference;
2. one fixed-capacity sign classifier plus separate harmful/favourable
   magnitude heads; and
3. one fixed-capacity direct nonlinear conditional-value challenger.

Compare hazard-only gating, harmful-sign gating and full expected-value ranking
on identical actions. Calibrate out of fold, report BTC and ETH separately and
permit a coin-specific rejection. Do not launch a generic model or
hyperparameter sweep.

## 6. Policy composition

Inventory skew and harmful-flow protection remain separate modules:

- inventory skew determines desired exposure and quote placement;
- the harmful-flow score can override the threatened side;
- the opposite side remains live when safe;
- intermediate risk reduces size before full cancellation;
- high risk cancels every live order generation on the threatened side,
  including an inventory-skewed order;
- inventory-reducing protection remains explicit and is not silently removed;
- cancel and repost use different thresholds or a short confirmation rule to
  avoid flicker;
- repost occurs only after risk falls below the lower threshold and ordinary
  placement rules again permit the quote.

The economic decision rule is:

```text
cancel when expected avoided harm
            > lost spread capture + queue-reset/repost cost.
```

The integrated evaluator owns the complete action comparison:

```text
delta_EV(cancel vs keep)
    = expected avoided fill-conditional harm
    - expected sacrificed favourable fill value
    - lost spread capture
    - queue-reset/repost cost
    + marginal inventory-risk benefit
    - action/traffic cost.
```

The predictor supplies estimates, never a cancel boolean. Skew retains
ownership of desired inventory exposure and placement. The policy layer alone
chooses `keep`, `resize`, `cancel`, `hold` or `repost` after applying inventory,
traffic and lifecycle constraints.

Because queue-reset cost is not yet directly measured, report results both
before it and across a declared sensitivity grid. Do not call a gross result
net-profitable.

## 7. Latency analysis

Evaluate assumed cancel-effective latencies of
`5, 10, 20, 30, 50, 75, 100, 150, 250 ms`. The grid answers sensitivity and
break-even questions. It cannot prove preventability from public market data.

For each latency report:

- fills already executed before the decision;
- fills received before assumed cancel effectiveness;
- fills definitely stale by event-time evidence;
- unresolved fills;
- economically preventable fills under the simulation;
- loss capture, profitable-fill sacrifice and cancel/repost traffic.

Real cancel-send, venue ACK/reject/effective timestamps and owned-fill receipts
remain necessary for causal validation. Those execution measurements belong
outside this research-only repository if they require a live adapter.

## 8. Evaluation and gates

Global ROC AUC, Brier score and average regression error are diagnostics, not
the primary economic verdict for a concentrated-tail problem.

Report at cancellation budgets of 5%, 10% and 15%, plus the unchanged q>0.5
action when applicable:

- share of total negative drift captured;
- precision and recall for economically harmful fills;
- PR AUC and lift over base rate;
- profitable spread capture sacrificed;
- fill and share retention;
- adverse-cost/spread-capture ratio `rho = A/S`;
- gross cancel value per decision and per retained fill;
- inventory, effective cancellations, reposts and queue resets;
- per-day results, not only pooled results.

Every scored policy must be compared with random abstention matched on the
same cancellation or fill-retention fraction. Merely quoting less is not
selection skill. A useful selection model must reduce `rho` relative to that
matched control; strategy viability requires reaching `rho < 1` at material
retention after declared costs.

The strategy baseline remains `QR_CANCEL_HOLD_X_SKEW`, with
`QR_SKEW_ONLY` mandatory. No candidate is promotable unless it improves the
baseline on independent days without worsening the declared inventory and
traffic limits.

### 8.1 Required integration ablation

Run every arm on the same neutral `QR_SKEW_ONLY` opportunity population, with
an independent event clock per arm:

1. `QR_SKEW_ONLY`;
2. `QR_CANCEL_HOLD_X_SKEW`;
3. fill-hazard-only cancellation with neutral placement;
4. conditional-value cancellation with neutral placement;
5. conditional-value cancel x frozen skew;
6. conditional-value cancel x frozen skew x fair-price residual; and
7. random cancellation matched on action count, side, hour and cancellation
   budget.

Also report the marginal deltas `hazard -> conditional value`, `cancel ->
cancel x skew` and `Identity -> fair-price challenger`. The full arm cannot hide
a failed component behind another module's gain.

Full replay output includes complete maker P&L, spread capture, post-fill
markout, fill/share retention, `rho`, effective/stale/unresolved cancels,
hold/repost/queue-reset traffic, terminal and peak inventory, inventory loss,
and per-day latency x cost sensitivity. `net_cancel_cents` alone is not a
strategy-P&L verdict.

## 9. Validation discipline

- Existing 2026-08-20 through 2026-08-25 data is consumed for the harmful-fill
  line. The current Phase-2 result includes one 14.4-hour 2026-08-25
  development span and has `G=0` complete UTC validation days.
- Do not tune features, thresholds, horizons or model capacity on new forward
  outcomes after the candidate is frozen.
- Admit only complete UTC days whose earliest required receipts postdate the
  candidate freeze.
- Treat UTC day as the primary independence cluster; window-level intervals
  are descriptive when the day count is inadequate.
- Account for every candidate in the forward race when constructing nulls and
  interpreting a clearing result.
- Preserve exact candidate builder code, dependency hashes, feature schema,
  split receipt and candidate hash before the first forward score.

More forward data is necessary to test generalization. More data from the same
seen days does not repair model selection, timing ambiguity or an invalid
freeze.

Parallel work is allowed for implementation, selftests and preregistration.
Outcome-driven selection is serialized: no lane may inspect a later day and
then change another lane's candidate for scoring on that same day. A newly
defined candidate starts its own forward clock after its committed freeze.

## 10. Implementation order

1. Close the remaining receipt/runtime fail-open seams before a downstream
   artifact depends on Phase 2: require every conditional-model artifact in
   the hash set, cover every bound input in fit-side drift detection, enforce
   the repository root for every result-bearing module, make population/reach
   disclosure generator-owned and rerun the increment null in the same chain.
2. Preserve and reproduce the accepted side-specific decision/exposure dataset
   and `PRED_STATE_V1` feature-as-of/reconciliation tests.
3. **Conditional-value lane:** preregister and fit the sign/magnitude
   decomposition in §2.1; evaluate hazard-only, sign-only and full-value
   increments with matched controls.
4. **Fair-price lane, in parallel:** write the successor timestamped interface,
   preserve `Identity`, and test only the predeclared challengers and feature
   ablations in §4.2.
5. **Skew lane, in parallel:** freeze `QR_SKEW_ONLY` placement semantics and
   inventory limits; implement integration/parity tests without selecting new
   bands or thresholds on consumed days.
6. Build the common action-value interface and the seven-arm ablation in §8.1.
7. Integrate only conditional-value and fair-price increments that pass their
   own gates; use `Identity` when no fair-price challenger passes.
8. Run complete stateful cancel/hold/repost replay, then the latency and
   queue-reset-cost grids.
9. Freeze the surviving candidate set, record multiplicity and score it
   unchanged on at least five complete later UTC days.
10. Measure real execution latency separately before making any claim about
    actually preventable fills.

### 10.1 Parallel-lane dependency rule

The conditional-value and fair-price lanes may be built concurrently. Skew
integration and the common replay harness may be developed against typed stub
outputs. Final economic scoring waits for immutable module artifacts:

```text
conditional value ─┐
fair price ─────────┼─> action-value policy ─> stateful replay ─> forward gate
frozen skew ────────┤
latency/cost model ─┘
```

Parallel construction does not authorize joint tuning. Each module must have a
standalone ablation, a positive control, a known-bad refusal and a committed
freeze before its first eligible forward day.

## 11. Expected outcome and stopping rule

The most likely performance gain comes from the decision-time target and
PM-native event features, not from nonlinear capacity alone. The nonlinear
model is useful only if it identifies the harmful 7–15% tail substantially
better than matched random cancellation.

Stop the route if, after complete forward day clusters, the best frozen model
cannot improve `rho` over matched random abstention or cannot approach
break-even at a material quote-retention level. A negative result closes this
feature/data design; it does not license continued tuning on the same forward
days.
