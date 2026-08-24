# BE-FlowAndFills adverse-movement extension

Status: **DESIGN / NOT DECISION-ELIGIBLE**

Scope: research only. This plan adds no venue adapter, order sender, cancellation
command or live execution path. `contracts/contracts.yaml` v24 is authoritative
for types and module ownership.

## 1. Placement and responsibility

Adverse-movement prediction is a capability inside `BE-FlowAndFills`, not a new
belief or decision module:

~~~text
DA-StateView                         knowledge-time Binance + PM state
    -> DA-FlowActionGrid             immutable AdverseFeatureRow + exact action
    -> BE-FillFit                    offline AdverseMoveFit
    -> EV-AdverseMove               independent forward validation
    -> BE-FlowAndFills.evaluate      complete KEEP/CANCEL ActionOutcome
    -> DE-DecisionScheme             chooses an action only from full economics
~~~

The fit answers what is likely to happen if an exact resting action remains.
It never chooses `CANCEL`. DE owns that choice, and no DE consumer receives raw
features or a bare toxicity score.

## 2. Estimand

For candidate action `a`, maker side `s`, state known at `t`, cancellation
latency `L`, prediction horizon `Delta`, and markout horizon `h`, estimate:

~~~text
P(toxic fill if a is kept | x_t, a)
E[maker-signed markout_h | fill, x_t, a]
E[(-markout_h)+ * prevented_fill_size
  * 1{counterfactual fill occurs after t + L} | x_t, a]
~~~

The last line is gross adverse damage potentially avoidable by cancellation.
It is non-negative and does not pretend that the canceled fill's spread,
rebate, or queue value is preserved. Those terms belong to `ActionOutcome` and
`WealthLedger` when KEEP and CANCEL are compared.

This is a joint action/fill/markout target. An unconditional direction forecast
multiplied by an independent fill probability is not admissible: fills are
endogenous and are most likely in the states whose movement is adverse.

## 3. Inputs

All inputs are admitted through `StateView` on local receipt/knowledge time.
`DA-FlowActionGrid` freezes them into `AdverseFeatureRow` objects tied to the
exact shadow action; unavailable rows are retained. The direct-event candidate
uses the existing Binance futures WebSocket capture:

- `bookTicker`: event-driven best bid/ask and sizes;
- `trade`: raw matched trades and aggressor direction;
- `depth20@100ms`: top-20 depth snapshots;
- Polymarket price changes, trades, displayed depth and the exact resting action;
- time remaining, moneyness, tick, spread and input staleness;
- the measured end-to-end cancellation-latency profile.

Initial feature families are multi-scale Binance returns, signed aggressive
volume, burst intensity, microprice/book imbalance, depth depletion, PM queue
and imbalance state, and Binance-implied-versus-PM price divergence. Exact
windows and transformations must be frozen in `feature_schema_hash` before a
scored period. The identical builder hash is required offline and at decision
time. No feature may include the event that reveals or causes the fill being
predicted.

The deployed 1 Hz relay may produce a labelled diagnostic fit with
`feed_class=SAMPLED_RELAY`; it cannot validate the direct-event-WS question.

## 4. Candidate and output

Start with an interpretable regularized logistic model for the joint toxic-fill
probability and a regularized conditional-markout model. A more complex learner
is a separate frozen candidate, not an after-the-fact replacement.

Every estimate binds:

- the exact action hash, maker side, level, size and state;
- `as_of`, prediction horizon and markout horizon;
- the model and source-profile hashes;
- the measured cancellation-latency profile;
- maximum feature knowledge time and input staleness.

If the model, source schema, latency profile or admitted state does not match,
`BE-FlowAndFills` returns `Unavailable` rather than extrapolating.

## 5. Evaluation and promotion

`EV-AdverseMove` evaluates a frozen candidate on forward UTC days. It reports:

- toxic-fill Brier score and calibration error;
- gross avoidable adverse damage;
- KEEP-versus-CANCEL PnL delta under the frozen policy and economic schedule;
- per-fill and share-weighted results, with a day-clustered interval;
- coverage, refusals, source staleness and the latency profile used.

Classification accuracy is diagnostic. Promotion requires the frozen
KEEP-versus-CANCEL replay to improve net share-weighted value after lost spread,
rebate, queue position and actuation latency, with its pre-registered uncertainty
bar satisfied. The existing ten-forward-day floor applies unless a separately
reviewed protocol imposes a stricter requirement.

Artifact states remain monotone:

~~~text
DEVELOPMENT -> CANDIDATE_FROZEN -> VALIDATED
                                  | INSUFFICIENT_EVIDENCE
                                  | MODEL_REFUTED
~~~

Until `AdverseMoveFit.status == VALIDATED` and its matching evaluation verdict
is `PASS`, the decision-facing capability is unavailable. Adding this structure
therefore does not reopen the current passive-MM or cancellation verdict; it
defines the evidence needed to test the remaining direct-WS route.
