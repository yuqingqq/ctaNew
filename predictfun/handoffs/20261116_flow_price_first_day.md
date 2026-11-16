# Joint flow/price model: first chronological test day

As of **2026-11-16T09:35:10.243484+00:00**. Test date **2026-10-27**;
train Oct23–25 (357,041 eligible actions), calibrate Oct26 (137,025).
Test: **118,062 eligible actions from 287 markets**, out of 178,934 generated
probes and 288 selected markets. The remaining market had no eligible probe
because its candidates were already gone or lacked book coverage. All probe
statuses are retained. This is one previously consumed development day, with
point estimates only; it is not untouched validation or a strategy replay.

The first-day run was moved ahead of the remaining four folds at the user's
request. The frozen model recipe, labels, dates and parameters were unchanged.

## Reading

The joint model learns useful flow/price structure and avoids much of the
optimism caused by treating flow and price changes as independent. Its local
value ranking is in the right direction. However, value MSE is essentially
tied with a direct regression fitted to the same data. This is evidence for
retaining the joint formulation as a research candidate, not a large demonstrated
improvement in economic prediction or a solution to quoting churn.

## What the models mean

- **JOINT:** 15 probabilities over incoming volume at/through the focal price
  (zero / up to30 / above30 shares) crossed with side-signed midpoint-change
  classes. Conditional response heads are integrated over these probabilities.
- **INDEPENDENT:** separately predicted flow and price marginals, multiplied
  together; the same conditional response heads as JOINT. This isolates the
  importance of their dependence for this representation.
- **MECHANICAL:** the joint classifier using order state, queue, price, spread,
  liquidity scale and phase, without the richer market-dynamics inputs.
- **DIRECT_VALUE:** one squared-error value regression on the same full inputs,
  training dates and local-value target. It is a newly fitted comparator,
  not the old STOP_EV strategy or a policy replay.
- **EMPIRICAL:** training-only joint frequencies by side/rung/phase. Its value
  readout uses the same conditional response heads; it is not a constant-value
  baseline. Zero and training-average value baselines are reported separately.

The forecast is conditional on information received by the decision time.
Public flow is measured over [t+37ms,t+1s]; midpoint change ends at t+1s. Own
local value is the focal neutral order's actual modeled tranches over that
interval, each marked at its own fill time+1s. Value units below are **cents per
order action**, including zero-fill actions, unless explicitly stated otherwise.

## Distribution and risk prediction

| Metric | JOINT | INDEPENDENT | MECHANICAL |
|---|---:|---:|---:|
| Joint categorical log loss, lower better | 1.6647 | 1.7929 | 1.8381 |
| Public flow reaches price: AUC | 0.8892 | 0.8889 | 0.8529 |
| Adverse midpoint change below −0.5c: AUC | 0.7859 | 0.7873 | 0.6635 |
| Own fill: AUC | 0.8378 | 0.8368 | — |
| Severe local loss: AUC | 0.8113 | 0.8068 | — |

JOINT reduces joint log loss by 7.15% versus INDEPENDENT and 9.43% versus
MECHANICAL. Marginal ranking is almost unchanged versus INDEPENDENT; the main
gain is the joint relationship. These improvements do not themselves measure
trading profit.

Severe means local value below −30c per original 30-share order. There are
5,333 severe outcomes (4.517% of actions). JOINT predicts 3.918% on average;
INDEPENDENT predicts 2.345%. JOINT therefore still understates average risk.
In its highest predicted-risk decile, predicted/observed rates are
16.45%/18.81%; in its lowest, 0.181%/0.127% (11,807 actions in each).

## Economic calibration

| Metric | Realized | JOINT | INDEPENDENT | DIRECT_VALUE |
|---|---:|---:|---:|---:|
| Mean own filled quantity, shares/action | 2.5850 | 2.2705 | 2.0969 | — |
| Mean local value, cents/action | −4.5533 | −3.9252 | −0.2210 | −3.8374 |
| Local-value RMSE, cents/action | — | 44.7796 | 45.0861 | 44.7848 |
| Local-value Spearman correlation | — | 0.1569 | 0.1315 | 0.1433 |

JOINT's MSE advantage over DIRECT_VALUE is just **0.0235%**, effectively a tie
on this day. JOINT beats the zero and training-mean value baselines by 6.05%
and 5.16% in MSE, respectively; much of the outcome variance remains unexplained.

The aggregate ratio `sum(predicted own value) / sum(predicted own quantity)`
is −1.729c/share for JOINT versus −0.105c/share for INDEPENDENT. The corresponding
realized quantity-weighted local markout is −1.761c/share. Thus much of JOINT's
aggregate loss understatement accompanies underprediction of filled quantity.
This is a ratio over all decision actions, not a comparison of unconditional
forecasts against an outcome-selected filled-only population. It is not
whole-portfolio P&L or a comparison with observed top makers.

Realized mean value increases strictly through the ten JOINT score deciles:
−22.025, −8.552, −5.837, −4.278, −2.758, −1.405, −0.937, −0.635, −0.320,
then +1.215c/action. Deciles contain 11,806 or 11,807 actions. The highest
decile's predicted mean is +0.521c/action. The prediction is not reversed in
this aggregate ranking. The declared 200-draw within-market/side/rung diagnostic
also finds positive association; it is not a validation or profitability test.

## Positive-score diagnostic

| Predictor | Positive-score actions | Share of eligible actions | Predicted mean c/action | Realized mean c/action |
|---|---:|---:|---:|---:|
| JOINT | 4,331 | 3.67% | +1.562 | +3.339 |
| INDEPENDENT | 95,144 | 80.59% | +1.436 | −2.136 |
| DIRECT_VALUE | 8,624 | 7.30% | +1.646 | +0.893 |

JOINT's positive subset contains 634 filled actions and 87 severe-loss actions.
These are different selected populations, not count/exposure-matched policy
comparisons. Their means cannot establish that JOINT is a better trading rule
than DIRECT_VALUE. In particular, 3.67% is a **fraction of sampled order states**,
not quote uptime. Turning this into another positive-score-only entry gate
could remain selective; no quote-availability improvement has been demonstrated.

## Remaining weaknesses

- **Late-contract difficulty persists with full-phase training.** JOINT's local
  value RMSE by minute is approximately 28.89, 26.64, 32.77, 43.37 and 89.54c per
  action. Minute-five n=15,910; its value correlation is0.128. DIRECT_VALUE has
  slightly lower minute-five MSE. This is not solely an untrained-phase transfer
  result: all five phases are represented in the training data here.
- **Some conditional responses are inconsistent.** In 108/118,062 cases
  (0.0915%), value exceeds the bounds implied by predicted quantity and the
  price's 0–100c payoff range. Mean excess over all actions is0.00448c; maximum
  is61.57c. In65 cases, predicted severe-loss probability exceeds predicted
  fill probability; maximum excess is12.72 percentage points. There are no
  expected-own-quantity-above-public-quantity violations. Independent response
  heads need a coherent constraint or shared scenario accounting before action
  optimization; no post-hoc clipping was applied to these predictions.
- **Probe-state coverage is limited.** Assigned ages are0/250/500/1000/2000ms,
  rounded to the next50ms review. This does not validate predictions for orders
  resting much longer, or new-entry action advantages. Size is30 shares.
- **Execution truth remains uncertain.** QueueM, M50 and estimated/imputed
  trade timing are assumptions; 117,710 eligible actions have estimated timing
  somewhere in the reference prefix. All statuses and timing provenance remain
  in the data. No profitability, settlement, inventory-controller or latency-
  robustness claim is made by this prediction-only experiment.

## Provenance and continuation

22 synthetic controls passed, plus valid/planted/nonfinite controls for the
response-consistency instrument. The first-day audit verified all five input
days' raw hashes, data identities, fitted artifacts, all prediction identities,
and probability/value reloads on512 spread-out test rows; an altered-probability
falsifier was detected. The underlying builder reconciles every eligible
action's quantities and own-tranche value to the neutral engine.

Source372d7427; freeze d883b942; first-day scheduler1d7075d5. First-day readout
source b7009409 includes the additional audit/display diagnostics, committed
before this readout. Primary fit790s, peak1,770MiB; first-day readout22s,
peak889MiB. No model or prediction adjustment followed the first-day results.

Artifacts: [report](../reviews/flow_price_20261116/first_day_v1/report.json),
[derived interpretation](../reviews/flow_price_20261116/first_day_v1/interpretation.json),
[calibration](../reviews/flow_price_20261116/first_day_v1/calibration.csv),
[chart](../reviews/flow_price_20261116/first_day_v1/predictions.png), and
[receipt](../reviews/flow_price_20261116/first_day_v1/receipt.json).
Report SHA256:
`b64f914807e48c944e31c9d06b449cf706b0f762ff5c93c7400ba2dc69be5c03`.
Model/data outputs are under ignored `data/predict_fun/research/flow_price_20261116_v1/`.

The remaining four frozen chronological folds continue automatically, with the
same recipe. Next modeling issue to address after their readout: consistency
and calibration of conditional responses, followed by broader order-age support
and action-value tests. No switch to a new model family is justified by this
one-day comparison alone.
