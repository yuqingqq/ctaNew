# First joint flow/price prediction experiment

User authorized building the model and checking predictions before action-policy
work. This is the implementation of stages A–C of the joint-flow plan, with a
coarse joint target rather than a full event-path simulator. No trading-policy
change, replay-performance selection, or new live integration is included.

One deterministic age probe per corrected neutral order generation; six
30-share levels per side, all five minutes, no learned cancellation. Reuse the
existing nine consumed dates and chronological folds: test Oct27, Oct31, Nov1,
Nov2 and Nov3; train before the preceding calibration date. Protected forward
dates remain unopened. Retain all statuses and zero-fill actions.

Market inputs are the existing 71 causal fields, 12 order-state fields and six
absolute-size/liquidity/phase fields. Public trade prints label future flow;
they are not treated as promptly received input signals. Full-contract feature
rows are rebuilt from the archived sources with the corrected fair-price basis.

Joint target: public volume reaching the focal order price in [t+37ms,t+1s],
with classes zero / (0,30] / above30 shares, crossed with five side-signed price
movement classes, split at -2, -0.5, +0.5 and +2 cents. Price movement is the
midpoint at t+1s minus the midpoint known at t, side-aligned. This is an observed
transition target under the declared timestamp convention, not an inferred
list of unobserved additions/cancellations. It does not identify causality.

One 15-class LightGBM challenger with pinned structural parameters and 200
rounds. Comparators: empirical side/rung/phase distribution, mechanically
informed joint model, and independent full-feature flow/price distributions.
Temperature calibration is selected only on the preceding calibration day.

Conditional response heads estimate public volume, own quantity, local value,
severe-loss probability and own-fill probability given current features and the
scenario class. They are trained on all eligible actions, including no fills;
realized scenario classes are used only during fitting. At prediction time all
15 possibilities are enumerated and integrated using forecast probabilities.
The independent-marginal ablation uses those same response heads. An additional
direct squared-error value head is the scalar-model comparator. Conditional
responses in scenarios absent from training support remain a limitation to
inspect, not a claimed fully calibrated distribution of profit.

Own value is quantity-weighted, each neutral fill in [t+37ms,t+1s] marked at its
own fill time+1s. Severe loss means below -30 cents per original 30-share order.
Mean-value targets retain the raw loss tail. Future mark coverage is required
uniformly through t+2s even when an action has no fill; expiry and missing marks
are statuses. No per-fill markouts are substituted with settlement values.

Prediction evaluation: categorical logloss and marginal calibration, volume
and own-fill prediction, local-value MSE/rank/mean calibration, severe-loss
Brier/AUC, by date, side and contract minute. A declared 200-draw within-market,
side and rung value-permutation diagnostic removes between-stratum ranking.
It is consumed-data association, not a significance claim for trading gains.
All economic sums here are local action labels, not portfolio P&L.

Current recorded order clocks (place58/cancel37/notice100/review50) are fixed.
M50, estimated/imputed trades and queueM remain assumptions. There is no model
promotion decision in this experiment. A good prediction result would justify
the separate action-controller experiment and timing/queue sensitivities.

Freeze is the committed full source and `reviews/flow_price_20261116/freeze_v1.json`.
Run synthetic accounting/target/calibration checks, then a two-market Oct23
smoke. Any failed smoke is preserved with a superseding version before a full
run. Build nine days sequentially, fit five folds sequentially, produce a report
and independently verify source/data/model/output identities and calculations.
Use the Codex numerical slot only; resource use is measured by `run_light.sh`.
