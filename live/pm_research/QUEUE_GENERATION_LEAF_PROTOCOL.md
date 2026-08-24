# Iteration 009 — generation-compatible nonlinear model protocol

**Status: FROZEN BEFORE FIT at 2026-08-24T12:39:00Z. Research only.**

## Hypothesis and one change

Iteration 008 cannot test nonlinear signal value because the inherited
LightGBM has `min_child_samples=200` and the frozen generation populations are
123 BTC / 72 ETH economic rows. Change only `min_child_samples` to 20, the
LightGBM sklearn default and a fixed order-of-magnitude reduction. No values
are swept and no development outcome chooses it.

All other tree parameters, the 50-window sample, first-generation row rule,
69 features, same-generation target, economic weights, H/L cells, q>0.5 gate,
and train/development split remain iteration 008 exact.

This is a model-only gate. It fits and scores first-generation rows but does
not replay a policy. Stateful replay is authorized only as a separately frozen
next iteration if every model gate passes for a coin: positive weighted Brier
skill, positive selected gross value on both development days, positive gain
versus the train-selected constant and old v5 on both days, and a q>0.5
fraction strictly between 2% and 98%.

All dates are seen development data; no result is forward or decision eligible.
