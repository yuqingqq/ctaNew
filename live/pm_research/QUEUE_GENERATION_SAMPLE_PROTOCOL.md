# Iteration 008 — five-window generation sample protocol

**Status: FROZEN BEFORE MATERIALIZATION at 2026-08-24T12:25:00Z. Research only.**

## Hypothesis and one change

Iteration 007 left only 30 BTC and 23 ETH economic training generations. Keep
the full iteration-007 model, first-generation row rule, features, labels,
threshold, H/L cells, and isolated policy unchanged, but select the earliest
five recorded windows per coin per UTC day instead of the earliest one.

This yields 15 training and 10 visible-development windows per coin. Five is a
pre-frozen fivefold support increase, not selected from model outcomes. If a
selected window lacks complete point-in-time PM/HF inputs it is reported and
the iteration fails; it is not replaced after observing labels.

Candidate: `QR_CANCEL_QGEN5_X_SKEW`. Baseline and comparator are rebuilt on
the same 50-window sample with one isolated replay per arm. All model and
adoption gates from iteration 007 remain exact. Day metrics first average
within UTC day; the two development-day signs remain the adoption units.

These dates are already seen and never become forward. No hyperparameter,
threshold, row selector, H/L, skew, hold, queue, or incentive change is allowed.
