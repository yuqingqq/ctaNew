# Value-weighted harmful-flow cancellation — development result

**Status: `DEVELOPMENT / INSUFFICIENT_EVIDENCE`; decision eligible: no.**
The experiment excludes maker rebates and liquidity rewards. It adds no live
order, cancellation, venue, or execution path.

Protocol: `plans/BE_ADVERSE_MOVE_PLAN.md` section 13. Implementation:
`adverse_move_harmful.py`. Local receipt:
`data/pm_5min/derived/adverse_move_harmful_development_v5.json`. Artifact ID:
`cec1873b04cc5f0d532c632813f191b0319e393d058d4f9ffda997b7d0bd68b0`.

## Experiment

V5 uses the same 231,092 exact-event action rows and the same already-seen
three-day train/two-day development split as v3/v4. On latency-preventable,
nonzero-value fills it classifies `cancel_value > 0`. The primary LightGBM
weights each training observation by `abs(cancel_value)`; its fixed 0.5
threshold therefore targets the sign of conditional expected economic value.
An unweighted classifier, v4 hurdle regressor and v3 direct regressor are
refitted comparators. No parameter, threshold or cell is selected on holdout.

## Result

| test | result |
|---|---:|
| Positive value-weighted Brier skill | 3/52 cells |
| Positive realized cancellation value | 46/52 cells |
| Selection gain over training-selected constant | 5/52 cells |
| Selection gain over constant on both days | 0/52 cells |
| Aggregate gain over unweighted sign model | 33/52 cells |
| Aggregate gain over v4 hurdle | 29/52 cells |
| Aggregate gain over v3 direct tree | 34/52 cells |
| Full harmful-flow gate | 0/52 cells |

Weighted Brier skill ranges from -0.2947 to +0.0131 and value-weighted AUC from
0.349 to 0.630. Positive realized value alone is not evidence of selection:
most training populations choose `ALWAYS_CANCEL`, so the policy must improve on
that constant rather than merely avoid some adverse fills.

The strongest near-pass is ETH H=250 ms/L=100 ms:

- weighted Brier skill +0.01175; value-weighted AUC 0.628;
- 2,019 nonzero-value training rows and 947 holdout rows; economic-weight
  effective sample sizes 962 and 321;
- +0.1232 c/decision realized cancellation value and +0.00808 c/decision
  aggregate selection gain over the constant;
- beats the unweighted classifier, v4 hurdle and v3 direct tree on both days;
- but loses to the constant by 0.00354 c/decision on 2026-08-24, so it fails
  the predeclared robustness gate.

ETH H=250/L=75 has the best weighted skill (+0.01306) but its constant and
unweighted comparisons reverse by day. The fastest narrow lead is BTC
H=50/L=25: weighted skill +0.00013, AUC 0.545 and +0.00039 c/decision aggregate
gain, but it loses to both the constant and unweighted classifier on one day;
its economic-weight effective sample sizes are only 195 train and 85 holdout.

## Interpretation

Economic weighting is directionally better than counting every harmful fill
equally: it beats the unweighted model in 33/52 cells and all prior model
families in several narrow cells. It still does not robustly identify which
fills should be cancelled. In particular, the sub-50 ms route has only a tiny,
low-effective-sample hint; the clearer ETH signal occurs at a slower 250 ms
fill horizon with 100 ms assumed, not measured, cancel latency.

The present feature set supports preventable-fill forecasting but only weak,
regime-sensitive harmful-flow discrimination. Dynamic cancel×skew replay, real
latency measurement and a decision-facing cancellation layer remain blocked.
Further iterations on these same two visible days would be model selection on
the holdout. If research continues, the ETH H=250/L=100 and BTC H=50/L=25 cells
should be preregistered as diagnostic cells and scored unchanged on new days,
not tuned again on this sample.

## Verification

- The new module contributes 15 deterministic synthetic, gate and receipt
  checks.
- The full adverse/skew self-test suite, contract self-tests, canonical
  contract diff, Python compilation and artifact/provenance integrity pass.
