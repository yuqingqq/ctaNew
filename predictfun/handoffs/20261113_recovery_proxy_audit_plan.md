# Fixed recovery proxy — supplemental forecast audit

Declaration complete as of 2026-11-13T03:28:16.740240+00:00, source commit
`75c39780809a25123cef11b7bc3c3bf64d6d3e33`, before the recovery fit. Seven
positive/refusal control groups passed. Do not rerun `declare`; the remaining
command is `run`, after all recovery data artifacts exist. No historical
forecast-quality result has been produced by this supplement yet.

Operational update, 2026-11-13T04:03:24.911686+00:00: the separate committed
launcher `2f30cab9` is live as PID 1289091, waiting for the remaining eight
recovery-data days. `proxy_audit_runner_started.json` records its source,
controls and dependencies. Do not launch a duplicate while that process is live.


This diagnostic evaluates the recovery estimator already specified by the
frozen model experiment. It adds no model fit, policy, operating threshold,
calibrator or selection gate. The full sequential strategy comparison remains
the decision evidence.

Declare the audit before the first recovery-model fit. The source is
`predictfun/research/pf_recovery_proxy_audit.py`; its `declare` command requires
a committed source, checks the existing experiment freeze, runs positive and
known-bad controls, and refuses if the recovery fit receipt already exists.
The resulting receipt is `reviews/model_recovery_20261113/proxy_audit_declaration.json`.

The population is every neutral sampled action on the seven dates after the
Oct23/24 seed dates: Oct25 is a separate calibration diagnostic; Oct26/27/31
and Nov1/2/3 are the six consumed test dates. Keep all statuses and save all
forecasts. Errors use the explicitly eligible short targets, including no-fill
zeros. Do not select completed fills or a future-surviving price level.

The target is the same original-generation fill value in the 250 ms after
actual cancellation effect that trains the fixed recovery model. Compare its
forecasts with the eligible short-target mean from the seed training data only.
Reconstruct every market feature from the corrected causal grid, and verify the
exact model, training-data and later action identities. Report bias, RMSE, MAE,
squared-error improvement, fixed prediction bins and the observed value where
estimates are nonnegative. Keep the preceding calibration date separate.

Intervals resample all six UTC test days with 1,000 draws. Their estimand is
the equally weighted mean of daily squared-error improvements, distinct from
the pooled action-weighted metric. No action-level confidence interval,
per-fill accuracy claim or new action-selection null is introduced. Positive
and known-bad controls cover perfect/flat/wrong forecasts, zero boundaries,
empty/nonfinite input, incomplete/duplicate days and corrupted feature equality.

Run through the shared wrapper in the Codex slot after all nine recovery data
builds exist, or in a genuinely idle spare slot if the bounded diagnostic is
advanced. The commands are:

```bash
python3 -m predictfun.research.pf_recovery_proxy_audit declare
python3 -m predictfun.research.pf_recovery_proxy_audit run
```

The unchanged primary driver does not run this supplement automatically.
The separate `model_recovery_proxy_audit_driver.py` launcher waits for all nine
verified recovery-data receipts and a free auxiliary slot, then invokes the
already declared `run` command through the shared wrapper. It records its
source/control/launch identity and refuses complete or partial audit reruns.
The launcher does not change the frozen forecast audit or primary experiment.
Once its started receipt exists and its process is live, do not launch a second
copy or run the audit manually. Complete the audit before the final human
review and bind its receipt there. Audit source changes require a superseding
declaration; launcher orchestration is separately committed.

This is still a neutral resting-order proxy. Neither forecast calibration on
these actions nor a positive nonnegative-estimate region identifies fresh-entry
value or performance on the states visited during recovery waits. Their local
values are not additive portfolio P&L. These consumed dates do not establish
forward or live improvement. No change to the frozen candidate follows from
this diagnostic.
