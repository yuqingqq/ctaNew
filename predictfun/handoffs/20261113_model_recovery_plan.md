# Frozen recovery-model experiment

As of **2026-11-13T07:31:18.208066+00:00**, all nine recovery-data days,
six chronological value fits and both local forecast audits are complete.
All four integration scenarios pass. The first full primary replay, Oct26,
is active on core8 under replacement driver **3159222**. The frozen queue
contains eighteen full scenario-day runs, followed by portfolio reporting and
independent verification. No full recovery performance result exists yet.

The empty-fill settlement correction passed raw parity at 07:31:17.944617 UTC:
all 18 raw parquet tables and the decompressed recovery-event stream match
the original failed attempt exactly. The corrected smoke retains its zero-fill
arm and all eight windows per each of four policies. Original failed source,
artifacts and logs remain preserved. Read [the correction review](20261113_recovery_accounting_repair.md).
Scientific source **78285fe0**, models, targets, policy and success criteria
remain fixed; supplemental accounting source **a4c190ec** changes no strategy.

The [local action-model review](20261113_recovery_local_review.md), as of
07:02:52.217695 UTC, covers 9,945 eligible / 32 missing test actions across six
consumed dates. Reduced local cancellation does not establish a matched-count
advantage over the direct model. The [fixed-proxy audit](20261113_recovery_proxy_audit_review.md),
as of 07:23:08.921632 UTC, covers 9,977 test actions, all short-target eligible;
its day-level forecast-improvement interval includes zero and its nonnegative
prediction region has negative aggregate observed value. Neither finding is
portfolio P&L or untouched-day validation.

Freeze as of 2026-11-13T00:32:13.765800+00:00; source commit
`78285fe02d00962abc5f7148e6001a8f39a0ddc6`.
The complete data, target, fit, policy, report, verification and orchestration
pipeline is committed on `research/predict-model-upgrade-20261112` in
`/tmp/cta-model-upgrade-20261112`. The declaration binds
302 source files, 836
inherited input identities, registries, prior receipts and the execution wrapper.
Authoritative receipt: `../reviews/model_recovery_20261113/declaration.json`.

The first temporal model failed its complete primary/stress comparison. This
experiment tests a separate candidate; that earlier failure and its artifacts
remain unchanged.

| Item | Frozen choice |
|---|---|
| Population | Same neutral default generations and deterministic quarter-sample; one probe per generation; complete exclusion ledger |
| Full target | Ten-second KEEP minus PULL, strict average one-/five-second fill marks minus 0.01c change in actual inventory squared |
| Short component | Original-order fill value in the 250 ms after actual cancellation effect |
| Remainder | Full value minus short value, including later original fills, replacements, both sides and inventory cost |
| Recovery | One market-only short-value model trained on Oct23–24 and fixed for every later label and replay |
| Re-entry | After an explicit pull: 250 ms of nonnegative recovery estimates, gaps no greater than 50 ms; negative/unknown resets; initial quoting stays default |
| Fits | Six expanding outer dates; earlier core only, preceding date retained for diagnostic calibration predictions; pinned 300-round LightGBM parameters |
| Direct policy | Pull on negative predicted full value |
| Component policy | Pull on negative predicted short-plus-remainder value |
| Primary selective policy | Pull only when predicted short value and predicted total value are both negative |
| Full execution | Default/three candidates, six dates, primary clocks, conservative clocks/queue, and full-path age500 sensitivity |
| Null and intervals | 500 action-count-matched draws within day/side/hour/age/inventory sign; 1,000 UTC-day bootstrap draws; all six dates required |

The same two fitted heads feed component and selective policies. Their
comparison isolates the policy gate. Direct uses one fitted head, so comparing
direct against the two-head models confounds decomposition with capacity.
Original temporal/static/chronological-guard comparators remain explicit and
use the same earlier core/calibration dates.

The recovery estimate averages neutral original-order value using market
features. It is a proxy for recovery, not an identified fresh-order entry value.
The label fork makes one intervention and then returns to neutral decisions;
full replay tests repeated learned interventions. Finite-horizon local values
are not additive portfolio P&L and do not establish value beyond the horizon.

The target's inventory penalty and the final risk gate measure different
quantities. `pf_model_recovery_labels.branch_value` penalizes the change in
squared net inventory at the ten-second endpoint. The engine's
`pf_order_policy_hooks.Hook.finish` integrates squared actual inventory over
the full 240.1-second replay, and the final report compares that exposure.
An accurate endpoint-penalized value estimate therefore does not itself
guarantee nonincreasing realized exposure. The separate full-path risk gate
is necessary; this clarification changes no coefficient, label or criterion.

The computed success gate requires positive common-window marked-value
differences, nonincreasing inventory-square exposure and positive lower 95%
UTC-day intervals versus both the chronological guard and the direct model in
all three execution scenarios. It also requires no lower first 60-second any-quote
presence and strictly fewer learned pull decisions than the original temporal
policy in every scenario. Every condition is reported separately. Integrity
PASS, exploratory success and live/forward promotion are different fields.

All 288 windows per date/policy/scenario remain in accounting. Reports retain
missing-mark counts, corrected tie settlement, strict fill marks, inventory,
terminal exposure, phase quote unions, placement/cancel counts and learned
pull decisions. The age500 path suppresses young own-fill credit while public
queue depletion and later decisions continue through the engine. It is a
sensitivity, not an empirical venue rule or a formal P&L bound.

Fifteen controls passed before declaration. The actual historical seed smoke
passed at 2026-11-13T00:36:27.200405+00:00: four windows, 279 neutral generations and 16 sampled
actions. All 16 KEEP forks and captured targets reconcile. Feature attachment
passed at 2026-11-13T00:38:09.391762+00:00, preserving all 16 rows. The independent smoke
checkpoint passed at 2026-11-13T00:39:39.654643+00:00 and binds 14 artifacts. No recovery/value
model is fitted yet; this smoke establishes integration, not performance.

The pure orchestrator started at 2026-11-13T00:35:11.662322+00:00 (PID 3592076). Its 60 stages
include the original study's final integrity check, two full seed-day builds,
one fixed recovery fit, nine recovery-target builds, six chronological value
fits, local reporting, four replay smokes, eighteen full scenario-days and
complete portfolio/evidence reports. It waits for prior workers and uses only
the Codex/core 8 slot for full jobs. The completed early smoke and data receipts
will be recognized and skipped. Logs/partial outputs are never automatically
rerun. No protected data or live changes.

Remaining original study work as of 2026-11-13T00:48:17.793445+00:00: Nov1/2/3 age500, complete age
report/corrected settlement, aggregate presence and final integrity/human review.
The three earlier age dates have passed; 15 of 18 full original runs are complete.
The new orchestrator waits for their completion rather than competing for core 8.

A successful exploratory candidate still needs a complete freeze and at least
five later untouched complete dates, plus unresolved live queue/fill/cancel
calibration. No claim of top-maker parity or live improvement is made.

Execution update as of 2026-11-13T01:22:19.451204+00:00: Nov1 age500 is complete and independently
checked, bringing the original study to 16 of 18 runs and 7,025 reconstructed
inference samples. Nov2 is running; Nov3 and final audits remain before full
recovery training starts. The frozen design and initial integration results
above are unchanged.

Execution update as of 2026-11-13T02:01:27.523255+00:00: Nov2 age500 is complete and checked,
bringing the original study to 17 of 18 runs and 7,325 reconstructed inference
samples. Nov3 is running; final reports/audits follow before full recovery
training. The [separate cancellation-timing review](20261113_model_cancel_timing_review.md)
makes the entry/re-entry limitation explicit and changes no frozen design.
