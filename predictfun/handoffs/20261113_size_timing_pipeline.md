# Larger-order timing pipeline checkpoint, 2026-11-13 10:45 UTC

The previous goal turn made concrete progress: native age/partial sampling,
three counterfactual branches, exact capture identity, and serialized targets
were implemented and tested. The goal remains active. No historical size
experiment, historical model fit, policy gain, untouched-day validation or live change
is claimed. The original frozen recovery experiment remains unchanged.

## Native action population

`pf_size_priority_action_ledger.AssignedActionLedger` assigns each generation
at submission either an age rule or the first notified partial-fill milestone.
The caller declares the age list and partial weight. Assignment and action
subsampling use different SHA256 blocks. Every generation remains represented;
there is at most one first action, including unknown, withheld, censored and
not-sampled statuses. Age actions can occur before any fill. Native tests show
an initial exhausting sweep has a pre-fill action under the assigned rule while
the partial-only path has no such action.

`pf_size_priority_action_observer` installs this ledger without changing the
sized neutral policy. `pf_size_priority_action_forks.RestingReset` supports
untouched orders; the original partial-only `ResidualReset` remains strict.
The saved capture distinguishes `size-residual-capture-v1` from the new
`size-assigned-residual-capture-v2`. Population and action tables retain
`decision_kind` and `assigned_age_ms`, including generations with no action.

## Three counterfactuals and common re-entry

`pf_size_priority_threeway` constructs KEEP, residual RESET and PULL in the
same native engine. KEEP reconciles to the neutral future. RESET uses actual
cancel effectiveness, last-notice reservations and same-price residual repost.
PULL withholds the action side until the existing causal pressure trigger is
continuously clear under the declared observation-gap rule. This fixed rule
is in `pf_size_priority_withdrawal`; the test configuration is250 ms clear and
at most50 ms between observations. Those are components, not a frozen
historical experiment declaration. A future policy must use the identical
re-entry rule. Initial quoting remains the default; this gate starts only
after an explicit pull.

Pressure observations and gate states are captured and reconstructed. Native
PULL cancellation records bind target identity, side, price, initial quantity,
activation and actual request/effect clocks. Placements must match captured
new generations and may not precede verified clearance. Unavoidable prefix
fills, cash, inventory paths and peak risk are checked. Native three-way
capture and independent neutral runs have identical requests and fills in the
fixtures; source variants M/L compile.

`pf_size_priority_protocol_identity` additionally binds RESET's accepted target,
actual reservation release and replacement quantity/identity. An audit found
that economic reconciliation alone did not previously validate every capture
metadata field. The new check and corrupted-metadata controls close that gap
before any historical size build. The original analytic fixtures remain
available in git; the dataset fixture now supplies complete native metadata.

## Labels and durable artifacts

`pf_size_priority_threeway_target` supplies:

- Full KEEP-minus-PULL utility over the declared horizon, using strict per-fill
  mean1s/5s marks and the unchanged endpoint inventory-square penalty.
- Original-order KEEP value after effective cancellation over the declared
  short window. Unavoidable fills are excluded. Missing marks and a truncated
  short window remain explicit statuses; a valid no-fill window is zero.
- The remainder (full value minus short original-order value), which includes
  later fills, replacement policy and inventory effects. It is not pure queue
  value.
- The separate KEEP-minus-residual-RESET comparison, which includes lost
  priority and the real replacement interruption.

`pf_size_priority_threeway_artifacts` stores the PULL path in
`withdrawal_captures.jsonl.gz`, linked by canonical SHA256 to the existing
paired capture. It writes `withdrawal_targets.parquet` and an explicitly
BUILT_UNVERIFIED construction receipt. Every action/status and zero-action
window survives. All targets reconstruct from saved captures; corrupt target
values are refused even after rehashing the output file. Missing captures,
manifest fields and protocol changes are refused.

`pf_size_priority_threeway_manager` is the complete injected-input native
capture manager for those artifacts. It has an independent sized neutral
reference mode. The shared feature attachment remains compatible and now uses
a batched concat, removing the observed pandas fragmentation overhead without
changing outputs. Historical entry points and experimental checks are still
required; this manager does not authorize undeclared numerical research.

## Executed controls

All sources were committed before their bounded numerical controls. Original
logs, sources and prior receipts remain; no old artifact was overwritten.
These overlapping suites are implementation evidence, not independent samples
of strategy performance. Receipt logs, source-at-commit hashes and all three
receipt copies were verified.

| Receipt | Tests | As of UTC | Result |
|---|---:|---|---|
| `size_priority_assigned_action_controls.json` |9|09:35:16.805847|PASS|
| `size_priority_assigned_native_controls.json` |25|09:43:49.637263|PASS|
| `size_priority_identity_withdrawal_controls.json` |36|09:46:11.019754|PASS|
| `size_priority_threeway_controls.json` |9|09:49:45.247505|PASS|
| `size_priority_threeway_target_controls.json` |8|09:54:22.257752|PASS|
| `size_priority_threeway_artifact_controls.json` |8|09:59:29.996203|PASS|

Receipts: `predictfun/reviews/model_recovery_20261113/`.
Source milestones:0ecbf83d (assignment),fcf6bf49 (native assigned actions),
e4d7edc4 (strict reset identity),45a8c985 (withdrawal gate),87c5c94d/1ece196b
(native three-way/PULL identity),d29e2904 (targets),28c0acd5 (linked artifacts).
The assigned-native launcher144726 waited for the complete best-gap chain115494
and then exited successfully. No control job or exec session remains pending.

## Original taker identity and timing provenance, verified 10:14 UTC

Source `19ffa1d5` adds `pf_size_priority_trade_identity`,
`pf_size_priority_provenance_artifacts` and `pf_size_priority_provenance_manager`.
The generated adapter preserves the original leg index, mapped taker group,
original `how`/time, websocket and estimated group anchors, and both declared
time shifts before the native engine overwrites `how` and drops `tkg`.
It reconstructs the replay clock and distinguishes websocket-anchored,
estimated-anchored and untimed fallback trades. Original `how` also retains the
leg-match versus cell-match distinction. A websocket anchor is provenance, not
an assertion of exact live execution time.

Native five-field trade tuples and their stable sort remain unchanged. Only
trade-event payloads carry metadata. Each captured fill is linked to its source
leg, including through fills and multiple fills on one taker order. Unmapped
negative groups remain explicitly unidentified and are excluded from distinct
identified-taker counts. Per-leg quantity conservation is checked across own
orders. All trades still run through the same matcher; timing subgroups do not
replace complete-policy results. These fields are evaluation evidence, not
predictor inputs.

A saved catalog per neutral window binds neutral fills and every KEEP/RESET/PULL
capture; the writer releases the in-memory catalog after the window is checked
and saved. Empty windows remain visible. Seven synthetic controls pass as of
2026-11-13T10:14:38.563696+00:00, including unchanged native requests/fills,
duplicate-leg stable ordering, original-anchor reconstruction, distinct-taker
counting, rehashed catalog corruption and source-anchor refusals. Receipt
`size_priority_trade_identity_controls.json`, SHA
`5a462d4220fb448f17fc5ef3f874f5e6943c6da883b58152f5d3df3f010d7500`.
Wrapper:244 MiB,10.09 seconds, auxiliary core9. No historical payload read.

## Chronological fitting component, verified 10:19 UTC

Source `c4a9d2d0`, `pf_size_priority_fit`, prepares a one-to-one identity-checked
join of size features and withdrawal targets. Pooled samples are namespaced by
explicit size profile because native generation IDs can coincide across the
five/50-share reference policies. All exclusions remain in prediction files.

Three fixed regression components use the same eligible core actions and pinned
300-round LightGBM parameters: static-plus-size withdrawal value,
history-plus-size withdrawal value, and history-plus-size immediate original-
order value. The earlier profile mean is a baseline. Calibration is the preceding
day; test is later; both size profiles must be represented in each split.
Day-boundary target/mark overlap is refused. The models emit estimates only;
this module does not select policy thresholds or define the final candidate race.
The full historical declaration still needs to fix policy variants/multiplicity.

Four synthetic fitting controls pass as of2026-11-13T10:19:04.246489+00:00.
They reconstruct serialized forecasts, retain unavailable targets, and reject
future dates, target-day overlap, changed profile/action/status identity,
changed input/model/parameters and corrupted predictions even after rehashing.
Receipt `size_priority_fit_controls.json`, SHA
`08c552b6f9e2d041798cd65ea7e132de2fb21c08e8b2c95458acc2ab4ddfc89c`.
Wrapper:219 MiB,1.53 seconds, auxiliary core9. Synthetic tiny fits exercise the
pipeline only; they are not evidence of predictive skill or historical training.

Both receipts, their logs, committed source identities (54/56 files respectively)
and all three receipt copies were checked. The302 original frozen recovery
sources remain unchanged. No bounded control job or exec session remains active.

## Model policy integration, verified 10:44 UTC

The [model policy checkpoint](20261113_size_model_policy.md) records the native
size-aware estimator/policy integration, strict profile binding, repeated shared
withdrawal/re-entry, complete deterministic inference audits and chronological
guard benchmark. Final13-test model-policy,13-test guard and11-test sampling
suites pass; their populations overlap. Two failed preparation attempts remain
preserved with their corrected sources. No historical larger-order model fit
or policy gain is implied.

## Current main experiment and next work

Recovery5/18 scenario-days are complete. Nov2/X completed at
2026-11-13T10:28:15.667677+00:00: four arms x288 windows, exact default parity,
454 reconstructed inference samples and reconstructed recovery events. The
fifth-run review verifies25 named artifacts and all correction links. Driver
3159222 and Nov3/X worker591257 remain live at this checkpoint; revalidate before
relying on this operational state. No final policy verdict yet.

Next: preserve original taker/timing provenance in the full-policy replay, then
complete the historical runner, input/corrected-grid verification, neutral parity,
chronological folds, native full-policy and risk accounting, and the complete
committed declaration with nulls/multiplicity. See the model policy checkpoint
for exact interfaces and remaining limits. Historical work is restricted to the
nine consumed dates. Full numerical runs use core8; core9 is for bounded
controls/preparation only when free. No size historical run, new historical fit,
untouched-day validation, strategy improvement or live change is claimed.
