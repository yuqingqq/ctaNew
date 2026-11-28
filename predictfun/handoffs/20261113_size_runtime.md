# Sized native runtime checkpoint, 2026-11-13 11:10 UTC

The goal remains active. Native size/cap descriptors, corrected feature
arithmetic, independent neutral parity and complete portfolio accounting now
have passing synthetic controls. No historical size-model fit, larger-order
profit advantage, matched-risk improvement or live change is established.

## Runtime and accounting

`pf_size_priority_runtime.Execution` binds day, scenario, profile, mode,
placement/cancel/notice/shift clocks, queue rule, window count and optional
fill-age sensitivity. Protected dates and unsupported combinations refuse.
Actual arm descriptors replace CAP10 with the declared integer cap and carry
the actual fixed quantity in both endpoints of the AM size descriptor. The
sized policy hook still owns native quantities and reservations.

`installed()` composes the known-empty and late-notice reservation fixes, then
recomputes inherited imbalance deltas using their original float32 subtraction.
Corrected context retains float64 precision. It refuses double installation and
restores installed modules on exit. `environment()` sets all execution clocks,
clears inherited PF settings and restores them, argv and timing tables even on
failure. `execute()` checks manager/source/arm identity and restores SafeReader.
The eventual historical driver must use one fresh process per engine job:
restored environment variables do not invalidate Python's imported clock caches.

Captured targets use the original-taker three-way manager. Independent default
and learned replays use the full-policy original-taker observer. Fill-age stress
is supported only on full policy replays. It is not a target-construction option:
the duplicated fork engine does not have the unique anchor required by that
adapter. This refusal is explicit before execution.

`neutral_exact()` compares native windows, fills and requests, ignoring only arm
descriptor columns, then compares actual generation identity, accepted clocks,
quantities, values, lifetime and first cancellation. The capture and reference
jobs must have identical dates, size, cap, clocks, scenario and windows.

`pf_size_priority_accounting` retains original settlement and portfolio
reconciliation, replacing only the old five-share continuation ceiling with
the actual order's initial quantity. It checks generation coverage/quantity,
inventory cap, terminal inventory and inventory-square exposure. It derives
venue exposure as interval unions, reconciles existing unions, partitions quote
presence by entry/reduce-only phases, and preserves zero-fill windows and
missing-mark counts. Empty-fill settlement uses the tested accounting adapter.
The original native fill order is separately verified by full-policy provenance;
the inherited accounting's tie-sorted inventory peak is not substituted for it.
Equal configured limits do not imply equal realized risk.

## Controls, with failed attempts preserved

| Receipt | As-of UTC | Outcome | Source |
|---|---|---|---|
| size_priority_runtime_controls.json | 11:04:00.189351 | FAILED, 7 tests / 1 error | d6f480ff |
| size_priority_runtime_controls_v2.json | 11:04:36.394501 | PASS, 7 tests | a493c867 |
| size_priority_native_runtime_controls.json | 11:08:29.019256 | FAILED, 2 tests / 2 errors | a3436fa9 |
| size_priority_native_runtime_controls_v2.json | 11:09:15.407718 | PASS, 2 tests | 022f6e91 |

All times are on2026-11-13. The first runtime attempt exposed the unsupported
age adapter on the duplicated target engine; source a493c867 refuses that
combination. The first native fixture incorrectly passed the market-score
threshold to the pressure-clear RecoveryGate, whose constructor takes only
clear duration and maximum observation gap. Source022f6e91 corrects the fixture;
it does not alter a policy or fitted model. Original receipts/logs remain.

Seven runtime controls cover actual descriptors, source compilation, invalid
date/configuration refusals, failure restoration, exact delta arithmetic with a
known-bad precision control, independent native parity with changed-fill refusal,
quantity/cap/cash/population corruption, and empty/nonempty saved accounting.

Two native controls execute six synthetic market windows through actual
generated engine functions and full arm configurations. In the same-budget
fixture, five shares exhaust at the first taker and miss the next; fifty shares
retain their residual and fill against both. Captured three-way construction
matches the independent default requests/fills/windows and generation economics.
Zero-age stress exactly matches the base;500ms stress suppresses the planted
young fills and preserves accounting. These are mechanism/integration checks,
not historical population results or proof of a net queue premium.

Runtime-v2 receipt SHA:
`472012d1cc6b4bb9e0e49383594e454b4f59c85ccfbfaccdb26c6a90b9696801`.
Native-runtime-v2 SHA:
`f355e1a805ff9a1f7a3f4fe087ceab84ab796cf4dbbbdae3fbb885646a6b009f`.
Committed source/log identities and three mirrors were checked. Final native
receipt binds70 source files; wrapper244MiB/14.93s on free auxiliary core9.
No historical payload was read. The original302 frozen recovery sources remain
unchanged. No auxiliary job remains pending.

## Main recovery experiment and next work

Nov3/X completed at2026-11-13T11:03:48.631717+00:00, four policies x288 windows,
438 reconstructed inference samples, reconstructed recovery events and exact
neutral parity. `sixth_full_run_review.json` independently checks25 named
artifact hashes, three receipt copies and four corrupted accounting-link
refusals; its SHA is
`9c27853d590d14d5e3044f7b15fef3810c2894e3f3dc20e2db44eee2adb1e9bb`.
This is a receipt/artifact review, not a second replay. All six primary dates
are complete,6/18 scenario-days overall. Driver3159222 and Oct26/stress
worker911474 were confirmed live at this checkpoint. No final policy verdict.

Next connect the historical data/fold runner and finish its complete declaration:
authenticate corrected-grid/source/input and earlier-only guard artifacts;
run independently sized defaults before target captures; assemble verified
feature/withdrawal tables and chronological fits; load the bound predictor for
full policy replays; implement local matched-action nulls, day-cluster intervals
and whole-policy realized-risk comparisons. Runtime/accounting are now present;
do not rebuild them. No complete historical size experiment declaration or
executable stage driver exists yet. The complete pipeline, candidate/scenario
multiplicity and nulls must be committed before historical size execution.
Only the nine consumed dates remain authorized. Full numerical experiments
stay on core8; core9 remains bounded controls/preparation only.


Continuation at11:33 UTC: the complete historical pipeline is now committed,
control-checked, frozen and queued. See `20261113_size_experiment_frozen.md`.
Source8a8f4c18, declaration cea782c5..., driver1167074 waiting behind the live
recovery chain. No historical size stage has started; goal remains active.
