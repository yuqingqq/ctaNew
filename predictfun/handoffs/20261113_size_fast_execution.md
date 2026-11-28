# Size inference optimization prepared, 2026-11-13 12:00 UTC

Operational status is superseded by the [completed handoff checkpoint](20261113_size_fast_handoff.md)
at12:19 UTC. The implementation and preserved-attempt record below is unchanged.

The goal remains active. The complete five/50-share experiment remains frozen
at8a8f4c18, declaration SHA
`cea782c5c074158c44b52f302e79907f23808123535e6cbf01a77f5c40b5183c`.
No historical size stage has started. A computational adapter has passed its
controls and is queued for supplemental declaration and an idle-driver handoff.
Do not change its queued source files. No model, feature definition, threshold,
target, task order or scientific success gate changes.

## Implemented and checked

`pf_size_priority_fast_features.py` transforms the exact committed
`SizeHistory.batch` source to use a unique-key row container and the existing
exact-container History calculation. It keeps the same arithmetic, including
float32 imbalance deltas, missing history and invalid-state refusals. Only
policy inference gets the optimized class. Batch training construction and
the saved-inference verifier keep the original class.

The final suite has8 passing tests, as of2026-11-13T11:56:03.007071Z, source
5cd38db25e0c3813c5b340bee966081765f30a47. Its receipt is
[size_priority_fast_execution_controls.json](../reviews/model_recovery_20261113/size_priority_fast_execution_controls.json),
SHA `3777296bf53a159feab403d903129057f9a7b8da8d822fe2b775698dfe8b391e`.
It binds85 source files. Wrapper: auxiliary core9,245MiB peak,38.90s wall.

- 400 synthetic feature queries across both sizes/sides, partial and untouched
  state, and warm-up/history boundaries compare all112 columns:44,800 values
  agree exactly, including missing values. In that fixture the pandas path
  took1.6925s and the container path0.1557s (10.87x for this component).
- Future-data and hidden-state mutations stay inert; a changed current signal
  changes output. Wrong float precision, changed values, invalid quantities,
  stale clocks and ambiguous source transformations are detected/refused.
- Four native synthetic policy windows compare original/optimized paths for
  five and50 shares. Requests, fills, generation/portfolio tables and complete
  decompressed policy/trade evidence agree exactly. A state-dependent predictor
  exercises both KEEP and PULL. The original verifier accepts both outputs.
- All169 declared tasks retain their arguments and order. Unknown tasks,
  corrupt computational provenance and incomplete controls are refused.
  Both reports reconstruct with supplemental metadata; altered report values
  remain rejected. Final evidence includes the supplemental artifact hashes.

The component timing does not predict a10.87x whole-replay speedup. Engine
matching, target forks, accounting, training and independent checks retain
their costs. This is implementation evidence, not strategy performance.

## Preserved attempts

The first five-test run failed in the test helper when comparing serialized
missing values (`None`) with NumPy `isnan`; the four feature tests passed, and
the native test stopped at its corruption check. Receipt as-of11:51:09.232713Z,
source7c0c4652, SHA
`7920d10ac23637f9d9faac376da55ecf3c94995f99cfbeacd8f2100a2184eff0`.
The helper now converts serialized numeric values/missing values to float,
with an explicit missing-value positive control and changed-value refusal.
The superseding five-test run passed as-of11:52:34.140336Z, source17b63bc5,
SHA `c646f0cc11789da5ba276930867f27381c82d60b7b226b3c7ffeb09f4fc318b8`.
Both attempts and their logs/three receipt mirrors are preserved. The final
eight-test suite then checked the full computational execution adapter.

## Live preparation and handoff

At this checkpoint, pure preparer **1413387** is live and sleeping, source
**9b9649c724649a3e22f40ffae91a9a5a8d363aca**. Its startup receipt is
`model_size_priority_20261113/fast_preparation_started.json`, as-of
2026-11-13T11:59:05.948852Z. Log: `logs/fast_prepare_driver.log`.
It waits for the entire other-owner PULL-DEPTH shell chain **1381042** and a
free auxiliary slot. A manual declaration attempt was refused by the slot
wrapper before launch; no declaration or historical output was created.

After release the preparer will:

1. Verify queued source/controls and declare `fast_execution_declaration.json`
   under the bounded auxiliary wrapper, before any size historical work.
2. Authenticate original waiting driver1167074 by exact command, require no
   children or historical size outputs, stop/recheck it, then terminate only
   that idle driver. Preserve its original source/start/log and publish
   `fast_execution_driver_handoff.json` with the observed process evidence.
3. Launch `model_size_priority_experiment_fast_driver.py`, verify its live PID
   and `fast_driver_started.json`, then publish `fast_preparation_finished.json`.
4. The replacement still waits for the entire recovery chain and its final
   verified evidence. It runs the original169 tasks serially on core8 through
   `pf_size_priority_fast_execution`, adding computational provenance in-band.

The replacement has **not** started at this checkpoint. Original size driver
1167074 is still live. Recovery driver3159222 and Oct27/stress worker1235114
remain live;7/18 full scenario-days are complete. Current worker CPU time
continues advancing. Original302 recovery sources and385 size sources were
verified unchanged. No live trading or protected-day work.

Next: verify the preparation process/receipts before assuming a handoff occurred;
do not start a second size driver. Continue reviewing completed recovery stages
and the first historical size smokes. Model improvement remains unproven.
