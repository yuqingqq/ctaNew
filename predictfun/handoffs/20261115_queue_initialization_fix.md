# Queue initialization correction

New entry and STOP replay engines initialize queue ahead from the last book
snapshot at or before venue activation. A later snapshot can no longer put
post-activation additions ahead of a resting order. Existing queue depletion,
partial fills, shared trade volume, placement/cancellation/notification latency
and policy decisions are preserved.

The corrected engines are versioned separately because historical entry labels
and the currently running STOP-PLAY chain import the original files. Existing
artifacts and ongoing frozen experiments retain their original queue semantics.
Use the new entrypoints below for corrected runs; the old commands intentionally
continue to reproduce the historical experiment.

## Implementation

- `research/pf_queue_activation.py`: shared initialization, contract
  `activation-snapshot-queue-v1`.
- `research/pf_entry_queue_activation_lane.py`: entry engine, derived from
  `pf_entry_repair_lane.py` at `e315abe8`; new execution contract
  `entry-activation-queue-v2`.
- `research/pf_stop_queue_activation_lane.py`: STOP engine, derived from
  `pf_stop_model.sim_joint_g` at `e315abe8`.
- `research/pf_queue_activation_replay.py`: executable entry/STOP integration.
  Uses the corrected engines explicitly, records the queue contract, refuses
  old/mixed output namespaces, and restores original module bindings on exit.

At-price displayed size already present at activation stays ahead. Same-stamp
frames retain the existing engine ordering (`frame time <= activation`). A
partial fill keeps the same order and its remaining priority. Reposting waits
for the existing cancellation acknowledgement and then joins the queue at the
new activation time. Unknown depth remains an explicit zero versus blocked/inf
assumption; a later visible frame may bound that unknown queue through the
existing rule M, but does not rewrite the original queue estimate.

Each accepted order records its initialization contract and snapshot time.
The old next-frame uplift was execution-model look-ahead, not permission to
feed future snapshots into model features. Snapshot timing, cancellations and
actual venue queue priority still require separate validation.

## Verification

Source `00971693`, repost-fixture correction `e3efba98`, replay integration
`07f287be`. As of **2026-11-15T12:08:34Z**, all **12 synthetic regression tests**
pass in the Codex core-8 slot, peak RSS 130 MiB. Both sides and both engines are
covered. Controls reproduce the old incorrect uplift and explicitly reject it.
Tests cover pre-existing queue, activation-time boundaries, empty displayed
queue, post-activation additions, partial fills, cancel/repost, unknown-depth
scenarios, later visibility, future perturbations, invalid assumptions,
unchanged-case legacy parity, and executable routing/restoration.

The first test attempt caught an incorrect expected repost timestamp in the
fixture: cancellation acknowledgement must arrive before reuse of the price.
The fixture was corrected; execution latency was not shortened. That failed
attempt is retained alongside the final passing log in
`reviews/queue_initialization_20261115/`.

This establishes the initialization behavior, not better P&L. No historical
performance replay or model refit used the correction in this session.

## Corrected run commands

Freeze the new experiment's source and input identities before historical
execution, and run serially in the assigned research slot.

For an entry replay with the already fitted local models:

```bash
PF_SLOT=codex predictfun/research/run_light.sh entry_queue_activation_20261027 -- \
  /home/yuqing/pricer-sol/venv/bin/python3 \
  predictfun/research/pf_queue_activation_replay.py entry 20261027 \
  --out data/predict_fun/research/entry_queue_activation_20261115_v1
```

That is a fixed-model execution sensitivity. It does not turn the old training
labels into corrected labels. Retraining requires regenerated neutral action
references, labels and sampling identities under the new execution contract;
do not reuse the earlier label-parity/reuse shortcut across this change.

For STOP-PLAY, run calibration before each corresponding test, then report:

```bash
PF_PLAY_V=6 PF_PLAY_SIZE=30 PF_PLAY_QUNK=zero PF_SLOT=codex \
  predictfun/research/run_light.sh stop_queue_activation_cal_20261027 -- \
  /home/yuqing/pricer-sol/venv/bin/python3 \
  predictfun/research/pf_queue_activation_replay.py stop cal 20261027
```

Use `stop test 20261027` for that test and `stop report` after all declared
folds. Artifacts have the additional suffix `_qinit_activation_v1`; existing
outputs are refused. Movement predictions may be reused because their labels
do not depend on our simulated queue. Policy calibration and execution must be
rerun. The other session can use its assigned slot when its current chain ends.

For new STOP-EV code, import `sim_joint_g` from
`pf_stop_queue_activation_lane`; for new entry label code, import
`pf_entry_queue_activation_lane` and freeze a fresh dataset. Old research
engines are retained for reproducibility, not the recommended queue contract.
