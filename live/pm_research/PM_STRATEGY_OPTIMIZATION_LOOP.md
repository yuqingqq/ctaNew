# PM strategy optimization loop

**Status: ACTIVE — launched 2026-08-24T11:08:17Z. Research only.**

This is the persistent, single-agent optimization ledger for the Polymarket
five-minute market-making research. It does not authorize live trading,
exchange adapters, order submission, cancellation, or deployment code.

## Objective and incumbent

Optimize fill-weighted five-second maker PnL and inventory risk without
recovering performance through impossible queue priority, post-selection on
seen days, or unmeasured execution assumptions.

- incumbent research baseline: `QR_CANCEL_HOLD_X_SKEW`;
- required no-cancel comparator: `QR_SKEW_ONLY`;
- queue semantics: join behind displayed depth at an occupied touch; only the
  inventory-reducing side may improve one tick, and only when spread >=2 ticks;
- signal cells: BTC H=50/L=25 ms and ETH H=250/L=100 ms;
- v5 threshold: q>0.5 unless an iteration explicitly freezes one signal change;
- inventory skew band: five shares;
- current evidence: three model-training days and two repeatedly inspected
  development days, one window per coin/day; no forward holdout.

The starting receipt is
`data/pm_5min/derived/policy_optimizer_queue_realistic_v1.json`, artifact
`42f56c10e3cc3cfb6b2846248189dde7917d61d759e66e1ff4330a3f01f965d3`.

## One iteration

1. Choose one falsifiable structural change from a named module.
2. Freeze its protocol, candidate name, parameters, data, comparators, and
   adoption bars before running it.
3. Implement it as a new research module. Historical receipts and arms remain
   exact parity controls.
4. Run lifecycle tests, deterministic replay, artifact/provenance checks, and
   all relevant parent tests.
5. Score all five visible days but keep training, development, and future days
   separate. Never relabel a seen day as forward.
6. Record `ADOPT_DIAGNOSTIC`, `REJECT`, or `BLOCKED`. An adoption changes only
   the loop's research incumbent; it never makes a policy decision eligible.
7. Select the next hypothesis from failure attribution, not from a blind
   parameter sweep.

## Candidate adoption bars

A candidate can replace the research incumbent for one coin only if all are
true on the two visible development days:

- PnL delta versus that coin's incumbent is positive on both days;
- mean PnL remains above `QR_SKEW_ONLY`;
- mean terminal absolute inventory does not increase;
- effective cancellation count and cancel/repost traffic do not increase; and
- no parity, determinism, provenance, or lifecycle control fails.

The all-five-day mean is reported as mechanism context, not an adoption gate.
Any candidate chosen after viewing August 23/24 remains development-only and
must later be frozen and scored unchanged on new complete days.

## Hard guards

- No same-price zero-queue placement is executable. Historical `FRONT` arms are
  upper bounds only.
- No maker rebates or incentives are added unless the user changes scope.
- No model hyperparameter or threshold sweep on August 23/24.
- No look-ahead: all features retain their point-in-time as-of timestamps.
- No latency claim: L is cancel-effective latency and stays `ASSUMED` until a
  separate environment measures receive-to-not-fillable time.
- No promotion from these ten windows. The v5 harmful-flow model failed its
  original gate.
- No live trading code or exchange adapters in this repository.

## Iteration queue

| iteration | one change | rationale | status |
|---|---|---|---|
| 001 | minimum hold equals the prediction horizon | a signal predicting harm over H should not repost an inventory-increasing side before H merely because q flickers below 0.5; inventory-reducing release remains immediate | REJECTED BTC + ETH |
| 002 | exact internal deadline event for the minimum hold | iteration 001's next-event wake-up delayed release 171-916 ms beyond the deadline; isolate the intended 50/250 ms treatment | REJECTED BTC + ETH; zero fill/PnL change |
| 003 | probability hysteresis around q=0.5 | exact timer confirms scheduler granularity has zero economic value here; target rapid q boundary reversals directly | REJECTED; BTC passes 4/5 bars, inventory fails |
| 004 | half-order inventory skew band on fixed hysteresis | target iteration 003's sole BTC failure without relaxing its frozen inventory bar | REJECTED; BTC still misses inventory |
| 005 | action-conditioned prevented-value model on queue-realistic JOIN fills | correct the model target for the now-executable action population; no more threshold/band tuning on visible days | REJECTED; parent parity also failed |
| 006 | independent arm event clocks | a cancel timer from one replay arm currently changes when another arm applies its skew intent; repair comparator validity before more optimization | CORRECTED; baseline unchanged |
| 007 | one training row per queue-order generation | iteration 005 has tens of thousands of overlapping decision rows but only hundreds of economic generations; align the fit unit with false-to-true edge arming | REJECTED; 30/23 economic train rows |
| 008 | five windows per coin/day | generation-level fit is support-limited; increase independent order generations without changing model/policy | REJECTED; tree cannot split |
| 009 | generation-compatible tree leaf floor | pinned 200-row leaf minimum exceeds both economic generation samples; test nonlinear capacity with fixed default 20 | MODEL GATE FAILED |
| 010 | independent-day continuation | further selection on Aug 23/24 would be model/sample tuning; accumulate new complete PM+HF days before freezing the next split | WAITING FOR NEW COMPLETE DAY |

## Stop and continuation rules

The loop continues while there is a new, bounded hypothesis that can be tested
without violating the guards. It pauses only for a genuine data dependency,
new authority, or a result showing the remaining high-value work requires real
queue/ACK measurement outside this repo. Running collectors may supply future
complete days; an iteration frozen before those days may score them once without
retuning.

Iteration outcomes append below and update
`PM_STRATEGY_OPTIMIZATION_STATE.json`.

## Iteration log

### Iteration 001 — horizon-aligned minimum hold

- Frozen: 2026-08-24T11:08:17Z.
- Protocol: `MIN_HORIZON_HOLD_PROTOCOL.md`.
- Candidate: `QR_CANCEL_MINH_HOLD_X_SKEW`.
- Receipt: `policy_optimizer_min_horizon_hold_v1.json`, artifact
  `aff9ced94067ef6370e2fe4b361f39e1d31bb3cb3fc8ba42a27183acab52c383`.
- Result: REJECT BTC and ETH. Development deltas versus incumbent are
  -5.67/+15.00 c for BTC and -113.17/+121.35 c for ETH; both increase terminal
  inventory. Mean release lateness beyond the minimum is materially larger than
  H, motivating an exact internal timer rather than adoption.
- Result detail: `MIN_HORIZON_HOLD_RESULTS.md`.

### Iteration 002 — exact horizon timer

- Frozen: 2026-08-24T11:19:15Z.
- Protocol: `EXACT_HORIZON_TIMER_PROTOCOL.md`.
- Candidate: `QR_CANCEL_MINH_TIMER_X_SKEW`.
- Receipt: `policy_optimizer_exact_horizon_timer_v1.json`, artifact
  `df296faf442abc0135d421d43613b7865c499e05e0c0ad06ac5b9975d6693d08`.
- Result: REJECT BTC and ETH. Exact-timer value versus iteration 001 is 0.00 c
  on all ten windows; all fills and PnL are identical. Scheduler granularity
  did not cause the minimum-hold reversal.
- Result detail: `EXACT_HORIZON_TIMER_RESULTS.md`.

### Iteration 003 — q hysteresis

- Frozen: 2026-08-24T11:27:39Z.
- Protocol: `HARMFUL_Q_HYSTERESIS_PROTOCOL.md`.
- Candidate: `QR_CANCEL_QHYST_X_SKEW`.
- Receipt: `policy_optimizer_harmful_q_hysteresis_v1.json`, artifact
  `f4abe2473d28e530db8fc4b29475836fac088949cef78ae37b47d6a0e4ca025e`.
- Result: REJECT. BTC improves the incumbent on both development days and cuts
  effective cancels 772 -> 143, but mean terminal inventory increases
  15.81 -> 18.44. ETH reverses and also increases inventory.
- Result detail: `HARMFUL_Q_HYSTERESIS_RESULTS.md`.

### Iteration 004 — half-order skew band

- Frozen: 2026-08-24T11:34:22Z.
- Protocol: `HYSTERESIS_HALF_BAND_PROTOCOL.md`.
- Candidate: `QR_CANCEL_QHYST_SKEW2P5`.
- Receipt: `policy_optimizer_hysteresis_half_band_v1.json`, artifact
  `f142630125362f6737954eb015d14366d99bacef0f10fcec7910923dabd7d011`.
- Result: REJECT. BTC improves both development days and cuts traffic but still
  misses terminal inventory by 0.94 shares. ETH passes inventory and loses PnL
  on both days. No further threshold/band variant is permitted on these days.
- Result detail: `HYSTERESIS_HALF_BAND_RESULTS.md`.

### Iteration 005 — queue-action-conditioned harmful model

- Frozen before fit or replay: 2026-08-24T11:50:00Z.
- Protocol: `QUEUE_ACTION_HARMFUL_PROTOCOL.md`.
- Candidate: `QR_CANCEL_QACT_X_SKEW`.
- Required change: train/score the cancellation decision only on actual
  queue-realistic joined, inventory-increasing eligibility states from an
  independent `QR_SKEW_ONLY` shadow trajectory, adding actual generation,
  remaining size, depleted queue, order age, and inventory context.
- Receipt: `policy_optimizer_queue_action_harmful_v1.json`, artifact
  `bc30eb68c4bf3722e74d4c8b0b7d974c5c4433633a455309a0c6c9d42877b022`.
- Result: REJECT BTC and ETH. BTC has -5.11% weighted Brier skill and loses to
  the train-selected always-cancel constant on both development days. ETH
  emits zero cancels. The stateful comparison is additionally invalid because
  the no-cancel parent parity check found cross-arm clock contamination.
- Result detail: `QUEUE_ACTION_HARMFUL_RESULTS.md`.

### Iteration 006 — replay clock isolation

- Frozen before replay: 2026-08-24T12:06:00Z.
- Protocol: `QUEUE_ARM_ISOLATION_PROTOCOL.md`.
- Required correction: run every cell on an independent state/signal/cancel
  event heap so one arm's cancel-effective timer cannot resync another arm.
  Existing next-own-event post-fill semantics are preserved.
- Receipt: `policy_optimizer_queue_isolated_v1.json`, artifact
  `7a1091458fcac46b8bdcf8be269c88f57590ae49972d0ac75e63152899272556`.
- Result: `CORRECTED_BASELINE`; all isolation controls pass and both baseline
  cells reproduce the historical metrics exactly on all ten windows. Future
  candidate comparisons must use one replay loop per arm.
- Result detail: `QUEUE_ARM_ISOLATION_RESULTS.md`.

### Iteration 007 — generation-deduplicated action model

- Frozen before fit or replay: 2026-08-24T12:17:00Z.
- Protocol: `QUEUE_GENERATION_MODEL_PROTOCOL.md`.
- Candidate: `QR_CANCEL_QGEN_X_SKEW`.
- Hypothesis: overlapping 10 ms rows repeatedly weight the same resting order
  generation and same future fill, while the policy can submit only on a
  false-to-true edge. Fit once at first eligible observation per generation,
  while retaining the exact-event inference clock.
- State: IMPLEMENTATION; no result was inspected before freeze.

- Receipt: `policy_optimizer_queue_generation_model_v1.json`, artifact
  `3e78dd1bb8e9d616eb98ce1df42abeadfc00b5365c55508c05b894bc5147338f`.
- Result: REJECT. BTC has 30 economic training generations and becomes
  never-cancel; ETH has 23 and becomes always-cancel. Both have AUC 0.5 and
  effectively zero Brier skill. All isolated replay controls pass.
- Result detail: `QUEUE_GENERATION_MODEL_RESULTS.md`.

### Iteration 008 — expanded within-day support

- Frozen before materialization: 2026-08-24T12:25:00Z.
- Protocol: `QUEUE_GENERATION_SAMPLE_PROTOCOL.md`.
- Candidate: `QR_CANCEL_QGEN5_X_SKEW`.
- One change: use the first five recorded windows per coin/day instead of the
  first one, preserving all generation-model and isolated-policy semantics.
- Receipt: `policy_optimizer_queue_generation_sample_v1.json`, artifact
  `2e748945748e4d9730c6fa4ec927036d89905a2c2d50732f24446fedf94089f6`.
- Result: REJECT. The fivefold sample raises economic training generations to
  123 BTC / 72 ETH, but inherited `min_child_samples=200` makes every tree a
  constant. Both q>0.5 fractions are zero and AUC is 0.5.
- Result detail: `QUEUE_GENERATION_SAMPLE_RESULTS.md`.

### Iteration 009 — compatible nonlinear capacity

- Frozen before fit: 2026-08-24T12:39:00Z.
- Protocol: `QUEUE_GENERATION_LEAF_PROTOCOL.md`.
- Candidate model: `QGEN5_LEAF20`.
- One change: `min_child_samples=20` instead of 200; every other model, data,
  target, split, and threshold field remains fixed.
- Receipt: `adverse_move_queue_generation_leaf_v1.json`, artifact
  `ad3b781fe5e02619dac919cbe1679624f73fb82b133ee5a673e46ad36e1ad7eb`.
- Result: model gate fails. Leaf-20 is nonconstant, but BTC/ETH weighted Brier
  skills are -14.05%/-59.92% and AUC is 0.505/0.399. Both lose selected gross
  value on August 23. No policy replay was performed.
- Result detail: `QUEUE_GENERATION_LEAF_RESULTS.md`.

### Iteration 010 — independent data continuation

- State: DATA ACCUMULATION.
- Both PM and HF collectors are running. August 23/24 are permanently seen;
  further model-family, leaf, threshold, H/L, or per-day sample-size changes on
  them are closed.
- Next action: after a new complete overlapping UTC day exists, freeze the
  exact expanded training/independent-forward split before materialization.
