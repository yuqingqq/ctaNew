# PM/HF data correctness repair — 2026-08-24

## Outcome

The frozen 50-window research pipeline is now mechanically conformant: every
queue-action trace reproduces the isolated authoritative `QR_SKEW_ONLY` fill
path.  The cancellation-effectiveness conclusion is still **unavailable**, not
validated, because public market data contains no measured cancel-effective
acknowledgement for our hypothetical order.

This repair changes the interpretation of the earlier harmful-flow results.
They remain negative development diagnostics, but they are no longer evidence
that a public-tape model can measure preventable fills.

## Correctness defects repaired

1. **HF collector stamp point.** Historical HF rows were stamped after JSON
   parsing while PM rows were stamped immediately after `ws.recv`.  HF v2 now
   stamps immediately after `ws.recv` and before parsing.  Every collector
   process appends its exact start boundary and stamp semantics to
   `data/mm_hf/collector_runs.jsonl`; historical rows remain explicitly legacy.

2. **Silent empty exchange metadata.** A restricted Binance REST response was
   previously accepted as an empty successful `exchange_info` snapshot.  The
   collector now requires a successful HTTP response and all requested symbols,
   writes atomically, and fails loudly without overwriting valid metadata.  The
   endpoint currently returns HTTP 451 on this host, so this metadata is
   honestly unavailable.  The two historical empty snapshots were preserved as
   `*.json.invalid-empty` so consumers no longer see them as valid metadata.
   Websocket collection remains healthy.

3. **Queue warm-up mismatch.** `build_pm_tape` warmed book state from -60 s but
   discarded trades before window open.  The authoritative replay retained
   both.  The trace now retains the same -60 s trade warm-up.

4. **Lossy replay state.** The feature tape correctly deduplicates unchanged
   top-of-book states, but the queue engine must resynchronize after every
   applied PM mutation because a preceding fill can change inventory skew.  A
   separate lossless replay-state clock now preserves every such mutation while
   leaving the compact feature state unchanged.

5. **Epoch-float event reordering.** Receipt nanoseconds were divided by `1e9`
   before subtracting the 2026 epoch.  Distinct sub-microsecond events could
   collapse to one float and be reordered by event type.  Replay and signal
   clocks now subtract integer nanoseconds first and convert only the small
   relative interval to seconds.  The source-profile hash records this change.

6. **False preventability labels.** A fill notification received after a
   decision may describe a trade executed before the decision.  The new
   `da_execution_timing.py` uses PM event time only to reject fills proven stale.
   It deliberately has no positive `PREVENTABLE` state: without our own cancel
   acknowledgement, unresolved candidates remain optimistic diagnostics and
   cannot pass the causal timing gate.

7. **Numerical gate false positive.** Brier improvements below `1e-9` were
   reported as positive due only to floating-point noise.  Model gates now use
   the existing numerical epsilon.

## Validation

- DA timing selftests: delayed notification, receipt-before-effective,
  unavailable event time, unavailable clock calibration, and collector stamp
  boundary all fail closed as intended.
- Alignment selftests: 29 checks, including knowledge-time reads, coverage
  edges, and legacy/v2 manifest segmentation.
- Feature/action/generation selftests pass, including a regression that
  preserves two receipt events only 100 ns apart.
- Full frozen-sample replay: 50/50 windows pass exact trace/authoritative fill
  parity.  The audited fill population is 5,866 BTC fills and 1,254 ETH fills.
- All seven generation-sample controls pass: frozen sample, five windows per
  coin/day, generation identity, deterministic isolation, all-false parity,
  model receipt round-trip, and trace/authoritative parity.
- Corrected generation receipt:
  `data/pm_5min/derived/policy_optimizer_queue_generation_sample_v1.json`,
  artifact ID
  `3414fc563b673e7f010614b367b2191dee69d5f52cf6544d0f25162d45d887ba`.

The public PM tape itself is demonstrably delayed and reordered.  Across the
50 windows, all 60,358 retained PM trades had event timestamps, but venue event
time moved backward in receipt order 1,066 times for BTC and 143 times for ETH.
Delay above each window's minimum clock bracket reached p99 1,464 ms for BTC
and 316 ms for ETH.  This is why receipt-after-cancel cannot be treated as
fill-after-cancel.

## Corrected model read

On the repaired generation sample, BTC has 52/31 economic train/dev rows and
ETH has 59/39.  Zero economic rows have positively measured cancel timing, so
both causal gates fail.  On the still-optimistic unresolved labels, both fitted
models are constant (`ROC AUC = 0.5`); Brier skill is approximately
`9.2e-10` for BTC and `5.0e-10` for ETH, which is numerical zero.  BTC selects
always-cancel and ETH never-cancel.  Both strategies remain rejected.

The DA timing filter rejected 1,888 BTC candidate assessments and 287 ETH
assessments whose venue events were definitely no later than the decision or
assumed cancel-effective time.  It retained 1,633 BTC and 1,350 ETH assessments
as unresolved diagnostics; none is promoted to measured prevention.

## Data needed next

More public rows can improve the adverse-movement prediction study, but they
cannot validate cancellation effectiveness.  That requires a separate measured
execution dataset containing, for each owned order: local cancel-send time,
venue cancel acknowledgement/rejection time, user-channel fill receipt, venue
fill/event identity, and clock-quality metadata.  Until that exists, the valid
research target is future adverse movement or post-fill markout—not claimed
prevented fills.
