# Fill timing and queue calibration: capture request

User priority, 2026-11-10: complete calibration before more strategy optimization.
Status: ingestion and identity controls prepared; **calibration is not complete**.
No new Tokyo capture with `wallet_events.jsonl.gz` was found in local run/export
directories, and no connected document session was available. The known exports
are the older Nov8 runs. This is an availability finding, not evidence about what
is currently running on Tokyo.

## Export a completed run from the logging build

Native main79a05d3 includes A4b wallet-event capture and A3 decision reasons. Keep
the actual run header/build identity: pulling a commit does not establish that a
running process used it. No new live trading, restart or deployment is requested
by this document.

For one completed run, preserve and export:

1. `~/pfmm/live/run-<start>/inputs.jsonl.gz` and `outputs.jsonl.gz`.
2. The same run's `wallet_events.jsonl.gz` and `decision_reasons.jsonl.gz`.
3. `~/pfmm/live/journal-<start>.bin`, plus a strict JSON export. From the matching
   native checkout, use `target/release/pmmm-tools journal-export JOURNAL --out kernel.jsonl`.
   This command refuses torn/undecodable journals. `journal-tail` is human-readable
   and omits routine records; it is not the calibration input.
4. The corresponding segment of `live.log`, health records, run exit/drain status,
   and clock synchronization/offset diagnostics if available.
5. Public Predict order-book and trade/settlement records for the same markets and
   interval, preserving original source and receipt times, transaction/settlement
   identities, order hashes where available and `lastOrderSettled`. Core depth
   snapshots alone may not retain all required public identities.

Provide paths, run ID, build commit, UTC start/end, completeness status and SHA256
manifest. Keep large captures outside Git; commit only the compact manifest or
export request. Do not export credentials or environment secret files.

## Data scope before analysis

Nov9–13 are reserved by the existing frozen v3 forward procedure. The current
TODO explicitly says they stay unread until `run_forward_v3.sh`. A new run from
those dates is not automatically admissible for calibration. Identify the exact
run/time range first; using it now requires an explicit scope decision and a
record that the affected data/day has been consumed, not untouched validation.
The new ingestion CLI refuses this protected interval and has no bypass flag.
An authorized scope change must be reflected in a new declaration and code review.

## Prepared ingestion tool

`research/pf_wallet_calibration.py` reads the real A4b JSONL schema and a strict
kernel journal export. It retains every Control.Place action, signed hash, native
fill notice and separate wallet Submitted/Success/Failed timestamps. Duplicate
notifications of one order/settlement pair do not create extra fill legs.

A missing wallet fill is never automatically a verified negative. Dropped
records, incomplete footer, reconnect/liveness uncertainty, missing terminal
status and unmatched kernel fills remain explicit statuses. Source event time,
receipt time and the kernel's truncated status timestamp are retained separately.

An explicit scope JSON supplies `start_ns`, `end_ns`, and `kernel` / `wallet`
objects, each containing `path` and `sha256`. The whole closed run must fit that
declared interval. Run from the research checkout:

```bash
/home/yuqing/ctaNew/predictfun/research/run_light.sh fill-calibration-ingest -- \
  python3 -m predictfun.research.pf_wallet_calibration \
  --scope /absolute/path/declared-scope.json --out /absolute/path/new-output
```

The wrapper changes directory to the shared root. If the code is only in the
isolated checkout, use `-- bash -c 'cd /tmp/cta-fill-calibration-20261110 && exec python3 -m predictfun.research.pf_wallet_calibration --scope /absolute/path/declared-scope.json --out /absolute/path/new-output'`.
The tool refuses existing destinations, changed input hashes and out-of-scope
event envelopes before decoding their payload. Output is an ingestion audit,
**not a calibrated fill model**.

## Calibration acceptance sequence

1. Verify capture completeness, all action identities and filled/unfilled coverage.
2. Establish status-clock semantics and lifecycle bounds on exact order/settlement
   identities. A millisecond Submitted timestamp may still be a settlement-status
   timestamp; precision alone does not establish execution time. Receipt minus
   event time includes clock offset and source processing, not just network delay.
3. Replay the actual submitted/accepted/cancelled order paths against the same
   public tape. Preserve pending cancellations, partial fills, lots and queue state.
4. Predeclare queue/timing assumptions and evaluation before fitting. Measure
   false and missed fills, filled shares, timing error and queue-age/cancel-pending
   breakdowns across all actions. Preserve unidentifiable clocks as intervals or
   unavailable statuses; do not fit a delay to force invalid clocks into bounds.
5. Validate on later untouched complete days before using the model to rank
   strategy P&L. Consumed-data fitting is not independent validation. No further
   capture optimization is promoted while these checks remain open.

Research branch: `research/predict-fill-calibration-20261110`.
