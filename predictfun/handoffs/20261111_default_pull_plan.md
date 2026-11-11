# Quote by default, learn when to pull

User requested a long, tracked research loop on 2026-11-11. The completed
own-order/calm study is retained at commit d475eccb with canonical
`reviews/own_orders_calm_20261111/comparison_v4.json`; no candidate was promoted.
Its narrow soft-pull band does not answer this broader question.

The new loop lives on `research/predict-default-pull-20261111`. Reproduction:
`python -m predictfun.research.pf_default_pull_spec` after committing the full
pipeline, then `predictfun/research/run_default_pull.sh`. Numerical commands
must run through the root `run_light.sh`. Raw output is ignored under
`data/predict_fun/research/default_pull_20261111/`.

1. Run synthetic controls and an eight-window exact frozen-reference parity check.
2. Build nine complete consumed dates, including every submitted generation.
   Sample one probe age per neutral generation before knowing its outcome.
3. Fit market-plus-age and queue models using expanding chronology: three
   initial training dates and six test dates. Retain all zero-fill outcomes.
4. Replay eight full portfolios: frozen reference, matched guard, no signal
   cancellation, pressure triggers, negative guard-value floor, market model,
   queue model and queue model with a cancellation margin.
5. Repeat all eight under conservative trade timing and queue assumptions.
6. Report paired marked value, settlement/carry, strict markouts, spread,
   inventory exposure, order lifetime and churn. Publish all comparisons,
   statuses, null diagnostics and computed success predicates.

The matched strategies share small JOIN orders, retained prices within two
ticks, no ladder, and a reserved inventory range of +/-10 shares. This is a
research design change, explicitly separated from the frozen ladder benchmark.
They preserve minimum order value, stale-data exits, in-flight reservations,
reduce-only after60s and stop at240s. Models evaluate accepted resting orders
every100ms; cancellation waits1s before re-entry. Local training labels compare
KEEP with CANCEL+same-price RESET, including unavoidable fills and lost queue
position. Local labels approximate future portfolio interactions; full replays
determine portfolio performance.

The durable driver writes `progress.json`, an append-only `progress.jsonl`,
per-stage logs, and updates HANDOFF/STATUS plus
`handoffs/20261111_default_pull_progress.md` after every stage. Failed checks
stop later work; incomplete artifacts are retained, never overwritten. The
driver refuses another instance and waits for the shared numeric slot.

No threshold tuning after results. All dates are consumed; no validation or
live-promotion claim. Existing fill/queue calibration limitations remain. No
protected raw data, live strategy, rebates or loss-budget changes. Observed
maker fills do not establish their private cancellation policy.
