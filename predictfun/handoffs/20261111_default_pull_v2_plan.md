# Default-pull loop: superseding reservation correction

As of 2026-11-11 18:10 UTC. Source 14da2a06, declaration 6a10e9f1.
Branch `research/predict-default-pull-20261111`.

The initial eight-window smoke stopped on its inventory invariant: the new
prototype could release a cancelled order's reservation before the last
possible fill notification. Its artifacts remain under
`data/predict_fun/research/default_pull_20261111/`; they are not a valid
strategy comparison. No model was fit from that run.

The v2 hook retains cancelled exposure until effective cancellation plus the
configured fill-notice delay. It reads client-notified quantities and follows
the pending-to-accepted object identity. The frozen reference is passive and
unchanged. A positive regression and an intentionally broken reservation case
exercise the correction; the original13 controls also pass.

The superseding eight-window replay PASSED exact reference win/fill/request
parity, independent accounting, the +/-10 new-strategy inventory invariant,
and197 direct saved-feature/cache identities. Peak RSS1,576MiB. These are
engineering checks, not performance evidence.

The full nine-day neutral build has started. Canonical output is now
`data/predict_fun/research/default_pull_20261111_v2/`; declaration and smoke
receipts are in `reviews/default_pull_20261111_v2/`. The original model
parameters, thresholds, null, dates and evaluation design are unchanged.
All future stages are serial. Every completed test updates HANDOFF/STATUS,
`handoffs/20261111_default_pull_progress.md`, and append-only progress events.
A failure stops later stages and preserves the artifacts.

Run/restart using `predictfun/research/run_default_pull_v2.sh` in a persistent
execution session. Completed log checksums are verified. An incomplete stage
requires an explicit recovery or superseding correction; it is never silently
re-run over existing results. The final reporting stage will produce
`comparison_final.json` and `handoffs/20261111_default_pull_final.md`.

The earlier own-order/calm study remains complete at d475eccb; no candidate
was promoted. No live strategy, rebate, loss-budget or protected-data change.
