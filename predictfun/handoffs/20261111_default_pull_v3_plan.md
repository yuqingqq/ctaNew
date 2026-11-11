# Default-pull loop: equivalent cached feature calculations

As of 2026-11-11 18:28 UTC. Source b3820276, declaration39c8fa44.
Branch `research/predict-default-pull-20261111`.

The complete slow v2 Oct23 build passed all288 windows for both the passive
reference and default-quote neutral path, with7,221 direct feature identities.
It took840.73s on the single numeric core. It is preserved as an immutable
reference under `data/predict_fun/research/default_pull_20261111_v2/`.

The v3 implementation caches displayed depth at each price and the minimum
observed since an order's birth. Every query ends at its own known book frame.
Backward queries used in labels recompute only their past slice. No feature,
policy, label, model, threshold, null, date or execution assumption changed.
The v2 cancellation/late-fill reservation correction remains active.

All13 original controls, the reservation regression/falsifier, and9,180 exact
original-versus-cached feature queries pass. The feature tests include
nonmonotone queries, changing or absent prices, unknown depth and future-data
mutation. A deliberately global future minimum must be distinguishable.

The eight-window smoke passed. Before any additional day or model fit, the
full288-window Oct23 build must match v2 exactly: all generation rows, engine
windows/fills/requests and independent accounting rows. The old artifacts'
hashes were frozen in the v3 declaration before this comparison. A mismatch
stops the loop; no silent acceptance based on similar P&L.

Canonical active output is now
`data/predict_fun/research/default_pull_20261111_v3/`, with receipts in
`reviews/default_pull_20261111_v3/`. Reproduction uses
`predictfun/research/run_default_pull_v3.sh`; the prior driver was retired only
after its complete day finished. All old artifacts remain available.

After all26 stages, run `pf_default_pull_completion_v3` through `run_light.sh`.
It applies the already committed full artifact/model/action audit to v3 and
checks the frozen runtime-parity identities. Actual resting time is computed
as the union of intervals; aggregate order-time can overlap during cancellation.

This is an engineering performance change, not a new strategy candidate.
The full six-day policy and timing/queue comparisons remain pending.
No live change or protected-data read. The completed earlier own-order/calm
experiment remains preserved at d475eccb with no promoted candidate.
