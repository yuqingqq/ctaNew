# Incremental sizing: capture mechanics study

Incremental sizing is the preferred candidate for the next implementation review:
start with one order at the desired size, and add a top-up only when an existing
small order grows. This preserves the small order through later size switches
without paying for two tickets on every initial large quote. It is implemented
in the research planner, not the Rust port or the fill simulator.

Results as-of **2026-11-09 14:42:01 UTC**: **220 consumed Nov8 windows**, first
60 seconds of each, four lanes, **279,059 recorded decisions**, zero excluded
windows. Every policy receives the same recorded prices, sizes and guard pulls.
Acknowledgements, pending reservations and capacity are modeled; no fills are
simulated. The recorded desired path contains factual inventory feedback, so this
is an order workload comparison, not a counterfactual inventory path.

## Comparison

| Sizing policy | Requests | Change | Active time at desired price / desired time | Mean oldest active quote | Mean reserved shares, both sides |
|---|---:|---:|---:|---:|---:|
| Single-order reference | 68,299 | — | 73.31% | 399 ms | 32.46 |
| Always base + top-up | 75,660 | +10.78% | 77.73% | 522 ms | 29.90 |
| Incremental top-up | 65,211 | −4.52% | 76.17% | 515 ms | 31.30 |

All three retain exactly the same desired targets and desired quote time.
Incremental produces **3.90% more active time at the intended price**, **28.97%
greater mean oldest-quote age**, and **3.57% less reserved share-time** than the
reference. Total active share-time rises **2.94%**. Off-target/cancel-pending quote
time also rises **0.82%**; this is not a proof of lower exposure or adverse selection.
Oldest-quote age measures the oldest surviving ticket, often only the small base;
it is not queue rank or the age of all 25 shares.

Requests fall in **188** windows, are equal in **25**, and rise in **7**. In the
122-window v6 run: requests −5.31%, aligned quote time +4.27%; in the 98-window
basis run: −3.66% and +3.48%. These are paired descriptive engineering outcomes
on one consumed partial day, not independent validation or confidence intervals.

Always splitting preserves slightly more quote time but creates more requests
overall: frequent pulls/reprices cancel both tickets. Its in-flight refusals are
3,780 versus 180 for the reference and 179 for incremental. A size-switch request
saving does not imply an aggregate request saving.

The initial screen also tried stricter entry buffers with the original exit line:

| Entry buffer, cents; always layered | Requests vs reference | Desired quote time retained |
|---|---:|---:|
| 0.005 | −15.69% | 79.82% |
| 0.01 | −31.51% | 67.24% |
| 0.02 | −47.38% | 48.33% |
| 0.04 | −70.21% | 21.90% |

Keep these buffers off in the candidate. Fewer requests partly reflect forgone
quoting opportunities; this does not establish better capture. In addition,
reconstructed guard scores disagree with recorded quote-on decisions at the
original pull line on **12,044 / 137,325 wanted-lane updates (8.77%)**. No joined
score was missing/stale. Exact captured guard values are needed before drawing
conclusions about the live entry rule from these reconstructed scores.

## Candidate behavior and review points

- Initial desired 25: one 25-share order. Initial desired 5: one 5-share order.
- Existing same-price 5 grows to 25: keep its identity; add 20 if minimum order,
  capacity and in-flight rules permit. A partially filled base is not topped up.
- Desired size falls back to 5: cancel the top-up; retain the compatible base.
  An original whole 25-share ticket still needs replacement when its remaining
  amount exceeds the smaller target. This candidate does not eliminate every
  resize cancellation: 1,104 remain versus 6,008 in the reference.
- Reprice or guard pull: cancel immediately. Repricing starts one whole ticket.
  No smoothing, grace period for bad signals, or delayed exit is introduced.
- If either split component fails the $1 minimum, use one valid target-sized
  ticket. Keep partial components without continuously replenishing them.
- Count pending creates, pending cancels and the late-fill reservation window
  across all lanes. Reduce-only retention must also subtract existing cancelled
  reservations before keeping active orders against the remaining inventory.
  Newly requested cancellation cannot remove instantaneous fill exposure.
- The model uses 40 ms placement, 22 ms live cancellation, 95 ms pending-create
  cancellation, 1 second cancellation reservation grace, 150-share side cap and
  three pending creates per side. It omits venue throttling, backoff, post-only
  rejection and counterfactual changes to ladder prices.

The initial always-layered port specification remains a separate implementation
proposal. This study adds a lower-request alternative; it does not silently
change its approved semantics. Compare both before finishing A2. No live code,
loss budget, quote thresholds or deployed settings were changed by this study.

## Verification and provenance

- Research branch `research/predict-quote-capture-20261109`; checkout
  `/tmp/cta-quote-capture-20261109`.
- Incremental producer freeze `5e651fae`, declaration commit `f5857810`.
  Six initial arms plus one follow-up chosen after the first screen: total
  multiplicity seven. Same consumed day; no new holdout or profit-based selection.
- **20 controls pass**, covering order identity, partial fills, minimum size,
  pending reservations, shared reduce-only room, pulls/reprices, score freshness,
  input cutoffs and price-aligned occupancy. Invalid clocks and malformed inputs
  are refused. A corrupted aggregate is refused by each artifact verifier.
- Original reference and always-layered outputs reproduce exactly across all
  **440 window-arm rows and 23 columns** in the follow-up. Artifact hashes,
  paired identities, counts and ratios reconcile independently.
- The initial six-arm v1 receipt is **invalid**: the reference mistakenly kept
  a smaller order on size increases. The v2 receipt supersedes it in-band with
  corrected cancel/repost behavior and an added regression control. Only v2
  numbers are used here. Old artifacts remain for provenance.
- Price-aligned audit declared after the overall quote-time result, before its
  own computation. It excludes time after pulls or at stale prices from aligned
  coverage. Its original order counts and desired time reproduce exactly.
- No protected Nov9–13 payload was read. All jobs finished; one process at a
  time, peak RSS below 200 MiB for the follow-up; total outputs about 8.5 MiB.

Evidence: `reviews/quote_capture_20261109/receipt_v2.json` and `verification.json`;
`reviews/quote_incremental_20261109/{receipt,verification,coverage_receipt,report}.json`.
Parquets and execution logs are under the corresponding directories in
`data/predict_fun/research/` (not committed).

Implementation: `research/pf_quote_capture.py`, `pf_quote_capture_study.py`,
`pf_quote_incremental_study.py`, `pf_quote_coverage.py`, and the two artifact
verifiers. Controls can be reproduced without reading market data:

```bash
predictfun/research/run_light.sh quote_controls -- bash -c \
  'cd /tmp/cta-quote-capture-20261109; exec python3 -m unittest predictfun.research.tests.test_quote_capture predictfun.research.tests.test_quote_capture_study predictfun.research.pf_quote_coverage'
```

Frozen study entrypoints deliberately refuse overwriting results. Reproduction
uses the recorded source commit, declared inputs and a separate output location;
do not delete or overwrite old receipts. The historical fill-clock gate remains
intact. Higher trading P&L, better fills and reduced inventory variance require
new matching-event evidence and validation of filled and unfilled actions before
the simulator can rank this candidate economically.
