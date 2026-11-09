# Optimization loop: bounded retention with paired fills

As-of **2026-11-09T18:04:28.592256+00:00**. Seventh engineering pass: **228 synthetic traces** (28 explicit,200 seeded), zero exclusions, **33 Python checks** and one Rust positive/refusal test. Native **f20fdb1** matches all **3,701 requests**, **16,628 effective decision states**, **801 fill notices**, **5 retries**, and **73 cancellation-only probes**. There were zero create rejections in this declared population; rejection behavior remains covered by the preceding B4 component suite, not a new LC50 rejection stress test.

The new generated research engine `research/pf_step4_ladder_continuity.py` adds opt-in **LC50** on top of **LZ**. The existing paired engine and frozen v3 are unchanged. This integrates the previously studied50ms policy with actual synthetic fill notifications and paired targets; it is not another economic comparison or parameter sweep.

## Behavior checked

- An accepted small touch order may survive a PM90-only soft exit for at most50ms while the score remains above the lower line (PM90 minus0.01). Other guards, the desired price/size, stop time and reduce-only phase still apply.
- Recovery at exactly the deadline is processed before expiration. A recovery one millisecond later cannot extend the hold. Old timer events cannot close a later hold early.
- Partial fills do not trigger replenishment while the guard is off. If the existing logical pair already describes the retained member, its fill budget is preserved. Otherwise the held logical target shrinks to that member's known remaining quantity, and its other member is cancelled.
- A whole fill closes the hold without reposting while the guard stays off. A late fill on a previously cancelled order can activate CAP10 and immediately cancel a different retained order.
- Prediction/fill probes may only cancel. They cannot initiate or reprice quotes. They cover lower-line changes that do not create an original PM90 decision, as well as notified inventory changes.
- No-LC complete returned outputs and native trace sequences match the parent paired engine on six mirrored cases. Unsupported LC50 rule combinations refuse. Historical file-output parity remains outside the check.

The saved artifacts contain **172 holds**, all closed with duration<=50ms and no new order on their logical lane during retention. Closure counts below describe deliberately constructed software tests, not expected venue frequencies or trading performance.

| Closure reason | Synthetic holds |
|---|---:|
| Same-price recovery | 87 |
| Deadline | 23 |
| Member no longer available after fill | 6 |
| Lower score boundary | 44 |
| Stop / reduce-only phase | 4 |
| Changed desired price / size / hard guard | 8 |

## Evidence and limits

Native reports reproduce byte for byte from the saved synthetic traces. Request/event counts and notified position/cash ledgers reconcile independently. The lifecycle checker rejects an extended deadline and an inserted off-guard top-up; identity/count corruption and a changed native request size also refuse. All172 hold openings match accepted orders and their known remaining quantities; exact-time cancellations reconcile where required.

The Rust oracle consumes Python's effective desired quotes and cancellation-only instructions. It independently exercises native pair mapping, requests and feedback, but does **not** independently implement LC50 policy or the prediction/guard calculation. The Python controls establish those specific scheduling and policy examples. This does not establish arbitrary-policy equivalence or deployed scheduler behavior.

No market or protected Nov9–13 data was read. Synthetic fills remain uncalibrated, and this receipt reports no P&L or inventory-variance benefit. Kernel/wallet integration, cross-market venue limits, historical output-file parity and deployed equal-clock ordering are not verified. The preceding frozen fixed-size rejection exception remains explicit and unchanged.

## Sources and tracking

Source **a565e3f6** and declaration **b3a52e2c** precede formal outcomes. Artifact verifier **c54daff1** records the saved-output checks. During development, the first late-cancel fixture accidentally made the old create post-only-ineligible before it could fill; its public book was corrected before source freeze. No frozen result or market evidence was changed.

- Current combined index: `reviews/optimization_loop_20261109/findings_v7.json`, superseding v6 by adding this pass. Earlier optimization values remain in its linked provenance chain.
- Declaration, receipt and verification: `reviews/optimization_loop_20261109/paired_continuity/`.
- Generated source: `python3 -m predictfun.research.pf_ladder_continuity_build`.
- Validation stages: `pf_continuity_pairs_validation` (`build`, `declare`, `controls`, `run`) and `pf_continuity_pairs_receipt`, serially under `run_light.sh`. New attempts require new destinations; frozen artifacts refuse overwrite.
- Research branch `research/predict-opt-loop-20261109`, isolated checkout `/tmp/cta-opt-loop-20261109`. No production changes or deployment.

All numeric jobs completed18:04:26Z. Peak RSS228MiB for the native build,143MiB for artifact verification;4.5MiB of new data. Slot is free now; recheck process state before C1. No strategy parameter sweep added; cumulative distinct settings remain21.

Next: audit remaining optimization and promotion gates against the completed bounded loop, and prioritize evidence that can change the economic decision. The prior measured coverage benefit still came with increased active exposure; this engineering pass does not reverse that trade-off. Fill calibration and actual timer behavior remain necessary before a profit or risk-reduction claim.
