# Optimization loop, superseding findings v2 — 2026-11-09

As-of **2026-11-09T16:40:02.131643+00:00**. **220 consumed Nov8 windows**, **279,059 original decisions**, **13,077 neutral touch-pull actions** per arm/scenario; zero excluded windows. One partial day, no confidence interval or independent validation.

The loop completed three declared rounds. The main new candidate is a **50 ms maximum grace that processes updates at the exact deadline before expiring**. If the quote remains off, cancellation still occurs at 50 ms; a later event cannot undo it. This changes event priority, not timestamps or the maximum grace.

| Policy, base latency | Request change | Original aligned time change | Off-target time change | Active share-time change |
|---|---:|---:|---:|---:|
| Reference | +0.00% | +0.00% | +0.00% | +0.00% |
| 100 ms unrestricted | -6.99% | +2.65% | +33.87% | +3.40% |
| 100 ms individually reducing | -0.36% | +0.13% | +1.36% | +0.15% |
| 100 ms full reservation room | -0.05% | +0.02% | +0.19% | +0.02% |
| 50 ms timer first | -0.01% | +1.23% | +25.79% | +2.18% |
| 25 ms timer first | +0.00% | -0.00% | +16.72% | +1.11% |
| 50 ms updates first | -6.98% | +2.37% | +25.66% | +2.74% |

At base latency, updates-first 50 ms retains **89.44%** of the 100 ms coverage gain while reducing its **additional off-target quote time by 24.23%**. It bridges **2,376** same-price reopens stamped exactly at the deadline. No claim of higher trading P&L follows from these metrics.

## What each loop round found

1. **Inventory gating is restrictive.** Individually reducing grace retained 4.86% of unrestricted coverage gain (208 admissions); requiring enough room for all same-side reservations retained 0.68% (29). Both added no non-reducing share-time at base latency. This protects the conditional exposure measure but sacrifices most quote-continuity benefit.
2. **Duration alone was misleading.** Timer-first 50 ms had zero same-price bridges in all three latency scenarios; timer-first 25 ms added exposure with essentially no base-latency coverage gain. The guard updates on a 50 ms grid, so expiry priority needed a separate experiment.
3. **Equal-clock ordering is material.** The follow-up processes every recorded event with the exact deadline timestamp, then cancels any order still under grace. Events after the deadline are never used to extend it. This is an exploratory policy discovered from round 2, not an independently validated result.

| Assumed latency | Request change | Original aligned time change | Off-target time change | 100ms coverage gain retained | Extra off-target time removed vs100ms |
|---|---:|---:|---:|---:|---:|
| base | -6.98% | +2.37% | +25.66% | 89.44% | 24.23% |
| faster | -7.01% | +1.12% | +67.89% | 90.90% | 23.36% |
| slower | -5.04% | +2.79% | +4.03% | 85.20% | 27.50% |

Latency assumptions are create / accepted-cancel / pending-create-cancel: base 40/22/95 ms, faster 20/10/50 ms, slower 80/50/160 ms, all with 1 s reservation grace. Each scenario has its own paired reference. These are assumed stress cases, not venue quantiles or separate days.
Computed predicate—fewer requests and more original aligned time versus reference, plus less off-target time than 100 ms, in all three cases: **True**.

## Verification and remaining limits

- 37 combined controls pass, including equal-clock reopen, hard/off cancellation at deadline, rejection of a later reopen, inventory reversal/overshoot, pending cancellation reservations and exact exposure accounting.
- Native observer supplement captures 93 fill-notice states at 0–65 s; original decisions/requests/books/grid probes remain exact. Actual fill notices update only the factual inventory path, never hypothetical orders.
- Artifact checks reconcile prior baselines, all first-two-round overlap, action identities, exact native inventory references, maximum 50 ms grace, coverage gain/loss and summaries. Corrupted counts and mismatched verification links are refused.
- The factual inventory path is exogenous and historical match timestamps remain inadequate for calibrated fill/P&L tests. Active exposure still increases; this does not prove lower inventory variance, favorable settlement inventory, or preserved trading profits.
- The model uses the earlier research sizing planner, not the new Rust A2 paired-order implementation. A2/B4 integration and calibrated fills/forward evidence remain necessary. No live code/deployment, frozen-engine, loss-budget or protected Nov9–13 changes.
- A live port must explicitly order guard refresh before the corresponding deadline check. Replay at nominal native ticks does not validate actual wall-clock timer/processing latency.
- Cumulative distinct policies 21. The first two rounds were declared before outcomes; deadline ordering was separately declared after observing round 2. No candidate is promoted automatically.
- This v2 supersedes the earlier two-round findings. Their numbers remain valid. An exclusive-create helper refused the mutable register update after the first report had been written; that publication issue is corrected here and the old log/outputs remain preserved.

Register and receipts: `reviews/optimization_loop_20261109/plan.json`, `findings_v2.json`, and `{native_inventory,inventory,duration_latency,deadline_order}/`. Full first-two-round table: `handoffs/20261109_optimization_loop.md`. Research branch: `research/predict-opt-loop-20261109`.
