# MAKER-DECISION-TIMING-v1 — USER direction, 2026-11-23

USER: “Yes we need to estimate the time when they quote, do decisions etc.”

This is a design/handoff amendment, not a new result or an implementation freeze. Source state inspected: `46cd3cc56`, following the review of `0e21193f2d403d08bb5172aaf727a6a93110e8a9`. Development owns implementation. Continue the already authorized, frozen MATCHED-MARKET-TURNOVER-GAP-v1 comparison; queue this timing packet after it, and include the results in one combined review if practical. Do not interrupt or alter the comparison's frozen estimand.

## Question and first deliverable

Estimate **when the observed makers could have submitted quotes, reached queue priority, and submitted inventory-changing takes**, and the conditions they could have known then. Do not rename first/last fills as POST/CANCEL decisions. The deliverable is an evidence-backed action timeline with time ranges and alternative histories, not a fitted imitation strategy.

First produce one bounded **timing precision and decision-path packet** on the existing Oct 24–27 BOOK markets, for the ex-ante top three from the accepted identities file. Reuse saved v2 lifecycle, fills, frames/levels, book/Binance grids and their receipt hashes. No new data, full replay, fitting or untouched validation.

Use all existing top-three lifecycle rows for a cheap coverage census. For detailed reconstruction, freeze at most four maker-order cases and four take-record cases per address per day: **at most 96 cases**. Select market-first round-robin, deterministic hash-ranked within market, with seed 20261123 and exact keys `(day, market, address, order_hash)` / `(day, market, address, settlement_id)`. Selection must precede any economic, queue-front or precision ranking. Preserve every selected status. Do not replace inconvenient cases with easier ones. Display a few complete paths from this declared sample, not only the most persuasive examples.

## Separate the clocks

For each relevant event record these fields, provenance and bounds:

| Event | Evidence / permitted inference |
|---|---|
| Quote accepted into the venue | Book-capacity/price-time constraints and the first match constrain an acceptance/posting window, conditional on the declared clock scenario. |
| Quote submission decision | Acceptance window minus a separately declared outbound-delay range. With no supported delay bound, submission time remains unidentified; do not assert a calibrated private timestamp. |
| Match | Preserve the existing evidence window, UUID upper bound, receipt time and their provenance. Multiple records with the same settle millisecond are not strictly ordered by that timestamp. |
| Cancel effect | Removal window, only where cancellation is distinguished from exhaustion/expiry. Otherwise retain the alternatives. |
| Cancel submission | Requires an effect window and a declared cancel-delay assumption; an anonymous level drop is not a named cancel request. |
| Fill/order notice | Private arrival is generally unknown. Distinguish reconstructed venue inventory from client-known inventory and pending exposure. |
| Aggressive take decision | Group the observed execution by the actual record/order identity. Bound submission only with an applicable execution/acceptance delay model; do not subtract a universal 50 ms from settlement. |

Other makers' network/processing delays are not measured by our collector lag. Any delay sensitivity must reuse named declared profiles or be frozen before measurement and shown separately from data-identified bounds. Existing own-order ground truth can test the reconstruction instrument on its already consumed artifacts; it does not establish other makers' private latency distribution. Do not infer an exact reaction time from a feature association changing at L=250 ms.

## Tighten only when the evidence permits

1. Start with all existing lifecycle scenarios and statuses. Retain the union of feasible histories; do not silently choose S_FIFO because it has a narrower posting interval. Preserve disjoint candidate windows where present. A contradiction becomes a status, not an averaged timestamp.
2. In the detailed sample, use local book additions/depletions, trade quantities and price-time constraints to eliminate inconsistent candidate placements. **Reconcile all orders sharing that price level jointly**: the same displayed quantity or addition cannot independently be assigned to several makers. Unknown/unfilled orders remain possible. A level addition alone does not identify its owner.
3. Keep the inferred original-size range separate from cumulative executed quantity. A repeated hash is evidence for the same signed order identity; any assumption about uninterrupted resting or impossibility of cancellation/re-entry must be explicit. The existing identical-public-tape/different-command control must retain ambiguity.
4. Before adding a new expensive solver, compare these constraints with what the accepted lifecycle already uses. If no new identifying evidence is available, report that limitation rather than rewriting the engine to obtain narrower-looking ranges.

## Queue and inventory are mandatory

For each selected maker order report: posting/acceptance window; price/depth at each feasible placement; size range; queue-ahead bounds; first and subsequent fill windows; remaining quantity bounds; potential departure/re-entry; and time before the price becomes the touch, where identifiable.

Distinguish **front at acceptance**, **front reached after earlier volume clears**, **front after a partial execution under FIFO**, and **unidentified/conflicting**. A first fill or zero lower bound alone does not prove front at acceptance. Report queue-age ranges, not one invented age. Do not turn a gap or resync into queue replenishment or a reset of the maker's priority.

For takes reconstruct the recorded inventory before/after, pending-notice uncertainty and previous own fills. Separate:

- opposing-side action;
- actual reduction of absolute recorded position;
- full close / reversal / overshoot;
- addition to existing exposure;
- ambiguous state because of notice/timing uncertainty.

A take that reduces exposure is an observed action, not proof that inventory caused it. Ask whether it followed a fill burst, a market move or both, without selecting a trigger after seeing the answer. Report the delay range from the preceding relevant own fill and the contemporaneous BTC/Predict-book state range.

## Associate conditions with decisions honestly

The reconstruction may use later public evidence to constrain historical events. **Decision features may use only information available by the candidate decision time.** Keep those operations separate.

For each feasible time/history, evaluate the unchanged market feature builders on the correct clock. Report which conditions hold across *all* feasible histories, which change sign/category, and which lack a receipt-time bound. Never treat the midpoint, latest posting bound, or first fill as the true command time. Keep venue-state descriptions distinct from inferred received information. Do not derive entry/exit training labels from these filled-order examples.

Do not claim “they post when X” from a positive-only fill sample. An unconditional action-rate/model test would require a separately declared market-clock opportunity population, including non-actions and unobserved participation. That is outside this first packet.

## Evidence, checks and stop rule

Outputs: `cases.csv` with keyed event windows/statuses/provenance, `coverage.csv`, a small set of full timeline tables, and a receipt binding the sampling manifest, inputs, source and checks. No fabricated probability/confidence score. A reconstructed feasible set is not a statistical confidence interval.

Report by address, day and event type: total / bounded / unbounded / conflicting cases; window widths **≤50 ms, >50–250 ms, >250–1,000 ms, >1,000 ms**; original-denominator case counts and associated executed shares; queue classification coverage; and stability of conditions across feasible histories. Show submission-time coverage separately from venue-effect coverage. These are descriptive precision bins, not tuned trading thresholds or evidence of predictive skill.

Bundle essential fixtures: known commands covered by their reconstructed windows; ambiguous identical-public histories remain ambiguous; overlapping orders cannot reuse capacity; gaps/resyncs cannot prove a posting/removal; same-ms ties; receipt-earlier-than-UUID fallback; inventory overshoot and notice timing; feature perturbations after a candidate decision cannot change that decision's inputs. Keep positive and known-bad controls. Refuse a false precise interval rather than drop its row.

**Stop after this packet if most histories cannot distinguish the proposed timing/queue mechanisms.** Say which additional observable would resolve them. Do not start repeated tuning of an unidentifiable latent timestamp or change assumptions to make the intervals narrow. A narrow subset can support a bounded descriptive comparison with its coverage shown; it cannot stand in for all makers' decisions.

## Resources and ownership

The USER has requested this timing analysis. No further permission is needed for the bounded saved-data scope within the existing approved limits. Preserve the comparison's **≤0.10 h** reservation. Proposed timing packet ceiling **≤0.035 core-hours all-in**, including tests and a **0.010 h review reserve**, only if the updated shared ledger fits the existing **1.85 h maker-trace / 5.25 h line** caps. Those are ceilings, not a direction to spend them. Current ledger arithmetic must include concurrent development and informational example probes; no double-counting earlier review reserves.

Developer: freeze the implementation, manifest, checks and projection before measurement. Pilot only the minimum needed to establish cost. If implementation cannot fit or the required evidence is absent, deliver the feasibility/coverage finding and resource estimate rather than launch a larger job. More resources, new data, a new policy/fit or live testing requires a separate USER decision. Use the assigned run_light slot; this review thread does not take the development slot or modify its implementation.

This amendment itself launches no reconstruction run. It records the user's next question and a concrete bounded deliverable; no new result or profitability claim is made.
