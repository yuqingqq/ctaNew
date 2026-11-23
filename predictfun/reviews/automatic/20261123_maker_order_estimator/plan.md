# MAKER-ORDER-ESTIMATOR — USER clarification, 2026-11-23

USER: “But we need to estimate when they pull, and repost, post size etc.”

This clarifies the required output of the existing [timing](../20261123_maker_decision_timing/plan.md) and [decision-logic](../20261123_maker_decision_logic/plan.md) packet. Development must build a bounded **order-history estimator**, not merely enumerate unobserved fields. Direct private command logs are not a prerequisite for estimation. Estimated actions must nevertheless be distinguished from recorded actions, and observational ties must not be presented as identified commands.

This is one refinement of the same queued packet, not an extra experiment or resource allowance. The current matched-market comparison continues unchanged. Preserve the fixed top-three/19-BOOK population, deterministic <=96-case sample, shared accounting and <=0.035 core-hour all-in ceiling, subject to the existing task/line ledger.

## Required reconstructed order ledger

For each selected order/history emit:

| Quantity | Required estimate/evidence |
|---|---|
| Original posted price and size | Price from identity where available; original-size estimate or tied alternatives, with cumulative executed quantity as a lower bound and book/priority size constraints. Executed total is not automatically the original size. |
| Posting / activation | Feasible venue-entry window and, only under a named delay model, the inferred submission window. Report point timestamps only as representatives of a specified history, never as observed. |
| Resting / queue progression | Remaining quantity and queue-ahead estimates through the path; front at entry versus front reached later; reset only when a compatible event actually requires it. |
| Pull | Estimated removal time and remaining quantity withdrawn, with separate cancellation / expiry / exhaustion hypotheses. A disappearance is evidence to explain, not sufficient alone to name a cancel. |
| Repost / replace / replenish | Candidate parent/child order identities, new price and size, entry window and gap/overlap with the old order. Distinguish replacement, replenishment, an already-resting second ladder, and an unrelated new order. New fill chronology alone does not establish a repost. |
| Decision context | Inventory/pending exposure, market/queue state and economic explanation from the logic packet, with timing sensitivity. These are assessed after the action reconstruction, not used to force the desired action history. |

Supply `estimate_kind`, bounds, source events, active assumptions, candidate alternatives and ambiguity reasons. Prefer a best-supported history when the observations distinguish it. If several histories fit equally well, report a tied estimate set rather than withholding all output or arbitrarily choosing one. A representative history is useful for visualization, but every inference that changes under its alternatives must be marked.

## Estimation method

1. Reconcile each order hash's fills/identity and the shared price-level observations. Build a local event inventory around the accepted time windows, preserving gaps, receipt/venue clocks and unknown external orders.
2. Enumerate or constrain plausible combinations of original size, additions, executions, removals and re-entry. Book-size change equals the net effect of additions, removals and executions **under the selected feed/clock convention**; do not force unsynchronized snapshots to balance as simultaneous events. Jointly allocate capacity: several makers cannot each claim the same anonymous addition.
3. Reject histories violating identity, quantity conservation, timing, accepted FIFO assumptions or known book evidence. Retain explicit unresolved-volume/gap states. Reuse the existing lifecycle's proven constraints before adding any solver.
4. Rank surviving histories by their agreement with independent public evidence. If a regularizer favors fewer hidden commands, round lot sizes or repeated clip sizes, expose it separately as a **prior/parsimonious preference**. It is not extra evidence and must not collapse equal-observation histories into false certainty. Test sensitivity to the prior. Do not fit a new predictor in this packet.
5. Use already accepted own-order ground-truth artifacts and synthetic known commands to check coverage, timing error and size error where truth is observed. Never extrapolate own-client latency or confidence calibration to all top makers. Do not give a percentage confidence without an applicable validation/calibration basis; use descriptive evidence grades instead.
6. Attach the economic chain only after the reconstructed histories are fixed. Never choose a cancel/post timestamp because it makes BTC anticipation, profitability or an inventory-control story look strongest. Subsequent fills may constrain the historical order, but later returns cannot become its decision-time information.

## Mandatory adversarial example

Retain the accepted identical-public-tape pair: an original 5-share order that fully fills versus an original 50-share order that fills the same 5 shares and cancels the other 45 unseen. Both must remain feasible when their public evidence is identical. If a parsimony prior selects the 5-share history, label that choice prior-driven and show the alternative. The estimator must also recover the correct size/cancel in a positive control where an additional observation genuinely distinguishes them.

Add a corresponding repost control: a new same-price order after a pull versus a previously resting second order that receives its first fill later. Preserve ambiguity where public observations do not distinguish these paths; identify the replacement only when the timing/identity/quantity evidence does. Keep large original size with many partial fills distinct from many small replacement orders.

## First useful output

Show a few complete, predetermined maker paths with an estimated ledger:

`post(size, price, time window) -> partial fills / remaining size -> keep or estimated pull(amount, window) -> candidate repost(new size/price/window) -> inventory-changing take`

Alongside each path, show the alternative that would materially change the inferred strategy and the evidence favoring one. Summarize coverage and errors with order/episode/share denominators. Explicitly estimate how much apparent churn could instead be exhaustion/replenishment, overlapping ladder orders or timing ambiguity; do not report a single exact cancel/repost count where the histories disagree.

The deliverable should answer **“what is our best supported estimate, what assumptions select it, and does the economic explanation survive the alternatives?”** Merely saying “we lack cancel logs” is insufficient. Equally, an uncalibrated confident command trace is insufficient.

## Integration

Freeze this estimator's finite hypothesis rules, ranking/tie behavior and checks before measurement. Start with the smallest existing sample needed to prove useful inference and its cost, then honor the original projection gate. No separate full-day rebuild, new data or policy replay. If the sample cannot support a narrower answer, quantify the limits and demonstrate the competing histories once; do not spend repeated rounds on cosmetic precision. Bundle implementation, evidence and material corrections in one acceptance review.

This note changes no implementation and launches no measurement. The USER's request authorizes this bounded estimation objective within the inherited limits; additional resources still require a separate decision.
