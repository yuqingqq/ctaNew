# MAKER-DECISION-LOGIC — USER clarification, 2026-11-23

USER: “Yes we need the chain of logics, not just what they do at what time.”

This supersedes the **primary deliverable** of [7d71ca40c's timing plan](../20261123_maker_decision_timing/plan.md). Timing reconstruction remains an instrument. The objective is to identify and challenge an economic explanation for a sequence of decisions. A precise timestamp catalogue alone does not complete the task. Development owns implementation; the current MATCHED-MARKET-TURNOVER-GAP-v1 experiment continues unchanged.

## Required unit: a connected decision episode

For each existing selected case, link the maker's relevant orders, both-side fills and takes through its inventory path. Preserve the timing plan's deterministic sample, population, ambiguity and resource limits. Cases sharing the same episode must be linked and counted without presenting them as independent evidence. Do not select only profitable completed round trips or narrate unrelated neighboring trades as one plan.

Every episode should answer:

**Observable state → feasible actions → economic trade-off → observed action → resulting state → next decision.**

For each link show:

1. What the maker could have known: price/flow conditions, quote price, queue bounds, remaining order quantity, inventory and potentially unnotified fills. Separate observable state from inferred private information.
2. What actually changed since the preceding decision: a price move, persistent one-sided flow, queue progress, a partial fill, increased inventory, reduced time remaining, or no identifiable change.
3. What alternatives were feasible: keep that order, cancel it, change price/size, wait, post the other side, or take liquidity. Respect outstanding reservations and timing; an alternative cannot cancel a fill that already happened.
4. Why the observed action could improve **the remaining economic outcome**, including subsequent fills and opposite-side turnover, inventory exposure, fees/impact and loss of queue priority. This is a hypothesis until discriminating evidence supports it.
5. What evidence supports or contradicts that hypothesis; which competing explanation remains compatible. Report the observed later outcome separately from the action's unknown counterfactual value. A profitable outcome does not prove the decision was optimal; a losing exit may prevent a larger loss.

## Competing explanations to test, not assumptions about their strategy

| Decision | Candidate economic logic | Distinguishing evidence / counterexample |
|---|---|---|
| Post before demand reaches a price | Secure queue priority at a price offering acceptable longer-horizon value, so later ordinary flow can generate turnover. | Was the order present before the flow or before its level became the touch? Could it instead have been a late quote, a reaction to a price change, or an inventory-offsetting order? First-fill timing alone cannot decide. |
| Retain after an initial fill or short adverse move | Preserve valuable remaining priority and future paired turnover when its expected benefit exceeds holding risk. | Follow the same order and whole book, including later losses and unclosed inventory. Compare feasible histories where early negative markout is temporary versus persistent. Repeated fills do not prove uninterrupted resting if another compatible history remains. |
| Pull or reprice | A price/flow change makes the remaining order unattractive, or inventory/capital constraints change the preferred exposure. | Did a relevant state change precede the feasible cancel decision? Would the same explanation predict pulling unaffected orders? Keep inventory-driven, price-driven, expiry/exhaustion and unidentified disappearance separate. |
| Take liquidity | Paying spread/fees/impact is worthwhile to reduce unwanted exposure, or to capture a directional opportunity. | Distinguish actual absolute-position reduction from opposing-side orders and overshoots. Does taking follow inventory accumulation without a new directional signal, a signal without much inventory, or both? Adding to exposure is evidence against a pure inventory-unwind explanation. |
| Resume or replenish | Risk/price conditions or inventory room again justify participating, while residual resting priority may still be worth keeping. | Identify the state change, where possible; do not infer a fixed timer from elapsed time or assume a new fill was a new post. |

These are declared rival mechanisms, not a new candidate strategy or thresholds to tune. The aim is to explain *which* mechanism the evidence can distinguish, including mixed or unidentifiable behavior.

## Economic and comparison discipline

- Use the matched-market study's common intervals, marks, fees, turnover and carry/residual accounting where available. Judge the **remaining book/episode**, not only the first fill or a 1-second markout. Short markouts diagnose arrival/adverse movement; they are not the entire value of retaining a queue position.
- Keep queue opportunity cost visible: cancellation may avoid a bad fill but also sacrifice later favorable fills. Public traces do not supply the missing counterfactual automatically. Label that value unmeasured when needed; do not subtract hindsight losses and call them avoidable.
- Evaluate candidate conditions over all supported decision-time histories. If the explanation is stable despite timing uncertainty, exact millisecond identification is unnecessary. If the sign or inventory state changes across histories, label the explanation timing-sensitive.
- Look for contrasting cases in the declared sample: similar market state with different inventory, similar inventory with different flow, similar price with different queue progress. These are descriptive checks, not causal estimates. If the sample lacks overlap, say so rather than selecting new flattering examples.
- Absence of a public fill is not a known WAIT or CANCEL. Any future unconditional policy/rule test needs a separate declared opportunity population. No new fitted decision model, counterfactual replay or significance test is authorized by this amendment.

## Concrete output and completion condition

Add a compact **logic table** to each reconstructed episode:

`decision_window | state/change | alternatives | proposed_trade-off | observed_action | resulting_state | supporting_evidence | competing_explanation | what_would_falsify | identification_status`

Use statuses such as `compatible`, `contradicted`, `timing_sensitive`, `insufficient_comparison`, and `unidentified`; never label private intent as observed. Across the fixed sample summarize mechanism support and contradictions with denominators and connected-episode counts, by maker/day. Show both profitable and losing continuations and unclosed positions. This remains four consumed days, not validation.

The useful answer is, for example, **“this sequence is consistent with building a queue position, harvesting repeated fills, then paying to reduce an inventory imbalance; these observations distinguish it from chasing price, and these links remain unidentified.”** A list of timestamps, an invented hidden policy, or a profitable-close-only story is insufficient.

If a mechanism cannot be separated, name the exact missing evidence or the single counterfactual needed next. Do not start repeated timestamp refinement when it cannot change the economic interpretation. Spend the bounded packet on the links that could distinguish explanations.

## Integration and resources

This is a refinement of the USER-requested timing packet, **not a second experiment or additional allowance**. The original <=96-case sample and <=0.035 core-hour all-in ceiling (including 0.010 review reserve) remain, conditional on the updated 1.85-hour task / 5.25-hour line ledger and the comparison's reserved <=0.10 hours. Freeze the implementation and reading rules before its new measurements. If a richer reconstruction will not fit, prioritize a few complete logic chains and honest coverage within the predetermined sample; report the resource gap before expanding.

Keep the current frozen comparison intact. Bundle material corrections and this logic-oriented deliverable into one development round/combined review where practical. This note changes no implementation and launches no run.
