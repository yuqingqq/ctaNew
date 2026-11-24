# POST/PULL TIMING BRIDGE — estimates and cumulative progress

2026-11-24. USER: estimate when makers post/pull to bridge more gaps, without rebuilding the full condition analysis, and show what has been accomplished. We now proceed with explicit conditional estimates rather than treating a complete private-event source as a prerequisite for every incremental step.

Source freeze **6579a3c03e7a2b053480343631bb164d06f08881**; checker-only supersession **ce04e043735dae365f433573a5834686973bad65**. [Contract](contract.md), [computed coverage](outputs/summary.json), [checks](checks.json), [all 88 signed-order chains with relative times](outputs/signed_chain_estimates.csv).

## What this round adds

The accepted reconstruction already contained posting/disappearance windows. This round makes them usable as an explicitly graded timing estimator, joins original signed size and expiry, separates exhausted endings, and connects the remaining estimated endings to successor posting windows. It does not claim the 1,774 posting bounds were newly discovered.

| Output | Coverage | Interpretation |
|---|---:|---|
| Bounded posting estimates | **1,774 / 4,149 filled orders** | Conditional on the inherited lifecycle, displayed-capacity and maximum-match-lag premises. Median window width **8.962s**; **940** windows <=10s, **6** <=1s. |
| Scenario-only posting estimates | **1,800 / 4,149** | Weaker assumptions; median width **1.926s** does not make them more reliable than the bounded tier. |
| Posting estimate refused | **575 / 4,149** | 100 unbounded/invalid, 421 conflict, 54 status-flagged. A narrow conflicting window never becomes a good estimate. |
| Signed terminal budgets exhausted | **41 / 88 fixed signed samples** | Suppress a terminal-pull estimate. Earlier withdrawal/re-entry remains possible. |
| Conditional withdrawal estimates | **40 / 88** | Original signed budget not exhausted, contractual expiry after the window, and timing/capacity gates pass in both views. |
| Incomplete signed endings still refused | **7 / 88** | Keep their conflicting/status-limited evidence; do not force a timestamp. |

The 40 conditional withdrawal estimates are the main new bridge. Their three-view combined windows have **median width 3.161s**, and **35/40 are <=10s**. The FIFO-only windows have median 2.587s and 36/40 <=10s. All 47 incomplete signed orders have expiry after their respective windows; expiry does not explain a disappearance inside those windows under the reconstruction. These are **withdrawal/cancel-or-invalidation estimates**, not directly observed user cancel commands. They additionally assume the relevant fill history is complete and the inherited no-reentry/display premises hold. Unknown/unrecorded execution or alternative lifecycle behavior can defeat that interpretation.

The signed sample was fixed eight per nonempty day/rank cell earlier. It is not a representative cancellation-rate sample, and the 40 are not 40 new ground-truth cancels. The full population is filled orders only; never-filled orders remain invisible.

## How to use the estimated times

Each row keeps **earliest feasible time, a midpoint display estimate, latest feasible time, and evidence grade**. The midpoint is a neutral location inside the interval, not a calibrated most-likely time or confidence estimate. Read market conditions over the full window, not only at the midpoint.

- The posting interval concerns possible acceptance/resting time. For a hypothetical submission-to-effect delay in [0,D], the command interval is [lower−D, upper], for D=0/250/1000ms. These are sensitivity profiles, not measured maker latency.
- Posting upper bounds use the inherited first-fill high clock bound. A first-record-to-midpoint gap is not measured resting time, and repeated fills do not prove continuous rest or queue-front ownership.
- Pull estimates refer to the conditional removal effect. The same delay profiles give possible command ranges. Do not equate UUID/settlement time with the private command clock.
- A successor's estimated posting window can overlap its predecessor's estimated ending. We retain that overlap rather than invent an immediate pull-then-repost sequence.

The full compressed outputs are [post estimates](outputs/post_estimates.csv.gz), [ending estimates by view](outputs/ending_estimates.csv.gz), and [all-view windows](outputs/union_windows.csv.gz). The smaller [signed chain table](outputs/signed_chain_estimates.csv) provides both post and pull times in **seconds relative to market opening**, signed quantity, ending class, successor and existing context classes for all 88 samples. Refused estimates are blank, not zero.

For the 40 withdrawal candidates, ten also have bounded posting estimates and 30 have scenario-only posting estimates. Their FIFO successor links are: **2 conditionally ordered after the ending, 21 timing-overlapping/tied, 3 with a refused successor posting estimate, and 14 without a usable recorded successor**. Even the two ordered links do not prove replacement intent; an independent new order is still possible. This is the current amount of the post→ending→next-order bridge, rather than a claim that all replacements are identified.

## What the estimated windows say about conditions

We reused the existing full-window context, without fitting or recomputing the market features:

- On the **1,774 bounded posting windows**, reconstructed inventory direction is consistently offsetting for **622**, consistently adding for **423**, flat for **10**, and timing-sensitive for **719**. This supports carrying inventory direction into each chain; it does not establish a population posting probability or private motive because these are filled orders.
- Five-second aligned BTC is flat throughout **966** of those windows, against throughout **29**, with throughout **30**, mixed in **745**, and missing in **4**. Thus stable context can be attached to some estimates; midpoint-selected trend labels would hide much of the uncertainty.
- Across the **40 conditional withdrawal windows**, the existing delay-union context finds own fills within five seconds throughout **27** windows. That is partly built into a terminal-after-fill population, not evidence of a causal fill-triggered cancel.
- Adverse five-second BTC is present throughout **2/40**, absent throughout **28/40**, and timing-sensitive in **10/40**. Inventory build-up is present throughout **9/40**, absent throughout **5/40**, and timing-sensitive in **26/40**. No simple “BTC turns against them, therefore they pull” rule is established here. Both priority views give these counts.

These are descriptive window classifications, not selected triggers or a new significance test. [Full context counts](outputs/terminal_context.csv) preserve every class and denominator. Opposite conclusions inside one feasible timing window stay timing-sensitive.

## What has been accomplished across the research line

| Part of the goal | Accomplished | Remaining limit |
|---|---|---|
| Explain the economic gap | [Matched-market comparison](../maker_trace_20261122/v2/report_matched_gap_interpretation_v2.md) separates completed turnover, fees and terminal inventory; top-maker completed turnover is better on the matched sample, while terminal inventory can erase gains. | Recorded maker fills versus our simulated fills; a descriptive gap, not proof of the cause. |
| Measure behavior across opportunities | [Population action map](../maker_condition_action_20261124/report.md): 17,100 maker-seconds, 11,400 market-side seconds, 3,796 retreat episodes and all64 condition/action comparisons. | Anonymous quote changes cannot automatically be assigned to a particular maker. |
| Understand inventory/taking | Large exposure is associated more clearly with reduced same-direction taking than an automatic liquidation rule; taking falls near END. Passive-fill bursts precede opposing takes, but the effect weakens with timing sensitivity. | Associations on four consumed days, not a complete causal controller or profitability proof. |
| Recover order terms and endings | Signed calldata binds original size to full order hashes and executions; 41/88 fixed terminal samples exhaust the budget. | Signed amount is not proof of displayed size, continuous rest or the cancellation history. |
| Estimate post/pull timing | This round exports graded estimates for all4,149 filled orders and narrows40 incomplete signed endings to conditional withdrawal windows, with expiry and conflict checks. | Approximate, conditional times; private command clocks are not observed. |
| Connect turnover to later actions | [Exact-order chain accounting](../maker_terminal_chain_20261124/report.md) distinguishes closing earlier inventory, opening new exposure and later takes against other book inventory. | Six detailed illustrations explain accounting; the population map, not those examples, supports behavior claims. |
| Explain queue-front advantage | Queue sensitivity and repeated-execution evidence exist; the reconstruction retains queue alternatives. | We have not established how often each maker arrived first, retained priority, or pulled/reposted without losing it. |

The resulting chain is now **estimated post window + context → observed fills → inventory/turnover → exhausted or estimated-withdrawal ending → conditional successor window → later observed takes**, with the uncertain links retained instead of omitted or falsely certified.

## Verification, provenance and resources

All4,149 posting identities, 6,640 ending-view rows (3,897 unique orders), 88 signed rows, interval/profile formulas, expiry/capacity gates, successor arithmetic, context exports and summary tables reconcile. FIFO has2,754 rows including non-ok statuses; the old2,741 headline counted a narrower eligible set. The three-view union is available for2,743 unique endings, while1,154 lack a required view; missing views are not replaced by FIFO. All88 signed cases have finite three-view unions, but only40 pass the withdrawal gates.

The independent checker and five falsifiers pass. The first checker reached its corruption test and raised TypeError on a missing/nonmissing comparison instead of its expected ValueError; it wrote no acceptance receipt. The committed v2 wrapper fixes only that refusal, adds two controls, and independently checks the all88 signed-chain rendering. The original run/logs and source remain unchanged; [supersession](checker_supersession.md). No model, new data, network request, tape rebuild, private credential or live activity.

All result-bearing Python used PF_SLOT=codex/run_light, mt_time_* tags counted once. [Ledger](logs/ledger.log): task **1.9133/2.00 core-hours**, line **5.2240/5.30**, standing0.05 reserve fits; this round approximately0.0010 hours against0.020 cap. Peak below253MiB. Local commits only; preserve other-session changes.

## ONE next incremental priority

**Tighten and test only these40 conditional withdrawal windows against nearby saved book/partial-fill evidence, using the41 fully exhausted orders as negative controls for false-pull inference.** Keep original windows beside any refinement; first verify the accepted displayed/pending-quantity units and time bounds, then use only justified remaining-size constraints. Report coverage gained, width reduced, contradictions and whether the existing BTC/inventory interpretation stays the same across the surviving times. No full condition rebuild, new fitted threshold, re-selection of examples or assertion of a private command. If no reliable narrowing is possible, retain these estimates and stop that refinement rather than loop over variants.

This saved-data incremental work is within the USER's current direction and can be declared as one bounded <=0.015core-hour packet, subject to the count/projection gate and unchanged2.00/5.30caps plus reserve. It does not require waiting for a complete private-event source. New data/access, live use or higher caps remain separate decisions.
