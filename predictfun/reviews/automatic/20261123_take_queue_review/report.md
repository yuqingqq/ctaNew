# TAKE-LOGIC v2 / QUEUE-FRONT v1 — combined review

2026-11-23 UTC. **Full pinned target: 0b2b0b13eb19f9f8c3df40d149a31c0b832905f8**.
Range: `af916aba6..0b2b0b13eb19f9f8c3df40d149a31c0b832905f8`.
Reviewed in detached checkout `/tmp/pf-queue-review-0b2b0b13`. Review only; no implementation or frozen receipt changed. Previously authorized static reviewers covered the two modules; only the root ran checks.

**Result:** accept TAKE-LOGIC v2 as a descriptive correction, including withdrawal of the inventory-size trigger. Do **not** accept the queue packet's certified front holders, maker posting/reposting mechanism, or front-versus-behind value interpretation. The saved measurements remain useful with narrower labels. Corrections and the posting-decision measurement should proceed in **one round**, not another cleanup-only cycle.

Population/as-of: ex-ante top three, 19 BOOK markets, Oct 24–27, four consumed days. TAKE finished 2026-11-23 16:00:39Z: 17,100 maker-seconds and 4,124 raw take records in the panel. QUEUE finished 17:22:46Z: 21,385 fresh-level episodes; the filled-order endpoint view has 4,149 orders. No untouched validation, interval, strategy replay, or profitability finding.

## Consolidated actionable checklist

Paths below are relative to `predictfun/reviews/maker_trace_20261122/`. BLOCKING means blocking the named inference or its use as a reliable action label; a checked downgrade to a scenario/observable is sufficient to continue the next measurement.

### 1. P1 — BLOCKING: an equal book-size decrease does not identify the trade that caused it

**Evidence:** `tools/queue_front.py:511-575`, especially `:545-559`; old/new-life assignment `:610-645`; certification `:659-685`. Contract amendment 2 `queue_front_contract.md:165-177`; receipt `v2/report_logic_chains_v3.md:44-50`.

`book_timed` requires a unique net decrease equal to the record's quantity and no competing recorded trade claiming that decrease. It does not rule out an anonymous cancellation, replenishment, or their netting within the frame interval. The contract's requirement that no other addition/decrease explains it is therefore not established. Old-life exclusion and the first-record ranking inherit this assumption.

The independent reproduction in [check_review.py](check_review.py) constructs **identical public frames and trade records with two different first fillers**:

- Fresh level displays 100 shares at 2,000 ms, 70 at 3,400 ms, and 50 from 4,200 ms onward. A's 30-share match has hard window [3,000, 5,000]; B's 20-share match [3,100, 5,100].
- History A: A fills at 3,300, then B at 4,100.
- History B: an order behind A/B cancels 30 at 3,300; B fills at 4,100; A fills at 4,500 and 30 new shares arrive at 4,501, before the next public frame. Initial ordering B/A is compatible with the same creation interval and time priority.
- Both histories satisfy the public sizes and the records' hard windows. The pinned code nevertheless certifies **A**, with `n_unexplained=0`, in both.

This is a synthetic falsification of identification, **not** an estimate of how many real labels are wrong. In the saved real table, **845 of the 888 certified levels use book timing**, so it is material to the headline.

A related static error remains even under the no-netting scenario: `:629-641` puts an unmatched record in the creation interval when only its **lower window endpoint** lies there, then orders it before all book-timed records. A window starting inside that interval can extend well beyond it. “Possible only inside the creation interval” is not checked.

**Disposition:** retain book-drop matches as conditional timing candidates. Carry alternative histories into old/new-life and earliest-address sets; do not convert uniqueness among net decreases into certainty. Add the masked-cancellation/replenishment and straddling-window controls. A conservative regrade is enough; an exhaustive hidden-order solver is not required for the next round.

### 2. P1 — BLOCKING for the queue-value comparison: later records are not necessarily orders behind the front

**Evidence:** `queue_front.py:690-699,1004-1005`; economics `:1085-1096`; receipt `:83-93`.

Every record after the selected earliest set is labelled `behind`, including later partial fills of the **same order hash**. A synthetic untouched 100-share order filling 30 and then 20 receives one front and one behind label. In the real saved table, **407/926 behind rows**, representing **7,488.58 shares**, have a hash in that level's initial front set; 188 enter the valued cohort. Those rows do not demonstrate loss of queue priority.

The arithmetic of the published gross markouts reproduces, but it compares selected earliest-record fills against later-record fills. It does not establish the economic effect of queue rank. The receipt also puts all-row counts beside values computed on a much smaller common cohort:

| Saved role | All rows | Valued rows | Valued shares | Gross 10 s c/share |
|---|---:|---:|---:|---:|
| front | 919 | 349 | 4,518.21 | +2.58855 |
| behind | 926 | 565 | 8,101.21 | -1.26162 |

The common cohort requires valid 1/10/60 s marks **and resolution** (`:464-493`), not merely a valid 10 s mark. Deep-level valued counts are 41/152 versus 12/28; rank 1 is 11/42 versus 9/42. Censoring and price/time composition are not matched away. These are inherited settlement-clock gross values, not values at an identified private match or posting decision.

**Disposition:** report earliest-set / later-record economics, explicitly split later fills of the same hash, and show valid/all counts and valid shares. Withdraw “does the front pay?” as a queue-effect inference. No new mark lookup is needed to correct this table.

### 3. P2 — BLOCKING for maker decision chains: the eventual first filler is not necessarily the creator of the level

**Evidence:** `queue_front_contract.md:47`; `queue_front.py:950-1001`; receipt `:70-81,116-118`.

Even if the first fill's owner were known, earlier unfilled orders may have cancelled and been replaced without a visible net-size change. Time priority identifies the surviving order at execution, not who created the cell, its posting time, or a continuous occupancy history. The code assigns the **anonymous creation frame's** improve/deep/re-establish context to the eventual selected holder. Re-establishment checks a prior clearing of the cell, not a cancellation and repost by the same maker.

Likewise 129 ms versus 348 ms is the saved **cell-event endpoint proxy** (`:972`), not a measured maker repost latency. Lower/upper interval columns exist at `:1000-1001`; they should be carried, without assigning ownership of both endpoints. “Improving” is the largest reported category (48%), not a majority.

**Disposition:** name these anonymous cell-creation contexts associated with scenario first fillers. Only claim a maker action/continuous priority where all retained histories support it. Do not attach the eventual fill owner's inventory to an earlier anonymous creation as though the creator were known. This boundary is essential for the planned posting-logic study.

### 4. P2 — BLOCKING for the claim about where makers posted: endpoint occupancy is not occupancy at posting

**Evidence:** `queue_front.py:1044-1066`; receipt `:52-63,116`.

`joined_rows` reads displayed size at `post_lo_union_ms`, the inferred posting-window **lower endpoint**. A positive size is labelled `joined_existing_queue`; zero is `cell_empty_before_window`. The cell may clear, refill, or change ownership before the actual posting time. The scenario union also needs its prior hard/constrained/scenario/conflict grade; a finite endpoint alone does not upgrade it.

“Cell empty at the inferred window's lower endpoint” is supported by this lookup, conditional on that window. “55% were posted with no queue and the rest joined one” is not. All 4,149 orders are selected by subsequently filling; unobserved/unfilled maker orders are outside this population.

**Disposition:** preserve occupancy as an endpoint observation with the lifecycle grade and timestamp. Rename the positive-size category symmetrically. Carry unknown states and refrain from using it as an actual-post queue label.

### 5. P2 — BLOCKING for the negative front-win conclusion; the computed answer is unidentified

**Evidence:** `queue_front.py:1021-1035,1125-1151`; receipt `:65-68,118`.

The receipt's “They do NOT win fresh-level fronts more often” contradicts its own `unidentified` reading: the scenario bounds 2.2–34.8% include the reported fill share of 11.7%. Failure of the lower bound to exceed a reference is not evidence of no advantage.

There is a population qualification too: the bound denominator contains identified **plus ambiguous** levels, while the fill share and permutation use attributed fills from identified levels. Identification itself depends on the addresses in the earliest set; the null permutes fill labels **after** that selection. It is a conditional label shuffle, not an unconditional chance of winning a front. Finding 1 additionally means the candidate histories are not complete, so these numbers are not certified real-world bounds.

**Disposition:** retain `unidentified`, give each denominator, and call the interval a bound within the declared reconstruction scenario. Do not tune an alternative queue-win test merely to replace this null result.

### 6. P2 — NONBLOCKING: amendments are adaptive development, not demonstrated outcome-blind identification fixes

**Evidence:** contract amendment 1 `:133-148`, amendment 2 `:150-186`; `logs/mt_qf_pilot1.log:18-37,158-227`, `mt_qf_pilot2.log:18-37,159-248`; output digest `queue_front.py:1387-1410,1580`.

Amendment 1's one-address earliest set is reasonable **at address level**, conditional on an adequate set of possible match times/lifetimes. It does not identify which order was first or who created the level. Amendment 2 is a reasonable hypothesis to investigate, but its no-netting premise cannot be called an identification fix (finding 1).

The pilots did not produce status counts only: both saved and printed economics, permutation p values and readings. Their stated motivation may still have been coverage; this review cannot infer what the developer attended to. The legitimate description is an adaptive consumed-day exploration, with the pilot versions retained and the full-run rule frozen. No claim of fraud or intentional outcome selection is made. This process limitation alone does not require a rerun on these same days.

### 7. P3 — NONBLOCKING: narrow the remaining take wording to its measured units

**Evidence:** `take_logic_v2.py:237-285,289-313`; inherited D1000 state construction; receipt `:33-39,120-123`.

- The lower-tail numbers reproduce. Large inventory is associated with roughly half as many **seconds containing same-direction takes**, not “almost never adds,” and not necessarily half the trade count or share volume.
- D1000 moves the entire state back one second: inventory, BTC, time-left and both ends of the recent-fill window. Its weaker burst result does not isolate the most recent second's fills or identify a reaction latency. State only that the association is sensitive to a one-second state lag.
- 2,765 is **extra raw records beyond the first per occupied second**, not all records sharing a second. There are 904 multi-record seconds containing 3,669 records. Outcome deduplication is correct.
- The lower-tail statistics were available from declared ratios, but the frozen decision rule tests increases. They remain descriptive associations under a fixed-path shift null, not newly established causal maker rules.

## What is closed and what survives

The earlier review's material TAKE issue is **closed**. `take_logic_v2.py:223-264` keeps raw ambiguous takes and recomputes opposing/same direction at each anchor. `:267-285` shifts the raw side schedule and reclassifies at the destination inventory. `:198-210` groups reduction subtypes before agreement; `:169-171,317-324` implements the clean-ledger sensitivity. The planted random-maker control reproduces the v1 spurious 1.92 ratio versus 1.01 with the corrected outcome; the positive policy control remains detectable.

The declared size-trigger decision was followed: pooled opposing size RR **0.90555**, upper-tail fraction **0.67465**, so the proposed increasing relationship is withdrawn. The lower-tail same-direction size ratio **0.51354**, near-END opposing **0.44478** and same-direction **0.03080** reproduce with lower-tail fraction **1/501**. These are useful constraints on candidate maker logic, not complete explanations of why they act.

Prior findings 2–5 are substantially addressed: venue/proxy clocks are named; full/clean panels are separate; base maker/take populations no longer sum overlapping attributes; 46 anchor-direction disagreements and three overshoots are disclosed; same-lot carry and literal zero taking near END are withdrawn. Finding 7 above finishes the remaining wording without delaying posting analysis. The previous pull-window association remains timing-sensitive and cancellation-compatible, not an identified private pull.

## Verification and limits

- Pinned suites **52/52 TAKE and 88/88 QUEUE** pass. Synthetic implementation checks do not establish behavioral identification; the new counterexamples demonstrate the missing coverage.
- **36 saved output/source hash checks** pass, including the panel/events and real queue rows used here. No large upstream tape or full lifecycle rebuild was run.
- Independently reconstructed six O1/O2 event vectors and **24 size MH ratios** across scopes, clocks and full/clean views; all agree. Independently regenerated the 500 real raw-schedule shifts for four D0/full/pooled size/end comparisons; ratios and both tails agree. See [evidence.json](evidence.json), [null_evidence.json](null_evidence.json), [check_null.py](check_null.py).
- Independently reproduced both queue counterexamples, the 407 same-hash later rows and the common-cohort economic values. These checks identify label/inference failures, not the true unobserved maker history.
- One review-fixture run failed after both suites passed: its synthetic lifecycle table contained a duplicated order hash. Only the review fixture was deduplicated; the first log is retained, and the passing suites were not repeated. No development source/output changed.
- All execution checks used the codex slot through `run_light.sh`; no model fitting, new-day measurement, tape experiment or strategy replay. Checks consumed about **9 CPU seconds** plus small bookkeeping. Charge **0.010 core-hours once**, within the requested 0.015; do not add the codex-tag logs again. The pre-review ledger is **1.8363 task h / 5.1470 line h**.

## ONE concrete next priority — posting opportunities linked to complete maker chains

**One frozen POSTING-OPPORTUNITY-AND-CHAIN packet on these same 19 markets**, bundling the corrections above. Its purpose is to learn which observable conditions precede quote creation/improvement and connect those opportunities to the makers' fill, possible withdrawal/repost, take and turnover paths. It is not another isolated take study or a queue-label cleanup gate.

Preparation contract:

1. **Neutral market-clock population:** every market-second × side over the full 300 s, including seconds with no fill and no observed creation. Reuse the existing book/BTC context builders and clocks. Outcomes are visible cell creation in the next second, separately improve / re-establish / deeper placement, with exact cell-event keys and counts beside the binary event indicator. Keep multiple types as overlapping outcomes, not extra seconds. Missing/stale book and creation intervals straddling an anchor are statuses. This is an observed market opportunity, not an identified maker decision timestamp. Adding at an occupied price is not a fresh creation; do not call its absence “no posting.”
2. **State precedes the opportunity:** freeze a small comparison set before reading outputs: spread, displayed depth at fixed offsets from the current touch, existing BTC-direction/flow measures, and time remaining. Use only available pre-anchor observations; never choose the feature's price level from the subsequent posting outcome. If a fair-value proxy already exists, name its exact frozen definition; do not fit one here or substitute the future price. Match comparisons on market, side, time-left and opportunity availability; keep day readings and all support counts. If reporting null tails, declare at least 500 matched draws and controls that remove a mechanically defined price-improvement effect.
3. **Maker identity is a separate layer:** all anonymous creations remain in the denominator. For top-three posting hypotheses, preserve the lifecycle hard/constrained/scenario grades and alternative histories. Unknown owner or unobserved posting is not WAIT. The eventual first filler is not automatically the creator. Only use a maker's reconstructed inventory at a creation when that identity/time link is justified; otherwise show candidate-specific context and bounds. Keep estimates useful, but explicitly conditional.
4. **Publish six complete chains**, using the existing predetermined cases (not new winners selected by economics): pre-state → post interval/price/size bounds → compatible queue histories and repeated same-hash fills → pull/exhaustion/repost alternatives → opposing/same-direction takes → completed turnover and residual value. Carry the longer 10/60 s and END views with valid denominators; no first-fill-only decision rule. Show where histories agree and exactly where a different hidden action changes the explanation. A lower bound from filled size is not a known submitted size.
5. **One combined acceptance:** include the cancellation/replenishment counterexample, a creation-window straddle, same-hash partial fills, no-fill/no-creation rows and a later-frame perturbation control. Every aggregate must reconcile to saved keyed rows. Publish supported associations, contradicted hypotheses and unidentified links separately. If maker ownership remains thin, finish with the anonymous creation results and bounded chains; do not start another tuned variant to manufacture identification.

Budget **<=0.040 core-hours all-in**, including <=0.010 for the next combined review; use a pilot/status projection before the full packet. No new days, model family, policy test, untouched validation or live use. This is within the USER's existing action-chain task and standing approval for short CPU top-ups (`PENDING.md:54`); **no additional USER decision is needed**. Record the task-cap top-up to **1.95 h** before launch if the standing 0.05 reserve is retained. At this ledger, current review + whole next packet totals **1.8863 task / 5.1970 line h**, or **1.9363 / 5.2470** including that reserve, within 1.95 / 5.25. Reconcile concurrent charges first; do not double-count this review.

**Largest evidenced limitation:** the public record does not uniquely link a book change to a particular maker's posting, cancellation or first match. This round overstates those links. The next measurement should extract actionable conditional logic from the observable opportunities and whole chains while keeping the unresolved ownership/timing alternatives visible.
