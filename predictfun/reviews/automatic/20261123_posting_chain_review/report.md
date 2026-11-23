# POSTING-OPPORTUNITY-AND-CHAIN v1 — combined review

Date: 2026-11-23 UTC. **Full pinned target: ed7d1ebaf34aa79cadec467ef41a894ea395423e**.
Range: `3f640ad9b..ed7d1ebaf34aa79cadec467ef41a894ea395423e`.
Source: detached checkout `/tmp/pf-posting-review-ed7d1eba`. Review only. Previously authorized reviewers inspected the three components statically; the root alone ran bounded checks. No implementation or frozen result changed.

**Result:** accept the neutral panel, its restricted-population descriptive statistics, the queue regrades and recorded economic views. The packet is useful, but it has not explained most price-improvement activity or identified top-maker posting policy. Three scoped interpretation/grade issues below should be bundled with the next measurement; they do not require a full rebuild or a cleanup-only round.

Population/as-of: 19 BOOK markets, Oct 24–27, four consumed days; posting receipt completed 2026-11-23 18:37:49Z and chains at 18:37:52Z. There are 11,400 market-side-seconds, 19,991 wholly assigned creations, 1,159 straddling intervals and 245 outside the panel. The six predetermined chain examples are from Oct 24, not six independent validation cases. No untouched-day evidence, intervals, new model fit, strategy replay or profitability claim.

## Consolidated actionable findings

Paths below are relative to `predictfun/reviews/maker_trace_20261122/`. BLOCKING refers to the named inference, not all saved output. Preserve frozen receipts and correct them in-band in the next combined packet.

### 1. P2 — BLOCKING for the improvement-coverage conclusion: the restriction removes most observed improvement activity

**Source:** `tools/posting_opportunity.py:399-439,721,789-790`; `v2/report_logic_chains_v4.md:34-46`. The selftest even has improvements beginning from an anchor spread of one (`posting_opportunity_selftest.py:445-450`).

The frozen rule permits improvement analysis only if the spread is already at least two ticks at the second's anchor. The outcome, however, is a creation anywhere during the **following second**. The spread can widen and then receive a better quote during that interval. Both best prices can also change within a public frame, without an observable intermediate wide spread.

Independent saved-panel recount:

| Unit | All observed improvements | Excluded by anchor availability |
|---|---:|---:|
| Event-seconds | 4,484 | **3,664 (81.7%)** |
| Creation events | 11,804 | **8,821 (74.7%)** |

There are 1,116 eligible **rows**, containing 820 improvement event-seconds. Thus “mechanically possible in only 1,116 rows” is false as a statement about the full following-second opportunity. The reported ratios correctly implement a narrower frozen population: **seconds starting with room to improve**. Insufficient support there does not mean improvement itself is rare or that its decision logic has been adequately tested.

**Disposition:** retain those ratios with their restricted denominator and publish the excluded counts. The next measurement should follow observed touch transitions and quote responses within the second, including joint repricing, rather than tune the existing one-second cutoff or drop the excluded activity.

### 2. P2 — BLOCKING for trade-caused clearing/reposting claims: `re_establish` is a print-compatible reappearance

**Source:** `posting_opportunity.py:708-721`; inherited `queue_front.py:497-508,757-761,958-963`; contract §1; report `:32,37,44,137`.

The outcome flag is inherited from v1. `_prints` contributes a recorded trade's quantity to **every** frame interval overlapping its hard execution window. A disappearing cell is considered explained when compatible print quantity is at least the net drop; this is not an allocation of that trade to a unique interval, nor a conservation proof that it emptied the level. A cancellation-to-zero can satisfy the same check when a trade from another part of the possible lifetime overlaps its interval.

Consequently the measured class is **recent cell reappearance with print-compatible clearing**. It does not establish “emptied by matched fills,” a repost by the same maker, or the proposed causal chain “adverse BTC caused the clearing.” The table's fair-gap and BTC ratios can remain as descriptive associations for the declared flag. Removing the causal wording is sufficient; do not rerun the whole panel merely to force an execution explanation.

The same distinction applies to `p_fair`: it is the existing trailing BTC reference, not a measured maker belief or an established true economic value. Its use is causal in the code's timestamp convention; its name alone does not establish an economic mechanism.

### 3. P2 — BLOCKING for posting-condition agreement: the six-chain pre-state step drops the posting-window grade

**Source:** `tools/chains_v3.py:658-708`; `tools/chain_correct_post.py:17-28,88-96`; `v2/chains/chains_v3/chains.md:165-173`; report `:110-112`.

`link_pre_state` enumerates saved seconds between the numeric scenario bounds and derives ONE-HISTORY/TIED from their feature vectors. It does not propagate whether those bounds are conflicting or only scenario-based. **Chain 3 has a G1_conflict POST, yet PRE-STATE is ONE-HISTORY.** A single vector inside a conflicted window is not agreement across all possible posting histories. G4 histories also allow earlier posting masked by replenishment outside the enumerated interval.

The receipt additionally says every pre-state is TIED, contrary to Chain 3's actual saved label. This matters for the USER's question about which market conditions caused posting, but does not change the neutral panel or recorded fill accounting.

**Disposition:** retain the displayed vectors as conditional window descriptions, carry the original grade, and include an unresolved/outside-window alternative when the window cannot bound the actual post. “Features constant within this scenario window” is distinct from “the posting state is identified.” Correct the summary from those graded records. No tape reread is necessary.

### 4. P3 — NONBLOCKING: carry unknown queue alternatives and label gross units clearly

**Source:** `chains_v3.py:858-864,1078,1091`; `queue_front_v2.py:230-236,305-314,327-338`.

- A literal `alt_netting_possible=False` can mean the upstream row had no evaluable earliest record: the boolean initializes false and the unknown/untimed branch returns before its calculation. The chain builder treats that false as excluding masked netting. For unknown rows use an unknown/not-evaluated state, not absence of an alternative. The current queue labels are already TIED, so this does not overturn their headline.
- The chain prose prints **gross** cents/share after net dollars. Add “gross c/sh” to avoid confusing that markout with completed-close net value. The calculation itself is consistent. Short displayed identifiers are disclosed and full keys remain in the artifacts; no identifier rewrite is necessary for this review.

## Acceptance of the main implementation and prior corrections

**Posting population and timing.** The full 19 × 300 × 2 panel, unique keys, all five outcome/count families, interval containment and side signs reproduce. Every D0 row has a usable book. The raw event census and the 19,282 unknown / 706 scenario / three constrained links reconcile. Types overlap intentionally; counts are not extra seconds.

The instrument measures **visible creations whose whole frame interval fits inside a second**. Straddling intervals are retained as statuses, but their touched seconds are not uncertain outcomes in the ratios: absent another contained event, they are zero for this narrower outcome. For example, 280 touched seconds have no contained `any` event. This is a declared observability restriction, not a population of all true posting actions. Adding at an occupied price is also outside fresh creation. Preserve these limits when interpreting activity rates.

**Features.** `posting_opportunity.py:361-394` uses the pre-anchor book and settlement-known flow; the entire state moves back under D1000. The fair-value path reuses `pf_mt_decision.btc_feats:435-453`: current/prior BTC grid values, return since market open and trailing 60 s variance. No new fit or forward-volatility window was found. These are reconstructed venue-clock observations, not proof of private maker receipt times. Depth/spread bins are fixed; half-tick spread boundaries correctly avoid raw floating-point equality problems.

**Statistics.** All **70 MH rate ratios** and **1,330 market RR/support records** independently reproduce. Five representative real 500-shift comparisons reproduce both tails. Circular shifts preserve the market-side event schedule while keeping the state path fixed. This is a declared descriptive timing null, not a causal experiment or an independent-day confidence statement. Market consistency uses the within-market MH ratio, not the old crude ratio. The timing-sensitive label correctly means a verdict changes under whole-state D1000; it does not measure a reaction delay. Time-left contrasts appropriately omit time-left from their own matching strata. All 35 comparisons are disclosed; no new winner was chosen here.

**Queue and maker identity.** Prior findings 1–6 are closed for scenario-conditional use: no holder certification, masked-netting alternatives, whole-window settlement test, later-same-hash split, valid/all economics, graded endpoint occupancy, cell-event speed, unidentified front-win answer and adaptive-pilot disclosure. The real-output adapter change was frozen before the run. `posting_opportunity.py:299-322,591-663` hash-binds the input, joins cell/frame/creation times, checks signer and window containment, and carries grades. Containment proves compatibility, not ownership. Keeping 96.5% unknown is correct; it must not become a maker WAIT label. No further queue-engine repair is justified by the remaining identifiability limit alone.

**Six chains and economics.** Selection is unchanged. Same-hash fills, unknown submitted-size bounds, pull/exhaustion and successor alternatives, take-clock differences, and horizon-valid populations are retained. `chains_v3.py:1114-1167` uses full leg keys for FIFO close attribution and keeps closes inside the episode, closes later, carried-lot closes and residuals separate. No new material accounting issue was found by static inspection; this review did not rerun all FIFO accounting. Finding 3 limits the posting-state story. Prior take wording corrections are complete.

## Bounded verification

- Pinned selftests **54/54 posting, 45/45 queue v2, 56/56 chains** pass: 155 checks, implementation evidence only.
- **50 receipt-named output/source hashes** agree. The saved panel/events and queue/chain artifacts used by the review are bound to the pinned receipts. Large upstream raw inputs were not rescanned.
- [check_saved.py](check_saved.py), [evidence.json](evidence.json) and [check_saved.log](check_saved.log) contain independent population, outcome, ratio, support, null and coverage recounts. No full measurement, replay, new data or model fit ran.
- Review-only instrument history is retained: the first launcher used system Python without numpy and stopped at import. The first recount then exposed two review-helper mismatches—string versus integer day keys and exact equality on raw floating spreads. Correcting those to the saved key type and declared half-tick bins reproduced the outcomes/ratios. Passing development suites were reused, not rerun; no development data changed.
- Approximately **12 CPU seconds** of checks plus small bookkeeping, via the codex slot. Charge this review **0.010 core-hours once**, within the requested 0.015. The pre-review ledger is **1.8633 task / 5.1740 line h**; codex logs are outside the task's `mt_` sum, so do not add both logs and the charge.

## ONE next priority — native-frame posting/repricing response chains

**One frozen TOUCH-TRANSITION-AND-RESPONSE packet**, on the same 19 markets, bundling the findings above. The question is concrete: **after the touch retreats, clears or moves, what observable market state precedes replacement/improvement, which maker histories are compatible with it, and how does that quote's subsequent fill/turnover chain develop?** This addresses the majority of improvement activity currently excluded and directly serves the USER's action-chain/queue goal. Do not start another isolated take study or attempt to certify hidden commands from unchanged evidence.

Preparation and acceptance contract:

1. **Use the native adjacent book-frame intervals**, with both sides and all valid no-change/no-response exposure retained. Freeze a mutually exclusive transition census before measurement: unchanged touch; one-side retreat/spread opening; one-side improvement; simultaneous two-side repricing; gap/unknown. Cross it with anonymous cell creation/reappearance separately. Same-frame retreat and refill must remain jointly timed; never invent an intermediate event order. Reconcile all 11,804 improvement creations and the 8,821 excluded by the prior anchor mask to this census, including explicit unsupported statuses.
2. **Define one nonoverlapping response episode per observed spread-opening event**, continuing until the first restoring/improving quote, a further touch transition, gap, or the frozen one-second response horizon; publish the full termination census. Keep simultaneous repricing as a separate population, not a forced sequential response. Use the pre-transition frame and unchanged BTC builder for state, with no next-frame feature leakage. Record interval bounds and elapsed exposure; do not compare raw numbers of high-frequency snapshots as equal units of time.
3. **Keep the same small state hypotheses and bins:** spread/depth, signed BTC move and the existing fair-gap reference, with time-left and side separate. For the sequential response question, match on the available price gap and time-left, not only on market; include the no-response opportunities. For simultaneous repricing, report a separate descriptive state/transition table. Declare any 500-draw null and known-bad controls before outputs. No threshold search, new fair-value model or strategy candidate.
4. **Link responses to complete observable maker continuations**, not just first-fill markouts: price/size and posting-time bounds; candidate owners with grades; repeated same-hash fills; possible withdrawal/exhaustion and successor histories; subsequent recorded takes; existing 10/60 s and END views, completed turnover and residual accounting. Unknown owner stays in the denominator. Do not attribute anonymous creation to the eventual filler as a fact. Use all supported linked cases for tables; update the same six examples to illustrate agreement and alternatives rather than select new profitable stories. Report shared valid cohorts and attribution overlap; do not sum a whole-book close value once for every candidate creation.
5. **One combined gate:** the chain must survive a same-second spread-one → retreat → improvement fixture, simultaneous best-price changes, masked cancellation/replenishment, conflicted posting windows, later same-hash fills, missing book and later-frame perturbation. The report must separate anonymous market behavior, maker-compatible histories and economically favorable recorded continuations. If ownership stays thin, finish the measured transition answer with bounds; do not tune another reconstruction merely to raise coverage.

Budget **<=0.040 core-hours all-in**, including <=0.010 for the next review; pilot/status projection before the full run. Existing stored frames/BTC/ledgers only; no new days, untouched validation, live use or policy replay. **No additional USER decision is needed:** the USER's standing short-CPU top-up authorization covers this bounded continuation of the action-chain task. Before launching, record cap amendments to **2.00 task h / 5.30 line h** if needed to retain the standing reserve. Current review + whole next packet projects **1.9133 task / 5.2240 line h**, or **1.9633 / 5.2740** including 0.05 reserve, within those caps. Reconcile concurrent usage first.

**Largest evidenced limitation:** most improvement happens inside the coarse second that this study conditions on, while 96.5% of visible creations lack even a candidate maker owner. The useful next step is to resolve observable transition chains and their conditional continuations—not infer a private maker policy or profitable rule from anonymous creation-frequency ratios.
