# LOGIC-CHAIN CORRECTION v1 + TAKE-LOGIC v1 — combined review

Date: 2026-11-23 UTC. Full pinned target: **cccac3665b967fc4858b2335add811779918b87a**.
Range: `6bb458b30..cccac3665b967fc4858b2335add811779918b87a`.
Source checkout: `/tmp/pf-takelogic-review-cccac366`. Review only; no implementation, frozen receipt, or other-session changes edited. Three previously authorized static reviewers covered the correction, statistics, and timing; only the root ran checks.

## Result

**Accept the corrected historical evidence and the new tables as descriptive measurements under their stated clocks. Do not yet accept “inventory size is the trigger.”** The principal remaining problem is the classification of the outcome, not an error in the MH formula. Small positions have many more unclassifiable takes; excluding those from the reduction outcome can create or amplify the apparent size relationship.

Population/as-of: the pinned receipt completed 2026-11-23 12:51:58Z; ex-ante top three, 19 BOOK markets, Oct 24–27 (four consumed days), 17,100 maker-seconds. Of 4,147 take records, 4,124 map into this panel and 23 precede it. No untouched validation, confidence interval, model fit, strategy test, or demonstrated profitability.

## Consolidated findings

BLOCKING refers to the named behavioral conclusion or using it as an established strategy rule. It does not invalidate all saved measurements or require a full rebuild. Bundle these items into the single next measurement below.

### 1. P1 — BLOCKING for the inventory-trigger claim: classification uncertainty depends strongly on inventory size

**Source:** `tools/chain_take.py:248-273,481-496`; `tools/take_logic.py:125-127,406-415,458-464,481-495,620-635`; `v2/report_logic_chains_v2.md:84,123-125`. Tool/receipt paths in this report are relative to `predictfun/reviews/maker_trace_20261122/`.

The prior builder marks a take `timing_sensitive` whenever its detailed class differs across recorded, certain, and simultaneous inventory variants. The new builder uses that fixed class as its outcome. Ambiguous takes are counted in the panel, but never contribute to the reduce or add numerator. D1000 and pending sensitivities change the state, **not this outcome classification**.

Independent saved-table recount:

| Pooled inventory | Classified reduce event-seconds | Timing-sensitive event-seconds |
|---|---:|---:|
| (0, 50] shares | 109 | 127 |
| >200 shares | 325 | 19 |

Rank 1 has the same imbalance: 89/102 at small positions versus 263/16 at large positions. These event-type counts may overlap within a second and must not be summed as disjoint time.

The MH size ratio reproduces: **1.4794 pooled, 1.2710 rank 1** for the classified-reduce outcome. As a **post-hoc diagnostic**, using the same saved joint cells and size strata gives **0.7550 pooled / 0.6507 rank 1 for any take**, and 0.0602 / 0.0582 for timing-sensitive takes. Thus the positive result is specific to the classified-reduction outcome, not evidence that large positions cause more taking overall. These extra ratios have no new null test or promotion claim.

Saved per-take classes sharpen the concern: of 838 ambiguous records overall, **689 have at least one reducing classification**; **48 are reducing in every available variant but differ only in the finer reduction subtype**. None enters the new reduce outcome. The random-maker selftest supplies its event types directly (`take_logic_selftest.py:139-182`); it does not exercise this upstream classification filter.

This does **not** prove the whole size association is artificial. It means the declared state sensitivities do not establish robustness to the material label uncertainty. Correct the estimand now; resolve this uncertainty before calling size a trigger.

### 2. P2 — BLOCKING for decision-time/causal claims; tables can be retained

**Source:** `tools/chain_context.py:19-21,33-38`; `tools/take_logic.py:268-290,303,425-455,696-723`; report `:61,84-89,123-133`.

The arithmetic is correct: `(x_lo_hard-ws-1)//1000` assigns `t < x_lo_hard <= t+1000`, and calling the strict-before inventory helper at `t+1` implements `hi <= t`. But **x_lo_hard is a lower possible match bound**, not an observed decision or submission time. The context library explicitly calls the inventory reconstructed venue inventory, not client-known inventory; public venue clocks do not identify the maker's receipt clocks. Pending variants also use retrospective reconstruction evidence.

Use “fixed market-second, pre-earliest-match proxy” rather than implying an identified decision-information set. The causality selftest correctly rejects later timestamped information in the chosen reconstruction; it cannot establish what a maker privately knew.

The following conclusions exceed the measured predicates:

- “Trigger = inventory size” should be an association conditional on the specified label and reference path, pending finding 1.
- “Fast reaction within ~1s” is not measured. Moving state back one second while leaving event labels/times fixed demonstrates alignment sensitivity, not response latency.
- “BTC direction is not the trigger” is not established by `not_supported` or `insufficient_support`. Say no BTC-direction association passed this declared screen. Rank 1's take-profit comparison has only four supported markets.
- Circular shifts preserve each pair's event pattern, but leave its inventory path fixed after moving the trades that helped generate it. This is the declared descriptive timing null, not a causal counterfactual or an independent-day probability. The 16/18 consistency count is the declared **crude**, not stratum-adjusted, within-market comparison.

### 3. P2 — NONBLOCKING for retaining tables; carry outcome/state disagreement and ledger qualification into the next measurement

**Source:** `tools/take_logic.py:125-127,290,620-678,763-792,871-875`; `tools/chain_take.py:248-260`; report `:46,55,69-70,122`.

There are **46 purported reducing takes in 13 event-seconds whose side equals the inventory sign at the new proxy anchor**. They add in that direction at the anchor, although their earlier saved classification says reduction. Zero reduce takes have q=0 at the anchor. The code records these distinctions correctly; the prose “each reducing take unwinds” does not carry them. Three `overshoot` records are deliberately mapped into reduce despite increasing absolute exposure under the old definition. This small mapping issue alone cannot explain the headline.

Also, **939 maker-seconds, 101 any-take event-seconds / 248 takes**, including 46 reduce event-seconds / 107 reduce takes, have inventory conditional on excluded earlier legs. The flag is counted but does not filter/stratify the new MH result. The previous correction's with/without table is not a clean-ledger sensitivity of this new ratio.

Keep the current outcome precisely named: “subsequent take classified reducing under the saved ledger variants.” Bundle a clean-ledger and classification sensitivity into finding 1; no separate fix-only cycle.

### 4. P2 — NONBLOCKING: the corrected execution census double-counts attributes

**Source:** `tools/chain_correct_post.py:110-112,568,954-957`; `v2/chains/correction_v1/census_graded.csv`; report `:36`.

The claimed **19,658 observed executions** sum both base events and overlapping take attributes. Saved rows contain **9,026 maker-fill rows** and **4,147 take records**; another 6,485 are attribute incidences: mint 3,709, multi-record 2,722, merge 16, self-trade 38. Attributes are not additional executions, and maker-leg rows and aggregated take records are different units.

Publish the two base populations separately; preserve attribute counts as overlapping attributes. Correct the computed census summary as well as prose when next touched. No tape reread is needed.

### 5. P3 — NONBLOCKING: keep end-profile and repetition wording within its units

**Source:** `tools/take_logic.py:763-805`; report `:96-118,128,133`.

The declared marginal profile does peak at 30–60 seconds left; it is valid to report it descriptively, not as an independently tested optimum. “Adds stop” is too absolute: the last-10-second row is 5 event-seconds per 1,000, not zero. “Markets with >=5” in the pattern table counts **event-seconds**, not individual takes; state the unit.

END, END−30s and END−60s inventory magnitudes support endpoint exposure and flattening counts. They do not prove the same lots were “mostly carried into resolution”: intervening turnover/replacement is possible. Say some markets end with sizeable recorded positions; no new lot-attribution experiment is required to correct this sentence.

## Prior findings and the frozen builder decisions

- Prior 1 (posting bounds/grades): **closed for the reported subset**. `chain_correct_post.py:317-337` requires a clean lifecycle-constrained class, not just a finite scenario union. The receipt withdraws the 70% proof, keeps 940/4,149 clean short-bound cases, and regrades detailed paths. Bounds remain conditional on the lifecycle premises; unknown cases are not evidence of long rest.
- Prior 2 (TAKE sign): **closed**. The old statistic is explicitly against the take side; no pooled sign flip is used to manufacture an inventory-aligned result.
- Prior 3–4 (pull references/matching): **closed descriptively**. Both reference families, actual fallbacks, support and day/timing sensitivity are published. BTC terminal-window enrichment survives the fill-anchored reference at the declared D=250ms on all four consumed days; it is still not an identified cancellation rule. Stable inventory enrichment and exhaustion as a universally weaker reference are withdrawn.
- Prior 5 (actions versus hypotheses): **substantially closed**. No identified reposts/private KEEP; tied links and the six predetermined paths have evidence grades. Finding 4 above is a new census-count issue.
- Prior 6 (excluded earlier legs): **closed for the old tables**, explicitly conditional; its application to the new hazard remains finding 3.
- Prior 7: new outputs are hash-bound; frozen old harness portability and masked-fill/replenishment controls are explicitly deferred in the correction contract. Those deferred items do not invalidate the current saved-data correction and need not restart this round.

The conservative pre-run choices are acceptable **for these estimands**: add rank to every pooled stratum; count NaN shift replicates on both tails; require informative events/support; use event-seconds for repetition; keep other statuses. D1000 moves all state reads consistently; it is a sensitivity, not known latency. Including overshoot is faithful to the new contract but needs the label qualification above. None of these choices repairs fixed event-class selection or identifies private decision times.

## Verification and limits

- Pinned selftests: **94/94 correction POST, 90/90 correction PULL/TAKE, 70/70 TAKE-LOGIC**, all pass after commit. Synthetic implementation evidence only.
- **40 receipt-named output/source hash checks**, all agree. The saved take input also matches the new receipt hash. Large upstream tapes/parquets were not rescanned; this review does not repeat the accepted lifecycle build.
- Independent saved joint-cell aggregation reproduces **16 D0 MH ratios** (size/end/5s with/against, reduce/add, rank1/pooled) to tolerance. All **120** variant/scope/type/ratio market-consistency recounts agree. Tail fractions equal the declared `(count+1)/501` convention represented by c_hi. Code inspection and synthetic tests cover rotation/count preservation; the 500 real shifts were not rerun.
- No full context, lifecycle, measurement build, replay, fit, or new-day run. [check_saved.py](check_saved.py), [evidence.json](evidence.json), [check_labels.py](check_labels.py), [label_evidence.json](label_evidence.json) contain the checks and row-derived diagnostics. One review-helper correction made “all available classes” ignore absent simultaneous classifications; its first log is retained, with no development output changed.
- Actual checks used about **8 CPU seconds** including synthetic children, plus negligible bookkeeping. Charge this review conservatively at **0.010 core-hours**, within the requested **0.015**. Before this charge the live ledger is **1.8064 task h / 5.1171 line h**. This review's codex-prefixed runs are outside the mt_ ledger: add the charge once, not its logs again.

## ONE concrete next priority

**One saved-data, ambiguity-aware inventory/take measurement**, <= **0.025 core-hours all-in**, including <=0.010 for its combined review. Finish the interpretation/count corrections above in the same round. This is completion of the authorized inventory-logic question, not a new strategy or new model candidate.

Freeze before computing the revised readings. Reuse the same 19 markets, timestamps, inputs, inventory/BTC builders, bins, 500 shifts, and market-support rules; no new days, window search, threshold tuning, or replay. A single bounded reconstruction of this small panel is acceptable if needed; save its event keys and state sufficient statistics so another complete reconstruction is unnecessary.

The decisive measurement should compare the **same fixed states** with outcomes that do not require agreement of the old fine-grained inventory classes:

1. Keep every raw take's time, side and size, including ambiguous records. Measure **opposing the anchor's position** versus same-direction taking, with q=0/unknown and excluded-ledger states explicit. This is a direction relative to the reconstructed anchor, not proof that the action reduced the maker's actual private inventory. Keep any-take incidence beside it.
2. Separately show actual-|position|-reduction classes across the saved feasible inventory variants, grouping fine reduction subtypes before agreement. Report disagreement/coverage instead of silently dropping uncertain outcomes. Do not sum overlapping event types; deduplicate the chosen outcome within each second.
3. For the direction-based null, circularly shift the **raw side-specific event schedule**, then recompute direction relative to the destination state's inventory. Do not shift an already position-labelled “reduce” vector and call that side-matched evidence. Preserve all per-pair raw event counts and report collisions/aggregation consistently. Retain the old calculation as the benchmark, not as the new null.
4. Publish the same ratios on the full retained-ledger panel and the clean-ledger subset. Require support counts and a random-side/timing control passed through the actual upstream classification path; include a fixture where small positions create more fine-class disagreement without an inventory-size policy. One combined acceptance review.

Declare the decision before results: if the size relationship fails once classification uncertainty is addressed, **stop treating it as a maker rule**; do not tune another bin/variant to recover it. If it survives with support, it becomes a qualified candidate association for later strategy work, still not a causal or profitable rule. Thin or timing-inconsistent results retain an unidentified status.

**No new USER decision is needed within this existing question and allowance.** Charging this review and the entire proposed follow-up gives approximately **1.8414/1.90 task h, 5.1521/5.25 line h** (and 1.8914 including the ledger's standing 0.05 reserve). Reconcile concurrent runs first. Extra data, untouched validation, a policy experiment, or a larger budget remains outside this priority.

**Largest evidenced limitation:** uncertainty is not uniform across the state space: the same small inventories being compared have the most uncertain trade classifications. The present measurements cannot separate a real inventory-driven decision rule from that selection effect, and public match bounds still do not identify private decision times.
