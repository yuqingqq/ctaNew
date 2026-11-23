# TOUCH-TRANSITION-AND-RESPONSE v1 — combined review

2026-11-23 UTC. **Full pinned target: f81b5b0d61ca14eb6039231e9daaeaa954b647ee**.
Range: `01a72b4e2..f81b5b0d61ca14eb6039231e9daaeaa954b647ee`.
Reviewed in detached checkout `/tmp/pf-touch-review-f81b5b0d`; no development implementation or frozen receipt changed. The previously authorized reviewers inspected separate components statically; bounded computation ran only in the root's codex slot.

**Decision: accept the descriptive census, episode accounting, restricted hazard calculations and chain regrades. Do not accept the headline as an identified top-maker repricing/reaction mechanism.** No material defect requiring a rerun of the native-frame measurement was found. Bundle the interpretation corrections and bounded continuation guards below with the next decision-chain measurement, not a cleanup-only round.

Population/as-of: 19 BOOK markets, Oct 24–27, four consumed days. Touch receipt completed 2026-11-23 21:21:32Z; chains completed 21:21:35Z. There are 3,796 retreat episodes, 999 same-side-first responses and only 12 top-three candidate response links. Point estimates only. No new data, untouched validation, model fit, strategy replay or profitability test.

## Consolidated findings

Paths below are relative to `predictfun/reviews/maker_trace_20261122/`. BLOCKING applies to the specified inference, not acceptance of all saved tables.

### 1. P2 — BLOCKING for coordinated-maker/majority claims: the headline changes both the unit and the identification grade

**Evidence:** `tools/touch_transition.py:386-422,641-704`; `v2/report_logic_chains_v5.md:26-34,104-105`.

The classifier identifies **net changes in the aggregate best bid and ask between adjacent snapshots**. It cannot identify simultaneous commands, a common maker, withdrawal versus depletion, or replacement of one maker's whole quote. Different hidden sequences can produce the same pair of endpoint books.

The 57% number reproduces: **6,723 / 11,804 improvement creations** occur in two-side-change intervals. It is not the share of all quote changes. The native census has **4,897 two-side pairs versus 8,298 one-side changed-touch pairs**. A frame can also contain several improvement creations. Thus “quotes move mostly as a whole” and “the whole quote shifts at once” overstate this evidence.

**Disposition:** retain the precise creation denominator and say “both aggregate best prices changed within the same observed interval.” For other-side-first episodes say “the next detected change was on the other side”; the group includes both improvements and retreats, not exclusively directional shifts. The maker layer remains conditional, with 910 / 999 responses owner-unknown and all 89 links scenario-graded. This correction does not require recomputation.

### 2. P2 — BLOCKING for maker reaction-speed claims: 54 ms measures frame endpoints, not a private action delay

**Evidence:** `touch_transition.py:432-521,798-810,1234-1256`; report `:43,53-60,105`.

The duration is response-frame time minus retreat-frame time. **710 / 999 responses are in the next frame.** Both the retreat and response occurred within observation intervals; the 54 ms median therefore does not identify a maker's reaction latency or an immediate response to a particular command.

Independent recount gives median endpoint difference **54 ms**, median lower elapsed bound **0 ms**, and median upper bound **104 ms** when uncertainty in the retreat's preceding interval is included. These are summaries of per-episode bounds, not an estimated latency distribution. The saved `exposure_hi_ms` is an endpoint duration measured from the observed retreat frame; it is not that physical-event upper bound.

Episodes terminate at competing transitions. The 1,922 other-side and 681 two-side endings are observations after a retreat, not established reactions to it. The arithmetic is correct under the frozen observation-clock definition.

**Disposition:** retain the endpoint statistic with its convention, and replace “answered within milliseconds” with “a subsequent touch change is usually detected within a few book frames.” Use both interval bounds in the next packet; do not fit a reaction-latency model from endpoint differences.

### 3. P2 — NONBLOCKING for the accepted descriptive result: the support/null design cannot establish a state-dependent maker rule

**Evidence:** `touch_transition.py:850-1001`, especially `:889-924,960-967`; contract §2; report `:58-70,106`.

The MH person-time formula and the within-stratum label shuffles implement the frozen design; all **10 D0/LAG ratios** independently reproduce. Each shuffle recalculates event counts and exposure time. The 30-exposed-second plus 30-reference-second per-market floor is also applied correctly.

Total exposure is **292.265 seconds**, with **7.189–27.483 seconds per market**, not the report's 3–14. Every market has less than 30 seconds even before splitting groups, so the support failure is structurally unavoidable for this dataset. It is a design limitation, not evidence that BTC/depth conditions contain no information. Keep every frozen `insufficient_support` verdict unchanged.

Furthermore, independently shuffling episode labels breaks the clustering of BTC/depth states; time-left is chronologically determined. The tail fractions are conditional reference diagnostics, not calibrated significance for a temporally dependent market process. The report already calls the ratios descriptive, which prevents a result-validity blocker. Lowering the floor until a displayed p-value passes would not repair this limitation.

**Disposition:** answer the next decision question with competing first-transition probabilities and explicitly counted informative episodes, as specified below. No significance or maker-rule promotion from this round's p-values. Correct the exposure range and retain the original failed gate.

### 4. P3 — NONBLOCKING on this saved population: continuation and adapter boundaries need small guards before wider reuse

**Evidence and verified disposition:**

- **END takes:** the header promises takes “to the end” (`touch_transition.py:52-55`), but `:1141-1151` uses `x_lo_hard_ms > creation_hi` without an END upper bound. Injecting one synthetic post-END record into the actual function changes the first candidate's count from **40 to 41**. However, independent inspection of **all 12 evaluated top-three candidate orders** finds **zero selected takes after or straddling END**. Current counts remain usable under this verified workaround. Bind the upper endpoint and retain unresolved boundary intervals as statuses before extending the population.
- **Value cohort:** `:1067-1081,367-380` supplies all selected maker legs to the horizon views without the clean overall-status/non-self filter used by `chains_v3.py:1114-1120,1144` and the economic accounting. All **160 candidate maker-leg rows** in this packet are status-ok and non-self; no current value is affected. Carry the explicit cohort guard when adding the requested economic continuation, not a full economic rebuild.
- **Response-context join:** `chains_v4.py:505-513` matches side and frame interval without checking the creation price against `resp_tick`. A simultaneous deep creation could inherit restoring-quote context. Here the only attached response, Chain 3, is **49c at both the cell and response best**, and is already context-only outside its conflicted post window. Add the price condition for “this cell restored the touch,” or label other same-frame cells merely contextual. No present chain grade changes.
- **Economic completeness:** `touch_transition.py:1079-1080` reports matched and residual **shares**, not the completed-close profit decomposition requested by contract §3. The report makes no new profitability claim, so it is incomplete explanation rather than false accounting. Complete that narrow saved-accounting attachment in the next packet if the exact leg keys are available; do not substitute markouts or summed book-level profits.

These items are one bounded integration bundle. None warrants restarting the full census or delaying the next authorized measurement.

## What is closed and what the saved evidence supports

All four findings from review `01a72b4e2` are closed for their stated scope:

1. The native-frame population recovers the omitted activity. **11,804 event-status improvements** reconcile, including **8,821 anchor-excluded** creations. Class counts reproduce: 6,723 two-side, 2,678 bid-only, 2,354 ask-only, 49 unsupported. The excluded split is 5,362 / 1,824 / 1,601 / 34. Creation intervals with other posting statuses are retained, not silently folded into this event-status denominator.
2. Recent reappearance is correctly relabelled print-compatible clearing; trade-caused clearing and maker-belief interpretations are withdrawn in-band.
3. `chains_v4.py:271-322` propagates the post-window grade and outside-window alternative. Chain 3 is TIED; no false ONE-HISTORY pre-state remains.
4. Uncomputed netting is `not_evaluated`, and gross cents/share labels are explicit (`:586-603`). Existing clean recorded-fill and episode take accounting is reused, not replaced by the new continuation table.

The classifier's gap/unknown/crossed statuses and joint net-change rule follow the contract. Outside-window pairs are separated. The native count is 117,888 including 10,095 outside; inside pairs total 107,793. Reconciliation uses cell/frame/time identity. Episode starts are one-side retreats; the first non-unchanged transition ends the episode, including further retreats. Independent saved-row checks found **zero overlapping episodes**, exact exposure arithmetic and correct pre-frame clocks.

The 3,796 endings reproduce: 999 response, 1,922 other side, 681 two side, 184 further retreat, 7 horizon, **3 empty-side censors** (the receipt's broad “gap” row). Response mix is 753 restoring, 139 partial and 107 overshoot. Unknown ownership stays in the denominator. Top-three 12-link counts are neither 12 identified posts nor 12 independent profitability cases.

Depth/spread/BTC context is from f−1; LAG uses f−2. The declared fair-gap feature combines that earlier BTC reference with the **post-retreat best**. It is known at the observed episode start, but should not be called a wholly pre-retreat feature. No decision-after-outcome input was found. Native frame timing remains a venue reconstruction, not private feed receipt evidence.

The integrator adaptation was included in the freeze before the full run. Per-creation reconciliation rows are de-duplicated to frame pairs before chain matching. No undisclosed outcome-driven threshold change was found. The preserved support failure is preferable to rewriting its gate after the pilot.

## Bounded verification and limits

- Pinned selftests: **56/56 touch, 49/49 chains**, all pass. They are implementation tests, not model skill or profitability evidence.
- **39 receipt-named output/source hashes** match. The used take parquet and **1.1 GB saved economic parquet** also match the pinned input hashes; the latter was streamed for hashing and only candidate columns/keys were read. No raw tapes loaded.
- [check_saved.py](check_saved.py), [evidence.json](evidence.json) and [check_saved.log](check_saved.log): independent episode/exposure/population/ratio recounts, current continuation-cohort checks and the actual-function END falsifier. No full measurement or 500-permutation rerun; the permutation implementation was inspected and covered by the pinned synthetic suite.
- Review-instrument v1 is retained as `evidence_v1.json` / `check_saved_v1.log`: it counted `is_improve` across all reconciliation statuses. The final probe applies the receipt's event-status population and asserts 11,804. This was a review denominator correction, not a development defect.
- Measured checks before staging: about **8.45 CPU seconds**, peak **433 MiB**. Charge this whole review conservatively as **0.010 core-hours once**, within the requested 0.015; do not separately add its codex logs to `mt_` totals.

## ONE next priority — COMPETING-TRANSITION-AND-CONTINUATION v1

**Question:** after a visible one-side retreat, which already-declared market conditions precede restoration versus continued repricing, and what happens economically along the candidate order chains? This advances the USER's post/pull/reprice/take logic goal; it does not manufacture private-action identification.

Freeze one saved-evidence packet on the **same 3,796 episodes / 19 markets**, with the following decisions settled before calculation:

1. **Outcome unit:** the next detected transition, mutually exclusive: same-side restoring / partial / overshoot; other-side improvement / retreat; two-side both-up / both-down / narrowing / widening; further same-side retreat; horizon without change; unknown/gap/window/book-end censor. Keep all rows. Competing moves are outcomes, not discarded nonresponses. If an existing saved class cannot distinguish a subcase, retain an explicit unknown instead of loading a new tape.
2. **State and contrasts:** retain all five frozen contrasts, their bins and D0/LAG definitions. No selection of the two smallest permutation fractions. Use market × side × retreat-size × time-left strata, omitting time-left only for its own contrast. Report state-conditional outcome fractions, matched same-side-response **risk differences**, day signs, side signs and contribution concentration. On common-support strata use fixed harmonic weights `n1*n0/(n1+n0)`; also show raw all-population counts. Keep unknown/censored mass visible with response-probability lower/upper bounds rather than treating missing outcomes as true zeros.
3. **Count-based support, frozen now:** descriptive comparison requires at least **30 episodes in each group**, **10 same-side responses and 10 competing nonresponse transitions in each group**, counted within mixed strata; and at least **5 markets with >=5 episodes in each group**. Show all counts, including excluded/unsupported counts. This is a reporting-adequacy floor, not a power guarantee or significance test. If it fails, report thin and stop; no further floor/feature tuning. There are only four related consumed days: no confidence interval or validated rule. Do not reuse independent-label permutation p-values as confirmation; this packet needs no new significance verdict.
4. **Timing:** report interval bounds using both retreat and response intervals, next-frame fraction and observer-clock lag separately. For a response, lower bound `max(0, response_lo - retreat_hi)`; upper `response_hi - retreat_lo`. Do not interpret medians of bounds as identified median latency. Compare D0/LAG direction descriptively; disagreements stay timing-sensitive.
5. **Finish the existing candidate continuation, without broadening it:** carry owner/grade and unresolved paths into the same six chains and 89 candidate links. Attach existing exact-leg FIFO completed-close net dollars/shares and residual values, with closes inside 10 s / 60 s / END kept apart from later closes and resolution. Preserve market-END bounds, fees, clean cohorts and take direction relative to known inventory; a subsequent take is not automatically an exit of that candidate. Deduplicate fill and close keys across repeated links; do not sum account-level P&L per candidate. If an exact economic link is unavailable, label it unavailable and keep the neutral transition measurement. No new large reconstruction is justified for 12 top-three links.
6. **Acceptance/stop:** reconstruct the fixed episode totals, mutual exclusion, class-direction identities, common-support denominators, censor bounds and exact leg uniqueness. Include known-bad cases for swapping outcome labels, dropping competitors, crossing END and duplicating a close. One combined receipt/review bundles all findings above. No fitted rule, policy replay, threshold change or promotion follows automatically. A thin or inconsistent answer ends this measurement line rather than triggering another tuned variant.

**Resources and authorization:** pre-review ledger is **1.8847 task / 5.1954 line**. With this review's 0.010 charge and a next packet capped at **0.030 core-hours ALL-IN, including <=0.010 for its review**, projected totals are **1.9247 / 5.2354**; with the standing 0.05 reserve, **1.9747 / 5.2854**, within **2.00 / 5.30**. Reconcile concurrent work before launch. No further USER decision is needed for this bounded saved-data packet under the standing authorization. New days, untouched validation, live use or a substantive strategy experiment remain separate USER decisions.

**Largest evidenced limitation:** aggregate book transitions are observable, but their maker-command chain is usually not: 91% of responses lack even a candidate owner, and the remaining links are scenario-conditional. The next packet must make the condition → observable transition → graded economic continuation chain clearer without claiming to recover the missing private decisions.
