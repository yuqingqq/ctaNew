# Review addendum: execution windows and remaining issues — 2026-11-18

User-directed follow-up to the review of
**d1de75049fdf5a2477e279ca108114b5aef09b27**, recorded at review commit
35c095cee0b6fc90f9dafce47e933a8d22f5294e. This document refines the next priority;
it does not change the pinned results, original receipt or implementation.

## Agreed direction: represent uncertainty instead of inventing an exact time

An exact match timestamp is **not required for every trade**. An evidence-backed
execution window can be sufficient when the action's economic conclusion is
stable throughout that window. A representative timestamp can drive one replay,
but must remain an assumption rather than replacing the recorded uncertainty.

Persist a ledger keyed by stable trade/event and leg identities, linked to the
neutral order/decision population. Each event should retain:

- Earliest/latest supported execution time, including open or unknown endpoints.
- Bound type: `evidence_bound`, `assumption_range`, `unknown` or `contradictory`.
- Optional representative execution time and the rule used to choose it.
- Raw source event/status and timestamps, local receipt time, clock domain and
  any clock-alignment uncertainty; never overwrite the source timestamp.
- Candidate identities when a link is ambiguous, evidence for each endpoint,
  relevant observed order lifecycle, and hashes of inputs/source/configuration.

Use independently observed lifecycle constraints for **actual identified
orders**. Do not clip a public trade into a hypothetical order's lifetime to
make the replay fill fit. Do not use current model predictions or P&L to choose
timestamps or interval widths. Empty intersections are contradictions to
investigate, not intervals to silently repair. A one-sided timing fact does not
justify inventing the missing endpoint.

Illustration only, with cancellation assumed effective at 137 ms:

| Execution window | Timing conclusion |
|---|---|
| 80–120 ms | The match precedes cancellation |
| 150–190 ms | Cancellation precedes the match |
| 90–180 ms | Ordering is ambiguous |

Cancellation may also have a window. On aligned clocks, `match_latest <
cancel_earliest` establishes that the cancellation is too late; `cancel_latest
< match_earliest` establishes the reverse ordering. Overlap remains ambiguous.
Touching endpoints require a supported venue tie-order rule; the simulator's
current equal-time convention is not independently established venue evidence.

For full economic evaluation, construct **jointly consistent event timelines**
within those constraints. Preserve the ordering/identity constraints connecting
legs of the same trade or matching operation; do not independently move each
order's copy of a shared event. Recompute queue consumption, fills, remaining
size, inventory and subsequent eligibility under each timeline. Keep observed
feature availability tied to its receipt/availability clock: moving a latent
execution time earlier does not make its information available earlier.

An earliest/latest/midpoint set or sampled timelines can provide sensitivity
results. These samples do not automatically establish exhaustive economic
bounds, and uniform sampling does not establish a true execution-time
distribution. Queue and lifecycle changes can make value non-monotonic in
timing. Claim robust decisions only to the extent the admissible scenario set
or bounding method supports the claim; otherwise say "stable in tested
scenarios." Retain ambiguous economic contributions in total accounting.

## Other material issues, in priority order

1. **Both execution and information arrival need treatment.** The replay uses
   placement 58 ms, cancellation 37 ms, pending landing 95 ms and own-fill
   notice delay 100 ms (`predictfun/research/pf_keep_pull.py:123`, `:138`).
   These are configured assumptions, not per-order observations. A fill-time
   window alone does not validate cancellation timing or what the controller
   knew. Preserve source/receipt clocks and distinguish actual from assumed
   activation, cancellation and fill-notice timing in each scenario.

2. **The current interval audit holds executions and eligibility fixed.** It
   starts with M50 fills and M50 full-fill notices
   (`pf_keep_pull_timing_audit.py:172-176`) and changes contribution times. It
   neither discovers fills absent at M50 nor recomputes quantities/queues.
   Its bounds are a useful conditional diagnostic, not full replay bounds.
   The execution-window implementation must replay the complete neutral order
   population, including unfilled orders and explicit excluded/unknown states.

3. **Event linkage and reproducibility remain incomplete.** The loader groups
   legs and propagates a minimum anchor
   (`pf_stop_ev_minutes_inputs.py:80-88`). The timing audit collapses candidate
   legs into a per-fill interval hull and does not save their identity ledger
   (`pf_keep_pull_timing_audit.py:108-121`, `:228-230`). A range built around a
   wrong link is still wrong. Retain candidate IDs, shared-event constraints,
   interval provenance and input/dependency hashes. Known historical
   lifecycle contradictions should be negative controls, not fitted away.

4. **The reporting population correction remains open.** AFTER/BEFORE/
   AMBIGUOUS describes all fills on pulled orders, whereas the headline calls
   it removed value. Preserve the removed-versus-kept cross-tab and use the
   correct denominator, as detailed in the main review. No new result is
   implied by this addendum.

5. **A stable subset is not automatically representative training data.**
   Timing certainty can be related to trade size, depth, source and volatility.
   Do not silently discard ambiguous or unfilled actions and then claim whole
   strategy skill. Keep neutral action rows, uncertainty statuses and coverage
   by day/side/source. Any later model must retain the declared comparison and
   be evaluated on later untouched days after freezing. Consumed data remain
   development data; agreement across clock assumptions is not live validation.

## One next priority for development

**Build and preserve an evidence-backed execution-window ledger, then validate
the ordering classifier on known-order lifecycle cases.** This is the next
bounded step, before a full replay or model change. It should accept separated
windows, mark overlap/unknowns explicitly, and refuse wrong identities,
contradictory windows or a delayed status timestamp substituted for a match.
Include cancellation/clock uncertainty and preserve original observations.

The deliverable is the versioned ledger plus interval/identity controls and
coverage, not a fabricated exact-time tape. It will identify which decisions
can already be evaluated and where additional timing evidence is required.
Queue-aware scenario replay follows that gate. A new capture is needed only
where existing evidence is insufficient for the chosen economic claim, not
as a blanket prerequisite for every decision. No live action is authorized or
performed by this documentation update.

## Checks and scope

Re-inspected the unchanged pinned timing/keep-pull/loader source at the cited
locations. This is a review-documentation update: no new empirical result,
training, tape read, implementation edit or test rerun. The main review's
bounded verification and limitations remain in force. Documentation diff and
review-only staged scope were checked; existing other-session edits preserved.
