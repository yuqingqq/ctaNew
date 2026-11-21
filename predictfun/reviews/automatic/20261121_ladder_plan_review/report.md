# Ladder plan review — proceed to the bounded test

2026-11-21. Reviewed committed target:
**e3c2b998ec6a31d8cc682f56e41abe0c74824549**.

**Decision: the research design is ready for implementation and the pilot.**
No blocking plan defect was identified. This accepts the experiment design,
not an unreviewed ladder implementation or its economic results. Development
may complete the existing correctness checks, freeze the executable package,
run the pilot and expand within budget without an intermediate review.

The authoritative documents remain the
[single-strategy plan](../20261121_default_post_ladder/plan.md) and
[metric definitions](../20261121_ladder_post_analysis/metrics.md). This review
clarifies their evidence requirements; it adds no strategy arm, new data day,
model fit, economic threshold, or separate experiment.

## What is sound

- The experiment directly tests the requested mechanism: default posting,
  two persistent prices per side, nominal 100-share orders, repeated partial
  fills and queue age. It does not require a learned entry score to participate.
- Inventory is managed across all levels, including pending orders and
  unnotified fills. The cap is shared, new sizes respond to known inventory,
  and partial fills preserve priority. Inventory and structural removals are
  explicit policy choices whose participation costs will be measured.
- Orders and inventory persist through the supported market lifetime, to
  opening +298s. Minute reports and 5s/30s markouts do not reset the strategy.
- One fixed policy, at most 96 preselected consumed-day markets, and an
  eight-market pilot keep the run focused. Selection is independent of fills
  and returns. Missing paths remain statuses rather than replacements.
- Economics reconcile to cash plus marked inventory. Quote uptime, actual
  quantities, tail losses and terminal inventory prevent a low-participation
  or unclosed-position result from being mistaken for model improvement.

## Consolidated execution checklist

1. **Existing prerequisite — result-critical, already in the plan.** Use an
   actual multi-order client, not the old single-order wrapper with new size
   constants. `pf_branch_engine.py:238`, `:248`, `:265` and `:327` retain one
   open gid per side; `:139` is the 30-share add path. The plan already requires
   the extension and focused checks at `plan.md:227-259`. Check shared public
   trade volume, per-gid notices/partials, aggregate reservations, receipt-only
   decisions and delayed cancellations together. The invariant includes both
   levels and pending removals. Do not accept economic output from an
   implementation that fails these checks. This is not a new fix/review round.

2. **NONBLOCKING clarification — make cancellation attribution useful.**
   The old client combines no-book, excessive-distance and crossing into one
   `structural` request (`pf_branch_engine.py:277-293`). The plan requires
   reasons (`plan.md:138-146`) and the metrics require reason-specific churn
   (`metrics.md:85-89`). In the same implementation, retain all triggering
   subreason flags and the first request's deterministic primary reason,
   together with gid, birth rung, received frame/time, price distance,
   inventory mode, request time, effective removal and acknowledgement.
   Preserve the existing policy/tie order. Repeated requests do not become
   independent cancellations. If flags can be reconstructed exactly from
   saved states, reconstruction is an adequate bounded workaround; a generic
   reason label alone cannot explain which structural rule causes churn.

3. **NONBLOCKING clarification — retain reusable comparison states.**
   Entry and fixed/hash-selected live checkpoints are already required
   (`plan.md:184-190`, `metrics.md:116-124`). Freeze their sampling rule and
   inclusion denominators before the pilot. Preserve input/source identity,
   receipt pointers and enough state/event records to reconstruct the first
   live review, actual pull and inventory-mode transitions without another
   branch simulation. Full feature vectors need not be computed at every
   unchanged tick. Keep ordinary, profitable, no-fill and still-resting
   checkpoints, not only states preceding losses. Check a small deterministic
   reconstructed subset against its saved client view in the existing receipt.
   Per-action logs diagnose the controller; a sampled live-state population
   with its denominators supports comparisons with states where it kept.

No additional plan-level blocker is open. Items 2-3 are implementation
clarifications for the same batch, not reasons to wait for another design
review or to build a large new test suite.

## Pilot and analysis contract

The existing eight-market pilot is the practical feasibility check. Confirm
the requested ladder/actual quantities, independent ledger reconciliation,
coverage through the supported endpoint, and usable logs; measure runtime and
memory before expanding. The full ceiling remains two aggregate core-hours,
also bounded by the unused previous allocation; pilot ceiling 0.25 core-hours.
The path builder's +298s endpoint and explicit gap/mark refusals are confirmed
at `pf_queue_place_test.py:65-89`. Do not remove these checks to force coverage.
If few paths or fills are usable, report that limitation and its reason; do
not infer an absence of economic opportunity from inadequate observations.
If projected cost exceeds the allocation, retain the pilot and report the
measured need rather than shortening each market or dropping expensive cases.

Retain machine-readable market membership/statuses, order and fill/notice
ledgers, decision/state records, feature metadata/sampling, minute equity and
the summary tables under bound code/config/input identities. Exact filenames
are development's choice. These are outputs of the existing run and summaries,
not a new artifact-framework requirement.

The combined analysis should decide which limitation dominates:

| Observed pattern | Next question supported by this baseline |
|---|---|
| Positive fill edge turns negative soon after filling | Which receipt-known flow/book states precede adverse fills, and could an action arrive in time? |
| Short markouts are positive, but later value falls with persistent inventory | Is inventory reduction or quote sizing the next controller lever? |
| Little at-best participation, many inventory/structural removals or pending gaps | Which policy constraint explains lost participation and queue age? |
| Repeated fills and ordinary gains are erased by a few markets | Is the tail concentrated in observable states, including matched ordinary/no-fill cases? |
| A level rarely fills or is usually remote from the best | Is the intended queue mechanism reached under this fixed ladder policy? |

These patterns guide investigation, not automated causal verdicts. A single
policy does not identify gains from the ladder or inventory control, nor the
losses a cancellation saved. Choose one subsequent declared comparison from
the evidence. Do not tune many guards on the baseline and call their best
in-sample result validation. This four-day consumed sample remains descriptive
development data; no new untouched validation is authorized by this review.

## Review work and next priority

Read the plan, metric contract, state/handoff, reliability/batching rules and
the matching/client/path boundaries cited above at the target. No simulation,
training, tape build or economic result was produced. Documentation and
review-owned state are the only changes; existing working-tree edits remain.

**ONE next priority:** development should implement/freeze
`LADDER_REST_INVENTORY_v1` with the existing checks and the two logging
clarifications, run the eight-market pilot, then the same declared sample if
the measured resource budget permits, and return one combined code/results
receipt identifying the largest loss/participation mechanism. **No additional
USER permission is needed within this scope.** Minor cleanup can remain listed
in that receipt and must not restart a fix-only cycle.
