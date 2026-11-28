# Critical review of JOINT-LADDER and ORDER-HARM

Source review and synthetic checks, 2026-11-14 around 05:55 UTC. Sources:
JOINT-LADDER `879fe15a` (after its initial fixture correction), ORDER-HARM
`67a6e6fb`. No historical replay or model fit was run by this reviewer.

The joint engine now contains both sides, shared position, price-priority
allocation of a trade leg and a duplicate-price placement check. ORDER-HARM
includes the best-price order and decision states throughout guard activity.
These address important structural gaps. Three material problems remain in
the new target pipeline; the first is demonstrated feature look-ahead.

## 1. Critical: queue features contain future information

ORDER-HARM reads `a0` from the completed reference simulation and calls
`pf_next_price.ahead_at` at decision time (`pf_order_harm.py:205-208`).
`pf_queue_place.py:125-138` initializes that `a0` using the next venue frame
when it is within 100 ms of activation, even if it is after the decision.
The inferred ahead feature inherits it. In addition, `ahead_at` uses venue
frame time rather than the client-known book clock, and explicitly ignores
trade consumption of queue ahead. It is not the current observable queue.

The source-function counterexample places an order at 200 ms, activating at
240 ms, and evaluates features at 250 ms. The book at time zero shows 100
shares. Changing only the future frame at 300 ms from 100 to 900 shares changes
both `a0` and `ahead` at the 250 ms decision from 100 to 900. All information
available by the decision is identical. This is a reachable defect, not an
estimate of how many historical rows it contaminates.

Fix: build features from a client-known event-state snapshot, separately from
the simulator's latent queue model. Replay known trade consumption and fill
notifications; carry remaining quantity and uncertainty. A future-only change
must leave every decision feature unchanged. Retain unknown depth as unknown,
rather than inheriting the simulator's zero-ahead fallback.

## 2. Critical for economic claims: overlapping rows reuse outcomes

The builder samples every 250 ms for each resting rung, and assigns each row
the next two seconds of reference-lane fills. The same order and physical fill
therefore appear in multiple action windows. No order-generation identity is
saved in the output key (`day, m, side, k, rung_i`). R622 then sums those rows'
DELTA to assess an offline keep/pull rule.

A source-function test with one 30-share fill losing 2c/share gives -2c/share
to the decision at 250 ms and again to the decision at 500 ms. Adding rows
reports -4c/share from that one fill. This may be a descriptive, time-weighted
decision-state score, but is not an additive policy value or independent-action
comparison. It also violates the repository's action-unit requirement.

Fix: preserve generation/fill identities and predeclare action sampling or
generation weighting. For economic improvement, run the sequential policy and
value each physical fill once. Repeated state predictions may be useful, but
their overlapping labels cannot become summed earnings or independent evidence.

## 3. Critical for labels near the end: unmatched terminal policies

KEEP is built from the original reference lane ending at 60 seconds.
PULL's `run_fork` replaces its end with decision time + 5 seconds, without
extending the decision grid. The builder only increments `truncated5`; these
rows still enter labels and fitting (`pf_order_harm.py:193-198`). There is no
corresponding primary-two-second eligibility status.

A source-function control at decision time 59 seconds shows the reference
ending at 60 seconds while the pull fork ends at 64 seconds, although its last
decision remains at 59.95 seconds. The inherited loader's trade tape is also
limited to the original first minute plus one second. Thus terminal policy
and observation coverage differ across branches; truncated outcomes are still
used as ordinary labels. Primary two-second labels near the boundary are also
exposed. Historical frequency and numerical effect have not been measured.

Fix: give both branches the same terminal policy, full observation horizon and
mark availability. Otherwise record a horizon-specific censored status and
exclude that label from fitting/evaluation while retaining the action row.

## Joint-engine checks still needed

- Its cap is explicitly a delayed soft trigger on known filled inventory,
  not an exposure reservation or hard bound. Six 30-share bids may be live
  while known inventory is zero even with cap=60. Pending/executable exposure
  can overshoot it. Report actual risk and overshoot; equal nominal caps do
  not establish matched risk.
- At the 05:53 UTC snapshot, four completed dates had zero parity mismatches,
  but only 293 of 20,736 arm/clock/window-side comparisons were checked;
  20,443 were skipped for interaction. This is expected by its declaration,
  but most paths need independent volume-conservation, priority, pending-cancel
  and position/cash reconciliation controls. Matching the old independent
  engine on a small non-interacting subset cannot validate joint execution.
- The inherited binary outcome conversion still prices every registry `Up`
  as 100 (`pf_kp_s4.py:1646`, `pf_joint_ladder.py:330`). The documented tie
  issue P10 must be corrected before using settlement marks as evidence.
- Future-order/policy effects remain approximate: the label's KEEP lane never
  pulls later, its fork has no shared portfolio, and fitting is leave-one-day-
  out on consumed dates. These are declared research limitations, not fixed
  by good offline prediction scores.

## Operational progress and verification

The joint driver's first self-test refused at 05:44:48 UTC. A committed
correction fixed two expected fixture times; it passed on restart at 05:45:23.
The historical chain then progressed normally. At the initial 05:53 snapshot,
four day receipts existed and the fifth day was running. ORDER-HARM was
committed; no row/model output was present in that snapshot.

At the receipt snapshot, 05:56:43 UTC, six dates had completed and Nov 1 was
running: 1,728 windows, 577 parity comparisons checked, 30,527 skipped, zero
mismatches. ORDER-HARM still had zero row/fit receipts and no report.

The synthetic instrument extracts the actual named source functions via AST,
with a minimal execution fixture. A causal equal-input control passes; the
future-dependent feature fails its invariance check; the cancellation-latency
exclusion control passes. Its first invocation lacked the fixture's `qrule`
binding and raised NameError before producing a result; adding that binding
fixed the fixture. No scientific source was altered. See
`research/model_order_harm_source_review.py` and
`reviews/model_order_harm_source_review_20261114/synthetic.json`.

Recommendation: treat ORDER-HARM's current construction as requiring revision
before interpreting model skill or economic selection. Preserve this version;
fix in a superseding target build and rerun its causal/identity/boundary checks.
No worker was stopped or restarted and no other-session source was edited.
