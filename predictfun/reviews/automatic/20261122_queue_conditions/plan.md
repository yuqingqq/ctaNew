# Queue position in the next condition diagnostic

2026-11-22 UTC. USER: “We also need to consider queue position, which is critical.”
Source inspected: **ed7c3db557ab610c8d14f7ae2d700330759447c9**.
This amends the NEXT diagnostic in
`../20261122_horizon_conditions/report.md`; that report, its frozen protocol,
results and the stopped positive-pressure candidate remain unchanged.
No numerical experiment, fit, replay or strategy change in this amendment.

## Why it belongs in the decision state

Queue position affects when and how much an order can fill. Market conditions
affect whether those fills have positive value. Cancellation also gives up an
existing queue position; a later post receives a new position. Evaluate these
together, per order/gid and side, including the two ladder levels and partial
fills. Do not assume either that the front is always safe or that a deep queue
is safe: the latter may fill mainly during a large adverse sweep. These are
hypotheses to test, not findings from a new run.

Order distance from the best price, birth rung, order age and shares ahead
are different quantities. Distance and age in the latest five views do not
substitute for queue position. The source already supplies queue inputs; the
current pressure rule reads only its pressure signal alongside structural and
inventory precedence. This amendment adds a queue-conditioned diagnostic,
not an assertion that past models never had queue features.

## Existing inputs and their limits

Verified in `predictfun/research/pf_ladder_rest.py` at the source above:

| Quantity | Source / unit | Interpretation |
|---|---|---|
| `fp_queue` | `:451–470`, `:493`, shares | Client estimate of shares ahead from received frames, a running-minimum proxy; not an observed exact rank. |
| `o_q_rel` | `:115–116`, `:486–493` | Same queue estimate divided by the established liquidity scale. Keep its original normalization. |
| `fp_remaining`, `fp_size` | `:456`, `:468`, `:493`, shares | Own quantity remaining from delivered fill notices; retain submitted size and known filled quantity separately. |
| `o_depth`, birth rung, price | `:466–468` | Current distance from received best versus rung at birth; never call either queue position. |
| `ctx_generation_age_ms` | `:117`, ms | Submission age; name activation-relative age separately if added. |
| Queue source age/status | `:451–470`, `pf_age` | Unknown/out-of-depth/stale observations remain explicit; zero estimate is only estimated front. |

The running-minimum estimator need not track every consumed trade or distinguish
all volume ahead from later arrivals. Its accuracy is an open measurement
limitation, not a newly demonstrated simulator defect. `queue_summary` at
`:541–560` stores lifetime first/last/front summaries. Its `last` or final
front fraction must never become an earlier decision input.

The simulator's `ahead0` is initialized with the activation-time venue frame
(`pf_branch_engine.py:126–134`), even though it is stored at submission by
`pf_ladder_pressure.py:243–244`. It is diagnostic only. The pressure receipt's
next-same-side-order pairing (`pf_ladder_pressure.py:442–449`) is not necessarily
the replacement at the same price/rung, and its ahead difference is not a
counterfactual queue cost. Preserve the later reviews' corrected clock labels.

## One integrated next diagnostic

Extend the decision-to-fill × post-fill-horizon analysis, not a separate search
over queue rules. Keep every neutral decision opportunity, original identity,
side/day, no-fill rows, unavailable observations and censor statuses. FULL's
reached states remain separately identified if present. No future-fill-selected
training population.

1. **Add the mandatory queue axis before computing new tables.** Keep raw ahead
   shares Q, remaining shares R, Q/R when R>0, the existing `o_q_rel`, known
   filled shares and both age/price-distance fields. Use fixed descriptive
   bands: Q=0 (estimated front), 0<Q<=R, Q>R, unknown, and invalid/nonpositive-R.
   Missing/nonfinite/negative estimates must not become Q=0. These are analysis
   bins, not proposed cancel thresholds. Save absolute quantities beside ratios.
2. **Test the joint state economically.** Cross those queue bands with the
   already declared Binance-pressure bands, separately by side/day; show the
   existing Predict-flow view as a declared secondary split. Report sparse
   cells as sparse instead of merging to maximize separation. Retain the five
   previous marginal views, but avoid a full Cartesian grid of every feature.
   Record the added queue view and interactions as exploratory multiplicity.
3. **Keep both timing axes.** Use the already specified fill-arrival intervals
   after the cancel effect (to 250 ms, to 1 s, to 5 s, later) and the saved
   50/100 ms, 1/5/30 s post-fill markouts. For each queue/condition cell show
   opportunities, at-risk follow-up, any-fill rate, filled quantity, partial
   fills, positive/negative/net value per opportunity and per filled share.
   Reconcile disjoint arrival intervals to the accepted remainder-of-generation
   window. Termination and missing horizons retain their statuses; do not
   describe observed pre-cancel fills as the unobserved always-kept fill rate.
4. **Queue progress only where causally reconstructable.** Initial-to-current
   ahead change and recent depletion over 250 ms are useful secondary inputs,
   but require a saved earlier receipt-time state of the same gid/price. Use
   no future lifetime summary or venue state. If the saved evidence lacks the
   necessary history, mark unavailable and finish the core tables; do not
   require a new tape replay simply to complete this optional field. Distinguish
   not-yet-born orders from missing history. Keep partial-fill notice latency.
5. **Queue quality diagnostics stay separate.** Where a saved venue state has
   the exact same gid and decision phase/time, compare it with the client
   estimate and report availability/disagreement. Such simulated truth is
   explanatory only and may share assumptions with the replay; it is not
   exchange-confirmed ground truth. No nearest-time or activation-to-decision
   substitution, and no venue queue in a signal or selection predicate.

Interpretation: first ask whether the signal separates expected value within
comparable queue states and arrival windows. A difference solely in fill rate
or filled quantity is not demonstrated adverse-movement discrimination. A
better queue gives access to both good and bad fills; report both. Do not
promote a queue-dependent rule from consumed-day descriptive cells.

## Controller implication and acceptance

The eventual action comparison is the value of **keeping this particular
remaining order at its existing queue position** against **cancel, then the
declared wait/re-entry continuation**, over a common economic horizon with
consistent inventory treatment. Include fills before cancellation takes effect
and the changed subsequent entry opportunities. If queue effects already enter
the two branch values through fill probabilities/timing, do not add a second
arbitrary “queue penalty.” This amendment does not authorize a new paired
replay or provide a measured monetary value for queue priority.

Within the one next development receipt, check same-gid joins, queue-zero versus
unknown, band boundaries, unchanged original opportunity membership and union
accounting, and receipt-clock causality. Use a future-only queue perturbation
that leaves decision inputs unchanged and a delivered-queue control that changes
them. Check partial fills only change known remaining after the notice. A
changed price/gid must fail a claimed matched-replacement comparison. Bundle
these with the existing arrival-boundary/missing-mark controls; no separate
minor-fix review cycle.

Implementation and measurements remain with the development session. Saved-data
scope and existing resource caps still apply. This amendment does not resume
the positive-pressure experiment stopped at its 0.14-hour round cap, change
its frozen primary endpoint, expand to new days, or authorize extra compute.
