# QUEUE-PRIORITY-SENSITIVITY-v1 — combined review

2026-11-22. Pinned target: **c7404f212700ebf1c6977e1806a28c23b5961f02**.
Range: **425e9fbcc59502ab1c03b2c0cd9f71f7a526d140..c7404f212700ebf1c6977e1806a28c23b5961f02**.
Sources inspected in detached checkout `/tmp/pf-qf-review-c7404f21`.
Freeze `b4ecbc34`, checker supersession `580411db`; source closure unchanged
between freeze and target. Development owns implementation. No historical
replay, fitting, label build, new day, push or external messaging in this review.

## Decision

**Accept the numerical sensitivity result: MIXED, still negative. No blocking
implementation/accounting finding and no required rerun.** Correct the causal
interpretation below before using this receipt to rule out a mechanism.

On the 19 complete markets from consumed Oct 24–27, as of this receipt:

| Metric | CURRENT_QUEUE | FRONT_AT_ACTIVATION | Difference |
|---|---:|---:|---:|
| END, cents | -86,109.61 | -65,937.09 | +20,172.51 |
| Eligible +1 s, cents | -85,164.28 | -61,812.39 | +23,351.88 |
| Eligible +1 s, cents/share | -1.077 | -0.528 | +0.549 |
| Filled shares | 79,399.47 | 117,505.63 | +38,106.17 |

END improves on 13 markets and worsens on 6; its day differences are
+23,892 / +1,131 / -2,956 / -1,895 c. Every arm remains negative on every day.
The declared `economic=MIXED`, `reading=mixed` is reproduced. The pooled
23.4% deficit reduction is **not an estimate that queue explains 23.4% of all
causes of loss**. This is a complete-policy counterfactual under fixed external
flow and the existing execution, sizing and inventory assumptions.

## Consolidated findings

1. **P2 — NONBLOCKING for numerical acceptance: the report overstates the
   causal conclusion.** `predictfun/reviews/ladder_queue_front_20261122/report.md:54,80,90`
   says the improvement is mostly orders filling before sweeps, that priority
   only changes same-price fills, and that priority does not address sweep
   or inventory losses. Only the *direct queue deduction in a single match*
   has that restriction (`pf_branch_engine.py:167-173`). The run intentionally
   feeds changed fills into subsequent inventory and decisions
   (`pf_ladder_queue_front.py:116-123,174-181`). Eligible sweep shares actually
   fall from 41,632 to 31,905, and sweep +1 s value improves from -86,932 to
   -82,694 c. Priority therefore changes sweep exposure through the feedback
   path. The volume/value identity at `:256-265` is an arithmetic decomposition;
   it cannot distinguish earlier depletion from changed sizes, prices, later
   generations or inventory restrictions. Its volume term is also based on
   CURRENT's rates; it is not an independently identified causal component.

   **Correction:** retain “not supported *by this test of the fixed policy*”
   at report line 76, and replace the categorical last paragraph with:
   “Front priority improves pooled fill mix but does not make this policy
   profitable at the tested execution settings; effects are mixed by day.
   It directly changes same-price allocation and indirectly changes later
   sweep/inventory exposure. The remaining 1-second loss is concentrated in
   through-price fills; this test does not establish whether that residual
   is preventable by faster actions, earlier information, pricing or inventory
   decisions.” No tuned queue variant follows this mixed result. The test is
   not a monetary upper bound and does not reject queue acquisition in general.

   The END class flip at report line 58 is numerically correct but is not a
   diagnosis of an inventory-management defect. Class END contributions mark
   all class fills to one terminal mark; opposite inventory from other classes
   can offset those contributions. In aggregate, the verified identity is:

   | Arm | Eligible +1 s | Subsequent movement to END | Late/unsupported +1 s | END |
   |---|---:|---:|---:|---:|
   | CURRENT | -85,164.28 | -1,416.84 | +471.51 | -86,109.61 |
   | FRONT | -61,812.39 | -5,175.38 | +1,050.68 | -65,937.09 |

   Thus the large opposite class movements mostly cancel at policy level;
   most of the aggregate deficit is already present at +1 s. No inventory
   ablation was run. Similarly, report line 56's 1/5/30-second averages use
   different native cohorts (`pf_ladder_queue_front.py:239-246`): call them
   native-horizon averages, not the measured trajectory of an identical cohort.
   For example FRONT same-price eligible shares are 85,088 / 84,257 / 78,789.
   See `class_horizon_denominators.csv`. These qualifications need a superseding
   interpretation, not another replay or a new diagnostic round.

2. **P3 — NONBLOCKING, defer repair: the independent checker covers less than
   the complete exported reading.** `fq_checks_v2.py:195-222` compares core
   economics and selected fields. It omits 5/30-second class values/censoring,
   lost/added economic sums (it trusts the stored `identity_gap`), and detailed
   daily signs and participation/inventory readings. Three in-memory corruptions
   pass their named families: add 99,999 c to a class's `mo30s`; add 99,999 c to
   `added_end` without changing stored gap; change FRONT's Oct 24 absolute
   sign to nonnegative. Evidence: `verify.py`, `evidence.json`.

   **Bounded workaround completed:** all 76 class rows including 5/30-second
   values and exclusions, all lost/added fields, the full predicate object and
   57 exported arm rows have been checked. Core economics was reconstructed
   independently from fills; full helper reproduction supplements that check
   for unchanged ancillary metrics. Actual exports and their committed receipt
   copies are correct. No rerun or fix-only round. Extend those comparisons
   with the three bad-value controls when this checker is next reused.

   Family F at `:181-192` checks class/size consistency against saved metadata,
   not a separately loaded public tape. The generator's direct `E.tr[it]`
   lookup and trade-side refusal (`pf_ladder_queue_front.py:106-113`) are
   correct and exercised by synthetic controls. This review additionally
   verifies event time/side/price/size consistency across both saved arms.
   Do not describe this as an independent reconstruction of the raw tape.

No other material issue found in the affected pipeline. These items do not
invalidate the MIXED result or justify restarting the experiment.

## Contract and integration findings

- **Treatment:** `pf_ladder_queue_front.py:83-103` calls the unchanged add9,
  excludes refusals/non-new/inactive actions, then zeros external ahead once.
  The engine initializes activation-clock state at submission internally;
  fills cannot occur before `act`. There is no future-outcome selection or
  repeated reset. `pf_branch_engine.py:159-171` keeps the running minimum,
  so zero persists. Own price/time priority and partial remnants are unchanged.
  Swap refuses a nonfresh engine. CURRENT's conditional override is inert.
- **Control/feedback:** CURRENT reproduces fingerprint, orders, fills, requests,
  value, guard log, commands and queue records on 19/19 markets. Each treatment
  arm constructs a fresh branch/policy; its own notices/reservations/inventory
  drive subsequent commands. Accepted FRONT generations: 4,874/4,874; all
  accepted final ahead values also remain zero. Original clocks and 200-share
  cap are unchanged. The later async/FAST10 proposals were **not** implemented
  or tested by this round; approval here was the two-arm 425e9fbc experiment.
- **Fills/accounting:** same-price equality and side-specific through relation
  match the engine. Own fills conserve public size per event, and each fill
  lies within activation/removal/END, has its order's price/side, and respects
  order quantity. Cash, position, marked value, END decomposition, common +1 s
  cutoff, native longer horizons, and multiset lost/added identity reconcile.
  “Through” is an execution relation, not proof every such fill is informed.
- **Sampling/projection:** the pilot is first two complete markets per date
  (8), rest is disjoint (11), union is the same 19. No Oct 27 rest job is
  needed. The original 96-market coverage census remains 19 complete / 54
  missing endpoint / 23 stale future book; no general full-day return claim.
  Projection recomputes to 0.212069h batch and 2.217638h line including the
  0.03h review reserve; both caps pass. Projection precedes the three rest jobs;
  continuation is a cost gate, not an outcome gate.
- **Checker v2:** only `_edit` and `main` function bodies differ from v1;
  the former fixes tuple-key assignment, the latter changes output names.
  Documentation/source-binding additions identify v2. No numerical predicate
  changes. V1's failed falsifier is retained; v2 was committed before it ran.
- **Prior issues:** the accepted queue/price audit's grouping-clock and detailed
  reader-coverage limitations remain deferred, not silently “fixed.” This run
  uses no grouping-derived queue rule. The new F/Q checker gap above has its
  own verified workaround. Baseline helper code remains frozen.

## Verification, limitations and resources

`verify.py` ran the pinned seven-group synthetic selftest, all 10 frozen checker
booleans and all 10 falsifiers. The 26 saved mutant outcomes were rejudged
against their indices, names and the pinned sources; no mutant rerun. It
checked source/manifest/part identity, committed measurement copies, 19 controls,
57 arm rows, 76 class rows and full readings. `diagnostics.py` verifies the
supersession, projection, resource log ledger and the extra bad-value controls.

Only saved records and small synthetic tapes were used. No market-data load,
teacher fit, strategy replay or untouched validation. Peak RSS 316 MiB.
Measured review checks: **9.09 user-CPU seconds = 0.002525 core-hours**, below
the 0.03 reserve. The source-log ledger has no duplicate job receipts: 2.171622h
before this review and omitted tiny planning checks; approximately **2.1742h**
after them and these checks, before final commit verification. Both the
0.25 batch cap and approved 2.25 line ceiling remain satisfied. Final commit
verification is recorded in `run_light.log`; it does not run research.

This supports a reliable **consumed-day sensitivity**, not predictive skill,
exchange-ground-truth queue ranks or live profitability. Zero cost/gross
accounting is unchanged. Unchanged external flow is particularly restrictive
when our hypothetical priority diverts substantial fills. Four consumed days,
only 19/96 complete markets, and no achievable-priority mechanism limit scope.

## ONE concrete next priority

**Prepare the previously requested faster, concurrent-execution sensitivity;
do not tune queue rank or fit another guard from this result.** The economic
question is whether the same signal/policy can avoid enough residual harmful
execution under a faster action contract, with and without the already-declared
front treatment. This tests a remaining assumption; it is not a claim that
latency must explain the losses.

Reuse this round's two hash-bound baseline ledgers. Add exactly the two
ASYNC_FAST10 cases from the
[concurrency amendment](../20261122_async_execution_amendment/plan.md): current
queue and the identical one-time-front treatment. Keep 10ms post/live-cancel
effects as the already-proposed sensitivity, 100ms fill/cancel notices, 50ms
grid, original feed/M50 clocks, price/size/guard/inventory rules and external
flow. Independent requests may overlap; retiring records must remain reserved.
Keep the 200-share cap and desired-exposure limit. Two fully reserved 100-share
bids still leave no capacity for another 100. No silent overlap-cap increase,
atomic replace assumption, faster information or notification shortcut.

First deliverable in the same development batch: source/config freeze and
small controls for concurrent independent commands, slot-vs-risk blocking,
pending fills, post/cancel races and strict removal, plus baseline parity and
effective settings in traces. Reuse the first-two-markets/date pilot and an
outcome-blind runtime projection. Then replay only the two new profiles on
the same 19 markets if the cost gate fits. No new data or parameter search.

Declare comparisons before outputs: combined execution-profile differences
at each queue assumption, queue difference under the faster profile, and the
interaction. Report absolute and incremental END/+1 s economics per day,
same/through quantities and values, participation, inventory, and slot/cap
blocks. The speed contrast includes concurrency, so it cannot isolate wire
latency. All-negative outcomes mean these assumptions have not rescued this
fixed policy; mixed outcomes start no tuned speed/queue variant. A nonnegative
development result would motivate timing feasibility and separately approved
untouched validation, not promotion.

**Needs the USER before launch:** the current approval covers this completed
queue experiment, not the additional execution-profile batch. A concrete
resource proposal is **<=0.25 additional development core-hours and a 2.50
cumulative line ceiling including review** (replacement for earlier unapproved
FAST10 budget proposals, not additive to them). This review does not approve
those caps. Preparation is possible now; no replay starts under an inferred
extension. POSITIVE-PRESSURE-GUARD-v1 remains stopped.

**Largest evidenced economic limitation:** even hypothetical front priority
leaves -61,812 c at +1 s and -65,937 c at END under the current execution
contract. Through-price fills dominate short-horizon losses. The test does
not identify how much of that loss is avoidable by actionable warning and
execution timing, nor show an achievable profitable queue policy.
