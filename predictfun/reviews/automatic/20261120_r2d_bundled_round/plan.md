# R2-D: one consolidated fix and measurement round — 2026-11-20

User instruction: **“bundle all issues, let the other session to fix all issues
in a round next.”** Development owner: Claude session aea5bfb9. This thread
owns review and acceptance. Deliver all items below together, then request
**one review of the combined code and distributed-run receipt**.

Basis: reviewed source **f96c38d01a7f27ed4fd39ddf2ddadeecf162b84b** and review
commit **4393ad19d97f35e07b476cb93670fb7fbe4f614f**.
[Evidence and complete sampling specification](../20261120_f96c38d0/report.md).
This document consolidates that contract and the user's request to accelerate;
it changes the delivery sequence, not the teacher, labels or sampling estimand.
Current uncommitted development work is preserved and is not certified here.

## Outcome of this round

Produce a reliable, resumable sampled-label builder **and economic readings
from the bounded sample spread across both full days**. Complete the known
acceptance, operational and CLI repairs together. Freeze the implementation
and plan, run the declared bounded integration, and submit one closing matrix
with the final commit and artifact identities. Do not stop for a separate
review after each fix, selftest, sampling adapter or resource correction.

The existing 128-pair integration and arm-binding fix remain accepted. The
remaining issues concern production acceptance and coverage. The next run is
a consumed-day target/readiness measurement, not a model performance test.

## Complete work list

| ID | Work in this round | Closure evidence in the combined receipt |
|---|---|---|
| B1 — execution provenance; **BLOCKING** | Use a common preflight to enforce the producing adapter/runner source closure, frozen teacher and libraries, sampling plan, keep/pull metadata, clocks/offsets and market-feature input identities. Preserve the hashes returned by the existing input builders. A changed plan/source/input uses a new output namespace. Bind completion records and the accepted dataset to those identities. | Intact resume passes; changed adapter, feature input, clock/offset, metadata and plan are refused before a new arm runs or an old record is admitted. Existing artifact-only readers must identify historical inputs explicitly; they cannot imply current-source equality without checking it. |
| B2 — record and finalization integrity; **BLOCKING** | Share strict validation across promotion, resume and accepted publication. Validate candidate/executed fork, arm names, schema/status, finite values, numerical Y and arm accounting, pre-fork equality, binding/causal/parity counters, quantity invariants and record bytes. Reconcile all selected keys/statuses, including duplicates and extras. Hold publication on undisposed stops; distinguish diagnostic output from accepted output. | Intact records pass. Persisted Y+1, wrong fork, reconciliation=false and binding_bad=1 are refused, alongside record/manifest corruption and an undisposed stop. Recompute numerical identities instead of trusting success booleans. Legitimate non-ok outcomes remain counted; failed invariants do not become censoring statuses. |
| B3 — durable stop history; **NONBLOCKING for existing labels**, close now | Archive prior events before starting any status-overwriting writer, including the hard-RSS watchdog and load/plan/finalize paths. Keep content-keyed events, plan-bound reasoned dispositions and HELD semantics. | Two successive stops at the same status filename both survive; disposition of the first does not clear the second. Intact stop/resume and explicit disposition paths still pass. Use a synthetic watchdog callback, not a real memory spike. |
| B4 — memory/process isolation and census guards; **NONBLOCKING for existing labels**, close now | Use isolated day/partition processes and release remaining world/factory references. Apply the declared resource coverage to loading, census, pairs and finalization. Retain never-reset process HWM, 1,536 MiB checkpoint and 2,048 MiB hard ceiling; retain 120 s/pair. Keep stopped records and their dispositions. | A small lifetime sentinel proves prior-day input objects are released when intended; bounded process logs and synthetic guard controls demonstrate coverage. Report measured peaks for each process. The old receipt's D remains FAIL; the new receipt computes its own D. No silent limit increase. |
| B5 — census CLI completion; **NONBLOCKING for existing labels**, close now | Fix the final print's removed-field access. Preserve the immutable manifest and deterministic streamed candidate output. | A tiny end-to-end CLI census exits successfully and its printed count agrees with the persisted manifest/candidate count. Existing-file refusal still works. |
| B6 — bounded test execution | Keep the required positive controls and falsifiers, but remove accidental production-sized work from mutation tests. In particular, the no-cap mutation must be detected on a tiny selection fixture or an explicit invariant before expensive pair execution. Run affected suites once on the final code. | Every mutant is classified as a specific caught assertion/refusal, escape, equivalent or harness failure. A timeout/crash is a harness failure, not a successful catch. Log durations and source hashes; retain failures. Preserve existing binding, reversal, parity and nine-state controls. |
| B7 — representative sampling, features and receipt | Implement the already-declared `r2_label_sample_v2`, its probabilities/weights and causal fork-feature output. Run the nested hourly distributed integration after B1/B2 pass locally. | Deterministic sample manifest, small inclusion/weight fixtures, interruption/resume identity, persisted 107-input schema and a complete economic/resource receipt described below. |

B1/B2 are the known blockers to **accepting new production results**. B3–B5
are bundled operational repairs and do not invalidate the saved labels. B6/B7
make this round both faster and informative. Do not reopen the closed
arm-binding, library-enforcement or value-denominator findings unless a new
reproduction shows a regression.

## Sampling and run size — unchanged from 4393ad19

- Population: all original shared worlds on consumed Oct24–25, both sides,
  five existing roll-in sources, original-time POST and the existing rotating
  every-fifth-tick census. Missing paths and other exclusions remain statuses.
- Full planned sample: hash-select at most **64 worlds per UTC hour per day**,
  at most **3,072 worlds / 18,432 pairs**. Seed/domain, canonical keys, tie
  order and probabilities are specified in the linked report. Preserve those
  choices; select before observing paired Y.
- Per selected world/side: eligible neutral original-time POST, one uniform
  hash-selected additional POST from the remaining eligible census and one
  uniform KEEP. If the mandatory POST is absent, use one draw from the available
  POST pool. Keep zeros; do not filter on fills, sign, score or realized loss.
- Retain the existing inclusion-weight contract:
  `w = 1 / (pi_hour * N_original_side_head * q_row)`.
  Mandatory POST has q=1; its additional draw has q=1/(N-1). Without a
  mandatory POST, q=1/N; KEEP q=1/N. Preserve inverse world probability.
- First and only real-data deliverable in this round: the **nested first-ranked
  one world per observed UTC hour/day**, at most **48 worlds / 288 pairs**.
  Run both full days' selected hours, not the midnight prefix. Persist ordered
  **107 causal inputs at the common fork before the forced action**, using the
  existing builders. This pilot's membership is narrower than the planned
  training sample; label its summaries accordingly.

The frozen `pi0_50_v1`, whole-portfolio paired Y, 5,000 ms END, 50 ms decision
grid, 58/37/100 ms timing and explicit zero costs remain unchanged. The detailed
freeze hash and input/target definitions are in the linked review.

## One acceptance gate and required readings

The receipt must have computed predicates for provenance, record/ledger
validity, population/weights, causal fork features, deterministic resume and
resource rules. Each has a working positive control and known-bad refusal.
Provide a B1–B7 closure table with final source lines, check/log paths and
artifact hashes. All actual bounded-run failures and attempts stay visible.

Alongside implementation checks, report the following **descriptively**:

- Planned versus realized worlds, markets, hours, original opportunities,
  selected candidates and final statuses by day/side/head/source. Include n
  and as-of, and distinguish repeated source rows from independent originals.
- Y=0 / Y>0 / Y<0 counts, distinct originals behind each, mean/median/tails
  in cents, and the contribution of the largest absolute effects. Show both
  days and both heads. Include source, horizon, generation and partial/queue
  coverage; report absent states explicitly.
- Both arms' value/cash/inventory components, fills/filled quantity where
  available, actions, generations and fallback statuses. These paired labels
  are not settlement P&L or a deployed controller's realized profitability.
- Raw and weighted summaries with their exact populations, raw weight totals,
  concentration and effective sample size. Pilot weights are not evidence that
  the pilot itself represents the full production sample. Do not create
  row-level confidence intervals from these two consumed days.
- Wall/core time, mean/max seconds per pair, load/census time, process peaks,
  all resource stops and the projected full-sample cost. Keep the **40 core-hour
  planning ceiling**, measured against the distributed pilot; do not adapt
  sample membership to the observed Y values.

This receipt answers whether the full sampled build can run reliably and
whether useful action-value variation appears across the distributed states.
Zero or negative Y is a result to report, not a failing implementation test.
It does not establish prediction skill: there is no candidate fit in this round.

## Execution sequence and review boundary

1. Implement B1–B7 together, using current work in progress where appropriate.
   Run tiny unit/control fixtures while editing. Keep the frozen teacher source
   closure intact; new orchestration remains outside it.
2. Run affected final-source synthetic suites once. Commit the implementation,
   sampling algorithm and immutable plan before paired outcomes are generated.
   Synthetic controls may use temporary directories; freeze/provenance requires
   the actual production builders in the repository.
3. Proceed directly to the <=288-pair distributed run when local reliability
   checks pass. Use isolated processes and at most the four already-authorized
   development slots. A review between steps 1, 2 and 3 is **not required**.
4. Commit one superseding receipt with the closure matrix and measured results;
   send one combined review request. One development round may use an initial
   freeze commit and a later result commit; it need not be one Git commit.

Return for review before running the remaining <=18,432-pair sample or fitting
a candidate. Report residual nonblocking cleanup in the same receipt; it does
not automatically require another fix-only round. A new issue blocks progress
only when a concrete reproduction shows it can invalidate labels, populations,
causality, accounting or accepted-result identity. Preserve runtime stops and
their declared operational handling. Do not add unrelated refactors, new model
families or feature searches to this package.

No additional USER approval, new schedule, external messaging or reviewer push
is needed. Development retains implementation ownership. The reviewer will
assess this combined deliverable once and identify the next scientific step.
