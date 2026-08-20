# PM_STRUCT_ITER8_REVIEW — v8 structure acceptance review

Object: `PM_ARCHITECTURE.md` v8 and `contract_inventory.py` at commit
`d6f1e79`. Date: 2026-08-20. Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA
theories, mechanisms and venues should be local plug-ins, while modules remain
independently testable and optimisable.

## Verdict

V8 closes several v7 findings in the canonical blocks: risk scenarios and
dynamic coupling are now fields of `DecisionProblem`; uncertainty draws and
adverse policy scenarios have distinct names; the venue-level rewards band is
removed; EV no longer feeds the runtime halt path; scenario scope, health
schemas and the ScenarioProvider registers are added.

It is not converged. The competition producer and consumer still disagree,
the risk-policy representation is neither knowledge-safe nor position-
independent, the kill path lacks the ports/ack semantics needed to execute it,
and the central decision axes remain closed lists rather than plug-in
protocols. The new inventory checker also provably misses the exact class of
type-narrowing regression it was introduced to catch.

At best, under the prior intended interpretation, the replay remains:

```
LOCAL 11 / SPREADING 0 / STRUCTURAL 2
target: 0 STRUCTURAL, <= 1 SPREADING
```

That score is optimistic because the literal closed solver/utility unions and
bare belief boundary undermine additional locality claims.

## MUST-FIX

### M8-1 — make the competition producer match the joint consumer

The canonical producer remains:

```
BE-Competition -> Known[Uncertain[CompetitionState]] | Unavailable
```

while `DecisionProblem` now requires:

```
Known[Uncertain[JointCompetitionState]] | Unavailable
```

No module owns the scalar-to-joint aggregation, so the stack cannot wire.
Either BE-Competition must directly produce the joint uncertain state or a
named, manifested aggregator must combine instrument-scoped marginals with an
explicit dependence contract.

`JointCompetitionState.dependence` is also placed inside each realised sample,
although dependence describes construction of the joint uncertainty rather
than a realised rival state. Keep the realised state as the scoped map and
attach the joint law to the distribution/aggregator. Restrict the map to node
kinds that can actually carry competition state.

### M8-2 — make adverse scenarios knowledge-safe, outcome-based and single-owner

Four problems remain in the canonical risk path:

1. `Declared(AdverseScenarioSet)` enters `DecisionProblem` without `Known[...]`,
   losing the bitemporal SP knowledge stamp required by R-KNOW.
2. `instrument_outcome_map` stores `ADVERSE|BENIGN`. Adversity depends on the
   current inventory and candidate action, so it cannot be stable declarative
   SP data. Store actual venue outcomes/token sides and calculate loss from the
   portfolio/action under that outcome.
3. `loss_limit` “MAY” live inside `AdverseScenario`, while
   `ScenarioLossLimit(ScenarioId)` also lives in `SP-Params`. This creates two
   owners and violates R-SSOT. Choose exactly one.
4. `Uncertain[AdverseScenarioSet]` is ambiguous: it appears to model
   uncertainty over sets rather than a joint outcome distribution. Define
   whether the estimator emits weighted joint outcomes, epistemic variants of
   a stress-policy set, or something else, and define incomplete-map handling.

Until these are fixed, change #7 remains STRUCTURAL.

### M8-3 — replace the unsound contract-inventory checker

The new checker does not inventory complete contracts. Executable audit found:

```
invalid baseline ref -> exit 0, DROPPED (0)
type:ScenarioDrawSet present? false
type:Uncertain present?        false
type:Known present?            false
field:command_id present?      false
field:retry_until_ack present? false
field:provenance present?      false
```

Most importantly, replacing the v8 joint competition field with the old bare
`competition: CompetitionState` produces an identical inventory. The checker
therefore misses the exact producer/consumer narrowing regression that caused
v6 to fail.

Root causes:

- `git show` return codes are ignored, so an invalid/missing baseline becomes
  an empty inventory and can pass;
- the type regex cannot parse generics such as `Known[V]` or
  `ScenarioDrawSet[T]`;
- fields are unqualified global names rather than owner-qualified paths, and
  fields without a colon are omitted;
- field types, signatures, variants, key structure and ownership are not
  compared;
- every fence is treated as canonical, so equations/module arrows become fake
  types, while inline contracts such as `Gate` are invisible;
- there are no regression tests or intentional-removal allowlist.

Use a machine-readable canonical schema, or parse owner-qualified complete
signatures with checked subprocess failures and tests for narrowing, moving,
renaming and deleting fields. The process guard cannot count toward convergence
until it catches the program's known regression classes.

### M8-4 — finish the executable halt/cancel contract

Runtime health is said to arrive through declared telemetry ports, but §9's
canonical port table declares neither module telemetry outputs nor an
OP-Monitor telemetry input. The fail-closed “unknown health” rule also needs
registered heartbeat/staleness deadlines; absence cannot emit a `HealthEvent`.

`cancel_all` is an untyped lowercase pseudo-record, causing even the new
checker to miss `command_id` and `retry_until_ack`. It requires acknowledgement
while §10 says acknowledgement is unobserved. Retry is unbounded and has no
backoff, deadline, terminal failure state, or interaction with the actuator's
rate budget.

Define `CancelAllCommand`/`CancelAllStatus`, the acknowledgement source,
idempotency key, bounded retry/backoff/rate policy, and the telemetry/halt port
manifests. If acknowledgement truly cannot be observed, define confirmation by
open-order reconciliation and the conservative state while confirmation is
unavailable.

### M8-5 — make utility/control genuinely open plug-in protocols

The core SOTA axes remain closed unions:

```
UtilityFunctional : RiskNeutral | CARA | PathFunctional
ControlSolver     : ClosedFormGLFT | PerLevel | HJBQVI
```

There is no behavioral protocol, implementation registry/factory,
configuration schema, or typed composition object. Adding an unlisted SOTA
theory therefore edits the shared canonical type and compatibility machinery.

`UnwrapPolicy` and `UnavailablePolicy` occur only in comments, not as fields on
a canonical `DecisionSchemeConfig`. R-COMPAT checks only pairs; a utility,
solver and coupling may be pairwise compatible but invalid as a triple.

Define open `UtilityFunctional` and `ControlSolver` interfaces with registered
plugin IDs/configuration and manifests. Add a canonical composition config
containing unwrap/unavailable policies, static coupling identity and n-ary
capability validation.

### M8-6 — make DecisionProblem a consistently point-in-time snapshot

Competition and dynamic coupling are `Known`, but `belief` remains a bare
`BeliefProcess` even though it is also fitted and time-varying. Constituent
timestamps do not give `p_hat`, link choice and aggregate estimator uncertainty
one enforceable knowledge time.

`ContractResolver` is a live handle rather than a resolved, spec-hash-pinned
map; unless it is explicitly snapshot-bound, contract values can change after
problem construction. `STATIC_INJECTED` likewise carries no graph reference,
hash or provenance into the decision/replay record.

Wrap the complete belief output, bind or pre-resolve contracts against the
problem's spec snapshot, and replace the sentinel with a typed static coupling
reference recorded in the run/decision provenance.

## SHOULD-FIX

1. `ScenarioDrawSet` uses `DrawId`, but the next comment still says
   `ScenarioId`.
2. The old statement that the records block was dropped v5 -> v6 remains in §3
   and contradicts the corrected v4 -> v5 lineage in §13.
3. `InstrumentId.venue_native_id` ambiguously permits a mutable slug, while
   `TokenId` omits Polymarket's native asset/token ID. Use immutable condition
   and token IDs; keep the slug as metadata.
4. Namespace `ParamId` by module/plugin, or add a strategy/module dimension, so
   parallel implementations can keep parameters at the same market scope
   without convention-based collisions.
5. The inventory command shown in the v7-to-v8 heading omits the script path
   when invoked from repository root.
6. Convert remaining behavior-bearing inline contracts—especially `Gate` and
   the DA/BE APIs—into the same machine-readable canonical source used by the
   inventory checker.

## 13-change acceptance replay

| # | change | v8 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL only under intended open-plugin reading; closed union remains M8-5 |
| 2 | fair-value source | LOCAL only after the belief snapshot boundary in M8-6 |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL only under intended open-plugin reading; closed union remains M8-5 |
| 5 | variance composition | LOCAL: `VarianceGroup` implementation/configuration |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | STRUCTURAL: declarative scenario type is not PIT-safe or position-independent; limit has two owners |
| 8 | pair representation / joint action | LOCAL through `CouplingGraph`, subject to typed static identity |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: scalar producer cannot supply joint competition consumer |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: `(ParamId, ScopeKey)` |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL or new implementations under manifested open registries |

## Keep unchanged

- risk scenarios and dynamic coupling as canonical `DecisionProblem` fields;
- separate Monte-Carlo draw and adverse-policy scenario names;
- scenario dimension in `ScopeKey`;
- instrument-scoped incentive facts and no venue-level rewards copy;
- EV excluded from runtime health;
- latched/fail-closed halt state and dual constraint/actuator edges;
- incentive cash flow into wealth, independent of utility preference;
- explicit uncertainty dependence and weighted draw identity;
- `(ParamId, ScopeKey)` identity and subset-order resolution;
- wealth-before-utility ordering and covariance-aware variance groups;
- cost-basis market risk without signed-loading cancellation;
- bitemporal specs, fitted-artifact guard and capability-sliced environment;
- field-level provenance, typed nullity, gate DAG and demand-driven build.
