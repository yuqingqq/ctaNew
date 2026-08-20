# PM_STRUCT_ITER6_REVIEW — v6 structure acceptance review

Object: `PM_ARCHITECTURE.md` v6 at commit `bf685a1`. Date: 2026-08-20.
Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories, mechanisms and venues
should be local plug-ins, while modules remain independently testable and
optimisable.

## Verdict

v6 materially improves the structure. It restores `ParamId`, replaces the
single-choice coupling enum with a hierarchical graph, separates additive
wealth from terminal utility, adds covariance and empirical falsification to
variance composition, requires dependence when combining uncertainty, adds
knowledge-time and fitted-artifact requirements to competition, packages an
incentive theory as one registration, replaces signed-factor loss netting with
adverse scenarios, and sketches the OP health/halt path. The seven v5
SHOULD-FIX items also landed in substance.

It is not converged yet. The new contracts are not carried consistently across
their consumer boundaries, and the adverse-scenario and halt paths do not yet
have executable ownership. Strict score as written:

```
LOCAL 11 / SPREADING 0 / STRUCTURAL 2
target: 0 STRUCTURAL, <= 1 SPREADING
```

After the MUST-FIXes, the intended score is approximately
`LOCAL 12 / SPREADING 1 / STRUCTURAL 0`: portfolio scenario risk remains the
one legitimate two-module change (`DE-Constraints` + `DE-Allocator`).

## MUST-FIX

### M6-1 — preserve competition knowledge and uncertainty at the decision boundary

`BE-Competition` now correctly returns:

```
Known[Uncertain[CompetitionState]] | Unavailable
```

but `DecisionProblem` immediately narrows that to bare `CompetitionState`.
This discards `t_known`, provenance, estimator uncertainty and unavailability
at the exact boundary where the document says they are required. It also
leaves the solver's unavailable policy undefined.

The `DecisionProblem` field must preserve the complete type (or accept an
equivalent typed, scoped joint representation). Unwrapping to an expectation,
quantile or scenario must be an explicit decision policy, not composition-root
coercion.

### M6-2 — incentive cash flow must feed wealth, not implement utility

The incentive registration currently declares:

```
cashflow_functional -> UtilityFunctional
```

This reconnects incentive theory to risk preference immediately after §7
separates the two. An incentive model should emit realised or uncertain cash
flows into `ActionOutcome.cash_flows` / `WealthLedger`; the independently
selected `UtilityFunctional` then evaluates the distribution of total terminal
wealth.

The scope is inconsistent as well. §3 correctly says rewards are instrument-
scoped and time-varying, but the registration points `contract_spec` at
`SP-Venue`, while a portfolio-coupled `DecisionProblem` carries one unscoped
`IncentiveContract` and one competition state. Use an `InstrumentId`-keyed
resolver/map or a typed joint state over the `CouplingGraph`, with bitemporal
spec resolution preserved.

Until both problems are fixed, change #9 still changes or leaks across the
decision interface and remains STRUCTURAL.

### M6-3 — give adverse scenarios a typed owner and consumer path

The risk cap now depends on `losers(s)`, `L_max(s)` and an unnamed "scenario
model". None is present in the shared algebra, module register, manifests or
`DecisionProblem`; scope, provenance, completeness, knowledge time and
dependence are therefore unenforceable. Under the loop rubric, an ownerless
quantity is STRUCTURAL.

Define either:

- declarative policy stress scenarios in SP, bitemporally versioned and scoped;
  or
- a knowledge-stamped scenario provider with an explicit joint law when the
  scenarios are estimated.

Give scenarios stable IDs and an instrument/outcome map, key `L_max` through
`SP-Params`, and name `DE-Constraints`/`DE-Allocator` as consumers. The cap can
then evaluate scenario PnL without signed-loading cancellation.

### M6-4 — make the OP halt and cancel paths executable

The new diagram ends at `DE-Constraints`, but §9 gives DE modules only
`StateView`, RNG and artifact ports. No declared port carries `HaltState` to
the constraint module. The prose also says cancel-all goes through
`DE-Actuator` without declaring a command edge to it.

A hard constraint can prevent new risk; it does not by itself cancel resting
orders, especially when the regular solver is unavailable during the fault.
Declare the health-event owner, a latched/fail-closed `HaltState` port, and a
priority `HaltState -> DE-Actuator.cancel_all` command. The actuator remains the
sole venue writer, so this does not violate the single-write-path invariant.

## SHOULD-FIX

1. `CouplingGraph` is currently a name plus an example. Give it a typed schema
   (node/hyperedge identity, relation kind, version/provenance and update
   semantics) and state whether it is injected when constructing
   `DecisionScheme` or carried in `DecisionProblem`.
2. `Uncertain.combine` requires dependence, but `scenarios()` does not expose
   scenario identity or weight at the type boundary. Define a typed
   `ScenarioSet`/`Dependence` representation so shared identity can actually be
   validated rather than trusted in prose.
3. §13 requires a full contract-inventory/removal log, but v6 does not include
   one. Add the v5 -> v6 inventory or link an auditable artifact. Also rename
   the stale `objective ledger` reference in R-ONCE to `WealthLedger`.

## 13-change acceptance replay

| # | change | v6 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL: `ControlSolver` |
| 2 | fair-value source | LOCAL: belief implementation |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL: `ControlSolver` |
| 5 | variance composition | LOCAL: `VarianceGroup` implementation/configuration |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | STRUCTURAL as written: adverse-scenario model has no typed owner; SPREADING after M6-3 (`Constraints` + `Allocator`) |
| 8 | pair representation / joint action | LOCAL under the stated `CouplingGraph` composition; make injection explicit per SHOULD-FIX 1 |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: competition wrapper is discarded and incentive cash flow/scope cross boundaries |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: `(ParamId, ScopeKey)` is restored |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL or new implementations |

## Keep unchanged

- `(ParamId, ScopeKey)` parameter identity and subset-order resolution;
- utility/control/coupling separation and nested pair/portfolio coupling;
- wealth-before-utility ordering and the three R-ONCE registries;
- covariance-aware variance groups with empirical validation;
- explicit dependence for uncertainty composition;
- cost-basis market risk and no signed-loading cancellation;
- bitemporal specs, knowledge-truncated state and fitted-artifact guard;
- capability-sliced environment, generic action evaluation and port manifests;
- field-level provenance, typed nullity and declared disputed-field handlers;
- gate DAG, STOP ownership, implementation-status table and demand-driven build.
