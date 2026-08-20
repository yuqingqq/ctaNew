# PM_STRUCT_ITER7_REVIEW — v7 structure acceptance review

Object: `PM_ARCHITECTURE.md` v7 at commit `c636500`. Date: 2026-08-20.
Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories, mechanisms and venues
should be local plug-ins, while modules remain independently testable and
optimisable.

## Verdict

v7 closes important parts of the v6 review. The complete knowledge/uncertainty
wrapper now reaches `DecisionProblem`; incentive cash flows enter
`ActionOutcome.cash_flows` and `WealthLedger` rather than utility; the halt
state reaches both constraints and the sole venue-writing actuator; coupling
has a typed hypergraph; and uncertainty exposes scenario identity and weights.

It is not converged. Several fixes exist only in explanatory prose rather than
the canonical types, the two meanings of `ScenarioSet` conflict, the restored
spec block reintroduces duplicate rewards ownership, and the OP source list
violates the dependency invariant. Strict replay score remains:

```
LOCAL 11 / SPREADING 0 / STRUCTURAL 2
target: 0 STRUCTURAL, <= 1 SPREADING
```

Scenario risk and portfolio incentive/competition still require interface
changes. The dynamic coupling omission is an additional contract failure, but
the historical pair-representation replay remains local through a statically
injected graph.

## MUST-FIX

### M7-1 — put risk scenarios in the canonical DecisionProblem

§8 declares:

```
DecisionProblem.scenarios: Known[Uncertain[ScenarioSet]] | Unavailable
```

but the actual `DecisionProblem{...}` schema in §6 has no scenarios field. The
default declarative SP scenario path also has no resolved type at that boundary;
only the optional estimated-provider form is shown. A prose field below the
canonical type cannot be validated by wiring or a `ModuleManifest`.

Add a typed risk-scenario field directly to `DecisionProblem`, covering both
resolved declarative policy scenarios and knowledge-stamped estimated
scenarios, with an explicit unavailable policy.

### M7-2 — separate uncertainty draws from adverse-risk scenarios

`ScenarioSet` has two incompatible meanings:

```
ScenarioSet{ draws: [(ScenarioId, weight, value)], common_random_id }
ScenarioSet[s].losers
```

The first is the generic result of `Uncertain[T].scenarios()`; the second is
treated as an adverse-scenario map, yet the declared `AdverseScenario` contains
`instrument_outcome_map`, not `losers`. `BE-ScenarioProvider ->`
`Uncertain[ScenarioSet]` then nests the ambiguous container inside itself.

Use separate contracts such as `ScenarioDrawSet[T]` and
`AdverseScenarioSet`. Define portfolio loss from the adverse scenario's
instrument/outcome map. Also make `L_max[s]` keyable: either add `ScenarioId`
to `ScopeKey`, use a typed `ScenarioLossLimit(ScenarioId)` parameter identity,
or store the limit in the adverse-policy record. As written, multiple
scenario limits at one scope are not uniquely addressed.

### M7-3 — make dynamic coupling a real decision input

V7 says a `PER_DECISION` `CouplingGraph` is carried in `DecisionProblem`, but
the canonical schema has no coupling field. A graph driven by market discovery
or resolution must also be knowledge-time safe; raw topology can otherwise
encode future membership.

Add `coupling: Known[CouplingGraph]` (or an equivalent view-bound type) to the
problem for `PER_DECISION`, while retaining constructor injection for `STATIC`.
The two modes and their manifest requirements must be mutually validated.

### M7-4 — type the portfolio competition and contract-resolution boundary

The text says competition is “joint over the `CouplingGraph`”, but the actual
type remains one `Known[Uncertain[CompetitionState]]` whose fields are scalar.
That cannot represent different rival scores, participation or eligibility
states across simultaneously coupled instruments.

Define a keyed/joint competition state with its dependence structure preserved
inside the uncertainty object. `ContractResolver` must also return
`Known[IncentiveContract] | Unavailable` and invoke the declared disputed-field
handler. Missing or disputed rewards facts are live states today, not
impossible branches. Until these types are real, change #9 remains STRUCTURAL.

### M7-5 — restore dependency and rewards SSOT invariants

Two regressions were introduced while closing v6:

1. §1 says EV is read by none, but `HealthEvent` lists EV as a source and OP
   then commands DE. This creates `EV -> OP -> DE`, allowing evaluation state
   to affect decisions. Runtime health must come from declared module telemetry
   ports. `EV-Gates.on_fail = HALT_PROGRAM` remains the separate programme-
   control path.
2. Rewards are explicitly instrument-scoped and time-varying, yet the restored
   records put `rewards_band` in `SP-Venue` while `SP-Instrument` owns
   `incentive_contract`. This duplicates the same fact and violates R-SSOT.
   Keep only a venue capability/default if needed; resolved band/rate/
   eligibility belong to the instrument contract.

## SHOULD-FIX

1. The inventory lineage is wrong: the records block was present at
   `b5b7968` and absent by v5 (`5f80dae`), so it was dropped v4 -> v5, not
   v5 -> v6. The v6 -> v7 table also says no contracts changed despite renamed
   or replaced fields. Identifier extraction is insufficient because old
   identifiers remain in explanatory prose; inventory the canonical schemas,
   keys, fields and rules instead.
2. Add `SP-Scenarios` and `BE-ScenarioProvider` to the layer and implementation-
   status registers. The detailed section names them, but the canonical module
   lists do not.
3. Define `HealthEvent` and `HaltState` schemas, reset authority, monotonic/
   latched transition rules and cancel acknowledgement/idempotency. A label and
   arrow are not enough to test the kill path independently.

## 13-change acceptance replay

| # | change | v7 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL: `ControlSolver` |
| 2 | fair-value source | LOCAL: belief implementation |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL: `ControlSolver` |
| 5 | variance composition | LOCAL: `VarianceGroup` implementation/configuration |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | STRUCTURAL: risk scenario types conflict and the canonical problem omits them; SPREADING after M7-1/M7-2 |
| 8 | pair representation / joint action | LOCAL through a static `CouplingGraph`; dynamic mode still needs M7-3 |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: portfolio competition and unavailable contract resolution are not typed |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: `(ParamId, ScopeKey)` |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL or new implementations |

## Keep unchanged

- full knowledge/uncertainty wrapper at the decision boundary;
- incentive cash flow into wealth, independent of utility preference;
- typed hierarchical coupling graph and static constructor injection;
- explicit uncertainty dependence, scenario IDs and weights;
- latched/fail-closed halt semantics and dual constraint/actuator edges;
- `(ParamId, ScopeKey)` identity and subset-order resolution;
- wealth-before-utility ordering and covariance-aware variance groups;
- cost-basis market risk without signed-loading cancellation;
- bitemporal specs, knowledge-truncated state and fitted-artifact guard;
- capability-sliced environment, generic action evaluation and port manifests;
- field-level provenance, typed nullity and disputed-field handlers;
- gate DAG, STOP ownership, implementation status and demand-driven build.
