# PM_STRUCT_ITER11_REVIEW — v11 structure acceptance review

Object: `contracts/` at commits `d5261e8` and `d70ff3c`, based on the v10
review commit `1c9e50c`. Date: 2026-08-20. Charter:
`PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories, mechanisms and venues should be
local plug-ins, while modules remain independently testable and optimisable.

## Verdict

V11 fixes many concrete v10 defects. It removes the obsolete heartbeat and
duplicate Hyperedge import, types cancellation variant payloads, restores
reconciliation/fail-closed semantics, separates static coupling configuration
from per-decision resolution, makes pinned outcomes callable with one artifact
identity, adds parameterised unwrap and per-input unavailable policies, selects
and pins the incentive plug-in, adds estimated-outcome completeness, resolves
scenario limits to typed money, wires constraints and allocation, restores the
wealth/variance registries, and broadens reference checks and self-tests.

It is not converged. The migration allowlist still masks arbitrary changes by
path and can pass the exact wrapped-to-bare narrowing this process exists to
prevent. The new “executable” rules are not executed, runtime producers can be
removed because their outputs are also declared `config_supplied`, and no port
or cross-snapshot invariants are checked. Scenario risk now has consumers but
the types disagree and duplicate the same constraint. Incentive selection is
present, but its competition/payout contributions never reach the outcome
model, so the mechanism extension remains structurally unwired.

The historical replay improves provisionally to:

```
LOCAL 11 / SPREADING 1 / STRUCTURAL 1
target: 0 STRUCTURAL, <= 1 SPREADING
```

Portfolio scenario risk (#7) moves from STRUCTURAL to SPREADING because the
owners and consumers now exist but need coordinated contract corrections.
Incentive theory (#9) remains STRUCTURAL.

## Executable audit

The expanded built-in tests and intended migration pass:

```
contract_check.py --selftest
  PASS  original four diff regressions
  PASS  undefined protocol/module/variant references
  PASS  duplicate local/external type
  PASS  empty rule
  PASS  identical input

contract_check.py 1c9e50c d70ff3c
  21 allowed removals, 9 allowed changes, 79 additions, exit 0
```

However, an unchanged current target fails because every historical migration
entry is considered stale:

```
contract_check.py d70ff3c WORKTREE
  STALE ALLOWLIST ENTRIES (30), exit 1
```

More importantly, changing the already-allowlisted
`DecisionProblem.coupling` from the intended `ResolvedCoupling` to bare
`CompetitionState` gives:

```
diff:       CouplingSource -> CompetitionState
allowlist:  accepted by field:DecisionProblem.coupling
invariants: []
unexplained changes: []
would fail: false
```

This reproduces the core “allowlisted path hides a different narrowing” failure
class with the current checker.

Other target mutations also pass `invariants()`:

```
false R-HALT semantics                         []
remove OP HaltState/CancelAllCommand producer  []
give BE-* a venue escape port                  []
undefined config_supplied type                 []
narrow PinnedOutcomes.evaluate behavior        []
```

PyYAML duplicate keys are still silently overwritten rather than rejected.

## MUST-FIX

### M11-1 — bind migration exceptions to exact versions and values

The stale-entry check is useful but does not repair the underlying allowlist
model. Exceptions remain `path -> reason`, loaded from the worktree. An
exception for one migration therefore approves every other change/removal at
the same path. The executable coupling mutation above passes even though its
new value has nothing to do with the documented migration.

Represent each exception as a one-shot migration:

```
from_version: 10
to_version: 11
operation: change
key: field:DecisionProblem.coupling
old: CouplingSource
new: ResolvedCoupling
reason: ...
```

Require exact normalized old/new values, operation and adjacent versions; load
the migration record from the target ref. Fail unused records only for their
declared version transition, and fail duplicate/conflicting migration records.
Then `HEAD -> WORKTREE` with no contract change can pass instead of treating
all prior transitions as stale.

Add a self-test proving that an exception for A→B rejects A→C. Without that
test, the checker still misses the exact type-narrowing regression at its
primary safety boundary.

### M11-2 — execute rule, producer and port invariants

The reference scanner now covers fields, methods, protocols, variants, module
signatures and registries. That is meaningful progress. Its semantic checks
remain labels rather than enforcement:

- a rule passes when it has any prose `body` or arbitrary `checks` string;
  no validator registry dispatches `loss_limit_owner` or `fail_closed`;
- `config_supplied` contains `HeartbeatPulse`, `HaltState` and
  `CancelAllCommand` even though they are runtime outputs, so removing their
  DA/OP producers passes;
- `config_supplied` entries themselves are not reference-checked;
- changing BE/DE wildcard ports to expose `venue` passes because ports are not
  validated against R-PORT or module capabilities;
- producer matching still uses string-base extraction rather than a parsed type
  AST;
- `ports._representation` mixes a metadata string into a map otherwise
  containing port lists; `flatten()` sorts that string character-by-character;
- duplicate YAML keys are accepted by `yaml.safe_load`.

Give each rule check a registered validator implementation and fail any
undeclared/unimplemented validator. Separate composition inputs from runtime
outputs, and require every non-config consumed value to have exactly the
intended producer. Parse type expressions once and use the AST for reference,
producer and compatibility checks. Validate wildcard expansion, module port
inheritance and forbidden capabilities. Use a strict duplicate-key loader.

Self-tests should include false rule semantics, removed runtime producers,
undefined config inputs, forbidden venue ports, malformed port metadata and
duplicate YAML keys.

### M11-3 — make the scenario-risk path type-consistent and single-copy

V11 correctly adds estimated-map completeness, `Known[Money]` resolution,
ConstraintSet membership, explicit DE-Constraints input and an allocator with
settlement facts. Four boundary mismatches remain:

1. `LossFunctional.loss` returns `Wealth`, while the resolved cap is
   `Money`. The hard comparison is not type-valid without an explicit
   valuation/currency conversion contract.
2. `ScenarioLossConstraint` is both inside
   `DecisionProblem.constraints.members` and a separate DE-Constraints input.
   DE-Allocator consumes another separate copy. No ID/hash/equality invariant
   prevents different limits or loss functions at the same decision.
3. `FeasibleSet.binding` is `list[Constraint]`, so a binding
   `ScenarioLossConstraint` cannot be represented even though ConstraintSet
   accepts it.
4. `ScenarioLossLimit.unit: str` duplicates `Money.currency`, while the
   backing `ParamValue.value: any` and `valid_for: ScopeKey` remain untyped
   and duplicate the store key's scope instead of expressing temporal validity.

Use one typed constraint identity referenced by the problem, constraint engine
and allocator; do not inject copies. Return a unit-bearing loss type compatible
with the limit, carry conversion explicitly if needed, and include the scenario
constraint in binding/dual output. Give parameter values validity intervals and
typed schemas. Until these coordinated fixes land, change #7 is SPREADING.

### M11-4 — connect the selected incentive extension to outcome cash flow

Adding `DecisionSchemeConfig.incentive_model`,
`ResolvedContracts.incentive_plugin`, the joint competition payout signature
and a non-generic `ModuleRef` closes important v10 gaps. The four
`IncentiveModel` contributions still terminate at the protocol:

- no module consumes `IncentiveModel` or its selected `PluginRef`;
- no resolver/composition contract proves that `contract_spec`, constraints
  and competition module all come from that selected version;
- `PinnedOutcomes.evaluate(ActionSet, StateView, SelfState, Duration)` receives
  neither `ResolvedContracts` nor joint competition, so it cannot call
  `PayoutFunctional.cash` and emit the promised incentive cash flow;
- no outcome-model capability requires incentive cash-flow integration;
- `ModuleRef` carries only a module ID, not the required competition
  capability or manifest.

There is also still no canonical plug-in registry/factory contract.
`PluginRegistry` is an opaque external name, no module uses
`ModuleManifest`, and `PluginRef.config: JsonSchema` stores a schema where a
validated configuration value should live. Registry entries should own the
schema; refs should carry a config instance and a typed protocol/registry ID.

Define an immutable `DecisionContext` or outcome-evaluation input containing
belief, joint competition, resolved contracts, portfolio and spec snapshot.
Wire the selected incentive extension's payout emitter into the outcome model
through a required capability, and validate plugin/version identity across
selection, resolution and evaluation. Until then change #9 remains STRUCTURAL.

### M11-5 — enforce one coherent decision snapshot and complete module manifests

Several paired fields describe the same decision but have no equality or
compatibility invariant:

- `DecisionSchemeConfig.coupling_mode` versus
  `DecisionProblem.coupling`;
- selected `incentive_model` versus
  `ResolvedContracts.incentive_plugin`;
- `DecisionProblem.spec_snapshot` versus
  `ResolvedContracts.spec_hash` and `PinnedOutcomes.spec_hash`;
- Unwrap/Unavailable policy keys versus the actual uncertain/unavailable
  DecisionProblem inputs;
- static coupling refs versus dynamic graph update semantics.

A mismatched combination is reference-valid and passes the gate. Define a
canonical problem-construction validator that checks these equalities,
knowledge-time cutoffs, required policy coverage, plugin capabilities and
n-ary compatibility before a solver can receive the problem.

The canonical module registry now has nine modules, but still omits the belief,
flow/outcome, solver, incentive resolver and most SP/EV contracts needed for
that construction. `ModuleManifest` remains unused by every registered
module. Runtime types duplicated in `config_supplied` also contradict R-SSOT.
Complete the minimum end-to-end path and make each module publish one actual
manifest rather than maintaining parallel ad-hoc fields.

## SHOULD-FIX

1. Update `PM_ARCHITECTURE.md`: it is still titled v10 and retains stale
   excerpts for identity, draws, compatibility, coupling, outcomes, incentives,
   scenarios, cancellation and ports. Generate excerpts from YAML or enforce
   consistency in CI.
2. Move `ports._representation` to schema metadata rather than storing a string
   among port lists.
3. Replace `JointCompetitionState.by_node: dict[NodeId,...]` plus a note with a
   typed competition-capable node ID.
4. Specify competition-aggregator composition of knowledge timestamps,
   provenance/error, staleness and partial unavailability.
5. Replace behavior-bearing `str`/`any` fields with typed IDs, units, schemas
   and discriminated values before implementations rely on them.

## 13-change acceptance replay

| # | change | v11 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL under intended open solver protocol; registry enforcement remains M11-4 |
| 2 | fair-value source | LOCAL: knowledge-stamped belief implementation |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL under intended open solver protocol |
| 5 | variance composition | LOCAL: canonical `VarianceGroup` now exists |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | SPREADING: owners/consumers exist, but loss/limit/binding types and duplicated constraint inputs require coordinated correction |
| 8 | pair representation / joint action | LOCAL through separated `CouplingMode` and `ResolvedCoupling` |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: selected extension has no path into outcome competition/payout cash flow |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: namespaced `(ParamId, ScopeKey)` |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL under intended module registry; manifest enforcement remains incomplete |

## Keep unchanged

- weighted individual joint outcomes with completeness handling;
- knowledge-wrapped declared/estimated risk inputs;
- typed resolved scenario money and explicit loss/constraint/allocator owners;
- named settlement facts and capital budget;
- selected/pinned incentive plug-in and joint competition payout signature;
- separated static `CouplingMode` and per-decision `ResolvedCoupling`;
- callable pinned outcome protocol with one plug-in/artifact identity;
- parameterised per-input unwrap and unavailable policies;
- open utility/control/incentive/outcome protocols;
- typed cancellation variants, reconciliation evidence and fail-closed intent;
- heartbeat registration/pulse separation and OP-owned cancel command;
- restored WealthLedger/VarianceGroup/RawEvent contracts;
- expanded reference checks, invariant self-tests and stale-allowlist detection;
- immutable market/token identity, namespaced params, field provenance, typed
  nullity, bitemporal specs, fitted-artifact guard, gate DAG, capability-sliced
  environment and demand-driven build.
