# PM_STRUCT_ITER10_REVIEW — v10 structure acceptance review

Object: `PM_ARCHITECTURE.md` and `contracts/` at commit `728222b`. Date:
2026-08-20. Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories,
mechanisms and venues should be local plug-ins, while modules remain
independently testable and optimisable.

## Verdict

V10 materially expands the canonical source. It changes estimated risk support
from scenario sets to individual joint outcomes, introduces explicit risk-limit
and incentive contracts, adds static/dynamic coupling and per-input unwrap
types, pins outcome implementation identity, gives heartbeat observation a
separate type, names the cancel-command producer, and makes the checker compare
registries, validation hooks, notes, rule bodies and version monotonicity.

It is not converged. The claimed canonical source is still incomplete and not
reference-closed; the new invariant checks only record fields and accepts
malformed protocols, variants, module outputs, registry references and port
graphs. The risk and incentive types are present but remain disconnected from
an enforceable point-in-time module path. Several v10 fixes also introduce new
internal contradictions: the same dynamic coupling value appears in config and
per-decision state, the pinned outcome record cannot evaluate an action, and
the discriminated cancellation status loses the payload types and
reconciliation semantics it replaced.

At best, preserving the intended behavior that remains outside canonical
wiring, the replay remains:

```
LOCAL 11 / SPREADING 0 / STRUCTURAL 2
target: 0 STRUCTURAL, <= 1 SPREADING
```

Portfolio scenario risk (#7) and incentive theory (#9) therefore remain the two
structural rows.

## Executable audit

The expected checks pass:

```
contract_check.py --selftest
  PASS narrowing, deletion, rename, move and identical input

contract_check.py 365a698 728222b
  5 allowed removals, 20 allowed changes, 261 additions, exit 0

contract_check.py 728222b WORKTREE
  0 removed, 0 changed, 0 added, exit 0

invariants(contracts.yaml)
  [], exit 0
```

The target-side invariant was then tested against mutations outside its
field-only scan:

```
mutation                               invariant errors
undefined protocol return type         []
undefined module output                []
undefined variant payload type         []
remove PluginRegistry declaration      []
duplicate local/external type          []
```

The current target already contains examples of these accepted faults:

```
DA-Feeds.produces RawEvent              # RawEvent undefined
Hyperedge                               # both a local type and external import
DA-Feeds module ports                   # [feed, telemetry_out]
DA-Feeds global ports                   # [feed]
```

The v9-to-v10 diff uses 25 allowlist entries, while eight additional entries do
not correspond to any actual removal/change and remain latent future bypasses.

## MUST-FIX

### M10-1 — finish the canonical surface and enforce reference closure

V10 adds many previously missing types, but the source still does not satisfy
its claim to define every type, module, port and rule:

- R-ONCE names `WealthLedger` and `VarianceGroup`, but neither type exists.
- The architecture names roughly thirty SP/DA/BE/DE/EV/OP modules; the canonical
  registry has seven and no SP, EV, allocator, belief or outcome-evaluator
  module.
- `ModuleManifest` is defined, but no module publishes or references one.
- `RawEvent` is produced without a type/import.
- `Hyperedge` is both defined locally and declared external, violating the new
  R-SSOT body.
- The obsolete `Heartbeat` record remains alongside
  `HeartbeatRegistration` and `HeartbeatPulse`, even though the allowlist
  claims it was removed.
- Module-local and top-level port declarations disagree for `DA-Feeds`.

Reference closure must cover fields, methods, protocol signatures, variant
payloads, generic bounds, module inputs/outputs/requires, registry identifiers,
rule validators and ports. Imported types need a source/version contract, not
only a name in `prelude.external`.

### M10-2 — turn target invariants and rule bodies into executable checks

`invariants()` scans capitalized references only inside `types.*.fields`.
It does not inspect protocols, methods, variants, modules, registries or ports.
Its producer check hardcodes exceptions and ignores qualified module-output
references. Consequently the current malformed target and every adversarial
mutation above pass with no errors.

The new rule bodies improve reviewability, but they are prose strings.
`R-SSOT.checks.loss_limit_owner` is flattened for diffing and never executed;
the same is true of R-KNOW, R-ONCE, R-PROV, R-PORT, R-COMPAT and R-OPEN. Define
validator IDs with implementations and fail if a declared rule has no
registered validator.

The allowlist remains keyed only by flattened path and loaded from the
worktree. An entry permanently permits unrelated future changes at that path
and can retroactively affect explicit-ref comparisons. Eight entries are
already stale, including `type:Heartbeat` and three of its fields. Require
every exception to be consumed exactly once and bind it to operation, exact
normalized old/new values, base/target version and rationale. Load it from the
target ref.

Also use strict duplicate-key YAML loading and add self-tests for every v10
invariant, version regression, order normalization, stale allowlists and rule
dispatch. The current self-test still covers only the original four diff cases.

### M10-3 — connect scenario loss to a typed, PIT-safe constraint path

Changing `JointOutcomeDistribution.support` to weighted `JointOutcome`
values is correct. The declared path still cannot enforce the advertised hard
cap:

1. `JointOutcome` has no completeness or `on_incomplete` policy, so estimated
   distributions lost the incomplete-map behavior promised by
   `RiskScenarios`.
2. `ScenarioLossLimit` contains only `ParamId` and `ScopeKey`; it has no
   resolved value, currency/unit, knowledge stamp or immutable spec snapshot.
3. `SP-Params.entries` stores `ParamValue.value: any`. Its
   `valid_for: ScopeKey` duplicates the scope already in the dictionary key
   and does not supply the validity interval the name previously represented.
4. No canonical resolver maps the limit reference to
   `Known[Money] | Unavailable` under the decision's `spec_snapshot`.
5. `ScenarioLossConstraint` is not a member/capability of `ConstraintSet`,
   is not listed as a DE-Constraints input, and the named `DE-Allocator`
   consumer does not exist in the module registry.

Define a resolved, unit-bearing scenario cap in the immutable
`DecisionProblem`, or define a spec-snapshot-bound resolver and make both
consumers explicit. Validate outcome-map completeness before evaluating the
loss. Until then change #7 remains STRUCTURAL.

### M10-4 — wire the incentive plug-in from selection through cash flow

`IncentiveModel` is now named, but nothing selects or consumes it.
`DecisionSchemeConfig` has utility and solver refs but no incentive-model ref;
there is no canonical SP-Strategy/composition binding, resolver module or
module manifest connecting its four contributions to BE-Competition,
DE-Constraints and `ActionOutcome.cash_flows`.

Additional boundary errors remain:

- `PluginRegistry` is an opaque external name; registry entry, factory,
  capability and config-validation behavior are still undefined.
- `PluginRef.config: JsonSchema` stores a schema where a configuration
  instance/value should live. The registry entry should own `config_schema`.
- `competition_model: () -> ModuleRef(BE-Competition)` is not a valid use of
  the non-generic `ModuleRef` type and hard-codes one module in a protocol
  signature.
- `PayoutFunctional.cash` consumes scalar `CompetitionState`, while the
  decision boundary carries joint uncertain competition. It needs an explicit
  instrument projection or the joint wrapper.
- `ResolvedContracts` pins a spec hash but not the selected incentive plug-in
  and version that produced the contracts.

Make the selected incentive extension part of canonical strategy config and
validate/wire all four declared capabilities. Until then change #9 remains
STRUCTURAL.

### M10-5 — separate static configuration from per-decision resolved state

`CouplingSource = Static(ref) | Dynamic(Known[CouplingGraph])` is used in both
`DecisionSchemeConfig` and `DecisionProblem`. A PER_DECISION graph must
therefore be stored in the supposedly static config as well as each problem,
or the config must mutate per decision. Use:

```
CouplingMode     = Static(StaticCouplingRef) | Dynamic
ResolvedCoupling = Static(StaticCouplingRef) | Dynamic(Known[CouplingGraph])
```

The outcome replacement also drops behavior: `PinnedOutcomes` records a
plug-in/artifact/spec hash but defines no evaluate method or outcome-model
protocol, so `DecisionProblem.outcomes` is no longer callable. Its
`PluginRef.artifact_id?` and mandatory `PinnedOutcomes.artifact_id` are two
unconstrained copies of the same identity.

Finally, `unavailable_policy` remains one global enum despite multiple
independently unavailable inputs, and its optional fallback is not conditionally
validated. `UnwrapPolicy` has per-input modes but no quantile level, scenario
count or shared-draw contract. Define typed per-input policies with their
required parameters and validation.

### M10-6 — preserve typed cancel semantics and make port SSOT real

Naming OP-Monitor as the `CancelAllCommand` producer and adding pulse,
registration and command ports closes the main v9 graph gap. Three regressions
remain:

1. The obsolete `Heartbeat` record was not removed, contradicting its
   allowlist entry and creating two liveness configuration types.
2. `CancelAllStatus` variants contain untyped payload names. The change removes
   the former `ImmutableId`, confirmation-source enum and optional integer
   types without replacing them in structured variants. The canonical note
   that venue ack is unavailable, reconciliation confirms cancellation, and
   UNCONFIRMED stays HALTED was also deleted.
3. `DA-Feeds` declares `telemetry_out` in its module record but not in the
   top-level port manifest. The checker compares both independently and does
   not enforce equality or producer/consumer port closure.

Represent union variants with owner-qualified typed payload fields, retain the
reconciliation/fail-closed invariant as an executable rule, remove the old
heartbeat type, and choose one canonical port-manifest representation.

## SHOULD-FIX

1. Generate the markdown excerpts from YAML or check them in CI. V10 changes
   only seven architecture lines, leaving stale contracts for identity,
   `draws()`, compatibility, coupling, pinned outcomes, incentives, scenarios,
   cancellation and ports.
2. Remove the duplicated “Earlier/Prior” score line in the v10 header.
3. Replace `JointCompetitionState.by_node: dict[NodeId,...]` plus an ignored
   eligibility note with a typed competition-capable node ID.
4. Specify how the competition aggregator composes knowledge timestamps,
   provenance/error, staleness and partial unavailability.
5. Replace behavior-bearing `str` and `any` fields with typed references,
   units and discriminated values before implementations rely on them.

## 13-change acceptance replay

| # | change | v10 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL under intended open solver protocol; executable registry/manifest remains M10-1/M10-4 |
| 2 | fair-value source | LOCAL: knowledge-stamped belief implementation |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL under intended open solver protocol |
| 5 | variance composition | LOCAL by intended `VarianceGroup`; canonical type is still absent |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | STRUCTURAL: limit is unresolved/untyped and its constraint/allocator path is unwired |
| 8 | pair representation / joint action | LOCAL through `CouplingGraph`, subject to config/problem separation |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: `IncentiveModel` exists but has no selection or module/cash-flow wiring |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: namespaced `(ParamId, ScopeKey)` |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL under intended manifest registry; canonical module coverage remains incomplete |

## Keep unchanged

- individual `JointOutcome` support points and explicit dependence;
- declared scenarios retaining their `Known` envelope;
- outcome-based adversity and decision-time loss functional;
- named `ScenarioLossLimit`, `ScenarioLossConstraint` and sole SP-Params
  storage intent;
- named open `IncentiveModel` with four contribution seams;
- open utility/control protocols and n-ary compatibility intent;
- immutable condition/token IDs, namespaced params and scenario scope;
- knowledge-wrapped belief/competition and resolved incentive values;
- explicit static/dynamic coupling variants and pinned outcome identity;
- heartbeat registration/pulse separation, observable last-seen state and
  OP-owned cancellation command;
- discriminated cancellation-state intent and fail-closed halt behavior;
- field provenance, typed nullity, bitemporal specs, fitted-artifact guard,
  gate DAG, capability-sliced environment and demand-driven build.
