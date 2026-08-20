# PM_STRUCT_ITER9_REVIEW — v9 structure acceptance review

Object: `PM_ARCHITECTURE.md` and `contracts/` at commit `8a7cb7e`. Date:
2026-08-20. Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories,
mechanisms and venues should be local plug-ins, while modules remain
independently testable and optimisable.

## Verdict

V9 makes real progress. It names the missing competition aggregator, wraps the
belief and declared risk policy in knowledge-time envelopes, replaces mutable
market identity with immutable condition/token IDs, opens the utility and
solver interfaces, snapshots resolved incentive contracts, and defines typed
halt/cancel records. The new checker also catches owner-qualified field
narrowing, deletion, rename and movement, and fails on an invalid Git ref.

It is not converged. The YAML claims to define every contract but contains only
a selected subset and leaves central referenced types, extension seams, modules
and rule bodies undefined. The checker does not inspect several semantic
properties introduced specifically to close v8 findings. The estimated risk
type is still a distribution over scenario sets, the sole loss-limit owner is
only an unchecked note, the incentive extension is absent from the canonical
source, and the halt-to-cancel path still has no command producer.

At best, preserving the intended reading of contracts that remain only in the
explanatory markdown, the replay remains:

```
LOCAL 11 / SPREADING 0 / STRUCTURAL 2
target: 0 STRUCTURAL, <= 1 SPREADING
```

The two structural rows remain portfolio scenario risk (#7) and incentive
theory (#9), although the immediate scalar-to-joint competition mismatch from
v8 is fixed.

## Executable audit

The advertised behavior passes:

```
contract_check.py --selftest
  PASS  detects narrowing (v6/v8 blind spot)
  PASS  detects deletion
  PASS  detects rename
  PASS  detects move to another owner
  PASS  no false positive on identical input

contract_check.py 8a7cb7e WORKTREE
  REMOVED (0), TYPE-CHANGED (0), ADDED (0), exit 0

contract_check.py definitely-not-a-ref WORKTREE
  FATAL: cannot read contracts.yaml at ref, exit 1
```

Adversarial mutations outside the self-test expose remaining blind spots:

```
mutation                         removed  changed  added
risk-owner note changed               0        0      0
n-ary validation -> pairwise          0        0      0
plugin registry deleted               0        0      0
version regressed                      0        0      0
new structural live-handle field       0        0      1   # non-fatal
```

Reordering a semantically identical `requires` map produces a false type
change because nested mappings are compared using Python `str()`.

## MUST-FIX

### M9-1 — make the canonical source complete and reference-closed

The document says `contracts.yaml` defines every type, field, module, port and
rule. It does not. Contracts existing only in markdown include `SpecRecord`,
`FieldState`, `SP-Params`, `ParamValue`, `ActionOutcome`, `Constraint`,
`ModuleManifest`, `CouplingGraph` and `IncentiveModel`, plus most of the
SP/DA/BE/DE/EV/OP module set and their manifests.

The YAML references core types without defining or importing them, including
`PluginRef`, `PluginRegistry`, `JsonSchema`, `BeliefProcess`,
`ActionOutcome`, `CouplingGraph`, `IncentiveContract`, `Decision`,
`StateView` and `ConstraintSet`. Primitive aliases may live in a shared
prelude, but no import/prelude contract exists, so a typo is indistinguishable
from an external type.

Only six modules appear in the canonical registry. Rules are IDs without
machine-readable enforcement bodies. The YAML needs strict schema validation,
declared imports, reference closure and complete contracts for every locality
claim being accepted.

### M9-2 — check semantic contracts, not only selected fields

`flatten()` ignores:

- `version`;
- `notes`, including the only canonical statement that the loss limit lives
  in SP-Params;
- `DecisionSchemeConfig.validation`, so n-ary compatibility can regress to
  pairwise;
- protocol `registry`, so both plug-in registries can silently disappear;
- rule enforcement bodies;
- schema/reference validity and module/port producer-consumer closure.

Additions are always non-fatal. That is correct for a local, schema-valid
extension, but not for an added duplicate owner, live handle or undeclared
dependency. Run a complete target-invariant pass in addition to backward diff.

The allowlist is keyed only by flattened path. One exception permanently
permits unrelated future changes to that path, and the allowlist comes from the
worktree even when comparing explicit refs. Bind exceptions to operation, exact
normalized old/new values, contract versions and rationale; load from the
target ref. Reject duplicate YAML keys and compare canonical serialization.

### M9-3 — define one joint outcome per estimated support point

`JointOutcomeDistribution.support` is:

```
list[(AdverseScenarioSet, weight)]
```

That remains a distribution over whole scenario sets — the ambiguity v9 says
it removed. A support point should be one `AdverseScenario`/joint venue outcome
or a separately defined `JointOutcome`; the set is the policy catalogue or
support collection, not one realization.

Declared scenarios now retain `Known`, outcomes are venue outcomes rather than
stable ADVERSE/BENIGN labels, and incomplete-map behavior is explicit. Those
are correct fixes. However, the sole loss-limit owner exists only in an ignored
note. YAML defines neither `SP-Params`, `ScenarioLossLimit`, the loss
functional, the hard constraint nor its consumer wiring. Change #7 therefore
remains STRUCTURAL.

### M9-4 — make extension and decision-snapshot contracts executable

YAML omits the documented one-registration `IncentiveModel`. Adding a
principal-agent/contest theory still has no canonical extension supplying its
instrument contract, competition model, constraints and cash-flow emitter.
`ResolvedContracts` references undefined `IncentiveContract`; it does not
replace registration of the theory producing those contracts. Change #9
therefore remains STRUCTURAL despite the correct competition aggregator.

The open decision axes also need these corrections:

1. `DecisionSchemeConfig.coupling_ref` is mandatory while
   `DecisionProblem.coupling` may carry dynamic `Known[CouplingGraph]`.
   Use an exclusive typed static/dynamic source variant.
2. One global `unwrap_policy` cannot select expectation for one uncertain
   input and shared scenarios for another. Scope policies by consumer/input.
3. `FALL_BACK` has no prior reference, maximum age, provenance or
   admissibility condition.
4. `DecisionProblem.outcomes` remains a live callable. Pin implementation,
   fitted artifact and spec/version identity for immutable replay.
5. Define `PluginRef`, registry/factory behavior, config validation and
   `ModuleManifest` capabilities canonically. A string named
   `n_ary_capability_check` is not an executable compatibility contract.

### M9-5 — wire heartbeat, halt and cancel as one closed graph

The record shapes improve, but no module produces `CancelAllCommand`:
`OP-Monitor` produces only `HaltState`, while `DE-Actuator` consumes the
command. Its manifest exposes `halt_in`, not a cancel-command input.

Either OP produces both state and command with a declared actuator command
port, or the actuator consumes `HaltState` and owns idempotent command
construction/retry internally.

`Heartbeat{port, period, staleness_deadline}` is a registration, not an
observed pulse: it has no observation time, sequence or last-seen state. There
is no pulse producer or monitor clock port, and `DA-Feeds` lacks the telemetry
output promised by the narrative. Unknown health cannot yet be computed
fail-closed.

`CancelAllStatus.confirmed_by` is mandatory even for `ISSUED`,
`UNCONFIRMED` and `FAILED_TERMINAL`. Use a discriminated union or require
confirmation evidence only for `CONFIRMED`.

## SHOULD-FIX

1. Synchronise or generate markdown excerpts from YAML. Contradictions include
   `scenarios()` vs `draws()`, pairwise vs n-ary compatibility, old mutable
   venue identity, `ContractResolver` vs `ResolvedContracts`, the old ADVERSE
   formula, the old ScenarioProvider type and lowercase `cancel_all`.
2. The v9 header should identify `PM_STRUCT_ITER8_REVIEW.md` and its six
   MUST-FIX findings, not the v7 review.
3. Define the competition aggregator's combined knowledge timestamp,
   provenance/error, staleness and partial-unavailability behavior.
4. Enforce eligible competition node kinds with a type, not an ignored note.
5. Define wildcard port expansion/precedence and validate every consumed
   type/port has exactly the intended producer.

## 13-change acceptance replay

| # | change | v9 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL under intended open solver protocol; registry/manifest closure remains M9-1/M9-4 |
| 2 | fair-value source | LOCAL: BE belief implementation with knowledge-stamped output |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL under intended open solver protocol |
| 5 | variance composition | LOCAL: `VarianceGroup` implementation/configuration |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | STRUCTURAL: distribution is over sets and sole parameter/loss owner is absent from YAML |
| 8 | pair representation / joint action | LOCAL through `CouplingGraph`, after coherent static/dynamic selection |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: aggregator is fixed, but canonical `IncentiveModel` registration is absent |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: namespaced `(ParamId, ScopeKey)` |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL/new modules under intended manifest registry; canonical completeness remains M9-1 |

## Keep unchanged

- owner-qualified, type-aware comparison and fatal Git failures;
- named competition aggregator and explicit dependence;
- knowledge-wrapped belief, competition and risk scenarios;
- outcome-based adverse scenarios and incomplete-map policy;
- immutable condition/token identity and metadata-only slug;
- namespaced `ParamId` and scenario scope;
- open utility/solver protocols;
- pre-resolved, spec-hash-pinned incentive values;
- typed static coupling identity and dynamic problem coupling;
- typed health/halt/cancel records with bounded retry and reconciliation;
- field provenance, typed nullity, bitemporal specs, fitted-artifact guard,
  gate DAG, capability-sliced environments and demand-driven build.
