# PM_STRUCT_ITER12_REVIEW — requested v12 / repository v11.1 review

Object: commit `ad5cbc0`, based on v11 review commit `73561cd`. The commit
itself is labelled `contracts v11.1`, not architecture v12. Date: 2026-08-20.
Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories, mechanisms and venues
should be local plug-ins, while modules remain independently testable and
optimisable.

## Verdict

This is a narrow but important process fix, not a new architecture iteration.
It replaces the permanent path-only allowlist with target-ref, version-bound,
exact old/new migration records. The previous coupling exploit is now rejected,
the unchanged HEAD-to-worktree check passes, duplicate migration signatures are
rejected, and self-tests cover A→B versus A→C and the wrong version step.

Neither architecture source changed:

```
PM_ARCHITECTURE.md          0 lines changed
contracts/contracts.yaml   0 lines changed
contracts version          11
architecture title         v10
new commit label           v11.1
```

Therefore documentation improvement is not “marginal”; in this iteration it is
**zero**. The machine-readable architecture is also unchanged. The meaningful
improvement is entirely in migration-gate safety.

M11-1 is closed. M11-2 through M11-5 remain, with one additional versioning
hole: additive contract changes can pass without a contract-version bump. The
historical replay remains:

```
LOCAL 11 / SPREADING 1 / STRUCTURAL 1
target: 0 STRUCTURAL, <= 1 SPREADING
```

Scenario risk (#7) remains SPREADING and incentive theory (#9) remains
STRUCTURAL.

## Executable audit

Expected results:

```
contract_check.py --selftest
  PASS original diff/reference cases
  PASS exact A->B migration
  PASS rejects A->C at the same path
  PASS rejects a different version step

contract_check.py 1c9e50c ad5cbc0
  21 exact removals, 9 exact changes, 79 additions, exit 0

contract_check.py ad5cbc0 WORKTREE
  0 removed, 0 changed, 0 added, exit 0
```

The former exploit is closed:

```
DecisionProblem.coupling:
  expected migration  CouplingSource -> ResolvedCoupling
  adversarial change  CouplingSource -> CompetitionState
  authorises(...)      None
```

Residual adversarial results:

```
add DecisionProblem.new_live_handle at version 11  exit would remain 0
duplicate key in contracts.yaml                      silently overwritten
false R-HALT semantics                               invariants []
remove OP runtime producers                          invariants []
give BE-* a venue escape port                        invariants []
```

## MUST-FIX

### M12-1 — publish a coherent architecture version and regenerate the docs

The requested object is called v12, the new commit calls itself v11.1,
`contracts.yaml` remains version 11, and `PM_ARCHITECTURE.md` is still titled
v10. No architecture or schema line changed.

The human document also retains contracts superseded several iterations ago:
mutable venue identity, `scenarios()` instead of `draws()`, pairwise
compatibility, the old static coupling ref, a live outcomes callable,
`ContractResolver`, the old ScenarioProvider type, lowercase cancellation and
old port manifests. It no longer explains the system represented by the
714-line canonical source.

Choose an explicit release policy:

- checker-only fixes are patch releases such as contract-tool v11.1 and do not
  count as architecture iterations; or
- an architecture v12 bumps `contracts.version`, updates the architecture
  title/lineage and lands actual contract changes.

Generate human-readable type/module/port/rule tables from YAML, then keep only
rationale and diagrams handwritten. Add a consistency check for version,
identity, decision fields, risk provider, incentive selection and halt ports.
For a modular research program, human collaborators need a reliable map; “YAML
wins” does not make an actively contradictory guide harmless.

### M12-2 — finish semantic validation and enforce versioning for additions

Exact migrations fix removal/change authorization. The remaining checker
boundary is still incomplete:

1. `StrictLoader` is used only for `migrations.yaml`; `contracts.yaml`
   still loads through `yaml.safe_load`, so duplicate canonical type/field keys
   are silently overwritten.
2. Additions are always non-fatal and do not require a version bump. Adding a
   new live handle to `DecisionProblem` at version 11 passes references and
   exits successfully. Require any flattened contract delta—add, remove or
   change—to advance the contract version, even when additions need no
   migration record.
3. Rules are prose/check strings, not dispatched validators. False R-HALT or
   R-SSOT semantics pass.
4. Runtime `HeartbeatPulse`, `HaltState` and `CancelAllCommand` remain in
   `config_supplied`, so their producers can disappear.
5. Port maps/wildcards are neither shape-validated nor checked against R-PORT;
   a belief or solver can gain a venue port without failure.
6. Type producer matching still relies on string splitting rather than the
   parsed type expression promised by the architecture.

Use the strict loader for all canonical YAML and test duplicate contracts.
Require adjacent/valid migration versions, required record fields and nonempty
reasons. Add tests for same-version additions, false rule semantics, removed
runtime producers, undefined config inputs, forbidden ports and malformed
`ports._representation`.

Also update the checker docstring: it still instructs intentional removals to
use the deleted `removals_allowlist.yaml`.

### M12-3 — make scenario loss type-consistent and single-copy

The architecture is unchanged from v11, so the risk discrepancies remain:

- `LossFunctional.loss` returns `Wealth`, while
  `ScenarioLossLimit.resolved` is `Known[Money]`;
- the same `ScenarioLossConstraint` appears inside
  `DecisionProblem.constraints` and as separate inputs to DE-Constraints and
  DE-Allocator, with no shared identity/hash;
- `FeasibleSet.binding: list[Constraint]` cannot represent a binding
  `ScenarioLossConstraint`;
- `ScenarioLossLimit.unit: str` duplicates `Money.currency`;
- `ParamValue.value: any` and `valid_for: ScopeKey` remain untyped and
  duplicate the parameter-store scope rather than temporal validity.

Use one immutable constraint reference shared by problem, constraint engine and
allocator. Return a unit-bearing loss type compatible with the resolved limit,
include scenario constraints in binding/dual output, and type parameter schemas
and validity intervals. Change #7 remains SPREADING.

### M12-4 — wire incentive selection into outcome cash flow

The selected `IncentiveModel` still has no executable path into outcomes:

- no module consumes the selected model or its four contributions;
- `PinnedOutcomes.evaluate` receives neither resolved contracts nor joint
  competition, so it cannot call `PayoutFunctional.cash`;
- no outcome-model capability requires incentive cash-flow integration;
- `ModuleRef` has no competition capability/manifest;
- selected and resolved plugin identities are not equality-checked;
- `PluginRegistry` remains opaque, every `ModuleManifest` is unused, and
  `PluginRef.config: JsonSchema` confuses a config instance with its schema.

Define an immutable decision/outcome context carrying joint competition,
resolved contracts, portfolio and spec snapshot. Require the outcome model to
integrate the selected payout contribution and validate plugin identity from
selection through resolution and evaluation. Define registry entry/factory,
typed protocol IDs, config values and real manifests. Change #9 remains
STRUCTURAL.

### M12-5 — validate one coherent decision snapshot and module graph

No validator connects:

- `DecisionSchemeConfig.coupling_mode` to
  `DecisionProblem.coupling`;
- selected incentive plugin to `ResolvedContracts.incentive_plugin`;
- `DecisionProblem.spec_snapshot` to resolved-contract and outcome-model
  spec hashes;
- unwrap/unavailable policy keys to all and only the relevant problem inputs;
- static/dynamic coupling mode to graph update semantics and knowledge time.

All mismatched combinations are reference-valid. Build one canonical
problem-construction validator that enforces these relationships, PIT cutoffs,
policy coverage and n-ary capabilities before solver invocation.

The module registry remains the same nine-module subset and no module publishes
the defined `ModuleManifest`. Belief, flow/outcome, solver, incentive resolver
and most SP/EV construction paths remain implicit. Complete the minimum
end-to-end graph and remove runtime outputs from `config_supplied`.

## SHOULD-FIX

1. Validate migration record schema explicitly rather than relying on failed
   matching: operation enum, required keys, integer adjacent versions, nonempty
   reason and old/new requirements by operation.
2. Add end-to-end tests loading migrations from a target Git ref, rejecting a
   missing migration file for a version-changing removal, detecting unused
   records and rejecting duplicate migration signatures.
3. Move `ports._representation` out of the port map.
4. Type competition-capable node IDs and knowledge aggregation behavior.
5. Replace remaining behavior-bearing `str`/`any` values with domain types,
   units and schemas.

## 13-change acceptance replay

| # | change | v11.1/requested-v12 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL under intended open solver protocol; registry enforcement remains M12-4 |
| 2 | fair-value source | LOCAL: knowledge-stamped belief implementation |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL under intended open solver protocol |
| 5 | variance composition | LOCAL through `VarianceGroup` |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | SPREADING: loss/limit/binding types and duplicated constraint inputs remain inconsistent |
| 8 | pair representation / joint action | LOCAL through `CouplingMode` and `ResolvedCoupling` |
| 9 | rewards -> obligation / principal-agent / contest | STRUCTURAL: selected extension still has no path into outcome competition/payout cash flow |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL: namespaced `(ParamId, ScopeKey)` |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL under intended registry; manifest enforcement remains incomplete |

## Closed in this iteration

- migration records are target-ref scoped;
- operation, key, exact old/new and exact version transition are matched;
- duplicate migration signatures fail;
- migration A→B no longer authorizes A→C;
- unrelated historical records no longer fail HEAD-to-worktree checks;
- the old permanent allowlist file is removed.

## Keep unchanged

- all v11 contract improvements listed in `PM_STRUCT_ITER11_REVIEW.md`;
- weighted joint outcomes and completeness handling;
- typed scenario limit, constraint and allocator ownership intent;
- selected/pinned incentive extension and joint payout signature;
- split coupling mode/resolved state and callable pinned outcomes;
- typed cancellation/reconciliation and heartbeat contracts;
- restored wealth/variance/settlement/capital types;
- expanded reference tests and fatal Git failures;
- immutable identity, namespaced params, provenance/null/version principles,
  gate DAG, capability-sliced environment and demand-driven build.
