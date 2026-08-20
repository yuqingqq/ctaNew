# PM_STRUCT_ITER5_REVIEW — v5 structure acceptance review

Object: `PM_ARCHITECTURE.md` v5 at commit `5f80dae`. Date: 2026-08-20.
Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — SOTA theories, mechanisms and venues
should be local plug-ins, while modules remain independently testable and
optimisable.

## Verdict

The eight v4 findings mostly landed. In particular, v5 separates utility,
control and coupling; gives incentives contract/competition/constraint/cash-
flow homes; capability-slices the environment; introduces a composite scope;
defines shared algebraic types; separates the three R-ONCE ledgers; and adds
field-level spec provenance.

It is not converged yet. Provisional score as written:

```
LOCAL 9 / SPREADING 3 / STRUCTURAL 1
target: 0 STRUCTURAL, <= 1 SPREADING
```

After restoring the parameter-name key, the score becomes approximately
`LOCAL 10 / SPREADING 3 / STRUCTURAL 0`. The three spreading changes are
portfolio risk, pair coupling and incentive theory.

## MUST-FIX

### M5-1 — restore ParamId to the parameter-store key

v4 used `(name, scope)`. v5 now declares:

```
SP-Params{ ScopeKey -> ParamValue }
```

Two parameters at the same scope — for example volatility and latency —
therefore collide. The canonical key is:

```
SP-Params{ (ParamId, ScopeKey) -> ParamValue }
```

This is the remaining STRUCTURAL failure because adding any second quantity
requires changing the storage identity.

### M5-2 — coupling is a graph, not a single-choice enum

A portfolio solver must preserve joint Up/Down atomicity inside each market.
Selecting `PortfolioJoint` instead of `JointPair` loses that nesting. Coupling
is hierarchical and may overlap:

```
portfolio
  +-- market A {Up, Down}
  +-- market B {Up, Down}
```

Use a typed `CouplingGraph`/hypergraph. This lets pair coupling and portfolio
coupling compose without either implementation knowing the other.

### M5-3 — terminal utility is not an additive economic term

The ObjectiveLedger currently lists:

```
markout · fees · rebates · incentive_payments · capital_cost · terminal_utility
```

The first five form terminal wealth/cash flow. `terminal_utility` is the
functional applied to that distribution, not a summand with the same unit.

```
WealthLedger = markout + fees + rebates + incentives - capital_cost
UtilityFunctional.evaluate(distribution_of(terminal_wealth))
```

This preserves the v5 utility/solver separation and prevents the v2 additive-
objective assumption from returning through the ledger.

### M5-4 — variance composition needs covariance and falsification

"Overlapping support is a wiring error" is too strong: correlated but distinct
components legitimately overlap. Conversely, disjoint human labels can hide
the original double count. A per-component `composition_operator` cannot
describe a joint covariance structure.

```
VarianceGroup{
  components, operator, covariance_model, estimand, validation_gate
}
```

The validation gate is a PIT/standardised-residual audit by horizon and state.
The registry catches declaration errors; the empirical audit catches false
declarations.

### M5-5 — Uncertain.combine requires a dependence contract

Combining marginal scenario sets independently is wrong for pair completion,
cross-coin risk and portfolio fills. `Uncertain[T]` must carry either shared
scenario identity or an explicit joint/coupling law. A combine operation must
refuse when dependence is unknown rather than assume independence.

### M5-6 — CompetitionState must be knowledge-time safe

Rival score `X`, total participation and our marginal effect are estimated,
time-varying quantities. `BE-Competition` currently returns deterministic bare
values, allowing an assumed equilibrium to enter a decision as measured fact.

Required return:

```
Known[Uncertain[CompetitionState]] | Unavailable
```

with fitted-artifact identity and `fit_data_through` where applicable.

### M5-7 — make incentive theory one registered extension

The correct four homes now exist, but change #9 still touches SP-Incentive,
BE-Competition, DE-Constraints and UtilityFunctional — four modules against an
ideal blast radius of one or two.

Define an `IncentiveModel` extension that registers four typed contributions:

```
IncentiveModel{
  contract_spec,
  competition_model,
  constraints,
  cashflow_functional
}
```

The contributions remain independently testable, but adding a new contracting
theory is one plug-in registration rather than a coordinated four-file edit.

### M5-8 — factor loadings are not loss scenarios

The hard cap currently uses `sum loading(i,f) * L_adv_i`. Signed loadings can
cancel hard loss exposure, and linear beta is not the same as a binary adverse-
resolution outcome.

Use explicit adverse scenarios, or a deliberately conservative absolute-
exposure rule. Do not multiply nonnegative cost-basis loss by unconstrained
signed loadings. The scenario model must declare which instruments lose
together and the cap must evaluate scenario PnL directly.

### M5-9 — define the OP data and command path

`OP-Monitor/KillSwitch` must observe health/state, and `OP-LatencyBudget`
consumes measurements. OP therefore cannot both function and "depend on none".

```
HealthEvent -> OP-Monitor -> HaltState
HaltState   -> DE-Constraints as a HARD constraint
```

Cancel-all still goes through `DE-Actuator`; OP publishes state/policy and does
not reach into the venue.

## SHOULD-FIX

1. Replace bare `jump_tail?` and `path_law?` with
   `Value | NullPin | Unavailable`.
2. Belief constituents used by EXP-BLEND must carry aligned `t_known` and an
   estimator-uncertainty field.
3. Add the venue-native market/condition identifier to `InstrumentId`.
4. Expand `Constraint` with limit/predicate, scope, provenance and dual units.
5. Put `review_date`, inference method, strata/spec/artifact hashes and freeze
   time directly in the `Gate` schema.
6. The document says implementations are marked `BUILT` versus `NAMED-SEAM`,
   but no implementation is actually marked.
7. Make `Disputed` candidates carry value, source and provenance explicitly;
   a decision consuming disputed facts should require a declared handler.

## 13-change acceptance replay

| # | change | v5 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | LOCAL: `ControlSolver` |
| 2 | fair-value source | LOCAL: belief implementation |
| 3 | sigma estimator | LOCAL: uncertainty implementation |
| 4 | continuous -> per-level | LOCAL: `ControlSolver` |
| 5 | variance composition | LOCAL after M5-4 |
| 6 | participation rule + size | LOCAL: constraint implementation |
| 7 | loss cap + portfolio aggregate | SPREADING: Constraints + Allocator |
| 8 | pair representation / joint action | SPREADING until M5-2 |
| 9 | rewards -> obligation / principal-agent / contest | SPREADING until M5-7 |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | STRUCTURAL as written due M5-1; LOCAL after fix |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL or new impls |

## Keep unchanged

- the three-way utility/control/coupling distinction;
- incentive contract, competition, constraint and cash-flow ownership;
- capability-sliced environment and port manifests;
- composite scope resolution with ambiguity failure;
- field-level provenance and declarative spec rules;
- shared type algebra and generic action evaluation;
- bitemporal specs, knowledge-truncated state and fitted-artifact guard;
- cost-basis risk, settlement ownership, gate DAG and STOP ownership;
- demand-driven specs and refusal to build speculative venue adapters.
