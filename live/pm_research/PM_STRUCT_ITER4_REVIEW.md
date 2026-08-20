# PM_STRUCT_ITER4_REVIEW — v4 structure acceptance review

Object: `PM_ARCHITECTURE.md` v4. Date: 2026-08-20.
Charter: `PM_STRUCTURE_REVIEW_LOOP.md` — new SOTA theories, mechanisms and
venues should be local plug-ins, while modules remain independently testable
and optimisable.

## Verdict

v4 is a substantial improvement. The v3 findings around bitemporal specs,
knowledge-truncated state, generic action evaluation, cost-basis risk,
settlement ownership, replay parity, fitted artifacts, portfolio allocation
and STOP ownership landed in substance.

It does **not** yet meet the convergence target. Generous provisional replay:

```
LOCAL 9 / SPREADING 2 / STRUCTURAL 2
target: 0 STRUCTURAL, <= 1 SPREADING
```

The two structural failures are: (1) value/utility and control/solver theory
remain one axis, so the adopted HJB-QVI + CARA combination is not composable;
and (2) rewards competition has no state owner or payout functional.

## MUST-FIX

### M4-1 — split utility, control and coupling into three axes

`HJBQVI` is a control/solution method; `CARA-CE` is a utility/value
functional. The adopted prediction-market theory requires both at once, but
v4 lists them as alternatives on one `valuation/control form` axis.

Required shape:

```
UtilityFunctional: RiskNeutral | CARA | PathFunctional
ControlSolver:     ClosedFormGLFT | PerLevel | HJBQVI
Coupling:          PerToken | JointPair | PortfolioJoint
```

All three declare compatibility. `DecisionProblem` should carry a
`ConstraintSet` or feasibility oracle, not a precomputed `FeasibleSet`:
feasibility is conditional on the candidate action set being evaluated.

### M4-2 — give rewards competition and payout an owner

The change-log item is not merely "rewards become a constraint". The adopted
principal-agent/Tullock model requires rival score `X`, our action's effect on
total participation, an eligibility obligation and an actual payment.

Required separation:

```
IncentiveContract = payout functional + eligibility obligations
BE-Competition    = rival participation / equilibrium state
DE-Constraints    = eligibility and obligation feasibility
UtilityFunctional = realised/expected incentive cash flow
```

A solver dual prices the opportunity cost of satisfying an obligation. It is
not the subsidy payment itself. Without `BE-Competition` in the decision input,
change #9 still changes an interface and is STRUCTURAL.

### M4-3 — capability-slice Environment; do not inject the god-object

`Environment{clock, feeds, venue, rng, artifacts}` is the correct composition
seam, but giving the entire object to every stateful module defeats two other
rules: a belief can bypass `StateView` by reading feeds, and a solver can bypass
`DE-Actuator` by touching the venue.

Only the composition root sees the full environment. Inject narrow ports:

- `DA-Feeds`: feed port;
- BE/DE: `StateView`, RNG and artifact resolver only;
- `DE-Actuator`: venue port;
- replay runner: replay clock and tape.

Declared port manifests make R-KNOW and the dependency rule enforceable.

### M4-4 — make parameter scope a composite key

`Global | Venue | Factor | Instrument` cannot represent parameters jointly
keyed by venue, factor, horizon, feed and deployment region. "Most specific
wins" is undefined when both `Venue(v)` and `Factor(f)` match. `PortfolioId`
also cannot appear in scope.

Use a typed composite `ScopeKey` with optional axes, fail on equally-specific
ambiguity, and record the full resolved key. Risk exposure is many-to-many:
one instrument may load on both an underlying factor and a common crypto
factor; it is not always many instruments to exactly one factor.

### M4-5 — add the missing variance partition

R-ONCE claims to cover variance and PnL through the declared partition, but
section 7 partitions markout only. The historical `sigma_perp + kappa` and
sum-vs-min errors can therefore recur unchanged.

A declarative ledger is sufficient; no `VarianceBudget` class is needed:

```
VarianceComponent{
  owner, unit_space, estimand, support, composition_operator, provenance
}
```

### M4-6 — distinguish markout partition from the objective ledger

The statement "anything not in the partition is not an economic term" is too
strong. Fees, rebates, capital lockup, incentive payments and terminal utility
are real economics but are not components of price markout.

Keep two non-overlapping ledgers:

1. `MarkoutPartition`: spread, transient/permanent adverse selection, snipe and
   own impact, summing exactly to markout.
2. `ObjectiveLedger`: markout, fees, rebates, incentives, capital cost and
   terminal utility, with units, measure and coverage keys.

### M4-7 — turn shared type names into usable contracts

Parallel module development still depends on shared semantics that v4 only
names:

- `Uncertain[T]`: `scenarios(n, rng)`, expectation/quantile and composition;
- `Unavailable`: upstream cause chain;
- `ActionOutcome`: fills, state transition, cash flows, latency, provenance and
  the markout partition;
- constraints: `HARD | SOFT`, units, usage and binding identifiers;
- belief nulls: typed `NullPin`, not a bare optional field;
- module manifest: inputs, outputs, required capabilities/ports, statefulness,
  artifacts and null semantics.

Without this minimum algebra, each solver must inspect concrete
implementations and the advertised plug-in boundaries do not hold.

### M4-8 — provenance belongs on each spec field

`Resolved(value)` has no source or provenance, while one record-level source
cannot describe fields reconciled from different authorities. R-PROV therefore
covers parameters but not decision-critical settlement/rewards facts.

```
Resolved(value, source, provenance)
Disputed(candidates, observed_at)
Unknown(reason, sources_tried)
```

The rewards programme may be venue-level, but band/rate/eligibility values are
instrument-scoped and time-varying. Declarative functions such as fee and tick
rules should be stored as `(family, params)`, not closures in the YAML spec.

## 13-change acceptance replay

| # | change | v4 result |
|---|---|---|
| 1 | binary GLFT -> published PM HJB | **STRUCTURAL** until utility and solver axes split |
| 2 | fair-value source | LOCAL |
| 3 | sigma estimator | LOCAL |
| 4 | continuous -> per-level | LOCAL after M4-1 |
| 5 | variance composition | LOCAL, but M4-5 enforcement missing |
| 6 | participation rule + size | LOCAL |
| 7 | loss cap + portfolio aggregate | SPREADING: Constraints + Allocator |
| 8 | pair representation / joint action | SPREADING: coupling + inventory/outcomes |
| 9 | rewards -> obligation / principal-agent / contest | **STRUCTURAL**: competition and payout absent |
| 10 | siting | LOCAL configuration |
| 11 | latency 120 -> 471 -> 1700 ms | LOCAL parameter |
| 12 | PnL-first -> mechanism-first | LOCAL gate registry |
| 13 | late flow/signal/cross-coin/portfolio components | LOCAL or new impls |

## Keep unchanged

- two-axis spec time (`observed_at` plus validity interval);
- `Known[V]`, knowledge-truncated `StateView` and exposure envelope;
- generic `evaluate(action_set, ...)` action vocabulary;
- cost-basis `L_adv`;
- settlement facts in DA and attribution in EV;
- live/replay/sim composition seam, deterministic replay and fitted-artifact
  guard;
- allocator, gate DAG and STOP ownership;
- demand-driven specs and refusal to build speculative venue adapters.
