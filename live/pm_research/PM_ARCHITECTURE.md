# PM_ARCHITECTURE v5 — structure for P-2026-003

Rewritten 2026-08-20 from `PM_STRUCT_ITER4_REVIEW.md` (8 MUST-FIX; v4 scored
LOCAL 9 / SPREADING 2 / STRUCTURAL 2 against a target of 0 STRUCTURAL, ≤1
SPREADING). Prior: v1 5/3/5, v2 7/4/2.

Everything on the review's **keep-unchanged** list is preserved verbatim in
substance: two-axis spec time · `Known[V]` + knowledge-truncated `StateView` +
exposure envelope · generic `evaluate(action_set,…)` · cost-basis `L_adv` ·
settlement facts in DA, attribution in EV · live/replay/sim seam with
deterministic replay and the fitted-artifact guard · allocator, gate DAG, STOP
ownership · demand-driven specs, no speculative venue adapters.

Naming: SP/DA/BE/DE/EV/OP modules; `EXP-*` experiments.

---

## 0. Rules and enforcement

| Rule | Enforcement |
|---|---|
| **R-SSOT** | specs/params are read-only handles; no restatement |
| **R-KNOW** | `Known[V]`; every read knowledge-truncated (§4); no API takes event time |
| **R-ONCE** | THREE declared ledgers (§7): markout partition, objective ledger, variance components |
| **R-PROV** | provenance per *field* (§3) and per param; `assumed` may not gate |
| **R-VERSION** | specs bitemporal: `observed_at` + validity interval |
| **R-NULL** | typed `NullPin` with bias direction, not a bare optional |
| **R-PORT** | modules declare a **port manifest**; only the composition root sees the whole `Environment` (§9) |
| **R-COMPAT** | the three decision axes (§6) declare pairwise compatibility; wiring rejects invalid combinations |

---

## 1. Layers

```
SP  SPECS     Venue · Instrument · Strategy · Params        (data, not a subsystem)
DA  DATA      Discovery · Feeds · Normalize · State(+SelfState) · Settlement
BE  BELIEF    Target · Uncertainty · Belief · FlowAndFills · Competition
DE  DECISION  ActionSpace · Constraints · DecisionScheme · Allocator · Actuator
EV  EVAL      Markout · Calibration · Attribution · Replay · Gates
OP  OPS       LatencyBudget · Monitor/KillSwitch
```
**Dependencies point downward only: SP ← DA ← BE ← DE. EV reads all planes and
is read by none. OP is readable by all and depends on none.** Settlement FACTS
are DA (read downward by `DE-Allocator`); performance attribution is EV.

---

## 2. Shared type algebra (M4-7 — parallel work needs these to be contracts)

```
Uncertain[T] { expectation(), quantile(q), scenarios(n, rng), map(f), combine(other, f) }
              # a distribution/bracket; NOT two hardcoded scenarios
Unavailable  { reason, since, cause: Unavailable? }        # upstream cause chain
NullPin      { field, assumption, bias_direction, declared_by }
Known[V]     { value, t_event, t_known, t_known_prov, t_known_err, source, provenance }

ActionOutcome{ fills, state_transition, cash_flows, latency_used,
               markout_partition, provenance }             # §7.1
Constraint   { id, kind: HARD|SOFT, unit, usage, binding_id }
ModuleManifest{ inputs, outputs, ports_required, capabilities_required,
                stateful, artifacts, null_semantics }
```
Every module publishes a `ModuleManifest`; wiring validates it. Without this
algebra a solver must inspect concrete implementations and the plug-in
boundary is fiction.

---

## 3. SP — bitemporal specs, provenance per FIELD

```
SpecRecord{ hash, observed_at, source, valid_from, valid_to, fields }
FieldState = Resolved(value, source, provenance)
           | Disputed(candidates[], observed_at)
           | Unknown(reason, sources_tried[])
```
Record-level source cannot describe fields reconciled from different
authorities — and `Disputed` is live today (Gamma vs CLOB registry on the
rewards band, both in `markets.jsonl`). Consumers must handle `Disputed`; they
may not silently pick.

Declarative rules are stored as **`(family, params)`**, never closures:
`fee_schedule = (PIECEWISE_MINPQ, {rate, size_rounding})`, `tick_rule =
(BANDED, {...})`. Rewards band/rate/eligibility are **instrument-scoped and
time-varying** even though the programme is venue-level.

`SP-Params{ ScopeKey -> ParamValue{value, provenance, owner, valid_for,
measured_at, fit_data_through?, artifact_id?} }` — see §4.

**Populate on demand.** EV-Markout needs ~4 facts, not 30 fields.

---

## 4. Identity, ScopeKey, exposure

```
VenueId · InstrumentId{venue, symbol, horizon, expiry} · TokenId{instrument, side}
RiskFactorId · PortfolioId · FeedId · RegionId

ScopeKey{ venue?, factor?, instrument?, horizon?, feed?, region?, portfolio? }
resolve: most-specific by SUBSET ORDER; ties (Venue(v) vs Factor(f) both
         matching, neither a subset of the other) FAIL LOUD — never silently
         pick. The fully resolved key is recorded in provenance.

exposure: MANY-TO-MANY  InstrumentId -> {(RiskFactorId, loading)}
          # one instrument loads on its own underlying AND a common crypto
          # factor; v4's many-to-one was wrong
```

---

## 5. DA / BE

**DA** — `view(now)` is the only entry; `StateView.coverage(field, span)` bound
to that `now`; `get -> Known[V] | Unavailable`; composition `t_known = MAX`,
`prov = weakest`, `t_known_err = combine`, staleness declared per field.
`SelfState.envelope(now) -> {submitted, acked, filled, in_flight,
worst_case_exposure}`. `t_known` on REST/on-chain history is IMPUTED/ASSUMED —
`t_known := t_event` is the type-laundered look-ahead; replay refuses inside
`t_known_err`.

**BE-Target** (`K`, `E[X_T]` from `w_declared`) · **BE-Uncertainty** (σ_eff,
`w_hat`; registers variance components §7.3) · **BE-Belief → BeliefProcess{
p_hat, link G, jump_tail?, path_law?, constituents{name→(value, weight,
provenance)}, staleness, nulls: NullPin[] }` · **BE-FlowAndFills**:
```
evaluate(action_set, StateView, SelfState, horizon) -> Uncertain[ActionOutcome] | Unavailable
```
pair-aware by construction (the PM book is unified across the token pair).

**BE-Competition (NEW — M4-2)** — the rival/equilibrium state the incentive
theory needs:
```
BE-Competition -> { rival_score X, our_marginal_effect(action_set),
                    total_participation, eligibility_state }
```
Without this in `DecisionProblem`, an incentive-scheme change alters an
interface and stays STRUCTURAL.

---

## 6. DE — THREE independent axes (M4-1)

v4 put `HJBQVI` (a control method) and `CARA-CE` (a utility functional) on one
axis, so the adopted prediction-market theory — which needs **both at once** —
was not expressible.

```
UtilityFunctional : RiskNeutral | CARA | PathFunctional
ControlSolver     : ClosedFormGLFT | PerLevel | HJBQVI
Coupling          : PerToken | JointPair | PortfolioJoint
```
All three declare pairwise compatibility (R-COMPAT); wiring rejects invalid
triples. A new utility does not reimplement a solver; a new coupling does not
reimplement either.

```
DecisionProblem{ view: StateView, self: SelfState, belief: BeliefProcess,
                 competition: CompetitionState,
                 outcomes: (action_set -> Uncertain[ActionOutcome]),
                 actions: ActionSpace, portfolio: PortfolioState,
                 constraints: ConstraintSet,      # ORACLE, not a precomputed set:
                                                  # feasibility is conditional on
                                                  # the candidate action set
                 incentives: IncentiveContract, horizon }
DecisionScheme.solve(DecisionProblem) -> Decision{ actions, duals, rationale }
```

**Incentives, decomposed (M4-2).** "Rewards are a constraint" was too coarse:
```
IncentiveContract  = payout functional + eligibility obligations   (SP-Venue)
BE-Competition     = rival score, our marginal effect               (§5)
DE-Constraints     = eligibility/obligation feasibility
UtilityFunctional  = the realised/expected incentive cash flow
```
The solver dual prices the **opportunity cost of satisfying an obligation** —
it is NOT the subsidy payment. Both appear, in different places.

`DE-Allocator` splits capital/risk across `{InstrumentId}` and `RiskFactorId`
(many-to-many loadings). `DE-Actuator` owns lifecycle, hysteresis, rate budget.

---

## 7. Three ledgers (R-ONCE)

### 7.1 MarkoutPartition — sums EXACTLY to a measured quantity
```
spread + transient_AS + permanent_AS + snipe + own_impact = markout(τ)
```
each with `{unit, conditioning_measure, sign_convention, coverage}`.

### 7.2 ObjectiveLedger — economics, NOT all of it inside markout (M4-6)
```
markout · fees · rebates · incentive_payments · capital_cost · terminal_utility
```
v4 said "anything not in the partition is not an economic term", which wrongly
excluded fees, rebates, incentives, lockup and terminal utility. Two ledgers,
non-overlapping, each with units/measure/coverage keys.

### 7.3 VarianceComponents — declarative, no class (M4-5)
```
VarianceComponent{ owner, unit_space, estimand, support,
                   composition_operator, provenance }
```
Overlapping support is a wiring error. This is what stops `σ_⊥ + κ` and the
sum-vs-min recurrences; v4 claimed R-ONCE covered variance but partitioned only
markout.

---

## 8. Risk — cost basis

```
m = min(q_up, q_down)                        # paired, riskless, redeems $1
L_adv = unpaired_cost_basis(q_up, q_down, m) # premium actually at risk
per market:  L_adv ≤ κ_$
per factor:  Σ_i loading(i,f)·L_adv_i ≤ L_max(f)      # many-to-many
```
`|u|·(1−p̂)` charged 0.02/share where ~0.98/share is lost — least where loss is
largest.

---

## 9. Environment — capability slices, not a god-object (M4-3)

Only the **composition root** sees the whole `Environment{clock, feeds, venue,
rng, artifacts}`. Modules receive narrow ports, declared in their manifest:

| module | port |
|---|---|
| DA-Feeds | feed port |
| BE-*, DE-* (except Actuator) | `StateView`, RNG, artifact resolver |
| DE-Actuator | venue port |
| replay runner | replay clock + tape |

Injecting the whole object would let a belief bypass `StateView` by reading
feeds, and a solver bypass `DE-Actuator` by touching the venue — defeating
R-KNOW and the dependency rule. Ports make both enforceable.

Live/Replay/Sim are implementations. Contracted: deterministic tie-breaking,
warm-state snapshot + restart parity, artifact resolution by `artifact_id` +
`fit_data_through`, RNG owned by the environment, spec-hash pinning per run.

---

## 10. EV / OP

`Gate{id, question, metric, threshold, unit, data_prereq, owner, preconditions,
on_pass, on_fail}` forming a **precondition DAG**; **STOP** is a first-class
gate with `on_fail = HALT_PROGRAM`, a named owner and a review date.
`EV-Attribution` decomposes over §7; read by nobody. `EV-Replay` builds a
ReplayEnv. `OP-LatencyBudget`: four legs, ack unobserved.
`PM_PREREG.md` does not exist — nothing is frozen yet.

## 11. Build order (demand-driven)
1. `Known[V]` + `t_known_prov` + `Unavailable` in DA (`recv_ns` is on disk).
2. EV-Markout / EV-Calibration on real state (~4 spec facts).
3. EXP-IMPACT (AS partition sign) · EXP-BLEND (belief constituents).
4. `L_adv` + `DE-Allocator` + STOP gate before any order.

## 12. Deliberately not built
No venue abstraction layer; no `VarianceBudget` class; no branches for
unreachable capabilities; solver impls marked BUILT vs NAMED-SEAM.
