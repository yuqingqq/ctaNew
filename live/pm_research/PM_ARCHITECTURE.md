# PM_ARCHITECTURE v6 — structure for P-2026-003

Rewritten 2026-08-20 from `PM_STRUCT_ITER5_REVIEW.md` (9 MUST-FIX + 7
SHOULD-FIX; v5 scored LOCAL 9 / SPREADING 3 / STRUCTURAL 1). Prior: v1 5/3/5,
v2 7/4/2, v4 9/2/2. The remaining structural failure was a REGRESSION — v5
dropped the parameter name from the store key that v4 had. Four consecutive
rewrites have now silently dropped something, so §13 adds a contract-inventory
diff to the process.

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
is read by none.** Settlement FACTS are DA (read downward by `DE-Allocator`);
performance attribution is EV.

**OP has a data path and a command path (M5-9)** — it cannot both function and
"depend on nothing", since the monitor observes health and the latency budget
consumes measurements:
```
HealthEvent → OP-Monitor → HaltState → DE-Constraints   (as a HARD constraint)
```
OP publishes state and policy; it never reaches the venue. Cancel-all still
goes through `DE-Actuator`, so the single write path is preserved.

---

## 2. Shared type algebra (M4-7 — parallel work needs these to be contracts)

```
Uncertain[T] { expectation(), quantile(q), scenarios(n, rng), map(f),
               combine(other, f, dependence) }   # a distribution/bracket
              # DEPENDENCE IS REQUIRED (M5-5): shared scenario identity or an
              # explicit joint/coupling law. combine() REFUSES when dependence
              # is unknown — it never assumes independence. Pair completion,
              # cross-coin risk and portfolio fills are all dependent.
Unavailable  { reason, since, cause: Unavailable? }        # upstream cause chain
NullPin      { field, assumption, bias_direction, declared_by }
Known[V]     { value, t_event, t_known, t_known_prov, t_known_err, source, provenance }

ActionOutcome{ fills, state_transition, cash_flows, latency_used,
               markout_partition, provenance }             # §7.1
Constraint   { id, kind: HARD|SOFT, limit_or_predicate, scope: ScopeKey,
               unit, dual_unit, usage, binding_id, provenance }
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
           | Disputed(candidates[{value, source, provenance}], observed_at)
           | Unknown(reason, sources_tried[])
```
Record-level source cannot describe fields reconciled from different
authorities — and `Disputed` is live today (Gamma vs CLOB registry on the
rewards band, both in `markets.jsonl`). A decision that consumes a `Disputed`
field must name a **declared handler**; silent selection is a wiring error.

Declarative rules are stored as **`(family, params)`**, never closures:
`fee_schedule = (PIECEWISE_MINPQ, {rate, size_rounding})`, `tick_rule =
(BANDED, {...})`. Rewards band/rate/eligibility are **instrument-scoped and
time-varying** even though the programme is venue-level.

`SP-Params{ (ParamId, ScopeKey) -> ParamValue{value, provenance, owner,
valid_for, measured_at, fit_data_through?, artifact_id?} }` — see §4.
**ParamId is part of the key** (v5 dropped it, so volatility and latency at the
same scope collided).

**Populate on demand.** EV-Markout needs ~4 facts, not 30 fields.

---

## 4. Identity, ScopeKey, exposure

```
VenueId · InstrumentId{venue, symbol, horizon, expiry, venue_native_id}
TokenId{instrument, side}   # venue_native_id = conditionId/market slug
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
p_hat, link G, jump_tail: Value|NullPin|Unavailable,
path_law: Value|NullPin|Unavailable,
constituents{name→(value, weight, provenance, t_known, estimator_uncertainty)},
staleness }` · **BE-FlowAndFills**:
```
evaluate(action_set, StateView, SelfState, horizon) -> Uncertain[ActionOutcome] | Unavailable
```
pair-aware by construction (the PM book is unified across the token pair).

**BE-Competition (NEW — M4-2)** — the rival/equilibrium state the incentive
theory needs:
```
BE-Competition -> Known[Uncertain[CompetitionState]] | Unavailable
CompetitionState{ rival_score X, our_marginal_effect(action_set),
                  total_participation, eligibility_state }
```
Estimated, time-varying, and often fitted — so it is knowledge-stamped and
uncertain (M5-6). Bare values would let an ASSUMED equilibrium enter a decision
as measured fact; fitted variants carry `artifact_id` + `fit_data_through`.
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
Coupling          : CouplingGraph          # NOT an enum — see below
```
All three declare pairwise compatibility (R-COMPAT); wiring rejects invalid
triples. A new utility does not reimplement a solver; a new coupling does not
reimplement either.

**Coupling is a hypergraph, not a single choice (M5-2).** Choosing
`PortfolioJoint` instead of `JointPair` would DISCARD the Up/Down atomicity
inside each market. Couplings nest and may overlap:
```
CouplingGraph:  portfolio ⊃ { market_A{Up,Down}, market_B{Up,Down} }
```
so pair coupling and portfolio coupling compose without either implementation
knowing the other.

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

**Incentives: four homes, ONE registration (M4-2 + M5-7).** The four owners are
right, but a new contracting theory must not require a coordinated four-file
edit. It registers as a single extension that supplies four typed
contributions:
```
IncentiveModel{ contract_spec,        -> SP-Venue (payout functional + obligations)
                competition_model,    -> BE-Competition
                constraints,          -> DE-Constraints (eligibility/obligation)
                cashflow_functional }  -> UtilityFunctional (realised/expected cash)
```
Contributions stay independently testable; adding a theory is one plug-in.
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

### 7.2 WealthLedger — additive cash, in one unit (M4-6 + M5-3)
```
WealthLedger = markout + fees + rebates + incentives − capital_cost
UtilityFunctional.evaluate( distribution_of(terminal_wealth) )
```
v5 listed `terminal_utility` as a SUMMAND beside cash terms. It is not: it is
the functional APPLIED to the terminal-wealth distribution, in a different
unit. Listing it additively would smuggle back the v2 additive-risk-neutral
objective through the ledger, defeating §6's utility/solver separation.
Each cash term carries units/measure/coverage keys.

### 7.3 VarianceComponents — declarative, no class (M4-5)
```
VarianceComponent{ owner, unit_space, estimand, support, provenance }
VarianceGroup{ components, operator, covariance_model, estimand,
               validation_gate }
```
Overlap is NOT automatically an error (M5-4): correlated-but-distinct
components legitimately overlap, and disjoint human labels can equally hide a
real double count — so a per-component operator cannot express joint
structure. The group declares a `covariance_model`, and `validation_gate` is an
empirical PIT / standardised-residual audit by horizon and state.
**The registry catches declaration errors; the audit catches false
declarations.** This is what stops `σ_⊥ + κ` and the sum-vs-min recurrences.

---

## 8. Risk — cost basis

```
m = min(q_up, q_down)                        # paired, riskless, redeems $1
L_adv = unpaired_cost_basis(q_up, q_down, m) # premium actually at risk
per market:  L_adv ≤ κ_$
per scenario: Σ_{i ∈ losers(s)} L_adv_i ≤ L_max(s)   # explicit adverse scenarios
# NOT Σ loading(i,f)·L_adv_i (M5-8): signed loadings can CANCEL hard loss
# exposure, and a linear beta is not a binary adverse-resolution outcome.
# The scenario model declares which instruments lose TOGETHER; the cap
# evaluates scenario PnL directly. Absolute-exposure fallback is permitted
# only as a deliberately conservative rule.
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
on_pass, on_fail, review_date, inference_method, strata_hash, spec_hash,
artifact_hash, frozen_at}` forming a **precondition DAG**; **STOP** is a first-class
gate with `on_fail = HALT_PROGRAM`, a named owner and a review date.
`EV-Attribution` decomposes over §7; read by nobody. `EV-Replay` builds a
ReplayEnv. `OP-LatencyBudget`: four legs, ack unobserved.
`PM_PREREG.md` does not exist — nothing is frozen yet.

## 11. Build order (demand-driven)
1. `Known[V]` + `t_known_prov` + `Unavailable` in DA (`recv_ns` is on disk).
2. EV-Markout / EV-Calibration on real state (~4 spec facts).
3. EXP-IMPACT (AS partition sign) · EXP-BLEND (belief constituents).
4. `L_adv` + `DE-Allocator` + STOP gate before any order.

## 12. Implementation status (SHOULD-FIX 6 — v5 claimed marking, did none)

| module | BUILT | NAMED-SEAM |
|---|---|---|
| DA-Feeds | PMMarketWS, PMPricesWS, BinanceWS, GammaREST, ClobREST | PolygonRPC |
| DA-Discovery | collect_pm discovery loop | — |
| DA-Normalize / DA-State / DA-Settlement | — | all |
| BE-* | — | all (Target, Uncertainty, Belief, FlowAndFills, Competition) |
| DE-* | — | all |
| EV-Markout / EV-Calibration | methodology only | harness |
| ControlSolver | — | ClosedFormGLFT, PerLevel, HJBQVI |
| UtilityFunctional | — | RiskNeutral, CARA, PathFunctional |

Nothing in BE or DE is built. The register describes contracts, not code.

## 13. Process: contract-inventory diff (new)

v2 dropped a MUST-FIX, v3 dropped settlement accounting, v4 dropped the
dependency rule, v5 dropped `ParamId` from the params key — **four consecutive
rewrites each silently lost something**, and the keep-list check missed the
fourth because it only covered items the reviewer had named.

A rewrite is not complete until the previous version's **full contract
inventory** (every type, key, field and rule) is diffed against the new one and
each removal is either intentional-and-logged or restored.

## 14. Deliberately not built
No venue abstraction layer; no `VarianceBudget` class; no branches for
unreachable capabilities.
