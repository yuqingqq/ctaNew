# PM_ARCHITECTURE — structure for P-2026-003

The explanatory prose baseline is version 12 (2026-08-20); the machine-readable
canonical contract is now **v21**. Versions 13–16 landed the sigma route and v17
lands the measurement-infrastructure boundary: nominal event/knowledge clocks,
strict source profiles, factual coverage separated from admissibility decisions,
and DA-Normalize/DA-Coverage/DA-State ownership. Version 18 adds point-in-time
coverage refusal and the harness-only event-time leak canary. Version 19 adds the
DA-Canary/DA-Orchestrate boundary and a commit-last, single-writer batch contract
across the requested coin universe. Version 20 adds the full-batch-gated Tier-2
boundary: model-free terminal markout, a point-in-time calibration scaffold,
dual weighting, explicit unavailable rows, and a second commit-last receipt.
Version 21 adds the marked flow-process boundary: same-state arrival/exposure
assignment, per-coin event intensity, conditional side/notional marks, derived
USDC throughput, and a forward-gated optional Hawkes residual.
When this prose and `contracts/contracts.yaml`
differ, the YAML remains authoritative. Score history on the 13-change replay:
v1 5/3/5 · v2 7/4/2 · v4 9/2/2 · v5 9/3/1 · v6–v10 11/0/2 · **v11 11/1/1**
(target 0 STRUCTURAL, ≤1 SPREADING). Open: M12-2 validator dispatch, M12-3
scenario-loss typing, M12-4 incentive→cash-flow wiring, M12-5 snapshot validator.
Prior: v1 5/3/5, v2 7/4/2, v4 9/2/2, v5 9/3/1.

> ## CANONICAL SOURCE
> **`contracts/contracts.yaml` is the machine-readable source of truth for every
> type, field, module, port and rule.** This document explains and motivates;
> it does not define. Schema blocks below are **illustrative excerpts** — if one
> disagrees with the YAML, the YAML wins. Verified by
> `contracts/contract_check.py`, which diffs owner-qualified fields WITH their
> types (so a narrowing is a detected regression) and carries a selftest.
>
> This resolves review M8-3: the v8 checker scanned markdown prose and was
> proven unsound — it reported ZERO changes when
> `Known[Uncertain[JointCompetitionState]]` was narrowed to `CompetitionState`,
> the exact regression class it existed to catch. Reproduced before replacing.

**The v7 diagnosis, adopted as a rule: a fix that lives in explanatory prose is
not a fix.** v7 declared `DecisionProblem.scenarios`, dynamic coupling and joint
competition in narrative blocks BELOW the canonical schemas, where no wiring
check or `ModuleManifest` can see them. In v8 every contract change lands
inside a fenced canonical block. Prose may explain a type; it may not
introduce one.

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
| **R-KNOW** | `Known[V]`; every production read is knowledge-truncated (§4); event-time reads exist only inside R-CANARY |
| **R-ONCE** | THREE declared registries (§7): `MarkoutPartition`, `WealthLedger`, `VarianceGroup` |
| **R-PROV** | provenance per *field* (§3) and per param; `assumed` may not gate |
| **R-VERSION** | specs bitemporal: `observed_at` + validity interval |
| **R-NULL** | typed `NullPin` with bias direction, not a bare optional |
| **R-PORT** | modules declare a **port manifest**; only the composition root sees the whole `Environment` (§9) |
| **R-COMPAT** | the three decision axes (§6) declare pairwise compatibility; wiring rejects invalid combinations |
| **R-IMPUTE** | `OBSERVED` requires wire `recv_ns`; non-observed knowledge carries a named positive-delay rule |
| **R-REFUSE** | `StateView` admits only when `t_known + refuse_k·t_known_err <= now` and counts refusals |
| **R-ADMISS** | coverage facts and selection decisions are separate; only frozen hash-matched rules evaluate, and exclusion requires both arms |
| **R-CANARY** | every replay diagnostic pairs `StateView` with a harness-only leaky `EventTimeView` and records whether selected states differ |
| **R-BATCH** | preflight all requested coins, hold one writer lock, validate every bundle, and emit the immutable cross-coin receipt last; uncommitted artifacts are resumable staging |
| **R-DERIVE** | Tier-2 derives only from a validated complete `full` batch, binds exact inputs, and commits across coins last |
| **R-GROSS** | terminal maker markout is the model-free gross identity; websocket fee zero is never applied |
| **R-DUAL** | every economic markout summary reports both per-fill and share-weighted estimates per coin and phase |
| **R-ONEROW** | calibration has exactly one row per window/frozen horizon; unavailable quotes remain named rows |
| **R-FLOW** | flow arrivals and exposure share one lagged knowledge-admissible state; only event intensity has a compensator; side/notional are marks; Hawkes is admitted only after a forward-valid baseline time change |

---

## 1. Layers

```
SP  SPECS     Venue · Instrument · Strategy · Scenarios · Params   (data)
DA  DATA      Discovery · Feeds · Normalize · State(+SelfState) · Settlement
BE  BELIEF    Target · Uncertainty · Belief · FlowAndFills · Competition
              · ScenarioProvider
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
HealthEvent (owner: OP-Monitor; sources: DECLARED TELEMETRY PORTS of
             DA/BE/DE modules · OP-LatencyBudget — NEVER EV)
   → HaltState        LATCHED, FAIL-CLOSED (unknown health ⇒ halted)
   → halt port ──┬──→ DE-Constraints          HARD constraint: no new risk
                 └──→ DE-Actuator.cancel_all  PRIORITY command
HealthEvent{ id, source_port, severity, observed_at, detail }
HaltState  { level: RUNNING|DEGRADED|HALTED, latched: bool, since,
             reason: HealthEvent[], reset_authority: OP_OWNER_ONLY }
             # monotonic: RUNNING→DEGRADED→HALTED escalates automatically;
             # de-escalation requires explicit operator reset (never automatic)
             # unknown health ⇒ HALTED (fail-closed)
cancel_all { command_id, idempotent: true, requires_ack: true, retry_until_ack }
```
**EV is not a health source (M7-5).** v7 listed EV among the sources, creating
`EV → OP → DE` and letting evaluation state reach decisions — a violation of
§1. Runtime health comes from each module's declared telemetry port.
`EV-Gates.on_fail = HALT_PROGRAM` remains the SEPARATE programme-control path,
operated by a human owner, not a runtime edge.

**Both edges are required (M6-4).** A hard constraint stops NEW risk; it does
not retract resting orders — and during a fault the solver may itself be
`Unavailable`, so a path that runs only through the scheme cannot fire. The
`cancel_all` command therefore bypasses the solver but NOT the actuator, so
`DE-Actuator` remains the sole venue writer.

The halt port is a declared port in §9 (v6's diagram ended at DE-Constraints
while §9 gave DE modules no port that could carry `HaltState`). OP publishes
state and commands; it never touches the venue directly.

---

## 2. Shared type algebra (M4-7 — parallel work needs these to be contracts)

```
Uncertain[T] { expectation(), quantile(q), scenarios(n, rng), map(f),
               combine(other, f, dependence) }   # a distribution/bracket
              # DEPENDENCE IS REQUIRED (M5-5): combine() REFUSES when unknown —
              # it never assumes independence. Pair completion, cross-coin risk
              # and portfolio fills are all dependent.
ScenarioDrawSet[T]{ draws: [(DrawId, weight, T)], common_random_id }
              # M7-2: the result of Uncertain[T].draws() — MONTE-CARLO DRAWS, keyed
              # by DrawId (not ScenarioId).
              # Distinct from AdverseScenarioSet (§8), which is a RISK POLICY.
              # v7 used one name for both and nested it inside itself.
Dependence   = SharedDraws(common_random_id) | JointLaw(copula|factor)
               | Independent(DECLARED)   # never a default
              # scenarios() exposes ScenarioId + weight so shared identity is
              # VALIDATED at the type boundary, not trusted in prose
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

**Records** (each a `SpecRecord`, so each is bitemporal and field-provenanced):
```
SP-Venue{ matching, tick_grid, min_size, rate_limits, fee_schedule(p,side,size),
          rebate_schedule, capabilities{CTF_PAIR, NEG_RISK, MAKER_REWARDS,...} }
          # M7-5: rewards_band REMOVED. Rewards are instrument-scoped and
          # time-varying (below), so a venue-level copy duplicated the fact and
          # violated R-SSOT. The capability flag stays; the resolved
          # band/rate/eligibility live in SP-Instrument.incentive_contract.
SP-Instrument{ settlement{source, statistic, w_declared, strike_rule, tie_rule},
               T, payoff, complement, incentive_contract }   # instrument-scoped
SP-Strategy{ utility, solver, coupling, constraints, action_space, impls,
             nulls{field: NullPin} }
SP-Scenarios{ ScenarioId -> AdverseScenario{...} }           # §8 (M6-3)
SP-Params{ (ParamId, ScopeKey) -> ParamValue{value, provenance, owner,
           valid_for, measured_at, fit_data_through?, artifact_id?} }
```
**ParamId is part of the key** (v5 dropped it, so volatility and latency at the
same scope collided) and is **namespaced by module/plugin** so parallel
implementations do not collide at one market scope. The `Records` block was
lost in the **v4→v5** edit (see §13 for the corrected lineage).

**Populate on demand.** EV-Markout needs ~4 facts, not 30 fields.

---

## 4. Identity, ScopeKey, exposure

```
VenueId · InstrumentId{venue, symbol, horizon, expiry, venue_native_id}
TokenId{instrument, side}   # venue_native_id = conditionId/market slug
RiskFactorId · PortfolioId · FeedId · RegionId

ScopeKey{ venue?, factor?, instrument?, horizon?, feed?, region?,
          portfolio?, scenario? }        # scenario axis added (M7-2)
resolve: most-specific by SUBSET ORDER; ties (Venue(v) vs Factor(f) both
         matching, neither a subset of the other) FAIL LOUD — never silently
         pick. The fully resolved key is recorded in provenance.

exposure: MANY-TO-MANY  InstrumentId -> {(RiskFactorId, loading)}
          # one instrument loads on its own underlying AND a common crypto
          # factor; v4's many-to-one was wrong
```

---

## 5. DA / BE

**DA** — `view(now)` is the only production entry;
`StateView.coverage(field, span) -> Known[Coverage] | Unavailable` is bound to
that `now`; `get -> Known[V] | Unavailable`; composition `t_known = MAX`,
`prov = weakest`, `t_known_err = combine`, staleness declared per field.
`EventTimeView` is constructible only by the replay canary harness and is never
available through a production port.
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

The Revision 3 flow seam is a per-coin marked point process. `DA-FlowNormalize`
materializes one `FlowArrival` per `last_trade_price` aggregate and joins it to
the same 250 ms-lagged midpoint state used for exposure. `BE-FlowFit` fits
events/s and the conditional type/side/monetary-mark laws; USDC/s is derived as
arrival intensity times expected native-price notional. `MICRO_002` and
`MARKET` are labelled subprocesses—independence is a tested null, not a
prerequisite for estimating either. A `HawkesResidualFit` is optional and may be
`NOT_ADMITTED`; it requires ten complete forward days and residual clustering
after baseline time change. The decision-facing `BE-FlowAndFills` remains
`Unavailable` until forward-valid artifacts exist.

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
UtilityFunctional : OPEN PROTOCOL  { evaluate(Uncertain[Wealth]) -> Value }
ControlSolver     : OPEN PROTOCOL  { solve(DecisionProblem) -> Decision }
                    # M8-5: registries with plugin ids + config schemas +
                    # manifests. builtins are REGISTERED, not enumerated in the
                    # type, so an unlisted SOTA theory does NOT edit shared code.
DecisionSchemeConfig{ utility, solver, coupling_ref: StaticCouplingRef,
                      unwrap_policy, unavailable_policy }
                    # M8-5: policies are FIELDS, not comments; validation is
                    # n-ary (a pairwise-valid triple can still be invalid)
Coupling          : CouplingGraph          # NOT an enum — see below
```
All three declare pairwise compatibility (R-COMPAT); wiring rejects invalid
triples. A new utility does not reimplement a solver; a new coupling does not
reimplement either.

**Coupling is a hypergraph, not a single choice (M5-2).** Choosing
`PortfolioJoint` instead of `JointPair` would DISCARD the Up/Down atomicity
inside each market. Couplings nest and may overlap:
```
CouplingGraph{ nodes: {NodeId -> TokenId|InstrumentId|PortfolioId},
               hyperedges: {EdgeId -> (nodes[], relation: ATOMIC|SHARED_RISK|
                                       SEQUENTIAL, params)},
               version, provenance, update_semantics: STATIC|PER_DECISION }
example:  portfolio ⊃ { market_A{Up,Down}ATOMIC, market_B{Up,Down}ATOMIC }
```
**Injection:** the graph is constructed with `DecisionScheme` when
`update_semantics = STATIC`, and carried in `DecisionProblem` when
`PER_DECISION` (e.g. a market resolves mid-horizon).
so pair coupling and portfolio coupling compose without either implementation
knowing the other.

```
DecisionProblem{ view: StateView, self: SelfState,
                 belief: Known[BeliefProcess],          # M8-6: fitted+time-varying,
                                                        # so ONE enforceable t_known
                 competition: Known[Uncertain[JointCompetitionState]] | Unavailable,
                 # M6-1: the FULL type crosses the boundary. Unwrapping to an
                 # expectation/quantile/scenario is an explicit UnwrapPolicy
                 # chosen by the scheme, never composition-root coercion.
                 # UnavailablePolicy{ HALT | FALL_BACK(prior) | REFUSE_ACTION }
                 #   must be declared by the scheme.
                 outcomes: (action_set -> Uncertain[ActionOutcome]),
                 actions: ActionSpace, portfolio: PortfolioState,
                 risk_scenarios: RiskScenarios,        # M7-1 — canonical, not prose
                 coupling: Known[CouplingGraph] | StaticCouplingRef, # M8-6:
                 # typed ref w/ graph_id+hash+provenance, not a sentinel
                 constraints: ConstraintSet,      # ORACLE, not a precomputed set:
                                                  # feasibility is conditional on
                                                  # the candidate action set
                 incentives: ResolvedContracts,   # M8-6: pre-resolved SNAPSHOT
                 horizon, spec_snapshot: Hash }   # not a live handle that can
                 # change after problem construction

RiskScenarios = Declared(Known[AdverseScenarioSet])          # M8-2: keeps the
              |                                              # SP knowledge stamp
                Estimated(Known[JointOutcomeDistribution])   # M8-2: weighted joint
              | Unavailable                                  # outcomes, not
              # "uncertainty over sets" (v8's Uncertain[AdverseScenarioSet] was
              # ambiguous). Incomplete maps handled by `on_incomplete`.
              # the scheme declares an UnavailablePolicy for this field too
JointCompetitionState{ by_node: {NodeId -> CompetitionState} }
              # M8-1: `dependence` REMOVED from the realised sample — it
              # describes CONSTRUCTION of the joint uncertainty, so it belongs
              # to the aggregator/distribution, not to a realised state.
BE-Competition            -> per-instrument MARGINALS
BE-CompetitionAggregator  -> Known[Uncertain[JointCompetitionState]] | Unavailable
              # M8-1: v8 had a scalar producer and a joint consumer with NO
              # module owning the aggregation — the stack could not wire.
              # The aggregator combines marginals under an EXPLICIT Dependence
              # and refuses when dependence is unknown.
DecisionScheme.solve(DecisionProblem) -> Decision{ actions, duals, rationale }
```

**Incentives: four homes, ONE registration (M4-2 + M5-7).** The four owners are
right, but a new contracting theory must not require a coordinated four-file
edit. It registers as a single extension that supplies four typed
contributions:
```
IncentiveModel{ contract_spec,      -> SP-Instrument-scoped ContractResolver
                competition_model,  -> BE-Competition
                constraints,        -> DE-Constraints (eligibility/obligation)
                cashflow_emitter }  -> ActionOutcome.cash_flows → WealthLedger
```
Contributions stay independently testable; adding a theory is one plug-in.

**Cash flow, not preference (M6-2).** v6 routed `cashflow_functional` to
`UtilityFunctional`, re-coupling incentive theory to risk preference one
section after §7 separated them. An incentive model emits realised/uncertain
CASH into `ActionOutcome.cash_flows` and thence `WealthLedger`; the
independently chosen `UtilityFunctional` then evaluates the distribution of
TOTAL terminal wealth.

**Scope (M6-2).** Rewards are instrument-scoped and time-varying (§3), so a
single unscoped contract cannot serve a portfolio-coupled problem:
```
ContractResolver: InstrumentId -> Known[IncentiveContract] | Unavailable
              # M7-4: missing/disputed rewards facts are LIVE states today, not
              # impossible branches; a Disputed field invokes the §3 declared
              # handler rather than being silently resolved
DecisionProblem.incentives: ContractResolver                 # not one contract
DecisionProblem.competition: joint over the CouplingGraph    # not one state
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
per scenario s ∈ AdverseScenarioSet:
   loss(s) = Σ_i L_adv_i · 1{ s.instrument_outcome_map[i] = ADVERSE }
   loss(s) ≤ ScenarioLossLimit(s.id)          # keyed BY SCENARIO — see below
# NOT Σ loading(i,f)·L_adv_i (M5-8): signed loadings can CANCEL hard loss
# exposure, and a linear beta is not a binary adverse-resolution outcome.
# The scenario model declares which instruments lose TOGETHER; the cap
# evaluates scenario PnL directly. Absolute-exposure fallback is permitted
# only as a deliberately conservative rule.
```
`|u|·(1−p̂)` charged 0.02/share where ~0.98/share is lost — least where loss is
largest.

**Scenario ownership (M6-3)** — v6 invented `losers(s)` / `L_max(s)` with no
owner in the algebra, register, manifests or `DecisionProblem`; an ownerless
quantity is STRUCTURAL by the loop rubric.
```
AdverseScenario{ id: ScenarioId, outcome_map: {InstrumentId -> VenueOutcome},
                 scope: ScopeKey, provenance, completeness, on_incomplete }
              # M8-2: stores VENUE OUTCOMES, not ADVERSE|BENIGN — adversity is a
              # function of (outcome, position, action) at decision time and
              # cannot be stable declarative data.
              # loss_limit REMOVED: it lives ONLY in SP-Params keyed by
              # ScopeKey.scenario. v8 had two owners ("MAY live here"), an
              # R-SSOT violation.
AdverseScenarioSet{ scenarios: {ScenarioId -> AdverseScenario}, dependence }
SP-Scenarios{ ScenarioId -> AdverseScenario }      # declarative, bitemporal
BE-ScenarioProvider -> Known[Uncertain[AdverseScenarioSet]] | Unavailable
              # ONLY when ESTIMATED; explicit joint law (§2 Dependence)
ScenarioLossLimit(ScenarioId) : a ParamId whose ScopeKey carries `scenario`
              # M7-2: without a scenario axis, multiple limits at one scope are
              # not uniquely addressable
consumers: DE-Constraints (per-scenario HARD cap) · DE-Allocator (budget split)
```
Declarative scenarios are the default; the estimated provider is the escape
hatch and is knowledge-stamped like any other belief.

---

## 9. Environment — capability slices, not a god-object (M4-3)

Only the **composition root** sees the whole `Environment{clock, feeds, venue,
rng, artifacts}`. Modules receive narrow ports, declared in their manifest:

| module | port |
|---|---|
| DA-Feeds | feed port |
| BE-*, DE-* (except Actuator) | `StateView`, RNG, artifact resolver |
| DE-Constraints | + halt port (read `HaltState`) |
| DE-Actuator | venue port + halt port (priority `cancel_all`) |
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
The flow protocol is frozen in `FLOW_MODEL_PROTOCOL_V3.yaml`; this does not
freeze or authorize the still-unbuilt decision/action layers.

## 11. Build order (demand-driven)
1. `Known[V]` + `t_known_prov` + `Unavailable` in DA (`recv_ns` is on disk).
2. EV-Markout / EV-Calibration Tier-2 artifacts on complete `full` batches.
3. Accumulate ten forward days; fit/score the per-coin B0–B3 flow baseline and
   M1–M3 conditional marks under the frozen Revision 3 protocol.
4. Fit a Hawkes residual only for coins that pass its admission gate.
5. EXP-IMPACT (AS partition sign) · EXP-BLEND (belief constituents).
6. `L_adv` + `DE-Allocator` + STOP gate before any order.

## 12. Implementation status (SHOULD-FIX 6 — v5 claimed marking, did none)

| module | BUILT | NAMED-SEAM |
|---|---|---|
| DA-Feeds | PMMarketWS, PMPricesWS, BinanceWS, GammaREST, ClobREST | PolygonRPC |
| DA-Discovery | collect_pm discovery loop | — |
| DA-Normalize / DA-State / DA-Settlement | Tier-1 v3, coverage, point-in-time views, closed-day coordinator | settlement spec adapter |
| BE-FlowFit | corrected descriptive `f_r`/same-state `f_p` measurement only | B0–B3/M1–M3 forward fit; optional gated Hawkes residual |
| other BE-* | — | Target, Uncertainty, Belief, FlowAndFills decision seam, Competition, ScenarioProvider |
| DE-* | — | all |
| EV-Markout / EV-Calibration | Tier-2 terminal markout + normalized book scaffold | fitted/scored arms after sufficient days |
| ControlSolver | — | ClosedFormGLFT, PerLevel, HJBQVI |
| UtilityFunctional | — | RiskNeutral, CARA, PathFunctional |

Nothing in BE or DE is built. The register describes contracts, not code.

## 13. Process: canonical contracts + structural diff

Five rewrites each silently lost something: v2 a MUST-FIX, v3 settlement
accounting, v4 the dependency rule, v5 `ParamId` AND the spec-record block.
Two process guards now exist, and the second exists because the first was
proven unsound.

**Guard 1 — canonical source.** `contracts/contracts.yaml` defines every type,
field, module, port and rule. The markdown explains; it does not define.

**Guard 2 — structural diff.** `contracts/contract_check.py <base-ref>` compares
**owner-qualified fields WITH their types**, so narrowing, renaming, moving and
deleting are all caught. Removals require an entry in
`removals_allowlist.yaml`. `--selftest` asserts the four regression classes,
including v8's blind spot. Subprocess failure is fatal.

**Lineage correction (v7/v8 reviews).** v7's log said the spec-record block was
dropped v5→v6. Wrong twice: it was present in v4 and gone by v5 — so **v4→v5** —
and it was never CANONICAL even in v4 (inline prose, not a fenced block).

**Why the v8 checker was replaced (review M8-3).** It scanned markdown prose,
ignored `git show` failures, could not parse generics, and used unqualified
field names. Audit, reproduced before accepting:

```
invalid baseline ref            -> exit 0, DROPPED (0)      # silent pass
type:Known / Uncertain present? -> False                    # generics invisible
narrow Known[Uncertain[JointCompetitionState]] -> CompetitionState
                                -> DROPPED (0), ADDED (0)   # IDENTICAL
```
It passed the exact regression it was built to catch. A verification that can
be fooled by its own output is worse than none, because it manufactures
confidence. Replaced, with tests.

Reproduce: `python3 live/pm_research/contracts/contract_check.py <base-ref>`
(exit 1 on any unexplained removal or type change).

## 14. Deliberately not built
No venue abstraction layer; no `VarianceBudget` class; no branches for
unreachable capabilities.
