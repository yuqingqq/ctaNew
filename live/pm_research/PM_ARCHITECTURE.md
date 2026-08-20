# PM_ARCHITECTURE v4 — structure for P-2026-003

Rewritten 2026-08-20 from a user review of v3 (9 MUST-FIX). v3 fixed the
*naming* of several contracts without making them *usable*; v4 gives them
signatures. Prior: v1 5/3/5, v2 7/4/2 on the 13-change replay.

Naming: SP/DA/BE/DE/EV/OP for modules; `EXP-*` for experiments.

---

## 0. Rules and enforcement

| Rule | Enforcement |
|---|---|
| **R-SSOT** | all spec records + params are read-only handles; no restatement |
| **R-KNOW** | `Known[V]`; every read is knowledge-truncated (§4); no API accepts event time |
| **R-ONCE** | variance AND PnL, via the declared **partition** (§7), not a class |
| **R-PROV** | `assumed` may not gate; fitted artifacts carry `fit_data_through` |
| **R-VERSION** | specs carry BOTH knowledge and validity axes (§3) |
| **R-NULL** | field-level, with bias direction |
| **R-REQUIRES** | capability declarations; wiring fails loud |
| **R-ENV** | every stateful module takes an `Environment`; nothing reads a wall clock |

---

## 1. Layers and the dependency rule

```
SP  SPECS (data: one YAML + a small loader, NOT a subsystem)
DA  DATA      Discovery · Feeds · Normalize · State(+SelfState) · Settlement
BE  BELIEF    Target · Uncertainty · Belief · FlowAndFills
DE  DECISION  ActionSpace · Constraints · DecisionScheme · Allocator · Actuator
EV  EVAL      Markout · Calibration · Attribution · Replay · Gates
OP  OPS       LatencyBudget · Monitor/KillSwitch
```
**Dependencies point downward only: SP ← DA ← BE ← DE. EV reads all planes and
is read by none. OP is readable by all and depends on none.**

(v4 initially dropped this rule — the third consecutive rewrite to lose
something silently — and immediately violated it by having `DE-Allocator` read
`EV-Settlement`. Resolution: **settlement FACTS are DATA, not evaluation.**
`DA-Settlement` owns resolutions/redemptions/lockup — 1,292 resolutions are on
disk — and `DE-Allocator` reads it downward. `EV-Attribution` keeps
performance measurement, which nothing reads back.)

## 2. Identity and cardinality (v3 had none — cause of 2 STRUCTURAL rows)

```
VenueId · InstrumentId{venue, symbol, horizon, expiry} · TokenId{instrument, side}
RiskFactorId          # e.g. BTC — MANY instruments map to ONE factor
PortfolioId
```
Cardinality is explicit everywhere: a strategy runs `{VenueId} × {InstrumentId}`
concurrently, and risk aggregates over `RiskFactorId`, not over instruments.
Cross-coin, multi-horizon and second-venue therefore become CONFIGURATION.

`SP-Params` is keyed by **`(name, scope)`** where
`scope = Global | Venue(v) | Factor(f) | Instrument(i)`; scope is part of the
key, not metadata. Resolution is most-specific-wins, and the resolved key is
recorded in provenance.

---

## 3. SP — specs as data, on TWO time axes

v3 replaced `valid_from/valid_to` with `hash + observed_at`. Both are needed:
the hash says *what we knew*, validity says *when the fact applied*.

```
SpecRecord{ hash, observed_at, source,           # knowledge axis
            valid_from, valid_to,                # event/validity axis
            fields: {name -> FieldState} }
FieldState = Resolved(value) | Disputed([values, sources]) | Unknown
```
`Disputed` is live today: Gamma and the CLOB registry disagree on the rewards
band after the 08-20 re-cut, and both sit in `markets.jsonl`. A consumer that
reads a `Disputed` field must handle it; it cannot silently pick one.

Records: `SP-Venue{matching, tick_grid, min_size, rate_limits,
fee_schedule(p, side, size), rebate_schedule, rewards_band, capabilities{...}}`
· `SP-Instrument{settlement{source, statistic, w_declared, strike_rule,
tie_rule}, T, payoff, complement}` · `SP-Strategy{scheme, constraints,
action_space, impls, nulls{field: (assumption, bias_direction)}}` ·
`SP-Params{(name,scope) -> ParamValue{value, provenance, owner, valid_for,
measured_at, fit_data_through?, artifact_id?}}`

**Populate on demand** — EV-Markout needs ~4 facts, not 30 fields.

---

## 4. DA — the knowledge-time contract, with the bypasses closed

v3 fixed `view()` but left three routes around it. All reads are now
knowledge-truncated, and absence is typed.

```
Known[V]{ value, t_event, t_known, t_known_prov: OBSERVED|IMPUTED|ASSUMED,
          t_known_err, source, provenance }
Uncertain[T]                       # carries the bracket/distribution, not 2 scenarios
Unavailable{ reason, since }       # typed; propagates; never silently a default

DA-State.view(now) -> StateView                       # the ONLY entry point
StateView.coverage(field, span) -> Coverage           # bound to the view's `now`
StateView.get(field) -> Known[V] | Unavailable

compose(f, inputs):  t_known = MAX(t_known_i)
                     t_known_err = combine(err_i)        # declared, not implicit
                     prov  = weakest(prov_i)             # OBSERVED > IMPUTED > ASSUMED
                     staleness/deficit = declared per field, never a scalar sum
```
No API accepts an event time. `t_known` on historical (REST/on-chain) data is
`IMPUTED` or `ASSUMED` — the natural `t_known := t_event` is precisely the
type-laundered look-ahead, so replay refuses when `r` falls inside
`t_known_err`.

**SelfState exposure envelope** (v3 named three clocks and used none):
```
SelfState.envelope(now) -> { submitted, acked, filled, in_flight,
                             worst_case_exposure }   # bracket, not a point
```
The ack leg is currently `ASSUMED 75 ms`, unobserved, and errs both ways.

---

## 5. BE — belief

`BE-Target` (`K`, `E[X_T]` from `w_declared`) · `BE-Uncertainty` (σ_eff,
`w_hat`) · `BE-Belief -> BeliefProcess{ p_hat, link G, jump_tail?, path_law?,
constituents{name -> (value, weight, provenance)}, staleness }` with
**field-level** nullity.

**BE-FlowAndFills — one vocabulary for ALL actions** (v3's `rest()` could not
price cancellation, crossing, mint/merge, or a batch/AMM action):
```
evaluate(action_set, StateView, SelfState, horizon)
    -> Uncertain[ActionOutcome] | Unavailable
```
Pair-aware by construction: the PM book is unified across the token pair (Up
@0.60 crosses Down @0.40 via CTF mint at match time), so `action_set` is
evaluated jointly, not per token.

---

## 6. DE — decision

Outer seam is coherent; internals are independently swappable.

```
DecisionProblem{ view: StateView, self: SelfState, belief: BeliefProcess,
                 outcomes: (action_set -> Uncertain[ActionOutcome]),
                 actions: ActionSpace, portfolio: PortfolioState,
                 constraints: FeasibleSet, horizon, env: Environment }

DecisionScheme.solve(DecisionProblem) -> Decision{ actions, duals, rationale }
```
Two INDEPENDENT axes (v3 conflated them):
- **valuation/control form**: `ClosedFormGLFT` · `PerLevelEV` · `HJBQVI` · `CARA-CE`
- **scope/action coupling**: `PerToken` · `JointPair` · `PortfolioJoint`

Each declares compatibility; the wiring rejects an incompatible pairing. So a
new control theory does not have to reimplement pair coupling, and vice versa.

`DE-Constraints.feasible(action_set, state) -> FeasibleSet` declares each
constraint's FORM; **the scheme computes duals** (`∂V/∂x` is a solver object).

`DE-Allocator` — splits capital/risk budget across `{InstrumentId}` and
`RiskFactorId`; owns the portfolio-level constraint. Without it, portfolio risk
was a structural change.

---

## 7. The PnL partition (R-ONCE for economics)

A markout ALREADY contains spread capture, adverse selection and own impact.
v3 assigned them to different owners without defining the split, which is the
double-count it claimed to prevent. `ActionOutcome` must declare a partition
that sums to the measured quantity:

```
spread + transient_AS + permanent_AS + snipe + own_impact  =  markout(τ)
```
Each component carries `{unit, conditioning_measure, sign_convention,
coverage}`. Anything not in the partition is not an economic term. Rewards are
NOT in it — they are a constraint, priced by the scheme's dual.

---

## 8. Risk — from cost basis, not from p̂

v3's `L_adv = |u|·(1 − p̂_side)` is ambiguous and effectively sign-reversed: a
held Up share at `p̂ = 0.98` is charged `0.02`/share while `~0.98`/share of
premium is lost if Down resolves. It charges LEAST where loss is LARGEST — the
same picking-up-pennies error, for the third time, inside its own fix.

```
m = min(q_up, q_down)                       # paired, riskless, redeems $1
L_adv = unpaired_cost_basis(q_up, q_down, m)   # premium actually at risk
per market:  L_adv ≤ κ_$
per factor:  Σ_{i ∈ RiskFactorId f} L_adv_i ≤ L_max(f)     # correlated tail
```

---

## 9. Environment seam (replay parity)

EV cannot drive the stack without inverting the dependency rule. It doesn't:
the stack depends on an `Environment`, and live/replay/sim are implementations.

```
Environment{ clock, feeds, venue, rng, artifacts }
   impls: LiveEnv · ReplayEnv · SimEnv
```
Contracted: deterministic tie-breaking; a warm-state snapshot with an explicit
restart-parity rule; fitted-artifact resolution by `artifact_id` +
`fit_data_through` (so a fitted module cannot re-enter the system with future
knowledge); RNG owned by the Environment; spec-hash pinning per run.
`EV-Replay` constructs a `ReplayEnv`; it never calls modules directly.

---

## 10. EV / OP — program control

**EV-Gates**
```
Gate{ id, question, metric, threshold, unit, data_prereq, owner,
      preconditions: [GateId], on_pass, on_fail }
```
Gates form a **precondition DAG**; a gate is unreadable until its predecessors
pass. The registry lives in `PM_PREREG.md` (does not yet exist — nothing is
currently frozen).

**STOP clause** — a first-class `Gate` with `on_fail = HALT_PROGRAM`, a named
owner, and a review date. Its absence was FATAL-3.

**EV-Attribution** — performance decomposition over the §7 partition; read by
nobody (per the dependency rule). Settlement FACTS live in `DA-Settlement`
(§1) and are read downward by `DE-Allocator`, so redemption delay and capital
lockup enter decisions.

**EV-Markout · EV-Calibration** ✅ mature. **OP-LatencyBudget** — four legs, ack
unobserved. **OP-Monitor/KillSwitch** — stand-down triggers.

---

## 11. Build order (demand-driven)

1. `Known[V]` + `t_known_prov` + `Unavailable` in DA — `recv_ns` is on disk.
2. EV-Markout / EV-Calibration on real state (~4 spec facts).
3. EXP-IMPACT (AS partition sign) · EXP-BLEND (belief constituents).
4. `L_adv` on cost basis + `DE-Allocator` + STOP gate before any order.

## 12. Deliberately not built
No venue abstraction layer; no `VarianceBudget` class; no branches for
unreachable capabilities; solver impls marked BUILT vs NAMED-SEAM.
