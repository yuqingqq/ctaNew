# PM_STRUCT_ITER1_A — structure review, lens A (EXTENSIBILITY)

Object under review: `PM_ARCHITECTURE.md` (5 planes, ~20 modules, R-SSOT /
R-EXPLICIT / R-ONCE / R-PROVENANCE).
Charter: `PM_STRUCTURE_REVIEW_LOOP.md`. Iteration 1, 2026-08-20.

**Verdict: the architecture localises 5 of 13 real changes. The 5 it localises
are all PARAMETER or ESTIMATOR churn. Every change that altered a model FORM,
added a SCOPE, or added an OBJECTIVE TERM spread or was structural.** The
decision plane is cut along the stages of the closed-form GLFT pipeline
(gate → reserve → quote → lifecycle), so it forbids the base model the program
has already adopted (arXiv 2607.17991 HJB) and the theories `PM_MECHANISM_THEORY.md`
selects for M-1/M-2/M-3/M-5, all of which are joint. Mechanisms are implicit in
module internals: there is no venue-spec object, and `tick` and `rewards params`
are smuggled into a DATA-plane struct (`WindowCtx`). 8 MUST-FIX.

---

## 1. Change-log replay

Scoring: **LOCAL** = one module, no interface change. **SPREADING** = 2–5
modules or an unowned shared type, but no plane/boundary moves. **STRUCTURAL** =
an interface signature changes, a quantity has no owner, or a new axis (scope,
objective term, mechanism) must be introduced.

| # | Change | Ideal | Modules actually touched | Interface churn | Score |
|---|---|---|---|---|---|
| 1 | binary-GLFT → published PM HJB (2607.17991) | 1 | C1, C2, C3, C4 (+ γ/γ_T in K1) | **yes** — `C3.reservation(...)` does not exist in an HJB; the object is `∂V/∂(q,m)` over the joint state | **STRUCTURAL** |
| 2 | fair value: Binance-synthetic → stream-anchored | 1 | B3, B1, B2 (σ_⊥ exists only in the synthetic construction), K2, D1 | no, but a hidden upward dep: B2's output set depends on B3's impl | **SPREADING** |
| 3 | σ: static → rolling HAR → variogram (estimand was wrong) | 1 | B2 (impls already listed) | no — but `sigma_eff(r)` never declares WHICH estimand, so §16.1's 4.12× error is invisible at the boundary | **LOCAL** |
| 4 | quoting: continuous-δ → discrete per-level EV | 1 | C2 | no *now* — because B5/B6 were written post-hoc as `λ_fill(ℓ,·)`, `ζ(ℓ,·)`. That indexes BELIEF by the DECISION plane's action enum; the next action-space change breaks both | **LOCAL** (latent) |
| 5 | v(t): sum → min-structure | 1 | ownerless — v(t) is in no module's `out`. B2 owns `sigma_eff`,`sigma_perp` only; C3 uses v(t) | **yes** — a variance object with no owner | **SPREADING** |
| 6 | pull: τ_min → (\|d\|,r) surface → participation region + **size** | 1–2 | C1, C2, C4 | **yes** — `C1.permitted(ℓ,side) → bool`. The endpoint of this change is that the answer is a SIZE, not a bool | **STRUCTURAL** |
| 7 | Q_max: variance cap → loss-given-adverse + portfolio aggregate | 1–2 | C4 — but C4 must now fan in across all live markets | **yes** — no module carries a SCOPE annotation; "portfolio" is a new axis, not a new impl | **STRUCTURAL** |
| 8 | pair: net q → (q_up,q_down) joint conditions | 2 | C3, C4, C5, C2 (the joint constraint `b_up+b_down ≤ 1−target` must live INSIDE the quote solve), X2 | **yes** — `Position` is an unowned shared type; **C4 still reads `L_adv = \|q\|(1−p̂)` in the retired net-q representation** (live bug in the doc) | **SPREADING** |
| 9 | rewards: PnL line → constraint w/ shadow price → principal–agent + contest | 1–2 | homeless. Band constraint → C1; shadow price λ_R → an objective that no module owns; accrual → X2/E1; `R/X → c` equilibrium → a belief about RIVALS, absent from the belief plane | **yes** — M-5 "changes the **objective**, not just a parameter" | **STRUCTURAL** |
| 10 | siting US → EU → suspended | 0 | K1/K2 only | value-only — but `B6.zeta_snipe(ℓ, L)` collapses K2's 4-leg budget back to a scalar `L` at the consumer, which is how one quantity re-acquires three values | **LOCAL** |
| 11 | latency 120 → 471 → 1700 ms | 0 | K1 | none | **LOCAL** |
| 12 | scope: PnL-first → mechanism-first | 0 | E4 | none *in principle* — but **E4 is the only module in the register with no `in`/`out` at all**, and this change silently deleted the kill gates (FATAL-3) | **LOCAL** (conditional on E4 acquiring a contract) |
| 13 | components ADDED late (flow, propagator, book microprice, short-horizon α, cross-coin, portfolio risk) | new module only | flow→B4 ✅; microprice→B3 impl ✅; propagator→inside B6 ✅; **α (μ̂)→ buried in B1's impl with no interface**; **cross-coin→ no home (belief plane is per-market)**; **portfolio risk→ see #7** | **yes** for 3 of 6 | **STRUCTURAL** |

**Tally: LOCAL 5 · SPREADING 3 · STRUCTURAL 5.**

### Where the boundary is misplaced (the failing rows, named)

- **#1, #6, #9 — the Decision plane is cut along pipeline stages, not along
  problem parts.** C1 Gating → C3 Inventory → C2 QuotePolicy → C6 Lifecycle is
  the decomposition of the *closed-form Avellaneda-Stoikov/GLFT family*: it
  presumes a reservation price exists as a standalone object and that
  admissibility can be decided before economics. `PM_MECHANISM_THEORY.md` M-2
  states the adopted object is a value function `V(t, q, m, book state, ℓ,
  Q_ahead)` solving an HJB quasi-variational inequality, and M-5's summary line
  is that rewards "change the **objective**, not just a parameter". Neither
  survives the cut. **There is no module that owns the objective function.**
  That is why change #9 had nowhere to land and why FATAL-2 (a constraint with
  no revenue term promoted to a policy) is structurally *available* — C1 is a
  bool gate evaluated before C2's economics, and the architecture's own defence
  is a comment ("must be paired with C2 economics, never used alone"), not a
  structure.

- **#5 — R-ONCE names a rule but the register does not enumerate the quantities
  it governs.** `v(t)`, `v_unwind`, `p̂(1−p̂)`, `σ_⊥`, `κ(r)` are five variance
  objects; B2's `out` lists two. The same failure mode has now fired three
  times (T-F9 `v(t)` sum-vs-min; §16.2 σ_⊥ double-count; running-vs-terminal
  penalty). A rule you must remember is not a structure.

- **#7, #13 — there is no SCOPE axis.** Every module is implicitly per-market /
  per-window. Portfolio risk (§14 R1), capital allocation across concurrent
  windows (R3), rate-limit budget across 28 quote streams (R5), and cross-coin
  belief (X7) are all a different cardinality, not a different implementation.

- **#13 — the dependency rule forbids an adopted component.** "Planes depend
  downward only … any arrow that violates this is a bug." But
  `PM_MECHANISM_THEORY.md` requires execution→belief feedback in four places:
  `A(Q_ahead) = E[p̂_fill − p̂_t | fill at queue position Q_ahead]` (M-1),
  P-M1c (markout monotone in `Q_ahead`), P-M3c (completion probability
  correlated with the informativeness of the first fill), P-M7b, plus own-impact
  / Kyle-λ (§13.1(3): "our size is NOT negligible vs the tape"). Under the rule
  as written these are all bugs. The fix is not to relax the rule but to move
  our own resting orders and positions into the STATE plane where belief may
  legally read them.

- **#2 — B2's output set depends on B3's implementation choice.** Whether a
  basis-noise term exists at all is a property of the fair-value *source*.
  Declared arrow is B3 → B2; actual coupling runs both ways. This is the same
  shape as MAJOR-1 (σ_⊥ owned by two components at once).

---

## 2. Plug-in tests

### Test 1 — a new fair-value theory (THE KEY ONE)

`B3 → p_hat, confidence, provenance`. **Insufficient, but not for the reason the
charter suggests.**

The naive upgrade — "return a distribution, not a point" — buys *nothing here*.
The settlement outcome is binary, so its distribution IS `p̂`; `p̂` is a
sufficient statistic and M-3 confirms the up/down covariance is exactly rank 1,
`Σ = v·[[1,−1],[−1,1]]`, so the pair needs no joint object either. What
consumers need and cannot obtain from `(p̂, σ)` is four other things:

1. **The link / quantile map `G`, not its value.** M-6's model object is
   `p̂ = G(d)` with a pre-committed heavy-tailed variant (Student-t_ν or an
   empirical quantile map) under T-F14/P-M6b. But C1's frontier, B6's `ζ_snipe`
   and `λ_bin` all use `φ(d)` — the *Gaussian* density — hard-coded in the
   consumer. Swap `G` to Student-t and three downstream modules are silently
   wrong. The density/quantile map must travel with the estimate.
2. **A tail functional, not a variance.** M-7's core P&L term is
   `ζ_snipe(ℓ) = (λ_J/λ_total)·E[(|J| − m(ℓ))^+]` — an expectation over the
   *jump-size law* above a moat. σ cannot produce it.
3. **The law of the future BELIEF PATH.** Running inventory penalty
   `v_unwind = ∫λ̂_bin²`, the queue option value `D_ℓ(Q_ahead)`, and the pickoff
   floor `E[|Δp̂ over requote latency|]` are all functionals of
   `{p̂_s}_{s∈[t,T]}`, not of `(p̂_t, σ_t)`.
4. **The constituents, not the blend.** `Blend(w)` is listed as an *impl* of B3,
   which HIDES `p_book` inside B3. E-X2 — the pivotal experiment, whose three
   outcomes decide whether this is an alpha strategy or a spread-capture
   strategy — requires a paired ΔBrier of `p̂_model` vs `p_book`.
   **E-X2 is not expressible through B3's interface.** It would have to be
   re-implemented outside the module, and then production and experiment
   diverge — the MAJOR-4 pattern, structurally invited.

Also: B3's `in:` is an *enumerated list of modules* (B1, B2, MarketState,
optional Binance alpha). A learned p̂ needs an open feature set, so every new
input becomes an interface edit. And a Kalshi-style settlement has no `(K, X_T)`
threshold shape at all — B1's `out: K, E_t[X_T]` is over-fitted to
"scalar above a strike" (see Test 5).

**Verdict: FAILS.** Fix = MF-3 (`BeliefProcess`), MF-4 (`Uncertain[T]`).
Side benefit: once the increment LAW is the primary object, "add a second
variance term" becomes a type error rather than a discipline problem — which
retires the failure mode that has now fired three times.

### Test 2 — a new mechanism (batch auction / maker-priority / negative-risk)

**No mechanism registry. Mechanisms are implicit in module internals**, and
partly in the wrong plane: `WindowCtx{t0, T, w, K, coin, tick, rewards params}`
puts venue rules inside a DATA-plane struct, mixing a window *instance* (t0, T,
K — changes every 5 min) with a venue *spec* (tick, rewards params, w
convention — changes when the venue changes its rules, as it did 2026-08-07).
Fees (M-4) and min-size / rate limits (M-2) appear in **no** module at all.

Blast radii today:

| New mechanism | Modules invalidated | Interface changes |
|---|---|---|
| Batch auction (Budish et al.) | D3 (`queue_est` meaningless), B5 (replace), B6 (`ζ_snipe`→0), C1 (frontier moot), C2 (action = a bid schedule), C6 (requote moot), K2 (latency moot) | 3+ |
| Pro-rata / maker-priority | D3.`queue_est`, B5, C2 (`ℓ` semantics change) | 1–2 |
| negRisk multi-outcome | C5 (`in: (q_up,q_down)` is hard-coded to 2 tokens), X2, C4, C3 (reservation over a k-simplex) | 2+ |

**Verdict: FAILS.** Fix = MF-2 (`VenueSpec` + per-module `REQUIRES`), so an
unsupported mechanism fails loud at wiring time instead of being silently
mis-modelled.

### Test 3 — a theory that spans modules (joint quoting + inventory)

**Yes — the plane decomposition forbids legitimate, already-adopted theories.**
Four of the eight adopted theories refuse the C1..C6 split:

- M-2 Guilbaud-Pham: `V(t, q, m, book, ℓ, Q_ahead)` HJB-QVI. Requote timing and
  level are one impulse-control problem → C2 and C6 are one module.
- M-1: `V(ℓ) = F_ℓ·[P_ℓ − CE_quote(q) − A(Q_ahead) + ρ(P_ℓ)] + D_ℓ(Q_ahead)`
  — level EV embeds the inventory certainty-equivalent (C3), the fee schedule
  (unowned), and a belief conditioned on our own fill (execution→belief).
- M-3: the pair is a **joint constraint inside the quote solve**
  (`b_up + b_down ≤ 1 − target`, gated on `P(complete)` vs legging cost `κ`),
  not a post-hoc merge scan. C5 sits *after* C2 in the register; it cannot
  express this.
- M-5: constrained control with a KKT shadow price; occupancy is an *integral*
  and fill EV is *per-fill* — theory note: "the two have no common unit until
  the M-5 sampling model is written down". So objective terms need declared
  UNITS, and one module must own their assembly.

**Verdict: FAILS.** Fix = MF-1 (re-cut Decision into
Objective / Constraints / Solver / Actuator + an explicit `ActionSpace`).
Under that cut, change #1 is one Solver impl, #9 is one objective term plus one
constraint, #6 is a constraint that returns a size, #4 is an ActionSpace swap.

### Test 4 — a new evaluation metric / a new gate

**E4 is the only module in the register with no interface** (`out:` absent).
`PM_PREREG.md` does not exist. There is no way to state what a gate *is*, so
"add a gate without touching experiments" has no meaning yet. Additionally
MAJOR-8 shows gates were scheduled *before their own preconditions* (freeze
2026-08-21, E0 reads 2026-08-22) and MAJOR-5 shows an inference convention
(notional weighting) transplanted across gates with different units. Both are
schema problems.

**Verdict: FAILS (unbuilt, and unspecified).** Fix = MF-7 (MetricRegistry +
declarative `Gate` with a precondition DAG, per-gate weighting/inference, and
`kind='stop'` so FATAL-3's STOP clause has a home and an owner field).

### Test 5 — a second venue running simultaneously (Polymarket + Kalshi)

Five breaks:
1. **No identity spine.** `QuoteIntent{token, side, px, size}` has no venue.
   Every module is an implicit singleton.
2. **K1 SSOT becomes wrong, not just incomplete.** "One owning module per
   quantity" with a scalar value means one `L` for two venues. SSOT must be
   *keyed*: `(venue, coin, horizon, date)`.
3. **X2 and C5 are CTF-specific.** mint/merge/redeem is a Polymarket
   mechanism; Kalshi has none. Belongs in `VenueSpec.conversion`.
4. **Risk must aggregate across venues, and cannot.** A Kalshi BTC-up and a
   Polymarket BTC-up are the SAME risk. There is no canonical
   `RiskFactorId{underlying, horizon, settle_time}`.
5. **B1 mixes world and market.** `E_t[X_T]` is a belief about the *underlying*
   (shared across venues); `K` and the payoff rule are *instrument* properties.
   Cutting them apart makes a second venue reuse the expensive half.

**Verdict: FAILS.** Fix = MF-6. Note this is the same fix the program needs
anyway for X8 (15-min / 4-hour horizons) and for the pre/post-2026-08-07 rule
epochs.

### Test 6 — swap the queue/fill bracket for a different uncertainty representation

B5: "bracketed (pessimistic/optimistic)", with a note that the bracket belongs
on `Q_ahead`. The *2-point* representation is an unnamed convention baked into
the interface. Consumers branch on it (`for b in {pess, opt}`), E3's queue
bracketing hard-codes it, and the inherited gate rule ("sign-flip across the
bracket = failure") is written for exactly 2 points. Moving to a posterior over
`Q_ahead`, a fill-time distribution, or a robust ambiguity set edits every
consumer.

**Verdict: FAILS.** Fix = MF-4 — one uniform `Uncertain[T]` with a required
`scenarios(n)` accessor. Consumers never branch on representation; the gate rule
generalises to "verdict stable across scenarios" (bracket = n·2). This is the
same fix as Test 1's, applied to a different quantity — which is the point: one
mechanism, not three.

---

## 3. Missing extension points

Predictable needs with no home today.

| # | Missing | Why it will be needed | Evidence it is already needed |
|---|---|---|---|
| X-1 | **VenueSpec / MechanismRegistry** | mechanisms are the program's subject | M-4 fees and M-2 min-size/rate-limits own no module; tick and rewards params hide in `WindowCtx` |
| X-2 | **Objective term registry** (name, value, sign, **unit**, source, R-ONCE key) | new theory usually = a new term | rewards shadow price, 0.35-tick rebate, merge option value, `D_ℓ` all currently homeless; M-5's occupancy-integral-vs-per-fill unit mismatch |
| X-3 | **Signal / feature registry** | the restored "signal half" (X2/X3/X6) | μ̂ is buried inside B1's impl; E-X3's decision rule is "adopt only in the strata where it wins" — that is config, not code |
| X-4 | **Regime / scenario tagger (D4)** | strata appear in gates, σ fitting, X3 adoption, stand-down | rule epoch (pre/post 2026-08-07), tick regime (0.01→0.001, 328 events/130 windows), 56 data gaps >5 s, scheduled events (CPI/FOMC — §12.3's named blind spot), MAJOR-6's two unmodelled regimes. Without an owner these get re-derived inconsistently per consumer |
| X-5 | **Window-phase / lifecycle owner** | legal actions differ by phase | MAJOR-6: `t<0` (strike forming) and `t>T` (redemption lag, measured **−806 bps**) have no model; "§2's state space stops at T" |
| X-6 | **StrategyConfig manifest** | §3 claims Option A/B is a config choice | there is no config object; today "selecting implementations" is an edit |
| X-7 | **Versioned model artifacts (K4)** with a PIT guard | B2 variogram, B3 blend `w`, B5 queue model, B6 ζ surface are FITTED, walk-forward (fit ≤ d−1, apply d) | K1 holds *parameters*; nothing holds *fits*. Repo pitfalls #1/#2/#5 are all this class |
| X-8 | **A/B harness contract on E3** | every module has ≥2 impls by design | must support paired same-path replay with **overlays disabled** (repo pitfall #4: path-coupled overlays amplify prediction noise 10–20×) |
| X-9 | **Rival / competition belief (B7)** | Tullock equilibrium `R/X → c`; the marginal sniper's latency differential (§12.1 I-3); "is the book picked over" (§17.4) | rival behaviour is currently smuggled into C1's frontier constants |
| X-10 | **Cross-market belief scope** | X7: a BTC move informs the ETH window | belief plane is per-market; §14 R1 uses a crude worst-case bound *because* no joint object exists |
| X-11 | **Provenance on DERIVED fields, not just params** | the venue's central economic sign rests on one field semantic | D2's `side_signed` (side×asset_id→direction): if WS `side` is the maker's, maker gross flips +95 → −95 bps. Under R-PROVENANCE `assumed` may not gate a decision — but D2's `out` carries no provenance |
| X-12 | **Invalidation / dependents index** | MAJOR-4: an amendment logged but never applied | if each module declares the fields it CONSUMES, changing a field mechanically lists what it invalidates |

Namespace hazard: `X1`/`X2` mean *propagator* and *PM book microprice* in
`PM_MM_PLAN.md` §13.2, and *VenueAdapter* and *SettlementAccounting* in
`PM_ARCHITECTURE.md`. Same corpus, same tokens, different referents.

---

## 4. Triage

### MUST-FIX

**MF-1 — re-cut the Decision plane along problem parts, not pipeline stages.**
Fixes changes #1, #4, #6, #9 and Test 3. Collapses C1+C4 (both emit constraints
on the same object) and C2+C3+C5+C6.

```python
ActionSpace{ kind: 'per_level'|'continuous_delta'|'batch_schedule'|'size_ladder'
             enumerate(state, venue_spec) -> [Action] }
Action = Quote{instrument: InstrumentId, side, px, size}
       | Cancel{order_id} | Convert{kind: 'mint'|'merge', n} | NoOp

# C-OBJ Objective — THE missing module. Owns the value being maximised.
Objective.terms(state) -> [Term{name, value(action) -> float, unit: '$'|'$/s',
                               sign, source: 'M-4'|'M-5'|..., provenance}]
  # registered once per name (R-ONCE enforced structurally, not by memory):
  # spread_capture · maker_rebate ρ(P) · taker_fee · adverse_selection ζ(P)
  # · running_inventory · terminal q²p̂(1−p̂) · rewards_shadow λ_R·x_t
  # · merge_option_value · queue_option D_ℓ
Objective.value(state, action) -> float          # units reconciled here

# C-CON Constraints — replaces C1 Gating + C4 RiskLimits.
Constraints.feasible(state, action) -> Feasibility{
    max_size: float,            # 0 == forbidden.  NEVER a bare bool  (change #6)
    binding: [name], shadow_price: {name -> float}, reasons: [...] }
  # a binding constraint REPORTS ITS PRICE -> the solver sees it ->
  # FATAL-2 ("a constraint promoted to a policy, with no revenue term")
  # becomes structurally impossible rather than commented against.

# C-SOL Solver — the base model. ONE impl swap == change #1.
Solver.solve(belief: BeliefProcess, self_state: SelfState, objective, constraints,
             action_space, horizon) -> [Action]
  impls: GreedyPerLevelEV | HJB_2607_17991 | ImpulseControlQVI | TwoSidedBand(OptionB)

# C-ACT Actuator — replaces C6.
Actuator.reconcile(desired: [Action], live_orders, limits: VenueSpec.limits)
    -> [OrderOp]
```

**MF-2 — add `VenueSpec` (cross-cutting) + per-module `REQUIRES`.** Fixes
Test 2; removes venue rules from `WindowCtx`.

```python
VenueSpec{
  venue_id, market_type,
  matching{ rule: 'price_time'|'pro_rata'|'batch_auction'|'maker_priority',
            batch_interval_ms: Optional[int], priority_key: [...] },
  grid{ tick(px) -> float,          # state-dependent: 0.01 -> 0.001 near extremes
        min_order_size, min_size_for_rewards, price_bounds: (0.01, 0.99) },
  fees{ taker(px, size) -> float, maker(px, size) -> float, settlement },
  conversion{ tokens: [id], relation: 'sum_to_one'|'negrisk_k'|'none',
              mint(...), merge(...), latency_ms, gas },        # k, not 2
  rewards: Optional{ band(mid) -> (lo,hi), score(z,s), min_size, rate, epoch },
  settlement{ source, estimand_expr, window_w, tie_rule, lock_dynamics,
              boundary_reader b(·) },
  limits{ orders_per_s, cancels_per_s, scope },
  phases: [ 'pre_window','in_window','in_lock','post_T','redeemed' ],  # X-5
  provenance: {field -> Provenance} }

# every module declares what it assumes about the mechanism:
class B5_FillModel: REQUIRES = {'matching.rule': ['price_time','pro_rata']}
# wiring a VenueSpec that fails a REQUIRES raises AT WIRING TIME. A new venue
# then tells you exactly which modules are invalid instead of mis-modelling.
```

**MF-3 — `B3` returns a `BeliefProcess`, not `(p_hat, confidence)`.** Fixes
Test 1 and makes E-X2 expressible against the production module.

```python
B1.settlement(as_of) -> SettlementSpec{
    payoff(token, terminal_state) -> float,   # generalises "Up iff X_T >= K"
    target: Uncertain[float],                 # E_t[X_T]
    estimand: 'E[X_T | F_t]',                 # DECLARED, not implied
    strike: Optional[float], locked_fraction, provenance }

B2.variances(as_of, r) -> VarianceSet{
    sigma_eff: Uncertain[float],
    estimand: 'sd_t[X_T | F_t]',   # NOT Var[X_{t+r} − X_t]; §16.1 was 4.12x
    sigma_perp: Optional[float],   # None when the source IS the settlement stream
    v_hold(q, p_hat), v_unwind(horizon),
    compose: 'min',                # DECLARED operator. Summation is illegal.
    provenance }

B3.fair_value(as_of, features: FeatureSpec) -> BeliefProcess{
    p_hat: Uncertain[float],
    link: LinkMap{ G(d) -> p, density(d) -> float, quantile(u) -> d,
                   family: 'gaussian'|'student_t'|'empirical' },   # M-6 / T-F14
      # consumers call belief.link.density(d), NEVER a hard-coded phi(d)
    increments{ var_term_structure(h) -> float,      # Var_t[p_hat_{t+h}]
                tail(h, threshold) -> float,          # E[(|J| − m)^+]  (M-7)
                jump_intensity(threshold) -> float,   # lambda_J
                paths(n, horizon) -> [array] },       # for solver + replay
    components: {name -> (p_component, weight, provenance)},  # p_model, p_book, p_flb
      # <- E-X2 (paired dBrier of model vs book) now runs against production
    estimand: 'P(token pays 1 | F_t)', provenance }
```
Note: a distribution over the *outcome* is not the fix — for a binary it is
`Bernoulli(p̂)` and adds nothing. The missing object is the law of the
belief PATH plus the link map.

**MF-4 — one uncertainty representation, used everywhere.** Fixes Test 6;
applies to `p̂`, `σ_eff`, `Q_ahead`, `ζ`.

```python
Uncertain[T]{ point: T,
              repr: 'point'|'bracket'|'dist'|'ambiguity_set',
              scenarios(n, rng) -> [(T, weight)],   # REQUIRED for every repr
              interval(alpha) -> (lo, hi),
              estimand: str, provenance }
# Consumers MUST NOT branch on `repr`. The inherited gate rule
# "sign flip across the bracket = failure" generalises to
# "verdict stable across scenarios"; bracket is just n=2.
```

**MF-5 — put our own state in the STATE plane so belief may legally read it.**
Fixes the rule-vs-theory contradiction in change #13 (own-impact, queue-reactive
intensities, `A(Q_ahead)`, `P(complete)`). Cheap.

```python
D3.as_of(t) -> State{ market: MarketState, settlement: SettlementState,
                      window: WindowCtx{t0, T, phase, K},   # venue spec REMOVED
                      tape: Tape,
                      self: SelfState{ live_orders, queue_positions, positions,
                                       recent_fills, rate_budget_used } }
# Dependency rule amended: Belief may read State.self. This is NOT an upward
# arrow — our own orders are observed data. The rule stays "no module reads a
# module it does not depend on"; only the placement of self-state changes.
```

**MF-6 — scope annotation + identity spine.** Fixes change #7 and Test 5.

```python
InstrumentId{venue, market_id, token_id}
RiskFactorId{underlying, horizon, settle_time}
# every module declares SCOPE in {tick, window, instrument, market, coin,
#                                 portfolio, program}
C4a MarketRisk    SCOPE=market
C7  PortfolioRisk SCOPE=portfolio
C7.limits(positions: {InstrumentId -> float}, beliefs, factor_map)
     -> ({InstrumentId -> max_size}, breaches, binding_factor)
   # L_adv per instrument = sum_token q_token * (1 − payoff_prob_token)
   # ^ fixes the LIVE BUG: C4 currently states |q|(1−p_hat) in the RETIRED net-q form
C8  CapitalScheduler SCOPE=portfolio     # R3 capital velocity
C9  RateBudget       SCOPE=venue         # R5, 28 quote streams
K1.get(name, key={venue, coin, horizon, date}) -> (value, Provenance)
   # SSOT must be KEYED, or a second venue silently re-creates the 120/471/1700 bug
```

**MF-7 — give E4 a contract.** Fixes Test 4 and houses FATAL-3.

```python
MetricRegistry.register(id, fn(run) -> series, unit, higher_is_better)
   # E1 registers markout(h), capture_ratio; E2 registers dBrier, calibration_slope
Gate{ id, metric, statistic, strata_ref: RegimeTagger.strata_id,
      inference{cluster: ['day','window'], bootstrap: 'block',
                family: str, alpha: float, weighting: 'equal'|'notional'},
      threshold, direction, kind: 'go'|'no_go'|'stop',
      owner: str, frozen_at: date, reads_after: date,
      preconditions: [gate_id] }     # MAJOR-8: a gate may not freeze before its
                                     # preconditions read. Checked, not trusted.
E4.evaluate(gate_id, artifact) -> GateResult{value, ci, verdict, provenance}
   #   verdict in {PASS, FAIL, PROVISIONAL, UNREADABLE}
E4.program_state() -> 'RUNNING'|'SUSPENDED'|'STOPPED'
# new metric = one register(); new gate = one registry row; experiments untouched.
```

**MF-8 — enumerate every variance and every objective term, with a composition
operator and a unit.** Fixes change #5; retires the failure mode that has fired
three times (T-F9, §16.2, running-vs-terminal). Covered by the `VarianceSet`
`compose` field in MF-3 and the `Term{unit}` registry in MF-1; listed separately
because it is a *register completeness* requirement on `PM_ARCHITECTURE.md` §2,
not only a signature.

### SHOULD-FIX

- **SF-1** `D4 RegimeTagger` (X-4): emits `RegimeTags{rule_epoch, tick_regime,
  data_gap, event_calendar, vol_regime, session}`; owns the strata ids that
  gates, σ fitting and E-X3's stratified adoption all reference. One owner, or
  they diverge.
- **SF-2** `SignalRegistry` (X-3): `signal(id) -> (value, horizon, eligibility
  strata, provenance)`, so μ̂ stops being buried in B1's impl.
- **SF-3** `K4 ArtifactStore` (X-7): `{fit_id, module, data_range, fitted_at,
  code_hash, params, applies_to}` with a PIT guard — an artifact may not be
  applied inside its own fit range.
- **SF-4** `E3.replay(config, range, n_scenarios, overlays='off')` and
  `E3.ab(cfg_a, cfg_b, ...) -> PairedResult` on an identical path; overlays OFF
  by default per repo pitfall #4.
- **SF-5** `StrategyConfig{variant_id, impls, params_ref, action_space,
  venue_spec_ref, gates_ref, artifact_pins}` — makes §3's Option A/B claim true.
- **SF-6** `B7 Competition`: rival latency differential, `X` (aggregate band
  occupancy from the book), the `R/X → c` equilibrium.
- **SF-7** `K2` hands consumers a `LatencyBudget` object (4 legs + rival
  differential), never a scalar `L`. `B6.zeta_snipe(ℓ, L)` currently re-collapses
  it.
- **SF-8** Cross-market belief scope (X-10) for X7 and for R1's correlated tail.
- **SF-9** Provenance on D2's derived semantics (X-11), starting with
  `side_signed`.
- **SF-10** Module `CONSUMES` declarations → a dependents index (X-12), so an
  amendment mechanically lists what it invalidates (MAJOR-4's requested rule
  "an amendment entry must cite the line it changed", made structural).

### NOTED

- **N-1** Build order §4 is right, and MF-2 belongs in step 1 with K1/K2: it is
  equally cheap and D2/D3 are blocked on it (tick grid, dedup keys, phases).
- **N-2** §3's "Option B is this system with simpler implementations of three
  modules" is optimistic under the current cut — B also has a different
  objective and different gates. Under MF-1 it becomes true.
- **N-3** `B4.burst_flag` is a boolean derived from a threshold, i.e. a decision
  parameter leaking into the belief plane. M-7 demotes the burst flag to a
  model-validity monitor; the interface should carry the continuous quantity.
- **N-4** E1/E2 are marked ✅ but consume `Fills`/`Positions`, which do not
  exist. The ✅ is inherited methodology, not a working module.
- **N-5** `X1`/`X2` namespace collision between `PM_MM_PLAN.md` §13.2 and
  `PM_ARCHITECTURE.md`'s Execution plane. Rename one set.
- **N-6** Horizon (5-min / 15-min / 4-hour, X8) is a config dimension for params,
  gates and strata; MF-6's keying covers it if `horizon` stays in the key.

---

## 5. What the architecture already gets right

Recorded so the loop does not churn it: R-SSOT + K1 genuinely localise changes
#10 and #11 — the two that produced FATAL-1 — and that is the single highest-value
thing in the document. B2's impl list localises #3. The B3-impl framing of the
A/B decision is the right *idea* (it just needs constituents exposed). The
Data-plane split (wire / normalise / point-in-time) is clean and its `as_of(t)`
rule is the correct construction-level defence. Build order §4 puts the cheap
error-retiring plumbing first, which is the correct response to
"the program built theory faster than it built arithmetic".
