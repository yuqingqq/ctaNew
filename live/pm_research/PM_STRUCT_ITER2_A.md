# PM_STRUCT_ITER2_A — structure review, lens A (EXTENSIBILITY)

Object under review: `PM_ARCHITECTURE.md` **v2** (rewritten 2026-08-20).
Charter: `PM_STRUCTURE_REVIEW_LOOP.md`. Iteration 2, 2026-08-20.
Predecessor: `PM_STRUCT_ITER1_A.md` (v1 scored LOCAL 5 / SPREADING 3 / STRUCTURAL 5,
0/6 plug-in tests).

**Verdict: real improvement, target missed.** Change-log replay
**LOCAL 7 / SPREADING 4 / STRUCTURAL 2** (target: 0 STRUCTURAL, ≤1 SPREADING).
Plug-in tests **1 pass / 2 partial / 3 fail** (was 0/6). The five root causes v2
names are genuinely fixed and I do not want the loop to churn them (§7). What v2
did **not** fix is my MF-6 (scope + keyed identity), which is the sole cause of
both remaining STRUCTURAL rows and of the second-venue failure; and the rewrite
introduced **six new defects**, one of which is an arithmetic error in the one
risk formula v2 chose to write out (`L_adv`, §5.1) that would forbid the
pair-harvest strategy the programme is built on.

**Scoreability caveat, stated up front.** v1 carried a per-module `in`/`out`
register; v2 does not. Eleven of ~20 modules have no signature at all
(`DA-Discovery`, `DA-Feeds`, `DA-Normalize`, `BE-Target`, `BE-Uncertainty`,
`DE-ActionSpace`, `DE-Objective`, `DE-Actuator`, all four `EV-*`, both `OP-*`).
`DE-Objective` — the module the whole re-cut exists to create — is specified as a
list of seven words. Where a row below is scored on a module with no signature I
say so; "unscoreable" is not "pass". This is SF-A1.

---

## 1. ACCEPTANCE TEST — the 13-change replay against v2

Same scoring as iteration 1. **LOCAL** = one module, no interface change.
**SPREADING** = 2–5 modules or an unowned shared type, no new axis.
**STRUCTURAL** = an interface signature changes, a quantity has no owner, or a
new axis (scope, objective form, mechanism) must be introduced.

| # | change | ideal | v2 modules touched | score | vs ideal |
|---|---|---|---|---|---|
| 1 | binary-GLFT → published PM HJB (2607.17991) | 1 | `DE-Solver` (impl ✅) **+ `DE-Objective`** (Σ-of-terms cannot express M-8's CARA `CE(q)`, which is the QVI's terminal condition and is *not* additive) **+ `DE-ActionSpace`** (in M-2, `ℓ` and `Q_ahead` are **state**, and the impulse is the control) **+ `DE-Constraints`** (`z ≥ 5`, `z ≥ 50` in-band and the tick lattice are the *admissible control set*, inside the operator) | **SPREADING** (4) | MISS |
| 2 | fair value: Binance-synthetic → stream-anchored | 1 | `BE-Belief` impl. `link: G` travels with the estimate so no consumer hard-codes Φ; σ_⊥ simply stops registering with the `VarianceBudget` | **LOCAL** | MEET ✅ |
| 3 | σ: static → HAR → variogram | 1 | `BE-Uncertainty` impl | **LOCAL** | MEET (caveat NF-6) |
| 4 | continuous-δ closed form → discrete per-level EV | 1 | `DE-ActionSpace` + `DE-Solver` + `DE-Objective` (GLFT terms are a **rate** over a continuum; per-level terms are **per fill** — no unit reconciler exists) | **SPREADING** (3) | MISS |
| 5 | v(t): sum → min-structure | 1 | `BE-Uncertainty.VarianceBudget` | **LOCAL** | MEET (caveat NF-6) |
| 6 | pull: τ_min → (\|d\|,r) surface → participation region + **size** | 1–2 | `DE-Constraints` — `feasible()` returns `max_size`, never a bool | **LOCAL** | MEET ✅ *the cleanest fix in v2* |
| 7 | Q_max: variance cap → loss-given-adverse + **portfolio aggregate** | 1–2 | `DE-Constraints` + `DA-State` (needs portfolio cardinality; is `SelfState` per-market or global? unstated) + `DE-Solver` (a cross-market cap couples `max_size` across concurrent solves) + no factor object for R1's "effective breadth ≈ 1–2, one factor" | **STRUCTURAL** — the SCOPE axis still does not exist; portfolio is a different **cardinality**, not a different impl | MISS |
| 8 | pair: net q → (q_up,q_down) joint conditions | 2 | `DE-Constraints` **cannot host it** (`feasible(action)` is one action; `b_up+b_down ≤ 1−target` constrains the action *vector*) → pushed into `DE-Solver=JointPairSolve` + `DE-ActionSpace` (a joint two-leg quote is not in the enum) + `P(complete)` has no owner in BE | **SPREADING** (3) | MISS |
| 9 | rewards: PnL line → constraint w/ shadow price → principal–agent + contest | 1–2 | `DE-Objective` (term `rewards`) + `DE-Constraints` (`moneyness_band`) **double-counted, no registry to stop it** + `SP-Venue` (has `rebate_schedule`, **no rewards-programme record**) + new `BE-Competition` for `X` in `R·x/X` | **SPREADING** (3 + 1 new) | MISS |
| 10 | siting US → EU → suspended | 0 | `SP-Params` value | **LOCAL** | MEET ✅ |
| 11 | latency 120 → 471 → 1700 ms | 0 | `SP-Params` key (+ `OP-LatencyBudget` as reader) | **LOCAL** | MEET ✅ (caveat NF-4: unkeyed) |
| 12 | scope: PnL-first → mechanism-first | 0 | `EV-Gates` | **LOCAL**, *unscoreable* — `EV-Gates` has no schema and `PM_PREREG.md` does not exist | MEET-conditional |
| 13 | components ADDED late | new module | flow → `BE-FlowAndFills` ✅ · propagator/impact → objective term `impact` ✅ · microprice → `BookOnly` ✅ · short-horizon α → `constituents` ✅ · **cross-coin (plan X7) → no home** · **portfolio risk (§14 R1/R3/R5) → a parenthesis inside `DE-Constraints`** | **STRUCTURAL** (4 of 6 land; the 2 that fail are the 2 that need a scope axis) | MISS |

**Tally: LOCAL 7 · SPREADING 4 · STRUCTURAL 2.** (v1: 5 / 3 / 5.)
Versus the charter's ideal blast radius: **7 MEET, 6 MISS.**

### Where it still spreads — three boundaries, named

**B-α — `DE-Objective` ↔ `DE-Solver` is not form-neutral.** (rows 1, 4, 9.)
`DE-Objective Σ terms … each pluggable and independently testable` commits to
**additive, risk-neutral, per-action** value. Three adopted theories are not that
shape:

- M-1: `V(ℓ) = F_ℓ·[P_ℓ − CE_quote(q) − A(Q_ahead) + ρ(P_ℓ)] + D_ℓ(Q_ahead)` —
  a *product* of fill probability and a bracket, plus a non-fill option term.
  Expressed as a Σ, every term must be pre-multiplied by `F_ℓ`, i.e. every term
  secretly depends on `BE-FlowAndFills` — which is *not* an argument to
  `DE-Solver.maximize`. So terms are not "independently testable"; they are
  jointly conditioned on a module the solver never sees.
- M-8: the terminal object is the CARA indifference `CE(q) = −(1/γ)ln(p̂e^{−γq}+1−p̂)`,
  and `PM_MECHANISM_THEORY.md` **explicitly rejects** the A-S/GLFT quadratic
  liquidation penalty. v2's term `inventory_penalty(running, terminal)` is
  precisely the rejected form. A certainty-equivalent is not a summand.
- M-5: the reward is an occupancy **integral** `R·∫(x_t/X_t)dN^sample`; fill EV
  is **per fill**. P-M5d: *"the objective needs a per-second unit to combine an
  occupancy integral with per-fill EV."* v2 has no unit on any term (my MF-8/X-2
  dropped).

**B-β — no SCOPE axis, no keyed identity** (rows 7, 13; Test 5). MF-6 is the one
iteration-1 MUST-FIX v2 silently dropped, and it is the sole cause of both
STRUCTURAL rows.

**B-γ — `DE-Constraints.feasible(action)` is per-single-action and returns a dual
it cannot compute** (rows 7, 8, 9). Detailed in §3(a)/§3(b).

---

## 2. PLUG-IN TESTS re-run against v2

### Test 1 — a new fair-value theory · **PASS (with two gaps)**
`BeliefProcess` delivers the four things v1 could not: the link `G` travels with
the estimate (kills the hard-coded-Φ bug across three consumers), a tail
functional, a path law, and `constituents`. A developer adds one `BE-Belief`
impl. Real fix.
Gaps: (a) `jump_tail: E[(|J|−m)^+]` is a **field, not a function**, but M-7's
moat is per level, `m(ℓ) = (P_ℓ − p̂) + f(P_ℓ)` — a single fixed-`m` scalar cannot
serve the level argmax it exists for. Needs `jump_tail(m)` (v1's `tail(h, thr)`).
(b) no uncertainty type on `p_hat` (see Test 6).

### Test 2 — a new mechanism · **PARTIAL**
`SP-Venue.matching`, `capabilities{}` and R-REQUIRES are the right machinery, and
"a new venue tells you which modules are invalid instead of mis-modelling it" is
now true *in principle*. Three concrete failures remain:

| mechanism | what a developer actually does |
|---|---|
| maker-priority / pro-rata | add an enum value + a `BE-FlowAndFills` impl. **PASSES.** |
| batch auction (E-M1's *"single highest-leverage binary"*) | `DE-ActionSpace`'s enum has no `bid_schedule`; `BE-FlowAndFills.rest()` returns a `queue_bracket` that is meaningless under batching, and the merge (NF-3) means you cannot null just that half. **FAILS.** |
| negRisk | `SP-Instrument.complement: Some(token)\|None` is structurally binary, so a k-outcome negRisk market cannot be represented — **while `SP-Venue.capabilities` advertises `NEG_RISK`**. A capability the instrument type cannot express. **FAILS, and it is self-contradictory.** |

R-REQUIRES also has **no declaration site**: the rule and the enforcement are
stated, but no module in §3–6 carries a `REQUIRES` field.

### Test 3 — a theory that does NOT decompose into the pipeline · **PARTIAL**
v2's best structural decision is moving `SelfState` into `DA-State`: it legalises
`A(Q_ahead)`, `P(complete)`, own-impact and P-M7b, which v1's dependency rule made
bugs. That is four adopted theories un-forbidden with one move.
What still has no home is the **equilibrium** class. M-5 is a Tullock contest:
`occupy iff R·(X−x)/X² ≥ c(|d|,r)` with Nash fixed point `R/X* → c`. Our own
effort enters `X`, so the payoff of an action depends on the action — a BE↔DE
fixed point. `EV reads all, is read by none` plus strict-downward dependency
forbids the iteration seam, and no module owns `X`. Also unhomed: M-6's boundary
reader `b(·)`, which is *"a parameter to be estimated, chosen to maximise
agreement with the official winner"* — an identification problem sitting upstream
of belief, fitted against settlement outcomes (i.e. an EV→SP edge; see NF-5).

### Test 4 — a new evaluation gate · **FAIL** (unchanged from v1)
`EV-Gates` = *"the prereg registry"*, one clause. No `Gate` record, no metric
registry, no strata reference, no inference block, no `kind='stop'`, no owner, no
precondition DAG (MAJOR-8: a gate froze before its own precondition read).
FATAL-3's STOP clause is *assigned* to `OP-Monitor/KillSwitch` — progress over
v1's ownerless state — but marked "(missing)". A developer inventing a gate
format ad hoc is exactly how MAJOR-5 (an inference convention transplanted across
gates with different units) happened.

### Test 5 — a second simultaneous venue · **FAIL**
`SP-Venue` per venue is right and `valid_from/valid_to` is right. But:
`SP-Params` is `name -> {value, provenance, owner_module, …}` — **unkeyed**. One
`L` for two venues. That is the *literal shape of FATAL-1* (three live values for
one quantity), which SP-Params exists to retire, reproduced one axis over. There
is still no `InstrumentId{venue, market, token}`, no
`RiskFactorId{underlying, horizon, settle_time}` (a Kalshi BTC-up and a PM BTC-up
are the same risk), and `DE-Constraints` is per-market. Same missing fix as rows
7 and 13. Note this is also what plan-X8 (15-min / 4-hour markets) needs.

### Test 6 — swap the queue/fill uncertainty representation · **FAIL (mild regression)**
v1 had "bracketed (pessimistic/optimistic)" as a *note*. v2 has **`queue_bracket`
as a named return field** of `BE-FlowAndFills.rest()`, and `EV-Replay` hard-codes
*"queue bracket + latency injection"*. The 2-point representation is now in two
interfaces instead of one comment. MF-4 (`Uncertain[T]` with a required
`scenarios(n)`) was dropped entirely, so moving to a posterior over `Q_ahead` or a
robust ambiguity set edits the return type, every consumer, `EV-Replay`, and the
inherited MF-6/H-PM3 gate rule *"sign-flip across the bracket = fail"*, which is
written for exactly two points.

---

## 3. FIX VERIFICATION — did they land, or are they claimed?

### (a) Does `DE-Objective` + `DE-Constraints{max_size, shadow_price}` genuinely host constrained control? — **NO. Traced.**

Trace: **the rewards band as a constraint with a price** (change #9, M-5).

| element needed | v2 home | lands? |
|---|---|---|
| band params `s_max`, `z_min ≥ 50`, `rate_per_day`, epoch | `SP-Venue` has `rebate_schedule` and a `MAKER_REWARDS` capability flag — **no rewards-programme record and no field for a band** | ✗ |
| the hard eligibility predicate (`s ≤ s_max`, `z ≥ z_min`, two-sided mandatory outside [0.10,0.90]) | `DE-Constraints.moneyness_band` | ✓ |
| the occupancy accrual `R·∫(x_t/X_t)dN^sample` over a **daily epoch** | `DE-Objective.rewards` — but the accrual integrates over an epoch while `DA-State.view(now)` is point-in-time and every other module is per-5-min-window. No owner for accrued occupancy | ✗ |
| rivals' aggregate score `X` | nowhere in BE | ✗ |
| the shadow price | `DE-Constraints.feasible() -> shadow_price` | **✗ — see below** |

M-5 defines the multiplier exactly: **`ψ = ∂V/∂x`**. That is a derivative of the
*value function*. A constraints module cannot return `∂V/∂x` without solving the
control problem — i.e. without being the solver. Either `DE-Constraints`
duplicates `DE-Solver`, or it hard-codes one solver's dual and silently breaks
when the solver is swapped. **The dual is a solver output.** The correct shape is
`Constraints.feasible(state, actions) -> FeasibleSet{max_size[], binding[]}` and
`Solver.solve(...) -> {actions, duals: {name -> price}}`; the *reporting* of the
price is what makes FATAL-2 structurally impossible, and that survives the move.

Two further failures of the same signature:

- **The pair as a joint constraint.** M-3: `b_up + b_down ≤ 1 − target`. This
  constrains a *pair of actions*. `feasible(action)` takes one. So the joint-ness
  migrated into `DE-Solver = JointPairSolve` — which means **swapping the solver
  silently drops a risk constraint**, the exact coupling the re-cut was meant to
  remove. v2 lists "the pair as a joint constraint inside the solve" among the
  four theories v1 forbade; v2 does not host it either, it relocates it.
- `feasible(action)` **has no `state` argument.** v2 makes a point of the
  converse omission (*"`size` is an argument — v1 omitted it, structurally
  forbidding own-impact"*), then drops `state` from the constraint that must
  read inventory, rate budget and phase.

### (b) Does swapping `DE-Solver` touch one module? — **NO.**

`maximize(Objective, Constraints, ActionSpace, BeliefProcess)` is a *numerical
optimiser's* signature, and `ClosedFormGLFT` is listed as an impl of it.
GLFT consumes `(σ, γ, k, A, τ)` and emits `δ*` in closed form; it cannot be handed
a `feasible()` oracle or an action enumerator, and it would have to ignore three
of its four arguments and re-derive them from `SP-Params`. Beyond that, the
Objective terms encode solver assumptions in three specific ways (B-α above):
the Σ-form, the unit (`$` per fill vs `$/s` occupancy vs rate-over-continuum), and
`inventory_penalty` being the GLFT quadratic that M-8 rejects.
Also missing from the signature: `SelfState` (the HJB state vector is
`(t, q, m, book, ℓ, Q_ahead)` — the solver cannot see `m`, `ℓ`, `Q_ahead` or the
book), the horizon, and `BE-FlowAndFills`.
**Fix:** `DE-Objective` declares a `form` (`per_action_scalar` |
`running_rate + terminal_CE` | `path_functional`) and a unit per term;
`DE-Solver` impls declare which forms they accept — R-REQUIRES applied to the
objective instead of only to the venue. Mismatch then fails at wiring instead of
being silently summed.

### (c) Is `constituents` sufficient for plan-E-X2 against the production module? — **PARTIAL.**

The central claim lands: with `constituents` on the production `BeliefProcess`,
the paired ΔBrier of `p̂_model` vs `p_book` runs against production rather than a
re-implementation, which retires the MAJOR-4 divergence pattern. Keep it.
Not sufficient as written:
1. **Weight and per-constituent provenance were dropped** from my MF-3 spec
   (`{name -> (p_component, weight, provenance)}` → `{model, book, ...}`). E-X2's
   deliverable *is* `ŵ`; you cannot read back which `w` produced a live quote, so
   the fitted blend is unauditable.
2. **No obligation on non-blend impls to populate `book`.** If production runs
   `StreamModel`, `constituents = {model}` and the experiment is off-production
   again — the failure the field exists to prevent.
3. **No alignment requirement.** A paired ΔBrier needs both constituents on the
   same information set. `Known[V]` provides the machinery; the contract does not
   require the constituents to share a `t_known`. A 30-s-stale book paired against
   a live stream is a contaminated pairing.
4. `ŵ` is a fitted parameter applied walk-forward. `SP-Params.provenance=fitted`
   exists, but there is no artifact store and no PIT guard (may not apply a fit
   inside its own fit range) — repo pitfalls #1/#2/#5 are this class. My SF-3
   was dropped.

---

## 4. NEW PROBLEMS INTRODUCED BY THE REWRITE

### NF-1 (MUST) — `L_adv` is now arithmetically wrong, and it forbids pair-harvest
v2 §5: *"`L_adv = Σ_side q_side·(1−p̂_side) ≤ κ_$` … expressed in the `(q_up,q_down)`
representation — v1's C4 still carried the retired net-`q` form."*
M-3 is exact, not approximate: `m = min(q_up,q_down)` is **riskless** (redeems for
`$m` at T) and `Var_t[terminal] = q²p̂(1−p̂)` with `m` contributing **exactly zero**.
Take `q_up = q_down = 100`: v2's sum gives `100(1−p̂) + 100p̂ = 100`, i.e. it charges
a fully-paired, guaranteed-redeemable position at full notional against the loss
cap. The true loss-given-adverse is `cost_basis − min(q_up,q_down)`.
Consequence: the risk limit **forbids the pair-harvest that §2.3 calls "the core
empirical object"** and that M-3/E-M3 are built to test. v1's net-`q` form hid the
locked edge; v2's per-side sum charges risk to riskless inventory. Both wrong, in
opposite directions. There are now four live definitions of one quantity
(plan §16.4 `|q|(1−p̂)`; `PM_DEEP_REVIEW:231` `max(q_up(1−p̂), q_down·p̂)`;
`ITER1_B:348`; v2's Σ). **The general lesson: R-SSOT as enforced governs
*values* (`SP-Params`), not *definitions*.** Every recurring corpus failure —
`v(t)` sum-vs-min, σ_⊥, `Q_max`'s exponent, three referents for `w`, and now
`L_adv` — is a definition collision, and `SP-Params` cannot see one.

### NF-2 (MUST) — `BeliefProcess` is over-specified, and R-NULL cannot reach inside it
`BookOnly` has no `jump_tail` and no `path_law`. R-NULL fires per **module**
(`SP-Strategy.nulls{module: assumption}`), so a *field* that is absent inside a
non-null module declares nothing. The `sniping` objective term then reads a
missing tail as zero — numerically identical to null-`FlowAndFills` ⇒ ζ = 0, *"the
most optimistic assumption in the programme, undeclared"*, which is the very
failure R-NULL was written for. Same hole on `BE-FlowAndFills.own_impact` (E-X6 is
unrun) and on `SP-Instrument.settlement` (below). **Fix: nullity and provenance at
FIELD granularity, not module granularity.** One mechanism, four holes.

### NF-3 (MUST) — merging FlowAndFills loses **partial nullity**
The merge is right (the split encoded the unconditional-ζ error; selection is the
dominant term — a 60–97 % haircut). But today `p_fill` is estimable from queue
simulation while `ζ`'s **sign** is an unrun experiment (plan-E-X1: reverting ⇒ ζ
gains a negative component; permanent ⇒ ζ is strictly a cost). Under v1 you could
null B6 alone and declare ζ = 0 visibly. Under v2 you either null the whole module
(losing the queue model) or ship an undeclared internal ζ = 0 inside a non-null
module. §7's `nulls = {}   # ζ must be measured, not nulled` is a *comment*, not
an enforcement point, and it is false today. Second loss: `rest()` prices only
maker actions, yet `DE-ActionSpace` advertises `cross`, `mint` and `merge` — the
taker and conversion economics (3.5 % taker, gas, redemption lag, measured
**−806 bps** post-T) have no module at all. v1's `X2 SettlementAccounting` was
deleted in the rewrite and not replaced. Fixed by NF-2's field-level fix plus a
`BE-FlowAndFills.take()`/conversion owner.

### NF-4 (SHOULD) — `DE-ActionSpace` is a flat primitive enum, and it is owned twice
`{quote(level,size) | cancel | mint | merge | cross | wait}` cannot express: a
**joint** action (the four sides are one decision — the pair constraint, B-12), a
**schedule** (batch auction), or **contingent/conditional** actions (post-only,
GTD, minimum-resting-time under P-M2b, "post the down leg iff the up leg fills").
Ownership: §5 says it is *"derived from `SP-Venue` capabilities"*; §2 makes
`action_space` a field of `SP-Strategy`. Venue-derived and strategy-selected are
different things, and per-level-vs-continuous-δ (change #4) is a *modelling*
choice that no venue capability implies. Two owners for one quantity, in the
document whose first rule is one owner per quantity.

### NF-5 (SHOULD) — the SP layer is a wiring god-object, and nothing owns fitting
`SP` is *"read by everything"*, and `SP-Strategy{objective_terms, constraint_set,
solver, action_space, module_impls, nulls}` is the **wiring of the entire system**.
So any module may legally read which solver is configured and branch on it —
the classic god-config, and it makes the "Option A/B is just a record" claim
fragile in the one way that matters. Split: `SP-Venue`/`SP-Instrument`/`SP-Params`
are facts, readable by all; the strategy record is *composition-root only* and not
readable by modules.
Second problem in the same layer: `SP` *"depends on nothing"* and `EV` *"is read
by none"*, yet `SP-Params.provenance` includes `fitted`, and the fitters are
`EV-Calibration` (isotonic link), the variogram (`ŵ`, σ̂), the blend `ŵ`, the queue
model and the ζ surface. **No module owns fitting**, so either there is an
undeclared EV→SP edge or the fit pipeline is outside the architecture — and the
PIT guard has nowhere to live.

### NF-6 (SHOULD) — the `VarianceBudget` fixes one of the three bugs it claims
§4: *"σ_⊥+κ, v(t), running-vs-terminal were one bug three times; this is its fix."*
`register()` raises `DoubleCount` **on overlapping support**. Checked one by one:
σ_⊥+κ is an overlap ✓; `v(t)` sum-vs-min is a wrong **composition operator** over
non-overlapping components ✗; running-vs-terminal is an **objective-side**
double-charge (§14 R6: *"running and terminal penalties price the same
uncertainty — still no procedure"*) that lives in `DE-Objective`, which has no
registry at all ✗. Also, support ≠ **estimand**: §16.1's 4.12× error was
`Var[X_{t+r}−X_t]` substituted for `sd_t[X_T|F_t]`, which a support check cannot
see. Fix: `register(component, support, estimand, compose_op)`, and give
`DE-Objective` the same registry with a **unit** per term.

---

## 5. INTERNAL INCONSISTENCIES / CONTRADICTIONS WITH THE CORPUS

1. **The namespace section, whose only job is naming, is wrong three ways.**
   (a) §0: *"the single-letter IDs in `PM_MM_PLAN.md` (B\*, C\*, X\*, M\*, R\*) are
   that document's own and DO NOT correspond."* False for B\*/C\*: the plan's build
   queue (*"E-M6 → E-F → E-X1 + E-X2 → B5/B6 → C4+R1 → E-X3 → B1/B3 → B4 → …"*)
   cites **v1 architecture modules**. The rewrite renamed all of them and shipped
   no v1→v2 map, so every cross-document reference in the corpus is now dangling.
   (b) §0 prescribes saying **"plan-X2"** — but in `PM_MM_PLAN` §13.2 `X2` is
   *"PM book's own information (microprice/OFI)"*, while the blend experiment is
   `E-X2` (§15). The prescribed disambiguation resolves to the wrong referent.
   (c) §4 says `plan-X2`, §8 says `plan-EX2`. `EX1`/`EX2` occur nowhere in the
   corpus. Three spellings, one experiment, one document.
2. **R-PROV's enforcement point references a concept the rewrite deleted.**
   *"`DE-Solver` refuses a **gate** whose inputs are `assumed`."* After the re-cut
   there are no gates in DE — gating became `DE-Constraints`, which is not named,
   and an `assumed` param feeding an *objective term* is not covered at all.
3. **`SP-Venue`/`SP-Instrument` carry `valid_from`/`valid_to` but no provenance
   field** — while `SP-Params` does. So the most decision-critical *assumed* facts
   escape R-PROV entirely: the settlement rule (E-M6 is the FOUNDATION GATE, i.e.
   the rule is a **hypothesis** with 12 preregistered cells; `w = 60 s` is
   *"a HYPOTHESIS, not a fact"*; `δ_tie` must be measured; `b(·)` must be fitted)
   and the rewards band (P-M5a: our markets are *absent from the rewards
   registry*; P-M5b unverified — and the corpus openly disagrees with itself,
   `PM_MM_PLAN` §8-CORRECTION vs `PM_SKETCH_REVIEW_ITER1_M`).
   v2 §2 asserts *"the venue … re-cut the rewards band 2026-08-20"* as fact; the
   corpus records it as contested.
4. **R-VERSION's enforcement kills two designed experiments and possibly all
   replay.** *"`EV-Replay` refuses to span a spec boundary"* — with no escape
   flag, dropping lens C's *"without an explicit flag"*. But E-M5(d) is the
   **band re-cut natural experiment**, which requires replaying across exactly
   that boundary; and if the `tick_size_change` 0.01→0.001 is a `SP-Venue.tick_grid`
   version, the measured **328 transitions across 130 windows** make a spec
   boundary fire ~2.5× per window and *no window is replayable*. Resolve
   explicitly: `tick_grid` must be a state-dependent function `tick(p)` (P-M2c:
   the switch is a function of price), not a versioned constant, and R-VERSION
   needs a cross-boundary mode that stamps and stratifies rather than refuses.
5. **Rewards is both an objective term and a constraint, with no registry to
   catch it** (§5 term list vs §7 `constraint_set`). Under M-5 the constraint's
   multiplier *is* the economics; charging both double-counts the subsidy. This
   is R-ONCE in the objective plane, which has no `register()`.
6. **Option B as specified cannot be wired today.** §7: `nulls = {}` — but
   `BE-FlowAndFills` is null *now* (E-X1 unrun, ζ's sign unknown), and
   `BE-Uncertainty = coarse` names an impl that appears in no impl list. §7 also
   sets `BE-Belief = StreamModel|Blend` — a record field cannot be a disjunction.
7. **Dependency direction contradicts its own diagram.** *"Dependencies point
   downward (SP ← DA ← BE ← DE)"* — the diagram draws SP at the top and DE at the
   bottom, and SP *"depends on nothing"*, so dependencies point **up** the drawing.
   Trivial to fix; it is the sentence a new developer reads first.
8. **No universe owner.** The corpus discipline is *"universe frozen at 7 coins …
   no coin enters mid-sample (the expanding-survivor artifact killed the OB-timing
   signal in this repo)"* and *"strata frozen before data is read and model-free
   (`|book_mid − 0.5|` — never model `|d|`, which lets the model grade its own
   exam)"*. `DA-Discovery` is the natural owner of a frozen, versioned PIT
   universe; nothing enforces either rule, and `DE-Constraints.moneyness_band`
   does not declare whether its band is model-free or `|d|`.
9. **§14 R3 (capital allocation across 288×7 concurrent windows — *"capital
   velocity, not risk, may be the binding constraint"*) and R6 (γ/γ_T joint
   calibration) have no home in v2.** R5 (28 quote streams) is assigned to
   `DE-Actuator`'s "rate budget", which is per-market — B-β again.
10. **`OP-LatencyBudget` vs `SP-Params`.** R-SSOT: `SP-Params` is *"the only legal
    source"*, readers *"cannot restate"*. Build order step 1 lists
    `OP-LatencyBudget` as a co-equal module owning four legs. It must be a *view*
    over four keyed params, or FATAL-1 returns as a second owner.
11. **`Known.t_known` has no provenance of its own.** `Known[V]` carries one
    `provenance` for the *value*. If `t_known` is computed as
    `t_event + assumed_latency` rather than stamped at arrival, the 1.7 s peek
    returns inside a type that certifies it away. `t_known` must be
    arrival-stamped by `DA-Feeds` and carry `measured|assumed` itself. Related:
    nothing forbids a live caller passing a future `now` to `view(now)`;
    `EV-Replay` should be the only component permitted to supply a synthetic one.

---

## 6. TRIAGE

### MUST-FIX (6)

| id | fix | why |
|---|---|---|
| **A2-1** | **`L_adv` per-side sum is wrong.** Risk is on `q = q_up − q_down`; `m = min(q_up,q_down)` is riskless. Write `L_adv = cost_basis − min(q_up,q_down) ≤ κ_$` (or the `(m,q)` decomposition) and reconcile the four live definitions in one place. | NF-1. As written the risk limit forbids pair-harvest. |
| **A2-2** | **Objective/Solver form contract.** `DE-Objective` declares `form ∈ {per_action_scalar, running_rate+terminal_CE, path_functional}` and a **unit** per term; each `DE-Solver` impl declares accepted forms; mismatch fails at wiring. Register terms by name+unit+sign+R-ONCE key (my MF-8/X-2). | B-α; rows 1/4/9; §5.5's rewards double-count; M-5's `$/s`-vs-`$` mismatch; M-8's rejected quadratic penalty. |
| **A2-3** | **Duals belong to the solver; constraints take state and action *vectors*.** `feasible(state, actions) -> FeasibleSet{max_size[], binding[]}`; `Solver.solve(...) -> {actions, duals}`. FATAL-2 is fixed by the *reporting* of the price, which survives the move. | B-γ; §3(a)/(b); the pair constraint stops hiding in `JointPairSolve`. |
| **A2-4** | **Nullity + provenance at FIELD granularity.** Every field of `BeliefProcess`, `RestingOutcome`, `SP-Instrument.settlement` and `SP-Venue` carries `measured\|fitted\|assumed\|absent(assumption)`. R-NULL fires per field, not per module. | NF-2, NF-3, §5.3. Retires the "silent zero" class (ζ=0, jump_tail=0, own_impact=0) and brings the unverified settlement rule and rewards band under R-PROV. |
| **A2-5** | **Restore MF-6: SCOPE axis + keyed identity.** Every module declares `SCOPE ∈ {window, market, coin, portfolio, venue, programme}`; `InstrumentId{venue, market, token}`; `RiskFactorId{underlying, horizon, settle_time}`; `SP-Params.get(name, key={venue, coin, horizon, date})`. | The only dropped iteration-1 MUST-FIX; sole cause of both STRUCTURAL rows and of Test 5; also what X8 (15-min/4-h) and §14 R1/R3/R5 need. |
| **A2-6** | **Publish a v1→v2 module ID map, and fix the namespace section.** `plan-E-X1`/`plan-E-X2` (matching the corpus IDs), correct the B\*/C\* claim, and map B1..B6/C1..C6/D1..D3/K1..K2/E1..E4 → SP/DA/BE/DE/EV/OP. | §5.1. Without it the plan's build queue, this loop's change log and every review cross-reference are dangling — a cost created by the rewrite. |

### SHOULD-FIX (8)

- **A2-S1** Restore the per-module `in`/`out` register. Eleven modules have no
  signature; `DE-Objective` — the point of the re-cut — is seven words. A
  structure review cannot score what has no interface.
- **A2-S2** Restore `Uncertain[T]` with a required `scenarios(n)`; delete
  `queue_bracket` as an interface name (Test 6; the bracket rule generalises to
  "verdict stable across scenarios").
- **A2-S3** `jump_tail(m)` as a function, not a field (M-7's moat is per level);
  type `path_law`.
- **A2-S4** `constituents: {name -> (value, weight, provenance, t_known)}`, and
  require every impl to populate `book` where a book exists (§3(c)).
- **A2-S5** `DE-ActionSpace`: joint/schedule/contingent actions; settle single
  ownership (venue-derived vs strategy-selected). Add a taker/conversion economics
  owner (`cross`, `mint`, `merge`, redemption lag, post-T phase).
- **A2-S6** Split `SP-Strategy` (wiring, composition-root only) from SP facts;
  name an owner for **fitting** + an artifact store with a PIT guard.
- **A2-S7** `EV-Gates` schema: `Gate{metric, statistic, strata_ref, inference,
  threshold, direction, kind∈{go,no_go,stop}, owner, frozen_at, reads_after,
  preconditions[]}` + a metric registry. Houses FATAL-3 and MAJOR-5/MAJOR-8.
- **A2-S8** `DA-Discovery` owns a frozen, versioned PIT **universe**; R-VERSION
  gains an explicit cross-boundary mode (E-M5(d)); `tick_grid` is `tick(p)`.

### NOTED (5)

- **N2-1** `BE-Competition` (rivals' `X`, latency differential, `R/X → c`) will be
  needed for M-5's endpoint; under the "new module, no edits" rule it is cheap —
  *provided* A2-2 lands, since the contest makes `rewards` an equilibrium object
  rather than a scalar.
- **N2-2** Objective term list is missing `D_ℓ(Q_ahead)` (M-1's join-vs-improve
  term, *"the term §3 is missing"*), `CE_quote`, and merge option value. Cheap by
  design — recorded so it is not forgotten.
- **N2-3** M-7 demotes the burst flag to a **model-validity monitor**; that is an
  `OP-Monitor` job and appears nowhere in v2 (carry-over N-3).
- **N2-4** §8 build order stops before the decision plane: `DE-Objective`,
  `DE-Solver`, `DE-ActionSpace`, `DE-Actuator` and `EV-Gates` are never built.
  Defensible as a near-term order; say so.
- **N2-5** Lens C's M-C4 (risk-factor exposure replacing `(q_up,q_down)`) was
  rejected in favour of `(q_up,q_down)`. Reasonable — but then cross-coin R1's
  *"effective breadth ≈ 1–2, one factor"* has no representation, which is half of
  row 13. A2-5's `RiskFactorId` is the minimum version.

---

## 7. What v2 genuinely fixed — do not churn these

Recorded so iteration 3 does not re-litigate settled ground.

1. **`Known[V]` + `view(now)` with no event-time API.** The correct fix to the
   worst defect in v1, and made *unwriteable* rather than *forbidden*. Add only
   `t_known`'s own provenance (§5.11).
2. **`SelfState` in `DA-State`.** One move un-forbids `A(Q_ahead)`, `P(complete)`,
   `D_ℓ`, own-impact and P-M7b — four adopted theories that v1's dependency rule
   made bugs. Best structural decision in the rewrite.
3. **`SP-Venue`/`SP-Instrument` as versioned data, not an abstraction layer**, with
   §9's explicit stop-signs. Retires the five-copy settlement rule; the
   `w_declared` ≠ `w_hat` split is exactly right (the corpus has three referents
   for `w`).
4. **`feasible()` returns `max_size`, never a bool.** Change #6 goes from
   STRUCTURAL to LOCAL on this alone, and it is the honest structural answer to
   FATAL-2.
5. **`BeliefProcess` with a travelling `link: G`.** Kills the hard-coded-Φ bug in
   three consumers and makes plan-E-X2 expressible against production.
6. **Merging fill-probability and adverse selection.** The split encoded the
   unconditional-ζ error where selection is the dominant term (60–97 % haircut);
   `size` as an argument un-forbids own-impact. Keep the merge — fix only its
   nullity granularity (NF-3).
7. **Every rule has a named enforcement point.** The idea is right even where a
   specific point is wrong (§5.2, §5.4); a rule with no enforcement point gets
   violated, and v1 proved it.
