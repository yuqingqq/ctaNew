# PM_STRUCT_ITER1_B — lens B (CONTRACTS & BOUNDARIES), iteration 1

Object under review: `PM_ARCHITECTURE.md` (2026-08-20). Supporting: `PM_MM_PLAN.md`
§2/§3 + §12–17, `PM_QUANT_REVIEW.md`, `PM_DEEP_REVIEW.md`.
Question this lens owns: **are the boundaries in the right places, are the
interfaces complete / minimal / non-leaky, and does the dependency rule bind?**

Verdict: the plane decomposition is sound and the module *set* is close to right.
The **contracts are not contracts** — they are names with a type sketch. Five of
them cannot express a fact the corpus has already measured, which means the
measurement has nowhere to land. **7 MUST-FIX, 8 SHOULD-FIX, 5 NOTED.**

Headline: **`as_of(t)` is the worst boundary error in the document.** It conflates
event time with knowledge time, and the measured gap is **1,700 ms p50** on the
settlement stream. `as_of(t)` as written is a look-ahead API, and it is the API on
which "no future data reachable by construction" is claimed.

---

## 1. Boundary placement

### 1(a) B1 SettlementTarget vs B2 Uncertainty — the variogram estimates σ² AND w

**Confirmed boundary violation, and the diagnosis in the prompt is one level too
shallow.** The problem is not "the joint estimator has no home". It is that
**`w` names two different quantities** and the register has one slot for them:

| quantity | what it is | epistemic status | rightful owner |
|---|---|---|---|
| `w_declared` | the venue's averaging convention (60 s per `resolutionSource`) | `inherited` from market metadata; verifiable on-chain | **B1** — it is part of *what settles* |
| `w_hat` | the MA-order recovered from the observable X stream by the variogram (56.31 ± 5.17 s on 13.5 h) | `fitted`, with a CI | **B2** — it is a nuisance parameter of an estimator |

R-SSOT says one owner per quantity. The live failure mode here is the inverse:
**one name, two quantities.** §16.1 even markets the collision as a feature ("it
simultaneously settles the §10/E-M6 ambiguity over whether the averaging window is
60 s or 300 s") — which silently promotes a *fitted nuisance parameter* into a
*settlement convention*. That is B2 deciding what the contract pays.

There is also a live estimator-level consequence the interface must state.
`(σ̂², ŵ)` from the joint fit are **correlated**; if the model then uses
`w = w_declared = 60`, `σ̂²` from the joint fit is the wrong number — it must be
re-profiled at fixed `w`. Two different σ̂² from the same data, and nothing in the
current interface distinguishes them.

**Corrected interfaces.**

```python
# ── B0 SettlementLaw ── pure, stateless, no owned parameters: a library, not a module.
#    Everything analytic about the TWAP settlement lives here, once.
def var_settle(r: Sec, tau: Sec, sigma2: Bps2PerSec, w: Sec) -> Bps2: ...
    # in-window   sigma2 * r**3 / (3*w**2)
    # pre-window  sigma2 * (tau - 2*w/3)
    # rolling-increment variogram V(r; sigma2, w)  <- ESTIMATION target only, never a belief

class B1_SettlementTarget(Protocol):
    def w_declared(self, ctx: WindowCtx) -> Param[Sec]:        # provenance=inherited|measured(on-chain)
    def strike(self, ctx: WindowCtx) -> Known[Price] | Unavailable
        # Known carries t_known: K depends on prices over [-60s, 0] and is NOT
        # knowable at t=0 under a 1.7 s relay lag. E-M6b becomes a typed field,
        # not an open question.
    def expect_XT(self, view: StateView) -> Known[Price] | Unavailable

class B2_Uncertainty(Protocol):
    # BELIEF path: sigma2 profiled at the DECLARED w. This is the only number p_hat may use.
    def sigma2_given_w(self, view: StateView, w: Param[Sec]) -> Param[Bps2PerSec]
    # DIAGNOSTIC path: the joint fit. Feeds K3, never B3.
    def variogram_joint(self, view: StateView) -> JointFit   # (sigma2_hat, w_hat, cov, gof)
    def convention_check(self) -> CheckResult   # |w_hat - w_declared| vs CI -> K3 alarm
```

**Rule that falls out:** a fitted quantity may never be substituted for a declared
convention. It may only *contradict* it, and a contradiction is a monitor event.

**Second, larger problem at this boundary: "σ_eff, and only σ_eff" does not make
R-ONCE true — it makes it unfalsifiable.** `σ_eff` is not one quantity. The corpus
has already found four disjoint variance sources and double-counted three of them:

| # | source | magnitude | who *actually* creates it |
|---|---|---|---|
| V1 | settlement diffusion of X_T | `σ²r³/3w²`; 0.007 bps at r=2 s | B0 law + B2 σ̂² |
| V2 | **epistemic deficit** — the locked integral is permanently `Δ_relay` short | ≈0.08 bps of X at 1.7 s (MAJOR-10) | **K2 / D3** (a latency fact) |
| V3 | basis innovation when B3 gap-fills from Binance | OU, `σ_⊥²·2(1−ρ(r))`: 0.08 bps @2 s, 0.44 bps @30 s | **B3** (it exists only because B3 chose to gap-fill) |
| V4 | edge-measurement noise ω (predictive vs realised σ; 1.37× at ω=0.5σ) | §16.5 | **E2** (calibration), not the belief path |

Three documented double-counts (σ_⊥ ×2, v(t), running-vs-terminal) are all the same
structural fact: **variance is composed by addition-at-will because the type
permits it.** A module that "owns σ_eff" while V2/V3/V4 are created elsewhere
cannot enforce anything.

```python
@dataclass(frozen=True)
class VarianceComponent:
    name: str                 # "settle" | "relay_deficit" | "basis_gapfill" | ...
    value: Bps2               # in ONE declared unit
    support: frozenset[str]   # the risk sources it covers, e.g. {"S_diffusion"}
    owner: str                # module id
    prov: Provenance

class VarianceBudget:
    """The ONLY legal way to add variances. Rejects overlapping support."""
    def register(self, c: VarianceComponent) -> None:
        if any(c.support & d.support for d in self._c):
            raise DoubleCount(c, conflicting)          # <- σ_⊥ + κ(r) fails HERE
    def total(self) -> Bps2: ...
    def explain(self) -> list[tuple[str, float, float]]   # name, bps², share of total
```

Re-scope B2 from *"σ_eff and only σ_eff"* to **"the VarianceBudget registry, plus
the V1 component"**. Other modules register their own components; only B2 may call
`total()`. `explain()` is what makes the endgame legible: at r = 2 s the budget
should say *92 % of my variance is `relay_deficit`, owner K2, provenance measured* —
a sentence no current interface can produce.

### 1(b) C1 Gating vs C2 QuotePolicy — the split is right, the CUT is wrong

Feasibility-vs-optimality is a good separation and I would keep it. But the current
boundary is drawn so that **C1 is allowed to reason in cents**: the participation
frontier `m(ℓ)/φ(d) ≥ k√(3L/r)` compares a moat (1 tick + fee credit, cents) against
a pickoff exposure (cents). Once a module that cannot see revenue is permitted to
compute a revenue-shaped quantity, it *will* become a policy. FATAL-2 is not an
oversight; it is what this cut invites. Evidence that the invitation was accepted:
the corrected frontier selects **5–10 % of window-time** and nobody asked what is
earned there, and the answer turned out to be ~+0.7 c/share against a ~97 c tail.

**Corrected cut — by kind of constraint, not by "where vs what":**

- **C1 may only encode constraints whose violation is not tradable at any price**:
  eligibility/jurisdiction, grid feasibility (`[tick(state), 1−tick(state)]`),
  lifecycle phase (no quotes in `[T−ε, resolution]`), stream/feed availability,
  rate & quota budget, and **exogenous risk budgets** (`L_adv ≤ κ_$`, `Σ_c ≤ L_max`)
  — budgets, note, not edge estimates.
- **Everything with an estimated price on it moves into C2's objective**: `ζ_snipe`
  becomes a cost term owned by B5/B6, not a gate. The participation frontier
  *disappears as a module-level concept* and survives only as the region where the
  EV happens to be negative.

```python
class C1_Feasibility(Protocol):
    def feasible(self, l: Level, side: Side, view: StateView) -> Feasible
    # Feasible = ok: bool + reason: InfeasibleReason (CLOSED ENUM)
    # InfeasibleReason ∈ {NOT_ELIGIBLE, OFF_GRID, PHASE_CLOSED, FEED_UNAVAILABLE,
    #                     RATE_BUDGET, LOSS_BUDGET, HALTED}
    # NOTE: C1 must not import B5/B6/B3-confidence. Lintable.

class C2_QuotePolicy(Protocol):
    def intents(self, view: StateView, belief: Belief, res: Reservation,
                feas: FeasibilityMap) -> QuoteSet     # all 4 sides, JOINTLY (see 2(f))
    # objective per level, in one unit (cents/share), all terms named:
    #   EV(l) = p_fill(l, size) * ( edge(l) - CE_shift(q, size) - E[markout | fill] )
```

`InfeasibleReason` being a closed non-economic enum is the enforcement point: you
cannot express "infeasible because the moat is thin" without adding a member, and
adding one is a reviewable act. This also makes §3's Option A/B claim *true* — see
§3(b).

### 1(c) B5 FillModel vs B6 AdverseSelection — merge them; they are two marginals of one object

They should be one module, and not merely for tidiness. The current split
**encodes a statistical error in the interface**. The architecture's own C2 formula
is

```
λ_fill(ℓ) · [ P_ℓ − CE_quote(q) − ζ(ℓ) ]
```

`ζ` here must be `E[markout | filled at ℓ]` — conditional on the fill. Two separate
modules producing `λ` and `ζ` from separate estimations will produce an
*unconditional* `ζ` (a markout surface averaged over all time) and multiply it by a
marginal fill rate. Fills are selected; that product is not the expectation of the
PnL. The deep review measured how large the selection effect is: unconditional mid
mispricing +1.8 c/share vs realised markout on fills +0.72 c/share at `t=290 s`
(a 60 % haircut), and −9.4 c vs +0.25 c at `t=30 s` (a 97 % haircut). **Selection is
the dominant term, and the interface splits it in half.**

```python
class B5_RestingOutcome(Protocol):     # absorbs B6; B4 FlowModel remains its engine
    def outcome(self, l: Level, side: Side, size: Shares, view: StateView,
                horizon: Sec) -> Outcome | Unavailable
# Outcome:
#   p_fill:        Prob            # over `horizon`, bracketed via Q_ahead uncertainty
#   markout:       Cents           # E[ mid_{t+h} - P | FILLED ]  <- conditional, by construction
#   markout_q:     dict[float, Cents]     # quantiles; the tail is the risk
#   decomposition: dict[str, Cents]       # {"spread_capture", "as_transient", "as_permanent",
#                                          #  "snipe"} — a PARTITION, must sum to markout
#   bracket:       tuple[Outcome, Outcome]  # pessimistic/optimistic on Q_ahead ONLY
#   prov:          Provenance
```

Three things this fixes: (i) `size` is now an **argument**, so own-impact /
Kyle-λ is expressible — the current `lambda_fill(ℓ, Q_ahead)` signature structurally
*forbids* the model §13.1(3) says is required (our min size 5 vs median trade 9
shares); (ii) `ζ_snipe` survives as a named element of a declared partition rather
than a second module free to be added on top (the R-ONCE failure mode again);
(iii) the pessimistic/optimistic bracket sits on `Q_ahead`, as the architecture
itself notes it should, instead of straddling a module boundary.

### 1(d) D3 StateBuilder — `as_of(t)` is NOT sufficient, and it is the worst error in the document

**`as_of(t)` does not name which `t`.** There are two, and they differ by a measured
**1,700 ms p50 / 2,330 ms p95** on the settlement stream (1,440 ms PM-side
publication + 257 ms transport). A TWAP tick stamped `t` is *knowable* at `t+1.7 s`.
Any implementation that filters by payload timestamp — the natural reading of
`as_of(t)`, and the only one that makes the returned state look tidy — is a
**1.7-second look-ahead**, and it is being sold as the guarantee that "no future
data [is] reachable by construction".

The magnitude is decisive rather than academic, because the endgame is exactly this
long: settlement vol at r = 2 s is 0.007 bps, and MAJOR-10 prices a 1.7 s head
start on the stream at ≈0.08 bps of X. **Inside the final ~2 s the peek is worth
more than the risk being modelled.** A backtest built on `as_of(t_event)` would show
the endgame as free money, and the replay harness (E3) would reproduce it faithfully.

`as_of(t)` also cannot express five facts the corpus has already established:
K's knowability (E-M6b), the permanently-short locked integral (MAJOR-10), the
state-dependent tick (328 transitions / 130 windows), per-feed staleness (the C1
pull trigger currently re-derives it), and the pre/post-window phases (8.3 % of
notional, §MAJOR-6).

**Corrected contract — make the illegal thing unrepresentable.**

```python
KnowledgeTime = NewType("KnowledgeTime", int)   # ns, OUR clock, = recv_ns + decode

@dataclass(frozen=True)
class Known(Generic[V]):
    value:   V
    t_event: EventTime        # venue/payload stamp — DATA, never a filter key
    t_known: KnowledgeTime    # when WE could have acted on it
    source:  SourceId
    seq:     int
    prov:    Provenance

class D3_StateBuilder(Protocol):
    def view(self, now: KnowledgeTime) -> StateView
    # INVARIANT (assertable, and asserted in tests): for every Known f returned,
    #     f.t_known <= now.
    # There is NO API that filters by t_event. None. That is the enforcement.

class StateView(Protocol):
    market:     Known[Book]           # incl. tick(state) — NOT a WindowCtx constant
    settlement: SettlementView
    ctx:        WindowCtx
    def staleness(self, field: str) -> Sec           # now - t_event of last update
    def knowledge_lag(self, src: SourceId) -> Dist   # monitored; K2 owns the estimate

class SettlementView(Protocol):
    locked:        Known[Price]       # integral over [T-w, t_event_last] ONLY
    covered_span:  tuple[Sec, Sec]    # what the integral actually covers
    deficit_span:  Sec                # = now - t_event_last  (the 1.7 s, exposed)
    # B2 registers VarianceComponent("relay_deficit", ...) FROM this field.
    # Today, every consumer silently assumes deficit_span == 0.

class WindowCtx(Protocol):
    t0: EventTime; T: EventTime
    phase: Literal["pre", "in", "post_T", "resolved"]   # MAJOR-6 lives here
    coin: str; rewards: RewardsParams
    # `tick` REMOVED — it is time-varying MarketState with its own t_known.
    # `w` REMOVED — it is B1's declared convention (1(a)).
```

`WindowCtx.phase` is not cosmetic: `t < 0` carries **7.4 % of notional** with no
formula (T-F6a) and `t > T` carries **0.9 % at −1.46 c/share (−806 bps)**, a pure
stale-quote harvest in a phase §2's state space does not know exists. With a phase
field, "cancel at `T−ε`, do not re-quote before resolution" is one line in C1's
closed enum. Without it, it is a rule with nowhere to live.

---

## 2. Interface completeness — what a real implementation needs and the contract does not give

### 2(a) Error / absence semantics: **no interface in §2 has a failure mode**

Every belief module returns a value. None returns "I cannot answer". The
consequence is visible in the current design: **B3's absence semantics are
implemented in C1** ("stream staleness → widen/pull"). A downstream module is
re-deriving an upstream module's health from raw feed state — that is R-EXPLICIT
violated by omission, and it is why "stream staleness" needs a hand-tuned `N`
that nobody owns.

```python
@dataclass(frozen=True)
class Unavailable:
    reason: Literal["STALE","GAP","WARMUP","OUT_OF_SUPPORT","UPSTREAM_UNAVAILABLE","NOT_IDENTIFIED"]
    since:  KnowledgeTime
    detail: str
# Every Belief-plane method returns `T | Unavailable`. Absence PROPAGATES with its
# cause: B3 returns UPSTREAM_UNAVAILABLE(B2:WARMUP) rather than a number.
# C1's FEED_UNAVAILABLE reason is then set BY the belief plane, not re-derived.
```

`OUT_OF_SUPPORT` is load-bearing and specific to this venue: `p̂ ≈ 2.5e-13`
(back-solved from the `Q_max = 9.9e15` demo) is not a probability any free-data
model can assert. The honest return is `Unavailable(OUT_OF_SUPPORT)`, not a
clamped 0.005 that a downstream cap will then multiply by 25×.

### 2(b) Units: mixed in the corpus, absent from the contract, and they have already caused two errors

`p̂` (probability) · `ζ` (cents/share) · `m(ℓ)` (cents) · `σ_eff` (bps) · `κ_$`
(USD) · `γ` (1/USD) · `q` (shares) · `L` (ms) · `v` (price²). The frontier
`m(ℓ)/φ(d) ≥ k√(3L/r)` mixes cents, bps and ms in one inequality. Two documented
errors are pure unit failures: §16.3 (taker fee multiplied by price — an ATM `r*` of
123 s that is really 46 s) and §16.4/§4(b) (`γ|q|v` has units of **price**, so
"a constant risk budget" is dimensionally a *skew cap*; QUANT states it plainly —
*"the label and the formula disagree"*).

Minimal fix, cheap and it would have caught both: `NewType` aliases
(`Prob`, `Cents`, `Bps`, `Bps2`, `USD`, `Shares`, `Sec`, `Price`) on every field and
signature in the register, plus a dimension test on each formula. Not a type system —
a naming discipline that makes a reviewer able to see the mismatch.

### 2(c) Provenance is on 2 of 22 modules, does not compose, and does not cover conventions

Three separate gaps:

1. **Coverage.** `provenance` appears on B1 and B3 only. It must be on every
   numeric output: σ̂ (B2), λ/markout (B5), each latency leg (K2), `κ_$`/`L_max` (C4).
2. **Composition.** *"`assumed` may not gate a decision"* is unenforceable without a
   combination rule. FATAL-1 is exactly this: the siting decision consumed a `p̂`
   that would have been labelled `fitted` while depending on an **assumed 75 ms**
   order-ack leg — the one leg never observed. Define the lattice
   `measured > fitted > inherited > assumed` and **`combine = min`**, computed
   automatically by every value type. Then a decision gate is
   `@requires_provenance(min=FITTED)` and the assumed input makes the call *raise*.
3. **Conventions, not just parameters.** The single largest sign risk in the program
   is not a number: it is whether `last_trade_price.side` is the **taker's** side.
   If it is the maker's, maker gross flips **+95 bps → −95 bps** and the program is
   dead. Its current status is *circumstantial* (63.7 % of BUY prints at the ask).
   D2 "owns the side×asset_id → direction mapping" — but there is nowhere in the
   contract to record that this mapping is `assumed`. **`Provenance` must attach to
   derived-field semantics** (`Trade.side_signed`, dedup keys, the `b(·)` boundary
   reader whose observations-vs-validFrom dimension is *not identifiable from our
   data at all*, MAJOR-7), not only to scalars.

### 2(d) C4 RiskLimits — `max_size(ℓ)` is the wrong shape three times over

The cap is (i) **sign-asymmetric** (`Q^long = κ_$/p̂` vs `Q^short = κ_$/(1−p̂)`;
at p̂=0.01 that is 250,000 vs 2,525 shares — a scalar `max_size` cannot express a
2-order-of-magnitude asymmetry between the two sides of *our own two-sided quote*);
(ii) **per-token, not per-level** — the binding object is inventory, and levels only
matter through the size they might add; (iii) **portfolio-coupled** —
`Σ_c |q_c|(1−p̂_c) ≤ L_max` cannot be evaluated by a per-market module at all.

```python
class C4_RiskLimits(Protocol):
    def headroom(self, token: TokenId, direction: Direction,
                 view: StateView, port: PortfolioState) -> Shares
    def l_adv(self, q_up: Shares, q_down: Shares, p_hat: Prob) -> USD
    def breaches(self, port: PortfolioState) -> list[Breach]
```

### 2(e) The multi-market scope has no owner at all — the whole register is implicitly one window

Three shared resources are contended across 7 coins × overlapping 5-min cycles, and
none has a module:

| resource | contended by | plan ref | owner in PM_ARCHITECTURE |
|---|---|---|---|
| aggregate loss budget `L_max` | all live windows | §14 R1 | **none** (C4 is per-market) |
| order/cancel rate budget | 28 quote streams (7×2×2) | §14 R5 | **none** (C6 is per-market) |
| capital tied to resolution | 288 windows/day/coin × 7 | §14 R3 | **none** |

A shared budget with no allocator is a race: whichever window's decision tick fires
first consumes it. Effective breadth across the 7 coins is **≈1–2, not 7** (one
crypto factor), so the aggregate binds *often*, not rarely.

```python
class C7_PortfolioAllocator(Protocol):        # NEW MODULE. Decision plane, above C1..C6.
    def allocate(self, requests: list[BudgetRequest], port: PortfolioState) -> Allocation
    # requests carry marginal EV per unit of budget; allocation is by shadow price,
    # not first-come. Deterministic tie-break (required by E3 replay determinism).
```

### 2(f) C2/C5 — the four sides are one decision, and the interface says they are four

`out: QuoteIntent{token, side, px, size}` (singular) invites per-side independent
evaluation. QUANT §4(c) shows why that is wrong: the second leg's reservation must
be evaluated at the **post-first-fill** `q`, and quoting both legs off the same
pre-fill `q` double-counts the inventory credit. Also `EV_pair = P(complete)·(1−a−b)
+ (1−P(complete))·EV_naked(first leg)` is a property of the *set*, and `1−a−b = 2δ`
identically — so the pair trigger is vacuous when evaluated leg-wise.
Emit `QuoteSet` (all four sides, one object, evaluated jointly).

C5's contract must additionally declare, because these are conventions with PnL
consequences and none is currently stated: `q_up, q_down ≥ 0` (**no naked short** on
this venue — shorts are longs of the complement; flag as P-M3d, unverified);
**average-cost** basis (FIFO gives a different realised/carried split);
**merge books no PnL** (it is a capital decision — $1 either way); partial-fill
residual is adversely selected by construction.

### 2(g) Other gaps, briefly

- **B3 has no link owner.** `p̂ = link(d)`. Φ under-prices the tail up to **28×**
  and misspecification biases the MLE **−3.5…−15 %**. The plan calls this "B4 (link)";
  the architecture's B4 is FlowModel. Nobody owns Φ. Add `B7 Link` (swappable:
  `Probit | StudentT(ν) | Isotonic`) with its own provenance.
- **E4 vs K3.** The FATAL-3 STOP clause is a *program* decision; K3's kill switch is
  a *trading* decision. Neither module is stated to own the other. Assign: E4 owns
  `stop_triggers()` (pre-registered, named owner); K3 owns `halt_triggers()` and
  evaluates both, because only K3 runs continuously.
- **X2 SettlementAccounting** has no post-`T`-pre-resolution contract (median trade
  lands at T+21 s, resolution at T+85 s).
- **D2** must expose the gap registry as a *queryable* object (`gaps(span)`), because
  B2 must skip any interval containing one of the 56 recorded >5 s gaps. Today that
  knowledge lives in an analysis script.

---

## 3. Leaky abstractions

### 3(a) `B3.confidence` — leaks, and is the reason the A/B swap is not clean

`confidence` has no declared type, unit or semantics. For `StreamModel` it would be
σ-derived; for `BookPlusFLB` a fit uncertainty; for `Blend(w)` something else again.
**C2 must know which implementation produced it to interpret it** — the definition
of a leak, in the exact place §3 promises a clean swap.

Fix: delete `confidence`. Return a quantity every implementation can produce and
that C2 consumes in one documented way:

```python
class Belief(Protocol):
    p_hat: Prob
    var_p: Prob2        # variance of the ESTIMATOR of p_hat (probability²), NOT of the outcome
    prov:  Provenance   # composed = min over inputs
# C2's single documented use: shrink edge toward the book by var_p /(var_p + var_book).
# An implementation that cannot produce var_p returns Unavailable(NOT_IDENTIFIED).
```

### 3(b) The Option A/B swap is a 3-module change, not a 1-module config choice

§3 claims *"Option B is this system with simpler implementations of three
modules"*, then its own table changes **five**: B3, B2, **C1**, **C2**, B4/B5/B6.
C1 and C2 changing is the tell — selecting a *belief* implementation should never
edit the *decision* plane. Root cause is 1(b): the participation frontier lives in
C1, and it is a StreamModel-specific object (it needs `φ(d)`, which needs `p̂` from
the stream model, which Option B does not compute).

With 1(b) applied, the table becomes: B3 = `BookPlusFLB`, B5 = measured-markout impl,
B4 = minimal, B2 = `StaticVol`, **C1 and C2 literally unchanged** — because C2's
objective reads `p_fill`, `edge`, `markout` regardless of who produced them, and
C1's enum is non-economic. Only then is §3's claim true.

### 3(c) Knowledge-time leak

C1's "stream staleness" rule exists because consumers are expected to *know* the
TWAP is 1.7 s late and to hand-tune `N` accordingly. With `StateView.staleness()` and
typed `Unavailable(STALE)`, `N` becomes a parameter of B3 (the module that knows what
staleness costs it) and C1 just reads `FEED_UNAVAILABLE`.

### 3(d) K2 latency leak

The frontier consumes a scalar `L`. There are **four** legs, and the corpus carries
**three mutually inconsistent values for one of them** (120 / 471 / 1700 ms), with
the 471 ms measured on the *spot mirror* and applied to the TWAP stream (3.6× slower).
A scalar `L` makes that error invisible.

```python
class K2_LatencyBudget(Protocol):
    def leg(self, name: LegId) -> Param[Ms]     # LegId ∈ {BINANCE_TO_US, RELAY_PUBLISH,
                                                #          RELAY_TO_US, US_TO_ACK}
    def path(self, p: PathId) -> Param[Ms]      # named composite; provenance = min over legs
# Consumers must name a PATH. `L` as a bare float does not exist in the codebase.
# US_TO_ACK is currently `assumed` -> any gate consuming it fails @requires_provenance.
```

---

## 4. Do the four rules bind? (No. Here is the enforcement point for each.)

All four rules are **aspirational as written** — each is a sentence with no
mechanism, and the program has violated all four. Proposed enforcement:

| rule | why it does not bind today | enforcement point |
|---|---|---|
| **R-SSOT** | nothing stops a module computing its own σ/latency; and the *identifier* namespace is itself not SSOT (see below) | **K1 as the only legal constructor**: `Param` objects can only be created by `ParamStore`; a lint bans numeric literals in module code outside the registry; docs cite `{{param:latency.relay_publish.p50}}` rather than typing "1,440 ms". Plus a machine-readable `modules.yaml` with an ID-uniqueness test. |
| **R-EXPLICIT** | modules can reach any import; FATAL-1's hidden Binance dependency was legal | **declared-input manifest + injection**: each module declares `INPUTS: list[FieldId]`; the harness passes a view exposing *only* those and raises on anything else. Plus an import-graph test (C1 ⊬ B5/B6; Belief ⊬ D1). |
| **R-ONCE** | "composition by declared operator" has no operator; addition is always available | **`VarianceBudget` (1(a)) and `LatencyBudget` (3(d))** — typed registries that reject overlapping support. σ_⊥-on-top-of-κ raises `DoubleCount` at registration, which is the exact bug that inflated σ_eff by 32–41 %. |
| **R-PROVENANCE** | provenance is on 2 of 22 modules, does not compose, does not cover conventions | **provenance as a field of the value type, combined automatically (`min` over the lattice)**, plus `@requires_provenance(min=FITTED)` on every decision entry point. Extend to derived-field *semantics* (2(c)3). |

**Two rules I propose adding**, because two of the findings above are not covered by
any of the four:

- **R-KNOWLEDGE-TIME** — *no state may be read by event time*. Enforcement: `StateView`
  exposes no event-time filter (1(d)); a test asserts `f.t_known <= now` on every
  field of every view produced during replay.
- **R-NO-ECONOMIC-GATES** — *a boolean gate may not be derived from an estimated
  revenue or cost quantity*; it may reference exogenous budgets. Enforcement: closed
  `InfeasibleReason` enum + C1's import restriction (1(b)). This is the structural
  form of FATAL-2, and without it FATAL-2 recurs the next time someone finds a
  safety condition worth respecting.

---

## 5. Dependency rule — cycles and violations

**Meta-finding first: the rule is stated at PLANE granularity, which is too coarse to
bind.** "Planes depend downward only" says nothing about B2→B3, C2→C3, C4→C5 — and
the intra-plane graph is where the real coupling lives. The Belief plane alone has
6 modules and at least 7 internal edges. **MUST-FIX: publish a module-level DAG and
test it** (`modules.yaml` + a cycle test in CI). A plane rule that cannot see a
B2↔B3 cycle is decoration.

| # | edge | verdict | resolution |
|---|---|---|---|
| 1 | **B2 → B3 → B2** | **REAL CYCLE.** B2's basis/gap-fill variance (V3) exists only because B3 chose to gap-fill from Binance; B3 needs σ_eff from B2. | Invert: **B3 registers** `VarianceComponent("basis_gapfill")` into the budget B2 owns. Data flows up; ownership stays put. No cycle. |
| 2 | **B2 → book-implied σ (MAJOR-2/H-3)** | **would be a cycle** (σ_book needs `E_t[X_T]` from B1/B3). | σ_book is an **Evaluation object (E2)**, never a belief input. If it ever feeds B2, H-3's circularity ("agreement scored as skill") becomes structural. Write it into the contract: *B2 may not read E2*. |
| 3 | **B5/B6 → our own size (own-impact, X6)** | **REAL CYCLE** if implicit: belief depends on our decision. | Legal *only* as an explicit argument: `outcome(l, side, size, ...)`. Size passed **down as data**, not as a dependency on C2. Current signature `lambda_fill(ℓ, Q_ahead)` has no size arg, so today the cycle is impossible *and so is the model*. |
| 4 | **C1 → B5/B6 (ζ_snipe)** | downward, therefore "legal" — and that is the problem. | Forbid by R-NO-ECONOMIC-GATES (1(b)). Legality under the plane rule is exactly how FATAL-2 passed review. |
| 5 | **C2 → C3 → C5 → C2** (second leg must see post-first-fill q) | **REAL intra-tick CYCLE.** | Fix by ordering + jointness: `C4.headroom → C3.reservation(q_now) → C2.QuoteSet (all 4 sides jointly) → C5.check → C6.emit`. Declare the tick order in the contract. |
| 6 | **E3 ReplayHarness → D1/K2** | **VIOLATION of "Evaluation is read by nothing".** E3 does not read the stack; it *drives* it (file feeds, injected latency, injected clock). | Introduce an **`Environment` seam** (clock, feeds, venue, RNG) implemented by `LiveEnv` and `ReplayEnv`. The stack depends on the seam; E3 binds it. Without this, replay gets built by reaching into modules and "replay reproduces live" quietly stops being true. |
| 7 | **K3 KillSwitch → X1 (cancel-all)** | **VIOLATION**: cross-cutting "depends on none" but must command Execution. | Split: K3 publishes `halt: Known[bool]`, C1 reads it (downward, `HALTED`); the cancel-all *command* goes through the `Environment` venue handle. State the direction explicitly — an unstated one becomes a back-channel. |
| 8 | **C4 → PortfolioState** | not a cycle; **a missing node** (2(e)). | Add C7 PortfolioAllocator above C1..C6. |

**Identifier collision (R-SSOT at the namespace level).** `PM_ARCHITECTURE` silently
re-used IDs that are live and load-bearing in `PM_MM_PLAN`, which the architecture
itself declares remains "the archive of findings":

| ID | PM_MM_PLAN §13–15 | PM_ARCHITECTURE §2 |
|---|---|---|
| **B4** | the **link** function Φ (§13.3 queue: "B4 (link)") | **FlowModel** |
| **B8** | order-flow distribution ("NEW component B8") | *(absent — it is B4)* |
| **X1** | propagator / transient impact | **VenueAdapter** |
| **X2** | PM book's own information (microprice/OFI) | **SettlementAccounting** |
| **X3** | short-horizon Binance alpha | *(absent)* |
| **C7** | requote hysteresis ("C7 is still empty") | *(absent — it is C6)* |

"Run X2 next" is now ambiguous between *the pivotal fair-value experiment* and *CTF
accounting*. This is not pedantry: cross-document references are the interface
between the plan and the architecture, and that interface is currently corrupted.
Renumber once, in a machine-readable register, and add an ID-uniqueness test.
(My §2(e) proposal deliberately takes `C7` — if the plan's `C7` is kept, pick
another. That decision needs an owner, which is the point.)

---

## 6. Statefulness and time

**No module in the register declares state, and none declares how it is rebuilt.**
The repo has been burned by restart bugs before (`CLAUDE.md` pitfall #3: an
overwrite-instead-of-merge destroyed 163/176 symbol histories) and by path-coupled
binary gates amplifying tiny differences 10–20× (pitfall #4). Both hazards are
present here: `permitted/not` and `halt/not` are exactly the bifurcating binary
overlays of pitfall #4.

| module | state | rebuildable from tape? | cold-start hazard |
|---|---|---|---|
| D1 | conn/seq/reconnect | n/a | — |
| D2 | dedup set (`transaction_hash`), gap registry | yes (unbounded memory — needs a TTL) | duplicate ⇒ **2× intensity** (§13.1(2)) |
| D3 | book, **`queue_est`** | book yes; **`queue_est` NO** (path-dependent, unobservable) | silently confident queue ⇒ wrong `p_fill` |
| B2 | walk-forward fit (≤ d−1), variogram windows | yes, with warmup | cold σ̂ ⇒ p̂ error; a 2× σ error is 16 c of p̂ |
| B4/B5 | Hawkes/EWMA intensity | yes, with warmup | cold λ **under**-states arrivals ⇒ over-quoting |
| C3/C5 | `q_up, q_down`, cost bases, cash | **NO** — must reconcile from X2/venue | wrong inventory ⇒ wrong reservation *and* wrong cap |
| C4/C7 | portfolio aggregate | from C3 + venue | aggregate breach invisible |
| C6 | open orders, rate-budget window | from venue | double-placement |
| K2 | measured latency EWMAs | yes, with warmup | assumed-provenance latency in a live gate (FATAL-1) |
| K3 | trip latch | **NO** — must persist | **restart un-trips the kill switch** |

```python
class Stateful(Protocol):
    WARMUP: timedelta                  # declared, per module
    COLD_START_SAFE: bool
    def snapshot(self) -> bytes: ...
    def restore(self, b: bytes) -> None: ...
    def warm(self) -> bool: ...        # C1 gates quoting on all(m.warm())  -> WARMUP reason
```

**Replay determinism contract (E3), stated as testable invariants:**

1. Given the same tape + `ParamStore` snapshot + seeds, the stack emits an
   **identical `QuoteSet` sequence**. Assert on **intents**, not PnL — pitfall #4
   says PnL deltas through path-coupled binary gates are 10–20× noise.
2. No wall-clock read outside K2 (lintable); no unordered iteration; RNG owned by
   the `Environment` seam.
3. `R-KNOWLEDGE-TIME` assertion active during replay: every field of every view
   satisfies `t_known <= now`. This is the test that would catch a `t_event`-filtered
   `as_of` — and it is cheap.
4. Restart-in-the-middle test: snapshot at `t`, restore, replay forward, assert the
   intent sequence matches the uninterrupted run. `COLD_START_SAFE=False` modules
   must produce `Unavailable(WARMUP)`, not a plausible number.

---

## 7. Triage

### MUST-FIX (7)

| id | fix | §  |
|---|---|---|
| **B-1** | **Knowledge time ≠ event time.** `Known[V]{t_event, t_known}`, `view(now: KnowledgeTime)`, no event-time filter API, `deficit_span` exposed. Measured 1,700 ms gap; look-ahead exceeds settlement vol inside the final ~2 s. | 1(d) |
| **B-2** | **`VarianceBudget` registry + split `w_declared` / `w_hat`.** Typed non-overlapping composition; belief uses σ̂² profiled at declared w; joint fit is a K3 diagnostic. Makes R-ONCE enforceable and blocks the 4th double-count. | 1(a) |
| **B-3** | **De-economise C1.** Closed non-economic `InfeasibleReason` enum; `ζ`/`ê` move into C2's objective; C1 may not import B5/B6. Structural form of FATAL-2; also makes the Option A/B swap real. | 1(b), 3(b) |
| **B-4** | **Merge B5+B6 into `RestingOutcome`**, conditional-on-fill, with `size` as an argument and a declared cost partition. Current split encodes the unconditional-ζ error; current signature forbids own-impact. | 1(c) |
| **B-5** | **Add `C7 PortfolioAllocator` + `PortfolioState`.** Three shared budgets (loss, rate, capital) across 7 coins have no owner; effective breadth ≈1–2. | 2(e) |
| **B-6** | **Module-level DAG + machine-readable ID register with a uniqueness test.** Plane-granularity rule cannot see the B2↔B3, C2↔C5, E3→D1, K3→X1 edges; and B4/B8/X1/X2/C7 collide with `PM_MM_PLAN`. | 5 |
| **B-7** | **Typed `Unavailable` on every Belief method + composing `Provenance` (min over lattice) extended to field semantics.** Without composition, "`assumed` may not gate a decision" is unenforceable — it is how FATAL-1 passed. | 2(a), 2(c) |

### SHOULD-FIX (8)

- **B-8** `WindowCtx.phase ∈ {pre, in, post_T, resolved}`; C1 closes `post_T`. (8.3 % of notional, one measurably toxic at −806 bps.)
- **B-9** `Environment` seam (clock/feeds/venue/RNG) so E3 hosts rather than reads; plus the `Stateful` snapshot/warmup contract and the 4 replay invariants.
- **B-10** Add `B7 Link` (Φ / Student-t / isotonic) with its own provenance — Φ under-prices the tail up to 28×, and nobody owns it.
- **B-11** Unit-tagged scalars across the register (`Prob/Cents/Bps/USD/Shares/Sec`) + a dimension check per formula. Would have caught §16.3 and §16.4.
- **B-12** `QuoteSet` (4 sides, jointly) replaces singular `QuoteIntent` as C2's output.
- **B-13** C5 declares: `q_up,q_down ≥ 0` (no naked short, P-M3d unverified), average-cost basis, merge books no PnL, partial-fill residual adversely selected.
- **B-14** Assign the FATAL-3 STOP clause: `E4.stop_triggers()` (pre-registered, named owner), evaluated by K3 alongside `halt_triggers()`.
- **B-15** `K2.path(PathId)` — named composite legs, no bare `L`; `US_TO_ACK` stays `assumed` until measured and therefore fails `@requires_provenance`.

### NOTED (5)

- `B3.confidence` → `var_p: Prob2` (naming, but it is the A/B leak's proximate cause).
- `tick` moves from `WindowCtx` to `MarketState` (state-dependent, 328 transitions / 130 windows).
- `D2.gaps(span)` as a queryable object — B2 must skip the 56 recorded >5 s gaps.
- D2's dedup set needs a TTL; it is currently unbounded.
- X2 needs a post-`T`-pre-resolution contract (median trade T+21 s, resolution T+85 s).

---

## Appendix — replaying the charter's change log against the CORRECTED structure

Scored as: modules touched if the fixes above are applied.

| # | change | blast radius | note |
|---|---|---|---|
| 3 | σ static → HAR → variogram | **1** (B2) | only if `w` is split (B-2); as written it is 2 (B1+B2) |
| 5 | v(t) sum → min | **1** (B2 budget) | `register()` raises on the double-count instead of shipping it |
| 6 | pull τ_min → surface → participation region | **1** (C2 objective) | as written it is 2 (C1+C2) — B-3 fixes |
| 7 | Q_max variance → loss cap + aggregate | **2** (C4 + new C7) | as written: unbounded, no portfolio owner |
| 8 | pair net q → (q_up,q_down) | **2** (C3, C5) | matches the charter's ideal |
| 11 | latency 120 → 471 → 1700 ms | **0** (one K2 leg) | *only* with B-15; a bare `L` makes it 3 documents |
| 13 | late-added components (flow, propagator, microprice, alpha, cross-coin, portfolio risk) | **new module each** | except portfolio risk, which needs B-5 first |

The two changes that would still hurt: **#2 fair-value source** (B3 swap drags C1/C2
until B-3 lands) and anything touching **knowledge time**, which currently has no
representation at all and would therefore be a whole-stack edit. Both are MUST-FIX.
