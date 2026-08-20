# PM_STRUCT_ITER2_B — lens B (CONTRACTS & BOUNDARIES), iteration 2

Object: `PM_ARCHITECTURE.md` **v2** (rewritten 2026-08-20). Prior: `PM_STRUCT_ITER1_B.md`
(7 MUST-FIX), `PM_STRUCTURE_REVIEW_LOOP.md`, `PM_QUANT_REVIEW.md`, `PM_DEEP_REVIEW.md`,
`PM_MM_PLAN.md`, and the sibling lenses `PM_STRUCT_ITER1_{A,C}.md`.

**Verdict.** v2 is a large, genuine improvement: the layer map is right, the Decision
plane is now cut along the problem, and the three headline fixes are all *present*.
But two of the three are **partial**, and the document has acquired a new instance of
the exact error class it was rewritten to kill.

- **1(a) knowledge time — landed, NOT airtight.** Four residual look-ahead paths, one
  of which (`t_known` is *fabricated* on backfilled data) makes the `t_known ≤ now`
  assertion pass trivially on the only history we have. The read API is clean; the
  *construction* of `t_known` is unowned.
- **1(b) `VarianceBudget.register()` — currently theatre.** "Declared non-overlapping
  support" over free-form strings is unfalsifiable: σ_⊥ tagged `{"basis"}` and κ(r)
  tagged `{"relay_deficit"}` are disjoint as strings and overlapping as variance, and
  that is precisely the 32–41 % inflation the registry exists to block. It passes.
- **1(c) `{max_size, shadow_price}` — partial.** Three escape routes survive, and one
  of them (a solver that clips at `max_size` and never prices λ) is *easier to write*
  than the correct behaviour, which is how FATAL-2 happened the first time.

**New finding of the same class as the old ones, and the headline for this iteration:**
R-ONCE now has an enforcement point for **variance** and none for **PnL**, and v2's own
`DE-Objective` term list already double-counts twice — `rewards` appears as an objective
term *and* as the priced constraint two lines below it, and `adverse_selection` sits
next to a `BE-FlowAndFills` that returns `E[markout | fill]`, which already contains it.
The v1 disease (σ_⊥+κ, v(t), running-vs-terminal — "one bug three times") has moved from
the variance plane to the objective plane, uncaught, because the fix was scoped to
variance rather than to *additive composition of owned quantities*.

**7 MUST-FIX, 9 SHOULD-FIX, 7 NOTED.**

One more finding worth the headline slot, because it is not a contract *gap* but a contract
that is **wrong about the venue**: `BE-FlowAndFills.rest(level, …)` is scoped to a single
instrument, and Polymarket's book is unified across the token pair — an Up bid at 0.60
crosses a Down bid at 0.40 via CTF mint, atomically at match time. Our fills arrive from a
book we are not quoting. A per-instrument fill model cannot be repaired by better
estimation, and this is the same failure shape as v1's missing `size` argument: the
signature forbids the correct model.

---

## 0. Scorecard on iteration-1 MUST-FIXes

| id | demand | v2 status |
|---|---|---|
| **B-1** | `Known[V]`, `view(now)`, no event-time filter, `deficit_span` | **LANDED**, incomplete — §1(a) |
| **B-2** | `VarianceBudget` + `w_declared` ≠ `w_hat` | **w-split LANDED** ✅ (§2, explicit). Budget landed **as declaration only** — §1(b) |
| **B-3** | de-economise constraints | **LANDED in shape**, three escapes — §1(c) |
| **B-4** | merge fill + adverse selection, `size` an argument | **LANDED** ✅ — `rest(level, size, horizon, state)`. Cost **partition** dropped — §2.6 |
| **B-5** | portfolio allocator + `PortfolioState` | **DROPPED.** `Σ_c` is named as a constraint in §5 with no module able to evaluate it — §3 row 7 |
| **B-6** | module-level DAG + ID register | **PARTIAL.** Namespace note ✅ (real fix, removes the "run X2" ambiguity). No DAG, no uniqueness test — §3 |
| **B-7** | typed `Unavailable` + composing `Provenance` | **DROPPED.** No contract in v2 returns absence; provenance has no combination rule — §2.5, §2.7 |

Sibling lenses: A's objective-ownership and C's SP-Venue/R-VERSION/R-NULL all landed.
C's `valid_for` on parameters (S-C6) and R-NULL's **bias direction** (M-C3) did not —
both matter below (§2.4, §2.6).

---

## 1. Verification of the three headline fixes

### 1(a) `Known[V]` + `view(now)` + `deficit_span` — is it airtight?

**What is genuinely airtight now.** The reordering attack fails. A consumer that holds a
collection of `Known` values *can* legally sort by `t_event` — it must, to integrate a
TWAP — but if the collection it holds was truncated by `t_known ≤ now` at the view
boundary, **no reordering of a truncated set can produce a future value.** That is a
real guarantee and it is the right one. The distinction to write into the contract is:

> **Ordering by `t_event` is legal. Comparing `t_event` against a decision clock is not.**
> Enforce by types: `EventTime` and `KnowledgeTime` are distinct `NewType`s with no
> cross-type comparison operator. `t_event <= now` must be a type error.

**Condition on which the guarantee rests, and which v2 does not state.** Truncation must
happen at the *collection* level, not only on scalar "latest" accessors. If `StateView`
exposes a `Tape`, a gap registry, a history buffer, or if a BE/DE module can be handed a
`DA-Feeds` or `DA-State` handle directly, the truncation is bypassed and the guarantee
is void. **SHOULD-FIX B2-8:** BE/DE modules receive a `StateView` and never a `DA-State`
handle; every history accessor on `StateView` returns a `t_known`-truncated sequence;
a declared-input manifest + injection makes "reaching for the raw feed" a wiring error
rather than an import.

Four residual look-ahead paths:

---

**L1 (worst). `t_known` is *fabricated* on every byte of history we own, and the
assertion cannot see it.**

`t_known` is an *observation* — `recv_ns + decode` on our box. It exists only for data we
were running to receive. But:

- the post-settlement-change backfill is **~12 days** and is pulled from Gamma/CLOB REST
  and on-chain (`PM_MM_PLAN` §"Historical backfill"): **there is no `recv_ns`.**
- the pre-2026-08-07 archive has none either.
- `DA-Discovery`/`PolygonRPC ⚠️` produce records whose only timestamp is the venue's.

The natural implementation is `t_known := t_event` (or `t_event + assumed_lag`). The
invariant `t_known ≤ now` then holds **by construction**, `EV-Replay`'s assertion passes,
and the 1,700 ms peek is back — now type-laundered and invisible to review. This is worse
than v1: v1's `as_of(t)` was at least *legible* as a bug.

`Known.provenance` does not cover this. It is the provenance of the **value**; the field
that needs one is **`t_known` itself**.

```python
KnownProv = Literal["OBSERVED",      # recv_ns on our box; the only safe one
                    "IMPUTED",       # t_event + a MEASURED lag distribution, with error
                    "ASSUMED"]       # t_event + a guess, or t_event itself

@dataclass(frozen=True)
class Known(Generic[V]):
    value: V
    t_event: EventTime
    t_known: KnowledgeTime
    t_known_prov: KnownProv          # <- MISSING IN v2. Load-bearing.
    t_known_err: Duration            # imputation sd; 0 for OBSERVED
    source: SourceId
    seq: int                         # total order within source; replay determinism
    provenance: Provenance           # of the VALUE
```

Gate, which is the point of the field:

> `EV-Replay` refuses — not warns — any result whose decision horizon `r` is within
> `k·t_known_err` of a non-`OBSERVED` input, and stamps every output with the worst
> `t_known_prov` among its inputs.

At `r = 2 s` with `t_known_err` on the order of the 1,440 ms publish leg, this refuses
the whole endgame study on backfilled data — **which is the correct answer**, and it is
the answer nobody will reach voluntarily once the numbers look good. `MAJOR-10` prices the
1.7 s head start at ≈0.08 bps against a settlement vol of 0.007 bps at `r = 2 s`: an
imputation error of the same order as the lag is not a precision issue, it is the entire
signal.

**MUST-FIX B2-1.**

---

**L2. `SelfState` has three knowledge times and v2 gives it one.**

Moving `SelfState` into `DA-State` is correct (§3 below agrees). But `SelfState` is the
one place where our knowledge is *asymmetrically* wrong in **both directions**, and a
single `Known` stamp cannot express it:

| fact | knowable at | status |
|---|---|---|
| **we submitted** an order | our submit clock — exact, ours | `OBSERVED`, zero lag |
| **the venue accepted** it | submit + `US_TO_ACK` | **never measured** — `assumed` 75 ms (US) / 3 ms (EU); FATAL-1(d) names it "the one leg we have never observed" (P-M2a) |
| **the order was live in the book** (what actually determines fills and queue position) | *earlier than the ack*, unobservable | not represented at all |
| **we were filled** | fill-notification lag, then on-chain for CTF position | separate, unmeasured |
| **our queue position** `Q_ahead` | never — reconstructed, path-dependent | the bracket exists for this |

**No own-order latency measurement exists anywhere in the corpus** — every figure quoted
(us-east-1→London ≈ 70–80 ms RTT, "Binance-event → PM-order-acknowledged ~120–250 ms") is
*inferred*, marked `[I]`. The only own-side timing we have is market-data WS delivery
(median 48 ms, p90 1,284 ms, **p99 8,971 ms** over 481,482 messages), and that is a
different stream whose interpretation is itself an open question (P-M7d, "run first — can
kill the thesis"). So `SelfState`'s knowledge times are not merely multiple; **every one of
them except `submitted` is currently `assumed`.**

Two opposite errors, both live:

- **Optimistic `SelfState`** (order marked live at submit) makes the fill model believe it
  is queued before the venue does ⇒ over-states `p_fill` and queue priority ⇒ replay
  **fabricates fills that could not have happened**. Note this is a look-ahead that the
  `t_known ≤ now` assertion *cannot* catch, because the timestamp is genuinely ours.
- **Pessimistic `SelfState`** (live only at ack) leaves us exposed but believing we are
  flat during the ack gap ⇒ `L_adv` and `κ_$` under-report exposure exactly when a burst
  is in flight.

The contract must carry both and route them differently:

```python
class SelfState(Protocol):
    submitted: Sequence[Known[OrderIntent]]   # t_known_prov=OBSERVED, ours
    acked:     Sequence[Known[OrderAck]]      # t_known from the venue feed
    fills:     Sequence[Known[Fill]]
    position:  Known[Position]                # on-chain confirmation for CTF
    def envelope(self, now) -> ExposureEnvelope
        # (lo, hi): lo = acked+filled only; hi = lo + ALL in-flight assumed filled
```

> **Routing rule (contract, not convention):** `DE-Constraints` evaluates on
> `envelope.hi` (pessimistic). `BE-FlowAndFills` evaluates on `acked`/`fills`
> (what the venue believes). No consumer may read a collapsed single `SelfState`.

**MUST-FIX B2-2.** Consequence for build order: `US_TO_ACK` is `assumed`, so under R-PROV
*any gate consuming the ack-gap envelope must fail `@requires_provenance`* until P-M2a is
measured — which is `OP-LatencyBudget`'s stated job and is already build-order step 1.
Good; the structure supports this once the field exists.

---

**L3. Derived quantities have no time-composition rule — and the intuitive rule is
backwards.**

Nothing in v2 says how a BE output is stamped. The task framing proposes "minimum
`t_known` of its inputs". **That is the wrong direction and would reintroduce the bug.**
Three different quantities compose three different ways:

| composed quantity | rule | why |
|---|---|---|
| `t_known(D)` | **MAX** over inputs | you cannot compute D before you know *all* of it. `min` claims D was actionable at the earliest input's clock — a look-ahead, dressed as conservatism |
| information age / staleness | **MIN over `t_event`** (report the full span) | D reflects the world no more recently than its *stalest* input |
| `deficit_span(D)` | **≥ MAX** over input deficits | the epistemic gap does not shrink under composition |
| `provenance(D)` | **MIN over the lattice** `measured > fitted > inherited > assumed` | one assumed input taints the result — this is FATAL-1's shape |

Note the asymmetry: **knowledge time takes MAX, provenance takes MIN.** Two composition
rules pointing opposite ways on the same object is exactly the kind of thing that gets
implemented once, wrongly, and never revisited. It must be computed **inside the value
type**, not by each module. **MUST-FIX B2-3.**

Concretely: `p̂` derived from a locked TWAP integral (`t_known` = last stream tick + 1.7 s)
and a Binance increment (`t_known` = Binance leg) is knowable at the **later** of the two,
carries the **older** `t_event` as its information age, and inherits `assumed` provenance
if either leg is assumed.

---

**L4. `deficit_span` is a scalar, and the coverage it summarises is not contiguous.**

v2 kept `deficit_span(field, now) -> Duration` and dropped `covered_span`/`locked`.
A scalar span implicitly asserts "the integral covers `[T−w, t_event_last]` with no
holes". The corpus records **56 gaps > 5 s** in the stream. A 5 s hole in the *middle* of
a 60 s TWAP is an 8.3 % weight deficit that `deficit_span` reports as **zero**, because
the last tick arrived on time. BE would price a gap it is not exposed to and not price the
one it is.

```python
DA-State.coverage(field, now) -> Coverage
# Coverage{ target: Interval[EventTime],      # what the estimand needs, e.g. [T-w, T]
#           covered: IntervalSet,             # what we actually have
#           weight_missing: float,            # fraction of the ESTIMAND's weight absent
#           tail_deficit: Duration }          # the 1.7 s at the right edge
# deficit_span() survives as `tail_deficit`; BE registers a VarianceComponent from
# `weight_missing`, not from the scalar.
```

Absence of coverage is the **expected state, not the exception**: full `[0, 300 s]` coverage
holds for **103/130 windows (79 %)**, and the `1013 slow consumer` disconnects recur on BTC
— the coin carrying 85 % of notional. A `Coverage` object that is usually complete would be
overhead; one that is incomplete 21 % of the time is load-bearing.

Also unstated: `DA-Normalize`'s gap registry must be **queryable**
(`gaps(span) -> IntervalSet`) so `BE-Uncertainty` can handle the 56 recorded gaps in a
variogram fit. Note the *handling* is not "exclude": outages cluster with bursts, so
dropping gap-containing intervals **selects on volatility and biases σ̂ down** — the same
direction as the dangerous case in QUANT §5.4, where under-estimating σ by 2× at |d| = 1
leaves us quoting **53 s deeper into the sniping zone**. The registry must therefore expose
gaps as an **indicator/weight** with a with-and-without report, not as a filter. Today the
whole thing lives in an analysis script. **SHOULD-FIX B2-9.**

**Related, and it breaks a different assumption than staleness does:** `DA-Normalize` must
declare that **per-source sequence order is not causal order.** Measured: the `price_change`
carrying `best_bid/ask` for a match **can be emitted before the `last_trade_price` for the
same match** — which silently corrupted the spread-capture / adverse-selection split
(capture measured at +0.11…+0.50 c against observed spreads of 1.1–1.8 c, far below a
half-spread). The `net` column was immune because it never uses the mid; the decomposition
was not. So `Known.seq` gives a total order for determinism and **does not** license
"the state just before this print". Any decomposition needs a **frozen-lag mid**
(e.g. mid as of `t_trade − 250 ms`) as a declared convention on `StateView`, not as a
per-analysis choice. This is one line in E-M1 and is currently absent.

---

**Also unstated: who supplies `now`?** If any module can call `view(now)`, a module can
call `view(now + ε)`, and in replay a module that reads a wall clock reads the operator's
clock, not the tape's. `now` must come from the `Environment` clock, be monotone
non-decreasing per tick, and modules must be handed an already-bound `StateView`.
Note `DE-Solver.maximize(Objective, Constraints, ActionSpace, BeliefProcess)` takes no
view and no clock at all (§2.9) — which is currently the *only* reason it cannot peek.

---

### 1(b) `VarianceBudget.register()` raising `DoubleCount` — checkable, or theatre?

**As specified in v2: theatre.** The rule is "raises `DoubleCount` on overlapping
support", and support (from my own ITER1 draft, which v2 adopted verbatim in spirit) is
`frozenset[str]`. A free-form string set is a *human declaration*, and the check is only
as good as the declaration. Two failure modes, both live:

1. **False negative — the original bug passes.** σ_⊥ (basis innovation) declares
   `{"basis"}`; κ(r) (relay deficit) declares `{"relay_deficit"}`. Disjoint strings.
   `register()` returns. But both are, over the same interval, partly the *same*
   Binance-vs-stream innovation, which is why adding them inflated σ_eff by **32–41 %**.
   The registry blocks nothing.
2. **False positive teaches mislabelling.** Two honestly distinct components that happen
   to share a tag collide; the cheapest fix is to rename a tag. The check then actively
   trains people to make the declaration less true.

A declaration-only check has no more force than the comment it replaced. **It needs two
layers: a support representation that is mechanically decidable, and an empirical test
that falsifies the declaration.**

**Layer 1 — decidable support.** Replace the string set with a typed product. The point is
that no human judgement about "is basis the same as relay deficit" is required; you must
name a *factor* and an *interval*, and both are checkable against objects the system
already owns.

```python
RiskFactor = Enum("X_SPOT", "X_TWAP_LOCKED", "BASIS_PERP_SPOT", "RELAY_GAP",
                  "MODEL_PARAM", "QUEUE", "OUTCOME_TIE")

@dataclass(frozen=True)
class Support:
    factor: RiskFactor
    span:   Interval[EventTime]        # the interval of X-path it covers
    op:     Literal["level", "increment"]

@dataclass(frozen=True)
class VarianceComponent:
    name: str
    value: Bps2
    space: Literal["X_bps2", "P_prob2"]   # <- v2 has no unit at all; see below
    support: frozenset[Support]
    owner: ModuleId
    prov: Provenance

# overlap iff  a.factor == b.factor  and  a.span ∩ b.span ≠ ∅  and op compatible.
# Mechanical. σ_⊥ and κ(r) BOTH claim (BASIS_PERP_SPOT | RELAY_GAP, [t_last, now],
# "increment") and now collide at registration, which is the point.
```

**Layer 2 — the falsification test, without which layer 1 is still a declaration.** The
budget is a *prediction* and PM's structural advantage is that ground truth exists (v2 §6
says so about calibration). Run the same test on variance:

```python
EV-Calibration.variance_audit(windows) -> VarianceAudit
# z_i = (X_T,i − E_t[X_T,i]) / sqrt(budget.total_i)
# report: var(z) with a day-clustered CI, PIT coverage at 50/80/95%, by (r, |d|) bin.
#   var(z) < 1  ⇒  budget too large  ⇒  DOUBLE-COUNT (the declared-disjoint kind)
#   var(z) > 1  ⇒  a component is MISSING from the budget
# A 32–41% sigma inflation shows up as var(z) ≈ 0.51–0.57 — unmissable on 1,206 windows.
```

> **The registry catches declaration *errors*. The PIT audit catches declaration *lies*.
> R-ONCE is not enforced until both exist**, and only the second is falsifiable. Wire
> `variance_audit` into `EV-Gates` as a standing gate, not a one-off study.

**Third gap: the budget has no unit.** Components live in different spaces — V1 (settlement
diffusion) and V3 (basis) are bps² of **X**; V4 (edge-measurement noise ω) is naturally in
**p̂** space. `total()` as specified will happily add them. Converting X-space to p-space
requires `G'` (delta method) — i.e. the **link**, which v2 places inside `BeliefProcess`,
a *consumer* of the budget. That is a cycle; see §3 row 2. Add `space` to the component
and refuse cross-space registration until the link is relocated.

**MUST-FIX B2-4.**

---

### 1(c) `feasible(action) -> {max_size, shadow_price}` — does FATAL-2's shape go away?

**Partly.** The shape is much better: a size and a price is the right object, and it is
what lets rewards be priced rather than gated. But a shadow price **can** be ignored, in
three distinct ways, and the v2 text asserts the property rather than enforcing it.

**Escape 1 — `max_size = 0` is a bool with extra steps.** Nothing in the type prevents
`{max_size: 0.0, shadow_price: inf}`. That is the participation frontier, re-encoded, and
the comment "NEVER a bool" is exactly the kind of instruction v2's own §0 says gets
violated within a day. Fix by **taxonomy at registration**:

```python
class ConstraintKind(Enum):
    HARD = auto()   # not tradable at ANY price: eligibility, tick grid, price bounds,
                    # lifecycle phase, rate limit, venue halt, kill-switch
    SOFT = auto()   # a BUDGET: L_adv, kappa_$, inventory, capital, rewards occupancy

# invariant, asserted at wiring:
#   kind == SOFT  =>  0 < shadow_price < inf  and  max_size > 0 whenever budget remains
#   kind == HARD  =>  shadow_price is None    (a price for it would be fiction)
# Adding a HARD constraint is a reviewable act, exactly as the closed InfeasibleReason
# enum was. "Infeasible because the moat is thin" has no HARD member to hide behind.
```

**Escape 2 — and this is the one that will actually happen — the solver clips and never
prices.** `DE-Solver` impls include `PerLevelEV` and `ClosedFormGLFT`. A greedy per-level
solver that takes `min(desired_size, max_size)` and never adds `−λ·usage` to its objective
is *behaviourally identical to FATAL-2* while satisfying every line of the v2 contract.
It is also the shorter implementation. The interface must make pricing unavoidable:

> **The shadow price must reach the solver through `DE-Objective`, not through
> `DE-Constraints`.** `DE-Constraints` emits a term
> `constraint_cost(action) = Σ_c λ_c · usage_c(action)` which registers into the objective
> term list. A solver that maximises the objective *necessarily* prices the constraint;
> it cannot skip it without ignoring the objective it was handed.

Plus the only test that can detect a clipping solver, which is cheap and replayable:

```python
# KKT audit, asserted on every replayed decision:
#   lambda_c >= 0
#   lambda_c * slack_c(action*) ~= 0        # complementary slackness
#   |dObjective/d(budget_c)| ~= lambda_c    # numerically, on a perturbed budget
# A solver that clipped shows lambda_c > 0 with slack_c > 0. Fails loudly.
```

**Escape 3 — the shadow price is not computable at the scope where `feasible()` sits, for
the two constraints that most need it.**

- `Σ_c q_c(1−p̂_c) ≤ L_max` **across coins** is a portfolio dual. A per-market
  `feasible(action)` cannot compute it — it does not know the other six coins' demand.
  With effective breadth ≈1–2 (one crypto factor), this binds *often*. **There is no
  portfolio module in v2** (my B-5 was dropped); §5 states the constraint and no owner
  exists. Whichever window's tick fires first consumes the budget — a race, not an
  allocation, and non-deterministic, which also breaks replay parity (§4).
- **rewards** (`R/X ≥ c(|d|,r)`) is an **occupancy integral over the window**, per
  `PM_MM_PLAN` Change 1. Its dual λ_R depends on the *path*, not on a single action.
  `feasible(action)` — v2's signature takes **no state, no view, no horizon** — cannot
  express it.

Corrected signature (the v2 one is not merely terse, it is missing its arguments):

```python
DE-Constraints.evaluate(action, view: StateView, self: SelfState,
                        port: PortfolioState, horizon: Sec) -> Feasibility
# Feasibility{
#   max_size: Shares,                       # 0 only if some HARD constraint binds
#   binding:  list[ConstraintId],           # WHICH — else no diagnosis, no KKT test
#   usage:    dict[ConstraintId, float],    # consumption per unit of action
#   shadow:   dict[ConstraintId, Price],    # SOFT only; unit declared per constraint
#   hard_reasons: list[HardReason] }        # closed enum
```

Units on `shadow` are mixed across constraints and v2 declares none: λ_R is
$ per occupancy-second; the `L_adv` dual is $/$ ; the inventory dual is cents/share.
Each `ConstraintId` must declare its `usage_unit`, and `constraint_cost` must reconcile
to the objective's numéraire. This is the M-5 "occupancy-integral-vs-per-fill unit
mismatch" that `PM_STRUCT_ITER1_A` X-2 already flagged as homeless; it is still homeless.

**MUST-FIX B2-5** (constraint kind + cost-into-objective + KKT audit + full signature).

---

## 2. Audit of the NEW contracts

Per contract: **completeness** (what a real impl needs and cannot get), **units**,
**absence/error semantics**, **leakage** (must a consumer know which impl produced it?).

### 2.0 The cross-cutting finding: R-ONCE covers variance and not PnL

v2 built one composition registry, for variance. Additive composition of *owned economic
quantities* is the same hazard, and v2's own §5 already contains two instances:

| # | double-count, live in v2 §5 | why |
|---|---|---|
| 1 | `rewards` is an **objective term** *and* the worked example of a **priced constraint** | "…it is what lets rewards be a constraint WITH a price" sits two lines under a term list containing `rewards`. Counted twice, or counted twice with opposite sign. `PM_MM_PLAN` Change 1 settles it: **constraint with a price**, and the PnL-line framing is "superseded twice" (`PM_DEEP_REVIEW` §contradictions, L174) |
| 2 | `adverse_selection` is an objective term while `BE-FlowAndFills.rest()` returns `E[markout \| fill]` | conditional markout **already is** spread capture net of adverse selection. Adding an `adverse_selection` term on top subtracts it twice — the identical shape to σ_⊥+κ |
| 3 | `impact` is an objective term while `rest()` also returns `own_impact` | is `E[markout \| fill]` already net of own impact? Undeclared. Same shape again |

**Fix: an `ObjectiveTermRegistry` with the same discipline as the variance budget.**

```python
@dataclass(frozen=True)
class Term:
    name: str
    unit: Literal["USD", "cents_per_share", "USD_per_s"]
    basis: Literal["per_fill", "per_share", "per_second", "per_window"]
    measure: Literal["conditional_on_fill", "unconditional", "occupancy"]
    covers: frozenset[EconSource]   # {SPREAD, ADVERSE_SEL, REBATE, FEE, OWN_IMPACT,
                                    #  INVENTORY_RUNNING, INVENTORY_TERMINAL, REWARD}
    owner: ModuleId
    prov: Provenance
    def value(self, action, view, self_state) -> float

class ObjectiveTermRegistry:
    def register(self, t: Term) -> None:
        # raises DoubleCount on overlapping `covers`
        # raises UnitMismatch unless unit/basis/measure reconcile to the numeraire
```

`basis` and `measure` are not pedantry: summing a per-fill term and an occupancy integral
without declaring the measure they are integrated against is a unit error of the same
family as §16.3 (taker fee multiplied by price: an ATM `r*` of 123 s that is really 46 s)
and §16.4 (`γ|q|v` has units of *price*, so "a constant risk budget" is dimensionally a
skew cap — QUANT: *"the label and the formula disagree"*).

**MUST-FIX B2-6.**

---

### 2.1 `SP-Venue`

**Completeness gaps.**

- `tick_grid` is written as a static field. Measured: the tick is **state-dependent**
  (0.01 → 0.001 near the extremes) with **328 transitions over 130 windows**, and
  `PM_MM_PLAN` Change 2 calls it "a **first-order economic parameter**, not a
  microstructure footnote" — far from the money the moat *is* the tick (the fee is ≈78 %
  of the moat ATM, ≈2 % at |d| = 3). Needs `tick(px) -> Price` plus
  `price_bounds: (0.01, 0.99)`.
- `capabilities: {CTF_PAIR, NEG_RISK, ...}` are **booleans**, but §5 says
  `DE-ActionSpace` is "derived from SP-Venue capabilities" and the action space contains
  `mint` and `merge`. **A boolean cannot parameterise an action.** Mint/merge need cost,
  gas, latency, atomicity, and `k` (neg-risk is k-way, not 2-way). Capabilities must be
  `{name: CapabilityParams | None}`.
- **`matching: PRICE_TIME|BATCH` cannot express this venue's actual matching rule.**
  The book is **unified across the token pair**: an Up bid at 0.60 crosses a Down bid at
  0.40 via **CTF mint, atomically at match time**. Consequences below in §2.6 — this is a
  MUST-FIX, because a per-instrument fill model is not merely incomplete here, it is wrong.
- Missing: `min_size_for_rewards` (**50 shares**) vs `min_order_size` (**5**);
  per-market `rate_per_day` ($10,000/day BTC, $1,666.67 ETH/HYPE, $833.33 BNB/DOGE);
  the reward weight **`((1.5 − s)/1.5)²`**, which **does not depend on `|d|` at all** while
  toxicity is φ(d)-driven; the **scope** of the rate limits (per key / per market / per IP
  — with 28 quote streams across 7 coins × 2 tokens × 2 sides, scope decides whether the
  budget is one resource or 28; measured **3,500 orders/10 s**, non-binding);
  self-trade prevention; cancel-on-disconnect; amend-vs-cancel semantics.

**A spec field is not a value — it is a claim from a source, and this venue's sources
disagree, go stale, and do not cover our markets. All three at once, measured.**

| what | evidence |
|---|---|
| **sources disagree, simultaneously checked** | CLOB registry `rewards_max_spread` = **1.5 c**; Gamma `rewardsMaxSpread` = **4.5 c**, on the same live markets |
| **the spec changed mid-programme** | registry rows carry `start_date: 2026-08-20` ⇒ the band was re-cut **4.5 → 1.5 c during collection**; Gamma served the stale value |
| **and neither source covers our instruments** | M2 CORRECTION: 5-min crypto markets are **absent from the rewards registry entirely** (pagination exhausted, **33 pages / 16,172 rows**); `GET /rewards/markets/<cid>` returns empty. *"Neither Gamma nor the CLOB registry provides verified rewards params for 5-min crypto markets, and even their reward-eligibility is unconfirmed."* |

A versioned record holding one value per field can represent none of these. It cannot say
"two sources disagree right now" (a monitorable event, and a real one), and — worse — it
has **no way to distinguish `assumed` from `unknown`**. The 1.5 c figure that a `Provenance`
would mark `inherited` was read off markets **that are not ours**. Under R-PROV that is not
a weak value, it is *not a value*; and yet the rewards objective/constraint is one of the
programme's two economic pillars (pool **$550k/month, August only**, ≈ **$34.7 per BTC
window** sampled ~5 times ⇒ a high-variance lottery).

```python
SpecField[V] = Resolved[V]{value, source, prov, valid_from, valid_to}
             | Disputed[V]{by_source: dict[SourceId, V], since}   # monitorable
             | Unknown{reason, sources_tried}                      # NOT the same as assumed
# SP-Venue.discrepancies() -> list[Disputed]  feeds OP-Monitor.
# R-PROV extends: a decision gate refuses `Unknown` outright, not merely `assumed`.
```

**SHOULD-FIX B2-10** (raise to MUST if the rewards term is retained in the objective —
an economic term whose parameters are `Unknown` is FATAL-1's shape with a worse input).

**Spec-plane tension.** `fee_schedule(p, side)`, `rebate_schedule` and `tick(px)` are
*functions*, but §2 declares specs are "**data**, not an abstraction layer" and §9 forbids
a venue abstraction. Both are right; the resolution is to store `(family, params)` pairs —
declarative, serialisable, versionable, diffable — not closures. Otherwise the "data"
plane quietly becomes code and R-VERSION cannot diff it. **SHOULD-FIX B2-11.**

**`EV-Replay` "refuses to span a spec boundary" is too strict to survive contact.** With
settlement changed 2026-08-07 and the rewards band re-cut 2026-08-20, the longest legal
replay window is **~12 days**, and the current one is **under a day**. Under that rule the
programme becomes unevaluable at the next venue change — and the pool is announced for
**August only** (`$550k/month`, M3), so at least one more boundary is scheduled. Refine to
**field-level dependency**: an analysis declares which spec fields it depends on; replay
refuses only if a *depended-on* field changed. A fee change must not invalidate a
settlement study. **SHOULD-FIX B2-12.**

### 2.2 `SP-Instrument`

- **Windows are not independent and nothing says so.** `PM_QUANT_REVIEW`: back-to-back
  5-min markets share a 60 s average — *"`X_0` of window k+1 **is** `X_T` of window k"* —
  giving a mechanical MA(1) `ρ = (w/6)/(T − w/3) = +0.0357` (MC +0.0347 ± 0.0018) and an
  outcome-sign correlation ≈ +0.023. Small per pair, but it is a *known* structural
  coupling with no field, and the clustering unit for every gate is the **day**, not the
  window. Declare it: `overlap_with: (prev_instrument, shared_span)`, and let EV read the
  clustering unit off the spec rather than each analysis choosing one.
- Missing: the **boundary reader `b(·)`** (MAJOR-7 — the observations-vs-`validFrom`
  dimension is *not identifiable from our data at all*, so it is permanently `assumed`
  and must be typed that way); **lock dynamics** (when the integral begins locking);
  **resolution lag** (median trade at **T+21 s**, resolution at **T+85 s**, post-`T`
  harvest measured at **−1.46 c/share = −806 bps** on 0.9 % of notional); **redemption**.
- `complement: Some(token) | None` cannot express a **k-way** neg-risk set. Portability
  limit, but it is a *contract shape* problem: `parity: SumToOne(tokens) | NegRisk(k) |
  None`.
- `payoff: BINARY|LINEAR` with `tie_rule` under settlement is fine; `δ_tie` still has no
  owning module (C's S-C8, unlanded).

### 2.3 `SP-Strategy`

- **No `valid_from`/`valid_to`**, while SP-Venue and SP-Instrument have them. If Option
  A/B is "a record, not a code branch" (§7 — correct, and the strongest structural claim
  in v2), then EV cannot attribute a day's fills to a strategy unless the record is
  versioned exactly as the venue specs are. R-VERSION must cover it.
- No binding to a **SP-Params version** and no **prereg id**. §6 notes `PM_PREREG.md` does
  not exist, so nothing is frozen; the strategy record is where a frozen configuration
  should be *named*.
- **`nulls{module: declared_assumption}` is a string, and strings do not compose with
  R-PROV.** This is the same theatre as `support`. R-NULL produces prose; R-PROV consumes
  a `Provenance`; nothing connects them, so a null's assumption cannot reach
  `@requires_provenance`. Make a null **pin a parameter**:

```python
nulls = { "BE-FlowAndFills": NullDecl(
              pins={"zeta": 0.0, "p_fill": 1.0, "own_impact": 0.0},
              prov=ASSUMED,
              bias="OPTIMISTIC") }     # C's M-C3 bias direction; unlanded in v2
# The pins land in SP-Params as `assumed` -> any decision gate consuming them RAISES.
# That is R-NULL and R-PROV composing, which is the only way either binds.
```

  **And v2's own worked example under-declares.** §0 says *"null `FlowAndFills` ⇒ ζ = 0"*.
  Nulling `FlowAndFills` pins **three** things — ζ = 0 *and* `p_fill` at some assumed
  constant *and* `own_impact` = 0 — because `rest()` returns all of them. A null that
  declares one of its three assumptions is the failure R-NULL exists to prevent, appearing
  in R-NULL's illustrative example. **SHOULD-FIX B2-13.**

### 2.4 `SP-Params`

`{value, provenance, owner_module, measured_at, source}`. Four gaps, and one of them
breaks §4 (replay):

1. **No key/scope.** A flat `name` re-creates the 120 / 471 / 1,700 ms collision the
   moment a second venue, coin or horizon exists (the 471 ms was measured on the Binance
   **spot mirror** and applied to the TWAP stream, 3.6× slower — a keying failure, not an
   arithmetic one). Needs `key = {venue, coin, horizon, epoch}`.
2. **No `valid_from`/`valid_to`.** R-VERSION's enforcement point is *"EV-Replay refuses to
   span a spec boundary"* — but SP-Params is not a spec under that rule, so
   `params.get(name)` resolves **today's** value inside yesterday's replay. That is a
   look-ahead in parameter space, and it is the repo's own pitfall class (#1/#2/#5).
   Needs `params.at(t) -> Known[value]`.
3. **`measured_at` ≠ `fit_data_through`.** For any `fitted` parameter (variogram σ̂,
   blend weight `w`, queue model, ζ surface — all walk-forward, fit ≤ d−1, apply d) the
   PIT-relevant field is the last date of the *fitting data*, not when the fit ran. v2 has
   only the latter. This is `PM_STRUCT_ITER1_A` X-7 (versioned model artifacts) and it is
   the mechanism by which EV's outputs legally re-enter the system (§3 row 4).
4. **No units and no uncertainty.** A value with no CI cannot propagate, and MAJOR-9
   (1.4 days is nowhere near enough to read the markout surface) then has no
   representation anywhere in the stack.

**MUST-FIX B2-3** covers (2)+(3) jointly with the composition rules; (1) and (4) are
**SHOULD-FIX B2-14**.

### 2.5 `BeliefProcess`

The object is right — `constituents` in particular makes plan-X2 (paired ΔBrier of model
vs book) expressible against the *production* module, which was the whole point. Gaps:

- **No absence semantics — this is the single largest omission in the new contracts.**
  What does `BE-Belief` return when the stream is stale? v2 has no answer, and the
  consequence is already visible in v1 and unrepaired: *stream-staleness handling lives in
  the decision plane* ("widen/pull" with a hand-tuned `N` that nobody owns). A downstream
  module re-deriving an upstream module's health from raw feed state is a leak by
  omission. Every belief method must return `T | Unavailable`:

```python
@dataclass(frozen=True)
class Unavailable:
    reason: Literal["STALE","GAP","WARMUP","OUT_OF_SUPPORT",
                    "UPSTREAM_UNAVAILABLE","NOT_IDENTIFIED"]
    since: KnowledgeTime
    cause: "Unavailable | None"     # propagates: BE-Belief -> UPSTREAM(BE-Uncertainty:WARMUP)
```

  `OUT_OF_SUPPORT` is venue-specific and load-bearing: `Q_max = 9.9e15` back-solves to
  **p̂ ≈ 2.5e-13** (|d| ≈ 7.2) — ten orders of magnitude beyond anything the data can
  support. No free-data model can assert that number. **And the clamp that would otherwise
  be used is not a safe default — it is an absorbing attractor.** At the floor, the
  cheapest ask has positive edge for *any* inventory (p̂ = 0.0050 ⇒ +0.50 c/share;
  p̂ = 0.0020 ⇒ +0.80 c/share), so the engine accumulates short-longshot inventory
  indefinitely — described in the corpus as "the single most dangerous interaction in the
  current design". A clamped p̂ is therefore worse than a refusal in a way that is
  *specific and measured*, which is why `Unavailable` cannot be optional.
  **MUST-FIX (part of B2-3's value-type work).**

- **Estimator uncertainty is gone.** v2 dropped `(p̂, confidence)` — correctly, it leaked —
  but replaced it with nothing. `DE` now has no basis on which to shrink toward the book,
  and MAJOR-9 has no home. Restore as `var_p: Prob2` (variance of the *estimator*, not of
  the outcome), with the single documented use `w = var_p/(var_p + var_book)`; an impl
  that cannot produce it returns `Unavailable(NOT_IDENTIFIED)`.
- **`jump_tail: E[(|J|−m)^+]` is a bare field with two unnamed arguments** (threshold `m`,
  horizon `h`). Must be `jump_tail(m, h)`. Its **unit** is also undeclared while its
  consumer (the sniping term) is in cents/share.
- **`path_law` is undefined.** `ClosedFormGLFT` needs a diffusion coefficient;
  `HJBQVI` needs a path generator; `PerLevelEV` needs a horizon variance. Left opaque,
  each solver will reach into the impl and the swap stops being clean. Minimum method set
  every impl can honour: `var_of_increment(h)`, `paths(n, h, rng)`, `jump_intensity(m)`.
- **`constituents` is a leak unless scoped.** A consumer reading `constituents["flb"]`
  must know which impl produced it. That is fine for EV and fatal for DE. Contract it:
  **`constituents` is EV-readable only**; `DE` may not branch on it.
- **`link: G`** — right call ("consumers must NOT hardcode Φ"; Φ under-prices the tail up
  to **28×** and misspecification biases the MLE **−3.5 … −15 %**). But its *placement*
  creates a cycle — §3 row 2.

### 2.6 `BE-FlowAndFills.rest(level, size, horizon, state)`

The merge landed and `size` landed — both were MUST-FIX and both are right (min order
**5 shares** vs median trade **9 shares** means own-impact is not optional). Four residual
problems:

1. **The cost partition was dropped.** The stated reason for merging B5+B6 was that the
   split "encodes the unconditional-ζ error" and lets ζ_snipe be *added on top*. Returning
   `E[markout|fill]` and `own_impact` as **separate** fields re-opens exactly that door
   (§2.0 row 3). Restore `decomposition: dict[str, Cents]` declared as a **partition that
   must sum to markout** — `{spread_capture, as_transient, as_permanent, snipe,
   own_impact}` — and register those names in the `ObjectiveTermRegistry`'s `covers`.
2. **The bracket is returned beside the answer instead of on it.** `queue_bracket`
   alongside `p_fill` forces every solver to combine them its own way, so the inherited
   gate *"sign flip across the queue bracket ⇒ fail"* cannot be applied uniformly. Return
   `Bracketed[Outcome]` (pessimistic/optimistic on `Q_ahead` **only**) with a
   `sign_stable()` method, so the gate is a contracted call.
3. **It answers one of six actions.** `DE-ActionSpace` = `{quote, cancel, mint, merge,
   cross, wait}`; `BE-FlowAndFills` models `quote` (rest) only. `cross` has no fill/slippage
   model, `cancel` has no queue-option cost (`D_ℓ`, requote hysteresis — "an impulse
   destroys queue value"), `mint`/`merge` have no cost or `P(complete)` model. The
   objective cannot price four of the six actions it is allowed to take. Widen to
   `BE-FlowAndFills.evaluate(action, view, self, horizon) -> Bracketed[Outcome] | Unavailable`.
4. **`state` must be a `StateView`, not `DA-State`** (§1(a)), and there is no
   `Unavailable` return.
5. **`rest(level, ...)` is scoped to ONE instrument, and the venue does not match that
   way — so the signature structurally forbids the correct fill model, exactly as v1's
   missing `size` forbade own-impact.** On-chain confirmed: the book is **unified across
   the token pair** — an Up bid at 0.60 crosses a Down bid at 0.40 via **CTF mint,
   atomically at match time** (`neg_risk = false`, so it is the plain binary CTF). Our
   resting Up bid is therefore filled by flow in the **Down** book, at a level we are not
   quoting, through a mechanism `SP-Venue.matching: PRICE_TIME` does not describe. Three
   consequences, all live:
   - `p_fill(Up @ a)` depends on the Down book's depth at `1 − a`. A per-instrument model
     gets it wrong in *both* directions and cannot be repaired by better estimation.
   - `E[markout | fill]` conditions on the wrong selection event: some of our fills are
     minted against a counterparty who wanted the *complement*, whose information content
     differs from a same-token taker's.
   - `DE-ActionSpace` listing `mint` as a discrete action is misleading — mint occurs
     **implicitly inside a match**. The explicit action is only the *deliberate* mint;
     the implicit one has no representation, and it is the one that happens.

   The fill model's unit of scope is the **market (token pair)**, not the instrument:
   `rest(quotes: QuoteSet, view, self, horizon)` over both tokens jointly. Note this is
   the same object §2.10 needs for `JointPairSolve`, and it is what makes `P(complete)`
   estimable at all (currently P-M3c, "kills the 'riskless' framing if it fails").
   **MUST-FIX B2-7.**

### 2.7 `DE-Objective`

Covered in §2.0. Additionally: no numéraire is declared; no sign convention; terms are
listed but not *registered*, so "a NEW theory that adds an objective term touches ONLY
this module" is true for the file and false for correctness — the new term can silently
overlap an existing one. The claim in the comment is the thing the registry must earn.

### 2.8 `DE-Constraints`

Covered in §1(c). Note the constraint set must be **individually togglable with reasons**
(C's S-C2): at a 4 h horizon `L = 1.7 s` vs `r = 14,400 s` makes the participation
condition non-binding for ~99 % of the window while inventory and drift start binding —
the rule *set* changes, not the rule parameters.

### 2.9 `DE-Solver`

`maximize(Objective, Constraints, ActionSpace, BeliefProcess)` — **the signature cannot
compute one of its own listed objective terms.** `inventory_penalty(running, terminal)`
requires `q`, and no argument carries `q`. Nor does anything carry `now`, a `StateView`,
a horizon, an RNG, or the portfolio. Corrected:

```python
DE-Solver.solve(belief: BeliefProcess, view: StateView, self: SelfState,
                port: PortfolioState, objective: Objective, constraints: Constraints,
                action_space: ActionSpace, horizon: Sec, rng: Rng) -> list[ActionIntent]
# + declared deterministic tie-break (required by replay parity, §4)
# + returns the KKT dual vector for the audit in 1(c)
```

### 2.10 `DE-ActionSpace`

- Missing **amend/replace** as distinct from `cancel`+`quote`: they differ in rate-budget
  cost *and* in queue priority (replace loses position), which is the entire requote-
  hysteresis economics.
- Missing the **joint action**. `JointPairSolve` is a listed solver, but if `Action` is
  singular there is nothing joint for it to return. `EV_pair = P(complete)·(1−a−b) +
  (1−P(complete))·EV_naked(first leg)` is a property of the **set**, and `1−a−b = 2δ`
  identically, so a leg-wise pair trigger is vacuous. The solver must emit a
  `QuoteSet` (all four sides, one object). My ITER1 B-12; unlanded.
- "Derived from SP-Venue capabilities" needs the capability *parameters* (§2.1), not
  booleans.
- Position conventions still undeclared and each has a PnL consequence: `q_up, q_down ≥ 0`
  (no naked short — shorts are longs of the complement; P-M3d, unverified),
  **average-cost** basis (FIFO splits realised/carried differently), **merge books no PnL**
  (it is a capital decision — $1 either way), partial-fill residual adversely selected by
  construction.

---

## 3. Cycles and dependency violations in v2

| # | edge | verdict | resolution |
|---|---|---|---|
| 1 | **`SelfState` in DA-State, read by BE** | **CLEAN as a read.** Our orders are observed data; the four theories (own-impact, `A(Q_ahead)`, `P(complete)`, P-M7b) need it. v2's call is right. | — |
| 2 | **DE-Actuator → DA-State (the WRITE)** | **REAL, unowned.** v2 says where SelfState *lives* and never says who *writes* it. If the Actuator writes it directly, DE→DA is an upward edge **and** live/replay diverge (live goes through the venue, replay through the back-channel). | The Actuator's outbox is part of the **`Environment` seam**; `DA-Feeds` reads the outbox as a feed (`t_known` = submit, `prov` = OBSERVED) and venue acks as another. The arrow goes through the *world*, not the code. Makes the in-flight set a first-class replayable object (§1(a) L2). |
| 3 | **BE-Uncertainty ↔ BE-Belief via `link`** | **REAL CYCLE, new in v2.** The budget must convert X-space bps² to p-space via `G'` (delta method), and `link: G` is a field of `BeliefProcess`; meanwhile `p̂ = G((E[X_T]−K)/σ)` needs σ from the budget. | The link is a **distributional-family declaration**, not a belief output. Hoist to `BE-Link` (impls `Probit | StudentT(ν) | Isotonic`, own provenance), depended on by Uncertainty *and* Belief. Also fixes "nobody owns Φ" (28× tail underpricing). |
| 4 | **BE-Uncertainty ← book-implied σ** | **would be a cycle** (σ_book needs `E_t[X_T]` and the book) **and** it structuralises H-3's circularity: agreement with the book scored as skill, while §6's calibration gate is *"paired ΔBrier vs book"*. | Write it into the contract: **σ_book is an EV object; BE may not read EV.** v2 does not say this. |
| 5 | **DE ← EV (fitted ζ surface, calibration map, markout params)** | **REAL and currently UNRESOLVABLE.** DE-Objective's `adverse_selection` is *calibrated by* EV-Markout; recalibrated p̂ comes from EV-Calibration. Either DE reads EV (violating "EV is read by none") or the fitted numbers have **no legal path into the system at all**. | The cycle must run **offline, through the spec plane**: EV writes `SP-Params` entries with `provenance=fitted`, `valid_from`, and `fit_data_through`; DE reads `SP-Params.at(t)`. This is the *only* mechanism that makes "EV is read by none" survivable, and it is also the walk-forward PIT guard. **v2 states neither half.** |
| 6 | **OP-KillSwitch → cancel-all** | **VIOLATION.** OP is "readable by all" but must *command* the venue. | Split: OP publishes `halt: Known[bool]`, DE-Constraints reads it as a HARD reason; the cancel-all command goes through the `Environment` venue handle. An unstated direction becomes a back-channel. |
| 7 | **`Σ_c` across coins has no owner** | **MISSING NODE** (my B-5, dropped). §5 states the constraint; per-market `feasible()` cannot evaluate it; nothing allocates the shared loss / rate / capital budgets across 7 coins × overlapping windows. | Add `DE-Portfolio` (state + allocator) above Constraints. Allocation by shadow price with a **deterministic tie-break** — first-come is a race and breaks replay determinism (§4). |
| 8 | **EV-Replay → DA-Feeds / OP-LatencyBudget** | **VIOLATION.** §1 says "EV reads all, is read by none", but Replay does not *read* the stack — it **drives** it (file feeds, injected latency, injected clock, injected RNG). | The `Environment` seam again: `LiveEnv` / `ReplayEnv` implement `{clock, feeds, venue, rng}`; the stack depends on the seam; EV-Replay binds it. Without this, replay gets built by reaching into modules and "replay reproduces live" quietly stops being true. |
| 9 | **plane-granularity rule vs intra-plane edges** | rows 3, 4 and 5 are all *intra-plane or spec-plane* edges that a plane rule cannot see. | Publish a **module-level DAG** (`modules.yaml`) with a cycle test. My B-6; the namespace note landed, the DAG did not. |

---

## 4. Replay-parity contract

**Is `assert t_known ≤ now` sufficient? No.** It is necessary, cheap, and catches exactly
one thing: a module *reading* ahead through the view. It catches none of the following,
and the first is fatal.

| # | leak the assertion cannot see | required contract |
|---|---|---|
| **P1** | **`t_known` fabricated on backfill** (§1(a) L1) — the assertion passes *by construction* on the only history we have | `t_known_prov` + `t_known_err`; replay refuses when `r` is within `k·t_known_err` of a non-OBSERVED input; every output stamped with the worst input provenance |
| **P2** | **Tape built by joining feeds on `t_event`** — a `merge_asof` on payload timestamps leaks before any module runs, and `direction="nearest"` leaks the future outright | The tape is **one stream totally ordered by `(t_known, source, seq)`**. Joining or resampling on `t_event` is banned in the replay builder and is lintable |
| **P3** | **Optimistic `SelfState`** — fabricated fills from orders the venue had not yet acked (§1(a) L2); the timestamp is genuinely ours, so the assertion is silent | replay drives `submitted → acked` through the injected `US_TO_ACK` leg; results inherit that leg's `assumed` provenance until P-M2a is measured |
| **P4** | **Warm-up state.** Replay starts cold; live has run for hours. Cold λ **under**-states arrivals ⇒ over-quoting; cold σ̂ ⇒ p̂ error (a 2× σ error is ~16 c of p̂). Divergence is silent *and favourable* | `Stateful{WARMUP, COLD_START_SAFE, snapshot(), restore(), warm()}` on every fitted module; wiring refuses to trade unless `all(m.warm())`; a not-warm module returns `Unavailable(WARMUP)`, never a plausible number |
| **P5** | **Parameter resolution.** `params.get(name)` returns today's value inside yesterday's replay | `SP-Params.at(t)` with `valid_from`/`valid_to` **and** `fit_data_through` (§2.4). R-VERSION currently covers specs only |
| **P6** | **RNG.** Bracket sampling, `paths(n,h)`, any stochastic solver | RNG owned by the `Environment`, seeded per `(window, tick)` so mid-tape restart reproduces. No module constructs its own |
| **P7** | **Non-determinism in ordering.** Unordered set/dict iteration (the dedup set on `transaction_hash`), first-come budget consumption across 7 coins (§3 row 7), solver ties | total orders everywhere; declared tie-break in `DE-Solver` and `DE-Portfolio` |
| **P8** | **Latency injection ≠ the live budget.** Replay can inject a latency the live system does not believe it has | the injected legs **are** `OP-LatencyBudget`'s legs, by reference, with provenance carried through |
| **P9** | **Asserting on the wrong object.** Repo pitfall #4: path-coupled binary overlays amplify prediction noise 10–20× | parity is asserted on the **`ActionIntent`/`QuoteSet` sequence**, not on PnL; the A/B harness runs paired same-path replay with overlays **disabled** |
| **P10** | **Which bracket?** The replay matcher is a *model* and produces two answers | the parity target names pessimistic or optimistic; the gate `sign_stable()` across the bracket is a contracted call (§2.6), not a convention |
| **P11** | **Restart parity** | snapshot at `t`, restore, replay forward, assert the intent sequence matches the uninterrupted run |
| **P12** | **Spec-boundary rule is too strict to use** (§2.1) | field-level dependency declaration instead of blanket refusal |

**Summary of the parity contract:** *given the same tape, the same `Environment`, the same
`SP-Params` resolved at the same `t`, the same restored warm state and the same seeds, the
stack emits an identical `ActionIntent` sequence* — plus P1's refusal rule, without which
parity is real and meaningless because both sides replay the same fabricated clock.

---

## 5. Over-contracting — where simple impls must fabricate

This is a real cost and v2 pays it in five places. In each case the fix is the same shape:
make the field optional and let the **consumer** declare what it REQUIRES — v2 already has
this pattern for venue capabilities (R-REQUIRES, wiring fails loud), so reuse it rather
than invent one.

| # | over-contracted | who must fabricate | fix |
|---|---|---|---|
| **O1** | **`BeliefProcess` in full** | `BookOnly` has no `d`, no `G`, no `path_law`, no `jump_tail` — the book *is* p̂. It must invent a link and a jump tail to satisfy the type. Worst case it returns `Φ` and a plausible tail, and those fabricated numbers then feed the sniping term. | fields optional / `Unavailable(NOT_IDENTIFIED)`; consumers declare `REQUIRES = {"jump_tail"}`; wiring fails loud, so Option B fails *at wiring* rather than at 3 a.m. |
| **O2** | **`constituents` mandatory** | `StreamModel` has one constituent — itself. `{self: p̂}` is noise. | optional, EV-scoped (§2.5) |
| **O3** | **`shadow_price` on every constraint** | HARD constraints (tick grid, phase, eligibility, halt) have no price. Forcing one yields `inf` (useless) or a plausible finite number (**worse than a bool** — the solver will trade against fiction) | HARD/SOFT taxonomy, §1(c). HARD returns `None` |
| **O4** | **`own_impact` from `rest()`** | Option B has no impact model and will return `0.0` — an `assumed` value entering the objective silently, which is the FATAL-1 shape | under R-NULL it must be a **declared pin** with a bias direction, not a float (§2.3) |
| **O5** | **`deficit_span(field, now)` for every field** | meaningless for locally-computed fields; will be stubbed to 0 and the stub will be believed | define only for fields with a declared **estimand interval** (the `Coverage` object, §1(a) L4) |

Borderline, called out so the loop can decide rather than drift: `provenance` on every raw
`Known`. It is cheap, but for a raw feed value it is always `measured`, so it will be a
hardcoded constant that then composes by `min` with real ones. The load-bearing provenance
is on **derived** values, **params**, and **field semantics** — the last being the one that
matters most and is still absent: if `last_trade_price.side` is the **maker's** rather than
the **taker's**, maker gross flips **+95 bps → −95 bps** and the programme is dead. Its
status is *circumstantial* (63.7 % of BUY prints at the ask). `DA-Normalize` owns the
`side × asset_id → direction` mapping and there is nowhere in v2 to record that the mapping
is `assumed`. **SHOULD-FIX B2-15** — provenance on derived-field semantics, not only on
scalars.

*(Nothing else in v2 reads as over-built. §9's refusal to build a venue abstraction layer
is correct and should be defended in the next iteration too.)*

---

## 6. Triage

### MUST-FIX (7)

| id | fix | § |
|---|---|---|
| **B2-1** | **`t_known_prov` + `t_known_err` on `Known[V]`, and an `EV-Replay` refusal rule keyed to them.** `t_known` is fabricated on all ~12 days of backfill and on all pre-08-07 history; the `t_known ≤ now` assertion then passes by construction and the 1.7 s look-ahead returns type-laundered. This is the same bug as v1's, one level deeper. | 1(a) L1 |
| **B2-2** | **`SelfState` carries `submitted / acked / fills / position` with separate knowledge times, plus `envelope()`.** Constraints read the pessimistic envelope; fill models read the venue-confirmed timeline. One stamp cannot express an unmeasured ack leg that errs in *both* directions. | 1(a) L2 |
| **B2-3** | **Composition rules inside the value type: `t_known` = MAX, staleness = MIN over `t_event`, deficit ≥ MAX, provenance = MIN over the lattice.** Plus typed `Unavailable` with a `cause` chain on every BE method. v2 states none of these; and the intuitive "MIN `t_known`" is the look-ahead, not the guard. | 1(a) L3, 2.5 |
| **B2-4** | **Make `VarianceBudget` falsifiable:** typed `Support{factor, span, op}` + a `space` unit + `EV-Calibration.variance_audit()` PIT gate. String supports let σ_⊥ and κ(r) register as disjoint, which is the exact 32–41 % double-count the registry was built to block. | 1(b) |
| **B2-5** | **Close FATAL-2's three escapes:** `ConstraintKind{HARD,SOFT}` (only HARD may return `max_size=0`, and it returns no price); shadow prices reach the solver **as an objective term**, not as a solver courtesy; KKT complementary-slackness audit on every replayed decision; and `evaluate()` gets its missing `view/self/port/horizon` arguments plus `binding`/`usage`. | 1(c) |
| **B2-6** | **`ObjectiveTermRegistry` with `covers` / `unit` / `basis` / `measure` and a `DoubleCount` on overlap.** R-ONCE is enforced for variance and not for PnL, and v2's own term list already double-counts `rewards` (term **and** priced constraint) and `adverse_selection` (a term beside a belief that returns `E[markout\|fill]`). | 2.0 |
| **B2-7** | **Scope the fill model to the MARKET, not the instrument.** The book is unified across the token pair — an Up bid at 0.60 crosses a Down bid at 0.40 via CTF mint, atomically at match time. `rest(level, …)` on one token structurally forbids the correct `p_fill` and mis-conditions `E[markout\|fill]`; implicit mint has no representation. Same failure shape as v1's missing `size`. | 2.6(5), 2.1 |

### SHOULD-FIX (9)

- **B2-8** `StateView` only to BE/DE (never a `DA-State` handle); truncation applies to history accessors, not just scalar getters; declared-input manifest + injection.
- **B2-9** `Coverage{target, covered, weight_missing, tail_deficit}` replaces scalar `deficit_span`; `DA-Normalize.gaps(span)` queryable (56 gaps > 5 s must be excludable from a variogram fit).
- **B2-10** `SP-*` fields carry a `source`; `discrepancies()` feeds `OP-Monitor` — Gamma served a stale rewards band after the 2026-08-20 re-cut and a single-value record cannot represent that.
- **B2-11** Spec functions stored as `(family, params)`, not closures — otherwise the "data" plane becomes code and R-VERSION cannot diff it.
- **B2-12** `EV-Replay` refuses on **field-level** spec dependency, not on any boundary; the current rule caps replay at ~12 days and, after 2026-08-20, under one.
- **B2-13** `NullDecl{pins, prov, bias}` replaces the free-text null assumption, so R-NULL composes with R-PROV; fix §0's own example, which declares one of `FlowAndFills`'s three implied pins.
- **B2-14** `SP-Params` keyed by `{venue, coin, horizon, epoch}` with units and uncertainty (the 120/471/1700 collision was a keying failure).
- **B2-15** Provenance on **derived-field semantics** (`Trade.side_signed`, dedup keys, the boundary reader `b(·)`), not only on scalars — the programme's central economic sign rests on one unrecorded field semantic (63.7 % of BTC `BUY` prints at the best ask vs 15.1 % at the best bid is *circumstantial*; if `side` is the maker's, maker gross flips **+95 → −95 bps**).
- **B2-16** `Environment` seam (`clock, feeds, venue, rng`) with `LiveEnv`/`ReplayEnv` — resolves §3 rows 2, 6 and 8 at once and is the precondition for the whole of §4.

### NOTED (7)

- `BE-Link` as its own module also fixes "nobody owns Φ" (28× tail underpricing, −3.5…−15 % MLE bias) — filed as §3 row 3 rather than separately.
- `tick(px)` and `price_bounds` into SP-Venue; `min_size_for_rewards` (50) ≠ `min_order_size` (5); reward weight `((1.5−s)/1.5)²` has no `|d|` dependence while toxicity is φ(d)-driven; rate limits measured non-binding (3,500 orders/10 s) but their **scope** is undeclared.
- The `Disputed`/`Unknown` spec states (§2.1) are the right home for the fact that `1.5 c` was read off markets that are not ours — do not let it enter as `inherited`.
- `SP-Instrument.overlap_with` — `X_0` of window k+1 **is** `X_T` of window k; every iid CI in the corpus is optimistic and nothing declares the coupling.
- Post-`T` phase contract: median trade T+21 s, resolution T+85 s, measured −1.46 c/share (−806 bps) on 0.9 % of notional; `WindowCtx.phase ∈ {pre, in, post_T, resolved}`.
- `parity: SumToOne | NegRisk(k) | None` replaces `complement: Some|None` (neg-risk is k-way).
- Dedup set on `transaction_hash` needs a TTL (unbounded) and a total order (replay determinism, P7).

---

## Appendix — is v2 better? Yes, and by how much

Scored the way the loop scores: modules touched under v2 **with the six MUST-FIXes above
applied**, for the changes that stressed v1 hardest.

| change | v1 | v2 as written | v2 + B2-1..6 |
|---|---|---|---|
| #9 rewards → constraint w/ shadow price → contest | **STRUCTURAL** (homeless) | 2 (Objective **and** Constraints — the double-count) | **1** (Constraints; term registry rejects the duplicate) |
| #11 latency 120 → 471 → 1,700 ms | 3 documents | 1 (OP-LatencyBudget) ✅ | 1 |
| #3 σ static → HAR → variogram | 2 | **1** ✅ (`w` split landed) | 1 |
| #5 v(t) sum → min | ships the bug | registers, **passes anyway** (string support) | **1**, and `register()` raises |
| #6 pull → participation region | 2 (C1+C2) | **1** ✅ (constraints de-economised) | 1 |
| #7 Q_max → loss cap + aggregate | unbounded | **no owner** (portfolio dropped) | 2 (Constraints + new DE-Portfolio) |
| #13 late components | new module each ✅ | new module each ✅ | ✅ |
| **knowledge time** | whole-stack | 1 type ✅ | 1 type — *provided* B2-1 lands, else 0 and wrong |

The decomposition is now good enough that the remaining findings are about **whether the
enforcement points actually fire**, not about where the boundaries sit. Three of them
(`register()`, `t_known ≤ now`, "never a bool") are rules that **pass while the underlying
error is present** — which is a more dangerous state than v1's, because v1's errors were
legible. That is the whole of this review.
