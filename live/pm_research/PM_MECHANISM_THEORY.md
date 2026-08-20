# PM_MECHANISM_THEORY — mechanism → SOTA theory map (P-2026-003, sketch-review iter 2, lens T)

Object: the mechanism inventory M-1…M-8 of `PM_MM_PLAN.md` §9. For each
mechanism: **(i)** the theory it selects, **(ii)** why that theory dominates the
obvious alternatives *for this mechanism*, **(iii)** the model object we
implement, **(iv)** the venue assumption the theory needs, stated as a
falsifiable proposition (`P-Mx*` — the design agent turns these into
experiments).

Scope per user 2026-08-20: **no PnL/capacity estimation**. Nothing below
forecasts earnings; the gates are mechanism-truth gates.

Builds on (not re-litigated): stream-anchored fair value (T-F12), discrete
per-level EV quoting (T-F8), `v(t)` min-structure (T-F9), the `(|d|, r)` pull
surface (T-F4), the on-chain fee ground truth and latency topology (M-lens
M1/M5).

Conventions: prices in $ of the $1 payout, so price ≡ probability numerically;
`q` = net inventory in shares; `τ = T − t`; `w` = TWAP window (60 s); `r = T − t`
inside the averaging window; `d` = standardised moneyness, `p̂ = Φ(d)`;
`v = p̂(1−p̂)`; `L` = our reaction latency (feed → decision → order ack);
`σ` = underlying vol in return units per √s.

---

## M-1 Matching: price-time priority, off-chain operator, on-chain settlement

### (i) Theory adopted

**Queue-position value** (Moallemi & Yuan) layered on a **queue-reactive order
book** (Huang, Lehalle & Rosenbaum) and framed by the **large-tick asset**
theory (Dayri & Rosenbaum).

- Moallemi & Yuan decompose the value of a resting order into a *static*
  component (spread earned minus adverse selection at the instant of trade,
  with AS **increasing in queue position** — a back-of-queue fill only happens
  when the whole queue is swept, which is the informed case) and a *dynamic*
  component (the option value of holding a locked-in position). Their headline
  empirical fact is the one that applies to us: **for large-tick assets queue
  value is of the same order of magnitude as the spread.**
- Huang–Lehalle–Rosenbaum supply the fill dynamics: the book is a Markov
  queueing system whose arrival/cancel intensities depend on the *state of the
  queues*, not just on distance from mid.
- Dayri–Rosenbaum give the regime label: when the effective spread is pinned at
  one tick, the economically meaningful spread is *implicit*, and quote-distance
  optimisation is the wrong parameterisation.

### (ii) Why it dominates the alternatives

The default (P-2026-002 inheritance) is the Avellaneda–Stoikov / GLFT
exponential fill intensity `λ(δ) = A·e^{−kδ}` — a function of distance-from-mid
only. That is a *small-tick* model: it presumes the maker chooses δ on a
continuum and that fills are Poisson in δ. Here the book is 2–4 ticks wide, δ
takes 2–4 values, and the marginal decision is not "how far from mid" but
"where in the FIFO queue at a price I have already chosen". **A λ(δ) model
cannot represent the decision at all** — it has no queue coordinate, which is
also the internal inconsistency T-F8 flagged between §2.4 and §3. Pro-rata
matching models (Guilbaud–Pham's pro-rata paper, Field/Large) are excluded by
the M-lens finding that matching is price-time.

### (iii) Model object

Level set per side per token: `ℓ ∈ {improve (1 tick inside best), join (at
best), behind (1 tick worse), out}`.

State added to §3: `Q_ahead` (shares resting ahead of us at our level),
`Q_behind`, book imbalance `I`.

Fill probability over decision horizon `h`, under strict price-time priority:

```
F_ℓ(h) = P( ∫_t^{t+h} dV^{aggr}_ℓ  +  ∫_t^{t+h} dC_ahead  >  Q_ahead )
```

with `V^{aggr}_ℓ` = aggressive volume consuming level ℓ and `C_ahead` =
cancellations ahead of us; both intensities estimated **conditional on book
state** (ticks-from-mid × queue size) per the queue-reactive model.

Per-level value (Moallemi–Yuan decomposition, in $ per share):

```
V(ℓ) =  F_ℓ · [ P_ℓ − CE_quote(q) − A(Q_ahead) + ρ(P_ℓ) ]   +   D_ℓ(Q_ahead)
         └──────────── static ────────────────────────┘        └─ dynamic ─┘
A(Q_ahead) = E[ p̂_fill − p̂_t | fill at queue position Q_ahead ]   (markout, position-dependent)
D_ℓ(Q)     = option value of the locked position if not filled this horizon
ρ(P)       = maker rebate, see M-4
CE_quote   = exact CARA indifference quote, see M-8
```

Decision: `ℓ* = argmax_ℓ V(ℓ)`, quote iff `V(ℓ*) > 0`.

**`D_ℓ` is the term §3 is missing, and it is exactly the join-vs-improve
term.** Improving resets `Q_ahead = 0` (max `D`) but costs a tick of edge and
takes the most-informed flow first; joining accepts `Q_ahead > 0` but earns the
tick. Since queue position is unobservable (no MBO feed), `Q_ahead` is
**bracketed**: pessimistic `Q_ahead` = full displayed size at ℓ at submission,
optimistic `Q_ahead = 0`. Note this re-interprets the plan's existing bracket:
**the bracket is on `Q_ahead`, not on `λ`.**

### (iv) Falsifiable venue assumptions

- **P-M1a (strict FIFO, no reordering).** After a trade of size `V` at price
  `P`, displayed size at `P` decrements by exactly `V` (net of concurrent
  add/cancel), and no order submitted later than ours ever fills ahead of us at
  the same price. Test: reconstruct the book from `book` + `price_change` +
  `trades`; look for decrements inconsistent with FIFO consumption.
- **P-M1b (continuous, not batched).** Match-event timestamps show no
  periodicity at any fixed grid between 1 ms and 1 s. A hidden batching
  interval would make PM a frequent batch auction and would qualitatively
  change M-7. Test: spectral / inter-arrival modulo test on `OrderFilled` block
  positions and WS trade timestamps.
- **P-M1c (queue value is first-order).** Measured fill markout is monotone
  increasing in estimated `Q_ahead`. If flat, `D ≈ 0`, queue value is
  negligible, and join-vs-improve reduces to a pure edge comparison — which
  would *simplify* §3 and is worth knowing.

---

## M-2 Order lifecycle: $0.01 tick, min size 5, cancel/replace latency, rate limits

### (i) Theory adopted

**Mixed regular/impulse control with a tick-valued spread** (Guilbaud & Pham,
*Quantitative Finance* 2013), extended with **latency-aware execution**
(Cartea, Jaimungal & Sánchez-Betancourt, *Latency and Liquidity Risk*; Gao &
Wang, *Optimal Market Making in the Presence of Latency*).

Guilbaud–Pham is the canonical large-tick MM control problem: the spread is a
tick-valued continuous-time Markov chain, the maker's control is
*which discrete level to occupy* (at best, or best ± 1 tick "for getting
execution order priority"), and requoting is an **impulse**, not a costless
adjustment.

### (ii) Why it dominates

Continuous-δ GLFT assumes an interior optimum in δ and free requoting. T-F8
already showed the interior optimum is fictitious here (`δ* ≈ 1/k ≈ 1 tick`,
fitted on 2–4 points). The deeper problem is the free-requote assumption: under
price-time priority **every cancel/replace destroys queue position**, so
requoting has a real, computable cost (`D_ℓ` from M-1) that a regular-control
model cannot see. Impulse control is the formalism that prices it. Latency-aware
formulations are needed because the impulse is executed at a state that has
already moved by `L`.

### (iii) Model object

Controls per (token, side): level `ℓ_t ∈ {−1, 0, +1, ∅}` on the grid
`P ∈ 0.01·ℤ` (state-dependent: the venue's `tick_size_change` to 0.001 near the
extremes — see P-M2c, this is **not** a detail, see M-7(iii)); size
`z_t ≥ z_min = 5` (and `≥ 50` if reward-qualifying).

Value function `V(t, q, m, book state, ℓ, Q_ahead)` solves an HJB
**quasi-variational inequality**

```
max{  ∂_t V + 𝓛^{ℓ} V ,   𝓜V − V  } = 0,
𝓜V(x) = sup_{ℓ'} V( x | level ℓ', Q_ahead ← back of new queue ),
terminal condition = the Bernoulli CE of M-8 (NOT a liquidation penalty)
```

**Implementable v1 (one line of code, and it is the whole point):** keep §3's
per-level argmax but add a **requote hysteresis band** — the discrete analogue
of the impulse-control continuation region:

```
requote from ℓ to ℓ'  iff   V(ℓ') − V(ℓ)  >  D_ℓ(Q_ahead)
```

i.e. never cancel unless the gain exceeds the queue value you destroy.

Consequence of the size floor: `z ≥ 5` (and `≥ 50` in-band) makes the control
set non-convex at the bottom — **the "quote a tiny size" defence does not
exist**. Combined with the fixed tick, level and participation are the only
defensive controls left (this feeds M-7).

### (iv) Falsifiable venue assumptions

- **P-M2a (latency budget).** Measured `order→ack` and `cancel→ack` RTT from
  our box has median ≤ X ms, p99 ≤ Y ms, with X, Y to be measured, not assumed.
  Until measured, `L` in every formula below is a free parameter and the
  participation frontier is unpinned.
- **P-M2b (no hidden lifecycle friction).** Cancels are free, there is no
  minimum resting time and no cancel-ratio penalty, and the documented rate
  limits (3,500 POST/10 s, 3,000 DELETE/10 s) are the real ones. A minimum
  resting time would add a lockout to the QVI and would invalidate any
  requote-based policy.
- **P-M2c (tick grid).** The tick switches 0.01 → 0.001 at documented price
  thresholds, and the reward scorer uses the same grid. **High priority**: the
  location of this switch is decisive for the M-7 participation frontier.

---

## M-3 CTF split/merge/redeem: Up + Down = $1

### (i) Theory adopted

**Multi-asset market making with a rank-deficient risk covariance** — Guéant
(*Optimal market making*, 2017) as the general framework, **Bergault & Guéant**
factor dimensionality reduction (*Math. Finance* 2021) and Bergault,
Evangelista, Guéant & Vieira (closed-form approximations) as the machinery —
with the *execution* side governed by **legging / execution risk in arbitrage**
(Kozhan & Tham, *Management Science* 2012).

The ETF creation-redemption analogue is the right economic intuition and its
empirical literature (Petajisto, *FAJ* 2017) is the right prior: with a *costly
or slow* primary market, secondary prices wander in a ~100–200 bps band. Our
primary market (CTF split/merge, atomic at match, gas < $0.01, redemption ≡
holding to T) is essentially **free and instantaneous**, so the band should be
≈ 0 and **the entire economics of the pair sits in legging, not in redemption.**

### (ii) Why it dominates

- *Two independent single-asset MM problems* (the naive route) is wrong: the
  risk covariance is exactly rank 1, `Σ = v·[[1,−1],[−1,1]]`. A 2-D inventory
  grid is an ill-conditioned parameterisation of a 1-D risk. Bergault–Guéant's
  factor reduction is the fix and here it is **exact, not approximate** — one
  factor.
- *Pure arbitrage* (buy the pair whenever `b_up + b_down < 1`) is also wrong:
  with two *maker* legs there is no simultaneity. The object is a two-leg
  acquisition with completion uncertainty — Kozhan–Tham's result (arbitrage is
  limited by completion risk, and the risk grows with the number of competitors
  chasing the same pair) is the correct frame and is directly measurable here
  as the pair-completion rate.
- The practitioner conversion/reversal (put-call parity) literature adds nothing
  formal beyond "quote the portfolio consistently"; it is not a source.

### (iii) Model object

Decompose inventory `(q_up, q_down)`:

```
m = min(q_up, q_down)      paired  → riskless, redeems for $m at T
q = q_up − q_down          net     → the ONLY risky coordinate
Var_t[terminal] = q² · p̂(1−p̂)        (m contributes exactly zero)
locked PnL per pair = 1 − (paid_up + paid_down)
```

CARA terminal wealth: `W_T = X_T + m + q·1{Up}` ⇒

```
CE = X_T + m − (1/γ)·ln( p̂·e^{−γq} + 1 − p̂ )
```

so the *reservation* sees only `q`, but the **EV of a fill must credit its
contribution to `m`**. Quote the four sides as **two instruments**:

- **Instrument A (directional):** priced off `p̂` with the M-8 CE reservation.
- **Instrument B (the pair):** a *joint* constraint, not two independent quotes:
  `b_up + b_down ≤ 1 − (target locked edge)`. Post both legs iff

```
P(complete) · [1 − b_up − b_down]   >   P(legged) · κ
κ = CE cost of carrying the naked leg to T  ( ≈ γ·v·|1| in CE units )
```

Note `κ` is *not* a chase cost: unwinding costs 3.5% taker (M-4), so the
realistic resolution of an unpaired leg is to hold it to resolution. The pair
is therefore a **bet on completion**, and Kozhan–Tham says completion
probability falls as competitors crowd the same pair.

### (iv) Falsifiable venue assumptions

- **P-M3a (unified book, atomic mint at match).** A marketable Up bid at `P`
  matches a resting Down bid at `1−P` via CTF mint in a single match event, with
  no extra fee or gas to either maker. Test: on-chain `OrderFilled` legs paired
  with a `PositionSplit` in one transaction, both maker legs at fee 0.
- **P-M3b (the no-arb band is ≈ 0).** `best_bid_up + best_bid_down` has
  negligible mass above 1 and `best_ask_up + best_ask_down` negligible mass
  below 1, net of taker fee. If true, the pair edge is only harvestable
  passively; if false, a taker-side conversion exists and the plan's priorities
  change.
- **P-M3c (completion is the binding cost — the real test).** Conditional on one
  intended pair leg filling, `P(second leg fills before T)` is materially < 1
  **and is negatively correlated with the informativeness of the first fill**
  (you complete when the market is quiet and get legged when it moves). This is
  the falsifiable version of "pair-harvest is nearly riskless". If it fails,
  §2.3's pair-harvest is a mirage.
- **P-M3d (redemption).** 1 Up + 1 Down held to resolution pays exactly $1, no
  fee, callable ~+85 s after window end.

---

## M-4 Fees: taker `∝ min(p, 1−p)`, maker rebate 20% of taker fee

### (i) Theory adopted

**Make/take fee non-neutrality under a binding tick** (Colliard & Foucault 2012;
Foucault, Kadan & Kandel, *JF* 2013) for why the split matters at all, and
**Glosten–Milgrom with a fee wedge** for what the fee does to the informed
trader's participation constraint (the "moat").

### (ii) Why it dominates

The textbook result is that only the **total** fee matters, because the maker
shifts the quoted price to neutralise any make/take split. That neutralisation
requires *continuous prices*. Colliard–Foucault show a binding minimum tick
breaks neutrality. Our tick is $0.01 on a ~$0.50 asset = **200 bps**, maximally
binding — so the rebate is a real transfer, not an accounting illusion, and it
must enter the objective rather than a reporting line.

Second, the fee's functional form here is not spread-like: `f(p) = 0.07·p(1−p)`
is proportional to the **Bernoulli variance**, the same object as our inventory
risk coefficient `v`. No generic fee model surfaces the consequence below.

### (iii) Model object

```
taker fee   f(p)  = 0.07 · p(1−p)        per share   (on-chain verified)
maker fee         = 0
maker rebate ρ(p) = 0.20 · f(p) = 0.014 · p(1−p)     per filled share
```

**1. Rebate-augmented level EV.** Every EV in §3 must read

```
V(ℓ) = F_ℓ · [ P_ℓ − CE_quote(q) − A_ℓ + ρ(P_ℓ) ] + D_ℓ
```

`ρ(0.50) = $0.0035/share = 0.35 tick`. On a 2–4-tick book that is **not** a
rounding term. §3 omits it entirely.

**2. The pickoff moat (Glosten–Milgrom + wedge).** A counterparty with private
value `p*` lifts our ask at `P` only if `p* − P − f(P) > 0`. So our ask is safe
against any information worth less than

```
m(ℓ) = (P_ℓ − p̂) + f(P_ℓ)          [$ of the $1 payout]
```

`m` is a **control** (choose the level), bounded above by the requirement that
`F_ℓ > 0`. Converting to underlying units with `∂p̂/∂F = φ(d)/σ_eff`, the moat in
bps of underlying is `B = m·σ_eff/φ(d)`. With the touch-joining case
`P_ℓ − p̂ = ½ tick = $0.005`, `σ_eff = 15.4 bps` (τ = 300 s, 50% annualised):

| `|d|` | `p̂` | `f(p)` | moat `m` | fee share of moat | `m/φ(d)` | moat in bps |
|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.01750 | 0.02250 | **78%** | 0.056 | 0.87 |
| 1 | 0.841 | 0.00934 | 0.01434 | 65% | 0.059 | 0.91 |
| 2 | 0.977 | 0.00156 | 0.00656 | 24% | 0.122 | 1.87 |
| 3 | 0.9987 | 0.00009 | 0.00509 | **2%** | 1.149 | 17.7 |

**Finding that corrects §1 / M-lens M1: the far-from-money moat is the TICK,
not the fee.** The fee is 78% of the moat ATM — where the moat in underlying
terms is smallest (0.87 bps) and therefore worth least — and collapses to 2% at
`|d| = 3`, where the moat is finally large. The fee's `p(1−p)` shape makes it
vanish exactly where protection becomes valuable. Corollary (see P-M2c): **if
the tick refines to 0.001 above `p ≈ 0.95`, the far-from-money moat drops ~10×
and most of the safe region disappears.** This is the single most
consequential unverified parameter in the venue.

**3. Rebate-vs-toxicity structure.** Rebate `ρ = 0.014·v`; pickoff exposure per
requote interval (derived in M-7) `= φ(d)·√(3L/r)` in-window. Their ratio
`∝ (v/φ(d))·√(r/L)`. Since `v/φ(d)` falls only slowly in `|d|`
(0.63 → 0.55 → 0.41 → 0.30 at `|d|` = 0, 1, 2, 3) while `√r` falls fast,
**the rebate/toxicity ratio is roughly flat in `|d|` and scales as `√r`** —
early-window fills are structurally better-compensated than late-window fills
at any moneyness. This is consistent with the M-lens census (largest notional
bucket is 0–60 s) and is a testable prescription.

### (iv) Falsifiable venue assumptions

- **P-M4a (fee law).** On-chain taker fee `= C·0.07·p_fill·(1−p_fill)` computed
  at the **fill** price; maker legs pay 0. Verified on 2 trades; extend to a
  census across price buckets and coins.
- **P-M4b (the rebate is real and reaches a maker).** Daily USDC rebate payout
  `= 0.20 × Σ_fills C·0.07·p(1−p)`, subject to the $1 minimum, with no further
  eligibility condition. **Currently UNVERIFIED and worth 0.35 tick ATM — i.e.
  comparable in importance to the rewards band.** Test: pick a large maker
  wallet on 5-min crypto markets, sum predicted rebate over a UTC day from
  on-chain fills, match against the next day's inbound USDC transfer.
- **P-M4c (no fee tiers).** The effective rate `fee/(C·p(1−p))` is 0.07 for all
  wallets independent of volume. Test: cross-sectional regression on wallet
  volume; slope must be 0.

---

## M-5 Rewards band = a quoting obligation (params UNVERIFIED)

> The most under-theorized mechanism in the plan, and the one whose correct
> treatment changes the **objective**, not just a parameter.

### (i) Theory adopted — three layers

**Layer 1 — what the scheme *is*: principal–agent contract theory for MM
incentives.** El Euch, Mastrolia, Rosenbaum & Touzi (*Math. Finance* 2021)
solve exactly this: an exchange designs a compensation scheme `ξ` contingent on
observable market-making behaviour, and the MM best-responds; the optimal
contract and the induced optimal quotes are obtained in quasi-explicit form.
Baldacci, Possamaï & Rosenbaum (*SIAM J. Financial Math.* 2021) extend it to
**several MMs playing a Nash equilibrium against one contract**, and find that
more market makers is not necessarily better under a contracting scheme — the
direct analogue of band crowding. Aïd, Bergault & Rosenbaum (2025) add the
competitive-spillover / free-riding structure.

**Layer 2 — our side of it: constrained stochastic control.** The band is a
**hard constraint on the admissible control set** (`s ≤ s_max`, `z ≥ z_min`,
two-sided mandatory when mid ∉ [0.10, 0.90]) plus an **integral (occupancy)
reward**. Base model GLFT / Guéant with the control set restricted; the
economics live in the **KKT multiplier (shadow price) of the constraint**.

**Layer 3 — the dissipation: proportional-share (Tullock) contest.** The payout
is literally `R·x_i/Σx_j`, so the contest formulation is exact, not analogical.

**Priors (not model objects): the DMM empirical literature.** Bessembinder, Hao
& Zheng (*JF* 2015; *RFS* 2020 NYSE discontinuity) — DMM contracts exist to fix
a coordination failure, and contractual features that raise DMM participation
raise depth and narrow spreads. Anand & Venkataraman (*JFE* 2016) — market
makers **withdraw in unison** when conditions are unfavourable, and the DMM's
economic function is precisely to be the one who cannot. Read across: our
obligation will bind exactly when we most want to leave.

### (ii) Why it dominates

- **Unconstrained GLFT is wrong because the constraint binds.** The qualifying
  band (1.5 c per the CLOB registry) sits strictly *inside* the observed 2–4 c
  market, so the reward-maximising quote is strictly tighter than the
  risk-optimal quote. The book confirms the field agrees: BTC depth within 1 c
  of mid ≈ 138 shares/side vs 1,290 within 4.5 c — depth deliberately sits
  outside the scoring band.
- **A "separate PnL line" is not a model.** It presumes separability; the
  constraint makes rewards and adverse selection the *same decision variable*.
  (This is the M-lens M1 finding, now with a formal reason.)
- **The DMM empirical literature cannot be the model object** — it is
  IO/event-study work with no quote rule. Use it for hypotheses.
- Tullock beats a generic Cournot/Bertrand story because the payout rule *is*
  proportional-share.

### (iii) Model object

Documented scoring, per random per-minute sample:

```
S(z, s) = ((s_max − s)/s_max)² · z ,   s = distance from mid (cents), z ≥ z_min
Q_min   = min(S_bid, S_ask);  single-sided credited at Q/c, c = 3
two-sided MANDATORY when mid ∉ [0.10, 0.90]
payout over epoch = R · Σ_samples x_i / Σ_samples X_i ,  X = total qualifying score
```

Constrained control problem:

```
max  E[ Σ_fills (edge − AS + rebate) ]  +  E[ R · ∫ (x_t / X_t) dN^{sample}_t ]  −  γ-risk
s.t. x_t = S(z_t, s_t)·1{eligible},  s_t ≤ s_max,  z_t ≥ z_min,
     two-sided when p̂ ∉ [0.10, 0.90]
```

First-order condition on qualifying score (KKT, shadow price
`ψ = ∂V/∂x`):

```
occupy the band  iff   R·(X − x)/X²  ≥  c(|d|, r)
small maker (x ≪ X):   ****  occupy iff  R / X  ≥  c(|d|, r)  ****
c(|d|, r) = expected adverse-selection + inventory cost per unit qualifying score
```

**Both sides are estimable from data we already collect.** `R` from the
registry (`rate_per_day` × window share, ≈ $34.7/BTC window subject to M9).
`X` **directly from the L1 book** — compute `S(z, s)` over all displayed size
within `s_max`. `c(|d|, r)` from PM-E2.5 shadow-quote markouts.

Because `R/X` is **`|d|`-independent** and `c(|d|, r)` is `φ(d)`-driven, the
rule defines a **band-occupancy frontier in `(|d|, r)`** — the M-lens M7 item,
now *derived* rather than asserted, and the primary object of PM-E3.

**Equilibrium sanity check (Tullock, `n` symmetric makers, unit cost `c`):**

```
x* = R(n−1)/(c n²),   X* = R(n−1)/(c n)   ⇒   R/X* = c·n/(n−1) → c,   rent per maker → R/n → 0
```

Reading: **in equilibrium `R/X ≈ c`, so the frontier is exactly the break-even
set.** Positive expected value can only come from a *differentially lower* `c`
(better `p̂`, lower latency, or a state where our `c` beats the marginal
maker's) — never from the band itself. This is the honest theoretical statement
of what the subsidy can pay for, and it determines what G3a/G3b are actually
testing.

**The forced far side.** Two-sidedness is a *hard constraint* outside
[0.10, 0.90] — precisely the decided-window regime where the M-lens says the
band is cheap. So the obligation forces us to post the **toxic** side (the side
that is picked off if the window un-decides). Its value is negative and must be
charged against the reward. §3 has no such term.

### (iv) Falsifiable venue assumptions — highest-value experiments in the program

- **P-M5a (the scheme applies at all).** 5-min crypto markets participate in a
  rewards program with the documented scoring. *Current evidence is against*:
  exhaustive pagination of `GET /rewards/markets/current` (33 pages / 16,172
  rows) does not contain our `condition_id`s, and `GET /rewards/markets/<cid>`
  returns empty. Decisive test: an observed 00:00 UTC on-chain USDC payout to a
  wallet whose only qualifying activity is 5-min crypto markets.
- **P-M5b (band params).** `s_max`, `z_min`, `rate_per_day` for OUR markets from
  a source the scorer actually uses. Gamma says 4.5 c, the CLOB registry says
  1.5 c for *other* markets; both are currently unusable for ours.
- **P-M5c (`X` is observable).** Total qualifying score computed from the public
  book equals the scorer's `X` — no hidden/undisplayed qualifying liquidity. If
  false, `R/X` is unknowable and the occupancy rule is unimplementable.
- **P-M5d (sampling).** ~5 per-minute samples per 5-min window ⇒ per-window
  reward is a lottery with CV ≳ 0.45. The reward must be modelled as a random
  variable, and the *objective* needs a per-second unit to combine an occupancy
  integral with per-fill EV.
- **P-M5e (the constraint binds).** The risk-optimal (no-rewards) quote distance
  `s*` is strictly greater than `s_max`. Measurable from PM-E2.5 shadow quotes.
  **If `s* ≤ s_max` the obligation does not bind and this entire apparatus
  collapses to unconstrained GLFT plus a free subsidy** — a cheap, decisive
  first test.

---

## M-6 Settlement: Chainlink 60 s TWAP, Up iff `X_T ≥ X_0`, `r³` lock-in, tie → Up

### (i) Theory adopted

- **Pricing: Asian digital, but solved exactly, not with Asian machinery.** The
  payoff is a digital on an arithmetic average. The general arithmetic-Asian
  problem has no closed form (hence Rogers–Shi / Vecer / moment-matching), but
  over a 60 s window with a diffusive underlying the average is Gaussian to
  `O(σ²w)` and the **exact conditional moments are already derived and
  MC-verified** in §2 / T-F1–F3. So the Asian literature's role is only to
  certify that the Gaussian/moment-matched form is the standard and that no
  better closed form exists.
- **The settlement statistic as a designed benchmark:** Duffie & Dworczak,
  *Robust Benchmark Design* (*JFE* 2021). Their result — under strategic trade
  splitting the best linear unbiased fixing is **volume**-weighted — implies a
  *time*-weighted fixing is **not** the robust optimum. The venue's
  snapshot → 60 s TWAP change reduces manipulability roughly with window length
  but does not reach the robust frontier.
- **Manipulation:** Kumar & Seppi, *Futures Manipulation with "Cash Settlement"*
  (*JF* 1992) — the exact structure: a cash-settled contract whose settlement is
  a spot statistic invites position-then-push; profits fall toward zero as
  manipulators multiply, but **liquidity damage persists even in the limit**.
  On-chain counterpart: Mackinga, Nadahalli & Wattenhofer, *TWAP Oracle Attacks:
  Easier Done than Said?* (IEEE ICBC 2022) — arithmetic-mean TWAP manipulation
  costs materially **less** than the naive linear-in-window bound.

### (ii) Why it dominates — and one explicit REJECTION

- A plain digital at `T` with a fudged `σ_eff` gets the in-window ATM
  sensitivity wrong by the `√3` factor (T-F4) and has no formula for
  strike formation before `t = 0` (T-F6a).
- **REJECTED: options pinning (Avellaneda & Lipkin 2003).** The `r³` behaviour
  in §2 is **variance lock-in of the settlement statistic** — a measurability
  effect: the averaging window mechanically freezes `X_T`. Avellaneda–Lipkin
  pinning is a **price-impact feedback** effect: aggregate delta-hedging by
  option market makers drags the *underlying* to the strike, and their model's
  singular drift is parameterised by open interest and a price-elasticity
  constant. These are different mechanisms with different signatures. The A-L
  mechanism requires our (and our competitors') hedging flow to move the
  settlement underlying — nil for BTC against a multi-venue Chainlink aggregate
  at prediction-market size. **Importing A-L would license quoting near the
  money late in the window, which is exactly the error T-F4 caught.** The
  correct model of any *endogenous* underlying motion here is Kumar–Seppi
  manipulation, which predicts pushing toward *or away from* the strike
  depending on the manipulator's binary position — not pinning.

### (iii) Model object

```
p̂_t = G(d_t),   d_t = ( E_t[X_T] − K ) / σ_eff(t)
G = Φ in v1;  heavy-tailed link (Student-t_ν or empirical quantile map) as the
   PRE-COMMITTED variant (T-F14), not a post-hoc patch
E_t[X_T], σ_eff  from the §2 laws with the stream-anchored construction (T-F12)
```

Two objects §2 lacks:

1. **Boundary-reader operator `b(·)`.** `K = X_0` and `X_T` are *reports*, not
   continuous readings. Define `b` = (last report with ts ≤ boundary | first
   with ts ≥ boundary) × (`observationsTimestamp` | `validFrom`). Since `X`
   drifts at `(S_t − S_{t−w})/w`, a Δ-second ambiguity moves the read by
   `≈ σ√w·Δ/w` (~0.7 bps at Δ = 8 s — T-F6b, the same order as the entire basis
   noise). **`b` is a parameter to be estimated**, chosen to maximise agreement
   with the official winner.
2. **Manipulation screen (Kumar–Seppi / TWAP-attack cost).** Per window,
   compare `Π_push` = (binary notional that flips) × (payoff/share) against
   `C_push` ≈ cost of moving the 60 s Chainlink aggregate by `|X_T − K|`. Flag
   windows where `Π_push > C_push`: a *risk filter* (do not quote) **and** a
   toxicity feature.

### (iv) Falsifiable venue assumptions

- **P-M6a (settlement identity).** The official CLOB winner equals
  `1{b(X_T) ≥ b(X_0)}` computed from our recorded `crypto_prices_twap_sixty`
  stream for ≥ 1 − ε of windows, under exactly **one** convention `b`. Residual
  disagreements must concentrate in a tie zone `|X_T − K| < δ_tie`;
  **`δ_tie` must be measured — it is the width of the unquotable region.**
- **P-M6b (variance law).** Standardised residuals of `X_T − E_t[X_T]` are
  Gaussian with variance `σ²r³/(3w²)` in-window and `σ²(τ − 2w/3)` pre-window.
  Falsifiers: fatter-than-Gaussian residuals (⇒ trigger the T-F14 link) or a
  variance that does not scale as `r³` (⇒ the whole endgame model is wrong).
- **P-M6c (no residual pinning / manipulation).** `(X_T − K)/σ_eff` has no
  excess mass near 0 and no dependence on binary open interest. Excess mass near
  0 ⇒ a pinning-like mechanism exists after all and A-L must be revisited;
  open-interest dependence ⇒ Kumar–Seppi manipulation is live.
- **P-M6d (the strike is knowable at `t = 0`).** `X_0` read from our stream at
  window open matches the venue's `K` within the stream's ~471 ms latency. If
  `K` is only knowable retrospectively, **no quoting is possible at window
  open** and the largest-volume bucket (0–60 s) is off-limits.

---

## M-7 Latency topology: PM London / Binance Tokyo / box us-east-1

### (i) Theory adopted

**Budish, Cramton & Shim** (*QJE* 2015) for the structural claim: continuous-time
*serial* matching mechanically creates sniping rents on symmetrically-observed
public information, which liquidity providers pay regardless of their
cleverness. **Aquilina, Budish & O'Neill** (*QJE* 2022) for the anatomy: races
are frequent (~1/min/symbol), extremely fast (modal race 5–10 µs), and latency
arbitrage accounts for ~33% of the effective spread. **Menkveld & Zoican**
(*RFS* 2017) for the comparative static that matters to us: faster
infrastructure helps snipers relative to slow makers, and the net spread effect
depends on the news-to-liquidity-trader ratio. **Foucault, Kozhan & Tham**
(*RFS* 2017, *Toxic Arbitrage*) for the measurable object: the fraction of
toxic-arbitrage opportunities that terminate with an arbitrageur's *trade*
rather than a *quote update* — literally "did the maker cancel in time" — with a
1% increase in that fraction associated with a 4% spread increase. Control-theory
side: Cartea–Jaimungal–Sánchez-Betancourt (latency, order failure) and Gao–Wang
(latency degrades MM performance for large-tick assets).

### (ii) Why it dominates — and what it invalidates

The alternative is §3's current treatment: a markout-fitted `δ_tox` plus
"pull-on-burst", described as "the single most important control". **That is
theoretically wrong for a maker who loses the race.** BCS's core insight is that
**cancellation is itself a race**: when public information arrives, our cancel
and the sniper's take are competing messages into the same serial matching
engine, and the slower one loses. Our budget — Binance (Tokyo) → us-east-1 →
PM (London), ~120–250 ms — versus a co-located sniper's milliseconds, means we
lose *every race we enter*. Pull-on-burst therefore cannot be modelled as a
defensive control; it only works against information that arrives *slowly*
(bursts lasting ≫ our RTT). What remains are the controls BCS / Menkveld–Zoican
identify: **width, size, and participation** — plus the discrete-tick moat of
M-4.

### (iii) Model object

**Per-level sniping cost (this is the correct `ζ`, replacing the fitted
constant):**

```
ζ_snipe(ℓ) = (λ_J / λ_total) · E[ ( |J| − m(ℓ) )^+ ]
m(ℓ) = (P_ℓ − p̂) + f(P_ℓ)                  sniper's moat  (M-4; the fee PROTECTS us)
J    = ∂p̂/∂F · ΔF_L                        p̂-jump over our reaction latency L
λ_J  = intensity of underlying moves large enough to clear m
```

§3 charges the taker fee as a cost but **never credits it as protection** — it
belongs inside `m`.

**Pickoff exposure is volatility-free.** With `∂d/∂F = √3/(σ√r)` in-window and
`ΔF_L ~ N(0, σ²L)`:

```
in-window :  std(Δp̂ over L) = φ(d) · √(3L / r)          ← σ cancels
pre-window:  std(Δp̂ over L) = φ(d) · √(L / (τ − 2w/3))  ← σ cancels
```

This is a genuinely useful structural result: **the participation frontier needs
no volatility estimate.** (Caveat: `σ` cancels only if the same `σ` governs both
the short horizon `L` and the remaining window. A vol burst breaks the
cancellation — which **rehabilitates the burst flag as a model-validity monitor
rather than as a defensive control**.)

**Quotable frontier (the closed form §3 asked for at T-F4, now with the moat
included):**

```
quote at level ℓ  iff   m(ℓ) / φ(d)   ≥   k · √( 3L / r )        (in-window)
                        m(ℓ) / φ(d)   ≥   k · √( L / (τ − 2w/3) ) (pre-window)
k = required moat-to-move ratio (k = 3 below)
```

With `L = 200 ms`, `k = 3`, and `m` = half-tick + fee at the $0.01 tick:

| regime | required `m/φ` | `|d|*` | `p̂*` |
|---|---|---|---|
| window open (τ = 300 s) | 0.083 | ≈ 1.7 | 0.955 |
| mid-window (τ = 150 s) | 0.128 | ≈ 2.0 | 0.977 |
| TWAP entry (r = 60 s) | 0.300 | ≈ 2.5 | 0.994 |
| r = 30 s | 0.424 | ≈ 2.6 | 0.995 |
| r = 10 s | 0.735 | ≈ 2.8 | 0.997 |
| r = 2 s | 1.643 | ≈ 3.1 | 0.999 |

Two readings, both load-bearing:

1. **ATM quoting is a shooting gallery at our latency.** At `d = 0`, `r = 10 s`
   the moat is `≈ 0.09 bps` of underlying versus a typical 200 ms move of
   `≈ 0.32 bps` — the moat is *one third* of a routine move. T-F4's warning,
   restated in economic units.
2. **The safe region depends critically on the tick (P-M2c).** The table assumes
   a $0.01 tick throughout. If the tick refines to $0.001 above `p ≈ 0.95`, the
   half-tick term drops 10× exactly in the region the frontier selects, `m/φ`
   falls by 3–9×, and `|d|*` moves out past 3.2 — i.e. **the quotable region
   could be essentially empty.** Verifying the tick-change threshold is
   therefore a first-order economic experiment, not housekeeping.

**Controls the theory endorses that §3 lacks:**

- **Size as the primary risk knob.** We cannot cancel in time, so the loss per
  race is `size × (|J| − m)^+`; `z` is the only continuous control we own
  (floored at 5, and at 50 if reward-qualifying — which is exactly why the M-5
  obligation is expensive: it forces the size knob *up* where the moat is
  thinnest).
- **Stale-feed pull is the one valid "pull".** When *our own* feed goes dark
  (stream staleness, T-F12), no race has started and cancelling is not a race —
  it is the only trigger that should keep the "pull" label.

### (iv) Falsifiable venue assumptions

- **P-M7a (we lose the race).** The lead-lag from a Binance mid jump to the first
  PM book/trade reaction has a mode well below our measured order RTT. Test:
  ms-resolution cross-correlation of 100 ms Binance mid changes against PM
  `price_change`/trade timestamps, versus our own measured `order→ack`. If PM's
  reaction is *slower* than our RTT, we are not the slowest participant and the
  frontier loosens materially.
- **P-M7b (sniping is present and priceable).** Fill markouts at h = 1, 5, 30 s,
  stratified by whether a Binance move exceeding the moat occurred within `L`
  before the fill, are significantly more negative in the "preceded" stratum.
  This is Foucault–Kozhan–Tham's toxic-arbitrage measurement adapted to our
  data, and it is the direct estimator of `ζ_snipe`.
- **P-M7c (no venue-side protection).** No speed bump, no last look, no minimum
  resting time, no randomised batching. (If P-M1b finds batching, the entire
  M-7 analysis flips in our favour.)
- **P-M7d (the WS tail is ours, not theirs) — RUN THIS FIRST.** The measured
  p99 = 8,971 ms CLOB WS delivery lag is a collector artefact, not venue egress.
  If it is real, our effective `L` is **seconds**, `√(3L/r)` grows ~5×, and no
  quoting is defensible at any `|d|` we can reach. **A single cheap test that
  can invalidate the whole maker thesis.**

---

## M-8 Expiry: inventory self-liquidates into a Bernoulli payoff

### (i) Theory adopted

**Exponential (CARA) utility-indifference pricing of an unhedgeable binary
claim** (Hodges & Neuberger 1989; Henderson & Hobson survey 2009) as the
terminal condition, embedded in a **finite-horizon Avellaneda–Stoikov /
Guéant (2017)** market-making problem. **Feil & Nendel (2026), *Optimal Market
Making in Prediction Markets***, is the closest published formulation — an HJB
for a MM whose price is a conditional outcome probability driven by a
transformed latent-belief diffusion, carrying **both** mark-to-market inventory
risk **and the settlement risk of positions remaining at resolution**, with
existence/uniqueness of the optimal bid and ask. Adopt it as the reference
formulation for §3's terminal term.

### (ii) Why it dominates

A-S and GLFT terminate with either mark-to-mid liquidation (`−q S_T`) or a
quadratic liquidation penalty (`−α q²`). Both encode *"you must unwind at the
end and unwinding costs money"*. None of that holds here:

- **there is no unwind** — the position converts to cash at $0 or $1
  automatically (M-lens M5: holding Up+Down to resolution ≡ merging);
- **unwinding early costs 3.5% taker**, so early unwinding is dominated in
  almost every state;
- **the payoff is bounded**, so the risk of a large inventory is bounded. A
  quadratic penalty is qualitatively wrong: it grows without bound in `q` while
  the true CE is asymptotically **linear** in `q`.

### (iii) Model object

```
terminal utility:  U(W_T) = −exp( −γ ( X_T + m + q·1{Up} ) )
CE(q)  = −(1/γ)·ln( p̂·e^{−γq} + 1 − p̂ )  =: −(1/γ)·ln g(q)
ask a(q) = (1/γ)·ln( g(q−1) / g(q) )        bid b(q) = (1/γ)·ln( g(q) / g(q+1) )
```

Three prescriptions that follow, all absent from §3:

1. **State-dependent inventory limits.** Local risk aversion is `γ·v` with
   `v = p̂(1−p̂)`, so a constant `|q| ≤ Q_max` is the wrong constraint. The right
   one is a constant risk budget:

   ```
   γ·|q|·p̂(1−p̂) ≤ κ    ⇒    Q_max(p̂) = κ / ( γ · p̂(1−p̂) )
   ```

   The limit widens ~4× at `p̂ = 0.9` versus ATM and diverges at the extremes —
   which is what makes M-5's decided-window band occupancy feasible at all. A
   flat `Q_max` is simultaneously too tight in decided windows and too loose ATM.

2. **Asymptotic linearity and the longshot asymmetry.** As `γq → ±∞`, `CE(q)`
   saturates and the exact quotes stay in [0, 1] automatically, whereas the
   linear `r = p̂ − γqv` can exit [0, 1] and understates the short-longshot
   reservation by ~2 ticks (T-F7). T-F14's Gaussian-tail error is on the **same
   side of the same trade**. With no liquidation leg there is no later
   opportunity to correct the mis-price — so the exact `g`-ratio is
   **structural**, not cosmetic.

3. **No terminal liquidation ⇒ the running problem is pure acquisition.** The
   control is "which fills to accept", not "how to end flat". The QVI's
   liquidity-taking impulse is essentially never optimal at 3.5%, so the model
   can be solved as a **make-only** control, retaining the taker leg only as an
   emergency at `|q| > Q_max(p̂)` and pricing it at the full 3.5%.

### (iv) Falsifiable venue assumptions

- **P-M8a (resolution is certain, complete and bounded).** Every window resolves
  on-chain within a bounded time (~+85 s observed) and a winning share redeems
  for exactly $1 with no fee or haircut. A failed/disputed/delayed resolution
  turns "self-liquidating" into a credit-and-settlement-risk position and
  invalidates the terminal condition.
- **P-M8b (no forced early exit).** No margin, no liquidation, no funding on a
  held position; inventory can always be held to `T`. (Believed true — positions
  are fully collateralised CTF tokens — but never verified.)
- **P-M8c (the CE is the right risk measure).** Per-window PnL conditional on
  end-of-window `|q|` matches the Bernoulli prediction `q²·p̂(1−p̂)` and not a
  diffusion-based prediction. Checkable on replayed inventory paths.
- **P-M8d (early unwind is genuinely dominated).** Median cost of flattening at
  `t < T` (taker fee + spread) exceeds the CE benefit of flattening for all
  `|q| < Q_max(p̂)`. Measurable from the recorded book.

---

## Summary table

| # | Mechanism | Adopted theory | Model object | Falsifiable venue assumption (headline) |
|---|---|---|---|---|
| M-1 | Price-time matching, off-chain operator | Queue-position value (Moallemi–Yuan) on queue-reactive dynamics (Huang–Lehalle–Rosenbaum), large-tick regime (Dayri–Rosenbaum) | `V(ℓ) = F_ℓ·[P_ℓ − CE − A(Q_ahead) + ρ] + D_ℓ(Q_ahead)`; bracket on `Q_ahead`, not on `λ` | **P-M1a** strict FIFO, no reordering; **P-M1b** no hidden batching; **P-M1c** markout increasing in `Q_ahead` |
| M-2 | Tick $0.01, min size 5, cancel/replace latency | Mixed regular/impulse control on a tick-valued spread (Guilbaud–Pham) + latency-aware execution (Cartea et al.; Gao–Wang) | HJB-QVI; v1 = **requote hysteresis**: requote iff `ΔV > D_ℓ(Q_ahead)` | **P-M2a** measured order/cancel RTT; **P-M2b** no min resting time or cancel penalty; **P-M2c** tick-change threshold (decisive for M-7) |
| M-3 | CTF split/merge, Up+Down = $1 | Multi-asset MM with rank-1 covariance (Guéant; Bergault–Guéant factor reduction) + legging risk (Kozhan–Tham); ETF creation-redemption as economic analogue (Petajisto) | State `(m, q)` = (paired, net); `m` is riskless; pair quoted as a **joint** constraint `b_up + b_down ≤ 1 − target`, gated on `P(complete)` vs legging cost | **P-M3c** pair completion probability < 1 and *negatively* correlated with first-fill informativeness (kills "riskless" framing if it fails) |
| M-4 | Taker fee `0.07·p(1−p)`, maker rebate 20% | Make/take non-neutrality under a binding tick (Colliard–Foucault; Foucault–Kadan–Kandel) + Glosten–Milgrom with a fee wedge | `ρ(p) = 0.014·p(1−p)` inside every level EV (0.35 tick ATM); moat `m(ℓ) = (P_ℓ − p̂) + f(P_ℓ)`; **far-from-money moat is the TICK, not the fee** | **P-M4b** the 20% rebate is actually paid to a maker on these markets (**unverified, worth 0.35 tick**) |
| M-5 | Rewards band = quoting obligation | Principal–agent MM contracts (El Euch–Mastrolia–Rosenbaum–Touzi; Baldacci–Possamaï–Rosenbaum) + constrained stochastic control (KKT shadow price) + Tullock proportional contest; DMM empirics as priors | **occupy iff `R/X ≥ c(|d|, r)`** (`X` computable from the book) ⇒ a band-occupancy frontier in `(|d|, r)`; equilibrium `R/X → c` ⇒ the band pays only for a differentially-lower `c`; forced two-sidedness outside [0.10, 0.90] carries a **negative** far-side value | **P-M5a** the scheme applies to these markets at all (current evidence: **absent from the registry**); **P-M5e** `s* > s_max`, i.e. the constraint binds |
| M-6 | Chainlink 60 s TWAP, `X_T ≥ X_0`, `r³` lock-in | Asian digital solved exactly via the §2 moments; benchmark design (Duffie–Dworczak: TWAP is not the robust optimum); manipulation = Kumar–Seppi + TWAP-oracle attack cost. **A-L options pinning explicitly REJECTED** (variance lock-in ≠ hedging feedback) | `p̂ = G(d)` with pre-committed heavy-tailed `G` variant; **boundary-reader `b(·)` as an estimated parameter**; tie-zone width `δ_tie` = unquotable region; manipulation screen `Π_push` vs `C_push` | **P-M6a** reconstructed winner matches the official winner under exactly one boundary convention; **P-M6c** no excess mass at `X_T ≈ K` and no open-interest dependence |
| M-7 | Latency topology; 1 bp BTC ≈ 3 ticks ATM | Sniping in continuous serial markets (Budish–Cramton–Shim; Aquilina–Budish–O'Neill), speed and adverse selection (Menkveld–Zoican), toxic arbitrage measurement (Foucault–Kozhan–Tham) | `ζ_snipe(ℓ) = (λ_J/λ)·E[(|J| − m(ℓ))^+]`; **volatility-free pickoff exposure `φ(d)·√(3L/r)`**; quotable frontier `m/φ(d) ≥ k√(3L/r)`; controls = width, **size**, participation — *not* cancellation | **P-M7d** the p99 = 8,971 ms WS lag is a collector artefact (**run first — can kill the thesis**); **P-M7a** we are strictly slower than the marginal sniper |
| M-8 | Expiry into a Bernoulli payoff, no liquidation leg | CARA utility-indifference pricing of an unhedgeable binary claim (Hodges–Neuberger; Henderson–Hobson) in finite-horizon A-S/Guéant; Feil–Nendel (2026) as the reference prediction-market formulation | Exact `a(q), b(q)` from `g`-ratios; **`Q_max(p̂) = κ/(γ·p̂(1−p̂))`** replaces a flat limit; make-only control (taker leg is emergency-only) | **P-M8a** every window resolves and redeems $1 with no haircut; **P-M8b** no margin/liquidation/funding on held inventory |

---

## Where §3 is theoretically WRONG or under-specified

Ranked by how much the mechanism changes the model.

**1. Rewards are treated as an outside option and a separate PnL line; they are
a CONSTRAINT with a shadow price.** (M-5)
§3 writes `ℓ* = argmax EV if max EV > outside option (0, or the rewards-band
value)` and "rewards booked as a SEPARATE PnL line". Both are structurally
wrong. The band is a hard constraint on `(s, z, two-sidedness)` and the payout
is a proportional-share contest whose marginal value is `R/X` — endogenous,
competitive, and **computable from the book we already collect**. Replace with
the constrained control and the occupancy rule `occupy iff R/X ≥ c(|d|, r)`; add
the mandatory-two-sided constraint outside [0.10, 0.90] with an explicit
negative value for the forced far side; and record the equilibrium implication
`R/X → c`, which is what G3a/G3b are really testing. Also: §3's objective mixes
a per-fill argmax with an occupancy *integral* sampled at random times — the two
have no common unit until the M-5 sampling model is written down.

**2. "Pull-on-burst" is not an implementable defence at our latency, yet §3
calls it "the single most important control".** (M-7)
Cancellation is a race we lose by 2–3 orders of magnitude (BCS). Replace with:
(a) a **participation region** in `(|d|, r)` from the closed-form frontier
`m/φ(d) ≥ k√(3L/r)`; (b) **size** as the primary continuous risk knob; (c) keep
"pull" only for the one non-race trigger — our own feed going stale. Demote the
burst flag to a **model-validity monitor** (it detects the vol-regime change
that breaks the `σ`-cancellation, which is a genuine and different job).
Additionally, `ζ` must become the per-level BCS sniping term with the taker fee
**inside** the moat: §3 charges the fee as a cost and never credits it as
protection.

**3. `λ_fill` has no queue state, so join-vs-improve — the only real decision on
a 2–4-tick book — is not modelled.** (M-1/M-2)
§3 writes `λ_fill(ℓ, queue; bracket)` but defines no queue object, and the EV
omits the *dynamic* (optionality) component `D_ℓ`, which is precisely what makes
joining preferable to improving. Add `Q_ahead` as a state, bracket it (the
bracket belongs on `Q_ahead`, not on `λ`), add `D_ℓ`, and add the requote
hysteresis rule.

**4. The maker rebate is missing from the objective.** (M-4)
`ρ(p) = 0.014·p(1−p)` = $0.0035/share ATM = **0.35 tick**. On a 2–4-tick book
that is not a rounding term; it is fill-contingent, so it belongs inside the
markout accounting, not beside it. Related correction: §1 and the M-lens
attribute the pickoff moat to the fee, but the fee is 78% of the moat ATM and 2%
at `|d| = 3` — **the far-from-money moat is the tick**, which makes the
tick-change threshold (P-M2c) a first-order economic parameter.

**5. Net inventory `q = q_up − q_down` cannot express pair-harvest.** (M-3)
Two fills at `b_up + b_down = 0.98` net to `q = 0`, register as zero in the
reservation — and lock $0.02 that the model never sees. The state must be
`(m, q)`, the pair must be quoted as a joint constraint, and the pair decision
must be gated on completion probability versus legging cost, not on the merge
being cheap (the merge *is* cheap; that was never the binding constraint).

**6. The inventory limit is a constant `Q_max`; it should be a constant risk
budget.** (M-8) `Q_max(p̂) = κ/(γ·p̂(1−p̂))`. A flat cap is simultaneously too
tight in decided windows — exactly where M-5 says occupancy is affordable — and
too loose ATM.

**7. The linear reservation `r = p̂ − γqv` is used where the exact CARA
`g`-ratio is closed-form.** (M-8, upgrading T-F7 from SHOULD-FIX to structural)
Its error is on the short-longshot side, the same side as T-F14's Gaussian-tail
error, and with **no liquidation leg there is no later correction**.

**8. The `r³` behaviour is described in language borrowed from options
pinning.** (M-6) It is variance lock-in of the settlement statistic, not
price-impact feedback. The two have opposite implications for endgame quoting.
State the mechanism correctly and put a Kumar–Seppi manipulation screen — not an
Avellaneda–Lipkin pinning model — in the endgame risk layer.

**9. No boundary-reader parameter and no tie-zone width.** (M-6) `K` and `X_T`
are *reports*; `b(·)` is an estimated parameter and `δ_tie` defines the
unquotable region. §2 carries the physics but not the reading convention.

**10. `γ` still has no calibration recipe** (standing T-F10), and it is now
load-bearing in three places instead of one: the CE quotes, `Q_max(p̂)`, and the
legging-cost `κ`.

---

## References

Market making, control, and queues

- Avellaneda, M. & Stoikov, S. (2008), *High-frequency trading in a limit order book*, Quantitative Finance 8(3). https://www.tandfonline.com/doi/abs/10.1080/14697680701381228
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013), *Dealing with the inventory risk: a solution to the market making problem*, Math. Finance & Economics 7(4). https://arxiv.org/abs/1105.3115
- Guéant, O. (2017), *Optimal market making*, Applied Mathematical Finance. https://arxiv.org/abs/1605.01862
- Bergault, P. & Guéant, O. (2021), *Size matters for OTC market makers: general results and dimensionality reduction techniques*, Mathematical Finance. https://arxiv.org/abs/1907.01225
- Bergault, P., Evangelista, D., Guéant, O. & Vieira, D. (2021), *Closed-form approximations in multi-asset market making*. https://arxiv.org/abs/1810.04383
- Guilbaud, F. & Pham, H. (2013), *Optimal high-frequency trading with limit and market orders*, Quantitative Finance 13(1). https://arxiv.org/abs/1106.5040
- Moallemi, C. C. & Yuan, K. (2016/2017), *A model for queue position valuation in a limit order book*. https://moallemi.com/ciamac/papers/queue-value-2016.pdf · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2996221
- Huang, W., Lehalle, C.-A. & Rosenbaum, M. (2015), *Simulating and analyzing order book data: the queue-reactive model*, JASA 110(509). https://arxiv.org/abs/1312.0563
- Dayri, K. & Rosenbaum, M. (2015), *Large tick assets: implicit spread and optimal tick size*, Market Microstructure and Liquidity. https://arxiv.org/abs/1207.6325
- Cartea, Á. & Wang, Y. (2020), *Market making with alpha signals*, IJTAF 23(3). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3439440
- Cartea, Á., Jaimungal, S. & Sánchez-Betancourt, L. (2021), *Latency and liquidity risk*, IJTAF. https://arxiv.org/abs/1908.03281
- Gao, X. & Wang, Y. (2018/2020), *Optimal market making in the presence of latency*. https://arxiv.org/abs/1806.05849
- Wang, Z., Ventre, C. & Polukarov, M. (2025), *Robust market making: to quote, or not to quote*. https://arxiv.org/abs/2508.16588
- Feil, D. & Nendel, M. (2026), *Optimal market making in prediction markets*. https://arxiv.org/abs/2607.17991

Incentive schemes, obligations, designated market makers

- El Euch, O., Mastrolia, T., Rosenbaum, M. & Touzi, N. (2021), *Optimal make-take fees for market making regulation*, Mathematical Finance 31(1). https://arxiv.org/abs/1805.02741
- Baldacci, B., Possamaï, D. & Rosenbaum, M. (2021), *Optimal make-take fees in a multi market-maker environment*, SIAM J. Financial Math. https://arxiv.org/abs/1907.11053
- Aïd, R., Bergault, P. & Rosenbaum, M. (2025), *Competition and incentives in a shared order book*. https://arxiv.org/abs/2509.10094
- Aqsha, A., Bergault, P. & Sánchez-Betancourt, L. (2025), *Equilibrium reward for liquidity providers in automated market makers*. https://arxiv.org/abs/2503.22502
- Bessembinder, H., Hao, J. & Zheng, K. (2015), *Market making contracts, firm value, and the IPO decision*, Journal of Finance 70(5). https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12285
- Anand, A. & Venkataraman, K. (2016), *Market conditions, fragility, and the economics of market making*, JFE 121(2). https://www.sciencedirect.com/science/article/abs/pii/S0304405X16300459
- Tullock, G. (1980), *Efficient rent seeking* — proportional-share contest; see e.g. survey https://doi.org/10.3390/g13060083

Fees

- Colliard, J.-E. & Foucault, T. (2012), *Trading fees and efficiency in limit order markets*, RFS 25(11). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1853124
- Foucault, T., Kadan, O. & Kandel, E. (2013), *Liquidity cycles and make/take fees in electronic markets*, Journal of Finance 68(1). https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2012.01801.x
- Glosten, L. & Milgrom, P. (1985), *Bid, ask and transaction prices in a specialist market with heterogeneously informed traders*, JFE 14(1). https://www.sciencedirect.com/science/article/abs/pii/0304405X85900443

Settlement, benchmarks, manipulation, pinning

- Duffie, D. & Dworczak, P. (2021), *Robust benchmark design*, JFE 142(2). https://www.nber.org/papers/w20540
- Kumar, P. & Seppi, D. J. (1992), *Futures manipulation with "cash settlement"*, Journal of Finance 47(4). https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1992.tb04666.x
- Mackinga, T., Nadahalli, T. & Wattenhofer, R. (2022), *TWAP oracle attacks: easier done than said?*, IEEE ICBC. https://eprint.iacr.org/2022/445
- Avellaneda, M. & Lipkin, M. (2003), *A market-induced mechanism for stock pinning*, Quantitative Finance 3(6) — **cited to be rejected for M-6**. https://www.cis.upenn.edu/~mkearns/finread/PinningPaper.pdf
- Petajisto, A. (2017), *Inefficiencies in the pricing of exchange-traded funds*, FAJ 73(1). http://www.petajisto.net/papers/etf26.pdf

Speed, sniping, arbitrage

- Budish, E., Cramton, P. & Shim, J. (2015), *The high-frequency trading arms race: frequent batch auctions as a market design response*, QJE 130(4). https://ericbudish.org/wp-content/uploads/2022/03/high_frequency_trading_arms_race_slides_seminar2015.pdf · http://econweb.umd.edu/~sweeting/hft-arms-race.pdf
- Aquilina, M., Budish, E. & O'Neill, P. (2022), *Quantifying the high-frequency trading "arms race"*, QJE 137(1). https://academic.oup.com/qje/article/137/1/493/6368348
- Menkveld, A. J. & Zoican, M. A. (2017), *Need for speed? Exchange latency and liquidity*, RFS 30(4). https://academic.oup.com/rfs/article-abstract/30/4/1188/2966376
- Foucault, T., Kozhan, R. & Tham, W. W. (2017), *Toxic arbitrage*, RFS 30(4). https://academic.oup.com/rfs/article-abstract/30/4/1053/2758635
- Kozhan, R. & Tham, W. W. (2012), *Execution risk in high-frequency arbitrage*, Management Science 58(11). https://pubsonline.informs.org/doi/10.1287/mnsc.1120.1541

Utility indifference

- Hodges, S. & Neuberger, A. (1989), *Optimal replication of contingent claims under transaction costs*, Review of Futures Markets 8.
- Henderson, V. & Hobson, D. (2009), *Utility indifference pricing: an overview*, in *Indifference Pricing: Theory and Applications*, Princeton UP. https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/henderson/publications/indifference_survey.pdf
