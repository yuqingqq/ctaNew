# PM_THEORY_CHECK — independent orchestrator verification (2026-08-20)

Written in parallel with the theory agent's `PM_MECHANISM_THEORY.md`, NOT from
it — iteration 2 reconciles the two. Purpose: verify the mechanism→theory
mappings I asserted in §9 rather than assuming them.

## 1. DIRECT HIT — our exact problem is already solved in the literature

**"Optimal Market Making in Prediction Markets", arXiv 2607.17991** —
binary CLOB (not AMM/scoring-rule), terminal resolution Y ∈ {0,1}, quotes
constrained to [0,1]. This is our setting, not an analogue. Adopt as the
program's base model, replacing the "binary-GLFT" improvisation in §3.

Setup: prices live in probability space via a latent belief diffusion
`dL = μdt + σdW`, `p = f(L)` (e.g. logistic), with μ chosen so p is a
**martingale**; volatility `ς(t,p)` vanishes at the boundaries 0/1.

Objective: `E[X_T + q_T·Y + Φ(p_T,q_T) − γ∫₀ᵀ q_s²ς(s,p_s)²ds]`
Terminal penalty: **Φ(p,q) = −γ_T·q²·p(1−p)**
Optimal quotes: `π*^b = argmax_π Λ^b(t,p,π)·[p − π − z_b]`,
`π*^a = argmax_π Λ^a(t,p,π)·[π − p − z_a]`, where
`z_b = [V(t,p,q) − V(t,p,q+Δ)]/Δ` (marginal cost of taking on inventory).
The 4-D HJB reduces to 3-D because cash enters linearly:
`Υ(t,p,q,x) = x + qp + V(t,p,q)`.

**Two independent confirmations of our review findings:**
- Its terminal penalty `q²p(1−p)` is *exactly* what the T-lens derived for
  v(t) (T-F9) — arrived at independently, same answer.
- Its optimal quote is an **argmax of intensity × (edge − marginal inventory
  cost)** — structurally identical to the discrete per-level EV form T-F8
  forced on us when the continuous-δ closed form died on the $0.01 grid. Our
  `ℓ* = argmax_ℓ λ_fill(ℓ)·[P_ℓ − CE(q) − ζ(ℓ)]` is the discrete-price
  instance of their Hamiltonian. The published model is the continuous-π
  parent; our tick grid just restricts the feasible set.

**Subtlety to carry (not in the paper's framing):** it charges BOTH a running
`γ∫q²ς²ds` and a terminal `γ_T q²p(1−p)`. For a martingale ending in {0,1},
`E[∫₀ᵀ ς²ds] = p₀(1−p₀)` — the two penalties measure the *same* uncertainty at
different points in time. They are therefore not independently interpretable;
γ and γ_T must be calibrated **jointly**. This is the same double-counting the
T-lens caught in our own v(t), surviving in a published model as separate
path-MTM vs terminal-settlement risk aversions. Adopt with eyes open.

**Gap the paper leaves us:** adverse selection is only *implicit*, via the
order intensities Λ(t,p,π); there is no information structure. Our ζ(ℓ)
markout-calibrated term and the M-7 sniping layer are exactly this hole, and
are our own work — not borrowable.

> **SUPERSEDED 2026-08-20 by `PM_MECHANISM_THEORY.md` (M-5).** My conclusion
> below — "that body of optimal-control theory does not meaningfully exist" —
> was a failure of my search, not a fact about the literature. The right body
> is **principal–agent market-making contracts**: El Euch–Mastrolia–Rosenbaum–
> Touzi (optimal make-take fees / MM regulation) and Baldacci–Possamaï–
> Rosenbaum, i.e. the exchange designing an incentive contract for a liquidity
> provider — which is precisely what a rewards band is. Add a **Tullock
> proportional contest** for the reward-sharing competition among makers, and
> constrained control (KKT shadow price) for the band itself. My two-policy
> comparison below is still a valid *special case* (it computes V_B − V_A) but
> it is the crude version: it cannot express the contest equilibrium, which is
> where the interesting result lives (see §11 of the plan: R/X → c, so the band
> only pays a maker with differentially lower pickoff cost). Section retained
> for the audit trail.

## 2. CORRECTION — my M-5 "DMM obligation literature" claim does not hold

I asserted the rewards band maps onto a designated-market-maker obligations
literature. Searching for it: **that body of optimal-control theory does not
meaningfully exist.** What exists is (a) institutional obligation *specs*
(e.g. LSE ETF market-maker obligations: max-spread bands 1.5/3/5/15/25% and
min quote size — documents, not models), and (b) MM control with *size*
constraints (arXiv 1802.08135, 1903.07222; soft inventory limits that taper
quote size). Neither is a theory of quoting under a subsidy-qualifying band.

**Correct formulation instead — and it is cleaner:** the rewards band is an
**action-space constraint that is optional**. So the problem is not
"constrained control" but a **two-policy comparison**, solving the same HJB
twice:
```
policy A: unconstrained quotes,      reward flow = 0
policy B: quotes ∈ band(spread ≤ s*, size ≥ n*),  reward flow = R(t)
adopt B iff  V_B(0,p,0) > V_A(0,p,0)
```
The value difference `V_B − V_A` *is* the implicit price of the subsidy, and
its sign is the whole M-5 question. Note this is precisely the M-lens's
MUST-FIX M1 (G3a must re-optimise rather than subtract a rewards line) —
arrived at from theory rather than from experiment design, which is a good
sign that both are right.

## 3. CONFIRMED — M-7 sniping frame is the correct one

Budish–Cramton–Shim (QJE 2015) define latency arbitrage as rents from
**symmetrically observable public information** — which is exactly our
situation (the Binance/Chainlink price is public; we are simply slower to act
on it). Their results that matter for us: sniping is a *speed contest we
cannot win*, and the liquidity provider's equilibrium response is to **widen**;
the arms race is privately rational and collectively wasteful. So the honest
M-7 question is not "how fast can we get" but "what is the optimal policy for
a maker who will be sniped" — i.e. quote width and pull rules parameterised by
*measured* one-way latency, which is what the (|d|, r) surface now does.
Follow-up not yet read: "Optimal Market Making in the Presence of Latency"
(arXiv 1806.05849) — PDF didn't parse via fetch; flagged for iteration 2.

## 4. Consequences for the plan

1. **§3 is re-based on arXiv 2607.17991**, not on an improvised binary-GLFT.
   Our contributions on top of it are the parts it lacks: stream-anchored p̂
   (§2), the tick-grid restriction of its Hamiltonian, ζ(ℓ) adverse selection,
   the (|d|,r) sniping pull surface, and the CTF pair-merge exit.
2. **M-5 becomes a two-policy value comparison**, not an appeal to obligation
   theory. This also supplies the missing formal statement of G3a-vs-G3b.
3. **γ and γ_T calibrate jointly** — a single risk-aversion calibration
   exercise, not two.
4. Sources: arXiv [2607.17991](https://arxiv.org/html/2607.17991);
   Budish–Cramton–Shim [QJE 2015](https://ericbudish.org/research/financial-markets/);
   [arXiv 1806.05849](https://arxiv.org/pdf/1806.05849) (unread);
   size-constrained MM [1802.08135](https://arxiv.org/pdf/1802.08135),
   [1903.07222](https://arxiv.org/pdf/1903.07222);
   LSE MM obligations factsheet (institutional spec).
