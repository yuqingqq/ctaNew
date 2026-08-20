# R3 — Optimal Quoting, Inventory Control, Perp-Specific Terms

Literature sweep for HF market making on Binance USDM perps (solo, no colo, tick
aggTrades + forward-collected L2@100ms/bookTicker). Companion to R1/R2/R4.
All formulas transcribed from primary sources; page/eq refs given. Conventions:
mid `S_t`, bid depth `δ^b = S − S^bid`, ask depth `δ^a = S^ask − S`, inventory
`q` (contracts, + = long), fill intensity `Λ(δ) = A e^{−kδ}` per side.

---

## 1. GLFT closed form (Guéant–Lehalle–Fernandez-Tapia)

Ref: arXiv:1105.3115 (v5), *Dealing with the inventory risk*, Math. Fin. Econ. 2013.

**Model.** `dS_t = σ dW_t` (arithmetic BM, no drift). MM posts one unit at
`S−δ^b`, `S+δ^a`; fills are Poisson with intensity `A e^{−kδ}` per side.
Inventory hard-bounded `|q| ≤ Q`. CARA utility `E[−exp(−γ(X_T + q_T S_T − ℓ(q_T)))]`.

**Exact solution** (Thm 1 + Prop 2, pp. 6–7). Define
```
α = (k/2) γ σ²                      # units: 1/time
η = A (1 + γ/k)^{−(1 + k/γ)}        # units: 1/time
```
Let `M` be the (2Q+1)×(2Q+1) tridiagonal matrix with diagonal `α q²`
(q = −Q..Q) and off-diagonals `−η`. Then
```
v(t) = exp(−M (T−t)) · 1            # vector (v_{−Q},…,v_Q), matrix exponential
δ^{b*}(t,q) = (1/k) ln(v_q(t)/v_{q+1}(t)) + (1/γ) ln(1 + γ/k)   (q ≠ Q)
δ^{a*}(t,q) = (1/k) ln(v_q(t)/v_{q−1}(t)) + (1/γ) ln(1 + γ/k)   (q ≠ −Q)
```
This is fully implementable: one `scipy.linalg.expm` per (T−t) grid, or ODE
integrate `v' = M v` backwards. Total spread `ψ*(t,q) = δ^a+δ^b` given by the
same log-ratios.

**Asymptotic / ergodic version** (Thm 2 + closed-form approx, pp. 10–11).
Quotes become t-independent far from T: `δ^{b/a*}_∞(q)` use the eigenvector
`f⁰` of the smallest eigenvalue of `M`; Gaussian approximation of `f⁰` gives
the famous closed forms:
```
δ^{b*}_∞(q) ≈ (1/γ) ln(1+γ/k) + (2q+1)/2 · S_glft
δ^{a*}_∞(q) ≈ (1/γ) ln(1+γ/k) − (2q−1)/2 · S_glft
spread_∞    ≈ (2/γ) ln(1+γ/k) + S_glft            # q-independent
where  S_glft = sqrt( (σ² γ)/(2 k A) · (1 + γ/k)^{1 + k/γ} )
```
Structure: **constant spread + linear inventory skew**. Reservation price
`S_t − q·S_glft`; each unit of inventory shifts both quotes by `S_glft`
against the position. Approximation is excellent for small |q|, degrades near
±Q (paper, Fig. 4–5).

**Parameter meanings / units** (tick-time units; be consistent):
- `σ`: mid (better: microprice) volatility, price·time^{−1/2}, at the quoting
  timescale (seconds).
- `A`: fill intensity at δ=0 (at-mid), fills/sec per side. Sets trade frequency.
- `k`: decay of fill intensity per unit depth, 1/price. Sets how expensive it
  is to quote deep. `1/k` = risk-neutral optimal half-spread.
- `γ`: CARA risk aversion, 1/price. Only enters through the two closed forms;
  `γ→0` gives `δ = 1/k + (2q±1)/2·sqrt(σ²/(2kA))·…` → spread floor `2/k`.

**Calibration recipe** (from the paper's empirical section + standard practice):
1. `σ`: realized vol of microprice over the expected holding time (10s–5min
   bars, robust to noise; NOT tick-by-tick mid which is bid-ask-bounce inflated).
2. `(A, k)`: from aggTrades + your book stream. For a grid of depths δ_j
   (e.g. 0.5–10 bps), count events per second in which cumulative aggressor
   volume at-or-beyond depth δ_j (from the side's touch) ≥ your quote size —
   this is the "volume-minute"-style proxy (see §4 paper below, same recipe).
   Fit `ln λ̂(δ_j) = ln A − k δ_j` by OLS/Poisson GLM. Do it per side and per
   vol regime; A and k move a lot intraday.
3. `γ`: not observable — back it out from a target inventory band. Stationary
   inventory std under the asymptotic strategy scales like
   `σ_q ≈ sqrt(λ_fill · τ)` with mean-reversion set by skew; simplest: pick γ
   so `S_glft × Q_target ≈` the adverse move you can tolerate, or scan γ over
   replay and pick by realized inv-RMS. Hard inventory cap Q is a free extra
   control — use it (GLFT is the model *with* limits).

**Guéant 2016 general intensities** (arXiv:1605.01862, *Optimal market making*).
Generalizes to arbitrary decreasing `Λ(δ)`, trade size Δ, and both CARA (ξ=γ)
and risk-adjusted-expectation (ξ=0) objectives. Reduced HJB (eq. 3.9):
```
0 = −∂_t θ(t,q) + ½ γσ² q² − 1_{q<Q} H^b_ξ( [θ(t,q)−θ(t,q+Δ)]/Δ )
                             − 1_{q>−Q} H^a_ξ( [θ(t,q)−θ(t,q−Δ)]/Δ )
```
Optimal quote = function of the **inventory value difference** p:
```
δ^{b*}(t,q) = δ̃*_ξ(p),  p = [θ(t,q)−θ(t,q+Δ)]/Δ
FOC (ξ=0):  p = δ + Λ(δ)/Λ'(δ)      (⇒ δ = p + 1/k for exponential Λ)
FOC (ξ>0):  p = δ − (1/ξΔ) ln(1 − ξΔ Λ(δ)/Λ'(δ))
```
General closed-form approximation (eq. 4.3–4.4) — only needs `H''_ξ(0)`:
```
δ^{b*} ≈ δ̃*_ξ( (2q+Δ)/2 · sqrt(γσ² / (2 H''_ξ(0))) )
δ^{a*} ≈ δ̃*_ξ( −(2q−Δ)/2 · sqrt(γσ² / (2 H''_ξ(0))) )
```
This is the practical bridge to *empirical* fill curves: measure Λ(δ)
nonparametrically, compute H and its curvature at 0 numerically, keep the same
two-term structure. Exponential Λ recovers GLFT exactly.

**Bergault et al. multi-asset** (arXiv:1810.04383, *Closed-form approximations
in multi-asset market making*). What changes with d assets: price risk pools
through the covariance Σ. Quadratic-Hamiltonian approximation gives (eqs. §3.3.3,
Model A = CARA):
```
δ^{i,b} ≈ (1/(γ z_i)) ln(1 + γ z_i/k_i) + √γ ( q'Γe_i + ½ z_i e_i'Γe_i )
δ^{i,a} ≈ (1/(γ z_i)) ln(1 + γ z_i/k_i) − √γ ( q'Γe_i − ½ z_i e_i'Γe_i )
Γ  = D₊^{−1/2} (D₊^{1/2} Σ D₊^{1/2})^{1/2} D₊^{−1/2}
D₊ = diag( 2 A_i C^i_ξ k_i z_i ),   C^i_ξ = (1+ξz_i/k_i)^{−(1+k_i/(ξ z_i))}
```
(z_i = trade size.) Half-spread per asset: liquidity term + `½√γ z_i e_i'Γe_i`;
**skew is linear in the whole inventory vector**: `skew_i = −√γ q'Γe_i` — you
skew BTC quotes for ETH inventory through the √-covariance Γ. Single asset
collapses to GLFT (D₊ = 2kA(1+γ/k)^{−(1+k/γ)} ⇒ √γσ/√D₊ = S_glft). For a
BTC+ETH+alts book this is the formula to use for portfolio-level skew.

---

## 2. Alpha/drift-augmented quoting

**The clean closed form with drift is Bergault §5** (arXiv:1810.04383, eq. 29–30
after quadratic approx; drift vector μ, `dS^i = μ^i dt + σ^i dW^i`):
```
δ^{i,b} ≈ [liquidity term] + √γ ( q'Γe_i + ½ z_i e_i'Γe_i − (1/γ) e_i' D₊^{−1/2} Â⁺ D₊^{1/2} μ )
δ^{i,a} ≈ [liquidity term] − √γ ( q'Γe_i − ½ z_i e_i'Γe_i + (1/γ) e_i' D₊^{−1/2} Â⁺ D₊^{1/2} μ )
Â = √γ (D₊^{1/2} Σ D₊^{1/2})^{1/2},  Â⁺ = pseudo-inverse
```
Single asset: both depths translate by the **same** amount with opposite sign,
i.e. the whole quote pair shifts toward the drift by
```
shift = μ / (γ σ sqrt(2 A k C_ξ))        # price units
```
Spread is unchanged. So under risk aversion, drift = **pure reservation-price
translation**, attenuated by γσ√(fill capacity).

**Risk-neutral / OU-signal version.** With an OU alpha `dα = −κ_α α dt + …`
entering the drift, the value function acquires `E_t[∫ drift] = α_t(1−e^{−κ_α(T−t)})/κ_α`,
so for T ≫ 1/κ_α the translation is `α_t/κ_α` = **total expected future price
move**, not the instantaneous drift. (This is the standard result threaded
through Cartea–Jaimungal–Penalva ch. 10 and the signal-execution literature.)

**Bottom line — the practical rule, stated precisely:**
- To first order, alpha-augmented GLFT = plain GLFT quoted around
  `S_fair = microprice + ρ_t` instead of mid, where
  `ρ_t = E_t[Σ future drift] ≈ α̂_t/κ_α` (α̂ = current instantaneous drift
  estimate, κ_α = its decay rate) — equivalently, if your signal is already a
  *cumulative* h-horizon return forecast `r̂` with h ≳ expected holding time,
  `ρ_t = r̂`. Under risk aversion cap the shift at the Bergault value
  `μ̂/(γσ√(2AkC))`. Spread and inventory-skew terms unchanged.
- Second, distinct effect that translation does NOT capture: when fill
  intensities themselves depend on the signal (informed flow hits you when α
  is large), the optimal action becomes **one-sided withdrawal** — pull the
  quote on the adversely-selected side entirely. This is the Cartea–Wang
  result, below. Implement as an override, not as a bigger shift.

**Cartea & Wang, *Market Making with Alpha Signals*** (local PDF read; Oxford
2020). Midprice = pure jump `dS = σ(dJ⁺−dJ⁻)`, jump intensities
`μ± = (α_t)± + θ`; alpha is OU-with-jumps driven by MO arrivals:
`dα = −κα dt + ξdW + η⁺(dM^{0+}+dM⁺) − η⁻(dM^{0−}+dM⁻)`. Controls: post/no-post
at the touch (binary l±) + take with MOs (impulse). HJBQVI (their eq. 8/11)
solved **numerically** — no closed-form depths. Qualitative structure of the
optimal policy (their Fig. 1): α≈0 → quote both sides; α moderately positive →
quote bid only (avoid being picked off on the ask); α beyond a threshold →
cross the spread with a buy MO. Thresholds tighten with inventory. MLE
calibration recipe for (κ, η±, θ) from MO/price-jump timestamps given in their
§2.1 (closed-form log-likelihood, their eq. 26–29) — directly reusable on
aggTrades. Nasdaq estimates: alpha half-life < 0.02 s (colo territory); crypto
perp order-flow alphas live at 100ms–minutes, so the structure ports but check
your own κ̂ before believing you can react in time.

**Cartea–Jaimungal–Ricci** (*Buy Low Sell High*, SIAM J. Fin. Math. 5(1), 2014;
*Algorithmic Trading, Stochastic Control and Mutually Exciting Processes*, SIAM
Review 60(3), 2018). MOs arrive as mutually-exciting Hawkes processes; midprice
jumps co-move with MO clusters. Optimal postings solved via DPE ODE systems.
Takeaways that survive: (i) widen spreads during activity bursts (Hawkes
intensity ↑ ⇒ short-horizon vol ↑ and toxicity ↑), (ii) skew with signed
order-flow, (iii) "buy low sell high" fill asymmetry is captured by
flow-dependent intensities, not by drift alone.

**Yu 2026, *Explicit Signal-Adaptive Sequential Optimal Execution Quotes***
(local PDF read; arXiv:2605.24242). One-sided (liquidation) quoting with
signal drift `g(s_t)` in the reference price, fills `λe^{−κδ}`, execution price
`M − a + bδ`. All four criteria (±CARA, ±running penalty) reduce to triangular
linear ODE systems ⇒ fully explicit quotes. Case I (risk-neutral, their
Lemma 3.2):
```
h(t,q) = (b/κ) ln w(t,q)
∂_t w(t,q) + (κ/b) g(s) q w(t,q) + λ e^{−κa/b − 1} w(t,q−1) = 0,  w(t,0)=1,
w(T,q) = exp(−(κ/b) q I(q))
δ*(t,q) = clip( (1/κ)[1 + ln(w(t,q)/w(t,q−1))] + a/b , δmin, δmax )
```
Signal enters via `g(s)q` in the triangular system; long-horizon behaviour:
ask deepens by ≈ `g(s)(T−t)/b` — quote at the *expected future* price.
"Frozen-coefficient" use: re-plug the current signal estimate into the explicit
formula each decision time (no re-solve). This is the right template for the
**inventory-unwind mode** of an MM (e.g. after a cap breach) with your alpha
plugged in.

---

## 3. Adverse selection inside the quoting model

**arXiv:2508.20225, *Optimal Quoting under Adverse Selection and Price
Reading*** (HTML version fetched). Adverse selection modeled as **post-fill
price impact**: after your quote at offset δ fills, the reference price moves
against you by markout function `ζ(δ)` (nondecreasing; per tier/size in their
multi-tier OTC setting); "price reading" = drift term `J(skew)` — others infer
your inventory from your skew. Solved as a **first-order perturbation** (ε) of
the baseline Guéant-type model:
```
δ* = δ*_baseline(q) + ε·[ global term ∝ D±f(q)  +  tier term ∝ (Λζ)'(δ*)/Λ'(δ*) ] + o(ε)
```
With exponential Λ and `ζ(δ) = α_ζ e^{βδ}`, everything is closed-form (their
quadratic-approx expressions). **Key structural answer: adverse selection does
NOT change the shape of the quote rule** — same inverse-intensity/value-difference
form — it adds an additive, inventory-linear correction whose sign depends on
whether informed flow is "slow" (β<κ: tighten to informed side, use it to shed
risk) or "sharp" (β>κ: widen the defensive side). **Calibration is exactly your
markout study**: estimate `ζ(δ)` = mean adverse mid move at h seconds after a
fill at depth δ, from your own fills (forward-collected) or from proxy fills
(aggTrades crossing depth δ).

**arXiv:2501.03658, Barucci–Mathieu–Sánchez-Betancourt, *Market Making with
Fads, Informed, and Uninformed Traders*** (v3, 2026; PDF read). Mid =
fundamental + OU fad `U_t` (`dU = −ηU dt + dB`). Two trader populations give
**toxicity-dependent fill intensities** (their eq. 4–5):
```
λ^a(δ) = φ e^{−kδ}            # uninformed: react to distance from MID
       + ψ e^{−kδ − γ_f σq U_t}   # informed: react to distance from FUNDAMENTAL
```
(similarly bid with `+γ_f σq U_t`). This is *the* tractable device: keep the
exponential-in-δ shape, multiply the informed component by exp(signal). Because
both populations share the same decay k, the FOC keeps the standard form
(their §3): `δ^{a*} = (1/k − V₁) ∨ −δ_∞` with V₁ = value difference — i.e.
**again only the value function (⇒ skew/spread levels) changes, not the rule's
shape**. Fad raises informed selling when price is above fundamental ⇒ optimal
response widens the bid and tightens the ask *before* inventory moves.
Approximate closed forms via quadratic ansatz (A Riccati, B linear in u, C
quadratic in u; their Props. 10–11, 15–16). Partial information: replace U by
its Kalman–Bucy filter — quote on the filtered fad. Spread is an explicit
increasing function of informed fraction ψ/φ (VPIN-style toxicity → spread map).

**Practical synthesis for us:** one quoting engine, GLFT-shaped; three
toxicity hooks, all parameter-level: (i) per-side effective intensity
`A_side(t) = φ + ψ·e^{∓γ_f·fad}` (fad proxy: microprice−mid, OFI, or
short-horizon alpha itself), (ii) markout-based correction from 2508.20225
calibrated off your own fills, (iii) one-sided withdrawal at signal extremes
(Cartea–Wang). No new solver needed.

---

## 4. Perp-specific terms

**Funding-aware quoting** — local PDF read: Nam Anh Le, *Funding-Aware Optimal
Market Making for Perpetual DEXs*, arXiv:2605.06405 (May 2026). Hyperliquid
ETH/BTC/SOL calibration. Setup: cash pays `−q_t f_t dt` with
**`f_t = S_t F_t`** (F = fractional funding rate/hour; cash-scaling is the
paper's main bookkeeping point — funding must be in the same units as spread
capture). Funding state OU: `df = κ(f̄−f)dt + σ_f dW`. Reduced HJB (their §5.2,
ansatz `v = x + qS + θ(t,q,f)`):
```
0 = ∂_tθ + κ(f̄−f)∂_fθ + ½σ_f²∂_ffθ − q·f − φq² + H^a + H^b
H^{a}(t,q,f) = sup_δ Λe^{−kδ}[ δΔq + θ(t,q−Δq,f) − θ(t,q,f) ],   θ(T,q,f) = −αq²
```
**The funding term enters exactly like a running drift on inventory value**
(`−qf`, same slot as the `+ασq` drift term in Cartea–Wang's eq. 11). Quote
recovery (their §5.3): `δ^{a*} = 1/k − A(t,q,f)`, `δ^{b*} = 1/k − B(t,q,f)`
with A,B = neighboring inventory value differences — classic AS in the f→0
limit. The **skew term** is the LQ-prototype cross term `a₄(t)·q·f` (their
appendix): positive funding makes long inventory carry-negative ⇒ ask
tightens / bid widens, proportionally to f. Empirics (100-seed holdout, two
fill proxies): funding-aware HJB beats plain AS on mean PnL with ~35% lower
inventory RMS on ETH/BTC; SOL "gain" is just a bigger risk point (risk-scaled
AS dominates) — honest nulls included.

**Port to Binance USDM** (my synthesis, consistent with the paper's structure):
funding is 8h-settled but accrues via the premium-index TWAP inside the window
and is **largely predictable intra-window** (Binance publishes the running
estimate). So don't carry an OU state — treat the predicted accrual rate
`f̂_t` (USDT/contract/hour, = S·F̂/8h-window logic) as a *known negative alpha
on inventory* and fold it into the §2 drift rule:
```
α_total(t) = α_price(t) − f̂_t·1{q-side pays}   →  reservation shift = −shift(f̂)
```
i.e. quote shifted away from the paying side; near the funding timestamp with
|F̂| large this is a hard skew (shed inventory before settlement, or hold to
*collect*). The paper's OU half-lives (2–6h HL, ETH/BTC/SOL; Binance ETH
7.96h in their footnote) say funding persistence ≈ your inventory horizon ×
100 — so for an HF book funding is a slowly-varying bias term, not a fast
state. Its jump diagnostics (1–2%/hour jump prob) warn: funding can gap with
the market — see cascades below.

**Liquidation-cascade risk / inventory limits.** No mature "MM inventory limits
under cascades" theory yet; nearest literature: arXiv:2603.15963 (*Risk-Based
Auto-Deleveraging* — ADL mechanics after Oct 10–11 2025 shock, insurance-fund
exhaustion), arXiv:2603.09164 (*Slippage-at-Risk* — forward-looking liquidity
risk for perp exchanges: leverage × liquidation × book-depth interaction),
plus the funding paper's jump evidence. Practical implications for us:
(i) size the inventory cap Q off **jump/cascade VaR**, not diffusion σ — during
a cascade your book widens 10–50x exactly when you're loaded, and fills are
one-sided; (ii) GLFT-with-limits is the right frame because Q is a first-class
model input — set `Q ≤ tolerable-loss / (cascade move × margin multiplier)`;
(iii) as a maker you also face ADL/socialized outcomes on the winning side of a
cascade — cap notional relative to book depth at ±1%, not just to equity;
(iv) kill-switch on cascade signatures (liquidation prints, OI drop, funding
gap) beats any smooth γ. DEFER formal modeling; adopt as engineering rules.

---

## 5. RL-based quoting — verdict

- arXiv:2305.15821 (Guo–Lin–Huang): DRL on raw LOB with CNN+attention
  ("Attn-LOB"), continuous action space, hybrid reward. Simulator-validated
  only; feature extractor is the contribution, not the control.
- arXiv:2207.09951 (Gašperov–Kostanjčar, IEEE CSL 2022): DRL on a multivariate
  Hawkes LOB simulator; beats benchmark MM strategies on risk-reward *in the
  simulator it was trained in*.
- arXiv:2510.27334 (Jafree–Jain–Firoozye 2025): PPO + self-imitation MM in a
  Hawkes LOB; RL market maker learns to *adversely select meta-orders*
  (exploits detectable drift). Interesting as a red-team result: whatever your
  execution leaves detectable, an RL MM can learn to pick off.

**Verdict: REJECT for now (as the control layer).** Every positive RL-MM result
above is trained and evaluated inside a hand-built simulator; the RL agent
mostly re-learns what GLFT+signal already gives in closed form (skew ∝ inventory,
widen in bursts, lean on flow), with far worse sample efficiency, no
interpretability, and severe sim-to-real risk for a solo operator without a
queue-accurate simulator. The decomposition closed-form-control + ML-signal
(supervised alpha into §2's shift) captures ~all of the value with testable
pieces. **Revisit if**: (a) you build a queue/latency-accurate replayer that
you'd trust to train against anyway, (b) months of own-fill data accumulate
(markouts, queue outcomes) so an offline-RL/policy-distillation step has real
data, or (c) the action space grows beyond two depths (multi-level, cancel
timing, take/make mix) where closed forms stop existing.

---

## 6. What breaks in Avellaneda–Stoikov in crypto, and fixes that matter

From practitioner sources (Hummingbot guides + quante.substack critique) +
the papers above:

| # | Broken assumption | Symptom in production | Fix (ranked by importance) |
|---|---|---|---|
| 1 | No adverse selection: fills assumed uninformed | inventory arrives exactly before adverse moves; spread PnL eaten by markouts | 1st: alpha shift + one-sided withdrawal (§2); markout-calibrated correction ζ(δ) (§3); this is the #1 fix |
| 2 | Constant A, k, σ | intensities are Hawkes-bursty + heteroskedastic; params stale within minutes | rolling re-estimation (minutes) + vol-regime buckets; widen on burst detection (CJR §2) |
| 3 | Symmetric quoting through trends (martingale mid) | quoting both sides through a trend = buying the whole way down | drift/funding shift of reservation price (§2, §4); trend filter as withdrawal override |
| 4 | Exponential fill curve exact | empirical λ(δ) not exponential (queue priority, tick granularity) | Guéant 1605.01862 general-Λ machinery: fit λ(δ) nonparametrically, keep H''(0) formula |
| 5 | Finite-T artifacts (quotes collapse near T) | AS finite-horizon terms are artifacts if you never stop trading | use ergodic/asymptotic GLFT forms (t-independent) |
| 6 | No queue position / latency | model thinks depth is the only fill control; reality: queue rank + cancel latency dominate at the touch | R2's territory; at minimum treat "at-touch vs deeper" discretely and measure fill-quality per level |
| 7 | No funding / perp carry | 8h funding is a predictable cash drift on inventory | fold f̂ into skew (§4) — cheap and strictly positive EV |
| 8 | Diffusion risk only | cascades/jumps size the real tail | hard Q from jump-VaR + kill-switch (§4) |

---

## 7. ADOPT / REJECT / DEFER

| Item | Verdict | Note |
|---|---|---|
| GLFT asymptotic closed form (spread + linear skew) as the quoting core | **ADOPT** | 5 params (σ, A, k, γ, Q); exact matrix-exp version as offline check |
| Guéant general-intensity approx (fit empirical λ(δ), use H''(0)) | **ADOPT** | upgrade path when exponential fit is poor |
| Drift injection as reservation-price shift (α̂/κ_α, risk-capped per Bergault §5) | **ADOPT** | the single highest-value augmentation |
| One-sided withdrawal at signal extremes (Cartea–Wang policy shape) | **ADOPT** | as thresholds/overrides, not by solving the HJBQVI |
| Funding as inventory carry: f̂ = S·F̂ folded into skew | **ADOPT** | Binance funding predictable from premium index; near-free |
| Markout-calibrated adverse-selection correction ζ(δ) (2508.20225) | **ADOPT (phase 2)** | needs own-fill or proxy-fill markout study first |
| Two-population toxic intensity (fads model) with filtered toxicity state | **DEFER** | adds a filter + state; revisit once a fad/toxicity proxy shows stable sign |
| Multi-asset Γ-skew (Bergault) | **DEFER** | adopt when quoting >1 symbol; single-asset collapses to GLFT anyway |
| Full funding-OU HJB grid (2605.06405) | **REJECT** | funding persistence ≫ HF holding times; the skew fold-in captures it |
| Cartea–Wang HJBQVI numerical solve in production | **REJECT** | half-life of their alpha < 20ms at Nasdaq scale; use policy shape only |
| RL quoting (any of §5) | **REJECT for now** | revisit under conditions listed in §5 |
| Formal cascade model for Q | **DEFER** | use jump-VaR sizing + kill-switch engineering rules now |

### Annotated primary refs
- GLFT: https://arxiv.org/abs/1105.3115 — exact + asymptotic quote formulas (§1).
- Guéant 2016: https://arxiv.org/abs/1605.01862 — general intensities, ξ=0/γ objectives, approx formulas.
- Bergault et al.: https://arxiv.org/abs/1810.04383 — multi-asset Γ-skew + drift term (§5 there).
- Cartea & Wang, *MM with Alpha Signals* (local PDF; Oxford OMI 2020) — signal-driven post/withdraw/take regions, MLE recipe for OU-jump alpha.
- Yu 2026: arXiv:2605.24242 — explicit signal-adaptive execution quotes (triangular systems).
- Adverse selection: https://arxiv.org/abs/2508.20225 (markout-perturbed quotes); https://arxiv.org/abs/2501.03658 (fads/informed/uninformed, filtered toxicity).
- Funding MM: arXiv:2605.06405 (local PDF) — f = S·F cash-scaling, −qf HJB term, Hyperliquid calibration.
- Cascades/ADL: https://arxiv.org/abs/2603.15963, https://arxiv.org/abs/2603.09164.
- RL: https://arxiv.org/abs/2305.15821, https://arxiv.org/abs/2207.09951, https://arxiv.org/abs/2510.27334.
- Practitioner: https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/ ; https://quante.substack.com/p/the-avellaneda-stoikov-algorithm.
