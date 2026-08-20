# PM_SKETCH_REVIEW_ITER1_T — theory/math lens, iteration 1

Object: `PM_MM_PLAN.md` §2 (market model) + §3 (binary-GLFT). Method: every
formula re-derived by hand and cross-checked by Monte Carlo / numerical
integration (BM paths, dt=0.01–0.05 s, N=8–20k; all quoted checks matched to
<1 SE). Numbers below use w=60 s, T=300 s, BTC σ = 30/40/60 %/yr =
0.53/0.71/1.07 bps/√s, basis residual std 0.7 bps and basis drift ~0.4 bps/85 s
(PM_REVIEW_ITER1 §A).

---

## F1. Pre-window variance σ²(τ − w + w/3) — CORRECT

For X_T = (1/w)∫_{T−w}^T S_u du, S BM with vol σ, at t ≤ T−w with a = τ−w,
b = τ:

Var_t[X_T] = (σ²/w²)·∫∫_{[a,b]²} min(u,v) du dv
           = (σ²/w²)·[(b³−a³)/3 − a²(b−a)] = σ²(b+2a)/3 = σ²(τ − 2w/3),

and τ − 2w/3 ≡ τ − w + w/3. The plan's decomposition is also exact, not just
mnemonic: Var = Var[S_{T−w}] + Var[X_T − S_{T−w}] = σ²(τ−w) + σ²w/3 with the
two terms independent. MC: 259.85 vs 260 (τ=300), 160.06 vs 160 (τ=200).
Endpoint σ²(τ−40 s) at w=60 correct.

## F2. In-window law σ²r³/(3w²) — CORRECT; cross-term is exactly zero

X_T = locked + (1/w)∫_t^T S_u du. Conditional on F_t the locked part is a
constant, so there is NO cross-term in Var_t — measurability, not approximation.
Open part: (1/w)[rS_t + σ∫_0^r B_s ds], Var[∫_0^r B_s ds] = r³/3 ⇒
Var_t[X_T] = σ²r³/(3w²). MC at r=30: 2.519 vs 2.5. Continuity at entry:
both regimes give σ²w/3 at τ=w ✓.

Caveat (feeds F8/F12): "locked is known" assumes the Chainlink path over
[T−w, t] is observed. The recorded 1 s stream gives X_t (already smoothed), not
S_t; the locked integral must be reconstructed as
locked = X_t − (1/w)∫_{t−w}^{T−w} Ŝ_u du with Ŝ_u = X_u + (F_u − TWAP_w(F)_u)
over the recorded per-second history (basis level cancels in that bracket).
Implementable, sub-bp error — but currently UNSTATED, and impossible during a
stream gap (see F12).

## F3. E_t[X_T] decompositions — CORRECT (both, incl. the w/2)

Pre-window: E_t[X_T] = S_t + μ·(mean of [T−w,T] − t) = S_t + μ(τ − w/2) — the
window's mean time is its center, so w/2 is exact. In-window:
E_t[X_T] = locked + (1/w)[rS_t + μr²/2] = locked + (r/w)(S_t + μr/2) ✓.
Continuity at τ=w ✓ (both give F + μw/2).

## F4. "Danger zone is ENTRY, not the final seconds" — HALF-RIGHT; the pull rule must be (|d|, r)-dependent, not τ-only

Local binary vol λ_bin = φ(d)·(∂d/∂F)·σ_F:

- pre-window: ∂E/∂F = 1 ⇒ λ_bin = φ(d)/√(τ − 2w/3) (per √s)
- in-window: ∂E/∂F = r/w, σ_eff = σr^{3/2}/(√3·w) ⇒ **λ_bin = √3·φ(d)/√r** —
  σ cancels.

Two facts, both derived and checked numerically:

(a) **Unconditional/typical-path: plan is RIGHT.** Starting ATM at window open,
E[λ_bin²(t)] = g(t)/(2π)·√(V/(2V₀−V)) (V = Var_t/σ²) rises ~2.6× from open to
entry, peaks exactly at r=w, then decays like √r. Integrated QV split: 0.187
pre-window vs 0.063 in-window (total = p(1−p) = 0.250 exactly) — 75% of
repricing variance is spent before the averaging window. From ATM-at-entry, the
last 10 s carry only 6% of remaining QV.

(b) **Conditional near-money: plan is WRONG.** λ_bin ATM does NOT pin — it
diverges like r^{−1/2}, and the √3 makes it **1.73× WORSE than a snapshot
binary at the same time-to-expiry** (sensitivity decays r/w but σ_eff decays
r^{3/2} — d reprices faster per unit F-move). Table (ATM, cents/√s,
σ-independent): entry 8.9; r=30: 12.6; r=10: 21.9; r=5: 30.9; r=2: 48.9.
And near-money endgames are not rare: P(|d|<1 | ATM at entry) = 29.5% at r=30,
15.5% at r=20, **5.4% at r=10** — one window in ~19, i.e. several times per
hour per coin. A resting quote in that state reprices ~22 c/√s; over the 471 ms
stream latency alone that is ~15 c of unobserved p̂ motion.

Fix: §2.1's prose ("averaging pins X_T … not the final seconds") licenses
quoting near-money late in the window; the model itself forbids it. The pull
rule must be a surface: pull when λ_bin(d, r)·√(latency + requote interval)
≳ tick (equivalently φ(d)√(3/r) over the reaction time), not a τ_min constant
"around T−60 s". τ_min-only is neither necessary (far-from-money late quotes
are safe and are where longshot rent lives) nor sufficient (ATM at r=10 s is
lethal).

## F5. Tie → Up — negligible; H-PM1's "real asymmetry at d≈0" overstates it

Settlement publishes `full_accuracy_value` at 1e18 scaling; P(X_T = K exactly)
at 18 decimals is ~0 at any practical σ. The ≥ convention shifts p̂ by P(tie) ≈
0. The REAL d≈0 hazards are the boundary-read convention and basis error
(F6/F12), already iter-1 data items #5/#9. Downgrade the wording; keep the rule
in the settlement-reconstruction test.

## F6. K = X_0 subtleties — two UNSTATED, one negligible

(a) **Pre-open quoting has no formula.** Books exist before t=0 (markets
pre-created ~5 min ahead) but K is not formed until t=0. For t ≤ −w the correct
variance is Var_t[X_T − X_0] = σ²(T − w/3) (= σ²·280 s, constant in t because
increments before −w hit both windows equally and cancel; MC 282.6 ± 4.4 vs
280 ✓), interpolating down to σ²(T − 2w/3) = σ²·260 s at t=0. If the MM quotes
pre-open, §2 needs this two-ended formula; if not, state "no quoting before
t=0". UNSTATED-ASSUMPTION.

(b) **Boundary-read jitter.** X_0/X_T are the stream reports nearest the
boundary; cadence 1 s with gaps to 8 s. X_t drifts at (S_t − S_{t−w})/w, so a
Δ-second offset moves the read by ~|S_t − S_{t−w}|·Δ/w ≈ σ√w·Δ/w: ~0.09 bps at
Δ=1 s, **~0.7 bps at an 8 s gap — same order as the whole basis noise** and
larger than σ_eff inside r≈20 s. Which report defines the boundary
(last-≤T vs first-≥T, observationsTimestamp vs validFrom) shifts K by exactly
this. Confirms iter-1 #5/#9 from the model side; the plan's §2 should carry the
number. UNSTATED-ASSUMPTION.

(c) 1 s-sample TWAP vs continuous integral: error std ~σ·Δ/√(12w) ≈ 0.03 bps —
negligible. CORRECT to ignore.

## F7. Reservation r = p̂ − γq·v — CORRECT to O((γq)²), but use the exact CE (it is closed-form and fixes the extremes)

CARA over W = q·1{Up}: CE(q) = −(1/γ)ln(p̂e^{−γq} + 1 − p̂) =: −(1/γ)ln g(q).
Indifference quotes: ask a(q) = (1/γ)ln(g(q−1)/g(q)), bid b(q) =
(1/γ)ln(g(q)/g(q+1)). Cumulant expansion: a,b = p̂ − γv·(q ∓ ½) + O(γ²q²κ₃),
v = p̂(1−p̂), κ₃ = v(1−2p̂). So:

- Linear-in-q with coefficient γv: CORRECT; the proper form is **q ± ½** —
  the missing γv/2 is the unit-lot risk half-spread, absorbed into δ* (state
  this so it isn't double-counted between r and δ*).
- No true circularity (v is evaluated at the model p̂, not the quote), BUT the
  linearization fails exactly where the plan needs care: v → 0 at extreme p̂
  turns inventory aversion OFF while the third cumulant blows up on the
  longshot-SHORT side. Numeric (p̂=0.03, γ=4e-4): q=−1000 exact ask 0.0441 vs
  linear 0.0416; q=−2500 exact 0.0776 vs linear 0.0591 — the linear form
  understates the short-longshot reservation by ~2 ticks. The exact g-ratio is
  two exponentials, bounded in [0,1] automatically (linear r can exit [0,1]),
  and costs nothing. SHOULD replace.

## F8. Half-spread (1/γ)ln(1+γ/k) on a 2–4-tick book — WRONG-IN-FORM; restate as a discrete per-level choice

The closed form presumes λ(δ)=Ae^{−kδ} with continuous δ and an interior
optimum. Here: δ takes 2–4 feasible values; a 2-parameter exponential fitted
through 2–4 points is ill-conditioned; and with γ ≪ k the formula gives
δ* ≈ 1/k ≈ 1–2 c — the same order as the tick, so rounding destroys whatever
optimality the closed form had. It also contains no queue concept, while §2.4
itself says join-vs-improve is the decision that matters (internal
inconsistency between §2.4 and §3).

Correct v1 formulation (uses only machinery the plan already builds): per side,
per tick level ℓ ∈ {improve, join, join-behind, out}:

EV(ℓ) = λ_fill(ℓ, Q_ahead; bracket)·[ P_ℓ − a(q) − ζ(ℓ) ]   (ask side; a(q) from F7)

with λ_fill from the queue-bracketed fill curve (pessimistic = full queue ahead
clears; moderate = pro-rata) and ζ(ℓ) the markout-measured adverse selection
conditional on a fill at level ℓ (fills at nearer levels are more informed —
ζ is level-dependent, which the additive δ_tox floor cannot express). Quote
ℓ* = argmax EV if max EV > outside option (0, or the rewards-band value).
The exponential fit survives only as a smoothing prior across levels. Note the
0.01→0.001 `tick_size_change` regime near extremes makes the grid
state-dependent — the discrete formulation handles this for free; the
continuous one does not know it happened.

## F9. v(t) = p̂(1−p̂) + max(0, holdout-λ_bin term) — WRONG (sign/structure)

For the martingale p̂ with Bernoulli terminal value, p̂(1−p̂) IS the total
remaining quadratic variation to resolution: E_t[∫_t^T λ_bin² ds] =
Var_t[p̂_T] = p̂(1−p̂) (verified numerically in F4: pre+in integrals = 0.250
exactly). Any unwind-before-T variance is a SUBSET of it. Adding a
positive-part λ_bin term double-counts risk. Fix: v = p̂(1−p̂) if holding to
resolution; v = ∫_t^{t+h} λ̂_bin² ds ≤ p̂(1−p̂) if unwinding at horizon h —
i.e. min-structure, never "+". (If the intent was model-error padding, label it
as that, not as risk decomposition.)

## F10. Dimensional audit — CONSISTENT, with two gaps

With all prices in $ of the $1 payout (prob ≡ price numerically), q in shares,
γ in $⁻¹ (CARA over dollars): γq·v is $/share ✓; γ/k dimensionless if the fill
curve is fitted with δ in $ ✓; δ_tox = Ê[|Δp̂|·1{burst}] is $ ✓; λ_bin is
prob/√s ✓. Magnitudes close: γ ≈ 4e-4 $⁻¹ gives 1-tick skew per 100 ATM shares
and δ* ≈ 1/k ≈ 1–2 c. Gaps: (i) no γ-calibration recipe in this plan (parent
program backs γ out of the inventory band — restate here, since v ≤ ¼ makes the
translation different); (ii) §2.2's "penalty is q²·p̂(1−p̂)" is missing the γ/2
(prose-level, but this repo punishes that).

## F11. Delta hedge δ = φ(d)·∂d/∂F — formula CORRECT; the remark under-describes what is unhedgeable

∂p̂/∂F = φ(d)·∂d/∂F with ∂d/∂F = 1/σ_eff pre-window, √3/(σ√r) in-window ✓
(note in-window hedge notional decays as r/w while probability-gamma grows).
Under-described: (a) hedging a DIGITAL near the money into expiry is the
textbook unhedgeable case — gamma ∝ (∂d/∂F)² ∝ 1/r, rebalance costs diverge at
4.5 bps taker exactly when the hedge is needed; delta-hedging only removes the
small-|move| linear risk far from the money, where risk is small anyway;
(b) the hedge is Binance-perp against a Chainlink-spot-aggregate settlement:
residual = perp-vs-spot basis (funding-linked, the measured +4.6 bps leg) ON TOP
of the aggregate-vs-Binance leg — a two-leg basis position the plan's one-line
remark doesn't decompose. Ablation-only status: appropriate. NOTED with fix to
the remark.

## F12. Basis layer — the load-bearing calculation; the plan's "F_t + basis correction" construction is the WEAK one

Setup: σ_eff(τ=300) = 8.6/11.5/17.2 bps (30/40/60% vol); at entry 2.4/3.2/4.8;
in-window at r=30: 0.84/1.13/1.69; r=10: 0.16/0.22/0.32.

(a) **Synthetic X̂ (Binance + calibrated constant basis), residual std 0.7 bps:**
ATM p̂ error = φ(0)·0.7/σ_eff = **1.6–3.2 c at window OPEN** (≥ 1.5 ticks, ~the
whole observed half-spread), and basis noise = 100% of remaining settlement
vol at **r ≈ 17/22/26 s** (60/40/30% vol); at r=30 the ATM p̂ error is already
16–33 c. Worse, the basis is a drifting level (−5.4→−5.8 bps over one 85 s
probe ⇒ ~1–1.4 bps per 300 s window), so even same-stream K-cancellation
leaves ~1+ bps of numerator error. Conclusion: **a synthetic-basis p̂ can never
quote near the money to tick accuracy at ANY τ in the window**, and is
model-noise-dominated (error > σ_eff) throughout r ≲ 20–25 s. The plan's basis
narrative treats this as a calibration refinement; it is a viability
constraint.

(b) **Correct construction (stream-anchored):** with the live
`crypto_prices_twap_sixty` feed in the quoting loop, anchor the LEVEL to the
stream and use Binance only for increments: Ŝ_t = X_t + (F_t − TWAP_w(F)_t)
(basis level cancels inside the bracket); locked part integrated from recorded
stream ticks (F2). Then the unpredictable basis innovation over the remaining
open window is not an error at all — it is genuine unspanned settlement
variance and belongs INSIDE σ_eff (σ_CL² = σ_F² + σ_⊥²); p̂ stays unbiased
w.r.t. our filtration. Residual exposure reduces to (i) stream latency
(~471 ms) and (ii) stream gaps (to 8 s): over a gap the maker is synthetic-only
again, i.e. case (a). So the pull rule acquires a new binding trigger the plan
lacks: **pull near-money quotes on stream staleness > ~2 s**, and fold
λ_bin·√(feed latency) into δ_tox for BOTH feeds (Binance and stream — two
latencies, plan names only one).

(c) Where basis dominates, answered precisely: never pre-window (0.7 <
2.4 bps min); in-window from r ≈ 17–26 s under (a); under (b) the analogous
ratio is √3·β/σ per unit-lag (β = basis innovation vol) — roughly CONSTANT in
r, so the stream-anchored model degrades gracefully instead of catastrophically.
β must be measured from the recorded stream-vs-Binance series (a variogram of
the basis increment vs lag — add to PM-E1).

## F13. μ̂ at 5-min horizon — mostly noise; default to 0 — UNSTATED-ASSUMPTION

For μ̂(τ−w/2) to move d by 0.1 at τ=300 requires |μ̂| ≈ 0.1·11.5 bps/260 s ≈
0.38%/day sustained — far beyond any signal this repo has ever validated at
ANY horizon (daily rank-IC ~0.03), let alone 5-min. An unregularized μ̂ mostly
injects variance into p̂: noise std_μ̂·260/σ_eff. Default μ̂=0; admit μ̂≠0 only
through the capped reservation-shift channel (Cartea–Wang), and state that
alpha enters ONCE (plan has both "p̂ embeds the alpha" and a ρ-cap — one entry
point must be named to avoid double-counting).

## F14. Φ misspecification — sign derived; it compounds F7 on the same side

5-min crypto returns are heavy-tailed. Variance-matched t₅ vs Gaussian:
f(0) = 0.490 vs 0.399 (+23% — ATM binary slope understated by Φ); shoulders
overstated (d=1: 0.127 vs 0.159); far tails understated ~4.5× (d=3: 0.0059 vs
0.0013). Consequences: (i) near d=0, Φ under-reacts to F moves — stale-quote
edge handed to snipers scales with the +23%; (ii) at |d| ≥ 2–3, Φ prices the
longshot at 0.1–0.6 c when the fat-tailed value is ~0.6–1+ c — selling
longshots at model price is systematically cheap, and F7's linearized CE
under-charges for exactly that inventory. Two independent approximations err on
the same side of the same trade (short-longshot). PM-E2's calibration by |d|
stratum will catch this empirically; the plan should pre-commit a fat-tailed
link (t-ν or empirical quantile map) as the F variant, not a post-hoc patch.

## F15. Small items

- **w hardcoded**: markets carry per-market stream metadata and a 30 s stream
  exists (`crypto_prices_twap_thirty`); all constants (τ−2w/3, r³/3w², w/2,
  √3) must be parameterized on window_s, not baked as 40 s/60 s. NOTED.
- BM vs GBM over 300 s: convexity ~½σ²τ ≈ 0.008 bps — negligible ✓.
- Down-book mirror (1−r) ∓ δ and pair-cost (bid_up + bid_down = 1 − 2δ < 1):
  both CORRECT given the verified unified book/mint-merge mechanics.
- Constant-σ-over-window assumption: unstated; vol bursts inside the window are
  handled operationally (burst-pull) but σ̂ staleness widens F14's tails —
  fine as v1 if stated.

---

## Triage

| # | severity | finding | fix |
|---|---|---|---|
| F4 | **MUST-FIX** | "danger = entry, not final seconds" is only the unconditional statement; ATM λ_bin = √3φ(d)/√r diverges (1.73× a snapshot binary), P(near-money at r=10s) ≈ 5%; τ_min-only pull is unsafe | pull rule = (|d|, r) surface: pull when φ(d)√(3/r)·√(reaction time) ≳ tick; rewrite §2.1 prose |
| F12 | **MUST-FIX** | basis math: synthetic X̂ p̂ carries ≥1.6–3.2 c ATM error at open and exceeds σ_eff inside r≈17–26 s; plan's "F_t + basis correction" is the weak construction | stream-anchored estimator (level from live TWAP topic, Binance for increments, locked from stream integral); σ_⊥ folded into σ_eff; NEW pull trigger on stream staleness; β-variogram added to PM-E1 |
| F9 | **MUST-FIX** | v(t) = p̂(1−p̂) + max(0, λ_bin term) double-counts: p̂(1−p̂) already equals total remaining QV (proved; integrals check to 0.250 exactly) | v = p̂(1−p̂) held-to-T, else ∫λ̂_bin² over unwind horizon; min-structure, never + |
| F8 | **MUST-FIX** | continuous-δ (1/γ)ln(1+γ/k) meaningless on a 2–4-tick grid (δ*≈1/k≈tick; 2-pt exponential fit; no queue; contradicts §2.4) | discrete per-level EV: argmax_ℓ λ_fill(ℓ,queue;bracket)·[P_ℓ − CE-quote − ζ(ℓ)]; exp fit demoted to smoother |
| F7 | SHOULD-FIX | linear r−γqv understates short-longshot reservation (~2 ticks at γ|q|~1) and exits [0,1]; exact CARA quotes are closed-form | use a(q), b(q) = (1/γ)ln g-ratios; note q±½ absorbed in δ* |
| F14 | SHOULD-FIX | Φ tails: f(0) −23%, longshot tails ~4.5× understated (t₅-matched); compounds F7 on short-longshot side | pre-commit fat-tailed link as PM-E2 variant |
| F13 | SHOULD-FIX | μ̂ needs ~0.4%/day-equivalent to matter; injects noise; alpha entry-point ambiguity (μ̂ vs ρ) | default μ̂=0; single named alpha channel |
| F6a | SHOULD-FIX | pre-open quoting has no formula; correct: Var_t[X_T−X_0] = σ²(T−w/3) for t≤−w (MC-verified) | add formula or state "no quotes before t=0" |
| F6b | SHOULD-FIX | boundary-read jitter ~0.7 bps at an 8 s gap ≥ σ_eff inside r≈20 s | carry the number in §2; aligns with data-loop #5/#9 |
| F10 | SHOULD-FIX | γ-calibration recipe absent; §2.2 penalty missing γ/2 | port parent inventory-band recipe; fix prose |
| F5 | NOTED | tie→Up negligible at 1e18 precision; H-PM1 wording overstates | downgrade wording; keep in E1 rule test |
| F11 | NOTED | delta formula right; digital-gamma unhedgeability + two-leg basis under-described | expand the remark; ablation status stands |
| F15 | NOTED | parameterize w (30 s stream exists); GBM/mirror/pair-cost all fine; state constant-σ assumption | one-line edits |
| F1–F3 | NOTED | all §2 variance/expectation formulas re-derived and MC-verified CORRECT (incl. the w/2 and the zero cross-term) | none |
