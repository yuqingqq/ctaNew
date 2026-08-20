# R1 — Fair value & short-horizon price prediction (MM literature sweep)

Agent R1 of 4. Date: 2026-08-19. Scope: micro-price, OFI, DL-LOB verdict,
trade-only estimators, propagator/transient-impact fair value.
Data context: Binance USDM perps; historical tick aggTrades (exact taker sign
via `isBuyerMaker`), 30s bookDepth bands (±0.2/1/2/3/5%, 2023→), forward L2@100ms
+ bookTicker starting now. No colocation; retail→VIP fees.

---

## 1. Stoikov micro-price (SSRN 2970694)

### 1.1 Construction

Definition: the micro-price is the long-horizon martingale expectation of the mid,

```
P_micro(t) = lim_{i→∞} P^i_t,     P^i_t = E[ M_{τ_i} | F_t ]
```

where τ_1..τ_i are the (random) times of the next i mid-price changes and
F_t = (M_t, I_t, S_t) with M = (P_b+P_a)/2, S = (P_a−P_b)/2,
I = Q_b/(Q_b+Q_a) (top-of-book imbalance). Two assumptions:
(A1) (M, I, S) is Markov; (A2) dynamics independent of price level, i.e.
E[M_{τ1} − M_t | M,I,S] = g1(I,S). Then

```
P^i_t = M_t + Σ_{k=1..i} g_k(I_t, S_t)
g1(I,S)   = E[ M_{τ1} − M_t | I, S ]           (first mid-change edge)
g_{i+1}   = E[ g_i(I_{τ1}, S_{τ1}) | I_t, S_t ] (recursion through the chain)
```

Discretize: I into n buckets, S into m tick values, state X = (I,S) ∈ {1..nm},
mid changes k ∈ K = {k : 0 < |k| ≤ 2m} (in half-ticks). Build the absorbing
Markov chain T = [[Q, R1],[0, Id]] from event-to-event transition counts:

```
Q_ij  = P( ΔM=0 ∧ X_{t+1}=j | X_t=i )      transient (book shuffles, no mid move)
R1_ik = P( M_{t+1}−M_t = k  | X_t=i )      absorb into a mid change of size k
R2_ik = P( ΔM≠0 ∧ I_{t+1}=k | I_t=i )      absorb into post-move imbalance state
```

Then, with k the vector of jump sizes:

```
g1 = (Id − Q)^{-1} R1 k               (first-step adjustment)
B  = (Id − Q)^{-1} R2                 (state-to-state operator across mid changes)
G* = P_micro − M = g1 + B g1 + B² g1 + ...   (iterate x ← g1 + B x to convergence)
```

Convergence (Stoikov Thm 2): B strictly positive, B^k → W (unique stationary
distribution) and **W g1 = 0** (adjustments mean zero under the stationary law).
Enforced in practice by symmetrizing: g1 antisymmetric in imbalance
(g1(i_I,i_S) = −g1(n−i_I,i_S)) and B_{(i_I,i_S),(j_I,j_S)} = B_{(n−i_I,i_S),(n−j_I,j_S)}.
With symmetrization the geometric series converges (sub-dominant eigenvalues < 1);
Stoikov also gives the spectral form G* = Σ_{i≥2} (eigen-projection of B) g1.
Empirics (BAC large-tick / CVX small-tick, Mar 2011): G*(I,S) estimated on 1 day
matches 1/5/10-min conditional mid-drift out of sample for a month; adjustment is
horizon-independent and stays inside [bid, ask].

### 1.2 Recipe on our data

- Feed: bookTicker stream (best bid/ask price+size) — this alone suffices.
  Sample in **event time** (every L1 change), not wall-clock.
- Majors (BTC/ETH perps): spread pinned at 1 tick most of the time → drop the
  spread dimension. **Imbalance-only chain**: n = 10 buckets, states = 10,
  K = {−1,+1} ticks. Q, R1, R2 are 10×10-ish counts; a day of BTCUSDT bookTicker
  (~10⁶–10⁷ events) is far more than enough. When S > 1 tick (vol bursts),
  either add a second spread state or fall back to weighted mid.
- Alts (multi-tick spreads): keep (I,S) grid, cap m at ~4 tick-states, pool the
  tail. Sparse cells: shrink G* toward the weighted-mid adjustment S·(2I−1).
- Re-estimate rolling (daily/weekly); transition matrices are regime-dependent
  (vol). Optionally condition on a vol bucket as a third (coarse) state.
- Sanity check: G*(I) must be monotone in I, antisymmetric, |G*| ≤ S.

### 1.3 Failure modes

- **Manipulable input**: L1 imbalance is spoofable; crypto books flicker. Cap
  effective size (e.g. min(Q, q95)) or use depth within a fixed bps band as
  Q_b/Q_a once L2@100ms accumulates (Blakely 2024 below = the same idea learned).
- **Markov/level-independence violations**: trends make (I,S) insufficient;
  micro-price is a *fair-value*, not an alpha — it removes bid-ask-bounce bias,
  it does not forecast beyond the imbalance information set.
- Non-stationary transition matrices; sparse extreme states; latency (martingale
  in event time ≠ realizable at 100ms+RTT).

Refs: [SSRN 2970694](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694);
slides [imperial.ac.uk Gatheral60](https://www.ma.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf)
(formulas above transcribed from these); code [github.com/sstoikov/microprice](https://github.com/sstoikov/microprice).

---

## 2. Order flow imbalance (OFI)

### 2.1 Cont–Kukanov–Stoikov OFI (arXiv 1011.6402)

Per L1 update n, the supply/demand contribution:

```
e_n = 1{P^B_n ≥ P^B_{n−1}} q^B_n  −  1{P^B_n ≤ P^B_{n−1}} q^B_{n−1}
    − 1{P^A_n ≤ P^A_{n−1}} q^A_n  +  1{P^A_n ≥ P^A_{n−1}} q^A_{n−1}

OFI_k = Σ_{n∈(t_{k−1},t_k]} e_n          ΔP_k = β·OFI_k + ε_k   (ΔP in ticks)
```

(covers limit adds, cancels, and trades symmetrically; a market sell ≡ cancel-buy
of equal size). Key numbers (50 US stocks, TAQ Apr 2010, Δt = 10s):
**contemporaneous R² ≈ 65%** avg (35–60% even excluding price-changing events);
quadratic term insignificant (65→68%) — impact is *linear* in OFI; trade
imbalance alone R² = 32% and becomes insignificant next to OFI (t-stat drops
4×). Impact coefficient β_i = c/AD_i^λ with λ ≈ 1: **slope = inverse depth**
(stylized model ΔP = OFI/2D). R² grows with Δt; results stable from ~0.5s to
10 min.

Caution: 65% is **contemporaneous** (explains, doesn't predict). Predictive R²
of lagged OFI on future returns is 1–2 orders of magnitude smaller and decays
within seconds–minutes (see 2.3).

### 2.2 Multi-level / integrated OFI

- **MLOFI** (Xu–Gould–Howison, [arXiv 1907.06230](https://arxiv.org/abs/1907.06230)):
  extend e_n level-by-level to depth m → vector OFI. On 6 Nasdaq stocks,
  out-of-sample fit improves with *every* additional level (diminishing returns);
  normalize each level by its average depth.
- **Integrated OFI** (Cont–Cucuringu–Zhang,
  [arXiv 2112.13213](https://arxiv.org/abs/2112.13213)): depth-normalize OFIs at
  levels 1..10, take the **first principal component** across levels (weights
  l1-normalized) → single scalar; beats best-level OFI for contemporaneous
  impact. Cross-asset: contemporaneous cross-impact adds ~nothing once
  integrated OFI is in; **lagged cross-asset OFIs do improve forecasts of
  future returns**, but the effect lives at short horizons and decays fast.
  (Matches our own repo finding of a 5–15 min BTC→alt flow lead.)

### 2.3 Trade-flow imbalance (TFI) and horizons — crypto twist

- TFI_k = Σ signed taker volume in (t_{k−1}, t_k] — exact from aggTrades
  (`isBuyerMaker`), no Lee–Ready needed.
- **Silantyev 2019** (XBTUSD, BitMEX, [Springer](https://link.springer.com/content/pdf/10.1007/s42521-019-00007-w.pdf)):
  in crypto perps **TFI explains contemporaneous price changes better than L1
  OFI**, especially at ≥1-min bins — the reverse of equities. Plausible cause:
  crypto L1 quote churn/spoof noise. Implication: weight the trade leg heavily;
  treat quote-OFI as a complement, not the core.
- **Kolm–Turiel–Westray 2023** ("Deep order flow imbalance",
  [SSRN 3900141](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141),
  Math. Finance 33(4)): 115 Nasdaq names; stationary **order-flow inputs beat
  raw LOB states** across architectures; and the **effective forecast horizon
  ≈ 2 average price changes** ahead — on BTCUSDT perp that is seconds, i.e.
  exactly the quote-resting horizon an MM needs, no more.

### 2.4 arXiv 2411.13594 (fetched, summarized)

Blakely, "High resolution microprice estimates … hyperdimensional Tsetlin
machines". Error-corrects the (I,S) micro-price with features = volume shares at
price ranks 1..L both sides, spread, last tick move; predicts a correction class
in {−2,−1,0,+1,+2} ticks vs the price N information-bars ahead; L2 error vs
future price improves **10–20% on TSLA**, less/noisier on a small-cap (Databento
L3, Sep 2024). Takeaway is model-agnostic: **deeper-rank imbalances carry
correction info beyond L1 micro-price** — reimplement as GBM/logit on the same
features once our L2@100ms history accumulates. The Tsetlin machinery itself:
skip. [arXiv 2411.13594](https://arxiv.org/abs/2411.13594).

---

## 3. Deep learning on the LOB — verdict

- **DeepLOB** (Zhang–Zohren–Roberts, [arXiv 1808.03668](https://arxiv.org/abs/1808.03668)):
  CNN (spatial LOB structure) + Inception + LSTM on 100×40 snapshots; strong
  FI-2010 F1 and claimed transfer across LSE names.
- **LOBCAST benchmark** (Prata et al, [arXiv 2308.01915](https://arxiv.org/abs/2308.01915)):
  re-ran **15 SOTA DL LOB models**; "all models exhibit a significant
  performance drop when exposed to new data" — the FI-2010 leaderboard does not
  survive fresh data; profitability analysis unconvincing.
- **LOBFrame** (Briola et al, [arXiv 2403.09267](https://arxiv.org/abs/2403.09267)):
  Nasdaq LOBSTER-grade data; efficacy is **stock-microstructure-dependent**, and
  "high forecasting power does not necessarily correspond to actionable trading
  signals" (their transaction-level practicality metric).
- **Crypto-specific** ([arXiv 2506.05764](https://arxiv.org/abs/2506.05764),
  BTC/USDT Bybit L2 @100ms–seconds): logistic regression / XGBoost with good
  preprocessing **match or beat DeepLOB-class nets**; "better inputs matter more
  than stacking another hidden layer".
- Kolm et al (2.3) says the same from the other side: representation (OFI) >>
  architecture.

**Verdict for a solo operator: REJECT DL-LOB.** The replicated-everywhere result
is that the edge is in stationary flow features; linear/GBM on OFI/TFI/micro-price
captures it with orders-of-magnitude less tuning, latency, and overfit surface.
Revisit only after a linear stack is live and profitable.

---

## 4. What is estimable from OUR data

### 4a. aggTrades only (full history → backtestable now)

| Estimator | What it gives | Notes |
|---|---|---|
| TFI (signed taker volume per bin) | contemporaneous+short-lag impact factor | exact signs; best crypto flow variable (2.3) |
| Sign autocorrelation C(ℓ) | flow persistence, sign predictor ε̂ | power-law tail; input to propagator + HDIM |
| Propagator G(ℓ) (§5) | transient-impact fair-value adjustment | full recipe below; trade event-time |
| Hasbrouck VAR (1991 JF) | permanent vs transient impact split | bivariate VAR of quote revision r_t and signed trade x_t, 5–10 lags, OLS; permanent impact = long-run IRF; needs a mid — proxy by trade-price midquote or use forward bookTicker era |
| Roll / trade-based spread | effective spread w/o book | cov(Δp_t, Δp_{t−1}) = −s²/4 |
| Volume clock | event-time base for all of the above | bucket by traded volume, not seconds |

Not estimable trade-only: micro-price, quote-OFI, queue dynamics.

### 4b. 30s bookDepth bands

**Not a fair-value input.** 30s cadence and ±0.2% band (=20 bp) vs BTC perp
spread ~0.01 bp — 3 orders of magnitude too coarse for quoting. Use only for
slow context: liquidity/capacity regime, depth-normalization of β (2.1), symbol
selection. (Consistent with this repo's L2 pilot conclusion: cost/capacity
realism, not alpha.)

### 4c. L2@100ms + bookTicker (accumulating from today)

Everything above plus: micro-price (§1, bookTicker is sufficient and is the
priority stream to keep robust), snapshot-differenced OFI at the 100ms grid
(differencing snapshots yields the *net* e_n sum per interval — that is the
standard practical OFI; intra-interval churn is lost but CKS's own robustness
across Δt says this is fine), MLOFI/integrated OFI, deeper-rank micro-price
corrections (2.4), queue-position estimates (R2/R3 scope). Start the estimators
shadow-computing on day 1; micro-price needs ~1 day of data, integrated OFI ~
weeks, DL never (see §3).

---

## 5. Propagator / transient impact (Bouchaud) — the "simulated flow impact" fair value

This is the academically-grounded version of "mid + expected impact of flow".

### 5.1 Model (TIM1)

Trade event-time t (one tick per aggTrade or per volume bucket), taker sign
ε_t ∈ {±1}, mid m_t before trade t. Return form (Taranto et al 2016, Eq. 2):

```
r_t ≡ m_{t+1} − m_t = Σ_{ℓ≥0} 𝒢(ℓ) ε_{t−ℓ} + η_t,      𝒢(ℓ) ≡ G(ℓ+1) − G(ℓ),  G(ℓ≤0) ≡ 0
```

equivalently m_t = Σ_{t'<t} G(t−t') ε_{t'} + noise: each trade impacts by G(1)
immediately, decaying along G(ℓ). Volume: multiply ε by f(v); empirically
impact is strongly concave — use f(v) = v^δ with δ ≈ 0.1–0.3 or ln(1+v/v̄),
fit δ by regressing single-trade mid response on v. G(ℓ) decays as a power law
ℓ^{−β} with **β = (1−γ)/2** where C(ℓ) ~ ℓ^{−γ} is the sign ACF (equities
γ ≈ 0.5 → β ≈ 0.25). Decaying G ⇔ "asymmetric liquidity": a trade following
same-sign flow impacts less.

### 5.2 Estimation recipe (exact, from Taranto et al)

Measure from aggTrades (+mid from bookTicker when available; historical fallback:
mid proxied by next-trade prices — noisier, adds an HF noise term already in the
model as D_HF):

```
R(ℓ) = E[ (m_{t+ℓ} − m_t) · ε_t ]        response function
C(ℓ) = E[ ε_t ε_{t+ℓ} ]                  sign autocorrelation
S(ℓ) = E[ r_{t+ℓ} · ε_t ]                differential response;  R(ℓ) = Σ_{0≤i<ℓ} S(i)
```

Then G solves the linear (Toeplitz) system — preferred, boundary-robust form:

```
S(ℓ) = Σ_{n≥0} 𝒢(n) C(n−ℓ),   ℓ = 0..L        (solve least-squares for 𝒢(0..L))
G(ℓ) = Σ_{n=0}^{ℓ−1} 𝒢(n)
```

(direct form: R(ℓ) = Σ_{0<n≤ℓ} G(n)C(ℓ−n) + Σ_{n>0}[G(n+ℓ)−G(n)]C(n).)
Practicalities: L ≈ 512–2048 trade lags; estimate per symbol per month; impose
smoothness by fitting G(ℓ) = g0·(ℓ0+ℓ)^{−β} after the raw solve; validate by
predicting the *negative-lag* response R(−ℓ) and the signature plot D(ℓ) =
E[(m_{t+ℓ}−m_t)²]/ℓ — TIM1 gives closed forms for both; mismatch there = flow
adapts to price (go HDIM).

### 5.3 HDIM and the sign predictor

Identity rewrite: r_t = G(1)(ε_t − ε̂_t) + η_t with
ε̂_t = −Σ_{ℓ>0} [𝒢(ℓ)/G(1)] ε_{t−ℓ}: only the **surprise** in flow moves price,
permanently. If signs follow a DAR process (ε_t copies a past sign at lag ~λ_ℓ
with prob ρ), TIM1 ≡ HDIM1 and ε̂_t = (2ρ−1) Σ_ℓ λ_ℓ ε_{t−ℓ}; λ solvable from
C(ℓ) via Yule–Walker C(ℓ) = (2ρ−1) Σ_n λ_n C(ℓ−n). Empirically HDIM fits
slightly better, most visibly on large-tick names — relevant for BTC/ETH perps
(1-tick books). Fit ε̂ as a plain AR on signs; it doubles as your
fill-toxicity flag.

### 5.4 Market-maker usage — fair value with expected flow impact

Expected mid drift h trades ahead, given past flow (rigid-flow TIM view):

```
E_t[m_{t+h}] − m_t  =  Σ_{ℓ≥0} [G(ℓ+h) − G(ℓ)] ε_{t−ℓ} f(v_{t−ℓ})     (decay of past impact)
                     + Σ_{k=1}^{h} G(h−k+1) · E_t[ε_{t+k}] · f̄        (impact of predicted flow)
```

with E_t[ε_{t+k}] from the sign predictor (§5.3, iterate the AR). Quote around
**F_t(h) = m_t + the above**, h = your expected quote-resting time in trades.
Interpretation: after a burst of buys, the first term is negative (transient
impact will partially revert — lean *into* fading it), while persistent-flow
prediction (second term) leans *with* the flow; G's shape sets the net. As
h→∞ with fully transient G: F_∞ = m_t − Σ_ℓ G(ℓ) ε_{t−ℓ} f(v_{t−ℓ}) — strip
the entire un-decayed transient component from the mid. In HDIM form the same
object is "mid consistent with the flow predictor", which is a martingale —
i.e. this estimator removes exactly the adverse-selection-by-flow bias.
Combine additively with the micro-price book adjustment (§1); they use disjoint
information (trades vs quotes) and both are anchored at the mid.

Refs: Bouchaud–Gefen–Potters–Wyart 2004 ([arXiv cond-mat/0307332](https://arxiv.org/abs/cond-mat/0307332));
Taranto–Bormetti–Bouchaud–Lillo–Treccani 2016 ([arXiv 1602.02735](https://arxiv.org/abs/1602.02735)
— equations §5.2 transcribed from this); Bouchaud–Bonart–Donier–Gould, *Trades,
Quotes and Prices*, CUP 2018, ch. 13 (propagator), ch. 16 (HDIM).

---

## 6. Component recommendations

| Component | Call | One-line justification |
|---|---|---|
| Imbalance-only micro-price from bookTicker (majors) | **ADOPT** | Exact fit for 1-tick pinned books; 1 day of data suffices; closed-form, fast |
| (I,S)-grid micro-price (alts, multi-tick) | **ADOPT** | Same machinery, add spread state; shrink sparse cells to weighted mid |
| TFI from aggTrades (volume clock) | **ADOPT** | Crypto evidence says trade flow > quote flow; exact signs; years of history to calibrate |
| Propagator G(ℓ) fair-value adjustment (§5.4) | **ADOPT** | The user's "simulated flow impact", rigorous; estimable trade-only → backtestable on full aggTrades history today |
| L1 OFI (snapshot-diff, 100ms grid) as regression feature | **ADOPT** | Cheap, linear, proven; but expect crypto attenuation vs TFI |
| HDIM sign-predictor ε̂ (AR on trade signs) | **ADOPT** | Needed by §5.4 anyway; doubles as toxicity gate on fills |
| Hasbrouck VAR permanent/transient split | **ADOPT (diagnostic)** | One afternoon of work; sizes adverse selection for R2/R3's spread model, not a quoting signal itself |
| Multi-level / integrated (PCA) OFI | **DEFER** | Real but incremental; needs weeks of L2@100ms history first |
| Lagged cross-asset OFI (BTC→alts) | **DEFER** | Positive evidence (CCZ + our own 5–15min lead) but second-order; add after single-asset stack works |
| Deeper-rank micro-price correction (2411.13594 idea, as GBM) | **DEFER** | 10–20% error reduction plausible; needs L2 history; skip the Tsetlin machinery |
| DeepLOB / TransLOB / any DL-LOB | **REJECT** | Benchmarked collapse on new data (LOBCAST); representation beats architecture (Kolm); latency+overfit cost for a solo operator |
| FI-2010-derived anything | **REJECT** | Dataset's leaderboard doesn't transfer even within equities |
| 30s bookDepth bands as fair-value input | **REJECT** | 20bp band / 30s vs 0.01bp spread / ms decisions; keep for liquidity context only |

**Integration sketch**: fair value F_t = M_t + G*(I_t[,S_t]) + propagator term
(§5.4) [+ β·OFI_recent]. Each term is linear-estimable, PIT-safe, and separately
validatable against forward mid at h ≈ 2 mid-changes (the Kolm effective
horizon). Quote bid/ask around F_t; skews and spread width are R2/R3 scope.
