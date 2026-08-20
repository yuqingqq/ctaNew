# R2 — Taker-flow modeling, fill probability, queue dynamics, adverse selection

Scope: SOTA for the "monitor incoming taker flow, form a distribution" idea, plus
fill/queue modeling and fill-conditional (markout) accounting for a Binance USDM
perp MM. Data assumed: historical aggTrades (tick), forward-collected depth@100ms
+ bookTicker. No colocation.

## 0. Context constants (Binance USDM, checked 2026-08)

- VIP0 fees: maker +2.0 bp, taker +5.0 bp (10% off with BNB). **No maker rebate
  at VIP0** — rebates (~−0.5 bp) exist only in VIP/liquidity-provider programs.
- BTCUSDT: tick 0.1 USD ≈ 0.02–0.03 bp of price; spread ≈ 1 tick nearly always;
  top-of-book queues are hundreds of k$ (Albers et al. 2025). Spread capture on
  BTC is negligible vs fees; economics live in rebate + adverse-selection control,
  or in alts where spread/tick is meaningfully wide in bps.
- aggTrades fields: price, qty, timestamp(ms), `isBuyerMaker`. `isBuyerMaker=false`
  = aggressive BUY (taker buy hit the ask). One aggTrade = fills of one taker order
  at one price; group same-timestamp same-side aggTrades to reconstruct sweeps.

---

## 1. Hawkes processes for taker flow

### 1.1 Baseline multivariate Hawkes (Bacry–Mastromatteo–Muzy review)

Intensity of event type i (e.g. taker-buy, taker-sell):

    lambda_i(t) = mu_i + sum_j INT_0^t phi_ij(t-s) dN_j(s)

Stationarity iff spectral radius of the branching matrix ||INT phi_ij|| < 1.
Branching ratio n = fraction of events endogenously triggered.

- Calibration: (a) parametric MLE with exponential kernels
  phi(t) = alpha*beta*exp(-beta*t) — log-likelihood has an O(N) recursion
  R_k = exp(-beta*dt_k) * (1 + R_{k-1}), so MLE is cheap; convex if beta fixed on
  a grid. (b) Nonparametric: Wiener–Hopf / EM (slow, data-hungry).
- Stylized facts: kernels are power-law-ish over 4+ decades; markets are
  near-critical (n ≈ 0.9–1); multiple timescales needed if you insist on
  exponentials (sums of 2–3 exponentials fit well).
- Crypto numbers from the literature: BTC trade flow branching ratio ≈ 0.8
  (80% endogenous, same ballpark as FX/equities); fitted exponential decay
  beta ≈ 2.3 /s → memory ~0.4 s for the fast component; BTCUSDT trade
  intensities O(10) trades/s at baseline with 10–100x bursts.
- Refs: Bacry, Mastromatteo, Muzy, "Hawkes processes in finance"
  https://arxiv.org/abs/1502.04592 ; branching-ratio estimation
  https://arxiv.org/pdf/1706.03411 ; crypto endogeneity
  https://www.tandfonline.com/doi/full/10.1080/1351847X.2020.1791925

### 1.2 Queue-reactive Hawkes (Wu, Rambaldi, Muzy, Bacry, arxiv 1901.08938)

Marries Huang–Lehalle–Rosenbaum queue-reactive with Hawkes. Two variants; the
practical one (QRH-II) is multiplicative:

    lambda_l(t) = f_l(q_a(t), q_b(t)) * [ mu_l + sum_m INT phi_lm(t-s) dN_m(s) ]

8 event types (mid up/down, limit/cancel/market x bid/ask). Kernels = sum of 3
exponentials with FIXED betas (Bund: 60/1500/5500 /s; DAX: 40/2100/5200 /s) so
estimation is convex; QRH-I fits by MLE, QRH-II by least squares
(R(theta) = sum_l [INT lambda_l^2 dt − sum_k lambda_l(t_k)]), which permits
negative (inhibitory) kernels.

Findings that matter for quoting:
- 60–80% of intensity is the Hawkes (history) term — flow clustering dominates.
- The state multiplier f_l for market orders and mid moves is almost entirely a
  function of book imbalance I = (q_b − q_a)/(q_b + q_a); it varies by up to 3
  orders of magnitude across I on large-tick assets. Limit/cancel rates instead
  depend on own-queue size.
- Mean reversion of price comes from the state term, not kernel inhibition.

Ref: https://arxiv.org/abs/1901.08938 (full text: ar5iv).

### 1.3 State-dependent Hawkes (Morariu-Patrichi & Pakkanen, arxiv 1809.08060)

Kernels themselves switch with a discrete state (spread regime, imbalance
regime); state flips at events. MLE developed; empirical result: excitation is
stronger in disequilibrium (wide spread / large imbalance). Same operational
content as QRH ("condition intensity on state"), heavier machinery.
Ref: https://arxiv.org/abs/1809.08060

### 1.4 Compound (marked) Hawkes for sizes (arxiv 2312.08927)

12-dim Hawkes; each event carries a size mark kappa sampled from a per-event-type
distribution: Dirac spikes at round sizes {1,10,50,100,...} + geometric tail
(MLE); cancels sampled uniformly from resting order sizes. Key empirical result:
**order sizes do NOT feed back into intensities** (Hoeffding independence test
stat 0.00068) — intensity depends on event times only. Nonparametric kernel
estimation (Kirchner-style, multi-scale grids) selects power-law
phi(t) = alpha*(1+gamma*t)^(−beta) 100% of the time; in-spread limit orders and
opposite-side top cancels have inhibitory kernels; intraday seasonality handled
with 13 half-hour bins (2–3x multipliers).
Ref: https://arxiv.org/abs/2312.08927

### 1.5 Honest assessment for a 100 ms–1 s quoting decision

What the quoter actually consumes is (i) expected taker arrival rate and
direction per side over the next O(1 s), and (ii) how that rate scales with book
state. The literature itself says: exponential-kernel intensity with 2–3 fixed
timescales is a sufficient kernel family (QRH uses exactly that); size marks are
independent of intensity (separate size model); the dominant state variable is
top-of-book imbalance. Full multivariate MLE buys *simulator fidelity*
(inter-event distributions, queue distributions, impact curves) — valuable for a
backtest simulator, marginal for the real-time quote itself. Near-criticality
(n≈0.8–1) also makes fitted mu/alpha unstable day-to-day; intraday
nonstationarity (2–3x) swamps kernel-shape refinements.

**Calibration recipe on our data** (aggTrades → per-side taker intensity):
1. Events: group aggTrades by (timestamp, side) → market orders; keep size sum.
2. Per side s in {buy, sell}, maintain K=2 EWMA intensities, half-lives ~1 s and
   ~30 s, event-driven update at each arrival after decay
   `lam = lam*exp(-dt/tau); on event: lam += 1/tau` (this IS the Hawkes
   exponential-kernel intensity estimate with alpha=1, mu=0).
3. Optionally fit (mu, alpha_1, alpha_2) by MLE with betas fixed at those two
   half-lives (convex; O(N) recursion) per symbol per week; check branching
   n = sum alpha_k < 1 and stability of parameters before trusting them.
4. State multiplier: bucket bookTicker imbalance I into deciles; estimate
   f(I) = (taker-arrival rate | bucket)/(unconditional rate) by counting.
   Multiply. This is QRH-lite and captures its headline finding.
5. Size distribution: empirical per-side histogram with round-size atoms;
   resample for the simulator; do not couple to intensity (1.4).

---

## 2. Simpler alternatives practitioners use

- **EWMA intensity** (above): equivalent to a 1-exponential Hawkes without
  separating baseline from excitation. Captures clustering, bursts, and flow
  asymmetry — the three first-order phenomena. Rule of thumb supported by the
  QRH decomposition (60–80% endogenous, imbalance explains the rest): a 2-scale
  EWMA per side + imbalance multiplier captures ~90% of what a full Hawkes gives
  a *quoting* decision; the residual is cross-excitation fine structure (e.g.
  taker-buy → cancel-ask rates) that mostly matters in simulation.
- **Burst detection**: flag lam_fast / lam_slow > k (k≈3–5) or Poisson tail
  P(N_obs | lam_slow) < eps → pull or widen quotes. Cheap toxicity tripwire;
  complements, not replaces, the intensity estimate.
- **Volume clock / bulk-volume buckets** (Easley, López de Prado, O'Hara, "The
  Volume Clock" / VPIN, 2012): sample features every V contracts instead of every
  Delta-t; VPIN = mean(|V_buy − V_sell|)/V over last n buckets. On crypto we have
  exact taker signs from aggTrades, so skip bulk classification; use volume-clock
  sampling for feature stationarity and VPIN as a slow (minutes) toxicity regime
  variable. Do not expect VPIN to be a per-fill signal — it is a regime dial.
- **Trade-sign runs / long memory**: sign autocorrelation is power-law
  (Lillo–Farmer); a practical predictor of next-taker-side is an EWMA of signed
  flow (or signed volume share) — this doubles as the flow-direction input to
  fair-price skew. Runs-based features (current run length) add little beyond
  the EWMA.

---

## 3. Fill probability and queue position

### 3.1 Queue value (Moallemi–Yuan 2016)

Order value decomposition (their eq. 1) — the cleanest mental model in the field:

    V_t = alpha_t * (delta_t − AS_t)
    alpha_t = P(fill), delta_t = P_ask − P_efficient (liquidity premium,
    ~half-spread + rebate), AS_t = E[P_fill_time − P_t | fill]  (adverse selection)

With Poisson trades (rate mu, size dist f), price jumps (rate gamma, P(up)=p_J+,
mean up-jump J+), and proportional queue cancellations (rate eta), the value is
quasi-linear in delta: V(q, delta) = alpha(q)*delta − beta(q), with alpha, beta
solving Volterra integral equations in queue position q (their Thm 1; solvable
numerically; closed form for exponential trade sizes, Thm 3:
alpha(q) = p_J+ + mu(1−p_J+)/(mu+2gamma) * exp(−b q)). Only *ratios* gamma/mu,
eta/mu matter, plus f, impact lambda, p_J+, J+.

Empirics (BAC, NASDAQ ITCH): alpha(q) decreasing → p_J+ asymptote; AS(q)
increasing in q (back-of-queue fills come from bigger sweeps); **front-of-queue
vs average queue position is worth 0.21–0.26 ticks — order of the spread**.
Implications: join early; a deteriorated queue position on a large-tick book is
often worth less than zero (cancel/replace); day-to-day variation is driven by
gamma/mu (jump-to-trade ratio) — worth monitoring live.
Ref: https://moallemi.com/ciamac/papers/queue-value-2016.pdf

### 3.2 State-dependent fill probabilities (Lokin & Yu, arxiv 2403.02572)

Book = interacting queues; all intensities state-dependent (queue size, spread,
distance): limit lambda_Q(X), market mu_Q(X), cancel phi_Q(X). Own order at best
quote with q ahead = pure-death process (later arrivals queue behind you);
first-passage to 0 has Laplace transform (their Prop 2):

    g_hat(s) = PROD_{k=1}^{q} (mu_k + phi_k) / (mu_k + phi_k + s)

Fill-before-adverse-mid-move probability = P(eps_i < sigma_j ∧ spread-events),
computed by combining g_hat with the opposite-queue depletion transform (a
continued-fraction birth–death first-passage, their Prop 1) and Exp(2*Lambda_s0)
in-spread arrivals; numbers come out via numerical Laplace inversion. Deeper
levels covered but empirically >90% of executions happen at the best quote.

**Calibration is just conditional counting** (their Sec 5.1): with spread S and
distance delta, lambda_S(delta) = N_limit(delta,S)/T_S; mu_S = N_mkt(S)/T_S *
(mean market size / mean limit size); theta_S(delta) = N_cancel(delta,S)/(T_S *
avg outstanding depth) — i.e. cancel rate proportional to queue size. Rolling
5-day or same-weekday windows both worked on EUR/USD (LMAX); accuracy good.
This is directly implementable in pandas on our depth@100ms + aggTrades.
Ref: https://arxiv.org/abs/2403.02572

Simpler cousin used by practitioners (hftbacktest GLFT tutorial): fit
lambda(delta) = A*exp(−k*delta) by counting taker arrivals reaching depth delta
(log-linear regression over a trailing ~10 min window, refit every few s); A and
k then feed Guéant–Lehalle–Fernandez-Tapia optimal half-spread/skew. Good
default before the full birth–death machinery.
Ref: https://hftbacktest.readthedocs.io/en/latest/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html

### 3.3 Deep survival models (KANFormer, arxiv 2512.05734)

Time-to-fill as right-censored survival; dilated causal conv + Transformer +
KAN heads; inputs = LOB snapshots + agent actions + queue position; evaluated by
C-index / td-AUC / Brier on CAC40 futures; beats prior deep survival baselines.
Needs labeled per-order data and heavy fitting — research-grade, not needed for
v1. Ref: https://arxiv.org/abs/2512.05734

### 3.4 L2-only queue-position approximation (no order IDs)

We never see our queue position directly; hftbacktest's models are the standard
approximations (verified in source `hftbacktest/src/backtest/models/queue.rs`):

- **RiskAverseQueueModel**: cancellations always assumed behind you; position
  advances only on trades. Worst-case bound.
- **ProbQueueModel**: on a depth decrease of chg (net of trades:
  `chg = prev_qty − new_qty − cum_trade_qty`), split it around you:

      prob = f(back) / (f(back) + f(front))         # share removed BEHIND you
      front_new = front − (1−prob)*chg + min(back − prob*chg, 0)

  with f(x) = x^n (PowerProbQueueFunc; n=1 linear), f(x)=log(1+x) (Log...), and
  variants prob = f(back)/f(back+front) (…2), prob = 1 − f(front/(front+back))
  (…3). Trades always consume from the front; `cum_trade_qty` prevents
  double-counting trades as cancels.
- How wrong: truth is bracketed by [RiskAverse (all cancels behind), all-cancels-
  ahead]. On Binance perps cancel rates are huge relative to trade rates
  (equities analogue: eta/mu ~ 100x in Moallemi-Yuan's Table 1 — same order
  observed in crypto), so the bracket is WIDE and the choice of f materially
  changes backtest fill counts. Treat queue model choice as a robustness axis:
  run RiskAverse (pessimistic) and PowerProbQueueFunc3 n=3 (moderate); a strategy
  that only works under optimistic queueing is not real.
- Ground truth is cheap on Binance: place min-notional live orders and record
  actual time-to-fill/queue outcomes (Albers et al. did exactly this; their fill
  indicator was predicted with R^2 = 0.946 by a linear model on (liquidity ahead,
  opposite-queue size)). Use forward-collected own-order data to pick and calibrate f.
  Refs: https://hftbacktest.readthedocs.io/en/latest/order_fill.html ,
  https://arxiv.org/html/2502.18625v2

### 3.5 Fills are not exogenous (the central fact)

Albers, Cucuringu, Howison, Shestopaloff, "The Market Maker's Dilemma"
(arxiv 2502.18625) — live experiment, Binance BTCUSDT perp, 232,897 min-size
maker orders placed signal-free (Feb 2024 + Aug 2024):

- Fill probability and post-fill return are strongly NEGATIVELY correlated: if
  the next tick move is against your side you fill w.p. ~1.
- 1 s markouts, all negative: best case (front of queue, favorable imbalance)
  −0.058 bp; back of queue −0.775 bp; adverse imbalance −0.5..−0.8 bp regardless
  of queue position.
- Naive symmetric quoting: −60% over 3.17 days, Sharpe −109. Imbalance-only
  maker strategies: negative. Imbalance-taker: +1 bp gross, −1.96 bp after fees.
- The exploitable residual: predicting *reversals* (imbalance says down but
  price mean-reverts) — logistic regression 68.6% / RF 70.2% accuracy on
  positive-markout fills at high fill probability. Feature groups: price
  dynamics (vol, amplitude, VWAP returns), trade volumes, momentum
  (autocovariance), queue imbalance.

Consequences for us: (1) backtests must trigger fills from actual trade prints
crossing our price (never "price touched = filled"), with a queue model, or
markout-conditional fill logic; (2) PnL accounting must be markout-based, not
spread-capture-based; (3) quoting logic must be conditioned on flow/imbalance
state — symmetric quoting is a money pump for takers.

---

## 4. Markout measurement from aggTrades alone

Every aggressive trade in aggTrades IS someone's passive fill. So the
*population* maker markout curve needs no own orders:

    For aggTrade j: price p_j, side s_j = +1 (taker buy) if isBuyerMaker=false.
    The counterparty maker was SHORT at p_j if s_j=+1 (their ask lifted).
    Maker markout at horizon h:  MO_j(h) = −s_j * (m_{t_j+h} − p_j)
    (equities convention: realized spread = 2*s_j*(p_j − m_{t+h}); same object).

- Mid m: forward data → mid or microprice from bookTicker (microprice
  (b*Q_a + a*Q_b)/(Q_a+Q_b) is the better fair-price proxy). Historical
  aggTrades-only → proxy m_t by the midpoint of the most recent taker-buy and
  taker-sell prices, or next-opposite-side trade price; both are noisy and
  slightly bias markouts toward zero at h < 1 s — report historical curves from
  h ≥ 1 s only, use forward-collected data below 1 s.
- Curve: h ∈ {100ms, 250ms, 500ms, 1s, 2s, 5s, 30s, 2min}, size-weighted and
  equal-weighted, per symbol.
- Conditioning (this is the actual research output): markout by (i) imbalance
  decile at fill, (ii) taker-intensity state (lam_fast/lam_slow), (iii) sweep
  size / number of price levels consumed, (iv) time-of-day. The conditional
  markout surface E[MO | state] is simultaneously (a) the adverse-selection
  input to quote skew and (b) the toxicity label generator for Sec 5.
- Anchors to sanity-check against: BTC perp 1 s maker markouts −0.06..−0.8 bp
  (Albers); markouts must beat (maker fee − spread/2) for viability: at VIP0
  +1.8–2 bp maker fee, BTCUSDT is DOA; the search space is high-spread alts
  and/or fee-reduced tiers.

No dedicated "crypto perp markout curve" literature beyond Albers et al. exists
as of 2026-08; the equities realized-spread literature (Hendershott et al.) is
methodological background only.

---

## 5. Toxic flow detection (Cartea, Duran-Martin, Sánchez-Betancourt, arxiv 2312.05827)

Setting: FX broker sees client IDs; decides internalize vs externalize.
- **Toxicity label**: trade at ask S^a_t is toxic on horizon G if
  tau+ = inf{u: S^b_u > S^a_t} <= t + G — counterparty could unwind at profit.
- **Features (183)**: 15 client-specific (inventory, past toxicity share, ...) +
  168 = 8 LOB stats x 3 clocks (time/volume/transaction) x 7 lookbacks.
- **PULSE**: Bayesian online logistic-output MLP; only last layer w (dim L) and
  a d-dim subspace z of hidden params (psi = A z + b, A from SVD of SGD warmup
  iterates) are updated, EKF-style, per observation (x_n, y_n), sigma = sigmoid:

      nu_n    = nu_{n-1} + Sigma_{n-1} h_n (y_n − sigma(nu' h_n))        # last layer mean
      Sigma_n^{-1} = Sigma_{n-1}^{-1} + sigma'(nu' h_n) h_n h_n'          # last layer precision
      mu_n    = mu_{n-1} + Gamma_{n-1} grad_z h_n (y_n − sigma(nu' h_n))  # subspace mean
      Gamma_n^{-1} = Gamma_{n-1}^{-1} + sigma'(nu' h_n) grad_z h_n grad_z h_n'

  (h_n = g(A mu + b; x_n)). ~120 dof updated, <1 ms/step. AUC ≈ 0.65–0.70 at
  30 s horizon vs logistic 0.52, RF 0.63 (RF retrained offline). Decision rule:
  internalize iff p_toxic + Phi*Q > threshold (inventory-adjusted cutoff).

**CLOB-maker translation** (no client IDs on Binance — toxicity attaches to
*fills/states*, not clients):
- Label: y_j = 1 if maker markout MO_j(G) < −c (c ≈ fee or 0), G ≈ 1–30 s.
  Sec 4 generates unlimited labeled fills from aggTrades without trading.
- Features: market state at fill (imbalance, lam_fast/lam_slow per side, signed
  flow EWMA, recent vol, spread, sweep size, levels consumed, size vs queue,
  time-of-day) — i.e. the paper's clock-based block; the client block is
  unavailable, which is precisely the part PULSE's edge came from (per-client
  memory). Expect a smaller gap over plain online logistic here.
- Use: p_toxic (or directly E[MO|state]) sets a per-side quote skew/widen and a
  pull-quote trigger; inventory term enters exactly as their Phi*Q.
- Verdict: adopt the *framing* (online-updated per-fill toxicity feeding skew);
  implement first as online logistic regression / recursive ridge on the Sec 4
  conditional-markout features; PULSE itself is a DEFER (nonstationarity may
  eventually justify it; equations above are complete if so).
  Ref: https://arxiv.org/abs/2312.05827

---

## 6. Component decisions

| Component | Verdict | Rationale / recipe |
|---|---|---|
| Per-side taker intensity: event-driven EWMA, 2 half-lives (~1 s, ~30 s) | **ADOPT** | = exponential-kernel Hawkes estimate; captures clustering + bursts; O(1) updates |
| Imbalance-bucket rate multiplier f(I) (QRH-lite) | **ADOPT** | QRH's dominant effect; conditional counting, no MLE |
| Signed-flow EWMA (direction) + burst tripwire lam_fast/lam_slow | **ADOPT** | flow-direction input to fair price; cheap toxicity gate |
| Full multivariate Hawkes MLE (exp kernels, fixed betas) | **DEFER** | only for backtest simulator fidelity; revisit when simulator is the bottleneck |
| Power-law kernels, nonparametric Hawkes, state-dependent-kernel MLE | **REJECT** | complexity >> quoting value at 100 ms–1 s; params unstable near criticality |
| Compound-Hawkes size marks | **REJECT** (use empirical per-side size histogram; sizes ⟂ intensity per 2312.08927) |
| lambda(delta)=A exp(−k delta) fit (GLFT-style), trailing 10 min | **ADOPT** | fill-rate-vs-depth input to spread/skew; log-linear regression |
| State-conditioned fill prob via pure-death product (Lokin–Yu Prop 2) | **ADOPT (v2)** | implementable by conditional counting + Laplace inversion; start with A,k model |
| Moallemi–Yuan queue-value V=alpha(q)delta−beta(q) | **ADOPT (concept)** | drives join/cancel logic; monitor gamma/mu, eta/mu live; full Volterra solve optional |
| KANFormer / deep survival fill models | **REJECT** for now | needs labeled per-order data; overkill |
| hftbacktest ProbQueueModel (f3, n≈3) + RiskAverse as pessimistic bound | **ADOPT** | only honest L2-queue treatment: report both; calibrate f with live min-size probe orders |
| "Touch = fill" backtest fills | **REJECT** | fills cluster exactly on adverse moves; Sharpe −109 naive quoting (Albers) |
| Markout accounting harness from aggTrades (Sec 4) | **ADOPT (first deliverable)** | zero-cost, generates the E[MO|state] surface + toxicity labels before any trading |
| Per-fill toxicity model → quote skew (online logistic on Sec 4 features) | **ADOPT** | CLOB version of PULSE framing |
| PULSE (subspace-EKF neural) | **DEFER** | equations in Sec 5; justify only if online logistic demonstrably decays |
| VPIN / volume clock | **ADOPT (regime dial only)** | slow toxicity regime; exact signs available, skip bulk classification |

**Build order implied**: (1) Sec 4 markout harness on historical aggTrades →
per-symbol viability screen vs fees (kills BTCUSDT at VIP0 immediately, ranks
alts); (2) intensity/imbalance state features + conditional markout surface;
(3) A,k fill model + queue-model-bracketed backtest; (4) toxicity-skewed quoting.
