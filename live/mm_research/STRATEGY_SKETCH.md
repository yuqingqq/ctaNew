# STRATEGY_SKETCH — P-2026-002 HF Market Making (canonical synthesis)

Review/synthesis of R1–R4 briefs, 2026-08-19. This is the single reference the
experiment-design phase (EXPERIMENT_PLAN.md) builds against. Sources:
`R1_FAIR_VALUE.md`, `R2_FLOW_FILLS.md`, `R3_QUOTING_INVENTORY.md`,
`R4_PRACTICE_ECONOMICS.md`, `PROGRAM.md`. Conflicts between briefs are resolved
in §3; where a brief overclaims, the claim is flagged here rather than adopted.

---

## 1. One-page summary

1. The strategy is a flow-aware passive quoter: estimate a fair value from book
   state and recent signed taker flow, post bid/ask around it with
   inventory/funding/toxicity skews, and account for profit exclusively via
   fill-conditional markout — never gross spread capture.
2. Fills are endogenous: you are filled precisely when flow is against you
   (Albers et al., live Binance BTC-perp experiment: naive symmetric quoting
   Sharpe −109; 1 s maker markouts −0.06 to −0.8 bp, all negative).
3. On Binance majors the spread (0.026–0.037 bps time-weighted, BTC/ETH) is
   40–70x smaller than the VIP0+BNB maker fee (1.8 bps): standalone touch-MM on
   liquid Binance names is arithmetically dead at any solo tier — REJECT, no
   model can fix it.
4. **Variant A (first deployment target): passive-execution overlay for the
   existing XS alpha book on Binance.** The repo capstone says the XS edge is
   ~+1 OOS Sharpe at ≤8 bps RT cost and ~0 at retail ~24 bps; passive entry/exit
   with the full MM stack is market making against a guaranteed uninformed
   client (ourselves), with no fee wall to clear — target effective RT cost
   2–4 bps + adverse selection + chase cost, gate at ≤8 bps.
5. Honest economics of A: even at full success the underlying book is marginal
   (capstone CI spans 0 at ≤8 bps); the overlay converts a dead book into a
   fundable-marginal one and is still the highest-EV first use of the stack
   because the client flow is captive and the test is cheap.
6. **Variant B (standalone MM): wide-spread names only.** Binance: full spread
   ≥ ~4 bps AND ≥VIP3+BNB (1.08 bps maker) — the tier needs $100M/30d volume,
   so near-term Binance-standalone reduces to new-listing LP promos (0.5 bps
   rebate). Hyperliquid: maker 1.5 bps base / 1.2 with modest staking, wider
   spreads (3–8 bps on mid/tails), no colocation moat, same Tokyo latency for
   everyone — the primary standalone candidate, DEFERRED until forward HL
   spread/markout data exists.
7. One quoting engine serves both variants: GLFT ergodic closed form (constant
   spread + linear inventory skew) quoted around fair value
   F = mid + microprice adjustment + propagator flow term, with alpha as a
   capped reservation-price shift, funding folded in as inventory carry, and
   one-sided withdrawal at signal extremes.
8. Adverse selection is handled at three parameter-level hooks (skew from the
   markout surface, widen on bursts, pull on toxic-state flags) — no new
   solver; RL and DL-LOB are rejected for this program.
9. Evaluation is markout curves + realized-spread decomposition with
   day-clustered t-stats, block-bootstrap CIs, per-quarter regime splits, and
   queue-model bracketing (RiskAverse vs ProbQueue) — a result whose sign flips
   across the bracket is not a result.
10. The E1–E5 kill ladder is unchanged from PROGRAM.md; the arithmetic gate
    (E1) runs on data already on disk before any model is built.

---

## 2. Component architecture

```
  Binance WS (live)                 Binance Vision (historical)
  bookTicker | depth20@100ms | trade         tick aggTrades (full history)
      │            │             │                   │
      ▼            ▼             ▼                   ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STATE: L1 (I, S, mid) · L2 snapshots (OFI, depth) ·         │
 │        signed trade stream (ε_t, v_t) on a volume clock     │
 └────────┬──────────────────────────────┬─────────────────────┘
          ▼                              ▼
 ┌───────────────────────┐   ┌────────────────────────────────┐
 │ (a) FAIR VALUE        │   │ (b) FLOW / INTENSITY STATE     │
 │ F = m + G*(I[,S])     │   │ λ_side 2-scale EWMA × f(I),    │
 │     + propagator P(h) │   │ size histogram, burst flag,    │
 │ ε̂ sign predictor      │   │ VPIN regime dial               │
 └──────────┬────────────┘   └──────┬─────────────────────────┘
            │        ┌──────────────┤
            │        ▼              ▼
            │  ┌──────────────────────┐  ┌───────────────────────┐
            │  │ (c) FILL/QUEUE       │  │ (e) TOXICITY          │
            │  │ λ(δ)=A·e^{−kδ} (v1)  │  │ online logistic       │
            │  │ queue bracketing     │  │ p_toxic; ζ(δ) markout │
            │  └──────────┬───────────┘  └──────────┬────────────┘
            ▼             ▼                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ (d) QUOTING ENGINE (GLFT ergodic)                           │
 │ center C = F + ρ(alpha, capped) − funding skew              │
 │ half-spreads δ*_{b,a}(q); one-sided withdrawal; unwind mode │
 │◄── (f) RISK: inventory cap Q (jump-VaR), kill-switch,       │
 │            rate budget 300 orders/10 s                      │
 └──────────────────────────┬──────────────────────────────────┘
                            ▼
              quotes (bid, ask, size) → venue / hftbacktest sim
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ (g) EVALUATION: markout MO(τ), es/rs/Λ decomposition,       │
 │ queue-bracketed replay, day-clustered + block-bootstrap CIs │
 │ — owns every gate number in §6                              │
 └─────────────────────────────────────────────────────────────┘
```

### (a) Fair value engine (R1)

Governing formulas:

```
F_t(h) = m_t + G*(I_t[,S_t]) + P_t(h)

G*     : Stoikov micro-price adjustment — absorbing Markov chain on state
         X=(I,S), I = Q_b/(Q_b+Q_a); g1=(Id−Q)^{-1}R1·k, B=(Id−Q)^{-1}R2,
         G* = g1 + B·g1 + B²·g1 + …  (symmetrized; |G*| ≤ S, monotone in I)
P_t(h) = Σ_{ℓ≥0}[G(ℓ+h)−G(ℓ)]·ε_{t−ℓ}·f(v_{t−ℓ})            (decay of past impact)
       + Σ_{k=1..h} G(h−k+1)·E_t[ε_{t+k}]·f̄                  (predicted-flow impact)
G(ℓ)   : propagator, Toeplitz solve of S(ℓ)=Σ_n 𝒢(n)C(n−ℓ) then smooth-fit
         g0·(ℓ0+ℓ)^{−β};  f(v)=v^δ, δ≈0.1–0.3;  ε̂ = AR on trade signs (HDIM)
h      : expected quote-resting time in trade events (≈2 mid-changes, Kolm)
```

- Inputs: bookTicker events (I, S, mid); signed trades (ε, v) — exact taker
  signs from `is_buyer_maker`. Outputs: F_t (price units), ε̂_t ∈ [−1,1]
  (sign forecast; doubles as a toxicity feature for (e)).
- Calibration: G* from ≥1 day of bookTicker per symbol, re-estimated daily
  (majors: imbalance-only 10-state chain; alts: (I,S) grid, sparse cells
  shrunk to weighted mid). G(ℓ), C(ℓ), δ per symbol-month from aggTrades —
  estimable on the full free history today. Validate G by predicting R(−ℓ)
  and the signature plot; mismatch → HDIM form.
- Update: F evaluated event-driven; G* daily; propagator monthly.
- Later regression features (not in v1 F): snapshot-diff L1 OFI at the 100 ms
  grid, integrated (PCA) OFI, deeper-rank corrections — DEFER until weeks of
  L2 accumulate. TFI is weighted over quote-OFI (Silantyev: crypto trade flow
  out-explains L1 OFI — reverse of equities). DL-LOB: REJECT (LOBCAST
  collapse; representation beats architecture).

### (b) Flow / intensity state (R2)

```
λ_s ← λ_s·e^{−Δt/τ};  on event: λ_s += 1/τ        s ∈ {buy,sell}, τ ∈ {1 s, 30 s}
rate(s,t) = λ_s(t)·f(I_t)                          f(I) = P̂(arrival | I-decile)/P̂(arrival)
burst flag: λ_fast/λ_slow > κ_b  (κ_b ≈ 3–5)
sizes: empirical per-side histogram (atoms at round sizes) — independent of intensity
```

- Inputs: signed trade events (grouped to sweeps), I from bookTicker.
  Outputs: λ_buy, λ_sell (events/s), f(I) multiplier, burst flag, size
  histogram; VPIN on a volume clock as a slow (minutes) regime dial only.
- Calibration: EWMA is parameter-free beyond the two half-lives; f(I) by
  conditional counting, weekly; optional exp-kernel Hawkes MLE with fixed
  betas as a stability check (branching n<1). Full multivariate Hawkes:
  DEFER to simulator fidelity only. Flag: R2's "2-scale EWMA + f(I) captures
  ~90% of a full Hawkes for quoting" is a rule of thumb motivated by the QRH
  decomposition (60–80% history term + imbalance state), not a measured 90%.
- Consumers: (c) intensity context, (d) widen-on-burst, (e) features,
  E4 simulator (arrival + size resampling).

### (c) Fill / queue layer (R2)

```
v1: λ(δ) = A·e^{−kδ};  fit ln λ̂(δ_j) = ln A − k·δ_j  (OLS, per side,
    trailing ~10 min of sweeps reaching depth δ_j, refit every ~5 s)
v2: Lokin–Yu state-conditioned fill prob — pure-death first passage
    ĝ(s) = Π_{k=1..q}(μ_k+φ_k)/(μ_k+φ_k+s), rates by conditional counting
sim: ALWAYS bracket — RiskAverseQueueModel (pessimistic) and
    ProbQueueModel PowerProbQueueFunc3 n≈3 (moderate)
```

- Inputs: trades + depth20 snapshots. Outputs: A(t) (fills/s at δ=0), k(t)
  (1/price) per side → (d); fill-probability curves → sim and (g).
- Queue economics: front-of-queue vs average position is worth 0.21–0.26 ticks
  (Moallemi–Yuan) — join early, cancel/replace when queue rank deteriorates;
  monitor jump/trade (γ/μ) and cancel/trade (η/μ) ratios live. Binance has no
  public L3, so own queue rank is never observed: the bracket IS the
  uncertainty statement, and f is later calibrated with min-notional live
  probe orders (Albers method; linear model on liquidity-ahead + opposite
  queue predicted fills with R²=0.946).
- "Touch = fill" backtests: REJECT. Deep survival models: REJECT for now.

### (d) Quoting engine (R3)

```
center:  C_t = F_t + ρ_t − s_f(q, f̂)
alpha:   ρ_t = clip( α̂_t/κ_α , ± μ̂/(γσ√(2AkC_ξ)) )      Bergault risk cap;
         if the signal is already an h-horizon cum-return forecast r̂ with
         h ≳ holding time, ρ_t = r̂
funding: f̂ = S·F̂ (USDT/contract/hr, predictable from the premium-index TWAP)
         enters as drift on inventory → skew away from the paying side
quotes:  δ*_b ≈ (1/γ)ln(1+γ/k) + (2q+1)/2·S_glft
         δ*_a ≈ (1/γ)ln(1+γ/k) − (2q−1)/2·S_glft
         S_glft = √( σ²γ/(2kA) · (1+γ/k)^{1+k/γ} )        (ergodic GLFT)
overrides: |signal| > θ_w → pull the adversely-selected side entirely
           (Cartea–Wang policy shape, threshold tightens with q);
           burst flag → widen; cap breach → unwind mode (Yu-2026 explicit
           one-sided quotes with g(s) signal drift)
```

- Inputs: F from (a); σ = microprice realized vol on 10 s–5 min bars
  (price·s^{−1/2}; never tick-by-tick mid); A, k from (c); q, f̂; toxicity
  corrections from (e); Q and rate budget from (f). Outputs: (bid px, ask px,
  size) per side or pull.
- γ is backed out from the target inventory band: pick γ so
  S_glft·Q_target ≈ tolerable adverse move; Q is a first-class input (GLFT is
  the with-limits model). Exact matrix-exponential GLFT kept as offline check.
- Update: decision tick event-driven or on the 100 ms grid, subject to the
  (f) rate budget; parameters quasi-static (seconds–minutes, see §3 C5).
- Empirical-λ upgrade path: Guéant general-intensity approximation (fit λ(δ)
  nonparametrically, reuse H''(0) closed form) when the exponential fit is
  poor. Multi-asset Γ-skew (Bergault): DEFER until >1 symbol is quoted.
  Funding-OU HJB grid: REJECT (funding persistence 2–8 h ≫ HF holding times;
  the skew fold-in captures it). Flag: the funding-aware "beats plain AS with
  ~35% lower inventory RMS" evidence is one paper's own simulator on
  Hyperliquid ETH/BTC — treat as a hypothesis for the E4 ablation, not a fact.
- RL as the control layer: REJECT for now (simulator-quality-limited;
  closed-form + supervised signal captures the value with testable pieces).

### (e) Toxicity / adverse-selection layer (R2 §5, R3 §3)

```
label:  y_j = 1{ MO_j(G) < −c },  G ∈ [1, 30] s,  c ≈ maker fee
model:  online logistic p_toxic = σ(w·x); x = (I decile, λ_fast/λ_slow per
        side, signed-flow EWMA, ε̂, recent vol, spread, sweep size, levels
        consumed, size vs queue, time-of-day)
ζ(δ):   Ê[adverse mid move at horizon h | fill at depth δ]  (markout-
        calibrated additive quote correction, per 2508.20225 — phase 2)
```

- Enters (d) at exactly ONE point: an additive per-side quote correction plus
  the pull trigger (resolution C4 — no simultaneous intensity-side and
  quote-side entry for the same feature).
- Calibration: population proxy fills from the (g) markout harness now
  (unlimited free labels from aggTrades); own fills from E5 later. PULSE
  subspace-EKF: DEFER unless plain online logistic demonstrably decays — its
  published edge came from per-client memory, which a CLOB maker does not have.

### (f) Risk layer (R3 §4, R4 §5)

```
inventory cap: Q ≤ tolerable_loss / (cascade_move × margin_mult)   — jump/cascade
               VaR, NOT diffusion σ (books widen 10–50x exactly when loaded)
notional cap:  ≤ x% of book depth within ±1% of mid (ADL/socialized-loss exposure)
kill-switch:   liquidation prints, OI drop, funding gap, spread blowout
               → cancel-all, switch (d) to unwind mode
rate budget:   300 orders/10 s and 1,200/min (Binance USDM default) —
               breadth × requote-rate allocation; 10 syms × 2-sided × 1
               requote/s ≈ 200/10 s before fills: rate limits, not CPU,
               bound quoting breadth
```

Formal cascade modeling: DEFER; these are adopted as engineering rules.

### (g) Evaluation layer (R4 §3, R2 §4)

Canonical formulas (q_t = +1 taker buy, −1 taker sell; maker is counterparty):

```
effective half-spread: es_t    = q_t (p_t − m_t)/m_t
realized half-spread:  rs_t(τ) = q_t (p_t − m_{t+τ})/m_t      = maker gross revenue
adverse selection:     Λ_t(τ)  = q_t (m_{t+τ} − m_t)/m_t ;    es = rs + Λ
maker markout:         MO_j(τ) = −q_j (m̂_{t_j+τ} − p_j)/p_j,
                       τ ∈ {0.1, 0.5, 1, 5, 15, 60, 300 s}
go/no-go per fill:     E[rs(τ*)] − maker_fee + rebate > 0 at the horizon τ*
                       at which inventory can realistically be flattened
```

Methodology rules (all mandatory, from repo discipline + R4):

- Parent-event collapse: merge prints with identical (timestamp, side) into
  one sweep before computing markouts, else SEs are fake.
- Mid: forward data → bookTicker mid (or microprice); historical
  aggTrades-only → midpoint of most recent taker-buy / taker-sell prints;
  report historical curves only for τ ≥ 1 s (staleness biases AS down below
  that); sub-second points from forward-collected data only.
- Inference: never per-trade iid SEs. Day-clustered t-stats (dof = #days) +
  block bootstrap (blocks ≫ max(markout horizon, autocorrelation time);
  ~30 min for ≤5 min markouts). Per-quarter regime splits with sign-stability
  demanded — pooled significance is not accepted.
- The aggTrades population markout is an UPPER bound on our economics (a
  marginal entrant at the back of the queue does strictly worse — "fast fills
  are bad fills"); the queue haircut comes from the (c) bracket in sim.
- Simulation: hftbacktest (Rust/Py), local-timestamp event streams, measured
  IntpOrderLatency (stress at 2–5x measured RTT; PnL-vs-latency curve must
  not be a cliff), fill share capped at printed volume × queue share, funding
  applied in post-processing, residual inventory closed at
  mid − half-spread − taker fee, per-day flat-close PnL.
- This layer owns every kill-gate number in §6.

---

## 3. Conflict resolutions

**C1 — R1 propagator vs R2 EWMA flow: complementary, with one shared input.**
Both consume the signed trade stream, but they output different objects: the
propagator maps flow history to an expected mid move in price units (fair
value); the EWMA intensities map it to per-side arrival rates in events/s
(fill model, burst detection). Resolution: one flow-state module computes the
shared primitives (sweep-grouped signed events); the propagator term is the
only flow input to F; the signed-flow EWMA does NOT enter F separately (it
would double-count the propagator) and survives only as a toxicity feature in
(e). Validation duty: the propagator term must out-predict a plain signed-flow
EWMA at h ≈ 2 mid-changes on the same data; if it does not, v1 falls back to
the simpler EWMA shift and the propagator is dropped, not stacked.

**C2 — R3's Binance quoting math vs R4's Binance-uneconomic verdict: both
stand; they answer different questions.** R3 specifies the engine assuming
positive per-fill economics exist somewhere; R4 shows they do not exist for
standalone MM on liquid Binance names at solo tiers (fee 1.8 bps vs half-spread
≤0.5 bps). Resolution: the engine is venue-agnostic; its deployment surface is
gated by R4's arithmetic — Variant A on Binance (no fee wall: we would
otherwise pay taker), Variant B only where break-even full spread
≈ 2×(fee − rebate + AS) clears (Binance ≥4 bps names at ≥VIP3, promo listings,
or Hyperliquid). No quoting model is permitted to override the E1 arithmetic.

**C3 — Microprice vs mid as quoting center: quote around F, decompose around
mid.** The mid is the bookkeeping anchor (G* and the propagator are both
defined as adjustments to m and are separately validatable against forward
mid); the quoting center is C = F + ρ − funding skew, i.e. microprice-adjusted
fair value, never raw mid. GLFT's σ is measured on microprice (bars, not
ticks) to avoid bounce inflation. On 1-tick pinned books |G*| ≤ S means the
microprice term mostly decides which side of the tick to lean on — that is
its job, not forecasting.

**C4 — Where toxicity enters: quote-side in v1, intensity-side deferred.**
R3 §3 offers two mathematically equivalent-shaped routes: modulate per-side
intensity A_side (fads/two-population) or correct the quotes (markout-ζ(δ),
withdrawal). Both briefs agree adverse selection does not change the shape of
the optimal rule, only its parameters. Resolution: v1 enters on the quote side
only — additive skew from E[MO|state], widen on burst, pull at extremes — one
entry point per feature so a toxic state is never counted twice (once in A,
once in skew). The two-population intensity model is DEFERRED until a
toxicity proxy shows a stable sign across quarters.

**C5 — GLFT's constant (A, k, σ) vs R2's bursty reality: quasi-static
parameter feed.** Intensities are Hawkes-bursty and heteroskedastic; GLFT
assumes constants. Resolution (R3's own fix #2): re-fit A, k on trailing
~10 min windows every few seconds, σ on rolling bars, evaluate the closed form
each decision tick with current values, and let the (b) burst flag trigger the
widen override for the transients the quasi-static feed misses. The ergodic
(t-independent) forms are used throughout — finite-T quote collapse is an
artifact for a maker who never stops (fix #5).

**C6 — Trade streams and markout mids: one canonical harness.** Live
collection uses @trade (raw per-match — @aggTrade delivers nothing on
URL-subscribed fstream endpoints, verified 2026-08-19); history is Vision
aggTrades. Both reduce to the same sweep events via (timestamp, side)
collapse. Markout formula is the (g) MO_j(τ) definition everywhere; R2 §4 and
R4 §3.2 describe the same object with cosmetically different denominators —
(g)'s form is canonical. Historical τ<1 s points are not reported.

**C7 — Quote-OFI's role vs TFI: TFI is the core, OFI is a candidate
complement.** R1 adopts L1 OFI as a regression feature; Silantyev says TFI
out-explains L1 OFI in crypto perps (quote churn/spoof noise). Resolution: the
trade leg carries the flow information in v1; OFI (snapshot-diff at 100 ms)
is added only if it demonstrates incremental prediction over TFI at h ≈ 2
mid-changes once L2 history accumulates. L1 imbalance inputs are spoofable:
cap effective size at min(Q, q95) and migrate to band-depth imbalance from
depth20 as data accrues.

**C8 — "Fees are the first wall" vs "Tokyo VPS mandatory": sequential, not
contradictory.** Fees decide the venue/variant (E1, no infrastructure
needed); latency and the confirmed US geo-block of fapi.binance.com decide
feasibility of any live step. Research and simulation run from the current
box on WS + Vision data; the Tokyo VPS ($50–200/mo, 5–30 ms order RTT vs
150–180 ms US) is purchased only when E-ladder progress requires measured
order latency (late E4) or live paper trading (E5).

---

## 4. Where the user's original idea landed

| Original concept element | Adopted form | Correction |
|---|---|---|
| Monitor incoming taker flow | (b) per-side 2-scale EWMA intensities + sweep grouping; exact signs from `is_buyer_maker` | none — strengthened: this is a bona-fide Hawkes-lite estimator |
| Form a distribution of flow | (b) intensity state × imbalance multiplier f(I) + empirical size histogram | sizes are independent of intensity (Hoeffding test, 2312.08927) — no joint distribution needed; two marginals suffice |
| Fair price = market price + simulated flow impact | (a) propagator fair value P_t(h) (Bouchaud TIM1/HDIM) + Stoikov microprice G* | "simulated impact" formalized as transient-impact decay + predicted-flow impact; the two terms can point opposite ways (fade transient vs follow persistent flow) |
| Post bid/ask around fair price | (d) GLFT ergodic quotes around C = F + ρ − funding skew | symmetric posting is a money pump (Albers); inventory skew, funding skew, and one-sided withdrawal are load-bearing, not optional |
| Profit = spread + rebate − adverse selection | (g) E[rs(τ*)] − maker_fee + rebate, markout-measured | no rebate exists at solo tiers on Binance/HL (institutional-share-gated); at VIP0 the maker FEE is 40–70x the majors' half-spread — venue/universe choice, not quoting skill, decides sign |
| "Impermanent loss" as the loss term | fill-conditional markout MO_j(τ) / adverse-selection Λ(τ) | IL is the AMM analogue (LVR); the CLOB object is adverse selection, measured per fill, horizon-resolved |
| Implicit: quote the majors | Variant A overlay (majors fine — we are the client); Variant B only on ≥4 bps-spread names / HL | at-touch MM on BTC/ETH rejected at any solo tier: 0.03 bps spread, queue war vs LP-program firms |

---

## 5. Data & infrastructure plan

| Resource | Status | Consumed by |
|---|---|---|
| Vision tick aggTrades, 16 syms × 30 d (2026-07-18→08-17), `data/mm_hf/vision/parquet/aggTrades/` | on disk | E1 markout/viability screen; (a) propagator + ε̂ calibration; (e) labels |
| Vision aggTrades, full history, any USDM symbol | free, fetch on demand | E1 universe extension; regime splits (2023 ≠ 2026) |
| Vision bookTicker archives | exist ONLY 2023-05-16→2024-03-30 (discontinued) | historical spread study (R4 §2.1); regime caveat mandatory |
| Live WS collection: 16 syms × (bookTicker + depth20@100ms + trade), `data/mm_hf/raw/`, ~0.5–1 GB/day gz | running since 2026-08-19 ~12:45 UTC (`collect_hf.py`) | (a) G* (needs ~1 day), (b) f(I) (~1 week), E2/E3 (2–6 weeks), E4 (1–2 months incl. stress days), current-regime spreads |
| Tokyo VPS ($50–200/mo) | not yet purchased | fapi REST access (US box geo-blocked, confirmed 2026-08-19), measured order RTT for E4 latency model, E5 paper trading |
| Tardis.dev historical Binance L2 | deferred | E3/E4 backfill IF E1 passes AND forward collection is the binding constraint (esp. stress-day coverage); hftbacktest has a native converter; buy one month × few symbols first |
| Hyperliquid forward spread/trade collection | not started | Variant B decision; start when E1 confirms Binance-standalone is dead but the program continues (cheap, and HL history is otherwise unobtainable) |

Collector caveats (from `collect_hf.py`, verified findings): fstream routed
`/market/ws` silently drops bookTicker/depth20 — combined
`/stream?streams=` URL form required; depth20@100ms is a self-contained
top-20 snapshot (no diff reconstruction) — sufficient for snapshot-diff OFI,
band imbalance, and top-level queue sizes; nothing on Binance provides L3, so
queue position is permanently model-bracketed. `recv_ns` local timestamps are
recorded per row — all simulation trades on local time (pitfall §4.2 #6).

Timeline of what unlocks when (calendar time is the long pole):
day 1: G*, E1 harness on historical aggTrades. Week 1–2: f(I), λ(δ) fits,
current-regime spread tables, E2 naive-quoter replay. Week 3–6: E3 signal
tests on accumulated L2. Month 2+: E4 full sim with stress-day coverage;
Tokyo VPS + latency measurement; Tardis decision.

---

## 6. Kill-gate alignment (E1–E5)

Gates restated in terms of §2 components. Numbers come from (g) only;
no gate is decided by a backtest that has not passed the queue bracket rule.

| Gate | The number | Producing components | Kill condition (unchanged from PROGRAM.md) |
|---|---|---|---|
| E1 (arithmetic) | per symbol × fee tier: E[rs(τ*)] − maker_fee (+rebate), bps, day-clustered CI, per-quarter sign stability; τ* from realistic flatten horizon | (g) on historical aggTrades; half-spread from 2023–24 bookTicker archives + forward collection | no symbol × reachable-tier cell > 0 → standalone Binance MM stops; program pivots to Variant A + HL data collection |
| E1-A (overlay arithmetic) | effective RT cost of the XS book with passive entry/exit: eff = Σ_legs [P(fill)·(fee_mkr − hs + AS) + (1−P(fill))·(fee_tkr + hs + chase)] | (g) + (c) fill model on the book's actual names/sizes/rebalance times | eff RT > 8 bps (capstone threshold) → overlay dead; XS book remains cost-bound |
| E2 (naive quoter) | net markout PnL of a no-alpha GLFT quoter, under BOTH RiskAverse and ProbQueue-f3 | (c), (d) with ρ=0, (g) on collected L2 | markout-adjusted spread ≤ 0 in every cell → a signal is required to proceed (E3), else die |
| E3 (signal lift) | Δ net markout from quoting around F vs mid; F−m prediction of forward mid at h ≈ 2 mid-changes | (a), (b) feeding (d); (g) scores | signal lift fails to flip any E2-negative cell positive (sign-stable across the queue bracket) → stop |
| E4 (full sim) | net PnL after measured latency (stress 2–5x), funding post-processing, flat-close inventory, fill-share caps | all of (a)–(f) in hftbacktest; (g) reports | net PnL ≤ 0 under the PESSIMISTIC queue model → stop; sign flip across the bracket → treated as ≤ 0 |
| E5 (paper live) | sim-vs-live tracking: fill precision/recall, first-fill timing error, markout delta (closed-action replay); rate-budget compliance | live quotes from Tokyo VPS; (g) + hangukquant closed-action methodology | unexplained sim-live divergence → do not fund; divergence explained by queue model → recalibrate f and re-run E4 |

The bracket rule is itself a gate condition everywhere: a PnL whose sign
depends on the queue model choice is recorded as a failure, not an average.

---

## 7. Open questions, ranked

1. Does any symbol × reachable-tier cell clear the fee+AS floor, and by how
   many bps? — E1 (the whole program hangs on this table).
2. What effective RT cost does passive execution achieve on the XS book's
   actual names and rebalance sizes (fill rate, chase cost, AS on fills) —
   ≤ 8 bps or not? — E1-A, refined by E4-A replay.
3. Does F = mid + G* + propagator predict forward mid at h ≈ 2 mid-changes
   out-of-sample, and does the propagator beat a plain signed-flow EWMA
   (C1's fallback test)? — E3.
4. How wide is the queue-model bracket on collected Binance L2 — do any E2/E3
   conclusions survive RiskAverse? — E2/E4.
5. What are current-regime (2026) spreads and markouts vs the 2023–24
   bookTicker-era numbers the E1 screen partly relies on? — E1 forward-data
   cross-check after 2–4 weeks of collection.
6. What is the measured Tokyo order RTT and the PnL-vs-latency curve —
   cliff or plateau at 5–30 ms? — E4 latency stress (needs VPS).
7. Do Hyperliquid mid/tail spreads and markouts clear 1.2–1.5 bps maker fee —
   is Variant B real on the only venue without a colocation moat? — HL
   forward collection + an E1-style screen (start on E1 completion).
8. Does the toxicity layer (online logistic on markout labels) add net PnL
   over imbalance-only skew, or is it redundant with (a)'s ε̂? — E3/E4
   ablation.
9. Is the funding-skew term worth its complexity on Binance 8 h funding at HF
   holding times (the cited evidence is one paper's own HL simulator)? — E4
   ablation.
10. What inventory cap Q does cascade jump-VaR imply on the pilot names, and
    does the kill-switch trigger fast enough on replayed stress days
    (2025-10-10-type)? — E4 stress replay; revisit at E5 with live data.

Deliberately not open: DL-LOB (rejected), RL control (rejected until a
validated simulator + own-fill history exist), at-touch majors MM (rejected),
30s bookDepth as fair-value input (rejected; liquidity context only).
