# Theory diff — P-2026-002 (Binance perps) vs P-2026-003 (Polymarket binaries)

## Component-by-component

| | P-2026-002 adopted | P-2026-003 adopted | Nature of the change |
|---|---|---|---|
| **Fair value** | `F = mid + G*(microprice) + P(h) propagator`, TFI; all from the SAME book we quote | `p̂ = Φ(d)`, stream-anchored on an EXOGENOUS Chainlink TWAP; σ_eff from a settlement variance law | **Source inverted**: endogenous book info → exogenous oracle. We now ignore the book we trade on (gap X2) |
| **Flow / intensity** | 2-scale EWMA (Hawkes-lite) × imbalance multiplier `f(I)`; size histogram; burst flag; VPIN dial | — nothing, until B8/E-F was added 2026-08-20 | **Dropped wholesale**, now being restored |
| **Fill / queue** | `λ(δ)=A·e^{−kδ}` log-linear; queue bracket; Moallemi–Yuan | `λ_fill(ℓ, Q_ahead)` discrete per level; queue bracket; Moallemi–Yuan | Same theory, discretised |
| **Quoting** | GLFT ergodic closed form; alpha as capped reservation shift; funding skew; RL rejected | arXiv 2607.17991 prediction-market HJB; discrete per-level EV argmax; participation region; rewards band as constrained control | **Re-based** on a model of the actual asset class |
| **Toxicity** | markout-calibrated `ζ(δ)`; online logistic | `ζ(ℓ)` (placeholder) + `ζ_snipe` from latency arbitrage | Same shape, different generator (see §2) |
| **Risk** | cascade jump-VaR cap; kill switch; rate budget | `Q_max = κ/(γp̂(1−p̂))` (defective) + portfolio layer §14 | **Regressed** — the Binance version was more complete |
| **Evaluation** | markout, es/rs/Λ, queue bracketing, day-clustered + block bootstrap, notional-weighted | identical, PLUS paired ΔBrier calibration | **Upgraded** (see §3) |

## 1. What genuinely differs (structural, not stylistic)

1. **Finite horizon with a binary terminal payoff.** Perps: ergodic, liquidate
   at mid, funding carry. Binaries: T=300 s, inventory resolves itself into
   Bernoulli, no liquidation leg → terminal penalty `q²p(1−p)` instead of
   `γσ²q²`, and the "never unwind" doctrine.
2. **Tick regime inverted — same conclusion, opposite cause.** Binance majors:
   spread pinned at 1 tick because the tick is far too FINE (0.01 bps). PM:
   books 2–4 ticks wide because the tick is enormously COARSE ($0.01 on a
   $0.50 asset = 200 bps). Both kill the continuous-δ closed form; both land on
   discrete per-level EV.
3. **Fee sign flipped.** Binance: maker pays 1.8 bps, no reachable rebate, fee
   = 4–70× the half-spread ⇒ arithmetically dead. PM: maker fee $0 + ~70 bps
   rebate + a resting subsidy, while the TAKER pays ~350 bps ATM ⇒ the maker
   side is subsidised and crossing is prohibitive.
4. **Adverse selection has a different generator.** Binance: Glosten–Milgrom
   informed traders with PRIVATE information. PM: Budish–Cramton–Shim snipers
   with the SAME PUBLIC information, only faster. Consequence: the defence is
   a participation region + size, NOT spread widening — you cannot widen your
   way out of a pure speed race.
5. **The complement structure has no perp analogue.** Up + Down = $1 gives a
   riskless mint/merge exit; perps have no such route.
6. **Volatility plays a different role.** Binance: σ enters the GLFT spread
   directly. PM: σ maps edge→probability (dominant error in p̂, up to 23 ¢),
   yet pickoff exposure is volatility-FREE. Bad σ mis-prices; it does not get
   you sniped.
7. **Subsidy as an obligation.** Rewards band = principal–agent MM contract +
   Tullock contest. No Binance analogue at reachable tiers.

## 2. What carried over unchanged (and should have)

Fills are endogenous; markout-only accounting; queue-model bracketing with
sign-flip = failure; day-clustered + block-bootstrap inference;
notional-weighted gates; fails-final / passes-provisional; prereg freeze;
Moallemi–Yuan queue value; RL and DL-LOB rejected.

## 3. What PM gains that Binance never had

**Ground truth.** Every window RESOLVES to a labelled outcome, so fair value
can be scored directly (Brier / log-loss, paired) instead of only inferred
through markout. This is a genuine methodological upgrade and is why σ can be
fitted by MLE on winners rather than on realized returns.

## 4. THE PATTERN — what the adoption quietly dropped

Replacing endogenous fair value (microprice + propagator, built from the book
we quote) with an exogenous settlement stream also discarded, silently, the
entire **flow-and-book-information half** of the Binance stack. Every current
gap clusters there and nowhere else:

- X1 propagator / transient impact — does a PM sweep revert?
- X2 microprice / OFI of the PM book — we ignore the book we trade on
- X3 short-horizon Binance alpha into `E[X_T]` — zeroed by default
- B8 order-flow distribution — absent until 2026-08-20
- X4 queue-reactive intensities — named, unspecified

We kept the QUOTING and RISK halves and dropped the SIGNAL half. That is
defensible if the oracle stream is genuinely sufficient — but it was never a
decision, it was a side effect of re-basing fair value. Restoring the signal
half is where any informational edge would live, since the quoting half only
converts an edge into fills.
