# SIGMA_PLAN — design of the volatility estimator (P-2026-003)

Status: **plan, not code.** Nothing is implemented. Three prior estimators
(`exp_blend_model.py`, `exp_blend_v2.py`, `exp_blend_v3.py`) were built and each
violated a decision that had already been made, so this document fixes the
purpose, the estimand and the validation protocol *before* an estimator exists.

Written 2026-08-20. BTC first; §10 says what generalises to the other six.
`SIGMA_DIAGNOSTICS.md` was not present when this was written, so the numbers
below are ones I measured directly (§0.1); §9 marks which of them a proper
diagnostics run could overturn.

---

## REVISION 2, 2026-08-20 — response to SIGMA_PLAN_REVIEW

**This block supersedes the v1 "FINALIZED" block.** `SIGMA_PLAN_REVIEW.md`
(`6bea435`) held six MUST-FIX items against v1 and recommended HOLD on estimator
implementation. **All six are accepted**, two with modifications, and one item of
the review's suggested *remedy* is declined with reasons (R1). Estimator
implementation stays on HOLD; what follows is spec repair only.

Three of the review's claims were verified rather than taken on trust, and all
three reproduce: the unit mixing (`:93` fits log increments, `:277` names `X` the
raw TWAP, `:287` divides by a bps `σ_eff`); the belief fallback
(`BE_BELIEF_PLAN.md:650` needs `σ_eff` when the book is `Unavailable` and calls
that not rare) against `contracts.yaml:396-400` "never hardcode the Gaussian Φ";
and the branch-point convention mix in §3's table.

### The finding that changes the most: the anchor ledger is closed-form

The review's S6 says D2's predicted `c(r)` breach was computed **before** the
anchor-error term the plan itself introduces, so it is not yet evidence about
`c(r)`. That is correct, and the term is stronger than "unmeasured": because
`P̂ = 2·S30 − S60` is a *fixed linear functional of the same latent path* as σ,
`ω_P` is **not a free parameter**. It is computable exactly, and now is —
`sigma_kernels.py`, closed form plus a Monte-Carlo selftest:

| quantity | continuous | 1 s discrete (exact) | BTC @ σ=1.089 |
|---|---|---|---|
| `ω_P = sd(P̂ − P_t)` | `σ√10` = 3.162σ | **3.0845σ** | **3.36 bps** |
| old S60 anchor `sd(S60 − P_t)` | `σ√20` = 4.472σ | 4.4162σ | 4.81 bps |

Four consequences, each of which changes a decision in v1:

1. **D2 is RETIRED, not carried forward.** Against the same 2.6 bps realised
   innovation D2 argued from: `c(30) = 2.17` on the diffusion line alone,
   **`c(30) = 1.14`** once the anchor line is included — inside `[0.8, 1.25]`.
   This does *not* establish that `c(r)` is in band (2.6 bps is itself
   provisional, and `ŵ≈47 s` is untouched by it); it removes the *prediction*.
   Budgeting for redesign on the strength of D2 would have been budgeting
   against an incomplete ledger.
2. **It supplies S3's bracket without identifying latent spot.** `σ√10` is a
   **floor** — what the extrapolation costs with perfect, synchronous, noiseless
   feeds. Real `ω_P` adds feed asynchrony, Chainlink aggregation and
   deviation-threshold staleness. The empirical S30/S60-vs-Binance residual
   (basis-contaminated, and *not* identified, exactly as S3 argues) is the
   **ceiling**. That is the review's option 3, and it is available now, so the
   design no longer waits on an identification it cannot achieve.
3. **§3's claim that the `(r/w)²` weight makes the nowcast "cost nothing late in
   the window" is backwards.** anchor/diffusion `= 3·a(r)/k_law(r) ~ 1/r`, so the
   anchor share **rises** as expiry approaches: **4.0% of variance at `r=270`,
   47.5% at `r=30`** — the horizon where §3 itself notes 1 bp is ~20
   probability-cents. Corrected in §3.
4. **`k ≈ 1.42` gets a mechanical account.** §7's source 2 is this same term.
   Under the old S60 anchor the omitted line was **65% of total variance at
   `r=30`**; outcome-MLE had no way to cover it except by inflating σ. This is
   §7's "decompose, never multiply" discharged for one of its three sources.

The ledger's independence claim survives (anchor error is pre-`t`, innovation is
post-`t`; MC corr −0.0013), but it is a **Brownian-model claim, not a type-system
fact** — asserted in the contract note, checked in the selftest, per S3.

### Item-by-item response

| item | disposition |
|---|---|
| **S1** units | **ACCEPTED**, remedy modified — see R1. One coordinate, frozen in `sigma_kernels.py` and typed as `UnitSpace` in `contracts.yaml`. |
| **S2** consumer contract | **ACCEPTED.** Moved to Phase 0A. `contracts.yaml` v13 adds a `BE-Uncertainty` module and replaces `PathLaw`. |
| **S3** anchor order + `ω_P` | **ACCEPTED.** Order reversed (verify semantics → then implement). Identification answered by the floor/ceiling bracket above, typed as `AnchorErrorBudget`. |
| **S4** MNAR pairing | **ACCEPTED without qualification.** The v1 sentence "it is paired, so it survives the MNAR gap" was wrong and is struck. |
| **S5** physical forecast + overlap | **ACCEPTED.** The live output is a **forecast of future physical variance**, not "realised σ"; §6 retitled; tick-count power language struck. |
| **S6** `c(r)` / H-3 gates | **ACCEPTED and strengthened** — the ledger is computed first, and D2 does not survive it. `c(r)` is `DIAGNOSTIC` until the day-cluster threshold. |
| SHOULD-FIX 1 kernels | **ACCEPTED, and it resolves exactly.** See below. |
| SHOULD-FIX 2–6 | **ACCEPTED**; applied in §5, §8, §9, §10. |

**SHOULD-FIX 1 resolves cleanly.** The discrete law is *already* continuous at
`r = w` — both branches give `(w+1)(2w+1)/(6w)` = 20.5028 s. v1's defect was
**mixing conventions**: discrete on the `r ≤ w` branch (`r=30 → 1.77`) and
continuous on the `r > w` branch (`r=60 → 4.87`, where the discrete value is
4.93). The correct `r > w` kernel is `(r − w) + (w+1)(2w+1)/(6w)` = `r − 39.4972`,
not `r − 2w/3 = r − 40` — the review's "roughly +0.5 s", exactly +0.5028 s.

**One further defect neither v1 nor the review caught.** The in-window anchor
coefficient is **not** `(r/w)²·ω_P²` in general. Strictly inside the window the
forecast also needs the trailing `(w−r)`-second TWAP, which is *reconstructed
from the same S30/S60 pair*, so the two reconstruction errors are correlated and
add. It happens to be exact at the only two in-window grid points — `r=30`
(where `S30` is observed) and `r=60` (where the trailing part vanishes) — which
is why v1 got away with it. At an interior `r=45` the true coefficient is
**6.64σ² against v1's 5.35σ², a 24% understatement**. Harmless on today's grid,
live the moment the grid densifies toward per-second quoting, which the MM design
implies. `sigma_kernels.anchor_error_coeff` computes the general case.

### R1 — DISAGREEMENT: the unit space should be normalised arithmetic returns, not log

S1 is right that v1 mixes three coordinates and that this must be fixed. I
decline its *example* remedy, `x = log(S/S_ref)`.

**Reason: the settlement mark is an arithmetic mean, and log space breaks every
exact identity this plan is built on.** `X_T = (1/w)Σ P` is linear in the path.
`log E[X] ≠ E[log X]`, so in log coordinates the TWAP kernel, the nowcast
`P̂ = 2·S30 − S60`, the `r=30` decomposition and the closed two-line ledger all
degrade from identities to approximations — and each is load-bearing precisely
*because* it is exact. S1's own acceptance criterion ("the reference probability
calculation reproduces the chosen formula from one typed fixture at every
horizon") is not satisfiable in a coordinate where the formula is approximate.

**I am not making a magnitude argument, and the numbers say I could not.**
Measured, the Jensen gap is `+0.00059 bps` — **0.024% of `σ_eff(30)`**, utterly
negligible. Log space would not produce a materially wrong answer. It would
produce a *derivation* in which nothing is exact, in a programme whose recorded
failure mode is specification error hiding inside a plausible approximation
(running-vs-terminal, `σ_⊥`+`κ`, `v(t)` sum-vs-min, and the anchor itself). The
cost of the arithmetic coordinate is zero, so there is no case for paying it.

**Adopted instead**, and frozen in `sigma_kernels.py`:

```
x_t   = 1e4 * (S_t - X_0) / X_0        model coordinate, bps, dimensionless
σ                                      bps / sqrt(second)
Σ(r), ω_P², nugget                     bps²
d     = (E_t[x_T] - x_0) / sqrt(Σ(r))  dimensionless
```

Per-window reference `X_0` (the strike, known at `t0`), so the coordinate is
dimensionless and cross-symbol pooling is legitimate *after* normalisation and a
type error before it. This satisfies S1's substance — one coordinate, typed
dimensions, poolable — while keeping every identity exact.

### Status of the v1 decisions

- **D1** (feasibility before machinery) — **stands**, and is refined by the
  review's Phase 0A/0B split; see §12.
- **D2** (expect a `c(r)` breach) — **RETIRED**. It was computed against an
  incomplete ledger; with the ledger closed the predicted breach is 1.14, in
  band. No redesign budget is reserved on its account.
- **D3** (size the build to σ's job) — **stands but is narrowed by S2**: the
  belief fallback needs σ for a *level*, not only a shape, so "σ is only
  plumbing" is not the whole consumer story. The single-scale baseline is still
  the incumbent the blend must beat.

Unchanged and adopted from v1: the estimand's closed two-line ledger (now with a
correct anchor coefficient); `w=60` fixed with `ŵ_free` diagnostic-only;
tape-fitted rather than outcome-fitted; measure-then-floor on the link; the
`(σ̂, SE(log σ̂))` two-output; and that **"the book wins" is not currently
claimable** — for the S4 reason as well as the mis-anchoring reason.

---

## 0. Executive answer

1. **σ is not for the level of p̂.** Of the twelve places the design consumes σ,
   the two economically load-bearing ones (FLB calibration, inventory cap
   `L_adv=|q|(1−p̂)`)
   consume **no variance at all**, and the four dynamics consumers
   (participation frontier, `r*`, `ζ` pickoff floor, `λ_bin`) need only the
   variance **shape** plus `d` — and `d` can be taken from the book.
2. σ's surviving jobs are: **(i)** the H-3 falsification test that is the only
   route back to a p̂ edge; **(ii)** a coarse **regime/state** level for sizing
   and stand-down; **(iii)** the **basis** term `σ_⊥`, which sets the staleness
   threshold and the horizon below which our whole price model is noise;
   **(iv)** the **shape**, which is *not estimated* — it is `w=60` from EXP-M6
   plus the exact discrete kernel; and **(v)** — added per S2 — the **level**
   for the BE-Belief stream fallback whenever the book is `Unavailable`, which
   `BE_BELIEF_PLAN.md:650` states is not rare. (v) is why D3 is narrowed: σ is
   *mostly* plumbing, not entirely.
3. **A large share of the model's deficit to the book was never a σ problem.**
   It was a forecast-anchor error: `E_t[X_T]` was set to the trailing 60 s TWAP,
   which lags spot by ~30 s. Correcting the anchor (§2) improved paired Brier at
   **every** horizon and cut the outcome-MLE inflation factor from 1.42 to 1.27.
   Three σ generations were chasing the wrong term. **It does not follow that σ
   was adequate**, and the residual model-vs-book verdict stays unadjudicated —
   the sample it would be read on is MNAR (S4).
4. Recommended estimator: **tape-fitted variogram for the level and its
   time-variation (option B), with an empirically-estimated horizon calibration
   multiplier `c(r)` (option A) absorbing the parametric shape error** — i.e. a
   hybrid, but composed level-from-B × correction-from-A, not the
   shape-from-A × level-from-regime composition in the brief. §4.

---

### 0.1 What I measured (before recommending anything)

Feed (`prices/crypto_prices_twap_sixty`, 20 files, 20260819_15 → 20260820_10):

| | value |
|---|---|
| symbols | 8 (`btc eth sol xrp doge bnb hype zec`) — **`zec` is priced but not traded** |
| ticks/symbol | 63,140–63,161 over **18.9 h** |
| inter-tick Δt | p50 **955 ms**, p90 1,571 ms, **max ≈ 44 s** |
| ticks that change value | **87.2 %** — a genuinely updating feed, not heartbeat-dominated |
| `recv_ns − payload.timestamp` | p50 **1,692–1,745 ms**, p90 2,171–2,233 ms (confirms the 1.7 s lag) |

Windows: 1,634 resolved, 7 traded coins, 226–239 per coin, **2 day-buckets**
(20684, 20685). Pooled up-rate **0.5459** (the +4.5 pp drift, confirmed).

BTC variogram of the S60 tape, `V(r)=Var[X_{t+r}−X_t]`, log increments. Last
column is σ implied under a 60 s-TWAP-of-BM; **flat ⇒ the law fits**:

| r (s) | 1 | 5 | 10 | 30 | 60 | 120 | 180 | 300 |
|---|---|---|---|---|---|---|---|---|
| `V(r)/r ×1e8` | 0.032 | 0.119 | 0.220 | 0.562 | 0.872 | 1.019 | 1.085 | 1.169 |
| implied σ (bps/√s) | 1.396 | 1.214 | 1.182 | 1.161 | 1.144 | 1.106 | 1.105 | 1.119 |

`V(r)/r` moves **36×** across the range; the implied σ moves **1.26×**, and only
1.06× once r ≥ 10 s. The TWAP-of-BM variance law is a very good description of
this tape. That is the empirical licence for option B.

Per-symbol variogram fit `(σ, ŵ, nugget)`, and regime persistence
(Spearman ρ between trailing σ̂ and the next 300 s squared innovation):

| symbol | σ (bps/√s) | ŵ (s) | nugget (bp) | fit rmse | ρ(15 m) | ρ(60 m) | p90/p10 of σ̂ |
|---|---|---|---|---|---|---|---|
| bnb | 0.939 | 55 | 0.00 | 10.6 % | +0.248 | +0.270 | 3.9× |
| **btc** | **1.089** | **47** | **0.00** | **7.4 %** | **+0.188** | **+0.233** | **3.4×** |
| doge | 1.451 | 57 | 0.00 | 10.1 % | +0.307 | +0.329 | 3.7× |
| xrp | 1.542 | 49 | 0.00 | 8.4 % | +0.326 | +0.276 | 4.0× |
| sol | 1.612 | 72 | 0.14 | 9.6 % | +0.317 | +0.291 | 3.6× |
| zec | 1.986 | 58 | 0.14 | 4.9 % | +0.265 | +0.277 | 3.4× |
| eth | 2.381 | 61 | 0.00 | 4.9 % | +0.261 | +0.279 | 3.8× |
| hype | 4.334 | 81 | 0.28 | 3.8 % | +0.341 | +0.401 | 4.3× |

Three things fall out immediately:

- **The σ level does not pool.** 0.94 → 4.33 bps/√s, a **4.6× spread**.
  Constraint 3 (per-symbol) is not a stylistic preference; pooling is a 4.6×
  error on the object p̂ is most sensitive to.
- **ŵ ranges 47–81 s.** BTC's 47 s is 13 s below the settlement `w=60`. §9-3
  argues this is most likely nugget confounding, but it is *unresolved* and it
  is a ≤1.6× variance bias on the in-window branch if it is real.
- **v3's fitted `w_fast=0.9` is contradicted.** The 60 min trailing window
  predicts forward vol **at least as well as** the 15 min window in 6 of 8
  symbols. That 0.9 was a pooled outcome-MLE artefact.

---

## 1. What is σ FOR? — answering the question before choosing a method

The brief lists five candidate purposes. I mapped every σ consumer in
`PM_MM_PLAN.md`, `PM_QUANT_REVIEW.md`, `PM_ARCHITECTURE.md`,
`PM_DEEP_REVIEW.md` and `contracts/contracts.yaml`. Result:

| consumer | σ object needed | LEVEL or DYNAMICS | edge-bearing? |
|---|---|---|---|
| `p̂ = Φ(d)` | terminal `Var_t[X_T]` | LEVEL | **demoted** (ŵ→0) |
| `L_adv=\|q\|(1−p̂) ≤ κ_$`, `Σ_c\|q_c\|(1−p̂_c) ≤ L_max` | **none** (linear in p̂) | LEVEL of p̂ only | **yes — risk cap** |
| FLB harvest (a small forecast-calibration effect) | **none** (book-bucket conditioned) | neither | **not established** — see note |
| reservation `p̂ − γqv(t)`, `v=p̂(1−p̂)` | none beyond p̂ | LEVEL | no |
| participation frontier `m/φ(d) ≥ k√(3L/r)` | shape + `d` | BOTH | no |
| `r*` quotable horizon | shape + `d` + `L` | BOTH | no |
| `ζ(ℓ)` pickoff floor `Ê\|Δp̂ over L\|` | short-horizon vol of p̂ | DYNAMICS | no |
| `λ_bin = √3·φ(d)/√r` | shape only | DYNAMICS | no |
| rewards band `R/X ≥ c(\|d\|,r)` | shape + `d` | DYNAMICS | no |
| stand-down `N` (stream staleness) | `σ_⊥` variogram | neither | no |
| hedge delta `φ(d)·∂d/∂F` | level | LEVEL | non-core |
| `μ̂` injection at short r | level at small r | LEVEL (inverse) | demoted |

**Candidate (a) — the level of p̂ — is out.** The book beats us by 2.5–3.2
Brier points at every horizon through three σ generations. `PM_DEEP_REVIEW.md`
H-3 states the reason sharply: after stream anchoring, `E_t[X_T]` and `K` are
both public feeds, so **σ_eff is the *only* axis on which p̂ can differ from the
book**. The programme's entire informational claim reduces to "we forecast
5-minute crypto volatility better than the Polymarket book does" — and that
claim has been tested three times and lost three times.

**Candidate (b) — inventory risk — is out as a σ consumer.** After the
`PM_QUANT_REVIEW.md` Q4 dimensional fix, the risk layer is `L_adv = |q|(1−p̂)`,
**linear in p̂ and free of any variance**. `q²p(1−p)` survives only inside the
CARA reservation term, where it is flat at the money. Sizing needs a *probability*,
not a *variance*, and the book supplies a better one.

**Candidates (c), (d), (e) are where σ actually lives — and (c)/(d) can be made
σ-free.** This is the plan's sharpest recommendation:

> **Take `d` from the book, not from us: `d_book = g⁻¹(p̃_book)`, where `g` is
> the *selected* `LinkFunction` and `p̃` is the recalibrated mid.** Then the
> frontier `m/g'(d) ≥ k√(3L/r)`, `r*`, `ζ` and `λ_bin` need only `w=60`
> (verified, not estimated) and `r`. They become **genuinely σ-free**, and the
> Q9 failure mode — under-estimating σ by 2× at |d|=1 leaves us quoting
> **52.9 s deeper** into the sniping zone — is *deleted rather than mitigated*.

**S2 correction: this is not `Φ⁻¹`, and it is not the whole consumer story.**
v1 wrote `Φ⁻¹` and `φ(d)` directly, which `contracts.yaml:396-400` forbids —
*never hardcode the Gaussian Φ* — and BE-Belief adopts a logit recalibration.
Dynamics must consume the selected link's `g_inv` and its typed derivative, or
the architecture must explicitly pin and version a Gaussian-only path law.
Reparameterising a probability through a probit is **not** evidence that the
book follows Gaussian dynamics in `x`.

And `BE_BELIEF_PLAN.md:650` requires `σ_eff` for the **stream-forecast fallback**
when the book is `Unavailable` — "and that is not rare": mean top-of-book age is
12–20 s in the ATM and extreme buckets, because a quiet book emits no events.
So σ is needed for a **level**, not only a shape, on a population selected by
book staleness — whose loss and validation set differ from the main path and
must be scored separately. The fallback **refuses** when its own inputs are
unavailable; it may **not** inherit a book-sourced `d` after the book has failed.
That is why D3 is narrowed rather than carried forward as written.

Q9 is the strongest argument in the corpus for a good σ, and it evaporates the
moment `d` stops depending on σ. Recalibration is required because `Φ⁻¹` of a
miscalibrated mid gives a biased `d`; we have measured that miscalibration
(FLB, −4.3 c at 0.1–0.2, +5.9 c at 0.6–0.7), so it is correctable with no model.

**So σ is for, in priority order:**

1. **H-3, the falsification test.** Does `ln σ̂ − ln σ_book` predict realised
   `|X_T − E_t[X_T]|` out of sample, day-clustered, where
   `σ_book(t) = (E_t[X_T] − K)/Φ⁻¹(mid_t)`? If not, there is no σ edge and
   therefore no p̂ edge, full stop — and σ's remaining role is pure risk plumbing.
   This is the single experiment that could reopen purpose (a). It requires a
   **realised** σ, not a predictive one (§6).
2. **Regime state for sizing and stand-down.** Trailing vol moves **3.4–4.3×**
   p10→p90 *within one coin*. Knowing we are in the top decile is worth acting
   on; knowing σ to ±3 % is not. Accuracy target here is ±25 %, robustness is
   everything.
3. **`σ_⊥` / basis.** Sets the staleness threshold `N` and the crossover below
   which basis noise exceeds settlement vol. At BTC's measured σ=1.089 bps/√s
   and σ_⊥≈0.7 bps, that crossover is **r ≈ 16 s** — below it, our price model
   is noise regardless of how good the estimator is.
4. **The shape** — supplied by `w=60`, fixed, from EXP-M6. Not estimated.

**Candidate (e), the FLB baseline, consumes no σ** — that part stands. But v1
called it "the one measured edge", and **that framing is withdrawn**
(SHOULD-FIX 5). The `+3.6 c/share at p ∈ [0.15, 0.35)` was measured on `book`
snapshots that are p90 6.2 s stale; rebuilt from the executable
`price_change.best_bid/ask` quotes the walk-forward gain is **0.0004 Brier** and
the effect is one-sided — a drift signature, not a bias. The current
`BE_BELIEF_PLAN.md` treats it as a **correctness module**, finds the executable
economics presently indistinguishable from zero, and shows selection destroys
most of the midpoint gap. So: *a small forecast-calibration effect that does not
need this estimator*, and σ work must not block it — but nothing in this plan
may lean on it as an established edge.

---

## 2. The finding that reframes everything: it was the anchor, not σ

All three prior versions set `E_t[X_T] = S60(t)`, the last observed 60 s TWAP.
That is **not** the martingale forecast. For a `w`-second TWAP of a driftless
process:

- for `r ≥ w`: `E_t[X_T] = P_t` — **spot**, not the trailing average. `S60(t)`
  lags spot by ~`w/2 = 30 s`.
- for `r < w`: `X_T` is *partly already observed*.
  `E_t[X_T] = [(w−r)·S_{w−r}(t) + r·P_t]/w`.

Meanwhile the code's variance factor `(r − 2w/3)` is the variance of
`X_T − P_t` — the **spot**-anchored innovation. The numerator was TWAP-anchored
and the denominator spot-anchored. That is a specification mismatch of exactly
the family this programme has flagged three times ("running vs terminal").

We do not observe spot, but we have two Chainlink TWAPs and can nowcast without
importing any Binance basis. Under a locally-linear path, `S30(t)` is the price
at `t−15 s` and `S60(t)` the price at `t−30 s`, so:

```
slope  = (S30(t) − S60(t)) / 15                        per second
P̂_t   = 2·S30(t) − S60(t)                             spot nowcast, basis-free
S_k(t) = P̂_t − (k/2)·slope                            any shorter TWAP
```

and at `r = 30 s` the interpolation is **exact**, needing no linearity
assumption at all: `E_t[X_T] = ½·S30(t) + ½·P̂_t`.

I tested this walk-forward (fit day 20684, test 20685, 1,575 windows, 6
horizons), holding everything else fixed. `OLD` = `S60` anchor, `NEW` = nowcast
anchor; `stat` = one static σ, `roll` = v3's trailing-15 min σ with a fitted
scale:

| r (s) | n | OLD stat | NEW stat | OLD roll | **NEW roll** | book* |
|---|---|---|---|---|---|---|
| 270 | 875 | 0.2410 | 0.2341 | 0.2369 | **0.2303** | 0.2062 |
| 240 | 875 | 0.2238 | 0.2106 | 0.2145 | **0.2019** | 0.1899 |
| 180 | 875 | 0.1896 | 0.1820 | 0.1744 | **0.1683** | 0.1503 |
| 120 | 875 | 0.1447 | 0.1308 | 0.1246 | **0.1078** | 0.1057 |
| 60 | 875 | 0.0839 | 0.0773 | 0.0708 | **0.0662** | 0.0531 |
| 30 | 875 | 0.0432 | 0.0331 | 0.0418 | **0.0279** | 0.0174 |
| **pooled** | 5,250 | 0.1544 | 0.1447 | 0.1438 | **0.1337** | ~0.1207 |

\* book column is from `EXP_RESULTS_2026-08-20.md`, a slightly different
(book-covered) subset — **not paired with these rows**, indicative only.

The **paired**, same-row comparisons are the valid ones, and both are
unambiguous: the anchor fix is worth **−0.0097** (static) and **−0.0101**
(rolling) pooled Brier, improving *every* horizon. It is roughly the same size
as the entire static→rolling-σ improvement (−0.0075) that took two versions to
obtain, and it is obtained from algebra on data we already had.

**And it moves the k puzzle.** The fitted multiplier on the trailing σ̂ falls
from **1.420 → 1.268** purely from correcting the numerator. σ was absorbing the
anchor error. §7.

**Consequence for this plan:** the anchor correction is a *precondition*, not a
deliverable of the estimator. Any σ fitted against the old numerator — including
every number in v1/v2/v3 and the `k≈1.12` finding — is contaminated. Fix the
anchor first, then re-read the "book wins" verdict once. It will very likely
still hold (the residual gap is still book-favouring at all six horizons), but
it must be re-read on a correctly-specified model rather than inherited.

---

## 3. The estimand — stated once, in one equation

**Unit space, frozen (S1/R1).** The model coordinate is the normalised
arithmetic return `x_t = 1e4·(S_t − X_0)/X_0`, in **bps**; `σ` is bps/√s;
`Σ(r)`, `ω_P²` and the nugget are bps²; `d` is dimensionless. `X_0` is the
strike, known at `t0`. Everything below is in `x`, never in raw price and never
in logs — see R1 for why. Implemented and unit-tested in `sigma_kernels.py`;
typed as `UnitSpace` in `contracts.yaml` v13.

Let `w = 60 s` (EXP-M6, fixed, never fitted), `r = T − t`, `x_T` the settlement
60 s TWAP in model coordinates, `x_0 = 0` the strike, and `Ê_t[x_T]` the §2
nowcast-anchored forecast.

> **The estimator targets the conditional settlement innovation variance of the
> forecast error:**
> ```
> Σ(r) ≡ Var_t[ x_T − Ê_t[x_T] ]  =  σ²·k_law(r)  +  σ²·a(r)
>
> k_law(r) = r(r+1)(2r+1)/(6w²)                for r ≤ w      diffusion
>          = (r − w) + (w+1)(2w+1)/(6w)        for r ≥ w
>
> a(r)     = anchor reconstruction coefficient, sigma_kernels.anchor_error_coeff
>          = (r/w)²·a_spot  at r = 30 only;  a_spot  for r ≥ w;  larger between
>
> σ_eff(r) = √Σ(r)          p̂ = G( (Ê_t[x_T] − x_0) / σ_eff(r) )
> ```
> `G` is the **selected `LinkFunction`**, not a hardcoded `Φ`
> (`contracts.yaml:396-400`).

Notes that make this the *right* object rather than a plausible one:

- **It is a conditional terminal variance, not a rolling increment variance.**
  The brief's trap is real and I reproduced its exact magnitudes. The rolling
  TWAP-to-TWAP variogram is `V(r) = σ²(r²/w − r³/(3w²))` for `r<w`; the
  conditional object is `σ²r³/(3w²)`. Their ratio in **σ** is
  `√(V/Σ) = 4.12× at r=10 s` and `2.24× at r=30 s` — matching the brief's
  4.1×/2.2× exactly, which confirms both the derivation and the diagnosis. The
  discrepancy is `X_t`'s trailing window rolling off: information already known
  at `t`. `V(r)` is the right thing to *fit σ to*; `Σ(r)` is the only thing to
  *price with*. They are never interchangeable.
- **The discrete form is used on BOTH branches** (SHOULD-FIX 1). Against the
  continuous kernel it is **+5.1 % variance at r=30 s, +15.5 % at r=10 s** —
  larger than most of the effects we are chasing, and free to include. v1
  applied it only for `r ≤ w` and used the continuous `r − 2w/3` above, which
  made its own table jump 1.25 % in σ at `r = w`. The discrete law is continuous
  there by construction: both branches give `(w+1)(2w+1)/(6w) = 20.5028` s, so
  the `r > w` offset is `r − 39.4972`, not `r − 40`.
- **The variance ledger has exactly two lines, and they are independent.**
  `x_T − Ê_t[x_T] = (future increment) + (reconstruction error)`; the first is a
  post-`t` innovation and the second a functional of the path up to `t`, so they
  add with no covariance term. That independence is a **Brownian-model claim,
  not a type-system fact** (S3) — asserted here, checked by Monte Carlo in
  `sigma_kernels.selftest` (corr −0.0013).
- **The anchor coefficient is `a(r)`, not `(r/w)²·ω_P²`.** v1's damped form is
  right only where the trailing part of the mark is *observed* (`r = 30`, where
  it is `S30`) or absent (`r ≥ w`). Strictly inside the window the forecast also
  needs the trailing `(w−r)`-second TWAP, reconstructed from the **same** S30/S60
  pair, so the two errors are correlated and add: at `r=45`, `a = 6.64σ²` against
  the damped form's `5.35σ²`, a 24 % understatement. Exact on today's grid,
  wrong as soon as it densifies.
- **The damping does NOT make the nowcast free late in the window.** v1 claimed
  it did. anchor/diffusion `= 3a(r)/k_law(r) ~ 1/r`, so the anchor share *rises*
  toward expiry: **4.0 % of variance at `r=270`, 47.5 % at `r=30`** — precisely
  the horizon where 1 bp is ~20 probability-cents.
- **`ω_P` replaces `σ_⊥` for the anchor; it is never added on top of it, and
  never added on top of `κ(r)`.** `κ(r) = 1 + σ_⊥²/σ_bin²` already contains the
  residual. This programme has committed the double-count three times
  (`σ_⊥+κ`, `v(t)` sum-vs-min, running-vs-terminal). **Treat any instinct to add
  a second variance term as suspect**; the ledger above is closed, and any
  addition to it requires a written justification of why it is not already
  inside `σ²` or `ω_P²`.
- **`ω_P` is bracketed, not point-identified** (S3). v1 said "measure it against
  the Binance feed, then subtract the known basis". That does not identify it:
  latent Chainlink spot is never observed, and time-varying basis plus Binance
  proxy error stay in the residual. Carried as `AnchorErrorBudget{floor,
  ceiling, identified=false}`:
  - **floor** — model-implied and exact, `a_spot = 9.5139σ²`, i.e.
    `ω_P = 3.0845σ` (**3.36 bps** for BTC). This is what `P̂ = 2·S30 − S60` costs
    with perfect, synchronous, noiseless feeds, purely from extrapolating. It is
    a floor because it assumes the feeds away.
  - **ceiling** — the empirical S30/S60-vs-Binance residual, basis-contaminated,
    hence an upper bound.
  Consumers propagate both ends until a two-proxy or error-in-variables estimate
  exists. For scale: the **old S60 anchor's** coefficient is `19.5028σ²`, so the
  nowcast halves anchor variance — and v1–v3 omitted the line entirely, which is
  §7's mechanical source 2 and 65 % of total variance at `r=30`.
- **Strike error is a third, separate line** for `t ∈ [0, Δ_K]`, `Δ_K ≈ 1.7–2.7 s`,
  and `Cov_t(X_0, X_T) ≠ 0` in the pre-open branch. Both are out of scope for
  the first estimator and are listed in §8.
- **No annualisation.** Report `σ` in bps/√s and `σ_eff` in bps at each `r`.
  For BTC at the measured 1.089 bps/√s:

  | r (s) | 30 | 60 | 120 | 180 | 240 | 270 |
  |---|---|---|---|---|---|---|
  | diffusion (bps²) | 3.12 | 24.32 | 95.47 | 166.63 | 237.78 | 273.36 |
  | anchor (bps², at the floor) | 2.82 | 11.28 | 11.28 | 11.28 | 11.28 | 11.28 |
  | **σ_eff (bps)** | **2.44** | **5.97** | **10.33** | **13.34** | **15.78** | **16.87** |
  | anchor share | 47.5 % | 31.7 % | 10.6 % | 6.3 % | 4.5 % | 4.0 % |
  | *v1 table (superseded)* | *1.77* | *4.87* | *9.74* | *12.88* | *15.40* | *16.52* |

  v1's row omitted the anchor line and mixed kernel conventions at `r=w`.
  Regenerate with `python3 live/pm_research/sigma_kernels.py --selftest`; the
  anchor row is the **floor**, so these `σ_eff` are lower bounds. At r=30 s,
  **1 bp of anything in the numerator is 0.41 in d-units, ≈ 16
  probability-cents** (v1 said 0.57 and ~20 c, computed against the understated
  `σ_eff`). That is still the real reason the anchor mattered more than σ.

---

## 4. Option comparison, and the recommendation

### A. Empirical settlement-innovation variance
Take completed windows, compute `X_T − Ê_t[X_T]` at each `r` on the grid, and
use the trailing sample variance.

*Pro:* model-free; targets `Σ(r)` exactly; needs no `w`, no BM assumption, no
kernel; immune to the ŵ≠60 problem; and it is the only method that can detect
that the parametric law is wrong.
*Con:* **it is starved.** ~190–239 windows per symbol *in total*, on 6 grid
points, from 2 days. Even treating windows as iid, a variance estimate has
SE ≈ √(2/190) ≈ 10 % in variance / 5 % in σ; but they are not iid — vol clusters,
288 windows a day ride one path, and the honest unit is the **day**, of which we
have two. Realistic SE is 20–30 %. It also cannot say anything about the current
regime — it is a trailing average by construction — and it produces no estimate
between grid points.

### B. Parametric variogram `V(r; σ², w)` fitted to increments
Fit the TWAP-of-BM law to the tape.

*Pro:* **it is where the information is.** 63 k ticks/symbol/day against 190
binary outcomes. My fits recover the shape to 3.8–10.6 % rmse across all eight
symbols, and BTC's implied σ is flat to 1.06× over r ∈ [10, 300] s — the law is
not an assumption we are tolerating, it is a description the data endorses. It
yields `ŵ` as an independent second check on EXP-M6, and it can be re-fit on any
trailing window, so it delivers the regime variation option A cannot.
*Con:* it assumes the shape. And the by-product check is *not clean*: ŵ ranges
47–81 s across symbols, BTC's 47 s sitting 13 s below the settlement `w`. If real,
that is a `(60/47)² = 1.63×` variance bias on the in-window branch — precisely
where we quote.

### C. Hybrid
*The brief's C* is empirical shape × regime-scaled level. I recommend a
**different composition**, because the diagnostics say the shape is the
*strong* part and the level's *time variation* is the weak part — the reverse of
what that composition assumes. See below.

### D. Regime conditioning
Not a rival option — a requirement any of A/B/C must satisfy. The measured
p90/p10 of 3.4–4.3× within a coin makes a static σ indefensible (constraint 4).
But the honest strength of the signal is modest: ρ(trailing σ̂, forward squared
innovation) = **+0.19 to +0.40**, and the **60 min window is at least as good as
the 15 min in 6 of 8 symbols**. Vol here is persistent but not sharply
forecastable, which caps what regime conditioning can deliver and directly
refutes v3's `w_fast=0.9`.

### Recommendation

> **B for the level and its time-variation; A for a horizon calibration
> multiplier; `w=60` fixed throughout.**
>
> ```
> Σ̂(r) = c(r) · [ σ̂²_blend · k_law(r; w=60) ] + weight(r) · ω̂_P²
> ```
> where `k_law` is the exact discrete kernel of §3, `σ̂²_blend` is a per-symbol
> multi-scale variogram estimate, and `c(r)` is an empirical multiplier ≈1
> absorbing everything the parametric law gets wrong at horizon `r`.

Composition, precisely, so nothing is double-counted:

1. **Level (fully per-symbol, no shrinkage).** Fit `V(r; σ², nugget)` with
   `w ≡ 60` **held fixed** on non-overlapping increments, WLS in log space, over
   `r ∈ [10, 600] s`. Per-symbol power is ample (63 k ticks/day), so this
   parameter is per-symbol with no pooling whatsoever. Emit `ŵ_free` from a
   *separate* free-`w` fit as a **diagnostic only** — never as an input.
2. **Time-variation (≥3 scales, constraint 4).** Fit the same variogram on
   trailing windows of **15 min, 1 h, 4 h** (and 5 min for BTC only, where tick
   density supports it), and combine as a weighted average of `σ²`, HAR-style.
   **Weights are fitted on the tape, not on outcomes**, by QLIKE loss against
   the next 300 s realised innovation — thousands of observations per symbol per
   day instead of 190 Bernoullis. This is the decisive reason v3's `w_fast=0.9`
   was noise: it was fitted on the wrong data.
3. **Calibration `c(r)` (option A's job).** For each `r` on the grid, compare
   the realised standardised innovation variance to the law's prediction over
   completed windows. `c(r)` absorbs the ŵ≠60 gap, jumps, oracle quantisation
   and discreteness residual, *without* letting any of them corrupt the level.
   This is where option A's model-freedom is worth its low power: it is
   estimating a **small correction around 1**, not a level.
   **Correction (S6): that does not buy the precision v1 claimed.** v1 wrote
   "20–30 % SE on the correction is 5 % on `Σ`". It is not: a 25 % relative SE
   on a variance multiplier whose mean is ~1 is a 25 % SE on `Σ` and ~12 % on
   `σ_eff`, regardless of where its mean sits. Any quoted SE must name its
   object — `c`, `c−1`, variance or volatility — and `c(r)` stays `DIAGNOSTIC`
   until the day clusters exist.
4. **Per-symbol, with declared shrinkage (constraint 3).** Every parameter has a
   per-symbol value. But `c(r)` and the blend weights cannot honestly be
   estimated from 190 windows/symbol, so they are **hierarchical**: a per-symbol
   posterior shrunk toward the pooled value, with the shrinkage weight
   *computed* from measured between-symbol dispersion vs within-symbol SE, never
   chosen. **Report the realised shrinkage weight per symbol** so a reviewer can
   see how much is genuinely per-symbol. Pooling as a *prior* is legitimate;
   pooling as a *parameter tie* (v3's error) is not.
5. **Two outputs, not one.** `PM_QUANT_REVIEW.md` Q9 proves the loss is
   asymmetric: under-estimating σ is the dangerous direction. Emit
   `(σ̂, SE(log σ̂))`, and let risk consumers use `σ_hi = σ̂·exp(z·SE)`. With §1's
   book-sourced `d` this matters far less — which is the point of that
   recommendation — but any consumer that must use *our* `d` uses `σ_hi`.

---

## 5. Link function

**Decision: do not fit a link. Measure it, then floor it.**

With a realised σ from the tape we can standardise: `z = (X_T − Ê_t[X_T])/σ_eff(r)`
over all completed windows and horizons. The empirical CDF of `z` **is** the
link. So:

- **Body (|z| < 2):** use the empirical/PIT-calibrated CDF, checked for
  uniformity by horizon and by symbol. This is estimable now — but **the fit
  must be walk-forward** (SHOULD-FIX 3). A pooled empirical CDF checked on the
  same two days it was estimated from is descriptive fit, not calibration.
- **Tail (|z| ≥ 2):** **not estimable on this sample and we should stop
  pretending otherwise.** ~190 windows/symbol puts perhaps 4–9 observations
  beyond |z|=2 per symbol, and none beyond |z|=3. Any Student-t `ν` fitted here
  is fitted to single-digit counts.
- **Resolution: pre-commit `ν = 4` as a declared conservative POLICY assumption
  — not "a standard crypto value" (SHOULD-FIX 2). Owner: BE-Uncertainty;
  sensitivity range ν ∈ [3, 6] reported. The `p̂` floor is a risk/decision
  policy and must never be read back as evidence about the link.** Impose a hard
  `p̂` floor at half a tick, 0.005. The floor binds at `|d| > 2.58`, which is
  exactly where the Gaussian tail becomes untrustworthy. Beyond that point the
  floor governs and **the link choice is irrelevant for every risk consumer** —
  which converts an unanswerable statistical question into a policy decision. It
  also kills the `Q_max = 9.9e15` pathology (`p̂ ≈ 2.5e-13`, `|d| ≈ 7.2`): no
  free-data model supports a probability statement at 1e-13.
- Do **not** fit `ν` by outcome-MLE. It will trade off against σ (both control
  spread), and with `SD ∝ 1/√days` at 2 days the trade-off is unidentified.
- **How we would test it, at 30 days:** PIT histogram by horizon with
  day-clustered CIs, plus a pre-registered Gaussian-vs-t(4) comparison **scored
  only on the body**, since the tail cannot be adjudicated. Report the tail as
  "assumed, not measured".

---

## 6. What the estimator forecasts — named correctly (S5)

**The live output is an ex-ante FORECAST OF FUTURE PHYSICAL VARIANCE, fitted on
the tape. Outcomes are never used to fit it.**

v1 called this "realised σ". That was wrong and it blurred the exact PIT
boundary this plan exists to protect: at decision time the next 300 s of
variance is unknown, so a trailing estimator whose weights minimise QLIKE
against the **next** 300 s realised variation is a forecast, and the future tape
quantity is its ex-post *target*, not its output. **"Not fitted to binary
outcomes" and "not predictive" are not synonyms.** Every artefact must carry
`forecast_as_of`, `target_start`, `target_end` and `fit_data_through`
(`PathLaw`, `contracts.yaml` v13), and the estimator refuses rather than
extrapolating past `fit_data_through`.

The distinction v1 was reaching for is real and survives under the right name:
the estimator is fitted on the **tape**, not on Bernoulli winners. That is what
the reasons below establish.

Reasons, in order of force:

1. **Information geometry.** `I(d) = [φ(d)d]²/(p(1−p))` is **zero at d=0**,
   peaks at |d|≈1.5, and integrates to ~0.31 units per window. ~320 windows are
   needed for `SE(log σ) = 0.10` and ~1,290 for 0.05 — and critically
   **`SD ∝ 1/√days`, not `1/√windows`**. At 2 days, an outcome-MLE cannot
   estimate *one pooled scalar* to better than tens of percent, let alone
   7 per-symbol levels × 3 blend weights × 6 calibration multipliers.
2. **Pseudo-replication.** There is exactly **one** `y` per window. v3 stacked 6
   decision rows per window as independent observations; a per-second
   implementation would stack ~300, inflating apparent information up to 300×.
   Every outcome-based number in v1–v3 is overstated by roughly √6.
3. **Circularity.** If the book already embeds a good σ, an outcome-MLE simply
   recovers the book's implied σ, and agreement gets scored as skill.
4. **It is the wrong object for the surviving purposes.** H-3 asks whether our
   σ̂ predicts *realised* `|X_T − Ê_t[X_T]|` better than `σ_book` does. A
   predictive σ fitted to outcomes has already absorbed the answer.

**What predictive σ is used for:** exactly one thing — as a **diagnostic**. The
ratio `k = σ̂_pred/σ̂_realised` is a *specification-error detector* (§7), reported
per horizon and per symbol, never applied as a multiplier. The two objects are
different and cannot be composed; the moment `k` is multiplied into `Σ̂`, the
estimator stops being realised and H-3 becomes untestable.

---

## 7. The k ≈ 1.12 puzzle

Three mechanical forces push `k > 1` *before* any economic story:

1. **Anchor error.** Demonstrated, not hypothesised: correcting the numerator
   moved `k` **1.420 → 1.268** on identical rows (§2). A lagging forecast is a
   noisier forecast, and MLE compensates by inflating σ toward 0.5.
2. **Measurement noise in the numerator.** With measurement error `ω`, an
   outcome-MLE returns `√(σ² + λω²)/λ` — **1.37× at ω = 0.5σ**. Mechanical.
3. **Gap selection bias.** Skipping gap-containing intervals (our tape has
   Δt up to 44 s) selects *against* high-vol regimes and biases realised σ̂
   **down**, inflating `k` from the denominator. Same direction.

And one force pushes `k < 1`: link misspecification biases outcome-fitted σ̂
**down 3.5–15 %**.

**Treatment: decompose, never multiply.** Build the ledger — anchor,
`ω_P`, strike error, gap selection, link — and see how much of `k` survives. My
prior is that little does: `k` fell 11 % from one specification fix, and two more
named mechanical sources remain unquantified. An "implied-over-realised premium"
is the *last* explanation to reach for, not the first, and at 2 days we could not
distinguish a 12 % premium from a 12 % estimator bias in any case. If a residual
`k` survives the full ledger at 30 days, *then* it is a finding — and it belongs
in the H-3 test as evidence of a σ edge, not in `Σ̂` as a fudge factor.

---

## 8. Validation protocol, honest at n ≈ 2 days

**Unit of inference.** Tape-based claims: the **window**, with a block bootstrap
over contiguous blocks. Outcome-based claims: the **day**, of which there are
two. Never the decision row. (The same lesson the CTA programme learned: overlap-
aware CIs, block bootstrap for multi-day horizons.)

**Effective sample size, not tick count (S5).** v1 repeatedly argued power from
"63 k ticks/symbol/day". That number is **not** the sample size for anything we
fit. Non-overlapping 600 s increments give ~**144 units per full day**, all
horizons ride the same path, and 1 s labels for next-300 s variation overlap
299/300 of their support — many numerical rows, few independent observations.
Every power claim below is stated in effective sample size by horizon, and the
fit protocol is: train through day `d−1`, **day-block test folds**, embargo
covering the longest label support, and either non-overlapping targets or
overlap-aware weights with block inference. **No blend weight may be called
per-symbol stable from the present two-day, one-regime sample.**

### What can be claimed now (2 days)
- **The variance law's shape.** Testable at ~15 horizons; the honest unit is
  ~144 non-overlapping 600 s increments/symbol/day, not the tick count. The
  1.06× flatness of BTC's implied σ over r ∈ [10,300] s is a real result.
- **Relative σ levels across symbols** (0.94→4.33 bps/√s is far outside noise).
- **Paired, same-row specification comparisons** — e.g. the anchor fix. Pairing
  removes the day-level variance that makes absolute claims impossible. This is
  the *only* outcome-based inference with usable power at n=2 days, and it should
  be the workhorse.
- **PIT uniformity in the body**, pooled across symbols, as a coarse check.

### What cannot be claimed now
- **Any Brier difference versus the book.** One test day; the day-clustered SE
  is undefined. `EXP_RESULTS`'s own caveat 1 already says this and it binds here
  too. The `NEWroll − book` column in §2 is **indicative, not paired, not
  significant**.
- **Tail calibration beyond |z| ≈ 2.5.**
- **The level of `k`.**
- **Any per-symbol difference in blend weights or `c(r)`** — hence the mandatory
  shrinkage.
- **That the anchor fix persists.** One day. It is algebraically motivated,
  which is why I trust it more than a fitted improvement of the same size — but
  it is still one day.
- **That "the book wins" is settled.** It was measured on a mis-anchored model.
  Re-read it once, on the corrected specification, before treating ŵ→0 as final.
- **Anything from the pre-repair book sample — and pairing does not rescue it**
  (S4). v1 said the re-read "is paired, so it survives the MNAR gap". **That
  sentence is struck.** Pairing makes the two forecasts share the same
  *observed* rows; it does not recover the busy BTC intervals the slow-consumer
  failure dropped (27 of 47 disconnects self-inflicted, 32 of 47 BTC). If
  relative model/book performance varies with volatility, staleness or activity
  — which is the whole hypothesis — conditioning on observed rows biases the
  paired delta. The anchor comparison stays valid because it is paired on the
  **settlement tape**, which did not have the gap; the book-beat verdict does
  not. It needs the dense knowledge-time top-of-book rebuilt from
  `price_change.best_bid/ask`, cause-stamped gaps, gap-complete or post-repair
  units only, selection deltas reported by activity/volatility, and a
  recalibrated-book baseline alongside the raw book.

### At 7 days
Fit blend weights per symbol on the tape (ample power there); PIT by horizon and
symbol; per-symbol level ratios with block-bootstrap CIs; `c(r)` pooled with
per-symbol shrinkage. Outcome-based day-clustered CIs with 6 test days remain
**descriptive** — report them, do not gate on them. **Pre-register the estimator,
the horizon grid and `w=60` before the 7-day read.**

### At 30 days
`SE(log σ) ≈ ±3 %` for one level parameter becomes reachable. Run **H-3**
properly: (i) agreement — regress `ln σ̂_eff` on `ln σ_book`, slope and R²;
(ii) direction — does `ln σ̂ − ln σ_book` predict realised `|X_T − Ê_t[X_T]|`
OOS, day-clustered?; (iii) control — p̂ must beat the **isotonic-recalibrated**
book mid, not the raw mid. Link body test Gaussian vs t(4). Tail still assumed.

### Gates
- **G1 (shape):** implied σ flat within ±15 % over r ∈ [10, 300] s, all symbols.
  *Already passes for BTC (1.06× over [10,300]); rmse 3.8–10.6 % pooled.*
- **G2 (calibration):** PIT uniform in |z| < 2 by horizon, per symbol after
  shrinkage, at 7 days.
- **G3 (ledger closed):** `ω_P` bracketed (floor/ceiling, `identified` declared);
  `c(r)` within [0.8, 1.25]; no variance component appears twice. A `c(r)`
  outside that band means the parametric law is failing, not that the multiplier
  needs widening. **Inference rule, frozen before the next read (S6):** `c(r)` is
  the residual left *after* the ledger,
  `c(r) = [Var(e_r) − a(r)·σ²·ω_scale] / [σ̂²·k_law(r)]` — it multiplies the
  **diffusion line only**, never the whole ledger, and the prose may not permit
  both readings. Its `status` is **`DIAGNOSTIC`, not `GATE`, until ≥10
  independent day clusters exist**; a point estimate crossing the band at 2 day
  clusters is not a go/no-go and will not be treated as one. Report the interval,
  and state explicitly that any quoted SE is on **`c` in variance units** —
  20–30 % relative uncertainty in a variance multiplier does **not** become 5 %
  merely because its mean sits near 1, which is what v1 §4-3 claimed.
- **G4 (H-3, at 30 days):** if the direction test is null, **σ is declared risk
  plumbing permanently** and no further σ-as-alpha work is funded. **Domain rules,
  pre-registered (S6):** `σ_book = (Ê_t[x_T] − x_0)/g⁻¹(mid_t)` is undefined near
  `mid = 0.5`, can go negative when book and stream disagree in sign, and
  explodes under tick quantisation. Fix an admissible moneyness band before
  looking; use the selected link, not `Φ⁻¹`; handle sign conflict and censoring
  explicitly rather than conditioning away the hard cases; score the **forecast**
  against future physical variation OOS; and use the **recalibrated executable
  book** as the control, not the raw mid.

---

## 9. Ways this design could still be wrong

1. **The nowcast could be worse than it looks.** `P̂ = 2·S30 − S60` assumes local
   linearity over 60 s and that both feeds sample the *same* path synchronously.
   Chainlink has heartbeats, deviation thresholds and multi-source aggregation;
   if S30 and S60 update on different triggers, the extrapolation amplifies noise
   faster than it removes lag. **This is now quantified on one side:** the
   model-implied floor is `ω_P = 3.08σ` (3.36 bps for BTC), i.e. the
   extrapolation already costs ~10 s of diffusion *before* any feed pathology.
   **Mitigation:** carry the floor/ceiling bracket, not a point estimate; the
   `r=30 s` case (`½S30 + ½P̂`) needs no linearity assumption and can be checked
   separately. If the ceiling comes in far above the floor, fall back to the
   `r=30 s`-exact form and interpolate conservatively.
2. **S30/S60 may not be exactly trailing 30/60 s means.** The aggregation is
   undocumented. If not, the `r=30 s` identity breaks and item 1's fallback goes
   with it. **Check first**, by reconstructing S30 and S60 from the 1 s Binance
   tape and regressing.
3. **ŵ ≈ 47 s for BTC.** Most likely nugget confounding — a variogram fitted
   without a nugget attributes microstructure variance at small `r` to a shorter
   kernel, and my nugget grid was coarse (0 for BTC). But if it is real, the
   in-window level is biased by up to 1.63× in variance. `c(r)` absorbs it *if*
   `c(r)` is well estimated, which at 2 days it is not. **This is the largest
   open technical risk in the design.** Resolve with a proper joint
   `(σ², w, nugget)` WLS fit on non-overlapping increments before trusting any
   in-window number.
4. **Two days is one regime.** A 2-day rally with +4.5 pp up-drift. Every
   regime-conditioning weight is fitted inside a single regime, and the anchor
   improvement could be drift-flattered.
5. **Vol persistence may be too weak for a 3-scale blend to earn its
   parameters.** ρ = 0.19–0.40 is real but modest, and the 60 min window already
   matches the 15 min. Constraint 4 mandates ≥3 scales; the evidence supports
   perhaps two. **Test the ≥3-scale blend against a single 60 min window and
   report the delta honestly** — if it is nil, say so rather than shipping
   parameters to satisfy a constraint.
6. **`hype` may not be the same process** (σ 4.33, ŵ 81 s, largest nugget). The
   shrinkage prior could actively harm it. Consider excluding it from the pooled
   prior.
7. **Gap handling can only bias σ̂ down** — the dangerous direction. My 1 s grid
   forward-fills across gaps up to 44 s, which understates short-`r` variance.
8. **Book-sourced `d` inherits the book's miscalibration.** `Φ⁻¹` of a mid we
   have *measured* to be 3–6 c miscalibrated gives a biased `d`. Isotonic
   recalibration is mandatory, and it is itself fitted on 2 days.
9. **Double-count risk is not eliminated, only currently closed.** `ω_P` is a new
   variance line. If anyone later adds `σ_⊥` or `κ(r)` on top, that is the
   fourth instance of this programme's signature failure. **Reconciled per
   SHOULD-FIX 6:** §1 lists `σ_⊥`/basis as a surviving σ *job* (it sets the
   staleness threshold `N` and the `r ≈ 16 s` crossover) while §11 defers it to
   PM-E1 and §3 excludes it from the settlement ledger. Both are correct and the
   separation is deliberate — `σ_⊥` is a **basis** object, not a settlement
   variance component. It is registered under a distinct owner in the
   `VarianceGroup` registry (R-ONCE) so no future consumer can add it to
   `Σ(r)`.
10. **Demoting σ-for-level could be wrong.** It rests on a book-beat comparison
    run against a mis-anchored model on one test day **and on an MNAR book
    sample** (S4). H-3 is the named re-entry point; the plan should not be read
    as closing that door, only as declining to fund it until H-3 reads.
11. ~~**The architecture has nowhere to put the answer.**~~ **FIXED in
    `contracts.yaml` v13** (M13-1). v1 flagged this and then scheduled the fix
    last, after the variogram, blend, calibration and link — which reversed the
    modularity goal (S2). A `BE-Uncertainty` module now exists, and `PathLaw` is
    a typed carrier with `unit_space`, `horizon_domain`, `fit_data_through`, the
    `AnchorErrorBudget` bracket, the `CalibrationCurve` status and a
    `settlement_var`/`increment_var` protocol that **refuses** out of domain. It
    replaced `{kind: str, params: dict[str, float]}`, which could not have held
    any of that. *Remaining gap, out of scope here:* `BE-Belief` itself is still
    absent from the `modules:` block — that belongs to the structure review loop,
    not to this plan.
12. **The `r = 30 s` "exact" claim was too strong** (S3). The *decomposition*
    into known and future halves is exact. Replacing latent spot with
    `P̂ = 2·S30 − S60` still assumes compatible feed windows, common event
    support, synchronisation and a locally linear path — and the extrapolation
    amplifies asynchronous updates. Exactness of the algebra is not exactness of
    the estimate.
13. **The unit-space choice is a live assumption, not a free lunch.** Normalised
    arithmetic returns keep every identity exact (R1), but they assume the
    per-window reference `X_0` is known and correct at `t0`; strike
    reconstruction error for `t ∈ [0, Δ_K]` therefore enters as a *coordinate*
    error, not only as the separate ledger line of §3.

---

## 10. BTC first — what generalises

**Generalises across all seven traded symbols:**
- The variance law's functional form. All eight symbols fit with 3.8–10.6 % rmse.
- The estimand `Σ(r)`, the trap, and the two-line variance ledger — algebra.
- The anchor correction. Purely algebraic in the two feeds; nothing coin-specific.
- `w = 60 s` — a venue settlement convention, not a market property. (Still worth
  reading the EXP-M6 per-coin breakdown; it was reported pooled at 99.8 %.)
- The estimator architecture, the link/floor policy, the validation protocol.

**Does not generalise — must be per-symbol:**
- **The σ level.** 0.94 → 4.33 bps/√s, 4.6×. Never pool.
- **`ŵ_free`.** 47 → 81 s. If item 9-3 turns out to be real rather than nugget
  confounding, `c(r)` is genuinely per-symbol and the shrinkage prior is harmful.
- **The nugget.** 0.00 for btc/eth/xrp/doge/bnb; 0.14–0.28 bp for sol/zec/hype —
  a microstructure difference, plausibly tick-size driven.
- **Regime persistence.** ρ 0.19 (btc) → 0.40 (hype). BTC is the *hardest* symbol
  to regime-condition, so BTC-first is a conservative ordering: whatever works on
  BTC should work better elsewhere.
- **The fast/slow balance.** 60 min beats 15 min for bnb/btc/doge/eth/hype/zec;
  15 min beats 60 min for sol/xrp. Genuinely per-symbol, and thin.

**A useful control, not an independent OOS asset:** `zec/usd` has 63,159 price
ticks but **no Polymarket market**, so it cannot contaminate the outcome
analysis and is worth using for tape-side claims (shape, level, `ŵ`, nugget,
regime persistence). But it is **not statistically independent** (SHOULD-FIX 4):
it shares the same dates, the same crypto regime and the same
estimator-selection process as the traded symbols. Treat a ZEC pass as evidence
against a coding error or a symbol-specific artefact, **not** as out-of-sample
confirmation of the law.

---

## 11. Deferred until more data

| item | why | unblocked at |
|---|---|---|
| Fitting `ν` / any tail parameter | <10 obs beyond \|z\|=2 per symbol | 30+ days |
| Per-symbol `c(r)` without shrinkage | ~190 windows/symbol | 30+ days |
| Any claim about `k` as a premium | 3 mechanical sources unquantified | 30 days + closed ledger |
| H-3 σ-edge test | needs day-clustered CI over ≥10 test days | 30 days |
| Pre-open branch (`−w < t < 0`), `Cov(X_0, X_T) ≠ 0` | not on the current decision grid | when quoting opens pre-window |
| Strike-reconstruction variance for `t ∈ [0, Δ_K]` | separate estimand, small | with the pre-open branch |
| `σ_⊥` / basis variogram → staleness `N` | needs paired Chainlink–Binance study | PM-E1 |
| Intra-window σ updating | ρ≈0.2–0.4 says the gain is small | after the 3-scale blend is shown to earn its keep |

---

## 12. Build order — REVISION 2 (definitions → admissibility → machinery)

Supersedes v1's five-step Phase 0. Estimator implementation remains on **HOLD**
until Phase 0A closes. Steps 1, 3 and 9 of v1 are **done**; the rest are
re-sequenced per the review, because v1 implemented the anchor *before*
verifying the feed semantics it depends on, and deferred the typed carrier to
last.

**Phase 0A — definitions and deterministic checks (no data required).**

1. ~~Freeze the unit space~~ **DONE** — normalised arithmetic returns, R1,
   frozen in `sigma_kernels.py` and typed as `UnitSpace`.
2. ~~Freeze the typed output boundary~~ **DONE** — `contracts.yaml` v13:
   `BE-Uncertainty` + `PathLaw` + `AnchorErrorBudget` + `CalibrationCurve`,
   with refusal behaviour and both migration records; `contract_check.py` passes
   with zero unexplained changes.
3. ~~Derive and unit-test the discrete kernels~~ **DONE** —
   `sigma_kernels.py --selftest`, 13 checks including continuity at `r = w`, the
   general anchor coefficient `a(r)`, and MC agreement on `ω_P`.
4. **Freeze the consumer matrix**, including the BE-Belief fallback as a
   *level* consumer with its own scoring population, and route every `Φ`/`φ`
   through the selected `LinkFunction`. Partially done in §1; the fallback's
   loss function is not yet written.
5. **Verify S30/S60 window semantics, timestamp alignment and common
   knowledge-time construction** against the 1 s Binance tape — **before** the
   anchor helper is written (S3). If they are not synchronous trailing
   arithmetic means, the helper is wrong by construction and item 6 changes.
6. **Implement the anchor** in a shared helper, only after 5 passes. Emit the
   `AnchorErrorBudget` bracket (floor already computed; measure the ceiling) and
   the declared covariance policy.

**Phase 0B — data admissibility and feasibility.**

7. **Rebuild the dense knowledge-time top of book** from
   `price_change.best_bid/ask`; classify gaps by cause; isolate post-MNAR-repair
   data and mark protected spans (S4).
8. **Fit the frozen per-symbol single-scale physical-vol baseline** on tape
   folds — day-block, embargoed, overlap-aware. This is the incumbent.
9. **Measure the complete variance ledger, then `c(r)`**, with block/day
   uncertainty. `status: DIAGNOSTIC`. The present two-day point estimate is
   **not** a gate (S6), and D2's predicted breach is retired.
10. **Re-read fallback calibration and model-vs-book scoring** on admissible
    dense-book rows only, against both the raw and the recalibrated book. Label
    `DESCRIPTIVE` until the day-cluster threshold is met.

**Phase 1 — machinery, only if Phase 0 is coherent.**

11. **Joint `(σ², w, nugget)` WLS variogram**, non-overlapping, `r ∈ [10,600] s`,
    per symbol. Report `ŵ_free` as a diagnostic; resolve §9-3 — still the
    largest open technical risk.
12. **Multi-scale QLIKE challenger** on *identical* embargoed folds, judged
    against the step-8 baseline (D3). If it does not beat it out of sample, it
    does not get built.
13. **Shrinkage only where between/within-symbol evidence supports it**;
    walk-forward PIT / G2; then the `p̂` floor and link policy.
14. **H-3 at the pre-registered horizon** with the §8 domain rules. A null
    permanently demotes σ to risk plumbing (G4).

**What can still stop this.** S30/S60 semantics failing at step 5 invalidates
the anchor and the nowcast together. Step 10 overturning "the book wins" would
re-scope σ, since §0's argument assumes the book supplies the level. Neither is
`c(r)` — that is now a diagnostic, not a gate, until the day clusters exist.

Phase 0 is worth doing regardless of whether Phase 1 is ever adopted. Per D3 and
the review's closing recommendation, effort saved here goes to **G-FF4, the
queue bracket**, which can still end the programme.
