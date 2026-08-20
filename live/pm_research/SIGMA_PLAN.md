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

## FINALIZED 2026-08-20 — decisions on top of the draft

Adopted as written except for the three items below. Rationale in the review
that accompanied this commit.

**D1 — Build order inverted: feasibility BEFORE machinery.** The draft built the
variogram and blend first, then measured `c(r)` and `ω_P`. Both of those are
cheap, run on data already collected, and **either can invalidate the design**,
so they move to the front. See the revised §12.

**D2 — `c(r)` is expected to breach G3, and that is the informative outcome.**
Two independent lines say `c(r)` will exceed the `[0.8, 1.25]` band at short r:
the §3 table gives `σ_eff(30) = 1.77 bps` while the measured realised innovation
at r=30 was 4.69 bps under the old anchor (even granting the nowcast's 45%
reduction that lands near 2.6 bps, i.e. `c(30) ≈ 2.1` in variance); and BTC's
free-fitted `ŵ ≈ 47 s` against the fixed 60 s implies a 1.63× in-window bias.
The plan's own rule stands — *a `c(r)` outside the band means the parametric law
is failing, not that the multiplier needs widening* — so the likely first result
is **redesign, not calibration**. Budget for that rather than being surprised.

**D3 — Size the build to σ's actual job.** §0 scopes σ down to risk plumbing:
the two edge-bearing consumers use no variance, and the dynamics consumers take
`d` from the book. If D1's measurements land in-band, a **single-scale
per-symbol estimate may suffice**, and the full multi-scale blend with
hierarchical shrinkage should be justified against that baseline before it is
built. Effort saved goes to G-FF4 (the queue bracket), which can end the
programme.

Unchanged and adopted: the estimand and its closed two-line variance ledger;
`w=60` fixed with `ŵ_free` as diagnostic only; tape-fitted (not outcome-fitted)
blend weights; the discrete kernel; measure-then-floor on the link; the
`(σ̂, SE(log σ̂))` two-output for asymmetric loss; and the validation table —
including its statement that **"the book wins" is not currently claimable**
because it was measured on a mis-anchored model.

---

## 0. Executive answer

1. **σ is not for the level of p̂.** Of the twelve places the design consumes σ,
   the two with a measured edge (FLB harvest, inventory cap `L_adv=|q|(1−p̂)`)
   consume **no variance at all**, and the four dynamics consumers
   (participation frontier, `r*`, `ζ` pickoff floor, `λ_bin`) need only the
   variance **shape** plus `d` — and `d` can be taken from the book.
2. σ's surviving jobs are: **(i)** the H-3 falsification test that is the only
   route back to a p̂ edge; **(ii)** a coarse **regime/state** level for sizing
   and stand-down; **(iii)** the **basis** term `σ_⊥`, which sets the staleness
   threshold and the horizon below which our whole price model is noise;
   **(iv)** the **shape**, which is *not estimated* — it is `w=60` from EXP-M6
   plus the exact discrete kernel.
3. **A large share of the model's deficit to the book was never a σ problem.**
   It was a forecast-anchor error: `E_t[X_T]` was set to the trailing 60 s TWAP,
   which lags spot by ~30 s. Correcting the anchor (§2) improved paired Brier at
   **every** horizon and cut the outcome-MLE inflation factor from 1.42 to 1.27.
   Three σ generations were chasing the wrong term.
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
| FLB harvest (+3.6 c/share at p∈[0.15,0.35)) | **none** (book-bucket conditioned) | neither | **yes — the one measured edge** |
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

> **Take `d` from the book, not from us: `d_book = Φ⁻¹(p̃_book)`, where `p̃` is
> the isotonic-recalibrated mid.** Then the frontier `m/φ(d) ≥ k√(3L/r)`,
> `r*`, `ζ` and `λ_bin` need only `w=60` (verified, not estimated) and `r`.
> They become **genuinely σ-free**, and the Q9 failure mode — under-estimating
> σ by 2× at |d|=1 leaves us quoting **52.9 s deeper** into the sniping zone —
> is *deleted rather than mitigated*.

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

**Candidate (e), the FLB baseline, consumes no σ.** Stated plainly, since the
brief asks: *the one measured edge in this programme does not need this
estimator at all.* σ work must not block the FLB harvest.

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

Let `w = 60 s` (EXP-M6, fixed, never fitted), `r = T − t`, `X` the settlement
60 s TWAP, `X_0` the strike, `Ê_t[X_T]` the §2 nowcast-anchored forecast, and
`ω_P² = Var(P̂_t − P_t)` the nowcast error variance.

> **The estimator targets the conditional settlement innovation variance of the
> *realised forecast error*:**
> ```
> Σ(r) ≡ Var_t[ X_T − Ê_t[X_T] ]
>
>       = σ² · r(r+1)(2r+1)/(6w²)  +  (r/w)² · ω_P²        for r ≤ w
>       = σ² · (r − 2w/3)          +        ω_P²           for r > w
>
> σ_eff(r) = √Σ(r)          p̂ = G( (Ê_t[X_T] − X_0) / σ_eff(r) )
> ```

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
- **The discrete form is used, not the continuous one.** `r(r+1)(2r+1)/(6w²)`
  against `r³/(3w²)` is **+5.1 % variance at r=30 s, +15.5 % at r=10 s** — larger
  than most of the effects we are chasing, and free to include.
- **The variance ledger has exactly two lines, and they are independent.**
  `X_T − Ê_t[X_T] = (future increment) + (r/w)·(P_t − P̂_t)`; the first is a
  post-`t` innovation and the second a pre-`t` estimation error, so they add with
  no covariance term. The `(r/w)²` weight is not decoration — it is why the
  nowcast's noise costs nothing late in the window, which is where we quote.
- **`ω_P` replaces `σ_⊥` for the anchor; it is never added on top of it, and
  never added on top of `κ(r)`.** `κ(r) = 1 + σ_⊥²/σ_bin²` already contains the
  residual. This programme has committed the double-count three times
  (`σ_⊥+κ`, `v(t)` sum-vs-min, running-vs-terminal). **Treat any instinct to add
  a second variance term as suspect**; the ledger above is closed, and any
  addition to it requires a written justification of why it is not already
  inside `σ²` or `ω_P²`.
- **`ω_P` is measured, not assumed.** It is a new term I am introducing, and by
  this programme's own history it will otherwise be assumed zero or counted
  twice. Measure it against the independent 1 s Binance feed
  (`prices/crypto_prices`, 68,514 BTC ticks, Δt p50 999 ms), then subtract the
  known Chainlink–Binance basis so `ω_P` is nowcast error only.
- **Strike error is a third, separate line** for `t ∈ [0, Δ_K]`, `Δ_K ≈ 1.7–2.7 s`,
  and `Cov_t(X_0, X_T) ≠ 0` in the pre-open branch. Both are out of scope for
  the first estimator and are listed in §8.
- **No annualisation.** Report `σ` in bps/√s and `σ_eff` in bps at each `r`.
  For BTC at the measured 1.089 bps/√s:

  | r (s) | 30 | 60 | 120 | 180 | 240 | 270 |
  |---|---|---|---|---|---|---|
  | σ_eff (bps) | 1.77 | 4.87 | 9.74 | 12.88 | 15.40 | 16.52 |

  (r=30 includes the +2.5 % discrete correction.) At r=30 s, **1 bp of anything
  in the numerator is 0.57 in d-units, ≈ 20 probability-cents.** That is the
  real reason the anchor mattered more than σ.

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
   estimating a **small correction around 1**, not a level, so 20–30 % SE on the
   correction is 5 % on `Σ`.
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
  uniformity by horizon and by symbol. This is estimable now.
- **Tail (|z| ≥ 2):** **not estimable on this sample and we should stop
  pretending otherwise.** ~190 windows/symbol puts perhaps 4–9 observations
  beyond |z|=2 per symbol, and none beyond |z|=3. Any Student-t `ν` fitted here
  is fitted to single-digit counts.
- **Resolution: pre-commit `ν = 4` (a standard crypto value) and impose a hard
  `p̂` floor at half a tick, 0.005.** The floor binds at `|d| > 2.58`, which is
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

## 6. Predictive vs realised σ — pick one

**The estimator targets REALISED σ, from the tape. Outcomes are never used to
fit it.**

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

### What can be claimed now (2 days)
- **The variance law's shape.** Estimated from 63 k ticks/symbol/day and testable
  at ~15 horizons with thousands of increments each. The 1.06× flatness of BTC's
  implied σ over r ∈ [10,300] s is a real result.
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
- **G3 (ledger closed):** measured `ω_P`; `c(r)` within [0.8, 1.25]; no variance
  component appears twice. A `c(r)` outside that band means the parametric law is
  failing, not that the multiplier needs widening.
- **G4 (H-3, at 30 days):** if the direction test is null, **σ is declared risk
  plumbing permanently** and no further σ-as-alpha work is funded.

---

## 9. Ways this design could still be wrong

1. **The nowcast could be worse than it looks.** `P̂ = 2·S30 − S60` assumes local
   linearity over 60 s and that both feeds sample the *same* path synchronously.
   Chainlink has heartbeats, deviation thresholds and multi-source aggregation;
   if S30 and S60 update on different triggers, the extrapolation amplifies noise
   faster than it removes lag. **Mitigation:** `ω_P` is measured against the
   independent Binance feed, and the `r=30 s` case (`½S30 + ½P̂`) is exact and can
   be checked separately. If `ω_P` is large, fall back to the `r=30 s`-exact form
   and interpolate conservatively.
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
   fourth instance of this programme's signature failure.
10. **Demoting σ-for-level could be wrong.** It rests on a book-beat comparison
    run against a mis-anchored model on one test day. H-3 is the named re-entry
    point; the plan should not be read as closing that door, only as declining to
    fund it until H-3 reads.
11. **The architecture has nowhere to put the answer.** `BE-Uncertainty` is not
    in `contracts.yaml`'s `modules:` block and `BeliefProcess` has no variance
    field. An estimator emitting a *law* `(σ̂², ŵ, c(r), ω_P)` rather than a
    scalar has no typed carrier today. **Fix the contract before building.**

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

**A free validation asset:** `zec/usd` has 63,159 price ticks but **no
Polymarket market**. It cannot contribute outcome validation, but it is a clean
out-of-sample instrument for every tape-side claim (shape, level, `ŵ`, nugget,
regime persistence) with zero risk of contaminating the outcome analysis.

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

## 12. Build order — FINALIZED (feasibility first, per D1)

**Phase 0 — do these before building anything (all on data already on disk).**

1. **Fix the anchor** (§2) in a shared helper. Everything downstream is
   contaminated until this lands.
2. **Verify S30/S60 semantics** against the 1 s Binance tape (§9-2). If they
   fail, the nowcast and the locked-integral reconstruction both change.
3. **Measure `ω_P`** against the independent Binance feed (68 k BTC ticks
   collected). Close the ledger before anything is added to it. Report its
   weight at `r > w`, where it enters **undamped** — that is where we quote.
4. **Measure `c(r)`** on completed windows under the corrected anchor, with a
   provisional single-scale σ̂. **This is the go/no-go for the parametric
   route** (D2).
5. **Re-read the book-beat verdict once** on the corrected specification. It is
   paired, so it survives the MNAR gap, and it decides the programme's identity.

**Gate.** If `c(r)` is in-band and the S30/S60 semantics hold → proceed to
Phase 1. If `c(r)` breaches → stop and redesign the shape, do NOT widen the
band. If step 5 overturns "the book wins" → re-scope σ before continuing, since
§0's whole argument assumes the book supplies the level.

**Phase 1 — machinery, only if Phase 0 clears.**

6. **Joint `(σ², w, nugget)` WLS variogram**, non-overlapping, `r ∈ [10,600] s`,
   per symbol. Report `ŵ_free` as a diagnostic; resolve §9-3.
7. **Multi-scale blend**, weights fitted on the tape by QLIKE, per symbol —
   justified against the single-scale baseline from step 4 (D3). If it does not
   beat that baseline, it does not get built.
8. **`c(r)` refit** with computed shrinkage; **PIT / G2**; then the `p̂` floor
   and link policy.
9. **Contract fix** so the law has a typed carrier (§9-11).

Phase 0 is worth doing regardless of whether Phase 1 is ever adopted.
