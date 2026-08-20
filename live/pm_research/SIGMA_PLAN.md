# SIGMA_PLAN — design of the volatility estimator (P-2026-003)

**Revision 4, 2026-08-20. Canonical.** Rewritten in place at Revision 3; this
revision resolves the one thing that rewrite got wrong. Revision 3 recommended
an empirical reduced-form anchor AND kept the structural variance ledger, which
are two estimators of the same quantity — combining them double-counts
(`SIGMA_PLAN_REVIEW_ITER3.md` M3-1). **§2.3 now picks one route and the choice
propagates everywhere.** There is one consumer matrix, one PRICING law and one
DIAGNOSTIC decomposition, and they are never summed. v1/v2 text is in git
history (`80f823e`, `cc1d0e7`); nothing here defers to it.

**Status: estimator implementation is on HOLD.** Phase 0A is open. The sampling
convention is **UNVERIFIED** and `sigma_kernels.py` is a **fixture**, not a
frozen spec. That is now *enforced* rather than asserted: `pricing_var()`
refuses under an unverified convention or an unfitted law, and the structural
function returns a dict with no total a pricer could reach for.

Review lineage: `SIGMA_PLAN_REVIEW.md` (S1–S6) → Revision 2 (`cc1d0e7`) →
`SIGMA_PLAN_REVIEW_ITER2.md` (M2-1…M2-6) → Revision 3 (`7474e49`) →
`SIGMA_PLAN_REVIEW_ITER3.md` (M3-1…M3-6) → this revision.

---

## 0. Executive answer

1. **σ is not for the level of `p̂` on the main path.** Of the twelve places the
   design consumes σ, the two economically load-bearing ones (FLB calibration,
   the inventory cap `L_adv = |q|(1−p̂)`) consume **no variance at all**, and the
   four dynamics consumers need only the variance *shape* plus `d` — and `d` can
   be taken from the book.
2. **σ's surviving jobs**, in priority order: (i) the H-3 falsification test,
   the only route back to a `p̂` edge; (ii) a coarse regime/state level for
   sizing and stand-down; (iii) the **level for the BE-Belief stream fallback**
   whenever the book is `Unavailable`, which `BE_BELIEF_PLAN.md:650` states is
   not rare; (iv) the basis term `σ_⊥`, which sets the staleness threshold; and
   (v) the shape, which is *not estimated* — it is `w = 60` from EXP-M6 plus a
   kernel whose sampling convention is still unverified.
3. **A large share of the model's deficit to the book was never a σ problem.**
   It was a forecast-anchor error. That does **not** show σ was adequate, and
   the residual model-vs-book verdict is **unadjudicated** — the sample it would
   be read on is MNAR (§8). No section of this document may treat "the book
   wins" as settled.
4. **The anchor is the live problem, and it is a MEAN problem, not a variance
   problem.** §2.
5. **One route prices, the other diagnoses.** The pricing law is the route-A
   reduced-form conditional fit; the structural `k_law/v/Ω` decomposition is
   route-B diagnostics; `c(r)` is the *agreement between them*, not a term in
   either. §2.3, §3.2. Neither is built until Phase 0A closes.

---

## 1. What σ is FOR — the consumer matrix (canonical, single copy)

| consumer | σ object needed | LEVEL or DYNAMICS | edge-bearing? |
|---|---|---|---|
| `p̂ = g(d)` main path | terminal `Var_t[x_T]` | LEVEL | **demoted** — `p̂` is algebraic in an observed book price |
| **BE-Belief stream fallback** | terminal `Var_t[x_T]` | **LEVEL** | **required**; separate scoring population |
| `L_adv=\|q\|(1−p̂) ≤ κ_$` | **none** (linear in `p̂`) | LEVEL of `p̂` only | yes — risk cap |
| FLB harvest | **none** (book-bucket conditioned) | neither | not established (see below) |
| reservation `p̂ − γqv(t)` | none beyond `p̂` | LEVEL | no |
| participation frontier `m/g'(d) ≥ k√(3L/r)` | shape + `d` | BOTH | no |
| `r*` quotable horizon | shape + `d` + `L` | BOTH | no |
| `ζ(ℓ)` pickoff floor | short-horizon vol of `p̂` | DYNAMICS | no |
| `λ_bin = √3·g'(d)/√r` | shape only | DYNAMICS | no |
| rewards band | shape + `d` | DYNAMICS | no |
| stand-down `N` (staleness) | `σ_⊥` variogram | neither | no |
| hedge delta `g'(d)·∂d/∂F` | level | LEVEL | non-core |

**Every link appears as `g`, `g_inv`, `g'`** — the *selected* `LinkFunction`.
`contracts.yaml` says **never hardcode the Gaussian Φ**, and BE-Belief adopts a
logit recalibration. v13 exposed only `g`/`g_inv`, which left every dynamics
consumer unimplementable except by reaching for `φ` and smuggling the Gaussian
back in. v14 added `g_prime`; **v15 makes it the single canonical derivative**,
since v14 also carried `density`, which for a CDF-valued link is the same
function under a second name with no equality rule to stop two implementations
disagreeing. Reparameterising a probability through a probit is **not** evidence
that the book follows Gaussian dynamics in `x`.

**Take `d` from the book:** `d_book = g⁻¹(p̃_book)` with `p̃` the recalibrated
mid. Then the frontier, `r*`, `ζ` and `λ_bin` are **σ-free**, and the Q9 failure
mode — under-estimating σ by 2× at `|d|=1` leaves us quoting 52.9 s deeper into
the sniping zone — is *deleted rather than mitigated*.

**The fallback is the exception that sizes the build.** When the book is
`Unavailable` — mean top-of-book age is 12–20 s in the ATM and extreme buckets,
because a quiet book emits no events — there is no `d_book`, and σ must supply a
**level**. Its loss and validation population differ from the main path and are
scored separately. The fallback **refuses** when its own inputs are unavailable;
it may **not** inherit a book-sourced `d` after the book has failed.

**On the FLB:** earlier drafts called it "the one measured edge". Withdrawn. The
`+3.6 c/share` was measured on `book` snapshots that are p90 6.2 s stale;
rebuilt from executable `price_change.best_bid/ask` quotes the walk-forward gain
is **0.0004 Brier** and one-sided — a drift signature. `BE_BELIEF_PLAN.md`
treats it as a **correctness module** whose executable economics are presently
indistinguishable from zero. It is *a small forecast-calibration effect*, it
consumes no σ, and nothing here may lean on it.

---

## 2. The anchor — a conditional MEAN problem (the load-bearing section)

### 2.1 What went wrong twice

**v1's error.** `E_t[x_T]` was set to the trailing 60 s TWAP, which lags spot by
`w/2 ≈ 30 s`, while the variance factor was spot-anchored. Correcting it gained
**−0.0101 Brier pooled, at every horizon** and moved the MLE inflation factor
1.42 → 1.27.

**v2's error, and it is the same error.** The correction used
`P̂ = 2·S30 − S60`, justified by a *locally linear path* — a trend model — and
then entered that estimator's **unconditional MSE** as a zero-mean variance
line. Under the driftless model used everywhere else in the ledger, trajectories
have no local derivative and `2/−1` is not the conditional mean.

Write the translation-invariant family (the level is not identified, so the
weights must sum to 1):

```
P̂(α) = S60 + α·(S30 − S60)
```

Then the error decomposes exactly:

```
P̂(α) − P_t  =  (α − α*)·(S30 − S60)   +   conditionally zero-mean residual
                └── CONDITIONAL BIAS ──┘
                    known at decision time
```

Under `disc1s_v0`, **`α* = 2700/1801 = 1.4991671`, not 2**. So `α = 2` carries a
bias of `0.5008·(S30 − S60)` — a quantity **we observe at `t`** — and v2 put it
in the variance. At BTC's σ that buried bias has **sd 1.22 bps** against a total
`σ_eff(30)` of ~2.4 bps: half the standard deviation at `r=30` was a predictable
mean error. Same defect class as the v1 bug, one level deeper.

Two further consequences:

- **The conditional variance is α-independent.** `P̂(α)` is a function of the
  conditioning variables, so every anchor in the family has the same conditional
  variance and they differ **only** in bias. The variance line is `8.2590σ²`,
  not `9.5139σ²`; the difference was squared bias.
- **`9.5139` was called a "floor". It is not.** `α*` achieves `8.2590` inside
  the very same model. Nothing in `sigma_kernels.py` is a bound (§3.3).

### 2.2 α is a parameter to be ESTIMATED, and bias is measured against it

`α*` is what *this* path model implies, not what the market must do. Real
Chainlink streams have heartbeats, deviation thresholds and multi-source
aggregation; the true conditional mean is an empirical object. **`α` is fitted
on the tape**, and its distance from 2.000 (the trend assumption) and from `α*`
(the model projection) is a **process diagnostic**, exactly like `ŵ_free`.

**Bias is defined against the SELECTED estimand, never automatically against the
model.** Revision 3's fixture computed `bias = α − α*_model` unconditionally, so
a fitted `α̂` was always labelled biased and applying the documented mean
correction algebraically dragged the centre back to the Brownian projection — no
empirical coefficient could ever become the zero-bias mean. `AnchorSpec.selected`
now names which estimand defines correctness; a fitted conditional mean is
unbiased with respect to itself, and `model_gap = α̂ − α*` survives as a
diagnostic that is **never applied as a correction**.

### 2.3 Two routes, and you must pick ONE

Revision 3 recommended the direct regression **and** kept the structural ledger.
Those are two estimators of the same quantity, and combining them double-counts:
the regression residual already contains future innovation, latent-path
uncertainty, stream error and every covariance among them.

| | **Route A — reduced form** | **Route B — structural** |
|---|---|---|
| object | fitted conditional law of `x_T` on `(S30, S60)` | `σ²k_law(r) + σ²v(r) + uᵀΩu` |
| needs the sampling convention? | **no** — a regression on published streams does not care how they are built | **yes** — you must know the true covariance's shape |
| identifies `Ω`? | **no** — `Ω` is *inside* the residual | **yes** — as the lag-0 nugget (§9-2a) |
| delivers | a **pricing** law | the **decomposition** |
| status here | **the pricing law** | **diagnostics only** |

> **DECISION: Route A prices; Route B is diagnostic; they are never summed.**

The consumer matrix decides it. The only consumer that needs a *level* is the
BE-Belief fallback, and it needs `Σ(r)`, not its parts. The decomposition is
needed only by `c(r)`, the `k` ledger and H-3 — none of which is a gate. Encoded
as `PathLaw.estimand_route` and rule **R-ROUTE**; `sigma_kernels.pricing_var`
refuses anything that is not a fitted `ReducedFormLaw`, and the structural
function returns a dict tagged `DIAGNOSTIC_ONLY` with no `total_var` key.

**What OLS does and does not give you.** OLS returns the best **linear
projection** and a **pooled** residual variance. That is the conditional mean
only if the conditional mean is linear, and the conditional variance only under
homoskedasticity. Otherwise the pooled residual is an *unconditional forecast
MSE* — the same category error §2.1 removed from the variance line, one level
up. Under the Gaussian fixture they coincide; the entire point of going
empirical is not to lean on the fixture. So Route A ships with **gates, not
footnotes**: cross-fitting, ≥10 day clusters, a residual conditional-mean test
and a heteroskedasticity test, per horizon and per symbol, day-blocked.
`pricing_var` refuses if any fails. Two day clusters yield a descriptive
coefficient, not a pricing-ready conditional law.

### 2.3 Bias never enters the variance

Whatever `α` is chosen, the residual bias `(α − α̂*)·(S30 − S60)` goes in the
**numerator of `d`** via `PathLaw.conditional_mean`, or the law refuses.
`AnchorSpec.bias_enters` admits only `MEAN_MODEL` or `REFUSED`. This is the
single rule that would have caught both v1 and v2.

---

## 3. The estimand — stated once

### 3.1 Unit space (frozen)

```
x_t = 1e4 · (S_t − X_0)/X_0     model coordinate, bps, dimensionless
σ                               bps / √second
Σ(r), nugget, feed error        bps²
d = (E_t[x_T] − x_0)/√Σ(r)      dimensionless
```

`X_0` is the strike, known at `t0`. **Normalised arithmetic returns, not log
returns.** The settlement mark is an *arithmetic* mean, so `log E[X] ≠ E[log X]`
and log coordinates would demote the TWAP kernel, the anchor decomposition and
the ledger from identities to approximations. This is *not* a magnitude
argument — the measured Jensen gap is `+0.00059 bps`, 0.024 % of `σ_eff(30)` —
it is that exactness is free here and this programme's recorded failure mode is
specification error hiding inside a plausible approximation. Typed as
`UnitSpace`; cross-symbol pooling before normalisation is a type error.

### 3.2 The law

**THE PRICING LAW (route A).** One line, and nothing is added to it:

> ```
> Ê_t[x_T] = S60 + α̂(r)·(S30 − S60)          fitted conditional mean
> Σ(r)     = residual variance of that fit    the WHOLE of it
>
> σ_eff(r) = √Σ(r)     p̂ = g( (Ê_t[x_T] − x_0)/σ_eff(r) )
> ```

`g` is the selected `LinkFunction`. Future innovation, latent-path uncertainty,
stream measurement error and every covariance among them are **already inside**
`Σ(r)`. Adding `k_law`, `v(r)` or `Ω` to it double-counts (R-ROUTE). Gated on
cross-fitting, ≥10 day clusters and the two residual diagnostics of §2.3.

**THE DIAGNOSTIC DECOMPOSITION (route B).** Never a pricing input:

> ```
> model_total(r) = σ²·k_law(r) + σ²·v(r) + uᵀΩu      DIAGNOSTIC_ONLY
>
> k_law(r) = r(r+1)(2r+1)/(6w²)              r ≤ w    post-t innovation
>          = (r − w) + (w+1)(2w+1)/(6w)      r ≥ w
> v(r)     = conditional variance of the anchor error   (α-INDEPENDENT)
> uᵀΩu     = stream measurement error, u = (α, 1−α), Ω the 2×2 feed covariance
> ```

`k_law` is **continuous at `r = w`** — both branches give
`(w+1)(2w+1)/(6w) = 20.5028` s — so the `r > w` offset is `r − 39.4972`, not the
continuous law's `r − 40`. v1 mixed the discrete branch below `w` with the
continuous one above and jumped 1.25 % in σ at the seam.

**Units, stated once (M3-3).** `σ²` is a **rate**, bps²/**second**, typed
`RateQuantity` — not the same type as a terminal bps² variance. `k_law` and
`v(r)` are dimensionless. **`Ω` is in bps², physically, everywhere**, and is
*not* multiplied by the rate. v3's code documented "σ² units" and multiplied by
`σ²` in the ledger, so an identity `Ω` at rate 4 contributed 9.9867 bps² instead
of 2.4967.

**Ω is a 2×2, not a scalar**, its contribution varies with the horizon weights,
and because `(1−α)` changes sign above `α = 1` the same covariance *raises* one
anchor's variance and *lowers* another's. It must be symmetric, finite and
**PSD** before use — v3 accepted a non-PSD matrix and returned a total variance
of **−120.9**.

**There is no hidden line.** v2 shipped a `settlement_var` that silently added a
`nugget` its own `ledger()` omitted. A nugget may be observation noise, feed
noise or small-scale process variance; those map differently into conditional
settlement uncertainty, so it appears only as a named component with a declared
estimand — and under route B it is not a nuisance at all but the **estimator of
`Ω`** (§9-2a).

### 3.3 The route-B numbers, and what they are not

**These are DIAGNOSTICS. There is no `σ_eff` for pricing yet, because no route-A
law has been fitted.** Under `disc1s_v0` (**UNVERIFIED**), the model anchor and
no feed error, BTC rate `σ² = 1.089²` bps²/s:

| r (s) | 30 | 60 | 120 | 180 | 240 | 270 |
|---|---|---|---|---|---|---|
| `α*(r)` | 1.2496 | 1.4992 | 1.4992 | 1.4992 | 1.4992 | 1.4992 |
| `k_law(r)` | 2.6264 | 20.5028 | 80.5028 | 140.5028 | 200.5028 | 230.5028 |
| `v(r)` | 2.0648 | 8.2590 | 8.2590 | 8.2590 | 8.2590 | 8.2590 |
| `model_total` (bps²) | 5.563 | 34.109 | 105.264 | 176.420 | 247.575 | 283.153 |

`α*` is horizon-dependent inside the window and constant outside it: at `r=30`
the trailing half of the mark *is* `S30`, observed, so `α*(30) = ½ + ½α*` and
`v(30) = ¼·8.2590` exactly. Revision 3 published a bolded `σ_eff` row here; that
invited exactly the use this row must not have, so it is gone.

**Why these cannot price.** The convention is unverified — a 1 s shift in
fast-stream support alone moves `α*` from 1.4992 to 1.4954 — and the model
anchor is not the estimand. Enforced, not merely stated: `pricing_var()` refuses
while `status != VERIFIED`, and the structural function returns a dict with no
`total_var` key for a pricer to reach for. For scale: at `r=30`, 1 bp in the
numerator is ≈ **17 probability-cents**, which is why the anchor mattered more
than σ and why a 1.22 bps buried bias was not survivable.

Regenerate: `python3 live/pm_research/sigma_kernels.py --selftest` (40 checks,
exact rationals, including the refusal paths).

### 3.4 The two objects are never interchangeable

The rolling TWAP-to-TWAP variogram `V(r) = σ²(r²/w − r³/(3w²))` for `r<w` is
what you **fit σ to**; the conditional `Σ(r)` is the only thing you **price
with**. Their ratio in σ is 4.12× at `r=10 s` and 2.24× at `r=30 s`. `PathLaw`
exposes both (`increment_var`, `settlement_var`) so no consumer has to choose.

### 3.5 Other lines, declared and excluded

- **`σ_⊥` / basis** is a *basis* object, not a settlement variance component. It
  sets the staleness threshold `N` and the `r ≈ 16 s` crossover below which our
  price model is noise. Registered under a distinct owner in the `VarianceGroup`
  registry (R-ONCE) so no future consumer can add it to `Σ(r)`.
- **Strike error** for `t ∈ [0, Δ_K]`, `Δ_K ≈ 1.7–2.7 s`, and
  `Cov_t(x_0, x_T) ≠ 0` in the pre-open branch: out of scope, §11. Note the
  normalised coordinate makes strike error a *coordinate* error too, not only a
  ledger line.
- This programme has double-counted variance three times (`σ_⊥+κ`, `v(t)`
  sum-vs-min, running-vs-terminal). **Treat any instinct to add a variance term
  as suspect**; additions require a written argument that they are not already
  inside `σ²`, `v(r)` or `Ω`.

---

## 4. Option comparison, and the recommendation

**A — empirical settlement-innovation variance.** Model-free, targets `Σ(r)`
exactly, immune to the kernel question, and the only method that can detect that
the parametric law is wrong. But starved: ~190–239 windows/symbol total, and the
honest unit is the **day**, of which there are two.

**B — parametric variogram fitted to increments.** Where the information is. The
fits recover the shape to 3.8–10.6 % rmse across eight symbols and BTC's implied
σ is flat to 1.06× over `r ∈ [10, 300] s` — a description the data endorse, not
an assumption tolerated. Yields `ŵ_free` as an independent check and re-fits on
any trailing window. But it assumes the shape, and `ŵ_free` ranges 47–81 s.

**D — regime conditioning** is a requirement on any of these, not a rival. p90/p10
of trailing σ is 3.4–4.3× *within* one coin, so a static σ is indefensible; but
ρ(trailing σ̂, forward squared innovation) is only **+0.19 to +0.40** and the
60 min window is at least as good as the 15 min in 6 of 8 symbols. That caps
what regime conditioning can deliver and refutes v3's `w_fast = 0.9`.

> **Recommendation, restated under §2.3's route decision: the PRICING law is the
> route-A reduced-form fit. B supplies the level, its time-variation and the
> decomposition as DIAGNOSTICS, and `c(r)` is the comparison BETWEEN them.**
>
> ```
> price with:   Sigma_hat(r) = residual variance of the route-A fit      (one line)
> diagnose with: model_total(r) = sigma2_blend*k_law(r) + sigma2*v(r) + u'Omega u
> c(r) = model-vs-empirical agreement, NOT a term added to either
> ```
>
> Revision 3 wrote `Σ̂(r) = σ̂²_blend·k_law + σ̂²v̂ + uᵀΩ̂u` here while §2.2
> recommended the regression. That was the M3-1 double count in its clearest
> form: two estimators of one quantity, presented as one formula.

Composition: level fully per-symbol, no pooling (the level spans 4.6× across
coins); time-variation by a multi-scale blend whose weights are fitted **on the
tape** by QLIKE, not on 190 Bernoullis; `c(r)` and blend weights hierarchically
shrunk with the shrinkage weight *computed* from measured between-symbol
dispersion versus within-symbol SE and **reported per symbol**. Pooling as a
*prior* is legitimate; pooling as a *parameter tie* is v3's error. Emit
`(σ̂, SE(log σ̂))`, since Q9 shows the loss is asymmetric.

**The multi-scale blend is a challenger, not a default.** It must beat the frozen
single-scale per-symbol baseline out of sample on identical folds, or it is not
built.

---

## 5. Link function

**Do not fit a link. Measure it, then floor it.**

- **Body (`|z| < 2`):** the empirical/PIT-calibrated CDF, checked for uniformity
  by horizon and symbol — **fitted walk-forward**. A pooled empirical CDF checked
  on the same two days it was estimated from is descriptive fit, not calibration.
- **Tail (`|z| ≥ 2`):** not estimable here and we should stop pretending. ~190
  windows/symbol gives perhaps 4–9 observations beyond `|z|=2` and none beyond 3.
- **Resolution:** pre-commit `ν = 4` as a declared conservative **policy
  assumption** — not "a standard crypto value". Owner: BE-Uncertainty;
  sensitivity `ν ∈ [3,6]` reported. Impose a hard `p̂` floor at half a tick,
  0.005. It binds at `|d| > 2.58`, exactly where the Gaussian tail becomes
  untrustworthy, so beyond that the floor governs and the link choice is
  irrelevant to every risk consumer. It also kills the `Q_max = 9.9e15`
  pathology (`p̂ ≈ 2.5e-13`). **The floor is a risk policy and may never be read
  back as evidence about the link.**
- Do **not** fit `ν` by outcome-MLE: it trades off against σ and at 2 days the
  trade-off is unidentified.

---

## 6. What the estimator forecasts

**The live output is an ex-ante FORECAST OF FUTURE PHYSICAL VARIANCE, fitted on
the tape.** Not "realised σ" — at decision time the next 300 s of variance is
unknown, so a trailing estimator whose weights minimise QLIKE against the *next*
300 s realised variation is a forecast, and the future tape quantity is its
ex-post *target*. **"Not fitted to binary outcomes" and "not predictive" are not
synonyms.**

Every artefact carries `as_of`, `fit_data_through` and a `target_interval`, and
every query carries a `ForecastRequest` with its knowledge cutoff. **R-WFWD:**
`fit_data_through ≤ as_of`, and no training data is read beyond the cutoff.
Being *used* after the cutoff is the point, not a violation — v2's contract note
said the law refuses after `fit_data_through`, which is backwards.

Outcomes are never used to fit it, for four reasons: information geometry
(`SD ∝ 1/√days`, and 2 days cannot pin one pooled scalar); pseudo-replication
(one `y` per window — v3 stacked 6 decision rows as independent, inflating
apparent information by ~√6); circularity (an outcome-MLE recovers the book's
implied σ and scores agreement as skill); and because H-3 asks whether σ̂ beats
`σ_book` at predicting *realised* dispersion, which a σ fitted to outcomes has
already absorbed.

`k = σ̂_pred/σ̂_realised` is a **specification-error detector**, reported per
horizon and symbol, **never applied as a multiplier**.

---

## 7. The `k ≈ 1.12` puzzle

Four mechanical forces act before any economic story:

1. **Anchor error — demonstrated, not hypothesised.** Correcting the v1 numerator
   moved `k` 1.420 → 1.268 on identical rows.
2. **Omitted anchor variance.** The v1 spec had no anchor line at all; under the
   old S60 anchor its coefficient is `19.5028σ²`, which at `r=30` is ~65 % of
   total variance. An outcome-MLE could only cover that by inflating σ.
3. **Conditional bias** (§2). A biased anchor inflates realised dispersion
   without inflating the true conditional variance — it looks like a σ premium
   and is not one.
4. **Gap selection.** Skipping gap-containing intervals selects against high-vol
   regimes and biases realised σ̂ **down**, inflating `k` from the denominator.

And one force pushes `k < 1`: link misspecification biases outcome-fitted σ̂ down
3.5–15 %.

**Decompose, never multiply.** My prior is that little of `k` survives the
ledger. An "implied-over-realised premium" is the *last* explanation to reach
for, and at 2 days a 12 % premium is indistinguishable from a 12 % estimator
bias. If a residual survives at 30 days it belongs in H-3 as evidence of a σ
edge, not in `Σ̂` as a fudge factor.

---

## 8. Validation protocol, honest at n ≈ 2 days

**Unit of inference.** Tape-based claims: the window, with a block bootstrap over
contiguous blocks. Outcome-based claims: the **day**, of which there are two.
Never the decision row.

**Effective sample size, not tick count.** Earlier drafts argued power from
"63 k ticks/symbol/day". That is not the sample size for anything we fit:
non-overlapping 600 s increments give ~**144 units per full day**, all horizons
ride the same path, and 1 s labels for next-300 s variation overlap 299/300 of
their support. Protocol: train through day `d−1`, **day-block test folds**, an
embargo covering the longest label support, and either non-overlapping targets or
overlap-aware weights with block inference.

### Claimable now (2 days)
- The variance law's **shape** (~144 non-overlapping units/symbol/day).
- **Relative σ levels across symbols** (0.94 → 4.33 bps/√s is far outside noise).
- **Paired, same-row specification comparisons** — the workhorse, and the only
  outcome-based inference with usable power here.
- **PIT uniformity in the body**, pooled, as a coarse check.

### Not claimable now
- **Any Brier difference versus the book.** One test day; the day-clustered SE is
  undefined.
- **Anything from the pre-repair book sample, and pairing does not rescue it.**
  Pairing makes the two forecasts share the same *observed* rows; it does not
  recover the busy BTC intervals the slow-consumer failure dropped (27 of 47
  disconnects self-inflicted, 32 of 47 BTC). If relative model/book performance
  varies with volatility, staleness or activity — the whole hypothesis —
  conditioning on observed rows biases the paired delta. The re-read needs the
  dense knowledge-time book rebuilt from `price_change.best_bid/ask`,
  cause-stamped gaps, post-repair or gap-complete units, selection deltas by
  activity/volatility, and a recalibrated-book baseline alongside the raw book.
  The **anchor** comparison stays valid because it is paired on the settlement
  tape, which had no such gap.
- Tail calibration beyond `|z| ≈ 2.5`; the level of `k`; any per-symbol
  difference in blend weights or `c(r)`; that the anchor fix persists.

### Gates
- **G1 (shape):** implied σ flat within ±15 % over `r ∈ [10,300] s`, all symbols.
  *Passes for BTC (1.06×).*
- **G2 (calibration):** PIT uniform in `|z| < 2` by horizon, per symbol after
  shrinkage, at 7 days, walk-forward.
- **G3 (route coherent):** `PathLaw.estimand_route` declared; sampling
  convention **VERIFIED** (route B only); `α` estimated with bias measured
  against the selected estimand; `Ω` in bps², PSD-validated, with its
  identification stated per route; and **no structural line added to a
  reduced-form residual** (R-ROUTE). `c(r)` is now defined as the *agreement*
  between the two routes,
  `c(r) = Σ̂_A(r) / model_total_B(r)`, expected ≈ 1 — **not** a multiplier
  inside either. It is a **model-adequacy diagnostic**: a `c(r)` far from 1
  means the parametric law misdescribes the tape, which is information about
  route B, not a correction to route A. Its `status` stays **`DIAGNOSTIC`**
  until ≥10 independent day clusters exist; a point estimate at 2 day clusters
  is not a go/no-go. Any quoted SE must name its object — 20–30 % relative
  uncertainty on a variance ratio is 20–30 % on the ratio and ~10–15 % in σ
  units, and does **not** shrink because the mean sits near 1.
- **G4 (H-3, at 30 days):** if the direction test is null, **σ is declared risk
  plumbing permanently**. Pre-registered domain rules: `σ_book` from the selected
  link is undefined near `mid = 0.5`, can go negative when book and stream
  disagree in sign, and explodes under tick quantisation — fix an admissible
  moneyness band first, handle sign conflict and censoring explicitly rather than
  conditioning away the hard cases, score the **forecast** against future
  physical variation OOS, and control against the **recalibrated executable
  book**, not the raw mid.

---

## 9. Ways this design could still be wrong

1. **`α` may be neither 2 nor `α*`.** The whole anchor now rests on estimating a
   conditional mean from two smoothed streams. If the streams are not jointly
   informative about spot in the way the projection assumes, the residual is
   neither zero-mean nor stationary.
2. **S30/S60 semantics are unverified.** The aggregation is undocumented, and a
   ~1 Hz publication cadence is a *cadence, not a kernel*: it does not establish
   60 equally-weighted samples, synchronous support, or right-alignment at `t`.
   EXP-M6 proves the published S60 endpoint reproduces settlement; it says
   nothing about how that endpoint is built. **A 1 s shift in fast-stream support
   alone moves `α*` by 0.004.** Check by reconstructing both from the 1 s Binance
   tape. This is Phase 0A step 5. **It gates route B entirely — the kernel, `v(r)`
   and `Ω` — and does not gate route A**, whose regression runs on the published
   streams whatever they turn out to be. That asymmetry is the practical reason
   §2.3 prices with A.
2a. **`Ω` is not identified from contemporaneous moments, and its route-B
   estimator is entangled with `ŵ`.** Counting: `Var(S30)`, `Var(S60)`,
   `Cov(S30,S60)` are **3 moments for 4 unknowns** (the rate plus `Ω`'s three).
   Under route B the extra information comes from the *time series*: if stream
   error is serially uncorrelated it appears **only at lag 0**, so extrapolating
   the true covariance to zero lag and taking the gap identifies `Ω`. That gap is
   the **nugget** — the same object already in the per-symbol variogram table
   (0.00 for btc/eth/xrp/doge/bnb, 0.14–0.28 bp for sol/zec/hype). Two caveats
   that make this a risk rather than a solved problem: it requires the true
   covariance shape, hence a **VERIFIED** convention; and a variogram fitted
   without a nugget attributes microstructure to a shorter kernel, so `Ω`, the
   nugget and `ŵ = 47 s` are **one identification problem, not three** — which is
   why item 3 must be resolved jointly. If stream error is serially correlated
   (plausible under deviation-threshold updates) it leaks into short lags and the
   separation fails. Under route A none of this arises: `Ω` is inside the
   residual and must never be added again.
3. **`ŵ ≈ 47 s` for BTC.** Most likely nugget confounding, but if real the
   in-window level is biased up to 1.63× in variance. **Still the largest open
   technical risk after the anchor.** Resolve with a joint `(σ², w, nugget)` WLS
   fit on non-overlapping increments before trusting any in-window number.
4. **Two days is one regime** — a rally with +4.5 pp up-drift. Every
   regime-conditioning weight is fitted inside it, and the anchor improvement
   could be drift-flattered. Note a *trend* extrapolator (`α = 2`) is exactly what
   a drifting sample would flatter, which is a reason to distrust v2's empirical
   support for it.
5. **Vol persistence may be too weak for a 3-scale blend to earn its parameters**
   (ρ = 0.19–0.40; 60 min already matches 15 min). Test against a single 60 min
   window and report the delta honestly.
6. **`hype` may not be the same process** (σ 4.33, `ŵ` 81 s, largest nugget); the
   shrinkage prior could harm it.
7. **Gap handling can only bias σ̂ down** — the dangerous direction.
8. **Book-sourced `d` inherits the book's miscalibration**; recalibration is
   mandatory and is itself fitted on 2 days.
9. **Double-count risk is closed, not eliminated.** `v(r)` and `Ω` are new lines.
10. **Demoting σ-for-level could be wrong.** It rests on a book-beat comparison
    run against a mis-anchored model on one test day *and* on an MNAR sample.
    H-3 is the named re-entry point.
11. **A green checker is not a sound boundary.** `contract_check.py` validates
    structural references and migrations; it did not catch that v13 reached past
    the StateView seam, duplicated link ownership or stated the fit cutoff
    backwards. This programme has now shipped **four** artefacts that reported
    success without checking the thing they existed to check — the v8 contract
    checker, the path-keyed allowlist, v2's tautological independence test (which
    drew the future innovation independently by construction, so it could not
    fail), and v13 itself. Assume the next one exists.
12. **`BE-Belief` is absent from the contracts `modules:` block.** Out of scope
    here; it belongs to the structure review loop.

---

## 10. BTC first — what generalises

**Generalises:** the variance law's functional form (3.8–10.6 % rmse across all
eight symbols); the estimand and the ledger; the anchor *algebra* (though not
its `α`); `w = 60` as a venue convention; the estimator architecture, link/floor
policy and validation protocol.

**Per-symbol, never pooled:** the σ level (0.94 → 4.33 bps/√s, 4.6×); `ŵ_free`
(47 → 81 s); the nugget (0 for btc/eth/xrp/doge/bnb, 0.14–0.28 bp for
sol/zec/hype); regime persistence (ρ 0.19 btc → 0.40 hype — BTC is the *hardest*
symbol to regime-condition, so BTC-first is conservative); the fast/slow balance.
**`α` must be treated as symbol-SPECIFIC — never pooled across symbols** — since
it depends on each stream's update behaviour. (Revision 3 wrote "assumed
per-symbol until measured otherwise", which read as licence to assume a value;
no `α` may be assumed for pricing at all — see §2.2.)

**`zec/usd` is a useful control, not an independent OOS asset.** 63,159 price
ticks and no Polymarket market, so it cannot contaminate the outcome analysis —
but it shares the same dates, regime and estimator-selection process. A ZEC pass
is evidence against a coding error or symbol-specific artefact, not out-of-sample
confirmation.

---

## 11. Deferred until more data

| item | why | unblocked at |
|---|---|---|
| Fitting `ν` / any tail parameter | <10 obs beyond \|z\|=2 per symbol | 30+ days |
| Per-symbol `c(r)` without shrinkage | ~190 windows/symbol | 30+ days |
| Any claim about `k` as a premium | 4 mechanical sources unquantified | 30 days + closed ledger |
| H-3 σ-edge test | needs day-clustered CI over ≥10 test days | 30 days |
| Pre-open branch, `Cov(x_0, x_T) ≠ 0` | not on the current decision grid | when quoting opens pre-window |
| Strike-reconstruction variance | separate estimand, and a coordinate error | with the pre-open branch |
| `σ_⊥` / basis variogram → staleness `N` | needs a paired Chainlink–Binance study | PM-E1 |
| Intra-window σ updating | ρ≈0.2–0.4 says the gain is small | after the blend earns its keep |

---

## 12. Build order

Estimator implementation is on **HOLD**. Phase 0A is open.

**Phase 0A — definitions and deterministic checks.**

0. **Route** — DONE (§2.3). `estimand_route: REDUCED_FORM` prices; STRUCTURAL
   diagnoses; R-ROUTE forbids summing them. Everything below is scoped by it.
1. **Unit space** — DONE (§3.1), typed as `UnitSpace`; rates typed separately
   from terminal variances as `RateQuantity`.
2. **Typed carrier** — contracts **v15**: `AnchorSpec` is horizon-indexed and
   defines bias against the SELECTED estimand; `ReducedFormLaw` carries the
   pricing route with its gates; `FeedErrorCov` fixes bps² once and states its
   identification; `g_prime` is the single canonical derivative; R-REQ and
   R-ROUTE join R-WFWD. **Not closed:** the runtime enforces refusal only in the
   fixture, and `BE-Belief` is still absent from `modules:`.
3. **Kernels** — REOPENED, and correctly so. The algebra is exact and
   unit-tested (40 checks including every refusal path) but conditional on an
   **UNVERIFIED** convention. Route B only.
4. **Consumer matrix** — §1 is canonical; the fallback's own loss function is
   still unwritten.
5. **Verify S30/S60 semantics** against the 1 s Binance tape: window endpoints,
   sample weights, update triggers, event-time alignment, knowledge-time
   construction. **This gates route B entirely and does not gate route A**
   (§9-2). Revision 3 said it "gates the kernel and the anchor together", which
   contradicted its own claim that the regression is convention-robust; the
   scoped statement is that it gates the *decomposition*, not the *fit*.
6. **Fit the route-A anchor and law** — regress observed `x_T` on observed
   `(S30, S60)` per horizon and symbol, cross-fitted, day-blocked, embargoed.
   Emit `AnchorSpec{selected: ESTIMATED}` with bias measured against the
   estimate, `model_gap` as a diagnostic, and `ReducedFormLaw` carrying
   `n_day_clusters`, `cross_fitted` and both residual tests. **Do not estimate
   `Ω` on this route** — it is inside the residual (§9-2a).

**Phase 0B — data admissibility and feasibility.**

7. Rebuild the dense knowledge-time top of book from `price_change`; classify
   gaps by cause; isolate post-MNAR-repair data.
8. Fit the frozen per-symbol single-scale baseline on day-block embargoed folds.
9. Build the route-B decomposition (only if step 5 passed) and compute
   `c(r) = Σ̂_A/model_total_B` with block/day uncertainty, `status: DIAGNOSTIC`.
   It measures agreement between the routes; it is added to neither.
10. Re-read fallback calibration and model-vs-book scoring on admissible rows
    only, against both raw and recalibrated book. `DESCRIPTIVE` until the
    day-cluster threshold.

**Phase 1 — machinery, only if Phase 0 is coherent.**

11. Joint `(σ², w, nugget)` WLS variogram, non-overlapping, per symbol; resolve
    §9-3.
12. Multi-scale QLIKE challenger on identical folds against the step-8 baseline.
13. Shrinkage only where between/within-symbol evidence supports it;
    walk-forward PIT/G2; then the `p̂` floor and link policy.
14. H-3 at the pre-registered horizon (G4).

**What can still stop this:** step 6's residual diagnostics failing would mean
the linear projection is not the conditional mean here, and route A would need a
richer conditioning set before it can price at all; step 5 failing removes route
B, hence `c(r)`, `ŵ`, `Ω` and the H-3 decomposition, but leaves route A intact;
step 10 overturning "the book wins" re-scopes σ, since §0's argument assumes the
book supplies the level. `c(r)` is **not** a gate until the day clusters exist.

Effort saved here goes to **G-FF4, the queue bracket**, which can still end the
programme.

---

## Appendix A — measurements this plan rests on

Feed (`prices/crypto_prices_twap_sixty`, 20 files, 20260819_15 → 20260820_10):
8 symbols, 63,140–63,161 ticks/symbol over 18.9 h; inter-tick Δt p50 955 ms, p90
1,571 ms, max ≈ 44 s; 87.2 % of ticks change value; `recv_ns − payload.timestamp`
p50 1,692–1,745 ms. Windows: 1,634 resolved, 7 traded coins, 226–239 per coin,
**2 day-buckets**, pooled up-rate 0.5459.

BTC variogram of the S60 tape, `V(r) = Var[x_{t+r} − x_t]`; last row is σ implied
under a 60 s-TWAP-of-BM, so **flat ⇒ the law fits**:

| r (s) | 1 | 5 | 10 | 30 | 60 | 120 | 180 | 300 |
|---|---|---|---|---|---|---|---|---|
| `V(r)/r ×1e8` | 0.032 | 0.119 | 0.220 | 0.562 | 0.872 | 1.019 | 1.085 | 1.169 |
| implied σ (bps/√s) | 1.396 | 1.214 | 1.182 | 1.161 | 1.144 | 1.106 | 1.105 | 1.119 |

`V(r)/r` moves 36× across the range; implied σ moves 1.26×, and only 1.06× once
`r ≥ 10 s`.

Per-symbol variogram fit and regime persistence (ρ = Spearman between trailing σ̂
and the next 300 s squared innovation):

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

The anchor-fix Brier table (walk-forward, fit day 20684, test 20685, 1,575
windows) is in git history at `cc1d0e7` §2. It is retained as evidence that the
*direction* of the v1 anchor fix was right; it is **not** evidence for `α = 2`,
which is the separate question §2.2 settles by estimation.

---

## Appendix B — what the reviews changed

| review | item | outcome |
|---|---|---|
| S1 | one unit space | normalised arithmetic returns (log declined: the mark is an arithmetic mean) |
| S2 | consumer contract first | v13→v14 carrier; fallback added as a LEVEL consumer |
| S3 | anchor order / `ω_P` identification | order reversed; identification superseded by §2.2's direct regression |
| S4 | MNAR pairing | accepted without qualification; the claim was struck |
| S5 | physical forecast, overlap | renamed; effective sample size replaces tick counts |
| S6 | `c(r)` / H-3 gates | `c(r)` diagnostic; H-3 given domain rules |
| M2-1 | `a(r)` is unconditional MSE | **anchor rebuilt as a conditional-mean problem** (§2) |
| M2-2 | 1 s convention frozen too early | `SamplingConvention`, status UNVERIFIED, step 3 reopened |
| M2-3 | floor/ceiling not ordered | replaced by non-ordered evidence + a 2×2 `Ω` |
| M2-4 | hidden nugget, no refusal | one ledger, exact domain validation, refusal tested |
| M2-5 | v13 boundary defects | contracts v14, eight sub-items |
| M2-6 | overlay, not canonical | rewritten in place at Revision 3 |
| M3-1 | reduced-form and structural combined | **§2.3 picks one route**; R-ROUTE; `c(r)` redefined as agreement |
| M3-2 | fixture could not express an empirical α | `AnchorSpec.selected`, horizon-indexed, bias vs the estimate, `conditional_mean` implemented |
| M3-3 | `Ω` units/PSD/identification | bps² once, PSD-validated, identification stated per route (§9-2a) |
| M3-4 | not fail-closed | `pricing_var` refuses on status/fit/gates; rates, PSD, conventions validated; contract `Unavailable` |
| M3-5 | request fields never compared | `check_request` + R-REQ, with negative fixtures |
| M3-6 | contradictions survived the scan | scan now covers plan, code, STATUS and HANDOFF |
