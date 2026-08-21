# PM_QUANT_REVIEW — quantitative-correctness lens (σ / variance law + quoting engine)

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


Object: `PM_MM_PLAN.md` §2 (settlement model + variance laws), §3 (quoting
engine as amended), §11 (three model changes), §12 (this session's amendments:
σ estimation design, the three model defects, EU siting). Prior theory review
`PM_SKETCH_REVIEW_ITER1_T.md` (F1–F15) is taken as a base and re-checked only
where §12 builds on it.

Method: every law re-derived by hand, then Monte-Carlo'd (driftless BM,
σ = 1/√s, dt = 0.05 s, N = 120–200 k paths, chunked); estimator designs
re-simulated on synthetic 1 s streams (13.5 h – 7 days × 8 coins, stochastic
vol); Fisher information computed by quadrature in log-space. Convention
throughout: `X_t = (1/w)∫_{t−w}^t S_u du`, `w = 60 s`, `T = 300 s`,
`τ = T − t`, `r = T − t` in-window.

Headline: **§2's variance laws are correct. §12.3's estimator does not estimate
them.** The quantity §12.3 names as "the target" is a different quantity, larger
than σ_eff² by 17× at r = 10 s, and E-M6c as specified will "refute" the r³ law
at ~100 σ for reasons that have nothing to do with the settlement mechanism.

---

## 1. The variance laws

### 1(a) Pre-window `Var_t[X_T] = σ²(τ − w + w/3)` — **CORRECT**; τ is to `T`

Derivation (as F1): with `a = τ−w`, `b = τ` measured backwards from the
settlement window,

```
Var_t[X_T] = (σ²/w²)∫∫_{[a,b]²} min(u,v) du dv = σ²(b+2a)/3 = σ²(τ − 2w/3)
```

and `τ − w + w/3 ≡ τ − 2w/3`. **τ is measured to `T`, the window END, not to
`T−w`.** Two checks settle it. (i) Continuity at window entry: at `τ = w` the
formula must collapse to the pure averaging variance `σ²w/3`, and it does
(`w − 2w/3 = w/3`); if τ were measured to `T−w`, entry would be `τ = 0` and the
formula would return a negative variance. (ii) The MC below, against the naive
"drift-to-window **plus** averaging" reading `σ²τ + σ²w/3` (the reading the
plan's own prose invites), which is off by 4× at entry:

```
  tau= 300.0  MC= 259.70 +-1.61 | tau-2w/3= 260.000 | naive tau+w/3: 320.000
  tau= 200.0  MC= 160.06 +-0.99 | tau-2w/3= 160.000 | naive tau+w/3: 220.000
  tau= 120.0  MC=  80.33 +-0.50 | tau-2w/3=  80.000 | naive tau+w/3: 140.000
  tau=  60.0  MC=  19.99 +-0.12 | tau-2w/3=  20.000 | naive tau+w/3:  80.000
```

Endpoint `σ²(τ − 40 s)` at w = 60 s: correct. **VERDICT: CORRECT.**

### 1(b) In-window `Var_t[X_T] = σ²r³/(3w²)` — **CORRECT**, cross-term exactly 0

`X_T = locked + (1/w)∫_t^T S_u du`. Conditional on `F_t` the locked integral is
a **constant**, so there is no cross-term — this is measurability, not an
approximation, and it holds for any filtration in which the elapsed part of the
averaging window is observed. Open part
`= (1/w)[rS_t + σ∫_0^r B_s ds]`, `Var[∫_0^r B_s ds] = r³/3`.

```
  r= 60.0  MC= 19.9889  law= 20.0000  ratio=0.9994
  r= 30.0  MC=  2.5044  law=  2.5000  ratio=1.0018
  r= 10.0  MC=  0.09269 law=  0.09259 ratio=1.0010
  r=  5.0  MC=  0.01167 law=  0.01157 ratio=1.0082
```

Two extensions the plan should carry:

1. The `r³` law is a property of the **averaging kernel**, not of `S` being a
   BM: `Var_t[X_T] = (1/w²)∫_t^T (T−u)² d⟨S⟩_u`. Simulated compound-Poisson
   jumps, t₃ increments and a staircase (6 σ deviation threshold + 60 s
   heartbeat) oracle all reproduce it (§2 below). §12.3's worry that "Chainlink
   heartbeat/deviation updates make the aggregate stepwise ⇒ α ≉ 3" is
   **unfounded**: quantising `S` does not move the law.
2. On a 1 s stream the **exact** law is discrete:
   `Var = σ²·r(r+1)(2r+1)/(6w²)`, which exceeds `σ²r³/(3w²)` by **+32% at
   r = 5 s, +15.5% at r = 10 s, +5.1% at r = 30 s, +2.5% at r = 60 s**. Not
   optional at the horizons where quoting happens.

**VERDICT: CORRECT (continuous form); UNDERSPECIFIED at 1 s resolution.**

### 1(c) `Var_t[X_T − X_0]` with both endpoints TWAPs — the plan has no formula, and "the covariance is zero" is **FALSE**

Windows `[−60, 0]` and `[240, 300]` are disjoint at `T = 300, w = 60` (they
would overlap only if `T < w`). **Disjointness does not make the covariance
zero** — that is true of *increments*, not of *levels*. For a BM started at `t`:

```
Cov_t(X_0, X_T) = (1/w²)∫_{-w}^{0}∫_{T-w}^{T} σ²(u − t) du dv = σ²(−w/2 − t)
```

which grows without bound as `t → −∞`. MC (σ=1):

```
  t= -300: Var[X0]=259.25 (260)  Var[XT]=560.72 (560)  Cov=269.30 (270)  Var[diff]=281.37
  t= -180: Var[X0]=140.18 (140)  Var[XT]=438.33 (440)  Cov=149.83 (150)  Var[diff]=278.84
  t=  -60: Var[X0]= 19.97 ( 20)  Var[XT]=322.72 (320)  Cov= 30.12 ( 30)  Var[diff]=282.44
```

F6a's `σ²(T − w/3) = 280 σ²` is **right**, but *because* the +2·Cov term cancels
the extra level variance, not because Cov = 0. Anyone re-deriving it under a
"disjoint ⇒ independent" reading will get `σ²(T + ...)`, unbounded in `t`.

New closed form for the pre-open interpolation region `−w < t < 0` (F6a gives
only the two endpoints; the plan gives nothing):

```
Var_t[X_T − X_0] = σ²[ T − t − 2w/3 + (−t)³/(3w²) − t²/w ],     a = −t ∈ (0, w)
d/da = (1 − a/w)² ≥ 0  ⇒ monotone from 260σ² at t=0 up to 280σ² at t=−w
```

```
   t= -60.0  MC= 282.04 +-2.26   law= 280.00
   t= -45.0  MC= 279.60 +-2.24   law= 279.69
   t= -30.0  MC= 278.47 +-2.23   law= 277.50
   t= -15.0  MC= 271.55 +-2.17   law= 271.56
   t=  -5.0  MC= 263.92 +-2.11   law= 264.60
```

Relevance is not academic: **E-M6b expects the strike to be unknown for
Δ_K ≈ 1.7–2.7 s after `t = 0`**, i.e. every window begins in exactly this
regime. (In that specific case the extra term is *feed-lag* uncertainty about a
path already realised — bounded by our own reconstruction error, not by
`σ²(−w/2 − t)` — which is a *third* variance to carry and is also unstated.)

**VERDICT: UNDERSPECIFIED (plan) / the "zero covariance" reading is WRONG.**

### 1(d) §12.3's estimator — **WRONG: it does not target σ_eff**

§12.3 states: *"Target (fixed): `Var[X_{t+r} − X_t]` where X is the 60 s TWAP —
i.e. σ_eff itself."* The identification is false. Derivation (t = 0, base at
−w, `W` a standard BM):

```
r ≤ w :  Var[X_{t+r} − X_t] = σ²( r²/w − r³/(3w²) )
r ≥ w :  Var[X_{t+r} − X_t] = σ²( r − w/3 )
```

(continuous at `r = w`, both giving `2σ²w/3`), against what is needed,
`Var_t[X_T] = σ²r³/(3w²)` in-window and `σ²(τ − 2w/3)` pre-window.

The gap is exact and structural:

```
X_{t+r} − X_t = ( X_{t+r} − E_t[X_{t+r}] )  +  ( E_t[X_{t+r}] − X_t )
                 ^ what we need (F_t-orthogonal)   ^ F_t-MEASURABLE roll-off
Var[increment] = Var_t[X_{t+r}] + σ²( r²/w − 2r³/(3w²) )
```

The second term is the deterministic roll-off of the trailing window — the part
of the average that is *already known at t* — and the rolling-increment sample
variance counts it as risk.

```
       r    MC roll   law roll     NEEDED  V ratio  sigma_eff x
     1.0     0.0165     0.0166     0.0001   179.00        13.38
     5.0     0.4054     0.4051     0.0116    35.00         5.92
    10.0     1.5698     1.5741     0.0926    17.00         4.12
    20.0     5.9453     5.9259     0.7407     8.00         2.83
    30.0    12.5205    12.5000     2.5000     5.00         2.24
    60.0    39.6560    40.0000    20.0000     2.00         1.41
   120.0    99.2913   100.0000    80.0000     1.25         1.12
   300.0   278.6909   280.0000   260.0000     1.08         1.04
```

**At r = 10–30 s — precisely the band §12.3 says is estimable now with ~1,500
non-overlapping samples/coin — the proposed estimator overstates σ_eff by
2.2–4.1×.** §12.3's own sensitivity note says a 2× σ error moves p̂ by up to
16 c (§5.3 below); this is worse than 2×.

Unbiasedness: the sample variance of non-overlapping increments **is** unbiased
for `Var[X_{t+r} − X_t]` (stationary increments, `ddof=1`) — so the estimator is
unbiased for the wrong estimand. It is *not* unbiased for the conditional
`Var_t[X_T]` and no amount of data fixes it.

Estimator variance / CI width (n independent samples, Gaussian):
`SD(log V̂) = √(2/n)`; 95% half-width on `V` = `1.96√(2/n)`, on `σ` = half that.

| n | ±95% on Var | ±95% on σ |
|---|---|---|
| 400 | ±13.9% | ±6.9% |
| 768 | ±10.0% | ±5.0% |
| 1,500 | ±7.2% | ±3.6% |
| 4,860 (r=10, 13.5 h) | ±4.0% | ±2.0% |

Under stochastic vol these are *within-regime* numbers only; across regimes the
day-level vol innovation does not average out (§3(d)).

**Constructive fix — and it needs no new data.** The rolling-increment
variogram is a **two-parameter curve in (σ², w)**, so fit the shape rather than
a power:

```python
def law(r, s2, w):                      # observable TWAP-increment variogram
    return s2*((r**2/w - r**3/(3*w**2)) if r <= w else (r - w/3))
# WLS in log space, weights 2/n_r, over r = 10..600 s, non-overlapping samples
```

On 13.5 h of ONE synthetic coin (exactly the data §12.3 says exists):

```
  n per r: [4854, 2427, 1618, 1078, 809, 539, 404, 269, 202, 161, 107, 80]
  sigma^2_hat = 0.9209 (true 1.000)  +-0.0665
  w_hat       = 56.31 s (true 60)    +-5.17 s
```

σ² and **w** are both identified from the observable `X` stream alone, without
ever needing `S`. σ_eff then follows analytically from §2's law. Two bonuses:
the intercept/slope ratio of the `r ≥ w` branch is `−w/3`, giving an
**independent test of the 60 s window hypothesis** (§10 discovery 1) from price
data alone; and the fit is a genuine goodness-of-fit test of the MA(w)
structure, which is what E-M6c was trying to be.

Where the conditional law itself must be checked directly (recommended as the
secondary arm), the estimator is the conditional innovation, which needs `S`:

```
e_r(t) = X_{t+r} − E_t[X_{t+r}] = (1/w) Σ_{u=t+1}^{t+r} (S_u − S_t)
```

`S` is observable on the Binance leg. On the Chainlink leg it is recoverable at
30 s resolution from the two recorded streams
(`60·X60(t) − 30·X30(t) = 30 × mean S over [t−60, t−30]`) — worth stating, since
the plan currently treats `S` as unobservable there.

**VERDICT: WRONG (estimand), with a fix that uses the same data.**

### 1(e) MA(60) bias factors — **CORRECT to leading order**

Naive RV of `X` sampled every δ and scaled by n understates σ² by

```
factor = 1 / ( Var[X_{t+δ}−X_t] / δ ) = (w/δ) / (1 − δ/(3w))
```

| δ | plan says | exact | MC |
|---|---|---|---|
| 1 s | ~60× | 60.34× | 60.64× |
| 5 s | ~12× | 12.34× | 12.39× |
| 30 s | ~2× | **2.40×** | 2.40× |
| 60 s | — | 1.50× | 1.51× |

Leading order is `w/δ`, exact carries the `(1 − δ/3w)⁻¹` correction. The 30 s
number should read 2.4×, not 2×. **VERDICT: CORRECT (minor: 30 s is 2.4×).**

---

## 2. α estimation (E-M6c) — **WRONG / not well posed**

Five separate defects, any one of which invalidates the headline deliverable
("α with a CI").

**(1) Wrong estimand ⇒ guaranteed spurious refutation.** Applied to the
rolling increments §12.3 specifies, the local exponent of the true law is

```
   local log-log slope alpha(r) = dlogV/dlogr
   r=   5.0  alpha_rolling=1.971   alpha_conditional=3.000
   r=  10.0  alpha_rolling=1.941   alpha_conditional=3.000
   r=  20.0  alpha_rolling=1.875   alpha_conditional=3.000
   r=  30.0  alpha_rolling=1.800   alpha_conditional=3.000
   r=  60.0  alpha_rolling=1.500   alpha_conditional=3.000
   r= 150.0  alpha_rolling=1.154   alpha_conditional=1.364
   r= 300.0  alpha_rolling=1.071   alpha_conditional=1.154
```

Analytically `α_rolling(r) = (2 − r/w)/(1 − r/3w)` for `r ≤ w` and
`r/(r − w/3)` for `r ≥ w` — it lives in **(1, 2]** and can never equal 3. Chord
slopes a 2-point fit would report:

```
   r=10->30  (the '~1500 non-overlapping samples' band) rolling=1.886  conditional=3.000
   r=10->300 (pooled across the w breakpoint)           rolling=1.523  conditional=2.335
   r=30->300                                            rolling=1.350  conditional=2.017
```

Simulated on a real path (2 M s, σ const):

```
  A rolling (plan)   rs=[10,20,30]  alpha_hat= 1.889
  B conditional      rs=[10,20,30]  alpha_hat= 2.910
  A rolling pooled with r=300                 alpha_hat= 1.481
  B conditional pooled with r=300             alpha_hat= 2.966
```

**α̂ ≈ 1.89 is the correct answer for the estimator the plan specifies, and it
would be read as refuting α = 3.** §12.3 says an α ≉ 3 result means *"the entire
endgame model — participation frontier, do-not-quote zone, Q_max behaviour — is
re-derived"*. As written, E-M6c fires that clause with certainty.

**(2) No regime split.** §12.3 lists the r = 10–30 s samples and the r = 300 s
samples in the same design with a single α. Even on the **correct** estimand,
pooling across the breakpoint `r = w` gives an answer that is a pure artefact of
how the design points are weighted — α = 2.02 for a 30↔300 chord, 2.34 for
10↔300, 2.97 for the 4-point LS fit (dominated by the three in-window points).
The two regimes have different laws (α = 3 in-window, α → 1 pre-window with the
`−2w/3` offset), and neither pooled number tests anything.
The right specification is not "piecewise power" either: the offset means the
pre-window branch is not a power law at all. Fit the **two-parameter variogram**
of §1(d) and report `(σ̂², ŵ)` with a shape-GoF statistic; α is the wrong
parameterisation of the object.

**(3) 1 s discretisation biases α by −0.10 — the same size as the effect α could
detect.** With the exact discrete law `r(r+1)(2r+1)/(6w²)`:

```
   r=  5: MC= 0.01526  discrete= 0.01528  continuous r^3/3w^2= 0.01157  bias +32.0%
   r= 10: MC= 0.10697  discrete= 0.10694  continuous          0.09259  bias +15.5%
   r= 30: MC= 2.63571  discrete= 2.62639  continuous          2.50000  bias  +5.1%
   alpha_hat with 1-s sampling = 2.9033  (true continuous alpha = 3)
```

(Fix: trapezoid/midpoint the innovation sum, or test against the discrete law.)

**(4) α is insensitive to everything except increment autocorrelation.** What
does *not* move it: heavy tails, jumps, staircase quantisation of the oracle —
i.e. all the mechanisms §12.3 names as the reason to measure it.

```
   compound-Poisson jumps         alpha_hat=2.898
   t3 increments                  alpha_hat=2.897
   staircase (6sd dev / 60s hb)   alpha_hat=2.905      [iid-BM reference 2.899]
```

What does move it, and only weakly:

```
   AR(1) increment phi=-0.4: alpha_hat=2.819      phi=+0.2: alpha_hat=2.946
   AR(1) increment phi=-0.2: alpha_hat=2.861      phi=+0.4: alpha_hat=2.998
```

So α's entire informative range is ±0.09 (a very large ±0.4 oracle-increment
autocorrelation), **the same magnitude as the discretisation bias (−0.10) it is
confounded with.** α is a precise measurement of (discretisation + oracle
autocorrelation), not a test of the settlement model.

**(5) Power — computed, and it is the opposite of the problem.** MC over
8 coins × 86,400 s/day, per-day log-vol N(0, 0.35²), coin loading ρ = 0.9 on the
common factor, in-window design B at r ∈ {5,10,20,30,45,60}, intercepts absorbed
per coin-day:

```
  days= 1 coins=8: alpha_hat = 2.9030 +- 0.0047   -> |3-2.5|/SE = 106 sigma
  days= 3 coins=8: alpha_hat = 2.9019 +- 0.0027   -> |3-2.5|/SE = 186 sigma
  days= 7 coins=8: alpha_hat = 2.9020 +- 0.0014   -> |3-2.5|/SE = 360 sigma
```

**The 95% CI half-width on α at one day of 8 coins is ≈ 0.009, so α = 2.5 is
separated at ~106 σ and even α = 2.99 is rejected.** The common vol factor is *not* a
constraint here because it loads on the intercept, not the slope — the 8-coins-
are-really-1.4 concern (§8 MF-8) does not transfer to a slope estimate. The
binding constraint on E-M6c is **specification, not sample size**, and with
this much power every specification error is reported as a significant finding.

**Also: name collision.** `PM_MECHANISM_EXPERIMENTS.md` already defines **E-M6c
= manipulation-successor screen** (H-PM1b). §12.3 defines E-M6c = variance-law
estimation. Rename (E-M6d).

**VERDICT: WRONG (misspecified estimand + no regime split + confounded with
discretisation); power is ample and is not the issue.**

---

## 3. The MLE-on-outcomes σ fit

### 3(a) Identification — the scale **is** identified; the degeneracy is with edge measurement error

A probit `y = 1{edge + σZ > 0}` with observed, varying `edge` and no intercept
identifies `1/σ`. But with `edge_obs = edge_true + η`, `η ⊥ edge_true`,
`Var(edge_true) = τ²`, `Var(η) = ω²`, `λ = τ²/(τ²+ω²)`:

```
P(y=1 | edge_obs) = Φ( λ·edge_obs / √(σ² + λω²) )   ⇒   σ̂ → √(σ² + λω²)/λ
```

```
  omega=0.00: sigma_hat= 0.992   analytic= 1.000   inflation= 0.99x
  omega=0.25: sigma_hat= 1.096   analytic= 1.093   inflation= 1.10x
  omega=0.50: sigma_hat= 1.367   analytic= 1.369   inflation= 1.37x
  omega=1.00: sigma_hat= 2.442   analytic= 2.449   inflation= 2.44x
```

This is **not a bug for quoting** — the MLE returns the *predictive* σ, which is
the right denominator for `p̂` given our own noisy edge. It **is** a fatal
inconsistency for §12.3's architecture, which uses the same symbol `σ_eff` for
three different objects:

| object | what it is | where §12.3 uses it |
|---|---|---|
| physical `σ_TWAP` | vol of the settlement TWAP | "shape from Binance", `κ(r)` ratio |
| predictive `σ_pred` | includes basis/strike/latency measurement error | "fit by MLE on realized winners" |
| `σ_⊥` | basis noise, added additively | "never fitted into the blend weights" |

The MLE leg and the `κ(r)`-ratio leg estimate **different quantities**, and
§12.3 blends them as if they were the same. They differ by exactly the
inflation factor above. Pick one target: for quoting, `σ_pred` is the correct
one, and then the Binance-shape/κ chain is a *prior* on it, not a measurement
of it.

**Is `w` identified from outcomes?** Weakly, and it trades off against the
scale. True w = 60, decision at r = 30 s, 200 k windows:

```
  assumed w'=  30: mean loglik=-0.12824   with free scale=-0.12824  fitted scale=0.996
  assumed w'=  60: mean loglik=-0.06772   with free scale=-0.06770  fitted scale=1.020
  assumed w'= 120: mean loglik=-0.38902   with free scale=-0.14075  fitted scale=3.948
  assumed w'= 300: mean loglik=-1.96346   with free scale=-0.28819  fitted scale=15.230
```

`w' = 120` recovers 62% of its log-likelihood deficit by rescaling σ by 3.95×.
The separation between hypotheses is ~0.06 nats/window, whereas **E-M6 separates
the same hypotheses deterministically at ≥ 99% per window**. Do not fit `w` from
outcomes; take it from E-M6 and hold it fixed.

**VERDICT: identified but CONFLATED — three distinct σ's under one symbol.**

### 3(b) The σ_⊥ additive floor — kills identification of the blend weights exactly where it is needed, and is double-counted

Two findings.

**(i) Information collapse.** For `σ_tot² = σ_⊥² + c·σ_shape²`, θ = log c, the
per-window Fisher information scales by `ρ² = (σ_eff²/σ_tot²)²`:

```
  sigma_perp=0.7 bps, sigma_eff=0.42 bps (r=30 s, 15%/yr): rho=0.265  info x0.070 -> n x14.3
  sigma_perp=0.7 bps, sigma_eff=0.84 bps (r=30 s, 30%/yr): rho=0.590  info x0.348 -> n x 2.9
  sigma_perp=0.7 bps, sigma_eff=2.40 bps (window entry)  : rho=0.922  info x0.849 -> n x 1.2
  sigma_perp=0.7 bps, sigma_eff=8.60 bps (window open)   : rho=0.993  info x0.987 -> n x 1.0
```

All identification of the blend weights comes from **early-window, high-vol**
observations. Quiet-regime late-window windows — the ones §12.3 flags as the
"real quiet-regime regime" — contribute ~7% of an observation each. And if σ_⊥
is fixed too high, the weights absorb the error with a *negative* bias; §12.3
explicitly forbids fitting σ_⊥, which guarantees that bias transfer. **Profile
over σ_⊥ or fit it jointly.**

**(ii) `κ(r)` already contains σ_⊥ — the additive floor double-counts it.**
`X_CL = X_bin + basis` ⇒ `Var_CL = Var_bin + Var_basis (+2Cov)`, so by
construction

```
κ(r) = Var_CL/Var_bin = 1 + σ_⊥²(r)/σ_bin²(r)
```

Then `σ̂_eff = √κ · shape` **already is** `√(σ_bin² + σ_⊥²)`, and §12.3's
instruction to add σ_⊥² separately on top applies it twice:

```
  vol= 15% r= 30s: sigma_bin=0.422 kappa= 3.75  correct=0.818  double-counted=1.076 (+31.6%)
       d=1.0: p_hat correct=0.8413  double-counted=0.7763   error = -6.51 c
  vol= 15% r= 10s: sigma_bin=0.081 kappa=75.17  correct=0.705  double-counted=0.993 (+41.0%)
       d=1.0: p_hat correct=0.8413  double-counted=0.7610   error = -8.04 c
  vol= 30% r= 60s: sigma_bin=2.389 kappa= 1.09  correct=2.490  double-counted=2.586 ( +3.9%)
       d=1.0: p_hat correct=0.8413  double-counted=0.8321   error = -0.92 c
```

**A 6.5–8 c p̂ error in the quiet regime, on a 2–4 c book.**

Also, the ratio-precision claim ("a ratio needs far fewer observations") is
**correct but conditional on paired sampling**:
`SD(log κ̂) ≈ √(4(1−ρ²)/n)`.

```
  rho=0.000: MC=0.05433  analytic=0.05164  -> +-10.65% (95%)
  rho=0.900: MC=0.02134  analytic=0.02251  -> +- 4.18%
  rho=0.990: MC=0.00757  analytic=0.00728  -> +- 1.48%
  rho=0.999: MC=0.00237  analytic=0.00231  -> +- 0.46%
```

If the two series are sampled on different intervals the common vol factor does
**not** cancel and precision collapses to the unpaired `√(4/n)` = ±10.7%.
§12.3 does not say "paired". State it.

**VERDICT: WRONG (σ_⊥ double-counted) + UNDERSPECIFIED (pairing, σ_⊥ profiling).**

### 3(c) Non-Φ link — the MLE is inconsistent, and the bias is in the longshot direction

Truth = variance-matched t₅, fit = Φ with a free scale; scale chosen to minimise
KL under a `N(0, τ²)` edge distribution:

```
  tau=0.5 : KL-optimal sigma_hat=0.8502 (-15.0%)
      d=0.5: true 0.72647  Phi-fit 0.72176   -0.47 c
      d=1.0: true 0.87342  Phi-fit 0.88023   +0.68 c
      d=2.0: true 0.97534  Phi-fit 0.99067   +1.53 c
      d=3.0: true 0.99414  Phi-fit 0.99979   +0.57 c
  tau=1.0 : KL-optimal sigma_hat=0.9011  (-9.9%)
      d=0.5: -1.60 c;  d=1.0: -0.70 c;  d=2.0: +1.14 c;  d=3.0: +0.54 c
  tau=2.0 : KL-optimal sigma_hat=0.9648  (-3.5%)
      d=0.5: -2.86 c;  d=1.0: -2.34 c;  d=2.0: +0.56 c;  d=3.0: +0.49 c
```

Sign and magnitude: **σ̂ is biased DOWN 3.5–15%**, `p̂` is too *low* in the
shoulders (|d| ≈ 0.5–1) by 0.5–2.9 c and too *high* at |d| ≥ 2 by 0.5–1.5 c.
In longshot terms at d = 3 the model prices the tail at 0.021 c against a true
0.586 c — a **28× under-pricing of the thing the engine is structurally short**
(§4(a)). A free scale cannot repair a link misspecification; it buys ATM
calibration at the cost of the tails, and the Fisher information for the scale
is **zero at d = 0** and peaks at |d| ≈ 1.5 — i.e. the MLE is anchored exactly in
the shoulder region where the t₅/Φ discrepancy changes sign. F14's SHOULD-FIX
(pre-commit a fat-tailed link) should be promoted to MUST-FIX for any use of the
MLE.

**VERDICT: WRONG (inconsistent; bias signed and quantified above).**

### 3(d) How many windows to pin the blend weights?

Per-window Fisher information for `log σ`, `I(d) = [φ(d)d]²/(p(1−p))`:

```
     info at |d|=0.0: 0.0000   <- ZERO at the money
     info at |d|=0.5: 0.1452
     info at |d|=1.0: 0.4386
     info at |d|=1.5: 0.6054   <- peak
     info at |d|=2.0: 0.5245
     info at |d|=3.0: 0.1311
```

`d_t ~ N(0, (V_0−V_t)/V_t)` for a market opening ATM, so the information has a
profile over the window:

```
 t(s)   r     V_t/s^2   sd(d_t)   E[info per window]
    0   300    260.000     0.000          0.00000
   60   240    200.000     0.548          0.14095
  120   180    140.000     0.926          0.25551
  180   120     80.000     1.500          0.31109   <- max
  240    60     20.000     3.464          0.22101
  270    30      2.500    10.149          0.08530
  290    10      0.093    52.981          0.01661
```

At the best decision time (`t = 180 s`, `r = 120 s`): **~320 windows for
SE(log σ) = 0.10, ~1,290 for SE = 0.05** — but only if each window contributes
**one** row.

> **PSEUDO-REPLICATION.** There is exactly **one** `y` per window. §12.3 does not
> say at which decision tick `p̂ᵢ` is evaluated; stacking the ~300 per-second
> decision rows of a window as independent observations inflates the apparent
> information by up to 300× (and the late rows carry ~0 information anyway,
> sd(d) = 10 at r = 30 s). Either one row per window, or a likelihood for the
> whole path — the latter requires the joint law and is much harder.

And the sampling unit for anything that varies at the day level is the **day**,
not the window (per-day log-vol dispersion 0.30, 2,304 windows/day):

```
   days=  1: sigma_hat mean=1.0536  SD across reps=0.3527   (windows =   2304)
   days=  5: sigma_hat mean=1.0333  SD across reps=0.1596   (windows =  11520)
   days= 20: sigma_hat mean=1.0210  SD across reps=0.0625   (windows =  46080)
   days= 60: sigma_hat mean=1.0174  SD across reps=0.0316   (windows = 138240)
```

`SD ∝ 1/√days`, **not** `1/√windows`. One level parameter needs ~60 days for
±3%; **§12.3's "~4 blend weights + coarse seasonal" is not estimable on 13.5 h
of data, nor on 5 days.** (Note also the persistent +1.7% upward bias at 60
days: a single blended σ across heterogeneous days is not the mixture's probit —
Jensen, in the direction of over-stating σ.)

**VERDICT: UNDERSPECIFIED (decision-tick unmarked ⇒ pseudo-replication risk);
day-clustered sample requirement ~60 days for one parameter.**

---

## 4. The quoting engine after the §12.2 fixes

### 4(a) The `[0.01, 0.99]` clamp — argmax fine, three real problems

The per-level EV is an argmax over a **finite** action set with an outside
option (0), so existence is trivial and the clamp cannot break it. The defects
are elsewhere.

1. **The floor is an absorbing attractor for short-longshot inventory.** Once
   `p̂ < tick`, the cheapest quotable ask is the floor and its edge is positive
   for *any* inventory:
   ```
      p_hat=0.0050: cheapest ask 0.010 -> +0.50 c/share edge
      p_hat=0.0020: cheapest ask 0.010 -> +0.80 c/share edge
   ```
   Combine with §4(b) (`Q_max ∝ 1/v` → 25× the ATM cap at p̂ = 0.01) and §3(c)
   (Φ under-prices that tail by up to 28×): the engine takes its largest
   position in the trade where the model error is largest and the
   loss-given-adverse is 99× the premium. **This is the single most dangerous
   interaction in the current design.**
2. **The grid is state-dependent.** `tick_size_change` 0.01 → 0.001 fires 328×
   over 130 windows (§10). A hard-coded `[0.01, 0.99]` is wrong in the 0.001
   regime — it forbids the only levels that exist near the boundary — and the
   feasible-level enumeration must be rebuilt on every tick-regime change. The
   clamp must be `[tick, 1 − tick]` with `tick = tick(state)`.
3. **`p̂` itself must be clamped, not just the quote.** §12.2 reports
   `Q_max = 9.9e15`, which back-solves to `p̂ ≈ 2.5e-13` (i.e. `|d| ≈ 7.2`). No
   free-data model supports a probability statement at 1e-13; floor `p̂` at
   (tick/2) or use the fat-tailed link.
4. **Argmax over 3–4 estimated EVs is a small-N selector.** With plausible fill
   counts the best-vs-second gap is ~2 SE — the failure mode this repo already
   records (vBTC rolling-IC selector: noise-dominated and value-negative). Use
   shrinkage across levels plus the §11 requote hysteresis, not a raw argmax.

**VERDICT: argmax CORRECT; clamp UNDERSPECIFIED (state-dependent tick, p̂ floor);
boundary interaction WRONG.**

### 4(b) `Q_max = κ/(γ·p̂(1−p̂))` — **WRONG** (dimensions, derivation, behaviour)

**Dimensional audit.** `γ [1/$]`, `q [shares]`, `v = p̂(1−p̂) [price²]`. So
`γ·|q|·v` has units of **price ($/share)** — it is the *reservation skew*, not a
risk. Capping it at κ caps how far the quote is pushed, which is a legitimate
constraint but is **not** "a constant risk budget". The label and the formula
disagree.

**Internal inconsistency.** The stated rationale is a constant risk budget. The
CARA risk charge is `R(q) = qp̂ − CE(q) ≈ (γ/2)q²v` locally; a constant budget on
*that* gives

```
Q_max = √(2κ)/(γ√v)   ∝ 1/√v      NOT   κ/(γv) ∝ 1/v
```

**Behaviour.** Calibrated so every rule gives 5,000 shares ATM (γ = 4e-4):

```
    p_hat        v |   plan 1/v quad 1/sqrt(v) | loss-cap LONG loss-cap SHORT | exact CARA SHORT
    0.500  0.25000 |       5000           5000 |          5000           5000 |             5000
    0.250  0.18750 |       6667           5774 |         10000           3333 |             4887
    0.100  0.09000 |      13889           8333 |         25000           2778 |             5885
    0.050  0.04750 |      26316          11471 |         50000           2632 |             7009
    0.010  0.00990 |     126263          25126 |        250000           2525 |            10310
    0.001  0.00100 |    1251251          79096 |       2500000           2503 |            15792
```

The formula is symmetric in the sign of `q` while the economics are violently
asymmetric:

```
   p=0.500: plan cap      5000 sh -> loss if Up =    2500 $   ratio  1.0x
   p=0.050: plan cap     26316 sh -> loss if Up =   25000 $   ratio 10.0x
   p=0.010: plan cap    126263 sh -> loss if Up =  125000 $   ratio 50.0x
```

**At p̂ = 0.01 the plan's cap permits a 50× ATM dollar loss on the short side.**
On the long side it is harmless (premium at risk `q·p̂ = κ/(γ(1−p̂))` is bounded)
— so a "hard capital cap layered on top" (§12.2 defect 2) fixes the symptom but
leaves the wrong *shape*: it will bind ATM long before it binds where the real
risk is.

**Correct replacements (explicit formulas).**

```
loss given adverse resolution      L_adv(q) = |q| · ( p̂     if q > 0
                                                      1 − p̂ if q < 0 )
constant-loss cap                  Q_max^long  = κ_$ / p̂        (grows: correct)
                                   Q_max^short = κ_$ / (1 − p̂)  (≈ κ_$: correct)

CVaR_β of a two-atom Bernoulli PnL, π_adv = P(adverse), G = gain if favourable:
   CVaR_β = L_adv                                        if π_adv ≥ 1 − β
          = [ π_adv·L_adv − (1−β−π_adv)·G ] / (1 − β)    otherwise

exact CARA risk charge (consistent with the g-ratio quotes already adopted):
   R(q) = q·p̂ + (1/γ)·ln( p̂ e^{−γq} + 1 − p̂ )  ≥ 0 ,  cap R(q) ≤ κ/γ
   asymptotics: R(q) → |q|·(1−p̂) + ln(p̂)/γ   (q → −∞)   [slope = loss/share]
                R(q) →  q ·p̂     + ln(1−p̂)/γ (q → +∞)
```

Numerically (`p = 0.01`, short): plan 126 k shares, quadratic 25 k, exact CARA
10.3 k, loss-cap 2.5 k.

Note that **CVaR adds nothing over the max-loss cap in the working range**: for
any β ≤ 0.95 and p̂ ∈ [0.05, 0.95] the adverse atom exceeds the tail mass and
`CVaR_β = L_adv` exactly; deeper in the tail CVaR *softens* the cap (`CVaR_95 =
190` vs `L_adv = 990` at p̂ = 0.01 short) — the wrong direction here, because the
tail is precisely where the model is least trustworthy.

**Recommendation:** price with the exact CARA `R(q)` (already adopted for the
quotes), and impose `L_adv(q) ≤ κ_$` as the hard cap. Do not use variance, and
do not use CVaR, as the binding constraint on a binary.

**VERDICT: WRONG (mislabelled, internally inconsistent, 50× loss asymmetry).**

### 4(c) The pair conditions — algebra CORRECT, decision rule VACUOUS, bookkeeping UNDERSPECIFIED

**Payoff algebra — correct.**

```
 BUY Up @a + BUY Down @b:  payoff = 1 − a − b in both states   (a+b<1 ⇒ lock)
 SELL Up @a + SELL Down @b (mint $1 first): payoff = a + b − 1 (a+b>1 ⇒ lock)
```

Both verified state-by-state; §12.2's correction of the earlier "short Up + long
Down" framing is right (that is a doubled directional bet, not a pair).

**But they are the same trade.** Under the unified book / CTF mint,
`SELL Up @a ≡ BUY Down @(1−a)`, so config 2 = config 1 on the complements with
cost `(1−a)+(1−b) = 2−a−b < 1 ⟺ a+b > 1`. There is **one** pair trade, not two.
Operationally the distinction is stronger than cosmetic: Polymarket has **no
naked short** (a SELL requires holding the token), so from a flat book only
config 1 is placeable. The state space should be `q_up, q_down ≥ 0` with shorts
expressed as longs of the complement. *(Flag as a mechanism proposition to
verify — P-M3d — not an assertion.)*

**The condition `a+b < 1` is vacuous as a trigger.** Any maker quoting inside
fair value satisfies it identically:

```
   p_hat=0.50, half-spread 0.01: bid_up=0.490 bid_down=0.490  a+b=0.980 = 1 − 2δ
   p_hat=0.20, half-spread 0.02: bid_up=0.180 bid_down=0.780  a+b=0.960 = 1 − 2δ
   p_hat=0.85, half-spread 0.01: bid_up=0.840 bid_down=0.140  a+b=0.980 = 1 − 2δ
```

`1 − a − b = 2δ` always. The binding objects are `P(both legs fill)` and the
adverse selection on the **first** leg:

```
EV_pair = P(complete)·(1 − a − b) + (1 − P(complete))·EV_naked(first leg)
   lock=0.02, EV_naked=-0.004  ->  breakeven P(complete) = 0.167
```

`PM_MECHANISM_THEORY.md` has this (P-M3c); §12.2 does not, and §12.2 is what the
engine was written from.

**Exact bookkeeping in (q_up, q_down) — currently unstated.**

```
state: q_up, q_down ≥ 0 ; cash C ; average cost bases c_up, c_down
fill BUY Up n @P :  c_up ← (c_up·q_up + n·P)/(q_up + n) ; q_up += n ; C −= n·P
merge m = min(q_up, q_down): q_up −= m ; q_down −= m ; C += m
        realised = m·(1 − c_up − c_down)                [AVERAGE-COST convention]
resolution:  C += q_up·1{Up} + q_down·1{Down}
risk coordinate q = q_up − q_down ; paired m is riskless
```

Three things §12 must add:

- **Cost convention.** Average-cost vs FIFO give different splits between
  realised pair PnL and the basis carried on the residual leg. Unstated. Fix
  average-cost and state it (it is also what makes `1 − c_up − c_down` a valid
  per-pair statistic for the E3 PnL equation, MF-5).
- **Merge is a capital decision, not a PnL decision.** Holding `m` to resolution
  and merging early pay identically ($1 either way); merging only frees
  collateral. Do not book a "merge PnL" event.
- **Partial fills / legging is adversely selected.** Residual `= n_up − n_down`
  is acquired at the price of the leg the counterparty *chose* to hit, so its
  markout is worse than the unconditional. And the second leg's reservation must
  be evaluated at the **post-first-fill** `q` — quoting both legs off the same
  pre-fill `q` double-counts the inventory credit.

```
     n_up= 50 n_down= 50 -> m=50 pairs, naked 0
     n_up= 50 n_down= 20 -> m=20 pairs, naked long Up = 30
     n_up= 50 n_down=  0 -> m= 0 pairs, naked long Up = 50
```

**VERDICT: algebra CORRECT; trigger condition WRONG (vacuous); bookkeeping
UNDERSPECIFIED.**

---

## 5. Other numerical errors in §2 / §3 / §11 / §12

### 5.1 §12.1's quotable-horizon table `r*` is wrong twice — ATM is 46 s, not 123 s

Reverse-engineering the published numbers pins the exact error: §12.1 computed
the taker-fee credit as `0.07·min(p,1−p)·p` — it multiplied by the **price**.

```
  |d|=0.0 target=123 s   fee = 0.07*min(p,1-p)*p : m=0.02750  r*=123.1 s  MATCHES
  |d|=1.0 target= 92 s   fee = 0.07*min(p,1-p)*p : m=0.01934  r*= 91.5 s  MATCHES
  |d|=2.0 target= 13 s   fee = 0.07*min(p,1-p)*p : m=0.01156  r*= 12.8 s  MATCHES
```

The on-chain formula verified in §1 is `fee = 0.07·min(p,1−p)·shares`, i.e.
**3.5 c/share ATM, not 1.75 c**. §11's own "fee ≈ 78% of the moat ATM" requires
3.5 c (`0.035/0.045 = 78%`; §12.1's 1.75 c gives 64%) — **§11 and §12.1
contradict each other by 2×.**

Second error: `r* = 123 s` and `92 s` **exceed `w = 60 s`**, so the in-window
`λ_bin = √3·φ(d)/√r` was applied outside its domain. Correct branch pre-window
is `φ(d)/√(τ − 2w/3)`:

```
    |d|=0.0: r= 60.0s  in-window=0.08921  correct=0.08921  overstated  0.0%
    |d|=0.0: r= 92.0s  in-window=0.07204  correct=0.05532  overstated 30.2%
    |d|=0.0: r=123.0s  in-window=0.06230  correct=0.04379  overstated 42.3%
```

Corrected table (k = 1, moat = 1 tick + on-chain taker fee, correct branch):

| \|d\| | p | fee (c) | moat (c) | US L=195 ms | EU L=11 ms | plan said (US/EU) |
|---|---|---|---|---|---|---|
| 0.0 | 0.5000 | 3.500 | 4.500 | **46.0 s** (in-window) | **2.59 s** | 123 / 7 |
| 0.5 | 0.6915 | 2.160 | 3.160 | 64.2 s (pre-window) | 4.10 s | — |
| 1.0 | 0.8413 | 1.111 | 2.111 | **65.6 s** (pre-window) | **4.34 s** | 92 / 5 |
| 1.5 | 0.9332 | 0.468 | 1.468 | 45.6 s (in-window) | 2.57 s | — |
| 2.0 | 0.9772 | 0.159 | 1.159 | **12.7 s** | **0.72 s** | 13 / 1 |
| 3.0 | 0.9987 | 0.009 | 1.009 | 0.1 s | 0.01 s | — |

Consequences:

- the headline ATM number is **2.7× overstated**; the EU/US ratio (17.7×) is
  unaffected, so the *direction* of the siting decision survives, but the case
  is "46 s → 2.6 s", not "123 s → 7 s";
- **`r*(|d|)` is non-monotone**, peaking at `|d| ≈ 0.82` (US 66.6 s, EU 4.5 s),
  above the ATM value — because the fee moat collapses faster than φ(d) does.
  §11/§12 both describe the frontier as if ATM were the worst case. The worst
  case is the **shoulder**, `|d| ≈ 0.8`, where ~22% of the window is unquotable
  at US latency vs 15% ATM.

### 5.2 §11's "fee ≈ 2% of the moat at |d| = 3" — it is 0.94%

```
    |d|=0.0: fee=3.50000 c -> 77.78 % of moat     |d|=2.0: fee=0.15925 c -> 13.74 %
    |d|=1.0: fee=1.11059 c -> 52.62 %             |d|=3.0: fee=0.00945 c ->  0.94 %
```

78% ATM confirms the 3.5 c reading; the |d| = 3 figure is 2× off. The
qualitative claim ("far from the money the moat is the tick, not the fee") is
strengthened, not weakened.

### 5.3 §12.3 "sensitivity of p̂ to a 2× σ error is up to 23 cents" — it is 16.1 c

`sup_d |Φ(d) − Φ(d/2)|` is attained at `d² = 8ln2/3`, `d = 1.3596`:
`Φ(1.3596) − Φ(0.6798) = 0.9130 − 0.7517 = 0.1613`. The argument is unaffected
(σ is still the dominant p̂ error term), but the number should be **16 c**.

### 5.4 §12.3 "pickoff exposure is volatility-free, so bad σ never gets us sniped" — **FALSE, in the dangerous direction**

`φ(d)√(3L/r)` is σ-free **given d**; but `d = edge/σ_eff`, so a σ error moves d,
moves φ(d), and moves `r*`:

```
   true |d|  sigma err  believed |d|  believed r*   true r*  over-quote
       1.00       0.50x          2.00         12.7 s      65.6 s      +52.9 s
       1.00       0.70x          1.43         51.3 s      65.6 s      +14.4 s
       2.00       0.50x          4.00          0.0 s      12.7 s      +12.7 s
       0.50       2.00x          0.25         60.1 s      64.2 s       +4.1 s
```

**Under-estimating σ by 2× at true |d| = 1 leaves us quoting 53 s deeper into the
sniping zone than the model would allow if σ were right.** Since §1(d) shows
the proposed estimator errs by 2.2–4.1× (over-statement) and §3(b)'s
double-count adds another +30–40%, while §3(c)'s link error biases σ̂ *down*
3.5–15%, the sign of the net error is currently unknown — which is the whole
problem.

### 5.5 σ_eff / σ_⊥ crossover numbers in §12.3 — **CORRECT**

At 15%/yr and r = 30 s: `σ_eff = σ·r^{3/2}/(√3·w) = 0.4223 bps` vs σ_⊥ = 0.7 bps
✓ (§12.3's "0.42 vs 0.70"). Crossover `σ_eff = 0.7 bps` at r = 42.0 s (15%),
26.5 s (30%), 21.8 s (40%), 16.7 s (60%) — consistent with F12's 17–26 s band at
30–60% vol.

### 5.6 Consecutive windows are not independent — mechanical MA(1) of +3.6%

Back-to-back 5-min markets share a 60 s average: `X_0` of window k+1 **is** `X_T`
of window k. Hence

```
Cov(X_{t2}−X_{t1}, X_{t3}−X_{t2}) = +σ²·w/6 ;  ρ = (w/6)/(T − w/3) = 0.0357
MC: +0.0347 +- 0.0018     ->  outcome-sign correlation ≈ +0.023
```

Small, but it means "window = the unit" (§8 MF-1) is not literally right; day
clustering covers it. **NOTED, not a MUST-FIX.**

### 5.7 Small items

- **§2's `E_t[X_T] ≈ F_t + μ̂(τ − w/2)` uses `F_t` (Binance mid) where the
  derivation needs `S_t` (Chainlink aggregate).** The stream-anchored
  construction three lines below defines `Ŝ_t = X_t + (F_t − TWAP_w(F)_t)`; §2's
  formula box should use `Ŝ_t`, else the basis level enters `d` directly. The
  `w/2` term itself is correct (F3).
- **"Skip any interval containing one of the 56 recorded gaps > 5 s" selects on
  volatility.** E0's own note is that outages cluster with bursts; dropping
  gap-containing intervals biases σ̂ **down** — in the same direction as §5.4's
  dangerous case. Use a gap indicator/weight, or report the estimate with and
  without, rather than dropping.
- **E-M6c name collision** with the manipulation-successor screen already
  defined in `PM_MECHANISM_EXPERIMENTS.md`. Rename the variance-law experiment.
- **"~1,500 non-overlapping samples/coin at r = 10–30 s"** is the r = 30 figure;
  13.5 h gives 4,854 at r = 10, 1,618 at r = 30.
- **`Q_max = 9.9e15` back-solves to p̂ ≈ 2.5e-13 (|d| ≈ 7.2)** — the engine is
  emitting probability statements 10 orders of magnitude beyond anything the
  data can support. Floor p̂ at tick/2.
- §12.3's "the loss function is probability accuracy, not vol accuracy" is a
  correct principle and is the strongest argument for making `σ_pred` (not
  `σ_TWAP`) the single named target — see §3(a).

---

## MUST-FIX table

| # | severity | finding | fix |
|---|---|---|---|
| Q1 | **MUST-FIX** | §12.3's stated target `Var[X_{t+r}−X_t]` **is not σ_eff**. Derived: `σ²(r²/w − r³/3w²)` for r≤w vs needed `σ²r³/3w²`. Overstates σ_eff by **4.12× at r=10 s, 2.24× at r=30 s** — the exact band the plan says it can estimate now | fit the 2-parameter variogram `V(r; σ², w)` on the observable stream (identifies σ² **and** w; 13.5 h of one coin gives σ² ±7%, w ±5.2 s), then get σ_eff analytically. Secondary arm: conditional innovation `e_r = (1/w)Σ(S_u−S_t)` on the Binance leg |
| Q2 | **MUST-FIX** | E-M6c as specified returns **α̂ ≈ 1.89** (correct answer for the wrong estimand) and would fire §12.3's "re-derive the entire endgame model" clause. Pooling r=10–30 with r=300 across the `w` breakpoint returns α ≈ 2.0 even on the right estimand. No regime split is specified | drop the power-law parameterisation entirely; report `(σ̂², ŵ)` + shape GoF. If α is kept, split at `r = w`, test against the **discrete** law `r(r+1)(2r+1)/6w²`, and pre-register that α's informative range is only ±0.09 |
| Q3 | **MUST-FIX** | `κ(r) = Var_CL/Var_bin` **already contains σ_⊥**; adding σ_⊥² again on top (as §12.3 instructs) inflates σ_eff by **+32% (r=30 s) to +41% (r=10 s)** at 15%/yr ⇒ **6.5–8 c of p̂ error** | use `σ_tot = √κ·shape` OR `√(shape² + σ_⊥²)`, never both. Also: state that κ requires **paired** sampling (unpaired precision ±10.7% vs ±1.5% at ρ=0.99) |
| Q4 | **MUST-FIX** | `Q_max = κ/(γ p̂(1−p̂))` is dimensionally a **price-skew cap, not a risk budget**; its own stated derivation gives `1/√v`; at p̂ = 0.01 it permits **50× the ATM dollar loss** on the short side | hard cap on loss-given-adverse-resolution `L_adv(q) ≤ κ_$` ⇒ `Q^long = κ_$/p̂`, `Q^short = κ_$/(1−p̂)`; price with the exact CARA `R(q) = qp̂ + (1/γ)ln(p̂e^{−γq}+1−p̂)`. CVaR is redundant (`= L_adv` for p̂∈[0.05,0.95]) and softens the cap in the tail |
| Q5 | **MUST-FIX** | §12.1's `r*` table used `fee = 0.07·min(p,1−p)·p` (fee × price) **and** the in-window λ_bin branch at r > w. ATM US `r*` is **46 s, not 123 s**; §11's "78% of the moat" contradicts §12.1 by 2× | adopt the corrected table (§5.1). Note `r*(|d|)` is **non-monotone**, peaking at \|d\| ≈ 0.82, not ATM — the shoulder is the worst case, not the money |
| Q6 | **MUST-FIX** | The MLE fits a **predictive** σ (absorbs basis/strike/latency error: 1.37× inflation at ω = 0.5σ) while the κ/shape chain estimates the **physical** TWAP σ. §12.3 blends them as one quantity | name ONE target. For quoting it is `σ_pred`; demote the Binance-shape/κ chain to a prior on it. Never add σ_⊥ to a MLE-fitted σ |
| Q7 | **MUST-FIX** | Boundary interaction: the `[0.01,0.99]` clamp makes the price floor an **absorbing attractor for short-longshot inventory** (+0.5–0.8 c/share edge at any q), where `Q_max ∝ 1/v` allows 25× ATM size and Φ under-prices the true tail by up to **28×** | clamp to `[tick(state), 1−tick(state)]`; floor `p̂` at tick/2; apply the loss-based cap (Q4); promote F14's fat-tailed link from SHOULD to MUST for any boundary quoting |
| Q8 | SHOULD-FIX | §12.3's MLE does not name the decision tick. One `y` per window ⇒ stacking per-second rows inflates information up to 300×. Fisher info is **0 at d=0**, peaks at \|d\|≈1.5, and by window position peaks at t = 180 s (0.311/window) | one row per window at a pre-registered decision time; ~320 windows for ±10% on log σ, ~1,290 for ±5%; **day is the clustering unit** — one level parameter needs ~60 days for ±3%, so "4 blend weights + seasonal" is not estimable on 13.5 h |
| Q9 | SHOULD-FIX | §12.3's "pickoff exposure is volatility-free so bad σ never gets us sniped" is false in the dangerous direction: a 2× σ **under**-estimate at true \|d\|=1 leaves us quoting 53 s inside the sniping zone | delete the claim; make `r*` sensitivity to σ̂ an explicit robustness requirement (quote on a conservative upper σ̂ for participation, the point estimate for pricing) |
| Q10 | SHOULD-FIX | The pair trigger `a+b<1` is **vacuous** (`1−a−b = 2δ` for any maker inside fair value). Config 1 and config 2 are the same trade on complements; PM has no naked short | replace the trigger with the completion-probability mixture (breakeven `P(complete) = 0.167` at lock 2 c / naked EV −0.4 c); carry `q_up, q_down ≥ 0`; fix the average-cost convention; second-leg reservation must use post-first-fill `q` |
| Q11 | SHOULD-FIX | 1 s sampling biases the in-window variance **+15.5% at r=10 s, +5.1% at r=30 s**; §1(e)'s 30 s bias factor is 2.40×, not 2× | use `r(r+1)(2r+1)/(6w²)` (or midpoint the innovation sum); correct the 30 s figure |
| Q12 | SHOULD-FIX | Pre-open (`t<0`) has no variance formula; every window also opens with `Δ_K ≈ 1.7–2.7 s` of unknown strike (E-M6b) | add `Var_t[X_T−X_0] = σ²[T−t−2w/3+(−t)³/3w² − t²/w]` for `−w<t<0` (MC-verified); and a separate strike-reconstruction-error term for `t ∈ [0, Δ_K]` |
| Q13 | NOTED | "cov(X_0, X_T) = 0 because the windows are disjoint" is **false** (`Cov = σ²(−w/2−t)`); F6a's 280σ² is right only because the covariance cancels | state the covariance explicitly so the result is not re-derived wrongly |
| Q14 | NOTED | Skipping gap-containing intervals selects **against** high-vol regimes (outages cluster with bursts) — biases σ̂ down | gap indicator/weight, or report with-and-without |
| Q15 | NOTED | `E-M6c` name collision (variance law vs manipulation screen); §2 writes `F_t` where it means `Ŝ_t`; "23 c" σ-sensitivity is 16.1 c; consecutive windows carry a mechanical +3.6% MA(1); §11's "2% at \|d\|=3" is 0.94% | one-line edits |

---

## What was checked and found CORRECT

- `Var_t[X_T] = σ²(τ − 2w/3)` pre-window, τ to `T` (MC to <1 SE at τ = 60–300 s).
- `Var_t[X_T] = σ²r³/(3w²)` in-window, zero cross-term by measurability;
  continuity at `τ = w`; and — new — invariance to jumps, t₃ tails and staircase
  quantisation of the oracle.
- `Var_t[X_T−X_0] = σ²(T − w/3)` for `t ≤ −w` (F6a).
- MA(60) understatement factors ~60× / ~12× at 1 s / 5 s sampling.
- `κ(r)`-as-a-ratio needs far fewer observations (±1.5% at n=1,500, ρ=0.99) —
  conditional on paired sampling.
- σ_eff vs σ_⊥ crossover numbers (0.42 vs 0.70 bps at 15%/yr, r = 30 s).
- Pair payoff algebra in both configurations (§12.2's correction of the earlier
  short-Up/long-Down framing).
- λ_bin branch formulas themselves (`√3φ(d)/√r` in-window, `φ(d)/√(τ−2w/3)`
  pre-window) — they are right; §12.1 applied the wrong one.
- Power for α: 95% CI half-width ≈ 0.009 at one day × 8 coins. Sample size is
  not the problem.

Reproduction scripts (MC + estimator simulations) were run ad hoc for this
review; the derivations above are self-contained and each numeric block is the
verbatim output of the corresponding check.
