# σ diagnostics — Polymarket 5-min binaries, BTC

Diagnostics only. Per the HOLD instruction no estimator was built and
`sigma_estimator.py` was **not** written; nothing here is a fitted model.
All numbers are BTC unless stated.

Data: `data/pm_5min/prices/crypto_prices_twap_sixty/` (18.91 h, 63 012 BTC
ticks, 2026-08-19 15:33 → 2026-08-20 10:28 UTC), `data/pm_5min/{markets,
resolutions}.jsonl` (213 usable resolved BTC windows), `data/mm_hf/raw/
bookTicker/BTCUSDT/` (21.3 h, downsampled to 1 s by `recv_ns`).

Scripts: `/tmp/…/scratchpad/{mc_var,diag,diag2..diag9}.py` (scratch — say the
word and I will fold the keepers into a committed diagnostics module).

Notation: `w = 60 s`, `X_t = S60(t)`, `r = T − t`, σ in **bps/√s** (no
annualisation anywhere).

---

## 0. Self-report against the standing constraints

| # | Constraint | Status here |
|---|---|---|
| 1 | Per-symbol, all parameters | **Not built** (HOLD). §4.2 quantifies the defect: v3's single pooled `k` spans **0.884 → 1.896 across the 7 coins (2.14×)**. Evidence supports the constraint. |
| 2 | ≥3 rolling scales, no static σ | **Not built** (HOLD). §3.3 measures which scales carry information (5/15/30/60/120/240 min) — evidence for the design, not a design. |
| 3 | Walk-forward only | No performance claim is made in this doc. Every number is a **descriptive statistic of the whole sample**, labelled as such. Nothing here may be quoted as OOS. |
| 4 | Knowledge time only | Held. All predictors read via `recv_ns` (§1.3). The *target* `X_T` is read by payload time — declared and justified in §2.1. Measured lag p50 **1745 ms**, p90 2233 ms. |
| 5 | Horizon-matched estimation | Held. No √n scaling anywhere. §5.1 shows v3's own estimator is nevertheless biased low and why. |
| 6 | No double-counted variance | **Checked and answered: do not add σ_⊥.** Measured stream noise = 0.093 bps sd = **0.08 % of the r=30 innovation variance**, and it is already inside the trailing estimate (§5.3). |
| 7 | No annualisation | Held — bps/√s and bps throughout. |
| 8 | Honest power | Held. n reported everywhere; CIs are window-bootstrap. **The headline k≈1.12 is not distinguishable from 1.0** (§4.4). One test day exists; no winner is declared. |

---

## 1. The variance laws: derivation and Monte-Carlo

### 1.1 Derivation

For a driftless BM `B` with variance rate σ² per second and
`M_t = (1/w)∫_{t−w}^{t} B_u du`, any linear functional `∫ f(u)B_u du` has
variance `σ²∫G(s)²ds` with `G(s)=∫_s^∞ f(u)du`.

**(a) Unconditional increment** (`f = +1/w` on `[t+h−w, t+h]`, `−1/w` on `[t−w, t]`).
For `h ≤ w`, G is a trapezoid: ramp `a/w` over length `h`, flat `h/w` over
length `w−h`, ramp down over length `h`:

```
Var[M_{t+h} − M_t] = σ²[ 2·h³/(3w²) + (w−h)h²/w² ] = σ²( h²/w − h³/(3w²) )
```

For `h > w` the two averaging windows are disjoint and `G` is ramp/flat/ramp
with flat length `h−w`, giving `σ²(h − w/3)`.

**At h = w = 60:** `w²/w − w³/(3w²) = w − w/3 = 2w/3 = 40`. **The 40σ² claim is
exactly right**, and the "40" is simply `2w/3` seconds.

**(b) Conditional on `F_t`** (the settlement law). For `r > w` the whole
averaging window is in the future and `E[X_T|F_t] = B_t`:
`Var_t[X_T] = σ²[(r−w) + w/3] = σ²(r − 2w/3)`. For `r ≤ w` part of the window
is already fixed: `Var_t[X_T] = σ²∫_t^T((T−s)/w)²ds = σ²r³/(3w²)`. Both match
the stated law.

### 1.2 Monte Carlo (20 000 paths, dt = 0.05 s)

| h | MC var | naive law | ratio | | r | MC var | cond law | ratio |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.405 | 0.405 | 1.000 | | 10 | 0.092 | 0.093 | 0.993 |
| 30 | 12.519 | 12.500 | 1.002 | | 30 | 2.503 | 2.500 | 1.001 |
| 60 | 40.335 | 40.000 | 1.008 | | 60 | 20.115 | 20.000 | 1.006 |
| 120 | 100.491 | 100.000 | 1.005 | | 180 | 137.914 | 140.000 | 0.985 |
| 300 | 281.120 | 280.000 | 1.004 | | 300 | 261.409 | 260.000 | 1.005 |

Both laws verified. Two by-products:

- **Non-overlapping h=w increments are NOT independent**: measured lag-1
  autocorrelation **+0.2508** (theory +1/4), lag-2 +0.002 (theory 0) — the
  filter is exactly MA(1) at this sampling. v3's docstring claim "*non-overlapping
  60 s increments, so no MA(60) bias*" is right about the *variance* and wrong
  about *independence*; the consequence is a real downward bias (§5.1).
- **`E[X_T|F_t] ≠ X_t`.** The correct forecast is the *spot* `B_t` for `r ≥ w`
  (plus a roll-off term for `r < w`). Using `X_t` instead inflates the required
  σ_eff by a factor that is strongly r-dependent:

| r | 270 | 240 | 180 | 120 | 60 | 30 |
|---|---|---|---|---|---|---|
| σ inflation from using X_t | 1.042 | 1.049 | 1.072 | 1.124 | **1.416** | **2.236** |

This single table is most of the answer to §4.

### 1.3 The knowledge-time grid

1 s grid built as "last tick with `recv_ns ≤ s`", invalidating any point whose
last tick is >5 s old. 68 057 points, **1.60 % stale**, tick age p50 487 ms,
p95 1601 ms.

---

## 2. Does the r³ law hold in the data? (the headline)

### 2.1 Construction

For each completed BTC window: `X_t` at **knowledge time** `t0 + t`, `X_T` at
**payload time** `T`. Reading the target by event time is legitimate — it is
the ex-post settlement value, an *outcome*, not an input — and it is the same
convention EXP-M6 used. Sanity check: `Up ⇔ S60(T) ≥ S60(t0)` reproduces
**213/213 = 100 %** of BTC winners.

Because the predictor lags ~1.75 s, the true horizon is `r + 1.75 s`; the
lag-adjusted column is given where it matters.

Tape anchor (non-overlapping 60 s increments over the full sample):
**σ = 1.1323 bps/√s**. Short-horizon fit `V(h)=σ²g(h)+2ν` over h=1…30 s gives
σ = 1.1518 bps/√s and noise sd 0.0930 bps.

### 2.2 Settlement-innovation variance, BTC

`V(r) = Var[log(X_T/X_t)]`, window-bootstrap 95 % CI on the ratios.

| r (s) | n | sd (bps) | kurt | V / **naive** law | 95 % CI | V / **cond** law (v3) | 95 % CI | σ error of v3 |
|---|---|---|---|---|---|---|---|---|
| 270 | 207 | 16.92 | 10.2 | 0.864 | [0.56, 1.25] | 0.939 | [0.61, 1.36] | 0.97× |
| 240 | 209 | 15.00 | 11.8 | 0.770 | [0.51, 1.18] | 0.848 | [0.56, 1.30] | 0.92× |
| 180 | 209 | 13.31 | 12.2 | 0.835 | [0.53, 1.27] | 0.954 | [0.60, 1.45] | 0.98× |
| 120 | 210 | 9.84 | 10.8 | 0.730 | [0.47, 1.08] | 0.912 | [0.59, 1.35] | 0.96× |
| **60** | 209 | 6.92 | 14.2 | 0.903 | [0.53, 1.38] | **1.806** | **[1.05, 2.76]** | **1.34×** |
| **30** | 211 | 4.69 | 24.8 | 1.327 | [0.65, 2.33] | **6.635** | **[3.23, 11.63]** | **2.58×** |

naive law = `σ²(r²/w − r³/3w²)` for r≤w else `σ²(r − w/3)`
cond law = `σ²r³/(3w²)` for r≤w else `σ²(r − 2w/3)`  ← what v3 uses

**Answer: the r³ law does not hold for the object v3 applies it to.** It is a
correct law — for the conditional variance given the *full* information set,
including spot. v3 pairs it with the forecast `X_t`, which is not the
conditional mean, so the matching law is the naive one. In the region where
the two laws are distinguishable (`r ≤ 60`, where they differ by 2×–6.6× in
variance) the data pick the naive law: ratio 0.90 and 1.33 versus 1.81 and
**6.64**. At r = 30 v3's σ_eff is **2.6× too small** — 1.79 bps against a
realised 4.69 bps.

For `r ≥ 120` the two laws differ by <10 % and the data cannot separate them
(every CI covers both). The apparent 0.73–0.86 shortfall against the naive law
there is **not** significant, and the independent non-overlapping tape estimate
(much larger n) says the law is fine:

| h (s) | 10 | 30 | 60 | 120 | 180 | 270 | 300 | 600 |
|---|---|---|---|---|---|---|---|---|
| n (non-overlapping) | 6649 | 2194 | 1098 | 553 | 361 | 244 | 216 | 105 |
| V / naive law | 1.043 | 1.030 | 1.000 | 0.977 | 0.997 | 0.905 | 1.038 | 0.994 |
| 95 % CI | [.95,1.15] | [.87,1.21] | [.79,1.26] | [.72,1.29] | [.69,1.37] | [.58,1.36] | [.65,1.48] | [.51,1.73] |

**The idealised MA-of-BM relation survives contact with the Chainlink
aggregate**, from 10 s to 600 s, within ±5 % at the horizons with real n. The
÷40 round trip is therefore *not* where the error is — but it is also
unnecessary, and §2.3 is the reason to keep the theory anyway.

### 2.3 What the theory buys that the empirical curve does not

Test F. Rebuild the forecast as `E[X_T|F_t]` under the MA model — spot for
`r ≥ w`, spot plus the roll-off integral for `r < w` — using the **Binance mid
path at knowledge time**, level-anchored to Chainlink over a trailing 300 s:

| r | n | sd[X_T − X_t] | sd[X_T − Ê] | shrink | naive-law sd | cond-law sd |
|---|---|---|---|---|---|---|
| 270 | 203 | 16.86 | 16.73 | 0.99 | 17.90 | 17.17 |
| 180 | 205 | 13.35 | 12.59 | 0.94 | 14.32 | 13.40 |
| 120 | 206 | 9.87 | 9.03 | 0.91 | 11.32 | 10.13 |
| 60 | 205 | 6.98 | 6.11 | 0.88 | 7.16 | 5.06 |
| **30** | 208 | **4.72** | **2.60** | **0.55** | 4.00 | 1.79 |

A correct conditional forecast cuts endgame settlement uncertainty **by 45 %
at r = 30**. That is the r³ pinning being real and harvestable — it just needs
the *mean* model fixed, not the variance model. This is a materially larger
prize than anything in the σ blend, and it lives exactly where pickoff risk
lives. (Residual 2.60 vs theoretical 1.79 = spot-proxy error: Binance perp mid
is not Chainlink's aggregate, and the reconstruction is 1 s-granular.)

---

## 3. How much does it move, and how stale is a trailing estimate?

### 3.1 Split sample and rolling spread

| r | sd 1st half | sd 2nd half | ratio | rolling-50 p10 | p50 | p90 | p90/p10 |
|---|---|---|---|---|---|---|---|
| 270 | 15.83 | 17.92 | 1.13 | 10.25 | 15.70 | 21.36 | 2.08× |
| 180 | 12.01 | 14.26 | 1.19 | 7.23 | 12.05 | 17.59 | 2.43× |
| 120 | 9.71 | 9.88 | 1.02 | 5.00 | 9.52 | 12.24 | 2.45× |
| 60 | 7.22 | 6.61 | 0.92 | 2.99 | 6.71 | 8.68 | 2.90× |
| 30 | 4.87 | 4.52 | 0.93 | 2.13 | 4.34 | 5.94 | 2.79× |

**Do not read that 2–3× spread as signal.** With kurtosis 10–25, a 50-sample sd
is enormously noisy. Permutation null (shuffle window order, recompute):

| r | 270 | 240 | 180 | 120 | 60 | 30 |
|---|---|---|---|---|---|---|
| observed p90/p10 | 2.08 | 2.05 | 2.43 | 2.45 | 2.90 | 2.79 |
| null median | 1.56 | 1.62 | 1.64 | 1.58 | 1.79 | 2.12 |
| null 95th pct | 1.92 | 1.88 | 2.02 | 1.95 | 2.29 | **2.67** |

Signal — but **most of the spread is sampling noise**, and at r = 30 it is
marginal (2.79 vs 2.67). A ~50-window rolling estimate of `Var[X_T − X_t]` is
a low-information object. This is the strongest argument against estimating
the settlement variance directly from ~50–150 completed windows.

### 3.2 The tape is a far better instrument

Predicting each window's own forward realised variation (measured from the
tape over `[t0, T]`, n≈30 per window — much lower noise than one innovation):

| trailing lookback | sd of log σ̂ | rank-corr w/ forward RV | rank-corr w/ \|X_T−X_0\| |
|---|---|---|---|
| 5 min | 0.647 | 0.425 | 0.188 |
| 15 min | 0.523 | 0.526 | 0.214 |
| 30 min | 0.480 | 0.548 | 0.250 |
| **60 min** | 0.450 | **0.565** | **0.260** |
| 120 min | 0.418 | 0.561 | 0.260 |
| 240 min | 0.323 | 0.420 | 0.186 |
| *ceiling* (forward RV vs \|X_T−X_0\|) | — | — | *0.551* |

n = 169 windows. Vol clustering is real: rank-corr(trailing 60 min σ,
|innovation|) = **+0.28** at r=270, **+0.23** at r=30 (SE ≈ 0.07). A trailing σ
captures ~47 % of the attainable correlation with |X_T−X_0|.

### 3.3 Which scales earn a place (in-sample R² of log forward RV)

| scales (min) | R² | coefficients |
|---|---|---|
| [15] | 0.295 | 0.61 |
| [60] | 0.323 | 0.74 |
| [240] | 0.148 | 0.70 |
| [15, 60] | 0.348 | 0.29 / 0.48 |
| [5, 15, 60] | **0.352** | 0.07 / 0.23 / 0.48 |
| [5, 15, 60, 240] | 0.352 | 0.07 / 0.23 / 0.48 / 0.00 |
| all six | 0.354 | (signs flip — over-parameterised) |

Evidence for a **3-scale HAR (≈5 / 15 / 60 min)**, weighted toward the slow leg.
The 4 h scale adds **nothing** on 18.9 h of data and should not be assumed in;
30/120 min are redundant with 60. In-sample R², one symbol, n=169 — indicative
of *which* scales, not of achievable skill.

---

## 4. The k ≈ 1.12 diagnosis

**Verdict: `k` is a link/mean-model calibration constant. It is neither an
estimator bias nor a volatility risk premium, and it must not be used to
correct the σ estimator.** Four pieces of evidence.

### 4.1 The trailing estimator is not biased in the way k suggests

Directly: realised innovation variance / tape-implied naive law = 0.73–1.33,
**every 95 % CI covers 1.0** (§2.2). There is no measurable
implied-over-realised premium and no material vol bias to absorb. (There *is* a
small, separate, real bias in v3's estimator — 6 % — but it has the wrong size
and the wrong sign structure to be k; see §5.1.)

### 4.2 Most of k is the wrong variance law, and it is r-dependent

Fitting `k` per horizon on BTC (1255 records, 15/60-min blend). The pooled
value is insensitive to the estimator form: 1.080 with an overlapping trailing
σ̂, 1.081 with v3's exact `trailing_sigma`.

| r | 270 | 240 | 180 | 120 | 60 | 30 | spread |
|---|---|---|---|---|---|---|---|
| k under **cond** law (v3) | 0.538 | 0.724 | 0.837 | 0.877 | 1.300 | **1.923** | 3.6× |
| k under **naive** law | 0.516 | 0.690 | 0.783 | 0.784 | 0.920 | 0.860 | 1.8× |

Switching to the law that matches the forecast **halves the r-dependence** and
lowers mean NLL 0.4373 → 0.4278. A single pooled scalar cannot represent a
3.6× ramp; the pooled k≈1.1 is a compromise that is wrong at both ends.

Same story across symbols — and this is the direct evidence for the per-symbol
constraint:

| coin | btc | eth | sol | xrp | doge | bnb | hype |
|---|---|---|---|---|---|---|---|
| k (cond law) | 1.081 | 1.324 | **1.896** | **0.884** | 1.043 | 1.276 | 0.995 |
| k (naive law) | 0.798 | 0.895 | 1.168 | 0.697 | 0.771 | 0.855 | 0.712 |
| median σ̂ (bps/√s) | 0.79 | 1.14 | 1.01 | 1.12 | 1.01 | 0.69 | 2.04 |

**2.14× spread; v3 forces one number on all seven.** The naive law lowers NLL
for **7 of 7** coins.

### 4.3 The residual is the Gaussian link meeting a leptokurtic centre

The binary likelihood does not identify a variance. `dP/dm` at the money is
`f(0)`, so what the MLE calibrates is the **central density**, whose
Gaussian-equivalent scale is `1/(√(2π)f(0))` — for a fat-tailed law this is far
below the sd:

| r | realised sd | central scale `1/(√2π f₀)` | MLE-optimal σ_eff | central/sd | MLE/sd |
|---|---|---|---|---|---|
| 270 | 16.92 | 14.71 | 9.66 | 0.87 | 0.57 |
| 180 | 13.31 | 11.01 | 11.03 | 0.83 | 0.83 |
| 120 | 9.84 | 8.18 | 6.53 | 0.83 | 0.66 |
| 60 | 6.92 | 4.95 | 4.64 | 0.71 | 0.67 |
| 30 | 4.69 | 3.31 | 2.33 | 0.71 | 0.50 |

Gaussian would give 1.00. The remainder (largest at r=270) is the second
mean-model effect: the TWAP roll-off creates **built-in continuation** —
regressing `X_T−X_0` on `X_t−X_0` gives β = **2.28** [1.02, 3.23] at r=270 and
1.03–1.21 elsewhere, all ≥1. Both effects push σ_eff below the realised sd;
the wrong-law effect pushes it up. **k≈1.12 is the net of three
misspecifications, no two of which are volatility.**

### 4.4 And it is not distinguishable from 1.0 anyway

BTC, window-bootstrap over 213 windows:

| law | fitted k | 95 % CI |
|---|---|---|
| conditional (v3) | 1.042 | **[0.816, 1.250]** |
| naive-MA | 0.772 | [0.592, 0.975] |

The premise "why is k ≈ 1.12" is under-powered on this sample. **Treatment:
fix the law and the link; carry no scale parameter at all until a k is fitted
per symbol and shown to exclude 1.**

---

## 5. Estimator-form findings

### 5.1 v3's `trailing_sigma` is biased low — exactly as MA(1) predicts

`pvariance` subtracts the sample mean from increments with ρ₁ = +0.25, so
`E[pvar] ≈ γ₀(1 − 1.5/n)`.

| lookback | n pts | measured v3 / zero-mean overlapping | theory `√(1−1.5/n)` |
|---|---|---|---|
| 15 min | 15 | **0.9402** | 0.9487 |
| 60 min | 60 | 0.9817 | 0.9874 |

Theory and measurement agree to <1 %. Confirmed independently by MC with the
true σ known:

| estimator (60 min lookback) | bias | sd(log σ̂) |
|---|---|---|
| v3: non-overlapping 60 s, mean-subtracted | **−1.9 %** | 0.0954 |
| non-overlapping 60 s, **zero-mean** | −0.6 % | 0.0946 |
| overlapping 60 s @1 s, zero-mean | −0.6 % | 0.0926 |
| **multiscale h = 10/20/30/60 s, overlapping** | −0.5 % | **0.0819** |

At a 15-min lookback the v3 bias is **−6.7 %** and multiscale pooling cuts
estimator noise by **16 %** (0.2089 → 0.1748). So: drop the mean subtraction,
sample overlapping, and pool horizons through the (now-validated) naive law.

Latent bug, flagged not fixed: the `tk[i] > t − 3000` freshness rule *skips*
stale points and then differences the survivors as if they were 60 s apart,
silently treating 120 s gaps as 60 s. At 1.6 % staleness the net effect is
inside the noise here (measured 0.9402 vs 0.9487 theory), but it will bite on a
thinner symbol or a worse connection.

### 5.2 Jumps: the TWAP smears them, but the tails survive

BV/RV jump share, 5-min segments:

| series | mean | p50 | p90 | >30 % |
|---|---|---|---|---|
| Binance raw mid (the real underlying), 5 s | 0.236 | 0.206 | 0.462 | 31 % |
| Chainlink S60 TWAP, 5 s | 0.001 | 0.000 | 0.000 | 0 % |

A jump in the underlying enters a 60 s TWAP as a 60 s **ramp**, so classical
jump tests on the settlement series find nothing by construction. Jump/diffusion
separation must be done on the underlying feed, not the TWAP.

Yet settlement innovations are strongly fat-tailed (kurtosis 10.2 at r=270,
**24.8** at r=30). Standardising by trailing σ does **not** fix it — kurtosis
*rises* (10.3 → 14.8 at 60 min lookback), consistent with σ̂ estimation noise
swamping any clustering benefit in the tails. Implication: **no σ blend fixes
tail calibration.** That is a link problem (Student-t / empirical CDF), and it
should be decided separately from the blend.

### 5.3 σ_⊥ — do NOT add it (constraint 6)

Fitting `V(h) = σ²g(h) + 2ν` on h = 1…30 s isolates a stream-vs-underlying
noise term of **sd 0.0930 bps = $0.63 on $68 000**. Against the smallest
settlement variance in the grid (r=30, V = 2.20e-7) that is **0.079 %**.

More decisively, `2ν` is *already inside* every trailing estimate — it is
estimated **from** the same increments the blend would use. Adding a σ_⊥ floor
on top would be the fourth instance of this program's double-counting error,
and it would be numerically irrelevant even if it were legitimate.

### 5.4 Time of day

18.91 h of data covers **16 of 24 UTC hours, once each** (n = 10–12 windows per
hour). Hourly forward RV ranges 0.164 → 0.782 bps/√s (4.8×), but that is one
draw per hour and is indistinguishable from ordinary vol clustering.
**A session factor is not identifiable on this sample.** Do not fit one.

---

## 6. TWAP-derived vs Binance-derived σ (BTC)

Synthetic 60 s TWAP of the Binance mid on the identical knowledge-time grid;
65 300 s overlap; median perp-vs-Chainlink basis −2.2 bps.

| h (s) | Chainlink S60 sd | Binance S60 sd | ratio | corr of increments |
|---|---|---|---|---|
| 10 | 1.43 | 1.45 | 1.013 | 0.927 |
| 30 | 4.01 | 4.08 | 1.017 | 0.964 |
| 60 | 7.10 | 7.09 | 0.999 | 0.974 |
| 120 | 11.06 | 11.05 | 0.999 | 0.989 |
| 180 | 14.26 | 14.37 | 1.008 | 0.993 |
| 300 | 18.92 | 18.84 | 0.996 | 0.997 |

Implied point σ: **Chainlink S60 ÷40 = 1.1323**, **Binance S60 ÷40 = 1.1196**
(−1.1 %), Binance **raw** mid ÷60 = 1.1934 (+5.4 %) bps/√s.

**They agree.** Two independent feeds, two independent estimators, within
1.1 %. Three consequences:

1. The Chainlink aggregate behaves like a clean 60 s moving average of the
   market price — the ÷40 round trip is *empirically safe*, not merely assumed.
   (It remains unnecessary: §2 targets `Var[X_T − X_t]` directly.)
2. The +5.4 % excess in the raw Binance mid is bid-ask/microstructure noise
   that the 60 s average removes — corroborating §5.3's tiny ν.
3. Binance is usable as an **independent σ source** and, more valuably, as the
   **spot proxy** that §2.3 shows is worth 45 % of endgame variance.

---

## 7. Summary for the design re-brief

**Settled by evidence:**

1. `Var[X_{t+h}−X_t] = σ²(h²/w − h³/3w²)`, and `h=w=60 ⇒ 40σ²` — derived,
   MC-verified, and confirmed on BTC tape from 10 s to 600 s within ±5 %.
2. **The r³ conditional law is the wrong law for v3's forecast.** v3 is 2.6×
   overconfident at r=30 and 1.34× at r=60. This is the single largest defect
   found, and it is bigger than k.
3. `k` is a link calibration, not a vol parameter; it is 2.14× dispersed across
   coins and not distinguishable from 1.0 on BTC.
4. **Do not add σ_⊥** — measured at 0.08 % of variance and already contained.
5. **Do not fit a session factor** — 16/24 hours, once each.
6. TWAP-derived and Binance-derived σ agree to 1.1 %.

**Open decisions I did not take (HOLD):**

- Estimate `Var[X_T−X_t]` from ~50–150 completed windows (model-free, but §3.1
  shows it is mostly sampling noise) versus from the tape and mapped through
  the now-validated naive law (far more n, small model dependence). The
  evidence leans to the tape; a hybrid that uses the window-based estimate only
  as a *level check* is probably the right compromise.
- Scale set: evidence supports ≈5/15/60 min, slow-weighted; 4 h is worthless
  here.
- The link. Kurtosis 10–25 is not fixable by any σ, and it is concentrated
  exactly at the short horizons that matter for pickoff.
- The mean model. §2.3 says a spot-based conditional forecast is worth more
  than the entire σ blend at r=30. Whoever owns `p̂` should see this.

**What generalises from BTC:** the variance algebra and the MA(1) bias are
structural — they hold for every symbol. The stream noise is ~$0.63 on BTC and
will be relatively larger on thin names (hype's σ̂ is 2.6× BTC's) — re-measure ν
per symbol before concluding it is negligible there. The k spread (0.884–1.896)
and the σ̂ spread (0.69–2.04 bps/√s) say nothing pooled will fit. **Untested
outside BTC: the innovation-variance curve, the scale weights, the jump share,
and the Binance agreement** — the last only exists for the 6 of 7 coins with a
Binance perp (hype has none in `data/mm_hf/raw/bookTicker/`).

**Power statement.** 18.91 h, 213 BTC windows, ~210 observations per horizon,
one underlying path, a single strongly trending session (+5.8 % BTC).
Innovation-variance CIs are ±35 % or wider. Everything above is direction and
mechanism. No number here is walk-forward, and none should be quoted as
performance.
