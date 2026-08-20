# BE-Belief — design plan

Program P-2026-003 (Polymarket 5-min crypto binaries). Planner deliverable:
recommendation + canonical contract. **No implementation in this document.**

Status of inputs: contracts.yaml v12 defines `BeliefProcess` but has **no
`BE-Belief` module record** (open MUST-FIX M11-4, `PM_STRUCT_ITER11_REVIEW.md:228`).
This plan supplies it.

---

## 0. Recommendation in one line

**BE-Belief produces the venue book, recalibrated: a one-parameter,
drift-neutral-by-construction map of the executable top-of-book, fitted
walk-forward and pooled across symbols — with the stream forecast retained
only as a weighted constituent whose weight is *measured* (currently ≈ 0) and
as the declared fallback when the book is Unavailable.**

Formally, option **(c)**, implemented in the shape of **(d)** so that the weight
on our own forecast is a running diagnostic rather than an assumption.

---

## 1. What BE-Belief should produce, and why

### 1.1 The four candidates against the evidence

| candidate | verdict | evidence |
|---|---|---|
| (a) stream-anchored p̂ | **reject as the level** | loses at every horizon through three σ generations: +0.0291 → +0.0277 → +0.0201 Brier vs book. Uniform, not horizon-specific (`EXP_RESULTS_2026-08-20.md:26-39`, commit `1ec5e95`) |
| (b) book as-is | **reject** | best available forecast, but a belief equal to the price generates zero disagreement. All P&L then comes from spread/rebate; BE-Belief becomes a pass-through and the FLB — the one measured edge — is discarded |
| **(c) book recalibrated** | **ADOPT** | the only measured, sign-stable, mechanism-backed disagreement we have. Drift-adjusted gaps run −0.066…−0.015 below the money and +0.012…+0.034 above it |
| (d) blend of (a) and (c) | **adopt as the *shape*, not as a claim** | the blend weight ŵ is E-X2's actual question (`PM_MM_PLAN.md:768-786`). Current evidence says ŵ→0. Keeping the slot costs nothing and makes the claim falsifiable rather than assumed |

### 1.2 The self-defeat check, stated honestly

A belief that tracks the book cannot profit from disagreeing with it. So the
recalibration IS the edge or there is none. Three consequences we accept up front:

1. **The FLB gap is an upper bound on P&L, not P&L.** The measured capture ratio
   is brutal: unconditional gap +1.8 c/share at `t=290s, mid∈[0.95,1.00)` vs
   realised markout **+0.72 c** in the matching state bin (60% haircut); gap
   −9.4 c at `t=30s, mid∈[0.15,0.30)` vs realised markout **+0.25 c**
   (**97% haircut**) — `PM_DEEP_REVIEW.md:177-183`. *Selection destroys most of
   the available edge at every moneyness.*
2. **BE-Belief must produce the UNCONDITIONAL belief `E[Y | book state]`, never
   the fill-conditional `E[Y | book state, FILLED]`.** The fill conditioning is
   BE-FlowAndFills' adverse-selection term. If BE-Belief bakes it in, the two
   modules double-count the same haircut and the program silently under-quotes.
   This is an ownership ruling, and it is the single most important line in this
   plan.
3. **BE-Belief is not the moat.** See §7.

### 1.3 What this makes BE-Belief architecturally

Under (c), `p_hat` is an **algebraic function of an observable price**, not a
model output. That has a welcome side effect: it dissolves the
BE-Uncertainty ↔ BE-Belief cycle recorded at `PM_STRUCT_ITER2_B.md:785` (the
budget needs `G'` to convert X-space variance to p-space; the belief needs σ
from the budget). With `p̂ = G_b(m)` there is **no σ in the level**, so there is
nothing to circle. `BE-Link` (proposed, not adopted) stays unnecessary.

---

## 2. Which price is the input

**Recommendation: the executable pair `(best_bid, best_ask)` at knowledge time,
reconstructed from `price_change` merged with `book` snapshots; the belief's
anchor is a declared `PriceSummary` over that pair, defaulting to `Mid`, with
`Microprice` a gated challenger and `LastTrade` refused.**

### 2.1 The incumbent series is stale, and it is stale by more than the effect

Every experiment in the corpus builds the book from `book` snapshot events only
(`exp_blend_model.py:86`, `book_mid_series` at `:69-100`). But `book` is ~4.1k
events/window against ~149.7k `price_change` deltas
(`PM_MECHANISM_EXPERIMENTS.md:47`) — and **`price_change` carries `best_bid` and
`best_ask` per asset directly**, so top-of-book needs no delta replay at all.

Measured now (`raw/20260820`, 15 window files, 238 asset series, knowledge time):

| quantity | value |
|---|---|
| `book`-snapshot inter-arrival, per token | p10 55 ms · **p50 547 ms · p90 6,240 ms** |
| `\|mid(t) − mid(t−0.5s)\|` | mean 0.0157 · p50 0.0050 · p90 0.0400 |
| `\|mid(t) − mid(t−2s)\|` | mean 0.0306 · p50 0.0150 · p90 0.0800 |
| **`\|mid(t) − mid(t−6s)\|`** | **mean 0.0558 · p50 0.0350 · p90 0.1350** |
| `\|mid(t) − mid(t−15s)\|` | mean 0.0880 · p50 0.0600 · p90 0.2100 |

A mid taken at the p90 of snapshot staleness is wrong by ~5.6 c on average and
13.5 c at p90. **The FLB being fitted is 3–7 c.** The measurement error is
larger than the effect.

And it *moves the answer*. Same 300 windows (day 20260820, grid
{240,180,120,60,30}s into the window), gap = realised − mid:

| mid bucket | dense (book+Δ) | snapshot-only | mean spread dense | mean spread snap |
|---|---|---|---|---|
| 0.1–0.2 | −0.082 | −0.056 | 0.032 | 0.034 |
| 0.2–0.3 | +0.017 | +0.001 | 0.041 | 0.049 |
| 0.3–0.4 | −0.074 | −0.061 | 0.044 | 0.047 |
| **0.4–0.5** | **+0.002** | **−0.046** | 0.044 | 0.057 |
| 0.5–0.6 | −0.056 | −0.031 | 0.039 | 0.055 |
| 0.6–0.7 | +0.105 | +0.116 | 0.042 | 0.052 |
| 0.9–1.0 | +0.021 | +0.030 | 0.017 | 0.019 |

Mean staleness in the dense series is **0.04 s** where the book is active. Note
the snapshot-only series also **inflates the apparent spread** by 1–1.5 c ATM,
because it pairs a bid and an ask observed at different instants.

**Action item, first and non-negotiable:** rebuild the book series from
`price_change.best_bid/best_ask` ∪ `book`. Every book number in the corpus —
including the FLB table and the book's Brier — must be re-derived. The book's
information advantage over our model is currently *understated*.

### 2.2 Mid is a poor summary, and the corpus's spread number is wrong

Measured spread (`raw/20260820`, from `price_change` best_bid/ask, 238 series):
**p25 0.04 · p50 0.05 · p75 0.08**. By moneyness (book snapshots, 60 files):

| `min(mid, 1−mid)` | n | spread p25 / p50 / p75 |
|---|---|---|
| 0.0–0.1 | 1266 | 0.020 / **0.030** / 0.050 |
| 0.1–0.2 | 2075 | 0.040 / **0.060** / 0.090 |
| 0.2–0.3 | 2583 | 0.040 / **0.060** / 0.100 |
| 0.3–0.4 | 2642 | 0.050 / **0.080** / 0.120 |
| 0.4–0.5 | 3518 | 0.040 / **0.070** / 0.130 |

The corpus asserts "2–4 c" everywhere (`PM_MM_PLAN.md:35`,
`PM_MECHANISM_THEORY.md:53`, `PM_VS_MM_THEORY_DIFF.md:22`) — inherited from
Dubach 2026, never measured here; the one previously-measured figure was 1.1–1.8 c
(`PM_DEEP_REVIEW.md:150`). **Both are wrong for the ATM region where the drift-
adjusted FLB is largest.** ATM the book is 6–8 c wide.

That reframes the whole edge. A 3–7 c gap *at the mid* is 0 to −4 c *at the ask*.
So:

> **The map must be fitted on the mid but VALIDATED at the executable prices.**
> The trading question is never "realised vs mid"; it is "realised vs ask"
> (buy) and "bid vs realised" (sell).

Measured, dense series, 300 windows:

| mid bucket | n | bid | mid | ask | realised | realised−bid (sell edge) | ask−realised (buy edge) |
|---|---|---|---|---|---|---|---|
| 0.1–0.2 | 89 | 0.133 | 0.149 | 0.165 | 0.067 | **−0.065** (sell wins) | +0.098 (buy loses) |
| 0.3–0.4 | 150 | 0.332 | 0.354 | 0.376 | 0.280 | **−0.052** (sell wins) | +0.096 (buy loses) |
| 0.6–0.7 | 118 | 0.628 | 0.649 | 0.670 | 0.754 | +0.126 | **−0.084** (buy wins) |
| 0.7–0.8 | 111 | 0.734 | 0.751 | 0.769 | 0.811 | +0.077 | **−0.042** (buy wins) |
| 0.9–1.0 | 248 | 0.958 | 0.967 | 0.975 | 0.988 | +0.030 | **−0.013** (buy wins) |

The FLB *does* clear the half-spread in these cells — before the 60–97%
selection haircut of §1.2. That haircut, not the gap, is the number that decides
deployment, and it is not BE-Belief's to produce.

### 2.3 Microprice: a gated challenger, not the default

No microprice, imbalance or OFI measurement exists anywhere in the program
(`PM_MM_PLAN.md:697`, `PM_VS_MM_THEORY_DIFF.md:66`) — zero data. Three measured
facts argue it will be weak *here*, unlike in a continuous market:

1. Top-of-book size is quantised at the reward-program minimums — measured p10/p50/p90
   of the touch = **4.98 / 29.98 / 50.0 shares** against `orderMinSize=5` and
   `rewardsMinSize=50` (`PM_MM_PLAN.md:677-679`). Touch size is largely
   reward-farming inventory, not informed demand.
2. Depth deliberately sits *outside* the touch: BTC median depth within 1 c of
   mid is **138 shares/side (~$69)** vs **1,290 within 4.5 c**
   (`PM_SKETCH_REVIEW_ITER1_M.md:150-152`). An imbalance computed at the touch is
   computed on the least informative 10% of the book.
3. The tick regime switches: `tick_size_change` 0.01 → 0.001 fires **328 times
   across 130 windows**, never reverting (`PM_MECHANISM_EXPERIMENTS.md:44,348-351`).
   Size-at-the-touch is not comparable across regimes.

So `PriceSummary` is a **registry-open variant** and `Microprice(alpha)` must
beat `Mid` out-of-sample on the same walk-forward protocol before it can be
declared. It is not the default.

### 2.4 last_trade: refused as the anchor

Two independent reasons, both already on the record:

- The trade-price conditioning was **explicitly retracted**: *"trade price is an
  outcome of the fill, not the state a maker chooses to rest in, and a sweep
  prints far from the mid it started at"* (`PM_DEEP_REVIEW.md:120-126`). The
  retracted number (+3.6 c/share at p∈[0.15,0.35)) still headlines
  `PM_MM_PLAN.md:960` — do not carry it into BE-Belief.
- Feed-ordering defect: *"the `price_change` carrying best_bid/ask for a match
  can be emitted **before** the `last_trade_price` for the same match"*
  (`PM_DEEP_REVIEW.md:149-157`). Any last-trade constituent must therefore carry
  a **declared frozen lag** (e.g. mid as of `t_trade − 250 ms`), which is a
  `LastTrade(lag: Duration)` variant, never a bare read.

### 2.5 There is exactly one price — no cross-check exists

Measured now: the Up and Down books are **exact complements**.
`price_change` entries for the two assets of a market mirror algebraically
(observed: asset A bid 0.43 / ask 0.63, asset B bid 0.37 / ask 0.57 ⇒
`bid_A = 1 − ask_B` exactly), and paired `book` snapshots taken < 2 s apart give
`mid_up + mid_down`: **p10 0.9900 · p50 1.0000 · p90 1.0100 · mean 1.0001**
(n = 3,003).

Consequence to state loudly: **the obvious robustness test — "does the Down
token show the same FLB?" — is algebraically vacuous.** There is no redundant
observation of the price, no cross-book arb residual, and no independent
validation of the anchor. Any "confirmation" from the Down side is a tautology.
This kills a test a reviewer will otherwise demand, and it removes a data source
someone will otherwise assume exists.

---

## 3. Functional form

**Recommendation: one-parameter anchored logit map on the direction-folded
coordinate.**

```
  deployed:   logit p̂ = b · logit m            ⇔   p̂ = m^b / (m^b + (1−m)^b)
  estimated:  logit p̂ = a + b · logit m        (a reported, NOT deployed — §4)
```

### 3.1 Why logit-affine and not isotonic

| form | dof | verdict |
|---|---|---|
| **`b` only, anchored** | **1** | **ADOPT.** Rotation about p=0.5; drift-neutral by construction |
| `(a, b)` free | 2 | estimate and report; `a` is the drift channel, deliberately not deployed |
| `b(r)`, 2 knots | 3 | challenger, gated on ≥7 days (§5) |
| isotonic | ~#level-sets | **reject as a deployable at this n**; keep as an overfitting *reference* |
| parametric non-logit (probit/power-of-odds) | 1–2 | equivalent at this precision; logit chosen because it makes drift and FLB orthogonal |

Isotonic is the wrong tool here for a reason that is visible in the data, not
just in theory. The *drift-adjusted* gap profile the programme reports is
−0.066, −0.043, −0.076, −0.034, −0.015, +0.034, +0.012, +0.021 — **not
monotone**. And my own 300-window dense measurement alternates harder:
+0.003, −0.082, +0.017, −0.074, +0.002, −0.056, +0.105, +0.059, +0.032, +0.021.
An isotonic fit on such a profile produces a staircase whose step boundaries are
determined by *which way the noise happened to fall*, and it will pool adjacent
buckets that differ by 8 cents. `PM_DEEP_REVIEW.md:394-403` correctly names
isotonic as the *competitor's* one-hour tool — that is an argument about how
cheap the edge is, not an argument that isotonic is the right estimator at
n = 1,477 windows.

Isotonic also **cannot separate drift from FLB**: it absorbs both into one
unconstrained curve, which is precisely the failure mode the brief warns about.

### 3.2 Why the logit parameterisation is the drift separation

In logit space the two confounds are geometrically distinct:

- **FLB = rotation** about `p = 0.5` → the slope `b`. `b > 1` ⇔ underconfident
  book ⇔ longshots overpriced *and* favourites underpriced, symmetrically.
- **Drift = translation** → the intercept `a`. A +4.5 pp up-drift shifts every
  bucket the same way in logit space and is largest in probability space ATM —
  the hump the brief describes.

They are orthogonal by construction, so the separation is *structural*, not an
ex-post subtraction. This is strictly better than the incumbent method
(`exp_blend_v2.py:100-110`), which subtracts `mean(φ(d)·μ/φ(0))` per bucket. That
adjustment assumes drift enters through *our Gaussian model's* `d`. But we are
recalibrating the **book**, and the book's drift response is φ(d) only if the
book is exactly Gaussian in d — which is the very hypothesis the FLB rejects.
A free intercept absorbs drift with no such assumption.

### 3.3 Calibrating the expected effect size

From the drift-adjusted profile: `m = 0.15 → p ≈ 0.10` gives
`b = logit(0.10)/logit(0.15) = 1.27`; `m = 0.65 → p ≈ 0.70` gives `b = 1.37`.
So the hypothesis is **b ≈ 1.3**, and the null is `b = 1`.

---

## 4. Drift separation — the design

Four layers, in increasing strength.

### 4.1 Structural (always on)
Logit parameterisation, §3.2. `a` = drift, `b` = FLB.

### 4.2 Direction-folding (the centrepiece)

For each `(window, decision time)` define the **longshot coordinate**
`q = min(m, 1−m)` and `y_q = 1` iff the *longshot side* won. Then:

- a pure **FLB** (rotation) predicts `E[y_q | q] < q` **uniformly, regardless of
  whether the longshot happened to be Up or Down**;
- a pure **drift** predicts `E[y_q|q] > q` when the longshot is Up and
  `E[y_q|q] < q` when the longshot is Down — *opposite signs*.

So fit `b` on the folded coordinate (drift-free by construction, one parameter),
and run the **Up-longshot vs Down-longshot split as the falsification test**.
This is the cleanest separation available on this data and it needs no model.

### 4.3 Deployment policy: pin `a = 0`

The deployed map carries no intercept. Justification: `a` is the net drift of a
20-hour rally in which **every coin is up** — measured up-rates btc 0.5375,
eth 0.5667, sol 0.5417, xrp 0.5500, doge 0.5154, bnb 0.5374, hype 0.5551;
pooled 0.5436 (n = 1,641). Deploying `a` is a bet that the rally continues, i.e.
a directional alpha claim — exactly the claim the ŵ→0 result says we cannot make.

Declared as a field-level `NullPin`:
```
  field: BeliefProcess.recalibration.drift_intercept
  assumption: a = 0 (drift-neutral anchor at p = 0.5)
  bias_direction: PESSIMISTIC     # under-states P(Up) in an up-drifting sample
  declared_by: BE-Belief
```
`a` is still *estimated and reported* every fit, because a large stable `a` is
either drift leaking in or a genuine venue-wide Up-token bid, and only calendar
time separates those.

### 4.4 Asymmetry test — the red flag already visible

A pure rotation implies `b_low = b_high` when fitted separately on `m < 0.5` and
`m > 0.5`. The drift-adjusted profile does **not** look like that: the below-the-
money arm (−0.066, −0.043, −0.076, −0.034) is **2–3× the above-the-money arm**
(+0.034, +0.012, +0.021). If `b_low ≠ b_high` survives folding and walk-forward,
then "favourite–longshot bias" is the wrong name for the phenomenon — it is an
asymmetric level effect, which drift *can* produce and a rotation cannot.

**This is the primary falsification test.** Run it before anything else.

### 4.5 A drift control that is knowledge-time legal

Conditioning on the window's own realised drift is look-ahead. The legal
version: stratify by the **prior hour's** realised return of the same coin,
read at `t0` from the settlement stream. If `b` is stable across those strata and
`a` tracks them, the separation works. If `b` moves with the prior-hour drift,
the FLB estimate is drift-contaminated and must be fitted conditionally.

---

## 5. Conditioning on time-in-window `r`

**Recommendation: `r` enters as a REGIME SPLIT and an exclusion, not as a free
slope — because the mechanism changes with `r`, it does not merely scale.**

### 5.1 The tick floor manufactures the favourite side

The book's Brier collapses to 0.0174 at `r = 30 s` — it knows the answer
(`EXP_RESULTS_2026-08-20.md:33`). At that point the *only* thing left to
mis-price is the last cent, and the grid decides it. On a 0.01 grid the best
quotable ask is 0.99, so a contract whose true probability is 0.998 is
mechanically under-priced by 0.8 c. Measured: bucket 0.9–1.0 has mid 0.967,
**ask 0.975**, realised 0.988 — an "edge" of exactly the size of the grid.

This matters enormously for the fit, because **the extreme buckets dominate the
sample**: 1,609 + 1,916 = **3,525 of 8,862 samples (40%)** sit in 0.0–0.1 or
0.9–1.0 (`EXP_RESULTS_2026-08-20.md:45,54`), and in logit space they carry by far
the largest `(logit m)²`, hence most of the Fisher information for `b`. **A
pooled fit is driven by the 40% of the sample where the bias is mechanical, not
behavioural, and where a 2 c edge cannot be quoted against a 1 c tick.**

Rulings:
1. Fit the behavioural `b` on the **core domain** `|logit m| ≤ 3`
   (`m ∈ [0.047, 0.953]`), declared as `Recalibration.domain`.
2. Fit the extreme domain under its **own** parameter and label it
   `tick_floor`, not `flb`. Refit within the 0.001 tick regime as the control:
   if the extreme-bucket bias shrinks by ~10× there, it is the grid.
3. Do **not** deploy the extreme-domain map until (2) returns.

### 5.2 Does `b` depend on `r`?

The apparent `r`-dependence in the corpus is confounded: as `r` falls the mid
distribution migrates to the extremes, so a fixed-bucket comparison across `r`
compares different populations. `b` is scale-free and *is* comparable across `r`,
which is the right diagnostic — and another reason to prefer logit-affine over
isotonic.

Prior: `b` should be largest early (most residual uncertainty, most room for
lottery-ticket demand) and smallest late — while the *tick-floor* bias runs the
other way. So the two mechanisms have opposite `r`-profiles and separating them
is exactly what §5.1's domain split buys.

**Gate:** `b(r)` with 2 knots is deployed only if it beats pooled `b` on
out-of-sample day-clustered log-loss at ≥ 7 days. At 2 days it is not estimable.

---

## 6. Fitting and validation protocol

### 6.1 Hard rules

1. **Walk-forward only.** Fit on days strictly `< d`; score day `d`. Never refit
   within a test day. An in-sample isotonic fit on 1.4 days will look
   spectacular and mean nothing.
2. **Knowledge time only.** All reads via `recv_ns`. Measured total observation
   lag on the settlement stream is **p50 1,700 ms / p95 2,330 ms**, of which 85%
   is PM-side publication delay (`PM_DEEP_REVIEW.md:35-45`). For `r < 1.7 s` we
   have observed *none* of the final segment — the steady state, not a gap.
3. **Admissibility.** Use the canonical four-condition rule
   (`PM_MECHANISM_EXPERIMENTS.md:113-118`), not the weak TWAP-only rule actually
   implemented at `exp_blend_v2.py:45-47` / `exp_blend_v3.py:66-67`. Report the
   excluded fraction and its bias direction with every table; exclusions are
   load-correlated (`1013 slow consumer` disconnects on BTC, which carries 85%
   of notional — `PM_MM_PLAN.md:493-497`), so all numbers are calm-market
   numbers until proven otherwise.
4. **One outcome per window.** The 5–6 grid points per window share a single
   Bernoulli draw. **Every n in the corpus is inflated 5–6×.** Cluster the
   likelihood and every CI on `window`, then on `day`. The correct unit for
   power is *windows*, and above that, *days*.
5. **Effective breadth ≈ 1–2, not 7** (`PM_MM_PLAN.md:747-749`). Seven crypto
   majors on one beta.

### 6.2 The challenger ladder (all scored on the same walk-forward split)

| # | model | expectation |
|---|---|---|
| 0 | raw book (`b=1, a=0`) | the null to beat |
| 1 | **anchored `b`, folded, core domain** | **the candidate** |
| 2 | `(a, b)` free | reported; `a` diagnoses drift |
| 3 | `b_low`, `b_high` separately | §4.4 falsification |
| 4 | `b(r)`, 2 knots | gated to ≥7 days |
| 5 | isotonic | overfitting reference; expected to lose OOS |
| 6 | per-coin `b` (shrunk) | gated to ≥30 days |
| 7 | blend: `logit p̂ = (1−w)·b·logit m + w·logit p̂_stream` | ŵ expected ≈ 0 |
| 8 | `Microprice(α)` anchor vs `Mid` | §2.3 gate |

Metrics: paired Δlog-loss and ΔBrier vs model 0, day-clustered; plus the
executable re-expression of §2.2 (`realised − bid`, `ask − realised`) per cell.
Log-loss is primary — it is the fit's own objective and it penalises the
overconfidence a too-large `b̂` would create.

### 6.3 What pools and what does not

| parameter | scope | why |
|---|---|---|
| `b` (FLB slope) | **pooled across all 7 coins** | the FLB is a property of the venue's participant mix (retail lottery-ticket demand + reward-farming makers), not of the coin. There is no coin-specific mechanism. With effective breadth 1–2, seven per-coin fits are seven noisy readings of one number |
| `a` (drift) | per-coin, **diagnostic only** | per-coin up-rates span 0.5154–0.5667 — that spread is drift, and it is exactly what we refuse to deploy |
| domain / tick regime | per-instrument | `SP-Venue.tick_grid` is state-dependent; the 0.01↔0.001 switch is per market |
| spread / anchor | per-instrument, **observed not fitted** | different coins have different spread distributions; handled by reading the live pair, not by splitting `b` |

Per-coin `b` enters as a *shrinkage challenger* (model 6) at ≥30 days, never as
the default.

### 6.4 Refit cadence and artifact discipline

Daily refit at the UTC day boundary on all admissible history; the map is an
immutable artifact with `artifact_id`, `fit_data_through`, `fit_n_windows`,
`fit_n_days`, and the parameter covariance. `ParamValue` already carries
`fit_data_through` and `artifact_id` (`contracts.yaml:251-259`) — **the fitted
`a`/`b` live in `SP-Params` keyed by `(ParamId, ScopeKey)`, not restated on
`Recalibration`** (R-SSOT; same precedent as `loss_limit` and `rewards_band`).
`Recalibration` holds *references*.

Fail-loud conditions → `Unavailable`:
- `fit_n_days < 2` or `fit_n_windows < 300` → `Unavailable(WARMUP)`
- book absent or `staleness > threshold` → `Unavailable(STALE_BOOK)`, cause
  propagated
- `m` outside `Recalibration.domain` → the identity map, flagged, never
  extrapolated
- `source_events = SNAPSHOT_ONLY` → belief is emitted but **marked**, because
  §2.1 shows it is a materially different object

---

## 7. Competitor baseline

The corpus's position is that this bias is *"public, monotone and capturable by
an isotonic map that any competitor can fit in an hour"* and that this is
bearish (`PM_DEEP_REVIEW.md:694-698`, `:374-381`). I agree with the fact and
partly disagree with the inference.

**Where it is right.** The recalibration is not a moat. It must not be the
justification for the programme, and it must not be the thing we spend budget
making fancier. Making the map more flexible buys nothing a competitor has not
already priced — which is a *second*, independent argument for the 1-parameter
form of §3.

**Where the inference needs care.** "Anyone can fit it" does not imply "it is
gone"; it implies it is competed down to the point where the marginal harvester
breaks even *after their costs*. The measured `b` is therefore already the
post-competition residual. Two operational consequences:

1. **Persistence is an empirical question with a decisive test: is `b_t`
   declining in calendar time?** Build that monitor from day 1. A positive but
   *decaying* `b` invalidates deployment even while the level is positive.
2. **It changes what BE-Belief is FOR.** BE-Belief becomes a *quoting-placement
   input* — it tells DE where resting is systematically wrong-sided — not an
   alpha source. The moat, if the programme has one, must live in
   BE-FlowAndFills (surviving the 60–97% selection haircut) or in the incentive
   term. BE-Belief's honest self-description: *"I am a commodity input. I remove
   a known bias so downstream sizing is not systematically wrong. I am not the
   edge."*

It also reinstates the guard that was demoted (`PM_MECHANISM_EXPERIMENTS.md:715`,
originally `PM_SKETCH_REVIEW_ITER1_S.md:144-146`): **the walk-forward
recalibrated book is now a mandatory baseline for any future alpha claim.** Any
model that beats the raw mid but not the recalibrated mid has demonstrated
public recalibration, not information. Under this plan that baseline is not a
deferred diagnostic — it *is* BE-Belief, so the guard is enforced by construction.

---

## 8. Interaction with BE-Uncertainty

Stated as a contract, so the σ planner can design against it.

**Under the recommendation, BE-Belief needs σ for the LEVEL in exactly one
place: the fallback.** Everything else is dynamics.

| what BE-Belief needs from BE-Uncertainty | why | if absent |
|---|---|---|
| **nothing for `p_hat` on the main path** | `p̂ = G_b(m)` is algebraic in an observed price. This is the cycle-breaker of §1.3 | n/a |
| `path_law.var_of_increment(h)` | carried through to the sniping / adverse-selection and inventory-horizon terms in DE and BE-FlowAndFills. BE-Belief **carries, does not compute** | `NullPin` at field granularity (NF-2, `PM_STRUCT_ITER2_A.md:254-261`) |
| `jump_tail(m, h)` | same — carried. Note the open defect: `contracts.yaml:385` still types it `float` with two unnamed arguments and no declared unit (`PM_STRUCT_ITER2_B.md:664-666`) | `NullPin`, bias OPTIMISTIC (a missing tail reads as zero) |
| σ_eff **only** for the fallback forecast | when the book is `Unavailable` — and it happens: measured 30 s mid staleness in early-window cells where no book has formed yet — `w` goes 0 → 1 and the stream forecast is the only level available | `Unavailable`, and the belief refuses rather than guessing |

So: **if BE-Uncertainty concludes σ is needed only for DYNAMICS, that is fully
consistent with this plan and no rework follows.** The blend weight `w` is not a
free tuning knob; it is 0 on the main path (measured, not assumed) and 1 in the
declared `FallbackPolicy`.

**Ownership boundary (R-SSOT).** BE-Belief owns the **parameter uncertainty of
its own fit** — `Var(b̂)` propagated to `Var(p̂)` by the delta method. That is not
a variance component of the *outcome* and must not be registered with
BE-Uncertainty's `VarianceGroup`. Conversely BE-Belief owns no σ_eff, no
`w_hat`, no `VarianceComponent`. Per `PM_STRUCT_ITER1_B.md:470`, if BE-Belief's
anchor choice creates outcome variance (e.g. a microprice anchor's estimator
noise), **BE-Belief registers that component into the budget BE-Uncertainty
owns** — data flows up, ownership stays put.

`Var(p̂)` has one documented use and only one:
`w = var_p / (var_p + var_book)` (`PM_STRUCT_ITER2_B.md:660-663`). It is exposed
as an aggregate `estimator_var`, with per-source marginals on `constituents`.

**BE may not read EV** — σ_book is an EV object (`PM_STRUCT_ITER2_B.md:786`).
BE-Belief's `consumes` list contains no `EV-*`.

---

## 9. Power: what is claimable now, at 7 days, at 30 days

Current inventory (measured on disk now): `markets.jsonl` 1,669 rows;
`resolutions.jsonl` 1,641 final, zero duplicate slugs; `window_start` span
**2026-08-19 14:25 → 2026-08-20 10:40 UTC ≈ 20.25 h**; 2 UTC dates; 1,744 raw
window files; 7 coins. (The corpus's "~1.4 days" label is not an elapsed time —
`PM_DEEP_REVIEW.md:11` gives 13 h 25 m and `PM_QUANT_REVIEW.md:899` says 13.5 h.
Read "1.4 days" as "spans two UTC dates, one walk-forward test day".)

Effective sample, after the §6.1 corrections: **~1,477 windows, one outcome
each**, ~1.5 effective coin-factors, **one walk-forward test day = one cluster**.

| | claimable | not claimable |
|---|---|---|
| **Now (2 days, 1 test day)** | sign and rough magnitude of pooled `b̂` on the train day; a point OOS Δlog-loss / ΔBrier vs raw book on the single test day; the §4.4 asymmetry check as a *flag*; the §2.1 stale-vs-dense rebuild, which is a data-quality fix and needs no power at all | **any CI** (one cluster); per-coin `b`; `b(r)`; drift/FLB separation with confidence; anything about P&L; any statement that survives a down-drift period |
| **7 days (~10.3k windows, 6 test days)** | day-clustered paired Δ vs raw book with a real (wide, 5 df) interval; the folded/unfolded asymmetry test with a CI; `a`-stability across days; the first read of `b_t` drift; the core-vs-extreme domain split | per-coin `b`; `b(r)` with 3+ knots; a capture-ratio CI |
| **30 days (~44k windows, 29 test days)** | usable day-clustered CIs; `b(r)` with 2–3 knots; per-coin `b` as a shrinkage challenger; a `b_t` trend test (§7); **and the actual gate — a markout-based capture-ratio estimate with a CI** | nothing about a regime we have not observed |

**The falsification that only calendar time can deliver:** 30 days is the
shortest window likely to contain a *down*-drift period. Until one is observed,
the drift/FLB separation rests on the structural argument (§4.1–4.2), not on
evidence. State it that way in every result table.

**And the one thing more days does NOT fix:** the capture ratio. The 60–97%
haircut is a selection effect, not a sample-size effect. No amount of data makes
the unconditional gap harvestable.

---

## 10. Ways this design could be wrong

Ranked by how much they would change the answer.

1. **The FLB is a tick-floor artifact.** 40% of the sample sits where the grid
   mechanically caps the quote, and those cells carry most of the Fisher
   information for `b`. Measured: bucket 0.9–1.0 ask 0.975 vs realised 0.988 —
   an edge the size of the grid. *Test:* refit on `|logit m| ≤ 3`; refit inside
   the 0.001 tick regime; if the extreme bias shrinks ~10×, it is the grid.
2. **It is a translation, not a rotation** — i.e. drift wearing an FLB costume.
   The drift-adjusted below-the-money arm is 2–3× the above-the-money arm.
   *Test:* §4.4 `b_low` vs `b_high`, and §4.2's Up-longshot / Down-longshot split.
3. **It is the rally.** Two days, every coin up, pooled 0.5436. *Mitigation:*
   pin `a = 0`. *Test:* requires an observed down-drift period — 30 days.
4. **Everything was measured on a stale book.** Snapshot-only series, p90 6.2 s
   stale, a 6 s-stale mid wrong by 5.6 c mean / 13.5 c p90 — larger than the
   effect. My dense re-measurement moves bucket 0.4–0.5 from −0.046 to +0.002.
   *Test:* the §2.1 rebuild. Every corpus number must be re-derived.
5. **The mid is not tradeable.** ATM spread 6–8 c, not the 2–4 c the corpus
   assumes. A 3–7 c mid-edge is 0 to −4 c at the ask. *Test:* §2.2's
   `realised−bid` / `ask−realised` re-expression, mandatory in every table.
6. **Selection eats it.** 60–97% measured haircut. BE-Belief cannot fix this and
   must not claim the gap as P&L (§1.2). *Owner:* BE-FlowAndFills.
7. **Pseudo-replication.** 5–6 grid points per window share one Bernoulli
   outcome; every n in the corpus is inflated 5–6×. Uncorrected, the whole
   result looks 2.3× more significant than it is.
8. **Late-window calibration is near-tautological.** At `r = 30 s` the book's
   Brier is 0.0174; "calibration" there is a tick-rounding residual, not a
   belief. *Mitigation:* always report `r`-stratified.
9. **`b` is being competed away.** A positive but decaying `b_t` invalidates
   deployment. *Test:* the §7 monitor, from day 1.
10. **"ŵ → 0" is narrower than it sounds.** It says *our settlement-stream
    forecast at 1.7 s lag* has no edge — 85% of that lag is PM-side publication
    delay we cannot remove. A Binance-lead forecast has never been tested
    (`PM_DEEP_REVIEW.md:648-663`: HYPE has no Binance leg and `data/mm_hf/vision/`
    is aggTrades-only). Do not let `ŵ→0` be quoted as "no alpha exists".
11. **No independent check on the anchor.** `mid_up + mid_down = 1.0000`; the
    Down-token confirmation is algebraically vacuous (§2.5). There is exactly one
    price and no redundancy.
12. **Pooling `b` across coins could be wrong** if the FLB is really a
    liquidity/spread effect rather than a participant-mix effect — spread differs
    materially by coin. *Test:* model 6 at 30 days; earlier, check whether adding
    spread as a covariate absorbs `b`.
13. **The map could be right and useless.** If `b̂ ≈ 1.05` after every correction
    above, the honest output is the raw book and BE-Belief is a pass-through with
    a monitor. That is an acceptable outcome and the contract must permit it
    (`RecalibrationForm.Identity`).

---

## 11. Canonical contract — `BE-Belief`

Schema style of `live/pm_research/contracts/contracts.yaml` v12. **Not applied to
that file; this block is the proposal.** All additions; per
`contracts/migrations.yaml`, adding types/fields/modules needs no migration
record, so this is a pure `version: 12 → 13` bump. Naming checked against R-SSOT:
no name here is declared elsewhere in v12, and none of `σ_eff`, `w_hat`,
`VarianceComponent`, `K`, `E[X_T]`, `w_declared` or `ScenarioLossLimit` is
touched.

```yaml
# ---------------------------------------------------------------- types (new)
types:

  PriceSummary:
    kind: open_protocol
    registry: PluginRegistry
    protocol:
      summarise: '(TopOfBook) -> float | Unavailable'
      config_schema: JsonSchema
    builtin_ids:
    - mid
    - microprice
    - bid
    - ask
    - last_trade
    notes: 'the scalar the belief is anchored to. `mid` is the DEFAULT. `microprice` is a
      GATED challenger: no imbalance/OFI measurement exists in-programme, touch size is
      quantised at the reward minima (p10/p50/p90 = 4.98/29.98/50.0 shares) and depth sits
      outside the touch (138 shares within 1c vs 1290 within 4.5c), so it must beat `mid`
      out-of-sample before it may be declared. `last_trade` REQUIRES a declared frozen lag:
      the price_change carrying best_bid/ask for a match can be emitted BEFORE the
      last_trade_price for the same match.

      '

  TopOfBook:
    fields:
      best_bid: float
      best_ask: float
      bid_size: float
      ask_size: float
      tick: float
      complement_of: TokenId?
      source_events: enum:SNAPSHOT_ONLY|SNAPSHOT_PLUS_DELTA
      t_known: Timestamp
      provenance: Provenance
    notes: 'OWNER IS DA-Normalize, not BE-Belief (R-SSOT); listed here because BE-Belief is
      its first declared consumer and the type does not yet exist. MUST be built from
      `price_change.best_bid/best_ask` UNIONED with `book` snapshots: snapshots alone are
      ~4.1k events/window vs ~149.7k deltas, p90 inter-arrival 6.24s, and a 6s-stale mid is
      wrong by mean 5.6c / p90 13.5c -- larger than the 3-7c effect being fitted.
      `source_events: SNAPSHOT_ONLY` is a materially different object and MUST be marked.
      The two tokens of a market are exact complements (mid_up + mid_down = 1.0000 median,
      mean 1.0001), so `complement_of` carries no independent information and MUST NOT be
      used as a cross-check.

      '

  RecalibrationForm:
    variants:
    - Identity
    - 'LogitAnchored(b: ParamId, anchor: float)'
    - 'LogitAffine(a: ParamId, b: ParamId)'
    - 'PiecewiseLogitAnchored(b_by_r: ParamId, knots: list[float])'
    - 'Isotonic(artifact: ImmutableId)'
    notes: 'DEFAULT is LogitAnchored with anchor = 0.5: logit p_hat = b * logit m, which is a
      ROTATION about 0.5 and is therefore drift-neutral BY CONSTRUCTION. LogitAffine adds the
      drift channel `a` as a TRANSLATION -- estimated and reported, NOT deployed (see the
      NullPin on drift_intercept). Isotonic is a REFERENCE challenger for measuring
      overfitting, not a deployable at n < 10 days: the measured gap profile is non-monotone,
      so isotonic step boundaries are set by noise, and isotonic cannot separate rotation
      from translation. Identity is a legitimate deployable outcome if b_hat is not
      distinguishable from 1.

      '

  Recalibration:
    fields:
      form: RecalibrationForm
      input: PluginRef
      fold: enum:NONE|LONGSHOT_FOLDED
      domain: '(float, float)'
      on_out_of_domain: enum:IDENTITY_FLAGGED|REFUSE
      scope: ScopeKey
      drift_intercept: float | NullPin
      fit_data_through: Timestamp
      fit_n_windows: int
      fit_n_days: int
      fit_cluster_unit: enum:WINDOW|DAY
      param_cov: dict[str, float]
      artifact_id: ImmutableId
      provenance: Provenance
    notes: 'the fitted VALUES of a and b live in SP-Params keyed by (ParamId, ScopeKey) with
      their own fit_data_through/artifact_id; this record holds REFERENCES only (R-SSOT,
      same precedent as loss_limit and rewards_band). `fold: LONGSHOT_FOLDED` fits on
      q = min(m, 1-m) with y = 1 iff the longshot side won, which is drift-free by
      construction. `domain` defaults to |logit m| <= 3: 40% of samples sit outside it, where
      the tick grid (0.01, switching to 0.001) mechanically caps the quote and manufactures
      an apparent favourite bias the size of the grid. `fit_cluster_unit` exists because the
      5-6 decision times inside one window share ONE Bernoulli outcome; an unclustered fit
      inflates n by 5-6x.

      '

  BeliefWarmupPolicy:
    fields:
      min_fit_days: int
      min_fit_windows: int
      max_book_staleness: Duration
      on_violation: UnavailableAction
    notes: 'fail-loud thresholds for the belief. Violation yields
      Unavailable(WARMUP|STALE_BOOK) with `cause` propagated, never a silently-degraded
      p_hat.

      '

# ------------------------------------------------------- types (extended)
# additive fields on the existing BE-Belief-owned type; no migration record needed
  BeliefProcess:
    fields:
      p_hat: float
      p_raw: float
      anchor: TopOfBook
      recalibration: Recalibration | Unavailable
      estimator_var: float | NullPin | Unavailable
      link: LinkFunction
      jump_tail: float | NullPin | Unavailable
      path_law: PathLaw | NullPin | Unavailable
      constituents: dict[str, Constituent]
      staleness: Duration
    notes: 'p_hat is the UNCONDITIONAL belief E[Y | book state]. It is NEVER the
      fill-conditional E[Y | book state, FILLED] -- that conditioning is BE-FlowAndFills''
      adverse-selection term, and the measured haircut from unconditional gap to realised
      markout is 60-97%. Baking it in here double-counts. `p_raw` is the un-recalibrated
      anchor, exposed so EV and DE can see the size of the disagreement without recomputing
      it. `anchor` is carried so consumers size against the EXECUTABLE pair: the ATM book is
      6-8c wide (measured p50 spread 0.05, 0.07-0.08 ATM), so a 3-7c edge at the mid is
      0 to -4c at the ask, and no consumer may treat p_hat - mid as tradeable.
      `estimator_var` is Var(p_hat) from the delta method on Var(b_hat) -- the uncertainty of
      BE-Belief''s OWN FIT, not of the outcome; it is NOT a VarianceComponent and is NOT
      registered with BE-Uncertainty''s VarianceGroup. Its single documented use is
      w = var_p / (var_p + var_book). `constituents` is EV-READABLE ONLY; DE MUST NOT branch
      on it. The stream forecast appears as constituents["stream"] with its measured weight
      w (currently ~0); w -> 1 only under the declared FallbackPolicy when the book is
      Unavailable.

      '

# --------------------------------------------------------------- module (new)
modules:

  BE-Belief:
    consumes:
    - DA-Normalize
    - BE-Target
    - BE-Uncertainty
    - SP-Params
    - SP-Instrument
    - SP-Venue
    produces: Known[BeliefProcess] | Unavailable
    requires:
      recalibration: Recalibration
      price_summary: PluginRef
      warmup: BeliefWarmupPolicy
      fallback: FallbackPolicy
      link: LinkFunction
    ports:
    - state_view
    - rng
    - artifact_resolver
    - telemetry_out
    notes: 'produces the venue book RECALIBRATED, not our own forecast: measured
      walk-forward, our stream-anchored p_hat loses to the book at every horizon through
      three sigma generations (+0.0291 / +0.0277 / +0.0201 Brier), so w_hat -> 0 and the
      recalibration IS the edge or there is none. p_hat is ALGEBRAIC in an observed price, so
      the LEVEL needs no sigma -- this dissolves the BE-Uncertainty <-> BE-Belief cycle and
      BE-Link is not needed. sigma enters the level in exactly ONE place: the FallbackPolicy,
      when the book is Unavailable. path_law and jump_tail are CARRIED from BE-Uncertainty,
      never computed here. Consumes no EV-* (sigma_book is an EV object; BE may not read EV)
      and takes NO venue port. t_known is derived per R-KNOW as MAX over live inputs
      (anchor.t_known, BE-Target, BE-Uncertainty); Recalibration.fit_data_through is a
      property of the artifact and does NOT enter t_known. Any future alpha claim must beat
      this module as its baseline, not the raw mid -- a model that beats the raw mid but not
      the recalibrated mid has demonstrated public recalibration, not information.

      '

# ------------------------------------------------------------ null semantics
# declared by BE-Belief, per R-NULL (field granularity, NF-2)
  _null_pins:
    drift_intercept:
      field: BeliefProcess.recalibration.drift_intercept
      assumption: a = 0; the map is anchored at p = 0.5 and carries no level shift
      bias_direction: PESSIMISTIC
      declared_by: BE-Belief
    jump_tail:
      field: BeliefProcess.jump_tail
      assumption: no jump component supplied by the active BE-Uncertainty impl
      bias_direction: OPTIMISTIC
      declared_by: BE-Belief
    path_law:
      field: BeliefProcess.path_law
      assumption: no path law supplied; horizon-variance consumers must refuse
      bias_direction: OPTIMISTIC
      declared_by: BE-Belief
```

**Conformance notes for whoever applies this to `contracts.yaml`:**

- `contract_check.py` recognises exactly four module keys — `produces`,
  `consumes`, `requires`, `ports`. `notes:` on a *module* is not flattened into
  the diff, so a notes-only change is invisible; do not encode a decision there
  that has to be enforced.
- The `_null_pins` block above is **not** a recognised key. It is written here so
  the pins are on the record; they must land either as `ModuleManifest.null_semantics`
  (defined at `contracts.yaml:308-316`, currently unused by every module — this
  would be its first user) or as a new `checks:` entry under `R-NULL`.
- `consumes` entries are module references, which the producer-closure check
  skips (`contract_check.py:130-131`). `DA-Normalize` and `BE-Target` /
  `BE-Uncertainty` have no module records yet; naming them here is safe but they
  remain open MUST-FIX M11-4 items.
- `TopOfBook` is declared here only because it does not exist. **Its owner is
  `DA-Normalize`.** Moving it later is a rename, which *does* need a
  `migrations.yaml` record.
- `PriceSummary` is an `open_protocol` + `PluginRegistry` per R-OPEN, so the
  microprice challenger is a plug-in registration rather than a schema edit.
- Bump `contracts.yaml: version: 13` and match it in `PM_ARCHITECTURE.md:3-4`
  and the commit label.

---

## 12. Sequenced work (for the implementer, not done here)

| # | step | gate | days of data |
|---|---|---|---|
| 1 | rebuild `TopOfBook` from `price_change` ∪ `book`; re-derive every book number | dense vs snapshot ΔBrier reported; no gate | 2 |
| 2 | re-express all FLB tables at bid/ask, `r`-stratified, window-clustered | mandatory format change | 2 |
| 3 | fit model 1 (folded, core domain), walk-forward; report models 0–3, 5 | `b̂ > 1` point estimate; §4.4 asymmetry flag | 2 |
| 4 | tick-regime control (0.001 subset) on the extreme domain | extreme bias shrinks ≥5× ⇒ it is the grid | 2–7 |
| 5 | day-clustered CI on Δlog-loss vs raw book | CI excludes 0 | **7** |
| 6 | `b(r)` and per-coin challengers; `b_t` decay monitor | beats pooled OOS | **30** |
| 7 | capture-ratio markout — **the actual deployment gate** | owned by BE-FlowAndFills | **30** |
