# Signal-diversity loop — charter

Opened 2026-08-06, after the cost/turnover loop (`COST_TURNOVER_LOOP.md`) closed with nothing adopted and
after a literature review + structural measurement reframed the problem.

## Why this loop exists (what the measurement established)

The binding constraint is **signal diversity, not risk modelling or construction**.

- **The risk-model idea is already dead — measured, not argued.** Mean pairwise cross-sectional correlation:
  raw returns +0.594 (n_eff 1.7 of 175), BTC-beta residual +0.392 (n_eff 2.5), **our actual training target
  `xs_z(alpha_A)` +0.0046 (n_eff 97.4)**. Adding K=3/5/10 PCA factors on top makes it *worse*
  (+0.0074/+0.0061/+0.0049) — rolling factor estimation error re-injects common variation. The deployed book
  regressed on PC1-5 has **R² = 4.1%**, Sharpe +1.55 → +1.52 with factors removed. Cross-sectional
  standardization already does the neutralization a Barra-style model would. **Do not build one.**
  (`live/factor_structure_probe.py`)
- **What is NOT diverse is the signal.** Three independent measurements this cycle: the full 14-feature model
  ≈ its best single feature (`v0_feature_ablation.py`); a plain slow low-vol sort out-ranks the whole
  175-model stack 2-4× on IC (`cl_iter2_factor.py`); the OI/ADV lead turned out to be that same factor again
  (corr −0.54/−0.59 with vol rank, residualized IC spans 0 in OOS). Realized IR ~2 at IC ~0.025 implies
  effective breadth ~6,400/yr against a nominal ~212,000.

## Literature calibration (net-of-cost Sharpe, published)

| work | net Sharpe |
|---|---|
| Attention Factors, ICAIF 2025 (current SOTA, US equities) | **2.3** (gross >4) |
| Deep Learning Statistical Arbitrage, Mgmt Sci | gross >4 |
| Crypto ML long-short, LSTM/GRU ensemble | 3.1-3.2 |
| Cointegration BTC-ETH | 2.23 |

Published net clusters **2-3.2**. Our held-out is +1.55 gross / ~+1.2 passive-net — about one Sharpe below
academic SOTA, not five. For uncorrelated sleeves IR = r·√N, so a 6-Sharpe book at r=2.3 needs **N ≈ 7
uncorrelated sleeves**. This loop is about raising r on one sleeve and finding genuinely uncorrelated ones.

## Discipline (inherited, non-negotiable)

Both eras; day-clustered CI for per-bar IC, 7d-block for Sharpe; paired deltas with CI on the DELTA;
pre-registered gate + falsifier BEFORE running; chronological hard split (select 2023-06→2024-12, hold out
2025-01→2026-06) for any level/deployability claim; **rank-IC parity does not imply book parity** — anything
that passes at IC level must then convert at book level net of cost. Four leads died at that step last loop.

## Directions

| # | direction | status |
|---|---|---|
| S1 | **Literature-backed characteristics we don't have**: turnover volatility, bid-ask spread (Corwin-Schultz), Amihud illiquidity done right, liquidity volatility, past alpha, size. The 2026 IRFA crypto study finds 2-3 factors absorb all alphas, led by turnover vol / spread / on-chain; the high-dimensional ML study names past alpha, illiquidity, momentum. Our 14 features contain none of the liquidity ones. | iter 1 — running |
| S2 | **On-chain orthogonal data** (new-address-to-price, active addresses) — the one literature-backed variable with a genuinely different economic root. Needs external data. | open |
| S3 | **Train on the trading objective, not MSE** — end-to-end portfolio objective with a turnover penalty. The SOTA paper attributes its net-Sharpe gain to joint learning of factors + trading policy. Attacks our measured binding constraint directly. | open |
| S4 | **Temporal structure** — our model sees 14 contemporaneous features and does no sequence modelling of the residual path. SOTA is a convolutional/attention model over residual time series. Partially explored (`resid_rev`). | open |
| S5 | **Sleeve count / correlation** — validate whether the candidate sleeves (XS reversal, intermediate momentum, carry) are actually uncorrelated, and what IR the combination supports. | open |

Closed upstream, do NOT re-open: V0 feature transforms, Ridge internals, convex sizing, pooled coef
shrinkage, L2/order-book alpha, pump→dump, dispersion/vol-management/PPP, the persistent-tilt book, the
low-vol factor sort, OI-universe, OI/ADV, multi-factor risk model (killed above).

---

## Iteration 1 — S1: the liquidity characteristics the crypto literature says dominate

**H1.** Our 14 features are all price/vol transforms. The crypto cross-section literature consistently ranks
*liquidity* characteristics — turnover volatility, spread, illiquidity — above them, and none are in our set.
Adding them raises book-level rank-IC in both eras.

**Features built** (`live/sd_features.py`, all on the 5m grid then `.shift(1)` and sampled to 4h, matching the
existing panel convention exactly):

| feature | definition | literature |
|---|---|---|
| `turnover_vol_7d` | 7d std of the daily log-change in daily dollar volume | IRFA 2026 top factor |
| `spread_cs_7d` | Corwin-Schultz high-low spread estimator, 7d mean, floored at 0 | IRFA 2026 top factor |
| `amihud_7d` | log mean of \|5m return\| / 5m dollar volume over 7d | "illiquidity" (dominant predictor) |
| `illiq_vol_7d` | 7d std of the daily Amihud measure | "volatility of liquidity" |
| `past_alpha_7d` | trailing 7d sum of the 4h BTC-residual alpha | "past alpha" (dominant predictor) |
| `log_dvol_7d` | log trailing-7d mean daily dollar volume | size |

**Assumptions to validate**
- **A1 validity gate.** Baseline V0_LEAN must reproduce rank-IC ≈ +0.030 RECENT / +0.021 OOS, or the harness
  is wrong and nothing else counts.
- **A2 not a vol restatement.** Every candidate that adds must ALSO be checked residualized on the vol rank —
  the OI/ADV post-mortem showed a "new" signal can be the low-vol factor in a new hat.
- **A3 no look-ahead.** All features `.shift(1)` on the 5m grid before 4h sampling; `past_alpha_7d` shifted a
  full 4h bar since the alpha label is forward-looking.

**Pre-registered gates**
- **G1.** Paired Δ rank-IC vs the V0_LEAN baseline, day-clustered CI excludes 0 in **BOTH** eras.
- **G2.** Any G1 survivor must still add once residualized on vol rank (A2).
- **G3.** Survivors must convert: held-out net@10k on top40/band improves with paired block CI > 0.
- **Falsifier.** No feature clears G1 in both eras → S1 is a null, the crypto liquidity literature does not
  transfer to a 4h BTC-residual book, and the loop moves to S2.

### Result — **G1 FAIL for all six. S1 is a NULL.** (`live/sd_iter1_chars.py`, `live/state/sd_iter1.log`)

Validity gate passed: baseline V0_LEAN reproduces +0.0302 RECENT / +0.0210 OOS.

**A2 — standalone, before the model.** Three of six carry real both-era information, but almost all of it is
the known vol factor:

| characteristic | RECENT own / vol-resid | OOS own / vol-resid | corr w/ vol rank |
|---|---|---|---|
| `past_alpha_7d` | −0.0233 / **−0.0145 SIG** | −0.0226 / **−0.0160 SIG** | +0.19 / +0.14 |
| `spread_cs_7d` | −0.0513 / −0.0061 SIG | −0.0506 / −0.0055 SIG | **+0.89 / +0.92** |
| `turnover_vol_7d` | −0.0289 / −0.0114 SIG | −0.0102 / +0.0005 spans0 | +0.36 / +0.22 |
| `illiq_vol_7d` | −0.0256 / −0.0123 SIG | −0.0038 / +0.0016 spans0 | +0.25 / +0.12 |
| `amihud_7d` | −0.0168 | **+0.0038 sign flip** | +0.14 / −0.01 |
| `log_dvol_7d` | +0.0010 | **−0.0163 sign flip** | +0.09 / +0.23 |

**G1 — incremental through the real pipeline. Nothing adds in both eras:**

| added to V0_LEAN | Δ rank-IC RECENT | Δ rank-IC OOS |
|---|---|---|
| turnover_vol_7d | −0.0011 [−0.0021,−0.0000] hurts | −0.0002 noise |
| spread_cs_7d | +0.0002 noise | −0.0004 noise |
| amihud_7d | −0.0014 [−0.0027,−0.0002] hurts | +0.0002 noise |
| illiq_vol_7d | −0.0019 [−0.0031,−0.0007] hurts | −0.0001 noise |
| log_dvol_7d | −0.0014 [−0.0025,−0.0003] hurts | +0.0002 noise |
| past_alpha_7d | −0.0000 noise | +0.0005 noise |
| **all six together** | −0.0024 [−0.0048,−0.0001] hurts | +0.0011 noise |

**The most informative single number** is `past_alpha_7d`: vol-residualized IC −0.0145/−0.0160 — comparable in
magnitude to the *entire* model's IC, stable in both eras, only +0.14-0.19 correlated with vol — and it adds
exactly **−0.0000 / +0.0005**. The existing 14 features already span it (V0_LEAN carries `return_1d`, `ret_3d`
and vol terms; a 7d past-residual sum is largely reconstructible from those).

This is now the **fifth** signal this cycle with strong standalone IC that contributes nothing incrementally
(after the persistent tilt, the low-vol sort, the OI-universe, and OI/ADV). The consistent explanation: the
crypto cross-section on free price/volume data is essentially ONE factor, our model already prices it, and
every "new" characteristic is another view of it. The crypto liquidity literature's top factors
(turnover volatility, spread) do NOT transfer to a 4h BTC-residual book — spread is the vol factor
(corr +0.9), and the rest are era-unstable.

**⇒ Adding more cross-sectional characteristics derived from price/volume is closed.** Only genuinely new
information (S2) or a change in how existing information is used (S3/S4) can move this.

---

## Iteration 2 — S3: train on the trading objective instead of MSE

**Why this next, and not S2.** S1 established that more price/volume characteristics cannot help, so the
remaining owned-data lever is *how* the existing information is used. The SOTA paper attributes its net-Sharpe
gain (2.3 vs the prior 1.25) specifically to **jointly learning the factors and the trading policy** rather
than fitting a predictor and bolting a portfolio on afterwards — "crucial to maximize profitability after
trading costs." Our incumbent fits RidgeCV to squared error on `xs_z` and applies cost afterwards. Cost is our
measured binding constraint, so an objective that internalizes turnover is the highest-prior owned-data move.
S2 (on-chain) is queued behind it because it is a data-acquisition problem — free historical per-token
address counts do not exist for a 176-token universe.

**H2.** Optimizing the per-symbol linear model against a net-of-cost portfolio objective (Sharpe of the
realized book minus a turnover charge at the true calibrated cost) beats optimizing it against MSE, on
held-out net Sharpe.

**Design — isolate the objective, hold architecture fixed.** The prior work (`gen_coef_shrink.py`) showed
pooled coefficients are far worse than per-symbol (+3.46 vs −0.71), so switching to a pooled policy would
confound the objective change with an architecture change. Instead: keep the identical per-symbol linear
parameterization and the identical features, **initialize at the RidgeCV solution**, and optimize the joint
portfolio objective by gradient ascent. Any gain is then attributable to the objective alone.

- weights: cross-sectionally demeaned score, L1-normalized per bar (dollar-neutral, differentiable)
- objective: `Sharpe(book returns on alpha_A) − λ · mean turnover`, λ set to the calibrated per-symbol cost —
  **not tuned**, so it cannot be fitted to the evaluation window
- no autodiff library is installed (no torch/jax), so gradients are derived analytically and checked
  numerically before use

### Pre-registered gates
- **G1.** Held-out (2025-01→2026-06) net@10k beats the MSE-fitted incumbent, paired 7d-block CI > 0.
- **G2.** The gain is not just turnover suppression already available for free — it must also beat the
  incumbent *with the band/EWMA turnover controls already applied* (the loop's existing best construction).
- **G3.** Both-era sanity: the objective-trained model's rank-IC must not degrade materially in either era.
- **Falsifier.** G1 fails → the objective change does not transfer to our setting; record and move to S4/S5.

### Result — **G1 FAIL, G2 FAIL, G3 FAIL. S3 as implemented is a NULL — it overfits.**
(`live/sd_iter2_objective.py`, `live/state/sd_iter2.log`)

Analytic gradient verified numerically before training (max rel err 4.45e-05; the run aborts above 1e-4).

**The training objective works exactly as designed — that is the problem.** In every one of the 18 folds the
training-window objective rises from ≈−0.09 at the Ridge init to ≈+0.30. Note the sign: **the Ridge solution,
evaluated as a net-of-cost continuous book, has NEGATIVE in-sample Sharpe** — independent confirmation that
cost is the binding constraint. But out of sample:

| | RECENT rank-IC | OOS rank-IC |
|---|---|---|
| MSE (incumbent) | +0.0229 | +0.0176 |
| trading-objective | **+0.0101** | **+0.0094** |

**G3 FAIL — the objective roughly halves predictive IC.** 2,450 free parameters optimized directly on a
Sharpe objective is textbook overfitting; the fixed rho=1e-3 shrinkage was too weak, and I will not retune it
after seeing the result.

Held-out 2025-01→2026-06, top-40:

| construction | gross | net@10k | turnover |
|---|---|---|---|
| continuous book, MSE | +1.90 | −2.59 [−4.56,−0.81] neg | 0.399 |
| continuous book, trading-objective | +1.15 | −1.80 [−3.39,−0.41] neg | 0.256 |
| incumbent BAND construction | — | −0.27 [−1.97,+1.23] | — |

- **G1 FAIL**: Δ(objective − MSE) +0.79 [−1.40,+2.94] spans 0.
- **G2 FAIL**: Δ(objective − band) −1.53 [−3.50,+0.40] — the trained policy is *worse* than the hand-picked band.

**What it did learn is instructive**: turnover 0.399 → 0.256, a 36% cut. The objective correctly discovered
that turnover is expensive — but that reduction is already available for free from the band (turn ≈0.26) and
the band gets it **without** paying half the predictive IC. A linear per-symbol policy can only reduce
turnover by shrinking/aligning coefficients, which damages the signal; it cannot express "don't trade when
the signal hasn't moved much", which is what a band does.

**Caveat on levels**: this script's own fold/merge path yields a restricted sample (band net −0.27 here vs
+0.85 in `cl_iter5_hardsplit.py`). The G1/G2 comparisons are paired on common bars so the *relative* reads
hold, but the absolute levels are not comparable to the main harness.

**Secondary finding worth keeping**: the continuous L1 book is a bad construction at our costs (net −2.59 vs
the band's −0.27). Concentration + a no-trade band beats spreading weight across the cross-section.

---

## Iteration 3 — S5: how many uncorrelated sleeves do we actually have?

**Why now.** Five iterations across two loops have established that no single additional signal, character-
istic, construction or objective moves this book. The literature calibration says the realistic route to a
6-Sharpe book is **not one better model** but `IR = r·√N` over N genuinely uncorrelated sleeves — at the
published SOTA r≈2.3, N≈7. That makes the decision-relevant question empirical and cheap: **how many
uncorrelated sleeves do we actually have, and what IR does their combination support?**

**H3.** The candidate sleeves are mutually near-uncorrelated, so combining them yields materially higher IR
than any one alone.

**Sleeves tested** (each built on the same held-out machinery, each net of calibrated cost):
1. **XS reversal** — the incumbent top40/band book.
2. **Intermediate momentum** — 14d trailing return, skip-recent (documented as a real, persistent,
   different-root sleeve in `docs/CONCLUSION_2026-08-03.md`).
3. **Carry** — cross-sectional funding rate (short high-funding / long low-funding).
4. **Low-vol factor** — the slow rvol sort from `cl_iter2_factor.py` (expected to be highly correlated with
   sleeve 1; included precisely as the negative control).
5. **Time-series trend** — 30d trend on the equal-weight basket, as a directional diversifier.

### Assumptions to validate
- **A1.** Correlations measured on NET return series, on the SAME bars, held-out window only.
- **A2.** Each sleeve's standalone net Sharpe reported with block CI — a sleeve with no edge adds nothing
  regardless of how uncorrelated it is.
- **A3.** The combination uses equal RISK weights fixed a priori, not optimized (optimizing weights on the
  evaluation window is the selection error iteration 5 of the previous loop already demonstrated).

### Pre-registered gates
- **G1.** At least 2 sleeves have standalone held-out net Sharpe > 0 with block CI excluding 0.
- **G2.** Mean absolute pairwise correlation of the positive sleeves < 0.3.
- **G3.** The equal-risk combination's net Sharpe exceeds the best single sleeve, paired block CI > 0.
- **Falsifier.** G1 fails → we do not have 2 independently profitable sleeves and the √N route is not open
  with owned data; report the honest sleeve count and what r each would need.

### Result — **G1 FAIL, G2 not evaluable, G3 FAIL — but the correlation matrix is the loop's key finding.**
(`live/sd_iter3_sleeves.py`)

| sleeve | held-out net Sharpe @10k |
|---|---|
| low_vol | +1.39 [+0.09,+3.09] SIG |
| xs_reversal *(the incumbent)* | +0.85 [−0.86,+2.43] spans0 |
| int_momentum | +0.47 [−1.27,+2.09] spans0 |
| ts_trend | −0.05 [−1.67,+1.69] spans0 |
| carry | **−2.45 [−4.15,−0.70] neg** |

**The sleeves are genuinely uncorrelated** — max |pairwise corr| across all five is **0.196**
(xs_reversal↔low_vol); every other pair is under 0.10. Diversification is NOT the obstacle.

**The obstacle is that the sleeves have no edge.** Only one clears zero; the equal-risk combination is −0.02
and loses to the best single sleeve by −1.38 [−3.00,+0.30] because carry drags it down. G1 fails: we do not
have two independently profitable sleeves.

**Caveat on `low_vol` +1.39**: this is the SAME signal the previous loop tested across both eras and rejected
(gross +0.35 OOS / −0.01 RECENT; four reweightings recovered nothing). One held-out window showing
significance does not overturn a both-era rejection — the prior test was the more thorough one. Treat it as
era-unstable, not as a validated sleeve.

**The √N arithmetic — the honest answer to "how do we get to 6.33":**

| per-sleeve net IR | uncorrelated sleeves needed |
|---|---|
| 1.0 | 40.1 |
| 1.5 | 17.8 |
| 2.0 | 10.0 |
| 2.3 (published SOTA) | 7.6 |
| 3.0 | 4.5 |

We have five mutually uncorrelated sleeves and **zero** with a defensible net edge.

---

## Iteration 4 — S4: residual-path term structure

### Result — **NULL.** (`live/sd_iter4_respath.py`)

Standalone, this is the **strongest signal anything in either loop produced** — every horizon significant in
both eras, same sign, clean monotonic reversal decay:

| | pa_4h | pa_8h | pa_12h | pa_1d | pa_2d | pa_7d | pa_14d |
|---|---|---|---|---|---|---|---|
| RECENT | −0.0421 | **−0.0456** | −0.0444 | −0.0446 | −0.0354 | −0.0233 | −0.0238 |
| OOS | −0.0281 | **−0.0294** | −0.0268 | −0.0235 | −0.0258 | −0.0227 | −0.0157 |

All 14 cells significant. No flip to continuation at any horizon — pure reversal, decaying with lag.

**Incrementally it is era-locked:**

| added | Δ RECENT | Δ OOS |
|---|---|---|
| pa_4h | +0.0018 [+0.0005,+0.0030] ADDS | −0.0003 noise |
| pa_8h | +0.0017 [+0.0007,+0.0027] ADDS | +0.0008 [−0.0001,+0.0016] noise |
| pa_12h | +0.0015 [+0.0004,+0.0027] ADDS | +0.0005 noise |
| pa_1d | +0.0012 [+0.0003,+0.0022] ADDS | −0.0002 noise |
| full path (7 lags) | +0.0020 [−0.0004,+0.0045] noise | +0.0008 noise |

Four horizons add in RECENT, none in OOS. Giving a linear model the *entire* term structure adds nothing
beyond noise in either era — so a sequence model over the residual path is not the missing piece: the
information is already spanned by `return_1d`/`ret_3d` plus the vol terms. (The 8h/12h members of this family
are already deployed in the v4 long book as `resid_rev_2`/`resid_rev_3`, so even a pass would have partly
re-derived an existing design choice.)

---

## Iteration 5 — S2: on-chain data

**CORRECTION.** Earlier in this loop I asserted that free historical per-token address counts "do not exist
for a 176-token universe". That was wrong. The CoinMetrics **community API** (no key, no cost) serves daily
AdrActCnt/TxCnt back to 2023 for **27 of 176 base assets** — 15% of the universe, but concentrated in the
majors that are the deployable universe anyway. The direction was testable and I tested it.

### Result — **NULL.** (`live/sd_onchain_fetch.py`, `live/sd_iter5_onchain.py`)

Baseline recomputed on the restricted 27-name universe (+0.0249 RECENT / +0.0268 OOS — note OOS is *higher*
on majors than the full universe's +0.0210, consistent with the signal being cleaner in big names).

Standalone: every feature is era-unstable. `adr_growth_7d` +0.0072 RECENT / −0.0024 OOS (sign flip);
`tx_growth_7d` +0.0057 / −0.0031 (sign flip); `adr_z_30d` spans 0 in both; `adr_per_dvol` +0.0058 spans0 /
+0.0189 SIG but its vol-residual flips sign (−0.0106 SIG RECENT / +0.0008 spans0 OOS) and it is 0.37-0.39
correlated with vol — the vol factor again.

Incremental: nothing passes both eras. `tx_growth_7d` adds in OOS (+0.0019 [+0.0001,+0.0037]) and is −0.0010
noise in RECENT. All four together: −0.0034 RECENT / +0.0012 OOS, both within noise.

**On-chain activity is the one input that CANNOT be a restatement of price/volume by construction — and it
still adds nothing.** That is the strongest available evidence that the ceiling is informational, not a
question of finding the right transform.

---

# Loop close-out (2026-08-06)

**Five directions, five nulls. Nothing adopted.** Combined with the cost/turnover loop: **12 iterations across
two loops, zero adoptions.**

| # | direction | verdict |
|---|---|---|
| — | multi-factor risk model | **killed before testing** — measured: target already factor-neutral (n_eff 97/175), book R² on PC1-5 = 4.1% |
| S1 | liquidity characteristics | NULL — all six fail both-era incremental |
| S2 | on-chain (27/176 assets) | NULL — era-unstable standalone, nothing incremental |
| S3 | trading-objective training | NULL — overfits, halves OOS IC, loses to the hand-picked band |
| S4 | residual-path term structure | NULL — strongest standalone signal in either loop, era-locked incrementally |
| S5 | sleeve count | NULL — sleeves uncorrelated (max 0.196) but only 1 of 5 has an edge |

## The one finding that explains all of them

**Seven signals with strong standalone IC and ~zero incremental value:**

| signal | standalone IC | incremental |
|---|---|---|
| residual path pa_8h | −0.046 / −0.029 both eras SIG | RECENT only |
| slow low-vol sort | 2-4× the model's IC, both eras | book Sharpe ~+0.3 |
| spread_cs_7d | −0.051 / −0.051 both eras SIG | +0.0002 / −0.0004 |
| OI/ADV | +0.044 / +0.023 both eras SIG | = vol factor (corr −0.54/−0.59) |
| past_alpha_7d | vol-resid −0.0145/−0.0160 both eras SIG | −0.0000 / +0.0005 |
| persistent tilt | ≈ the full model's IC both eras | era-locked at book level |
| on-chain tx/addresses | era-unstable | nothing |

The V0_LEAN 14-feature model **spans the free-data information set for this cross-section.** This is not "we
haven't found the right feature yet" — it is one factor, already priced. The `spread_cs_7d` case is the
cleanest illustration: the biggest raw IC in the entire loop (−0.051, both eras) and correlation +0.89/+0.92
with the vol rank.

## Honest position on the 6-Sharpe question

- Our ceiling on owned data: **+1.55 gross / ~+1.2 net with passive execution** (held-out, CI spans 0).
- Published SOTA net: **2.3** (Attention Factors, ICAIF 2025). Crypto ML papers report 3.1-3.2.
- 6.33 sits above everything published, on a 133-day sample whose Sharpe CI is [3.0, 9.7].
- The gap is **not a model we are missing.** It is execution (measured: +0.85 → ~+1.2 passive), sleeve count
  (we have <1 profitable sleeve, need ~8), and — for an ex-HFT desk — probably genuine liquidity-provision
  revenue, which is a different business from alpha and is capacity-limited.

## What would actually move the number (ranked, and none of it is more signal research)

1. **Real fill data.** Post `GTX` orders in small size through the existing paper-trading harness and record
   actual fills. Cheapest item on the list, and it settles the one number the passive-execution probe had to
   assume (klines show the price *touched* a level, not that we filled). Quantified benefit: +0.85 → ~+1.2.
2. **Paid data at scale** — full L2 history, historical positioning, on-chain for the whole universe rather
   than 15% of it. Every free-data avenue is now measured and closed.
3. **Sleeves from different markets/mechanisms**, not more cross-sectional signals on the same 176 perps. The
   sleeve correlations (max 0.196) show diversification would work if the sleeves had edge.

## Method note for the next loop

The gate that did all the work was **both-era + incremental**. Standalone IC passed in six of seven cases and
was wrong every time. Any future candidate should be judged on `Δ` against the incumbent in both eras before
anything else is measured — it is fast, and it would have killed all seven leads in the first hour.

Scripts: `live/sd_*.py`; logs `live/state/sd_iter*.log`; caches `live/state/cost_loop/sd*`. All uncommitted.

---

# Addendum — iteration 6: reading the papers properly, and the experiment I had skipped

The literature review above was done from abstracts and search summaries. Challenged on it, I read the full
texts. Three things changed.

## What the papers actually say

**Attention Factors (ICAIF 2025) — the central experiment varies the FACTOR COUNT:**

| K (factors removed) | gross SR | net SR |
|---|---|---|
| 8 | 3.35 | 1.94 |
| 30 | 3.97 | **2.28** |
| 100 | 4.52 | 2.19 |

Gross rises monotonically; **net has an interior optimum at K=30** of ~1000 names. Their explanation:
"these higher order factors capture weak signals and local dependency patterns." Method: conditional latent
factors from attention over characteristic embeddings, 8-year rolling window, LongConv sequence model over a
30-day residual window, costs of 5 bps per unit turnover + 1 bp shorting, objective = net Sharpe + 100 ×
explained variance.

**Their ablation is the most useful single table** — drop a characteristic group, measure net SR:
past returns 2.28 → **0.59** (74% of all performance); trading frictions → 1.34 (41%); value, profitability,
investment → negligible. Price-based reversal/momentum *is* the edge; fundamentals are not.

**Two corrections to what I told you earlier:**
- I cited the replication paper's "Sharpe > 10" alongside the others. Reading it: **it applies no transaction
  costs at all**, and its own authors flag overfitting, regime-specificity and possible leakage. Not a
  credible benchmark; it should not have been in that table.
- Their **two-step PCA** variant — the architecture closest to ours — nets ~1.5, and joint optimisation is
  what lifts it to 2.28. We are at ~1.2 net. The comparable-architecture gap is ~0.3 Sharpe, not the ~1.1 I
  implied against the 2.3 headline.

## The experiment I had skipped — and my own error

I killed the multi-factor direction earlier on a **breadth** argument (target already at n_eff 97/175). That
measurement is correct but answers the wrong question: the paper's claim is that the residual becomes more
**predictable**, not more independent. Alpha as a function of K was never tested. Scaled by universe size,
their K=30/1000 is ~K=5 for our 176 names.

### iter6 (`live/sd_iter6_factorK.py`) — train on xs_z(K-factor residual), evaluate all K on the SAME P&L

| | K=0 | K=2 | K=5 | K=10 |
|---|---|---|---|---|
| OOS gross | +1.47 | +1.69 | **+1.98** | +1.89 |
| OOS net@10k | +0.67 | +0.88 | **+1.12** | +1.03 |
| RECENT gross | +0.88 | +0.05 | +0.06 | −0.50 |
| rank-IC vs alpha_A (OOS) | +0.0211 | +0.0167 | +0.0165 | +0.0153 |

The OOS half **reproduces the paper's shape** — monotonic rise to an interior optimum at exactly the K their
result scales to — while RECENT does the opposite. Every paired delta spans 0. Note also the dissociation:
rank-IC against `alpha_A` falls monotonically with K while the OOS book improves, which is what you would
expect if the model is learning something better aligned with a factor-neutral book than with the BTC residual.

### iter6b (`live/sd_iter6b_hardsplit.py`) — the honest resolution

RECENT has 1596 bars and CI widths of ~4 Sharpe, so it cannot contradict OOS; this loop's own method note
says to use a chronological hard split for a level question. Select K on 2023-06→2024-12, evaluate on
2025-01→2026-06:

| | SELECT net@10k | HOLDOUT net@10k |
|---|---|---|
| K=0 | +1.08 | −0.00 [−1.54,+1.50] |
| K=2 | +1.01 | −0.18 |
| **K=5 (selected)** | **+1.16** | **+0.00 [−1.62,+1.68]** |
| K=10 | +1.11 | −0.40 |

The rule picked K=5 — correctly, it was best on the selection window. Held out it delivers **Δ +0.01
[−1.24,+1.30] over K=0**. Nothing.

**And this explains the iter6 era split**: the "OOS" era (2023-06→2025-09) largely *overlaps* the selection
window, so the monotonic improvement I saw there was selection-contaminated, not out-of-sample. The hard split
removes the overlap and the effect vanishes.

**Verdict: the weak-factor result does NOT transfer to a 176-name 4h crypto cross-section.** My earlier
dismissal was right, but for the wrong reason — the correct reason is that removing more factors does not
improve the alpha, not that breadth was already adequate.

## What survived the deeper read

- **VALIDATED**: past returns carry the edge (their 74% ablation) — consistent with our all-price/vol feature
  set and with S1/S2 nulls on liquidity and on-chain data.
- **REFUTED for us**: the weak-factor / high-K residual construction.
- **STILL UNTESTED** — the one genuine idea left from the papers: in their model, characteristics build
  **conditional factor loadings**, not return predictions. Their trading-frictions group costs 41% of net
  Sharpe when dropped *in that role*. We tested those same variables (spread, turnover-vol, Amihud) as
  predictors (S1, null). Using them as loading instruments — characteristic-sorted factor portfolios, then
  residualise, then predict — is a different mechanism and has not been tried here.

---

# Iteration 7 — multi-agent literature sweep, and its single survivor

An 11-agent workflow (5 readers × 5 adversarial screeners × 1 synthesis, 1.17M tokens) read the stat-arb,
crypto-factor, microstructure and portfolio-construction literature with this repo's falsified list in hand
and was instructed to KILL mechanisms by default. **14 of 15 were killed** as already-falsified, redundant
with the one factor, or not implementable on free data.

## What the sweep established about the literature (numbers the abstracts do not advertise)

| source | universe / freq | gross | **net** |
|---|---|---|---|
| Attention Factors K=30 (best net anywhere) | 500 US large caps, daily | 3.97 | **2.28** |
| — same paper, PCA benchmark | | 2.79 | **1.57** |
| — same paper, OU + threshold policy | | 1.26 | **−2.54 to −7.05** |
| DLSA (Mgmt Sci 2025), IPCA-5 | ~550 US stocks, daily | 4.16 | **1.11** |
| — same paper, PCA residuals K=10 / K=15 | | 3.36 | **−0.08 / −0.87** |
| Jensen-Kelly-Malamud-Pedersen, Markowitz-ML | US equities, monthly | 2.00 | **negative** |
| — same paper, Portfolio-ML | | 1.43 | **1.38** |
| Fieberg et al., JFQA 2025 (CTREND) | 3,245 crypto spot, **weekly** | 1.94 | ~1.45 @30/40bp |

Three load-bearing facts:
1. **The best properly-costed net Sharpe in this literature is 2.28**, and **not one of these papers reports a
   confidence interval on any Sharpe or Sharpe difference.** By this repo's standard none of the headline
   comparisons is established.
2. **No paper reports a net-of-cost Sharpe for a crypto cross-section at intraday frequency. Zero.** Every
   crypto number is weekly or daily spot. We operate where the literature has no result to beat.
3. **The incumbent is already at that frontier**: held-out gross +1.55, taker net +0.85, passive net
   +1.17..+1.31 vs Fieberg's full-universe crypto net ~1.2-1.45. Not underperforming — at the level, and the
   level is ~1.2-1.5, not 6.

Papers flagged for reporting no costs or non-implementable portfolios: Babiak-Bianchi's "OOS Sharpe >13" is on
characteristic-managed *factor returns*, not a portfolio; the order-flow paper reports break-even TC only, on
paid segmented-flow data; the cross-exchange paper's horizon is 500 ms and its live maker experiment earned
~1 bp on $1.5M over five days.

## The survivor: clock-phase order imbalance — **REJECTED, and the mechanism is now understood**
(Kim & Hansen, arXiv 2607.09426; `live/qoi_build.py`, `live/sd_iter7_qoi.py`)

Order imbalance in the first 10s of each quarter-hour boundary, where scheduled execution clusters. Screened
GENUINELY_NEW because every flow feature here averages over that phase (fl_tfi/fl_vpin 5-min, OB-flow 5-min,
bookDepth 30s). Built from raw aggTrades: 31 symbols × 1,246 days, 3.7M boundary windows.

**A0 measurability — PASS.** 31/31 symbols under the 5% abort threshold (0.002-0.05% missing windows,
0.16-4.1% empty).

**A1 — the pre-registered killer — FAIL, decisively.** Univariate slope in bps, day-clustered SE:

| dependent variable | RECENT | OOS |
|---|---|---|
| raw 4h return | +3.95 (t +1.20) ns | +3.86 (t +1.83) ns |
| raw, **cross-sectionally demeaned** | **−0.15 (t −0.20)** | **+0.09 (t +0.22)** |
| BTC-beta residual | +3.41 (t +1.97) SIG | +0.37 (t +0.29) ns |
| residual, demeaned — **OUR TARGET** | **+0.01 (t +0.02)** | **+0.20 (t +0.46)** |

**The effect is entirely market-wide.** Demeaning the cross-section takes the slope from +3.95 bps to −0.15 —
it does not shrink, it vanishes. Boundary-clustered algorithmic flow hits all names at once, and a
cross-sectional book subtracts exactly that common component by construction. This is a *structural*
incompatibility between the paper's setting (time-series, six contracts, no cross-section) and ours, not a
weak-signal problem. Caveat recorded: we do not cleanly replicate even the time-series effect (t = 1.20 / 1.83,
both insignificant), so the replication itself is uncertain — but the verdict does not depend on it.

**G1 — FAIL.** Standalone cross-sectional rank-IC ≈ 0 in both eras (qoi_last +0.0031 / +0.0009, both span 0).
One cell significant (qoi_mean4h RECENT −0.0116) and era-inconsistent. Look-ahead sanity check clean: IC
against the +1-bar-shifted target is |·| < 0.01 everywhere, far under the 0.10 rule.

**G2 — FAIL.** Incremental over V0_LEAN on the matched 30-symbol universe: every delta within noise in both
eras (largest +0.0002).

**Harness hazard found and fixed**: `v0_feature_ablation.gen()` reads `alpha_vs_btc_realized` internally and
wraps each symbol fit in a bare `except Exception: pass`, so renaming that column upstream made every fit fail
silently and returned an empty frame rather than an error. Only iteration 7 was affected (verified: iterations
1, 4, 5, 6 never renamed the column on the frame passed to `gen`). **That bare except should be narrowed** —
it can convert a schema error into a silent null.

**⇒ Per the pre-registered falsifier, the flow-as-alpha family is now closed permanently.**

---

# FINAL SYNTHESIS — both loops (2026-08-06)

**14 iterations across two loops. Zero adoptions.** That is the result, not a failure to find one.

## What is established
- **The 14-feature V0_LEAN model spans the free-data information set for this cross-section.** Eight signals
  with strong standalone IC contributed nothing incrementally: residual path (−0.046 both eras), low-vol sort
  (2-4× the model's IC), spread_cs (−0.051, corr +0.90 with vol), OI/ADV, past_alpha_7d, the persistent tilt,
  on-chain activity, and now clock-phase order imbalance. Each was a different view of one factor, or —
  in iteration 7's case — of a market-wide component the target removes by construction.
- **It is not a risk-model problem.** The target is already factor-neutral (mean pairwise corr +0.0046,
  n_eff 97/175); the book's R² on PC1-5 is 4.1%; K-factor residualisation fails the hard split (Δ +0.01).
- **It is not a construction or objective problem.** Turnover control, weighting schemes, and direct
  net-Sharpe policy optimisation were all tested; the last overfits and loses to a hand-picked band.
- **It is not a diversification problem.** Five sleeves are mutually uncorrelated (max 0.196) — and only one
  clears zero.
- **The incumbent sits at the literature's net frontier for its segment** (~1.2-1.5), which is well below the
  6.33 that prompted this work, and which no published crypto intraday cross-sectional result matches.

## What would change the picture — none of it is more signal research
1. **Real fill data.** Post `GTX` in small size through the paper harness and record actual fills. Settles the
   one assumption the passive-execution probe had to make (klines show a price *touched*, not that we filled).
   Measured benefit: +0.85 → ~+1.2 net.
2. **Paid data at scale** — full L2 history, historical positioning, on-chain across the whole universe rather
   than 15% of it. Every free-data avenue is now measured and closed.
3. **Sleeves from different markets or mechanisms**, not more cross-sectional signals on the same 176 perps.

## Method note
The gate that did the work was **both-era + incremental delta**. Standalone IC passed in seven of eight cases
and was misleading every time. Iteration 7 adds a second cheap killer worth running first on any future
candidate imported from a time-series paper: **regress the candidate on the cross-sectionally demeaned target
before anything else** — if the effect is market-wide it dies there, in one command, for free.
