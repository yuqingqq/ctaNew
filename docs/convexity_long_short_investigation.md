# Convexity Long/Short Investigation — Root Cause & Adopted Fixes

**Date:** 2026-05-29  **Model:** convexity_portable (per-sym Ridge, 17 V0 features, 6-sleeve
24h-hold, regime-gated, vol-stop).  **OOS:** 2025-10-04 → 2026-05-26.
**Split:** H1 = 10/04→01/22, H2 = 01/22→05/26.

## The alarm

Forward test showed H1 Sharpe **+2.70**, H2 **−2.70** (reproduced exactly in replay → harness
trustworthy). The strategy worked then collapsed. This documents the root cause and the two
validated fixes.

## Root cause

Both H1 and H2 are **bear markets** for alts (median alt −75% / −59% cumulative). Decomposing the
L/S edge into beta + selection:

| | H1 long selection | H1 short selection | H2 long selection | H2 short selection |
|---|---|---|---|---|
| alpha (bps/cycle) | +40 (t+2.2) | −3.9 (ns) | +7 (t+1.8) | −2.3 (ns) |

- **Long side carries real cross-sectional alpha; short side selection is ~always negative** — its
  only useful component was bear-market beta.
- The raw long edge flipped sign H1→H2 not because the signal "broke" but because long alpha shrank
  ~5× while the bear-beta drag stayed.
- **Mechanism (the real root cause):** the model rides a **regime-fragile** signal (mean-reversion of
  recent returns/funding). In H1 the beaten high-vol names bounce; in H2 the *same* picks keep
  crashing — the feature→return relationship **inverts**. A single static model can't track a
  non-stationary relationship.
- **Short side is structurally unfixable from this model:** any gate/percentile/threshold is
  downstream of the same prediction vector. Short conviction is anti-informative OOS (deepest-
  conviction shorts are the *worst* shorts; they're recently-*down* names, not the rare crash-zone
  pumps where short alpha lives).

## What was ruled out (all tested, all consistent)

| lever | result |
|---|---|
| Feature engineering (78 features, 8 categories) | no robust H2 long lift; significant features are momentum → invert in H2 |
| Preprocessing (XS-rank vs per-sym-z) | helps structurally (−5.8→−2.0 H2), not a regime fix |
| Model class (pooled LGBM vs per-sym Ridge) | LGBM better but same H2 collapse |
| Gating (z / percentile / absolute) | can't fix shorts (downstream of same preds); z-gate non-stationary |
| Retrain **cadence** (monthly, equal-weight) | H2 +1.9 only, hurts H1 — regime not staleness |

## What was adopted (validated, mechanistically understood)

**1. Design A — held alt-basket hedge** (`SIDE_MODE=long_basket_hedge`). Replace the value-negative
model short leg with a held, beta-neutral equal-weight eligible-basket short (6-sleeve aggregation
keeps it low-turnover). BTC-short rejected: alts fell harder than BTC, so BTC under-hedges
(residual beta +0.29, gross collapses).

**2. Recency-weighted training (~60d half-life)** via `--halflife-days`. Distinct from retrain
cadence — it **decontaminates** the fit by downweighting old-regime (2021–24) data. Improves *both*
regimes (it's a cleaner fit, not a regime trade). Lift is robust across 30/60/90d half-lives.

### Cumulative system performance (replay, real cost/sleeves/stop)

| config | H1 Sh | H2 Sh | **ALL Sh** | DD |
|---|---|---|---|---|
| A0 original baseline | +2.70 | −2.70 | **+1.28** | −3,000 |
| + Design A hedge | +2.41 | −0.44 | **+1.55** | −4,043 |
| + recency weighting (60d) | +2.90 | −0.39 | **+1.89** | −4,181 |

**Net: aggregate Sharpe +1.28 → +1.89 (+0.61), two understood changes.**

## A third lever found but NOT adopted standalone: the defensive factor

Oracle ceiling analysis (perfect-foresight top-K = +575/+495 bps H1/H2; model captures only 6%/1%)
shows the H2 opportunity did **not** vanish — the model's *capture* collapsed. The optimal longs
share a **stable cross-regime signature: low volatility (atr/idio/rvol) + high corr-to-BTC** — a
defensive/low-vol anomaly that, unlike momentum, does NOT flip across regimes.

A parameter-free defensive two-stage long (top-N by pred → pick K most-defensive) is the **only
config that makes H2 positive** (H2 Sharpe +0.56). But standalone it halves H1 (aggregate +0.82),
so it is not adopted. It is **regime-complementary** with the model:

| | high-dispersion regime (H1) | low-dispersion bear (H2) |
|---|---|---|
| model long + hedge (recency) | **+2.90** ✓ | −0.39 ✗ |
| defensive tilt | +1.05 | **+0.56** ✓ |

## Regime switch — tested and REJECTED

Diagnosis: H1 and H2 are *both* bear (H1 more so); raw BTC vol is **equal**; the real regime
difference is **model prediction dispersion** (H1 2.13 vs H2 0.50, 4.3×). `pred_disp` is the best
PIT switch signal (zero lag, +49.5 bps Q4−Q1 long-alpha spread).

But a `pred_disp` regime switch (model-long when dispersed, defensive when flat) **failed in system
replay** — every threshold gave aggregate Sharpe +1.83–1.88 (< recency-A +1.89) and made H2 *worse*
(−0.44 to −1.27). Two reasons:
1. `pred_disp` separates regimes *on average* but is a **noisy per-cycle router** — it mis-assigns
   cycles and ends up worse than both pure modes.
2. **Recency weighting subsumes the defensive benefit** — defensive lifts H2 to +0.56 on prod preds
   but only −0.21 on recency preds; the decontaminated model already captures most of the stable
   signal, leaving nothing for a switch to gain.

The rank-stack second layer (learned blend) was likewise rejected — stable weights but a dominated
static compromise (a fixed blend can't be regime-optimal; the non-stationarity recurs one level up).

## Adopted config (superseded — see TARGET REDESIGN below)

Design A (held basket hedge) + recency-weighted training (60d). Aggregate Sharpe +1.89 (H1 +2.90,
H2 −0.39). H2 negative. This was the local optimum **conditional on the residualize+z target** — and
that turned out to be the binding limiter (next section).

## TARGET REDESIGN — the ceiling broken (optimization loop, 2026-05-29)

The "+1.89 local optimum / everything exhausted" conclusion held the **target** fixed. The target
(per-symbol-z of BTC-residualized return) structurally strips the cross-sectional/defensive axis —
fixing it reopened real headroom.

- **xs_z target** (per-cycle cross-sectional z of raw fwd return) on the per-sym model lifts H2 IC
  ~10× and turns H2 positive. **+K=3** concentration (monotonic in K, robust to nested-OOS) gives the
  final config.
- **Extensible**: the rank/z target transfers to the per-sym model; a pooled-no-sym_id model passed
  leave-symbols-out (held-out-symbol IC = full-universe IC) — universe-portable.

The target redesign also fixed the **short side**: on the old resid_z model shorts were
anti-informative (forcing the passive hedge); on the xs_z model the bottom-K carries real short
alpha (H2 short edge +15.9 bps), so letting the model pick both legs beats the passive hedge.

**FINAL PRODUCTION CONFIG: per-sym Ridge on xs_z target + recency 60d + K=3 + MODEL long/short
(SIDE_MODE=default), beta-neutral leg sizing.** System replay 2025-10-04→2026-05-26: **aggregate
Sharpe +2.95 (H1 +3.36, H2 +2.53)** — up from +1.89. Cost-robust (3/4.5/9 bps: +3.02/+2.95/+2.65),
K=3 robust vs nested-OOS. The passive-hedge variant (+2.65) is the conservative fallback (more
consistent month-to-month, no single-month dependence).

**Robustness caveat (LOFO, iter-14):** the *aggregate* lift is robust — drop the best month (April)
and model L/S is still +2.00, beating recency-A's full +1.89. But the **H2 positivity is
April-dependent**: H2 +2.53 → −0.39 ex-April (≈ recency-A). So model L/S robustly improves aggregate
Sharpe and is the least-bad H2 of all configs even ex-April (−0.39 vs −0.70 vs −2.06), but it does
**not** robustly turn the bear regime positive — that headline leans on one strong month.

**Honest caveat:** a matched-basket placebo shows per-cycle *selection*-vs-random skill is weak
(+2.5 bps, t≈0.4) — the prior "+14 long-vs-median" was skew-inflated. So +2.65 is real and validated
but **substantially structural** (6-sleeve smoothing + beta-neutral hedge + K=3 concentration + broad
IC), not sharp name-picking. We capture little of the +495-bps oracle; cross-sectional-rank features
improved selection metrics but did **not** survive the full system. **Selection skill is the
remaining frontier** — likely needs orthogonal (paid on-chain) data, not more free-feature work.

## ORDER-FLOW — the orthogonal-signal breakthrough (2h push, 2026-05-30)

The "+2.95 is the free-data ceiling" conclusion was **price-feature-exhausted, not free-data-exhausted.**
aggTrade **order-flow** (TFI, signed-volume-z, VPIN, Kyle-λ, large-trade share, aggressor ratio;
`data/ml/cache/flow_*.parquet`, 69 of 156 syms) — genuinely orthogonal to price/funding, never tested —
adds real, robust alpha on the xs_z model:

- On the **69-sym flow universe**, V0+flow vs price-only: **Sharpe +3.50 vs +2.26**, H1 +4.55 vs +2.90,
  H2 +2.10 vs +1.27, **DD −2,384 vs −3,862 (38% less)**.
- **7/8 months positive uplift**, biggest in the *bad* months (Jan −8.51→−3.99, May −5.32→**+1.61**) —
  it **de-lumps** the strategy. LOFO +1.71 (survives best-month drop). Matched placebo: H1 long edge
  **+10.5 (t+1.9)** = the first real *selection*-skill lift.
- **The flow sleeve (+3.50 / DD −2,384) beats the price-only full-universe champion (+3.02 / DD −4,527)
  on Sharpe AND drawdown** — the smaller universe is more than compensated by the orthogonal signal.

**Best config now: xs_z + recency60 + K=3 + model-L/S + ORDER-FLOW features, on the 69-sym flow
universe = +3.50.** (Hybrid full-universe mixing flow-model and price-model preds is broken — +1.98 —
due to incomparable pred scales; fixing it via per-symbol pred-normalization is future work for
*capacity*, not Sharpe, since adding price-only names dilutes the flow-sym quality.)

Operational: needs **live aggTrade ingestion** (flow data ends ~2026-05-06); flow leg is 69/156 syms.
Lesson: orthogonal **microstructure** data is a distinct, productive free signal — the frontier the
price-only work had wrongly declared closed.

## TWO-BOOK diversification — current best (5h push, 2026-05-30)

The pred-merge hybrid (combining flow-model and price-model predictions in one cross-section) is a dead
end. The right way to use the full universe + flow is **two separate books combined at the PnL level**:

- **Book A — flow sleeve:** flow model (xs_z + K=3 + model-L/S + order-flow) on the 69 flow symbols. +3.50.
- **Book B — price book:** price model (same construction, price-only) on the 87 non-flow symbols. +2.21.
- **A↔B correlation = 0.17** → 50/50 PnL combine = **Sharpe +3.71, DD −1,417** (full 156-universe).

Diversification of two low-correlated, independently-validated books raises Sharpe *above either book*,
cuts DD 41% vs the flow sleeve, and **de-lumps** the strategy (6/8 months positive, the other two
breakeven; LOFO +2.64 — the most robust config produced). Weight-robust (untuned 50/50; any flow-weight
≥0.5 beats +3.50). Sound portfolio math, not a fit.

**Full arc: +1.89 → +2.95 (target redesign) → +3.50 (order-flow) → +3.71 (two-book diversified).**

Deploy two books (flow model on flow-syms, price model on non-flow-syms), 50/50, combine PnL. Needs live
aggTrade ingestion for the flow book.

**Full-flow fetch test (Phase VI):** fetched real aggTrade flow for 10 missing liquid symbols (pipeline
works end-to-end). Order-flow GENERALIZES — 8/10 new symbols carry positive flow IC (+0.0159 ≈ original
+0.0174), so it's a real *general* microstructure signal, not specific to the original 69. The kline
taker-imbalance *proxy* was rejected (too weak — lacks large-trade/VPIN/Kyle). BUT extending coverage is
**capacity, not Sharpe**: the flow sleeve dilutes with more names (K=3 composition), and the expanded
two-book (+3.69) ≈ the +3.71 champion. Fetch all 106 only if live-trading *capacity* requires it.

**Remaining Sharpe frontier:** the number is bounded by the per-sleeve IC ceiling AND the count of
*uncorrelated signal axes* (currently 2: price + flow, corr 0.17 — already captured by the two-book).
More names on the existing axes = capacity; raising Sharpe requires a **3rd orthogonal axis** — a
genuinely new data type (on-chain, options flow, basis microstructure) — not more flow coverage.

## P5 — Limitations & deployment plan (honest)

**Forward Sharpe expectation (by regime):**
- High-dispersion regimes (H1-like): strong, ~+3 (the bulk of the edge).
- Low-dispersion bears (H2-like): aggregate stays positive but **the bear-regime component is
  fragile** — in-sample H2 +2.53 is largely April; ex-April ≈ breakeven. Expect bear-regime
  contribution near zero, not the +2.53 headline. Honest blended forward Sharpe: **~+1.5 to +2.5**,
  wide because it depends on the dispersion regime mix and the residual edge being structural.

**What the edge IS and ISN'T:** the matched-basket placebo shows per-cycle selection-vs-random is
weak (long t≈0.4, short t≈1.6). The Sharpe is **substantially structural** — 6-sleeve smoothing,
beta-neutral sizing, K=3 concentration, broad IC — plus a modest, partly-April-driven selection edge.
We capture little of the +495-bps oracle. **Selection skill is the unbroken frontier**; free-feature
work (78 features + xsrank) does not close it — that likely needs orthogonal **paid** data
(on-chain/cohort), consistent with the vBTC line's conclusion.

**Kill-switch / monitoring (deploy):**
- Gross-exposure kill on equity drawdown (the existing iter-012 vol-norm stop) — keep it.
- Monitor `pred_disp` (model conviction): the iter-044 regime signal. Sustained low pred_disp =
  low-edge regime → consider de-risking (do NOT switch to a "defensive" model — that mis-routes,
  iter-045).
- Track realized per-cycle long/short IC on a trailing window; if it decays toward zero and stays
  there (signal death / universe drift), halt and retrain.
- Annual (or on universe drift) retrain; the config is extensible (leave-symbols-out proven), so new
  listings can be scored once they have the trailing features.

**Residual risks:** (1) the strategy is universe-composition sensitive (prior finding); (2) the H2
edge is regime- and month-fragile; (3) the absolute selection skill is low, so most return is
structural and could compress if costs rise or the basket/sleeve mechanics change.

**Deploy recommendation:** ship the xs_z + K=3 + model-L/S config (gen preds monthly-WF recency60
xs_z target; `SIDE_MODE=default STRAT_K=3`); keep passive-hedge xs_z (+2.65) and recency-A (+1.89) as
documented fallbacks; size for a forward Sharpe ~+1.5–2.0, not the in-sample +2.95; live-monitor as
above with a hard kill-switch.

## Honest bottom line

- Absolute prediction quality is **low** (captures 1–6% of the oracle) — most 4h cross-sectional
  alpha is unpredictable from free Binance features. That's a real ceiling.
- The *fixable* problem is that the model aimed its limited capacity at a **regime-fragile** signal.
  The adopted fixes (hedge + recency) lift aggregate Sharpe to +1.89; the defensive factor and a
  regime switch are the path to a positive-H2 without sacrificing H1.

## Phase VII — full-flow definitive test (2026-05-31)

Fetched real aggTrade flow for ALL 175 universe symbols (97/97 new built, 0 fail) and tested the
core integration question with full coverage: does flow in ONE unified per-sym book beat the
partial-coverage two-book? **No — decisively.** At the production config (K=3, SIDE_MODE=default),
scored on the production bot replay:

| config | Sharpe | note |
|---|---|---|
| two-book (flow BookA + price BookB, 50/50) | **+3.71** | CHAMPION, reproduced exactly |
| flow sleeve standalone | +3.50 | reproduced |
| unified V0-only (price, full universe) | +3.01 | reproduces the +3.02 baseline |
| **unified V0+flow (all syms)** | **+2.28** | full-coverage unified = HARMFUL |

Adding flow features to every symbol's Ridge **drops** Sharpe from +3.01 (price-only) to +2.28
(−0.73). Flow and price preds correlate ~0.86 on the same universe, so unified merging adds noise,
not an orthogonal axis. The flow edge only materializes through the **universe-split two-book**
(cross-sectional diversification, book corr 0.15). Full coverage = **capacity, not Sharpe** —
confirmed with full data; unified integration actively destroys the edge. **Two-book +3.71 remains
the champion.** Order-flow research is closed. The next real Sharpe lever is a *3rd orthogonal data
axis* (e.g. on-chain / Glassnode), not more flow coverage or more modeling on price+flow.
