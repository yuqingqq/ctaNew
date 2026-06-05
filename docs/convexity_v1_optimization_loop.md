# Convexity v1 — optimization loop (book-B-only + resid_rev + K=3)

Sustained, diagnosis-driven optimization on the **new v1 structure** (single low-vol book, resid_rev
dual-pred long ranker, base short ranker, K=3, 24h/6-sleeve, regime gate, full-V0). Honest walk-forward
baseline **Sharpe +3.46** (monthly re-rank, no look-ahead). Engine: `live/v1_opt_loop.py`.

**Standing rule:** a screen lift is NOT adoption. Any survivor must pass (1) per-fold ≥6/9 positive,
(2) matched-basket placebo p95, (3) no single-fold dominance — before touching the frozen v1.

## Why re-test prior ideas
Most of the historical ledger (`live/state/opt_loop/insights.md`, vBTC memory) was tested on the OLD
**two-book** or **51-symbol-panel** structure. The conclusions may not transfer to book-B-only on the
~78-name low-vol universe. This loop re-runs the transferable levers on v1.

## Catalog of prior ideas — re-test status on v1
| idea | prior result (old structure) | v1 re-test |
|---|---|---|
| Dynamic regime gate (bull/bear/side) | always on | **batch1 gate_off** (never removed on book-B) |
| Bull-mode (mom vs pred vs betaneut) | mom default | batch1 bull_pred / bull_bnmom |
| Vol-scaled leg sizing (inv_vol/sqrt/cap) | two-book: ~neutral | batch1 size_* |
| Falling-knife idio-vol long-skip | two-book mixed | batch1 idioskip80/90 |
| Dispersion/conviction gate (DISP_GATE) | vBTC conv_gate helped | batch1 dispgate |
| Equity-DD vol-stop overlay | iter-012 | batch1 stop_off / stop_tight |
| SIDE_MODE variants (longmom_shortmr, longdef_shortmr) | 2026-06-03 surgical | batch1 sidemode_* |
| Breadth / higher-K | **#182 REJECT** (K=3 best) | done |
| rvol rank stability (smooth/hysteresis) | **#180 REJECT** | done |
| Cutoff ensemble | **#181 ADOPT** (v2 form) | done |
| Maturity gate | **KEEP 180d** | done |
| Cross-exchange premium feature | **#185 REJECT** (redundant w/ resid_rev) | done |
| Feature mining / GBM / pooled / clustering | vBTC: real IC, no lift | closed (graveyard) |

## Iteration log
### Iteration 1 — `batch1` env-toggle screen ✅ NOTHING beats baseline (+3.46)
- **gate_off +2.22** (−1.23, maxDD −4759) → KEEP regime gate (earns keep even on low-vol book).
- **stop_off +3.17** (maxDD −4150) → KEEP DD-stop.
- **idioskip80/90 +2.91** (−0.55) → falling-knife skip HURTS here (flipped vs old high-vol book — no knives to skip).
- bull_bnmom +3.46 (=baseline); bull_pred +3.19.
- **size_invvol +3.37, maxDD −1488 (−33%)** — Sharpe-neutral (paired block-boot ΔSh CI[−1.18,+0.91], P(worse)=0.58),
  DD-improvement P=0.91 (<p95). RISK-overlay candidate for forward test, NOT a frozen-v1 change.
- dispgate +3.10 (6/9 folds).

### Iteration 2 — `batch2` SIDE_MODE re-test (via AB_SIDEMODE_B) ✅ none beat dual-pred default
- longdef_shortmr +3.34 (6/9); confidence_btc_hedge +2.51 (maxDD −870); longmom_shortmr +2.22; regime_switch +2.16 (**8/9 folds**).
- Dual-pred default is Sharpe-optimal; alternatives trade ~1 Sharpe for robustness (regime_switch) or low DD (conf_btc_hedge).

### Iteration 3 — `batch3` hold/sleeve horizon ✅ 24h confirmed optimal (inverted-U peak)
4h +2.22 → 12h +2.85 → **24h +3.46 (peak)** → 36h +3.05 → 48h +3.17. Shorter bleeds turnover cost, longer decays
the 4h signal. Production 6-sleeve/24h is optimal on book-B. No lift.

### Iteration 4 — `rrh` resid_rev horizon ✅ [8h+12h] confirmed optimal
[4h,8h] +2.86 → **[8h,12h] +3.46 (peak)** → [12h,16h] +3.30 → [4h,8h,12h,16h] +2.63 → [8h,16h,24h] +3.25.
Adding the 4h window HURTS (microstructure noise — the bid-ask-bounce concern). Production resid_rev windows optimal. No lift.

### Iteration 5 — POOLED model vs per-symbol Ridge ❌ REJECTED — deepest insight of the loop
Pooled per-cycle IC **+0.047 (58% higher** than prod +0.029) yet portfolio **−0.71** (vs +3.46). Leg decomp SYMMETRIC:
pool_longonly +0.67, pool_shortonly +0.81 (both legs crater ~−2.7) → the POOLING ARCHITECTURE itself fails, not one leg.
- Raw signal fine: naive gross K=3 L/S +3.58 > prod +2.40; decile-monotone; less concentrated (174 vs 130 names).
- Loss REGIME-concentrated: craters in the folds prod wins big (f0 +3858→−2331, f6 +3302→+7, f7 +2592→−44); loses
  7× more in hi-dispersion cycles (−20.6 vs −2.9 bps). Pooling flattens conviction (pred spread 0.028 vs 0.138, 5×).
- **ROOT CAUSE:** per-symbol standardization = "how unusual *for this symbol*" → regime-stable, scale-calibrated TAIL
  conviction. Pooled standardization = absolute cross-sectional extremeness → higher *average* IC but FRAGILE at the
  K=3 tails (regime-conditional). **For a K=3-extremes XS L/S, average IC is the wrong objective — tail calibration is.**
  This is WHY production is per-symbol, and why the IC≠Sharpe graveyard (GBM, pooled #167, OI) keeps failing.

### Iteration 6 — 2×2 isolation (standardization × coefficients) ❌ pooling rejected, root cause = COEFFICIENTS
[per-sym std, per-sym coef] +3.456 (prod) · [pooled std, per-sym coef] **+3.455 (≡ prod)** · [pooled std, common coef]
−0.71 · [per-sym std, common coef] −2.27. **The benefit is ENTIRELY the per-symbol coefficients; standardization is
irrelevant** (a per-symbol linear fit absorbs any feature scaling). Each symbol has a genuinely different feature→return
relationship; a common vector averages it away. (Corrected my premature "matched-pair" reading — the 4th cell falsified it.)

### Iteration 7 — partial coefficient shrinkage β=(1−α)β_sym+αβ_common ❌ REJECTED, monotone decline
α: 0 **+3.46** / 0.1 +3.29 / 0.2 +2.84 / 0.35 +2.05 / 0.5 +1.43 / 1.0 −0.71. Even a 10% nudge hurts. Per-symbol
coefficients are the genuine best estimate of each symbol's behavior — not noisy fits awaiting denoising. **Per-symbol
model beats both extremes AND every blend → true optimum. Architecture thread definitively closed.**

## DEFINITIVE VERDICT (2026-06-04, 4 iterations, ~34 configs + xexch)
**v1 (book-B + resid_rev[8h,12h] + K=3 + 24h/6-sleeve + regime gate + DD-stop, +3.46) is at its local optimum.**
Every lever tested fails to beat it: regime gate, DD-stop, K, leg-sizing, SIDE_MODE, idio-skip, dispersion-gate, hold
horizon, resid_rev horizon, cross-exchange feature. Non-Sharpe risk levers exist (`size_invvol` −33% DD Sharpe-neutral;
`regime_switch` 8/9 folds) — candidates for the FORWARD test, not backtest adoption. **Backtest construction space is
exhausted; further grinding = data-snooping risk. The composition-variance risk (placebo p83) can only be resolved by
forward data (#178), not another backtest.** Recommend pivot: loop → forward paper-test + operational (#178/#179).

### #185 cross-exchange premium — alignment CORRECTED (user caught timing bug)
Original premium was 4h-misaligned (venue close@period-end vs Binance close@hh:05) → IC inflated. Boundary-aligned
(all venues open@hh:00): univariate IC collapses >50% (okx −0.016, cb −0.019 OOS). Aligned portfolio retest HURTS more
(both +2.29 / long +2.64 / short +3.24). **>half the "strong" −0.04 IC was a bar-marking artifact. Reject reinforced.**
Timing confirmed: signal determined at hh:05 (resid_rev needs close@hh:05); hh:05 entry is the earliest causal action.

## Data-driven insights (running)
- Composition variance is the dominant risk (placebo p83) and is **irreducible via construction levers**
  on free data (breadth/#182, rank-stability/#180 both failed). It is cross-sectional, not temporal.
- The 4h residual-alpha signal is at its extraction ceiling: even external orthogonal data (Coinbase/OKX
  premium, #185) adds no portfolio lift — it's a noisier proxy for resid_rev (corr −0.55).
- resid_rev only works as a **dual-pred long ranker** (global feature corrupts shorts; #185 same lesson:
  signals must enter at the right layer, not mixed into the per-symbol Ridge).
- _(extended as the loop runs)_
