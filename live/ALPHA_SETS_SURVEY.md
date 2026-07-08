# Released alpha-factor sets — survey & port plan

Context: the convexity book farms BTC-beta-neutral cross-sectional alpha on a 175-sym 4h USDM panel.
GTJA Alpha191 (106-factor faithful subset) was ported and screened in a parallel session — survivors
after beta-neutralization + honest validation: alpha082/alpha065 (stochastic/reversal family),
alpha095 (liquidity vol). This doc surveys the OTHER publicly released alpha sets and what to port.

## The sets

| set | source | count | inputs needed | feasible here | verdict |
|---|---|---|---|---|---|
| GTJA Alpha191 | 国泰君安《基于短周期价量特征的多因子选股》(2017) | 191 | OHLCV+VWAP, benchmark | done (106 ported) | **DONE** |
| WorldQuant 101 | Kakushadze, "101 Formulaic Alphas" (arXiv 1601.00991, 2015) | 101 | OHLCV, VWAP, returns, adv{d}; 19 need IndClass/cap | 82 | **PORT (57 novel)** |
| Qlib Alpha158 | Microsoft Qlib benchmark config | 158 | OHLCV+VWAP only | 158 | **PORT** |
| Qlib Alpha360 | Microsoft Qlib | 360 | raw price/vol lags ÷ close, 60 bars | all | SKIP — normalized raw lags, an NN input representation, not standalone factors; our LGBM/screen frame gains nothing |
| Barra CNE5/USE4 styles | MSCI handbooks | ~10 styles | prices + fundamentals + cap | ~4 (mom, resvol, liquidity, beta) | SKIP — the price-computable styles (momentum, residual vol, liquidity, beta) are already in V0_LEAN (`ret_3d`, `rvol_7d`, `idio_vol_*`, `beta_to_btc_change_5d`, `obv_z_1d`) |
| 华泰/海通/光大 factor reports | CN broker research | var. | var. | — | SKIP — not released as reproducible formula lists; the reproducible CN-broker set IS GTJA191 |
| hf_features.py | in-repo legacy | 160+ | OHLCV | done | already mined in earlier phases |

## WorldQuant 101 — what it adds over GTJA191

GTJA191 borrowed ~25 formulas nearly verbatim from WQ101 (e.g. WQ#2≈G001, WQ#26≈G005, WQ#41≈G013,
WQ#42≈G120, WQ#55≈G176...). Those are already screened. The **57 novel implementable** alphas are the
distinct WQ structural families:

- **conditional sign-flip reversal** (#9, #10, #49, #51): delta(close) gated by ts_min/ts_max of deltas.
- **rank-power / rank-ratio composites** (#78, #81, #84, #85, #94): `rank(x)^rank(y)` interactions — a form GTJA never uses.
- **boolean cross-sectional comparisons** (#61, #62, #64, #65, #68, #74, #75, #86, #95, #99): `rank(a) < rank(b)` → ±1 signals.
- **adv{d}-conditioned liquidity interactions** (#7, #17, #25, #39, #43, #72, #77, #88, #92, #96, #98): price signal × volume/adv regime — GTJA mostly uses raw V.
- **decay_linear pipelines** (#57, #66, #71, #72, #73, #77, #88, #96, #98) with ts_rank wrappers.
- **multi-term linear blends** (#36): weighted sum of 5 rank terms.
- **candle anatomy** (#101, #33, #60): (close-open)/(high-low) family.

Excluded: #48,58,59,63,67,69,70,76,79,80,82,87,89,90,91,93,97,100 (IndNeutralize — no industry
classification for crypto; sector work in Phase F/G showed cluster-neutralization adds nothing here),
#56 (needs market cap). Excluded as GTJA-duplicates: #2,3,5,6,8,11,15,16,18,21,22,23,24,26,35,37,
40,41,42,46,47,52,54,55,83.

Window adaptation: same rule as the GTJA port — fractional windows rounded; daily windows kept in
bars except very long ones capped at 100 bars (200/230/250d → 60–100); adv120/adv180 → adv60.

## Qlib Alpha158 — what it adds

Systematic grid, not hand-crafted formulas: 9 KBAR candle-shape features + 4 price-level ratios +
29 rolling operators × windows {5,10,20,30,60}:

- trend regression on time: **BETA (slope), RSQR (R²), RESI (residual)** — genuinely absent from GTJA/WQ.
- quantiles: **QTLU/QTLD** (rolling 80/20 close quantile ÷ close).
- positional: IMAX/IMIN/IMXD (argmax/argmin of high/low), RANK (ts pct rank), RSV (stochastic).
- path stats: MA/STD/MAX/MIN/ROC.
- up/down asymmetry: CNTP/CNTN/CNTD (count), SUMP/SUMN/SUMD (RSI-style magnitude).
- volume: VMA/VSTD/WVMA (vol-weighted return vol), VSUMP/VSUMN/VSUMD (volume RSI), CORR (close vs log-vol), CORD (returns vs vol-changes).

Overlap with GTJA exists (RSV≈stoch family G047/G057/G082, SUMP≈RSI G063/G067, CNTP≈G053/G058) — the
marginal-IC screen orthogonalizes; family dedup happens at rep-selection like the alpha191 run.

## Test protocol (identical to alpha191 flow, separate file namespace)

1. `alpha101_lib.py` / `alpha158_lib.py` → factors on the same 175-sym 4h panel, PIT shift(1),
   memory-safe per-factor loop (float32, fixed master index, gc) — the alpha191_lib pattern.
2. Beta-neutralize per cycle against trailing 180-bar BTC beta (generalized `alphaset_betaneut.py`).
3. Screen: per-cycle Spearman raw_IC + marg_IC orthogonal to V0_LEAN (generalized `alphaset_screen.py`).
4. Validate dedup reps: non-overlap honest t (stride H=6), per-cycle orthogonal IC, 4-fold sign
   stability, within-cycle placebo (generalized `alphaset_validate.py`).
5. Cross-set dedup: survivors correlated against alpha191 survivors (alpha082/065/095) — keep only
   factors adding signal beyond BOTH V0_LEAN and the already-validated alpha191 reps.

Bar: |marg_t/no| > ~2.5, folds ≥3/4, |placebo_t| < 2 — same as alpha191.

## Results (2026-07-06)

Screens on betaneut factors (175-sym panel, 1595 cycles, target = 24h fwd alpha_vs_btc_realized):

**WQ101 (57 novel): REJECTED.** Screen showed 29 factors |marg_t|>3, but honest non-overlap validation
of 10 family reps: best |marg_t/no| = 2.09 (wq073) — nothing clears 2.5. The strong screeners
(wq061/027/036, raw_t/no ≈ 3.1–3.8) are redundant with V0_LEAN once orthogonalized. WQ's distinct
structural families (boolean rank-comparisons, rank^rank powers, adv-conditioned pipelines) add no
orthogonal signal on this panel.

**Alpha158: 4 honest survivors** (marg_t/no, folds, placebo all pass): q158_IMIN60 +3.92,
q158_SUMD20 +3.23, q158_RSV60 +3.03, q158_RANK60 +2.96 — all positional/asymmetry family. Note their
raw honest t is <2.5; the value is conditional (orthogonal complement), not standalone.

**Cross-set dedup vs alpha191 survivors (alpha082/065/095 added to conditioning set):**
RSV60 collapses (t +1.63 — it IS the alpha082 stochastic family, per-cycle corr −0.67);
SUMD20 borderline (+2.36, 4/4); RANK60 keeps +2.73 (4/4); **IMIN60 keeps +2.86 (3/4)** and is the
most independent (corr −0.23 to alpha082). Bars-since-low is the one genuinely new signal beyond
GTJA191 + V0_LEAN.

Verdict (pre-regime-check): candidate adds = q158_IMIN60, q158_RANK60 (marginal, conditional).

## Regime check (2026-07-06) — DOWNGRADES the verdict to REJECT ALL

The screen/validation window (2025-10 → 2026-06) is a single regime. Re-running the honest marginal
IC over the FULL 2021–2026 history (11,883 cycles), per year, non-overlap, orthogonal to V0_LEAN:

| factor | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | FULL |
|---|---|---|---|---|---|---|---|
| q158_IMIN60 | +1.1 | +1.4 | −1.8 | −0.8 | **+2.9** | +0.4 | +1.2 |
| q158_RANK60 | +2.4 | 0.0 | −1.4 | −1.5 | +1.1 | +1.0 | +0.8 |
| alpha082 (a191 ref) | −1.8 | −0.5 | +2.0 | +1.4 | −1.7 | −2.9 | −1.1 |
| alpha065 (a191 ref) | +0.8 | −0.7 | +2.1 | −0.4 | +2.3 | +0.9 | +1.8 |
| alpha095 (a191 ref) | +0.8 | −1.7 | −0.5 | −1.8 | −0.5 | −3.2 | −2.0 |

(cells = t-stat of per-year marginal IC)

- IMIN60's "survival" is a 2025 artifact; it sign-flips in 2023/24. RANK60 similar. Full-history t ≈ 1.
- The alpha191 survivors show the SAME pathology — alpha082 flips sign between 2023/24 (+) and
  2025/26 (−), the documented mean-rev sign-flip regime pattern. This implicates the parallel
  alpha191 session's shortlist too.
- Phase DDI showed per-cycle IC is unpredictable from regime features (R²=0.005), so a PIT regime
  switch to harvest these conditionally is not available.

**Final verdict: 0 regime-robust survivors across GTJA191 + WQ101 + Alpha158 (~320 factors).**

## Regime-CONDITIONAL screen (2026-07-06) — can a factor play in ONE regime?

Setup: production trend regimes (btc_ret_30d: bear<−0.10 | side | mild_bull | hot_bull≥0.15) +
idio-dispersion split, all PIT; all 321 betaneut factors × 6 buckets over full 2021–2026; per-regime
honest non-overlap t. Multiple-testing bar: ~1,900 tests ⇒ need |t|≥4 + episode consistency.
Script: live/alphaset_regime_screen.py; outputs live/state/longtail/regime_screen_{a191,wq,a158}.csv.

Screen-level specialists found (strong in one trend bucket, |t|<2 in all others):
q158_CORR5 bear −3.78 | alpha054, wq074 side | alpha082, alpha072 hot_bull-only (the alpha191
session's top pick is a hot-bull-conditional signal, not a general one). No bear-vs-bull sign-flippers.

Stage-2 (within-cycle V0_LEAN residualization, per-regime non-overlap t, EPISODE consistency across
contiguous regime runs, placebo):

| candidate | regime | t_in | t_out | episodes agree | verdict |
|---|---|---|---|---|---|
| q158_CORR5 | bear | −2.83 | −0.66 | 10/15 | weak — fails bar, episode-inconsistent |
| alpha054 | side | −2.99 | −0.60 | **29/37** | best structural story; IC only −0.014, below bar |
| wq074 | side | −0.02 | +0.69 | 15/37 | pooled-resid artifact, dead |
| alpha082 | hot_bull | −1.29 | −0.51 | 9/11 | direction consistent but magnitude DECAYED (−0.05 in 2021 → ~0 in 2023+) |
| alpha072 | hot_bull | −1.14 | −0.19 | 10/11 | same decay pattern |

**Verdict: regime-specialist structure is real at screen level but nothing clears the honest
multiple-testing bar in the strict within-cycle frame.** Closest: alpha054-in-side (29/37 episodes,
p≈4e-4 binomial, but IC −0.014 and "side" is 48% of cycles — a thin everywhere-in-normal-markets
effect, not a regime play). The bear-book angle (production is flat in bear; a bear alpha would be
directly monetizable via BEAR_MODE=side) finds no factor strong enough to build on.
Caveat: the earlier "TOP 20 by per-regime t" pooled-resid table is inflated for slow vol/liquidity-level
characteristics (alpha081/100/095 full_t ≈ 10–14 pooled vs ≈ −2 within-cycle) — pooled OLS doesn't
remove time-varying V0_LEAN loadings; only the within-cycle frame is decision-grade.

## Frame correction (2026-07-06): test vs the ACTUAL per-symbol ridge predictions

Production is per-symbol RidgeCV (price book V0_LEAN / flow book +VPIN,TFI, recency HL=60) — NOT
pooled, NOT cross-sectional. Neither of the OLS orthogonalization frames above nests it. The
decision-grade test: within-cycle orthogonalize factor & fwd target on BOTH production pred books
(live/state/convexity/hl{,_residrev}/v0full_hl60.parquet, 2025-10-04→2026-06-21, 1547 cycles),
honest non-overlap t. Reference: pred_s IC +0.034 (t +4.5), pred_l +0.032 (t +4.3).

| factor | margIC beyond preds | t | folds | note |
|---|---|---|---|---|
| **wq036** | **+0.0228** | **+3.79** | **4/4** | regime-sign-stable full-history (all 6 buckets +, full_t +5.2 pooled) |
| alpha082 | −0.0268 | −3.69 | 3/4 | adds on this window BUT hot-bull-conditional + decayed historically |
| alpha065 | +0.0200 | +2.52 | 4/4 | borderline |
| q158_IMIN60/RANK60/SUMD20/RSV60 | ≤0.009 | ≤1.2 | — | V0_LEAN-frame "survivors" collapse — ridge already spans positional structure |
| alpha054, q158_CORR5, wq073, alpha095 | — | ≤2.1 | — | dead vs preds |

KEY LESSON: frames disagree exactly as feared — the V0_LEAN-frame survivors are pred-redundant, and
the pred-frame survivor (wq036) was V0_LEAN-frame-rejected (t +1.93). Only the pred frame answers
"does it add to what we trade".

**wq036** (2.21·rank(corr(close−open, delay(vol,1),15)) + 0.7·rank(open−close) +
0.73·rank(tsrank(delay(−ret,6),5)) + rank(|corr(vwap,adv20,6)|) + 0.6·rank((ma(close,60)−open)·(close−open)))
is the single candidate passing all three: adds beyond production preds (+3.79, 4/4), regime-sign-stable
2021–2026, survives ~68-test selection burden (p≈0.01 Bonferroni).

## Adoption A/B (2026-07-06): wq036 REJECTED at book level

Full controlled test (gen_wq036_wf_preds.py + run_wq036_ab.sh): row-matched WF pred regeneration
(exact gen_alpha065 machinery — same cuts, HL=60, xs_z target, per-symbol RidgeCV; 267,925 rows × 6
pred sets), then identical v3_native strategy + matched execution, varying ONLY the pred set.

Pred level: adding wq036 does NOT lift IC — base +0.0366 (t +5.5) → bn +0.0355 / raw +0.0359;
pred corr 0.97. Book level (2025-10 → 2026-06, 1601 cycles):

| cell | daily Sharpe | totPnL bps | maxDD bps |
|---|---|---|---|
| baseline V0_LEAN | **+3.21** | **+25,791** | 3,762 |
| +wq036_bn | +2.79 | +22,018 | 3,665 |
| +wq036_raw | +2.75 | +22,067 | 3,713 |

Paired daily diff ≈ −14 bps/day (block-bootstrap CI mostly negative, crosses 0 at the top).
Both variants HURT by ~15% of PnL / −0.42 Sharpe.

**Mechanism — the architecture gap:** wq036's validated signal is a cross-sectional-rank relationship
(within-cycle orthogonal IC vs preds). A per-symbol time-series ridge cannot express cross-sectional
rank information — each symbol's fit sees only its own wq036 history vs its own xs_z — so the feature
adds a noisy per-symbol coefficient (~170 extra estimated params) and dilutes the model. Same class
of lesson as vBTC WINNER_23: orthogonal-complement IC ≠ model lift; here it's actively negative.

The only remaining route would be a construction-layer overlay (final rank = pred + λ·rank(wq036)),
but that introduces a tuned continuous parameter calibrated on the SAME single window that produced
the +3.79 — the exact pattern that failed nested-OOS repeatedly (K3 margin, V3.3 decay weights).
NOT recommended.

**FINAL: published-alpha mining is closed at every level tested — feature frame, regime-conditional
frame, pred-orthogonal frame, and book-level adoption. 0 adoptions from ~320 factors.**
Factor-mining published alpha sets against this residual target is closed. The only defensible use
would be as LGBM features where the tree can learn regime interactions — but the WINNER_23 test
already showed that fails honest OOS for signals of this magnitude.
Outputs: live/state/longtail/alpha{101,158}_screen_{raw,betaneut}.csv, alpha{101,158}_val.log.
