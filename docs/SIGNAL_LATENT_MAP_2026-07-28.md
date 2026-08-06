# Signal Latent-Structure Map + Validation — 2026-07-28

Deep-research pass (not test→check): built a latent map of the whole signal space, derived the model's
predictions, and validated them walk-forward. Purpose: understand *what* the deployed edge is and *why*
everything else dies, instead of enumerating more pass/fail hypotheses.

Scripts: `live/build_latent_map.py`, `live/build_regime_derive.py`, `live/validate_model.py`,
`live/validate_lgbm.py`. All uncommitted; results below are for review.

## The map (PCA of V0 features ± OB/flow, per-PC cross-sectional rank-IC)

- Signal space is **rank ~7.5** (14 V0 feats) → **~10.6** (V0 + 8 OB/flow) — NOT low-rank. But
  **predictability concentrates in one latent**: PC1 = a VOLATILITY factor (idio_vol/atr), raw IC
  +0.066/+0.071 — ~2× everything else. Small both-era tail: momentum/reversal (PC2), BTC-regime (PC4/5).
- **This unifies the graveyard:** OB/flow features load onto the *same* PC1 as `atr` — i.e. OB-reversal,
  illiquidity, synchronization are all **re-measurements of the volatility latent**. They didn't fail
  independently; they're one factor in many hats.
- CAVEAT: map ICs are RAW (in-sample PCA, no embargo) → magnitudes inflated; read for STRUCTURE.

## Derived prediction → tested → REFUTED (and corrected)

Model predicted the vol/reversal factors are regime-conditional (would explain the era "instability").
`build_regime_derive.py`: corr(factor-IC, BTC-vol regime) ≈ 0 both eras; no monotone bucket pattern.
**Refuted.** But it revealed something better: at the factor level the edge is STABLE both eras
(idio_vol −0.063/−0.065, return_1d −0.034/−0.049). The "fragility" was over-read from noisy composites
(full-model era-gap, leg decomposition = noise, overlay-distorted Sharpe). Only weak survivor:
low-vol slightly stronger in down-trends (corr −0.09/−0.14, same sign both eras).

## Walk-forward validation (LIVE model = per-symbol Ridge `gen_residual_target`, embargoed, day-clustered CIs)

The deployed crypto model is the **per-symbol RidgeCV** (`gen_residual_target`), not LGBM. On it
(`validate_model.py`): 2-factor [idio_vol + return_1d] ≈ full-14 both eras —

| per-symbol Ridge (LIVE) | RECENT | OOS |
|---|---|---|
| full-14 | +0.0302 | +0.0210 |
| 2-factor (low-vol + reversal) | +0.0297 | +0.0210 |
| full − 2factor (paired) | −0.0005 [−0.005,+0.004] | +0.0000 [−0.004,+0.004] |

Paired diff **spans 0 both eras → the 12 other features are DEAD WEIGHT as Ridge predictors.** The model can
drop to 2 features with no edge loss (parsimony/robustness win). Caveat: dead *as predictors* ≠ delete from
the whole stack — some (btc_rvol, idio_vol) may also feed regime gates / sizing.

Non-deployed footnote: a pooled LGBM (`validate_lgbm.py`, NOT the live model) showed the 12 features add a
tiny OOS interaction increment (~+0.003) — irrelevant to the live per-symbol Ridge, and it conflated
pooling×nonlinearity so not cleanly attributable. Ignore for the live strategy.

## Validated conclusion (live per-symbol Ridge)

- The deployed edge is **vol-PRIMARY (low-vol) + secondary short-reversal** (+0.030 recent / +0.021 OOS); the
  prediction loads ~2× more on low-vol than reversal (see 2026-07-30 addendum — corrects an earlier loose
  "co-equal / reversal-primary" read). The other 12 features are **dead weight as predictors** → simplify to 2
  with no loss. NOT fragile at the factor level (low-vol factor stable both eras).
- Constraint is **thinness + cost**, not instability.
- **More gross signal requires VOL-ORTHOGONAL data.** Screen any candidate (on-chain, options, positioning)
  by projecting onto {vol, reversal} and measuring residual predictive IC before building on it.
- Signal-level optimization on this data is **understood and closed** (this map explains *why*, beyond the
  prior "closed" verdicts). Next gross alpha = orthogonal data; net improvement = cost/execution + diversify.

## Addendum 2026-07-30 — thorough decomposition (`build_signal_decomp.py`), corrects the "factor pair" framing

Prompted by a user challenge (which leg is high-vol?). Measured signal loading, book composition, and return
attribution on COMPARABLE bases (per-bar corr / cross-sec pctile), both eras. Corrects two loose reads of mine
(incl. a wrong verbal "reversal-primary" claim):
1. **SIGNAL is VOL-PRIMARY.** corr(pred, factor): low-vol atr_pct/idio_vol **−0.15..−0.17 BOTH eras** >
   reversal return_1d/ret_3d **−0.08..−0.12**. Gap widest OOS (vol −0.17 vs rev −0.08). Low-vol leads; reversal
   secondary. (Earlier leg-check "reversal-primary" was an artifact of comparing raw return_1d medians to vol
   PERCENTILES.)
2. **BOOK:** short leg = higher-vol (pctile 0.63–0.65 vs long 0.51–0.53) AND recent-winner (0.54–0.57 vs
   0.46–0.48). Vol separation (Δ0.11–0.14) ≥ reversal separation (Δ0.05–0.11), esp. OOS. Short = the high-vol
   recent-winner leg; **vol is the wider separator.**
3. **ATTRIBUTION (key):** crude cross-sectional pure-factor quintile books **LOSE OOS** (pure-reversal Sharpe
   −0.90, pure-low-vol −1.16) while the per-symbol-Ridge strategy **EARNS +1.84** (R² of strat on the two pure
   factors = **0.07**; RECENT 0.22, where both pure factors do earn +2.83/+2.18). ⇒ **the edge is NOT a naive
   vol/reversal factor BET** — it is the per-symbol RidgeCV's calibrated residual modeling; "factor pair"
   describes the INPUTS/loadings, not the earning mechanism, esp. OOS. (btc_rvol = time-series regime var, no
   cross-sectional loading.)

## WHY the edge exists + where its relatives are (2026-07-30, `build_edge_why.py`)
Hypothesis: behavioral **overreaction / lottery-preference premium** — retail overpays for high-vol, recently-
pumped "lottery" names → they underperform; low-vol & reversal are two proxies for the SAME "over-demanded &
mispriced" state (⇒ one factor). Tested:
- **SIBLINGS redundant ⇒ it IS one behavioral factor.** Canonical free lottery factors: MAX (Bali, trailing max
  return) RAW IC −0.046 but ORTH-vs-edge only +0.007 OOS / +0.004 RECENT (ns) → MAX's lottery content ≈ the
  low-vol factor, redundant. idio-SKEW(30d) is the ONLY sibling with a genuine both-era orthogonal residual
  (−0.004 OOS / −0.008 RECENT) — real but TINY/sub-cost (a skewness "jackpot-shape" dimension slightly beyond vol
  level). So the same-root siblings add ~nothing = the deep reason the edge is one thin factor.
- **MECHANISM = OVERREACTION (confirmed both eras).** Edge rank-IC is stronger for BIG recent movers than calm
  names both eras (RECENT +0.031 vs +0.025; OOS +0.020 vs +0.018) → reversal of attention/overreaction moves.
- **NOT exclusively in illiquid names.** Illiquid > liquid only RECENT (+0.035 vs +0.023); OOS ~equal (+0.0185 vs
  +0.0199). So the durable OOS edge lives in TRADABLE names too (consistent with the liquidity-filter win) — the
  cost wall is thinness, not "alpha only in untradeable names."
## TOP-40 (deployable universe) mechanism — DIFFERENT from the long tail (`build_top40_why.py`, 2026-07-30)
The small-cap overreaction/lottery story is a LONG-TAIL mechanism; restricted to the top-40 by ADV (the deployable
book) the mechanism shifts:
- **VOL-DOMINATED, not reversal.** corr(pred, factor) in top-40: vol −0.10..−0.16 both eras >> reversal −0.005..−0.05.
  The low-vol tilt dominates even more than in the full universe. Short leg = higher-vol majors (pctile 0.64–0.65
  vs long 0.52–0.55); reversal separation weak (0.53 vs 0.51).
- **STRONGER IN CALM NAMES, not big movers** (RECENT big +0.006 vs calm +0.012; OOS big +0.013 vs calm +0.023) —
  the OPPOSITE of the full-universe overreaction finding. So in majors it is NOT a fade-overreaction edge; it's a
  steady LOW-VOL ANOMALY.
- **Earns via PER-SYMBOL calibration, not a naive factor.** Crude pure-factor books LOSE in top-40 (pure-low-vol Sh
  −2.28 RECENT / −1.78 OOS) while the per-symbol Ridge strat earns (OOS +1.79), R²≈0 — the model does per-symbol
  relative-value (each major mean-reverts to its own vol/funding-adjusted norm), which the naive cross-sectional
  factor doesn't capture.
- **⇒ WHY the edge is there in the majors — WITHIN/BETWEEN decomposition (`build_top40_dig.py`) nails it:**
  the VOL edge is **BETWEEN-name (STRUCTURAL), not within.** BETWEEN rvol/atr IC −0.03..−0.04 BOTH eras (strong);
  WITHIN ~0. So it is the CLASSIC low-vol anomaly among majors: persistently-volatile majors (SOL/DOGE/racier alts)
  underperform persistently-calm ones (BTC/ETH) on a BETA-ADJUSTED basis. **This resolves the "naive book loses /
  per-symbol earns" paradox: the loss is BETA** (high structural vol = high beta → raw short-high-vol is short-beta,
  bleeds in a rising market) even though the beta-adjusted alpha is +. The per-symbol model earns because it targets
  the BTC-RESIDUAL (beta-stripped) return where the structural low-vol alpha lives → **the residual target + beta-
  hedge is the mechanism, NOT per-symbol cleverness.** Beta-hedge is essential, not optional. SECONDARY: within-name
  reversal (return_1d WITHIN IC −0.04/−0.026 — self-normalized, smaller). Stronger in calm (structural premium not
  swamped by big moves). Persistent = BAB/low-vol risk premium: volatile majors are structurally over-demanded by
  leveraged/speculative flow → lower risk-adjusted returns; we hold boring / short exciting, beta-hedged.
  (Correction trail: self-normalized → relative-value → STRUCTURAL-low-vol-in-residual-space+beta — the last is what
  the decomposition shows.)
- **REGIME dependence (`build_top40_regime.py`) — the mechanism is 'bet the excitement deflates':** top-40 beta-
  neutral alpha by regime. RECENT: QUIET(low btc-vol) **+4.88** vs FROTHY(high) **−3.01**; and HIGH cross-sec
  dispersion **−12.76** (disaster) — i.e. in the recent era the edge WORKS in quiet and gets CRUSHED when the
  volatile majors disperse/run (froth PERSISTS/validates → shorts squeezed). OOS is the OPPOSITE (frothy strongest
  +7.20) because OOS froth DEFLATED (shorts won). So it's not 'quiet vs frothy' per se — it's whether the volatile
  majors' excitement DEFLATES (edge wins: quiet, or froth-that-reverts) or PERSISTS (edge loses: RECENT froth).
  For a QUIET current market the edge is in its favorable regime (RECENT-quiet +4.88); the tail risk is persistent
  froth. Regime-timing itself is non-stationary (OOS↔RECENT flip) so don't hard-gate on it.

## Long-tail "why" (retained for the full universe, NOT the deployable top-40)
- **⇒ "similar edges":** same-root (lottery/overreaction) proxies are redundant; a genuinely new edge needs a
  DIFFERENT economic root — carry/funding (tested thin), liquidity-provision/Amihud (capacity-walled), lead-lag
  /information-diffusion (cross-asset, partial), or **longer-horizon MOMENTUM/underreaction** (opposite behavioral
  force to our short-horizon reversal — the least-tested candidate).

## LEAD: skip-recent intermediate MOMENTUM — first different-root, cost-clearing, diversifying signal (2026-07-30)
`build_momentum_ts.py` / `build_momentum_net.py`. Crypto is cross-sectional REVERSAL at every RAW horizon (1d→60d
all negative IC), BUT the ORTHOGONAL residual (vs {return_1d,ret_3d,vol}) of the 14d return is POSITIVE both eras
(+0.0079 OOS / +0.0066 RECENT, CI excl 0) = "skip-recent" intermediate momentum (underreaction) hiding under the
dominant reversal. Constructed clean: `mom_14_3` = trailing return t−14d..t−3d, residualized vs the edge; SLOW ⇒
low turnover.
- **NET VIABILITY (mom_14_3, EWMA λ=0.85, turnover 0.054):** net Sharpe POSITIVE at RETAIL both eras (RECENT +0.44,
  OOS **+1.20** @24bps; +1.6/+1.8 @6bps). corr to strategy **+0.11** (orthogonal). **Blend DIVERSIFIES the strategy
  at NET: OOS CI>0 at ALL costs (Δ@24 [+0.25,+2.13]); RECENT positive but TIE (CI spans 0).**
- **First signal in the whole program that is orthogonal + clears retail cost + diversifies at net** — because it's
  SLOW (14d) so heavy EWMA smoothing (turn 0.054) leaves the thin gross intact while killing cost. Validates the
  mechanism reasoning: a DIFFERENT root + a SLOW signal is what beats the cost wall the fast signals hit.
- **CALIBRATION (not yet "validated"):** (1) significant diversification is OOS-only; RECENT is a positive TIE
  (likely power + already-Sh4 strategy) — so "both-era SIGNAL, OOS-significant DIVERSIFICATION," not clean both-era
  DIV; (2) win REQUIRES the smoothing (λ=0 is net-neg) — needs λ-robustness; (3) first-pass block-30 CIs; (4)
  mom_30_7 is weaker (14d is the sweet spot — mild horizon selection, though 14d was pre-flagged by the raw screen).
  Pipeline-incremental confirmation (add to V0 through the real model, both eras) = `build_mom_pipeline.py`.
- **PIPELINE CONFIRMATION = NULL (`build_mom_pipeline.py`):** adding mom_14_3 to V0 through the real per-symbol
  RidgeCV does NOT improve rank-IC — RECENT Δ −0.0002 [−.0016,+.0011], OOS Δ −0.0008 [−.0017,+.0001], within noise
  both eras. (Weak-ish evidence: a thin orthogonal feature in a heavily-regularized per-symbol model wouldn't move
  rank-IC even if it has standalone sleeve value — cf. the cross-sectional positioning G2 null. But it gives NO
  corroboration from the feature angle.)
- **CALIBRATED VERDICT: promising but UNCONFIRMED candidate, not validated.** Evidence FOR: both-era orthogonal
  screen IC + net-positive standalone (low-turnover) + OOS blend DIVERSIFIES (CI>0). Evidence AGAINST/missing:
  RECENT blend only a TIE, pipeline-incremental null, win depends on λ=0.85. Net: ONE era's significant
  diversification amid ties = could be a fragile/false positive. To settle (not yet run): λ-robustness (0.7/0.9),
  QUARTERLY orth-IC stability (is it persistent or a period artifact like the OB-flow non-stationarity?), honest
  both-era DIV CI. Do NOT bank it as an edge until those pass. Still the best orthogonal lead found — the only
  candidate that cleared retail cost AND diversified in any era.

## VALIDATION OUTCOME (`build_mom_validate.py`, 2026-07-30) — REAL persistent standalone edge, but a THIN separate sleeve
- **A. QUARTERLY STABILITY — PASSES the acid test.** Orthogonal IC positive in **16/22 quarters** (2021Q1–2026Q2),
  mean +0.008, WORST only −0.009 (no sign-flips) — genuinely PERSISTENT, unlike the OB-flow trap (which flipped
  +/−/+ by era). Stronger in 2024–25 (+0.012..+0.026) than 2021–23; last 2 quarters weak (26Q1 +0.002, 26Q2 −0.003)
  = some non-stationarity in STRENGTH, but SIGN is stable. Real signal.
- **B. λ-ROBUSTNESS + conservative (7d-block) CI:** STANDALONE factor net@24 is robustly POSITIVE across λ both eras
  (OOS +0.6→+1.4, RECENT +0.1→+0.6 as λ 0.7→0.9) — a genuine thin standalone edge that clears retail cost. BUT the
  BLEND-vs-strategy ΔSh is a **TIE at every λ, both eras** under the conservative 7d-block CI (OOS λ=0.9
  [−0.23,+1.56]; the earlier "OOS DIV" under 5d-blocks did NOT survive the more conservative CI). So it does NOT
  robustly IMPROVE the deployed book; combined with the pipeline null, it's a SEPARATE sleeve, not a booster.
- **VERDICT: FOUND a genuine "similar edge" — skip-recent intermediate MOMENTUM (underreaction, different root).**
  It is REAL (persistent quarterly), ORTHOGONAL (corr +0.11 to the strategy), and STANDALONE net-positive at retail
  (~+1 Sharpe OOS / ~+0.4 RECENT, robust across λ) BECAUSE it's slow (turnover ~0.05) → beats the cost wall. Honest
  limits: (1) THIN (~+1 OOS Sharpe); (2) does NOT significantly boost the existing strategy (blend tie, feature
  null) — a small independent sleeve; (3) strength non-stationary (weak last 2 quarters). The MECHANISM METHOD
  worked: overreaction edge ⇒ hunt the opposite root (underreaction/momentum) ⇒ make it slow to beat cost ⇒ a real
  orthogonal edge. Scripts: build_momentum_ts / build_momentum_net / build_mom_pipeline / build_mom_validate.

## EXTENSION (`build_edge_hunt.py`, 2026-07-30) — composite doesn't strengthen momentum; CARRY fails persistence
- **Momentum COMPOSITE (14d+30d) does NOT strengthen the edge.** orth IC OOS +0.0083 (≈ single-14d +0.0079),
  RECENT +0.0051 (ns) — marginally more quarterly-stable (18/22 positive, worst −0.005 vs 16/22, −0.009) but NOT
  stronger; RECENT screen weakened to ns. ⇒ the momentum edge is genuinely THIN; combining horizons can't lift it.
- **CARRY (funding_rate_z_7d as a slow cross-sectional sleeve) FAILS the acid test.** orth IC ns both eras
  (−0.0009 OOS / +0.0025 RECENT); QUARTERLY 11/22 positive (coin flip), big sign-flips by period (21 positive →
  22 strongly negative −.016..−.027 → 24 positive) = classic NON-STATIONARY/regime-dependent, the OB-flow trap.
  The slow-signal trick does NOT rescue carry (confirms memory: funding self-defeating/no headroom). Not an edge.
- **⇒ FREE-DATA EDGE MAP COMPLETE:** (1) main = ONE behavioral overreaction/lottery factor (low-vol+reversal);
  (2) ONE genuine thin orthogonal sleeve = intermediate MOMENTUM (underreaction), real+persistent but thin
  (~+1 OOS Sharpe standalone, not a booster); (3) CARRY = non-stationary, fails; (4) positioning/OB = real but
  sub-cost; (5) everything else redundant or cost-walled. The method (mechanism ⇒ different root ⇒ slow) is
  validated and has now been run to exhaustion on free data.
