# v4 limitations — clean-data re-evaluation (2026-07-10, AUTHORITATIVE)

Authoritative statement of v4's limitations on the **clean, deployed** data, replacing the earlier
(leaked-panel / half-cost) diagnosis. Every number here has a **committed generator** and is at the
pinned **0.5×9-bps** cost. Audit trail (how we got here — the label gap-leak, the 2× cost undercharge,
the ERT1 wrong-model, the promotion to production): RESEARCH_LOOP_20260707.md addenda 24–33 + git
history. Generators: `attribution_v4_regime.py`, `limitation5_concentration.py`, `reevaluate_clean.py`,
`reeval_stability.py`.

## Status (what is true right now)

- **Deployed:** the live pipeline runs on the **clean panel** (317 gap-corrupted labels fixed at source
  in X70 + promoted to the canonical panel), **per-symbol RidgeCV** models retrained on it, clean
  bootstrap books/seeds/state. A latent deep-bull empty-group crash in `convexity_paper_bot` was fixed.
- **Verified:** in **backtest / replay** — clean-vs-leaked A/B, committed generators, reviewer-closed.
  Clean is measurably **better** than leaked (recent 2.30 vs 2.26, better maxDD).
- **NOT yet verified:** **live-forward.** No forward trading on the clean models has occurred; the
  forward ledger is the standing gap. This is *backtest-verified, not live-proven*.
- **Forward expectation is WIDE** — the 2.30↔0.71 universe-meta swing showed real universe-composition
  sensitivity; do not read +2.30 as a tight point estimate.

## Performance (clean, deployed)

| frame | full-stack Sharpe | maxDD |
|---|---|---|
| **recent 2025-10+** (deployment case) | **+2.30** (deployed; +2.41 confirmatory) | −11,132 |
| **OOS 2023-25** (guardrail) | **+0.20** (better than leaked −0.09) | — |

Per-regime book (residual Sharpe / net bps, pinned cost):

| regime | RECENT Sh / net | OOS Sh / net |
|---|---|---|
| side | +3.55 / +14,432 | −0.78 / −8,621 |
| bear | −1.29 / −3,950 | **+1.82 / +3,802** |
| bull (mild) | +3.95 / +4,571 (GATED→flat) | −2.37 / −8,288 |
| deep-bull | −4.35 / −1,457 | −0.85 / −5,231 |

## The strategy in one sentence

v4 is a regime-switched, ~beta-neutral cross-sectional book whose edge is a **collection of
era-specific, event-concentrated regime edges** — no single regime is reliable across eras, and each
regime's positive net comes from a handful of dispersion months.

## The definitive limitations (all committed-verified on clean)

### #1 — Era-fragility (DEEPEST). The two main regimes have OPPOSITE era signs.
- **side** pays RECENT (+3.55) but LOSES OOS (−0.78); **bear** pays OOS (+1.82) but LOSES recent
  (−1.29). Opposite signs → **in any given era, one main regime works and the other doesn't; neither is
  reliable across eras.** No regime is both-era positive.
- Mechanism: v4 farms cross-sectional dispersion, whose *sign/regime* rotates with the market era; the
  model cannot know which era it is in. This is the driver of the 2022 holdout FAIL and the 0.5× gross
  cap.
- **Fixability: structural, LOW.** Modeling axis exhausted (~20 cells, 0 promotions); era-robust
  training FAILED honest gates (addendum 23z). Cross-asset diversification works (DIV1 +33%) but is
  OUT of the crypto-only scope; a within-crypto trend sleeve is not robustly deployable (DIV2/SWITCH1
  FAILED). → **managed** (cap + monitoring + kill-switch), not solved.

### #2 — Bull gate is a deliberate era-REFUSAL, not a defect.
- Mild-bull is +3.95 recent (mean +40/cyc, 80% positive-months) but **gated to flat** (BULL_GROSS_MULT
  =0). Because mild-bull is −2.37 OOS, gating **refuses the era bet** — exactly the #1 discipline.
  Forfeiting the recent +4,571 is the correct conservative choice, not a lost edge.

### #3 — Deep-bull is a directional beta lottery.
- The beta-neutral MODEL book LOSES in deep-bull both eras (−4.35 rec / −0.85 OOS), so production runs
  a `mom1d_long` overlay (long top-2 by return_1d, ½ gross). That overlay **earns via generic long-alt
  BETA, not selection** (Q3 ranking p=0.215; §6.1). return_1d is a feature (label-fix-independent), and
  deep-bull n≈47 is tiny. → a high-variance directional bet a beta-neutral strategy holds; era-neutral
  (bad-to-flat both eras), so simplifying it is not an era bet.

### #4 — Bear squeeze tail, UNHEDGED — and the label leak had been MASKING it.
- The short leg (bottom-K) can be squeezed: bear short-leg PnL is **left-skewed −1.02 OOS** (median
  +6.7 ≫ mean −9.8, CVaR5 −715). **The label leak had hidden this**: leaked OOS skew read +3.06 (falsely
  RIGHT-skewed) because the corrupt 2025-02-28 "short-a-22-day-decline" cycle was a +15% outlier. Clean
  data reveals the true left tail (recent skew −1.67 was never masked). The tail is **unhedged**: SQ1
  showed crowding predicts squeezes but is non-stationary on free funding (SK1 failed the recent
  holdout). → **DATA1 (paid liquidation/positioning data) is the one lever with real upside** for
  hedging this now-visible risk.

### #5 — Thin, event-concentrated alpha — across EVERY regime, not just side.
- Every regime's positive net is **~50–80% concentrated in its top-2 months**: recent side 73% (2026-04,
  2025-10), bear 58–80%, bull 72%. The edge is a **handful of dispersion events, not a steady stream** —
  so realized performance over any short window is dominated by whether a dispersion event landed. This
  is a signal/data limit (free 4h Binance perp), unfixable by config.

### #6 — The regime label is a 30-day LAGGING classifier (compounds #1).
- Every switch keys off `btc_ret_30d` + 3-cycle entry hysteresis + 6-sleeve/24h settle, so "which regime
  am I in" is a lagging guess — worst in fast transitions (a 2022-style cascade, a sharp flip), where
  the book can apply the wrong era's rules. Structural / label-fix-independent; correctly handled as
  conservatively as a lagging estimator allows.

## What can move these (honest map)

| limitation | binding? | the only lever with upside |
|---|---|---|
| #1 era-fragility | YES (deepest) | none within crypto-only modeling — **manage** (cap/monitor/kill-switch) |
| #5 thin/concentrated | YES | none (free-data ceiling) — accept + size for long flat stretches |
| #4 squeeze tail | YES (now visible) | **DATA1** — paid liquidation/positioning data (a real, paid decision) |
| #2 bull gate | no (deliberate) | keep gating (refuses the era bet) |
| #3 deep-bull lottery | no (era-neutral) | KEEP/DROP the mom1d beta bet — small either way |
| #6 lagging regime | structural | none (any faster estimator is adaptive-timing, failed 8×) |

## The bottom line

v4 is a **real but thin, era-fragile, event-concentrated** edge — now on **clean data with an honest
risk picture** (the #4 squeeze tail is no longer masked). Deployed recent backtest **+2.30** (forward
expectation WIDE); OOS a thin-positive guardrail (+0.20). The **binding constraints are structural**
(#1 era-fragility, #5 thin/concentrated alpha) — they are **managed, not solved**, because the alpha
ceiling is set by free 4h Binance-perp data on a single asset class, and every modeling/construction/
cost/architecture lever has been honestly exhausted. The **one lever with genuine upside is DATA1**
(paid liquidation data → the now-visible short-side squeeze tail, #4). Everything else is **operational**
(the forward ledger, to convert the backtest into live evidence and eventually release the conservative
0.5× cap on the §7 criteria). **No strategy change is warranted; v4 sits at its free-data local optimum.**
