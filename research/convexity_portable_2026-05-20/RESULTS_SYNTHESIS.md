# Convexity-Portable — Terminal Synthesis (2026-05-20)

## Outcome

**Direction CLOSED.** Two independent test configurations of event-conditional
sign-predictability on the broad 110-symbol Binance universe both fail the
pre-registered AUC gate; the marginal non-vol signal that exists is
fold-concentrated and fails LOFO sensitivity.

## Process

1. **Plan written** with universe-disjoint train (40 non-HL syms) + test (70 HL syms) framing
2. **3-agent review**: methodology APPROVE-WITH-FIXES, profitability **REJECT**, red-team APPROVE-WITH-FIXES
3. **Plan re-initiated** as E0 broad-universe falsification (user request — drop HL restriction)
4. **E0 v1 run**: AUC_full 0.5254 (below 0.53 gate), Δ vs vol +0.022 (above 0.015 gate). Mixed.
5. **3-agent result review**: methodology and profitability both ACCEPT-CLOSURE; red-team RE-TEST citing 3 specific pre-reg violations (winsor instead of rank-transform on heavy-tail features; 2 missing panel features; LOFO not run)
6. **E0 v2 run** with red-team's fixes
7. **Outcome**: CLOSE strengthened — AUC fell to 0.5227, worst-LOFO Δ +0.0139 (fails the 0.015 gate), drop-best-2 Δ +0.009 (null)

## Numerical summary

| Test | n_features | heavy-tail preproc | AUC_full | AUC_vol | Δ | worst-LOFO Δ | verdict |
|---|---|---|---|---|---|---|---|
| E0 v1 | 17 | winsor+z | 0.5254 | 0.5038 | +0.022 | — | FAIL AUC |
| E0 v2 | 19 | rank+z | 0.5227 | 0.5041 | +0.019 | +0.014 | FAIL AUC + LOFO |

## What this closes

Adds to the project's ledger of closed directions:

44. Cross-sectional residual-decile convexity detector (C0pre, 2026-05-19) — vol confound
45. OI/flow as portable orthogonal signal (oi_flow_test_v2, 2026-05-19) — underpowered
46. Lifecycle event-path direction prediction at 4h (Probe lifecycle, 2026-05-19) — +0.014 lift only
47. Trade-level feature separation of convex outcomes (Probe #1) — sub null-p95
48. Symbol-class signature predictiveness (Probe #3) — at placebo
49. PnL-mean-reversion as portable mechanism (Probe #6c) — half-of-sample only
50. R2a WINNER_21 + rvol_7d + ret_3d + btc_rvol_7d retrain (2026-05-20) — Δ −1.84 Sharpe
51. **Event-conditional sign-predictability via portable Ridge classifier (E0 v1+v2, 2026-05-20)** — AUC 0.52 with fold-concentrated Δ +0.02

## What's NOT closed

- Path (a) target_A unclip retrain — diagnostic done, full retrain not yet run (~3–6h compute)
- Path (d) operational deployment of V3.1 on 51-panel — engineering, not research
- Paid orthogonal data (deferred until extraction-layer fix proven needed)

## Mechanism finding (cross-cutting)

The combined results of R2a + E0 v1+v2 + Probe loop refine the project's understanding of where alpha lives on this data scope:

- **Magnitude is predictable** (vol clustering, event-detector works) — but magnitude alone has zero sign-predictive value (AUC_vol ≈ 0.50 across all event-conditional tests).
- **Sign is barely predictable** (AUC ~0.52-0.53 in best-case) — but the predictability is concentrated in specific time windows (fold 2 summer 2025, fold 6 Dec-Jan 2025-26, fold 8 Mar-Apr 2026) and doesn't aggregate to a strategy-grade signal.
- **The alpha that drives V3.1's +2.23 Sharpe is event-attribution alpha** — capturing specific pump events on rotation memes (VVV/AXS/PENDLE on 51-panel; SIREN/JELLYJELLY/AVAAI on 110-panel) — not a portable feature-based forecast.
- **Both LGBM and linear models exhibit the same pattern** because both are doing event-memorization, not portable feature extraction.

## Strategic implication

The path-(b) feature-track failure (R2a Δ −1.84) and the E0 dual closure together establish:

**The feature-extraction ceiling on free Binance perp 4h data is reached.** No combination of pre-event features, vol detectors, sign classifiers, or model architectures has produced AUC > 0.54 portably. The cohort-spread-→-tradeability gap (Phase Q + R2a + E0) is structural.

V3.1's +2.23 Sharpe IS the realistic ceiling on this data scope. Forward expectation widens to +1.0 to +2.2 with mean ~+1.5 per memory's universe-overfit analysis. Further alpha requires:
- Different data scope (paid orthogonal, longer horizon, different asset class)
- OR different problem framing (not 4h CS residual prediction)

## Scripts

- `research/convexity_portable_2026-05-20/PLAN.md`
- `research/convexity_portable_2026-05-20/scripts/E0_broad_universe.py`
- `research/convexity_portable_2026-05-20/scripts/E0v2_with_rank_heavy_tail.py`
- `research/convexity_portable_2026-05-20/results/E0_results.json`
- `research/convexity_portable_2026-05-20/results/E0v2_results.json`

## Recommended next step

**Operational deployment of V3.1 on 51-panel (path d).** Ship the +2.23-Sharpe strategy forward via paper-trading + HL execution + kill-switch. This is engineering, not research, but it:
- Validates the strategy in forward time
- Provides a baseline against which future research must beat
- Yields actual P&L data (or honest forward-fail data) to inform whether the universe-overfit caveat matters in practice
- Is needed regardless of whether further research finds incremental alpha
