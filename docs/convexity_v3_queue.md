# Convexity v3 optimization queue (from parallel module-review workflow, 2026-06-05)

8-agent parallel review of every module (features, model/target, regime, selection/sizing, risk/hold, universe,
cost/exec) against the full ledger. ~50 of 60+ proposed candidates dropped as dupes of rejected work.

## Honest assessment up front
**The genuine v3 alpha space is nearly exhausted.** Per-symbol coefficients are the entire edge (iter5–7 closed the
architecture thread); resid_rev is the sole orthogonal signal on free Binance data (every other family killed);
composition variance (placebo p83) is cross-sectional and irreducible by any construction lever. The v2 loop already
harvested the structural wins (equal-weight, stop-off-bear, bearK2, inv-vol). What remains = a few **discrete,
structurally-motivated levers in the regime/risk/cost layers not yet tested per-regime/per-symbol-cost**, plus
operational validation. No remaining feature-layer or model-architecture upside.

## Survivors (ranked by EV × (1 − overfit_risk))
| # | candidate | module | env/retrain | rationale | overfit | prio |
|---|---|---|---|---|---|---|
| 1 | **Wide-spread name down-weight/exclude** | cost_exec | small bot edit | #175 flagged ~7% of low-vol names with >9bps RT spread as exec hazards (resid_rev's quoted-spread fill breaks there) — recommended operationally but NEVER backtested as a lever. Discrete, low-overfit, cheapest likely real win. | low | **P1** |
| 2 | **Regime-specific hold (bear=12h, side/bull=24h)** | regime/risk | ~10-line edit | Audit #19 + FROZEN say "bear may prefer 12h" — never tested per-regime (24h sweep was global). Bear is live now. Discrete (3 fixed values). | low–med | **P2** |
| 3 | **Asymmetric hysteresis (N_bull/N_bear split)** | regime | tiny edit | Batch5 swept only a single symmetric N. Fast-in-bull / slow-out-bear is structurally distinct, low DOF. | low–med | **P3** |
| 4 | **Regime-flip transition de-gross (1–2 bar)** | regime/risk | env-adjacent | Smooths boundary whipsaw/turnover. Discrete ramp, cost-saving not alpha. | low | **P4** |
| 5 | **Regime-aware per-symbol Ridge (bull/side/bear coefs)** | model | **retrain** | The one architecture variant iter5–7 didn't close — stays per-symbol (no pooled-collapse) but lets coefs differ by regime (bull β−0.22 mom vs side/bear mean-rev). | **high** (bear ~1/3 data → noisy) | P5 |
| 6 | **Sleeve-maturity decay weighting** | risk_hold | small edit | hold3<hold6 implies fresher sleeves carry more; exp age-decay is the continuous analog, untested. 1 param. | low–med | P6 |
| — | **Matched-composition placebo (GATE, not a lever)** | universe | offline | The honest decision gate (#173/#178) — every P1–P6 result must beat placebo p95, not just baseline (p83 composition variance). | n/a | gate |

## Test methods (existing harness)
- **P1**: per-symbol rolling-20d RT spread from convexity_slippage; tag >p93 (~7% from #175); book-B membership excludes/half-weights tagged vs baseline; gate per-fold ≥6/9 + placebo p95 + confirm not dropping high-alpha names.
- **P2**: bot edit STRAT_HOLD_{BEAR,SIDE,BULL}, select by effective regime in sleeve agg; nested-OOS hold selection (mirage filter).
- **P3**: split apply_hysteresis into N_bull/N_bear; grid {1,2,3}×{3,4,5}; reject any cell >60% one-fold.
- **P4**: gross *= {0.5,0.75} for 1–2 bars on regime change; neutral-or-better Sharpe + lower switch turnover.
- **P5**: new gen_regime_ridge_wf_preds.py, fit per-(sym,regime) with higher α floor; mandatory nested-OOS + bear-fold isolation.
- **P6**: weight sleeve by exp(−age/τ); τ∈{2,3,4}; adopt only if >+0.05 Sharpe AND survives nested-OOS τ.

## Bottom line
**P1 is the single most likely real win** (flagged-but-untested, env-runnable, low overfit). P2–P4 are honest-but-modest
bear/transition refinements (the one layer still producing wins) — expect small Sharpe/DD gains, mirage-prone → nested-OOS
+ p95-placebo mandatory. P5 is the only swing-for-the-fences, with a strong prior *against*. Everything else proposed = re-tread.
