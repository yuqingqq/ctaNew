# R2 — Profit-Lever Stack: FINDINGS (2026-05-19) — BOTH LEVERS REFUTED

Pre-registered (PLAN.md R2): a lever passes iff it clears deployable criteria
1-6 with lift ≥ +0.3 over R1 (paired CI excluding 0) AND no LOFO single-fold
sign-flip; else refuted, recorded honestly. **Outcome: both refuted.**

## R2a — rvol_7d + ret_3d + btc_rvol_7d as MODEL features  → REFUTED

Clean A/B (identical production harness, +3 PIT features). **Catastrophic:**

| variant | Sharpe | folds+ | lift vs R1 |
|---|---|---|---|
| 24h uncapped flat45 | **+0.39** | 5/9 | **−1.84** |
| 24h cap-1/3 flat45 | +0.46 | 5/9 | −1.77 |
| 24h uncapped flat9 | −0.07 | 5/9 | −2.30 |
| 24h * realized | +0.27–0.34 | 5/9 | −1.9 |

Mechanism: `target_A` is rank-only / barely point-fittable (best_iter≈1, RMSE≈
target-std — same as baseline). `btc_rvol_7d` is **identical across all symbols
at a timestamp**; `rvol_7d`/`ret_3d` are slow regime variables. Injecting them
lets LGBM key on market-regime level instead of the cross-sectional residual
ranking the strategy monetises → ranking corrupted, Sharpe collapses. The
cohort Sharpe-spread +15.77 that motivated this was a *cycle-level* statistic
and explicitly does NOT transfer to a per-symbol model feature (predicted in
the Diagnosis; confirmed). **Refuted: lift ≈ −1.8 (pre-registered bar +0.3).**

## R2b — longer effective hold (equal-weight overlapping sleeves) → REFUTED

| hold | uncapped flat45 | folds+ | lift | realized | tail-stressed |
|---|---|---|---|---|---|
| 24h (ref) | +2.23 | 7/9 | — | **+2.13** | **+1.96** (7/9) |
| 48h | +1.21 | 5/9 | −1.02 | +1.14 | +1.03 |
| 72h | +0.90 | 4/9 | −1.33 | +0.85 | +0.76 |

Longer holds **monotonically hurt** (Sharpe and fold-breadth both degrade);
paired-diff CI excludes 0 negative at 48h/72h. The cost-amortization
hypothesis fails: signal decays faster than cost is saved beyond 24h. 24h was
already the sweet spot. **Refuted: no hold beats R1; 48h/72h significantly
worse.** R2c (R2a preds @ 48h) +0.80–0.90, lift −1.3 — inherits R2a's damage.

## Decisive positive by-product (hardens R1)

At 24h the R1 baseline survives the **realized √ADV cost (+2.13)** and a
**tail-stressed cost that triples the per-unit charge on top-vol-decile legs
(+1.96, 7/9 folds)**. This directly refutes the Round-3 red-team's single
biggest residual risk ("R2b/R1 a cost-amortization artifact of median-
calibrated costs that under-charges the convex tail names"). The edge is NOT
a cost-accounting artifact: it persists when the high-vol names that carry the
PnL are charged 3×.

## Conclusion

No lever improves the already-deployable R1 baseline — the highest-EV
un-refuted directions (rvol/ret-as-model-features; longer-hold cost-
amortization) are now **refuted with clean OOS evidence and pre-registered
gates** (no goalpost-moving; "else refuted" branch fired as written).
**The deliverable remains the R1 system**, now additionally validated against
realistic and tail-stressed execution cost. Per decoupled delivery: ship R1.

Data: `R2_results.json`, `R2_run.log`. Scripts: `R2a_retrain.py`, `R2_eval.py`.
