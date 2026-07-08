# Experiment plan: retarget the model to the residual (remove beta from the training label)

## Hypothesis
The two-book model trains on `xs_z(return_pct)` — cross-sectional z of **raw** return, which still contains BTC
beta — while the strategy farms the **beta-removed residual** (`alpha_vs_btc_realized`). The training label is
mis-aligned with the objective. Retargeting the label to `xs_z(alpha_vs_btc_realized)` should predict the residual
better and possibly lift strategy PnL.

Fast-check evidence (Path A, done): pooled WF ridge baseline IC vs forward residual alpha rose **+0.0027 ALL**,
**+0.0015 bull / +0.0029 side / +0.0027 bear** — small but **regime-consistent** (not one-regime noise).

## The exact change (one line, nothing else)
In the WF pred generators, training label:
`PAN["xs_z"] = z(return_pct)`  →  `PAN["xs_z"] = z(alpha_vs_btc_realized)`
Hold everything else identical: V0_LEAN features, per-symbol RidgeCV, HL=60, walk-forward cuts, two books
(base + residrev), per-symbol rstd normalization, execution.

## STEP 0 — confound checks BEFORE trusting any result (mandatory)
1. **PIT-cleanliness of `alpha_vs_btc_realized`.** The residual is built from a beta estimate; if that beta uses
   full-sample or forward data, the label leaks and the +0.003 IC (and any strategy lift) is fake — the same class
   of bug as the Sharpe-21 look-ahead. Verify the beta behind the residual is strictly trailing/PIT before Step 1.
2. **Normalization scale.** Residual has a different scale than raw return; confirm the per-symbol rstd machinery is
   applied to the new label consistently (rank-only z, per-symbol std) — memory says this is essential for
   cross-symbol learnability.
3. **Circularity caveat.** The fast IC was measured vs forward residual alpha while training on residual — some
   alignment is mechanical. The decision rests on strategy PnL (Step 2), which is NOT circular.

## STEP 1 — pred-quality confirm in the ACTUAL model (fast, ~4 min)
Retrain the base book both ways (per-symbol RidgeCV, not pooled) and confirm the +0.003 residual-IC holds in the
real model (pooled was a proxy). Reject here if it vanishes under per-symbol fitting.

## STEP 2 — strategy confirm (Path C, ~10 min)
Regenerate both books on the residual label; run frozen v3; compare to the raw-label baseline, **row-matched**:
- Metrics: net daily Sharpe (Δ), **by-regime PnL**, maxDD, per-fold / per-block.
- Baseline = current model (`xs_z(return)`) through the same frozen v3.

## STEP 3 — OOS (Path D)
Repeat Steps 1–2 on fullhist (2022–2026) before any production change.

## Decision criteria (adopt only if ALL hold)
- Step 0 clean (no label leak).
- Net Sharpe ≥ baseline (a small + or genuinely neutral is acceptable — see below), **maxDD not worse**, **no regime
  harmed**, per-fold not concentrated (≥ ~6/9 or block-stable).
- OOS-stable.

## Expected outcome & why it's worth doing anyway
- Realistic expectation: **small or neutral** Sharpe effect — the IC↔PnL gap (K=1/2 selection) that killed the
  features applies here too; +0.003 IC is small.
- BUT this is a **structural alignment fix, not a bolted-on feature**: training the model on the exact quantity it
  is meant to farm is correct on principle and reduces model risk. If PnL-neutral, keeping the residual label is
  still the more defensible design. It is a discrete architecture change (swap the label) → generalizes better than
  tuned params, low overfit risk.
- This is the ONLY model-side lead from the whole alpha thread; features (I1) are settled-rejected.

## Scope / cost
- Base book first (short ranker) for Step 1; both books for Step 2. ~2 WF retrains + 2 driver runs, memory-capped
  (ulimit -v, single-thread) per the OOM discipline.
- Separate from the K_short=3 lead (independent; can be tested before/after).

## Open question for the user
Confirm: run Step 0 (beta-PIT check) + Step 1 (per-symbol IC confirm) first, and only proceed to Step 2 if both
pass? Or go straight to Step 2 strategy backtest?
