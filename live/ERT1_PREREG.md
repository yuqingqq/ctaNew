# ERT1 — era-robust training: PRE-REGISTRATION (2026-07-10)

Binding pre-registration for retraining v4 with era-balanced sample weights, to attack limitation #1
(era-fragility): every regime's edge is era-specific — side +4.09 recent / −0.04 OOS, bear −0.88
recent / +4.46 OOS, mild-bull +4.36 recent / −1.47 OOS. Mechanism: the pooled model overfits whichever
regime temporally DOMINATES its training window, so it works in that era and fails in others. This is
the driver of the 2022-holdout FAIL and the 0.5× gross cap. Hypothesis: forcing each regime to
contribute equally to the training loss yields a more era-robust model (bad eras less bad) without
gutting the aggregate edge.

## Two risks named up front (they shape the gates)
1. **The 2022 holdout is SPENT** (one-shot, already used). Era-robustness' truest test is exactly the
   era we cannot re-gate on. So the BINDING gate is 2023-26 walk-forward FOLD-CONSISTENCY (a weaker but
   honest proxy); 2022 is reported DESCRIPTIVELY only, never tuned or gated on.
2. **Balancing can just FLATTEN predictions** (regress to mean — cf. the target-clip that broke the
   111-panel). Guarded by a no-degradation SECONDARY gate + a shuffled-regime PLACEBO (the SWITCH1
   discipline): if random-regime weights help as much, the effect is weight-noise regularization, not
   era-robustness.

## PINNED spec (binding — everything except the sample weights is v4-identical, per CLAUDE.md)
- **balancing axis:** btc_ret_30d regime bucket {bear <−0.10, side −0.10..0.10, bull 0.10..0.15,
  deep-bull ≥0.15}. **sample_weight(row) = 1 / freq(row's regime bucket in THIS training window)**,
  normalized to mean 1 (PIT — frequencies computed within each expanding train window only). Injected
  as `weight=` on the training `lgb.Dataset` in `_train`. Equalizes each regime's loss contribution.
- **everything else PINNED to production v4:** V0_LEAN 14 features, residual target xs_z(alpha_vs_btc),
  HORIZON=48, 5-seed ensemble (42,7,123,99,314), autocorr training filter (REGIME_CUTOFF=0.50), LGBM
  hyperparameters, expanding walk-forward with embargo + label purge. No other change. Single pinned
  balancing scheme — NO sweep of weight schemes/exponents (W1b).
- **measurement = BOOK LEVEL** (estimator law — NOT overlay replays, which path-couple and amplify
  noise): per-fold book rank-IC (Spearman pred vs residual-alpha target on the full cross-section) +
  aggregate top/bot-20% selection spread (the traded edge). Baseline (production weights=uniform) vs
  ERT (era-balanced), SAME harness, SAME folds.

## Pre-committed gates (2023-26 walk-forward; 2022 descriptive only)
**VERDICT METRIC = TRADED top/bot-20% SELECTION SPREAD, not rank-IC (review 984619c).** rank-IC has
failed to convert to tip value 4× (W1/M1/pooled-Ridge) and §5 established era-fragility is TAIL/tip-
driven, which full-cross-section rank-IC does not capture. So every improvement gate is on the traded
selection spread; rank-IC is reported as a DIAGNOSTIC only, never gated.
- **GATE E-1 (era-robustness — PRIMARY, on the TRADED edge).** ERT's BOTTOM-QUARTILE-fold top/bot-20%
  selection SPREAD (the traded edge in the "bad eras") improves vs baseline. Does the TRADED edge
  improve in bad folds — not just the ordering. If the worst folds' spread doesn't improve, ERT did not
  reduce #1 → FAIL. (Also report bottom-fold traded-book tail: bot-20% CVaR5, since #1 is tail-driven.)
- **GATE E-2 (no edge degradation — the flatten kill).** ERT MEAN top/bot-20% selection spread ≥
  baseline − 5% relative. If balancing flattens the average traded edge to buy bad-fold robustness,
  FAIL.
- **GATE E-3 (PLACEBO — decisive vs weight-noise, on the SPREAD metric).** Re-run with sample weights
  from SHUFFLED regime labels (same weight distribution, regime→row assignment permuted), N≥20
  shuffles. Real ERT's bottom-fold SPREAD gain must beat the p90 of the shuffled-regime placebos. If
  random-regime balancing lifts the bad-fold spread as much → regime info inert, effect is mere
  regularization → FAIL.
- **CONCENTRATION:** report per-fold Δspread (baseline→ERT); the E-1 gain must not be one-fold-driven.
  NOTE: bottom-quartile ≈ 2-3 of the 2023-26 folds (thin sample) — the concentration check + E-3
  placebo are what keep a 2-3-fold gain from being noise.
- **2022 DESCRIPTIVE (not gated — spent holdout):** report both models' 2022 book spread / per-regime
  net as corroboration only. A 2022 improvement is suggestive, never a pass.

## Honest ceiling (stated before running)
- Prior **LOW** (sharpened from low-moderate, review 984619c). MECHANISM MISMATCH: inverse-regime-freq
  weighting fits the rare-regime conditional MEAN/ordering — which the model already gets roughly right
  — whereas #1's binding failures are TAIL/cost-driven (§5: the MR mean holds in every regime; §1: the
  2022 failure was cost-domination / dispersion-collapse, NOT a training-imbalance overfit). So ERT1
  attacks the mean while #1 bites at the tails → even a 2023-26 fold-robustness PASS would NOT imply the
  2022-TYPE mechanism is addressed.
- The truest test (2022) is spent, so a PASS = "more fold-robust TRADED edge in 2023-26 without losing
  average edge," a WEAKER claim than "fixes era-fragility." Era-balancing may still just trade breadth
  for a flatter, safer-but-thinner model.
- PASS (E-1 ∧ E-2 ∧ E-3) = era-robust training carries real, non-random, non-concentrated robustness
  value → candidate for a full-stack replay + forward ledger (NOT an immediate production swap; the
  pinned-hyperparam fairness means any adopt still needs the overlay-level replay discipline).
- Any gate FAIL → not adopted; recorded negative-space result on the deepest limitation.

## Discipline
- W1b: no sweep of the balancing scheme to rescue a fail; single pinned inverse-frequency weighting.
- Book-level metrics only (estimator law); placebo + concentration mandatory (session lessons).
- Script: live/ert1_era_robust.py. AWAITING REVIEW before running.
