# Bottleneck Decomposition Plan — 2026-05-19

## Question (fixed)

Where is the binding constraint on this system — **feature engineering**
(little extractable signal exists), **model** (signal exists but is not
extracted), or **harness** (signal is extracted but destroyed by
selection/gating/cost, or only amplified by meme-convexity not skill)?
Deliver a quantified decomposition with a clear verdict and the single
highest-leverage fix, or an honest "the ceiling is the data."

## Current-system module evaluation (grounded in the R0–R3 + retrain evidence)

**Features** (`features_ml/`, `WINNER_21` = V6_CLEAN−drops + funding +
cross-BTC). PIT-clean (R0: `target_A` recompute 1.1e-5·std, prefix-causal
exactly 0). Known: per-cycle IC ≈ +0.023 (memory), low. Redundancy flagged in
prior Phase H (5 features R²≥0.5 by others). Unknown & to be measured: the
*maximum* cross-sectional IC any learner can extract from this set (the signal
ceiling).

**Model** (LGBM 5-seed, MSE on `target_A` = per-symbol-z, rolling-rank-style
target). **Hard new evidence:** across 225 real fits (R3c), median
`best_iteration = 7`, **20% stop at iteration 1**, RMSE ≈ target-std ⇒ the
model explains almost no pointwise variance. Either the signal isn't there
(feature ceiling) or MSE-on-z-rank-target is the wrong objective
(model/target-framing ceiling). Phase RANK (LambdaRank) was rejected
in-universe (−0.005 IC) but the *ceiling* was never measured.

**Harness** (rolling-IC top-15 [memory: S/N 0.32, noise-dominated] + conv_gate
+ filter_refill + PM_M2 + K=3 + 6-sleeve + cost). Produces in-universe +2.23,
but R1c showed this is a ~5-effective-name vol-convexity bet (Herfindahl
≈0.19) that *rotates* (VVV→AXS→PENDLE), and R3c showed it does not port
(−0.33 unseen). So the harness *amplifies* a tiny signal via convexity rather
than extracting broad skill — but the share attributable to (selection vs
construction vs cost vs convexity vs actual model edge) was never decomposed.

Hypothesis (to be tested, not assumed): **feature/signal is the primary
ceiling; the model is not the limiter (best_iter≈7 because there is little to
fit); the harness's +2.23 is mostly convexity amplification, not skill.**

## Method (pre-registered; oracle ladder)

All on the validated panel (`panel_variants_with_funding.parquet`, R0-clean
`target_A`/`alpha_A`), walk-forward `_multi_oos_splits` (9 OOS folds, embargo,
label-purge), 51-symbol universe. Each test states an absolute numeric
prediction; a miss rewrites the diagnosis, never the gate. Block-bootstrap
(block=11) CIs reported as information. No goalpost-moving.

### B0 — Signal ceiling (is the feature set the limiter?)
- B0.1 Univariate per-feature rank-IC vs `alpha_A`, per fold OOS; report
  max|IC|, mean|IC|, and the IC of a simple equal-weight composite of
  sign-aligned features.
- B0.2 **Best-learner ceiling**: on the SAME folds/features/target, fit (a)
  the production LGBM (baseline), (b) a strong-capacity LGBM (more leaves,
  lower LR, no early-stop cap, n_round high), (c) ridge on standardized
  features, (d) per-cycle cross-sectionally-z-scored-target LGBM. Report OOS
  per-cycle rank-IC of each.
- **Pre-registered:** if the BEST learner's OOS per-cycle IC ≤ +0.04 (≈ within
  noise of the +0.023 baseline), the **feature/signal ceiling binds** — model
  work cannot help. If some learner reaches IC ≥ +0.06 with CI excluding the
  baseline, the **model is a real lever**.

### B1 — Model extraction gap & target-framing (is MSE-on-z-rank the limiter?)
- B1.1 best_iteration / learning-curve diagnostic across folds (quantify
  underfit vs the target's pointwise unfittability).
- B1.2 Target reframings, same features/folds, OOS per-cycle IC AND the
  resulting harness Sharpe (production stack, fixed): {`target_A` z (prod),
  raw 4h `alpha_A`, sign(alpha_A) classification, per-cycle cross-sectional
  rank target, LambdaRank objective}.
- **Pre-registered:** if no reframing lifts OOS per-cycle IC by ≥ +0.015 over
  prod AND harness Sharpe by ≥ +0.3 (paired CI ex-0), the model/target line is
  **not** the binding constraint (confirms B0 if B0 says feature-bound).

### B2 — Harness attribution (oracle ladder; where is Sharpe made/lost?)
Holding model preds fixed, walk the ladder and record Sharpe at each rung
(equal & vol-norm sizing; flat-4.5 & realized-√ADV cost):
1. realized production (rolling-IC + conv_gate + refill + PM + 6-sleeve + cost)
2. + oracle universe (top-15 by *realized* next-window IC) — selection value
3. + oracle picks (rank by *realized* `alpha_A` instead of pred) — signal gap
4. + zero cost — cost drag
5. vol-normalized vs raw — convexity (meme) share
- **Pre-registered:** decompose +2.23 into {model-skill, selection,
  construction, cost, convexity}. If (oracle-picks − realized) ≫ (oracle-univ
  − realized) and ≫ any model lever, the bottleneck is **signal** (perfect
  signal would help a lot; better selection/model within real signal won't).
  If convexity share > 60% of realized Sharpe, the in-universe edge is
  **not skill** (consistent with R1c/R3c).

### B3 — Synthesis & decision
Name the single binding bottleneck with the numbers; give the one
highest-leverage action (e.g., "signal-bound on free 4h data ⇒ only orthogonal
data/horizon helps", or "model-bound ⇒ reframe target", or "harness-bound ⇒
fix selection"). No pre-written verdict.

## Process
Plan reviewed by 3 agents (methodology / profitability-alignment / red-team)
to alignment before B0 runs. After B0–B2, results reviewed by 3 agents vs
these pre-registered gates; any leaky/fudged measurement ⇒ re-initiate that
test. Reuse validated machinery (`phase_ah_sleeve`, `build_audit_panel`,
`R1_baseline_frontier`) to avoid new bugs.

## Out of scope
Re-running the closed portable-alpha conclusion; deploying anything; orthogonal
data acquisition (may be a recommendation, not executed).
