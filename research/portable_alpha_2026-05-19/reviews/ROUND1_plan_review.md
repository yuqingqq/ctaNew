# Plan Review — Round 1 (3 independent agents) — 2026-05-19

Reviewed: `PLAN.md` (v1). Three agents: methodology, profitability-alignment, red-team.

## Verdicts
- **Methodology:** PLAN-NEEDS-REVISION. Top fix: target `target_beta_btc`/`alpha_beta`
  does not exist; only substitute (`btc_target`/`target_A`) is per-symbol-normalized
  *pooled over full universe+period* → out-of-time AND out-of-universe leak that
  defeats T1's premise.
- **Profitability:** NEEDS-REFOCUS. Plan optimizes to *prove non-portability*, not
  find money. Both fail-actions = "stop" → third consecutive defeatist arc.
  Demotes/omits the highest-EV un-refuted levers.
- **Red-team:** DO-NOT-PROCEED (as written). T1 not novel and pre-falsified
  (Phase 1A/1B already +0.27/+0.40 vs required +1.0; no-sym_id already −0.39).
  Vol-norm sizing already rejected (DD_ROOT_CAUSE Test E). Pre-written conclusion.

## Convergent actionable critiques (must all be addressed)
1. **Drop portability-as-kill-gate.** It re-derives closed work (Phase UNI,
   51-vs-111, diag_3way/4_variant, BTC-only model on disk). Demote to a
   *robustness/sizing diagnostic*, not pass/fail.
2. **Do not blindly strip convexity.** Inverse-vol already −0.31 (DD_ROOT_CAUSE
   Test E). Report BOTH a notional-with-hard-concentration-cap variant (keeps the
   engine, controls the risk) and a vol-normalized variant; measure, don't assume.
3. **Center the un-refuted profit levers as PRIMARY work:** (a) longer-effective-
   hold cost-amortization sleeve (equal weights only); (b) `rvol_7d` & `ret_3d`
   as MODEL FEATURES in a retrain (never done — Phase Q used different features).
4. **Reframe to deployable criteria, not academic CI>0.** Curated universe +
   periodic walk-forward refit + concentration cap + kill-switch + cost sweep.
   Judge on pre-registered ABSOLUTE point-Sharpe / fold-breadth / cost-robustness
   / concentration numbers. CI reported as information, never a binary kill
   (it is statistically unachievable on ~1yr of data and would bury real edge).
5. **Fix methodology defects:** name exact panel + add a *recomputing, blocking*
   PIT check on the 3 flagged columns; recompute per-symbol target normalization
   on training-time rows only (per fold); sign-safe concentration (Herfindahl/
   Gini on gross |per-name PnL|) + drop-k stress; PIT-correct trailing vol spec
   (`.shift`, winsor floor); moving-block bootstrap at block = 1 hold + n_boot
   + one-sided LCB + N_eff + power analysis; keep 6/9 fold bar (not 5/9);
   lock ALL free params (panel, target recipe, vol window, seeds, n_boot) in a
   numeric table before any run.
6. **No pre-written conclusion.** Remove the "recommend orthogonal data, stop"
   terminus written before tests run (motivated-reasoning red flag).
7. **Guard the shared-BTC-factor false positive** in any disjoint-symbol eval
   (regress out market beta; report alpha-only).
8. **Confirmed real & must stay in scope regardless:** live `vBTC_paper_bot.py`
   ships K=4 / no-sleeve, NOT the claimed K=3 + V3.1 6-sleeve production stack.

Agent IDs (for targeted re-review): methodology `a8b6fa6404ffbdb04`,
profitability `ae87e5ab124bd1d92`, red-team `a80e855fe0cc9c046`.
