# Plan Review — Round 2 (re-review of v2) — 2026-05-19

Verdicts: Methodology **NEEDS-MINOR-FIXES** · Profitability **NEEDS-MINOR-REFOCUS**
· Red-team **PROCEED-WITH-CHANGES**. All Round-1 critiques resolved; v2 introduced
a new critical methodology flaw and the reviewers converge on 6 concrete fixes.

## Convergent mandatory fixes (all to be folded into v3)

1. **[CRITICAL — methodology N1] Drop the spurious per-fold target renorm.**
   `target_A` (`make_xs_alpha_labels`: `expanding/rolling .mean()/.std()
   .shift(horizon)`) is ALREADY PIT-correct per-symbol trailing. v2's "leak
   fix" fixes a non-existent leak and *degrades* the target. R0 must instead
   VERIFY the panel's `target_A` equals a fresh causal recompute of that recipe
   (≤1e-4·std), not introduce a new normalization.

2. **[KILLER — red-team F2 + profitability crit-4] The hard 1/3 cap is
   inverse-vol in a costume.** meme_mechanism (hit-rate 47.9% < coin-flip; PnL =
   pure vol-convexity) + DD_ROOT_CAUSE Test E (−0.31) prove any symmetric
   high-vol-tail truncation kills the engine. Making capped-Herfindahl≤0.25 a
   *conjunctive pass gate* is a rigged known-negative. Replace with a
   **risk-budget cap sweep** {1/2, 1/3, 1/5, 1/8 of book}; deliver the
   **Sharpe-vs-cap frontier**; deployable = "∃ a cap level with Sharpe ≥ +0.8 &
   acceptable DD" + an explicit cost-of-cap retention number (capped ≥ ~70% of
   uncapped).

3. **[red-team F4] Add Leave-One-Fold-Out (LOFO) to R2 gates.** The diagnostic
   that caught the last identical false-positive (Phase Q fold-6). Pre-registered
   kill: if removing any single fold flips the lift sign → reject the lever,
   regardless of aggregate placebo rank. Random drop-k is NOT a substitute.

4. **[red-team F3 — needs USER decision] R2b's 48h/72h holds cross the user's
   fixed-4h scope** (PROGRESS.md: longer horizon = out-of-current-scope, requires
   user re-authorization). 24h is already the baseline; 12-24h equal-weight
   already validated in `phase_ah_v3_robustness.py`. Only 48h/72h is new — and
   it is a scope change. Resolve via explicit user authorization or cut it.

5. **[red-team F5 / methodology N4] De-rig the Diagnosis premise.** Carry BOTH
   ex-VVV numbers — +2.57 (adaptive-refill) AND +1.63/+1.87 (fixed-universe) — as
   an unresolved range, not the favorable one as fact.

6. **[methodology N2/N3, profitability] Pre-register R2a feature builders**
   (per-symbol `rvol_7d`/`ret_3d`/`btc_rvol_7d`: source bars, window, `.shift(1)`,
   winsor); re-anchor R2a's prediction to feature-IC, not the cycle-level +15.77
   cohort spread. Widen bootstrap block to `ceil(hold/4h)+(n_sleeves−1)`. Add a
   realized-execution (inverse-ADV / spread-weighted) cost variant, not just flat
   bps. Decouple delivery: an R1 capped-frontier pass authorizes deployment-
   hardening in parallel with R2.

Two fixes (#4 scope, and the goal-tradeoff behind #2) require a user decision
before v3 is finalized. Round-2 agent IDs: methodology `adfd80729421253c0`,
profitability `aa08bb532ac9e4a7a`, red-team `ad7068e5cd3f50828`.
