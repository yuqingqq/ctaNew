# Bottleneck Decomposition — B3 Synthesis (2026-05-19, constrained per
# 3-agent results review)

> Wording constrained exactly as the methodology / profitability / red-team
> results-review mandated. The underlying tests are SOUND (methodology
> verified: B★ A0 reproduces R3c byte-exact; lockstep paired-Δ is pure
> feature effect; joint OLS R²≈0.002 → no hidden multivariate leak; no gate
> fudged). Only the *claims* are constrained — this is "no detectable
> fundable lever," NOT "proven ceiling."

## Question
Is the binding constraint **features**, **model**, or **harness**?

## What the measurements show (each is OUR run, not a doc citation)

1. **Model-free signal floor.** Every leak-free non-WINNER panel column:
   univariate |rankIC| ≤ **0.036** (mean 0.014); joint OLS of all 39 on
   `target_A` R² ≈ **0.002**. The *tested on-disk feature space* carries
   essentially no linear or low-order cross-sectional signal — and this is
   model-free, so not an LGBM artifact.

2. **Feature superset → no detectable portable lift, across 3 model
   classes.** Adding the 39 leak-free cols to WINNER_21, paired-Δ through the
   R3c portable protocol (group-disjoint, no sym_id, beta-neutral, costed,
   unseen symbols):
   - production LGBM Δ **−0.58**, Ridge Δ **−0.89**, high-cap LGBM Δ **−0.90**.
   - **All three paired block-bootstrap CIs include 0.** Honest reading:
     **no detectable portable feature lever among the tested columns** — NOT
     "features proven worthless / feature ceiling proven."
   - **Confound (stated, not hidden):** the uniform-negative Δ is consistent
     with dumping 39 near-zero-IC columns at once (dimensionality/variance
     inflation). "More of *these* features hurt" ≠ "better features are
     impossible." A one-at-a-time / IC-ranked-block test was not done; the
     claim is bounded to *this on-disk column set*.

3. **The result is power-limited.** Portability rests on ~**5 disjoint-symbol
   groups over one ~0.74-yr window** (R3c: 2/5 groups positive, mean −0.36,
   std 0.72 → cannot reject zero portable Sharpe either). The n_eff≈3682 /
   MDE≈3.7 bps figure in the JSON is overstated ~25× by a duplicate-timestamp
   join and must NOT be quoted as precision; the honest statement is "one
   ~0.74-yr / ~5-group window — not resolvable." This cuts conservatively
   (strengthens "not proven," creates no false positive).

4. **Model is not the lever, but the portable *level* is strongly
   model-dependent.** Closed RANK/SEG/CAL (IC ∓0.005) + B★/B★b: no learner
   surfaces a portable feature lift. BUT the A0 (WINNER_21) portable *level*
   swings **+0.56 (Ridge) / −0.33 (prod-LGBM) / −0.48 (high-cap LGBM)** —
   a 1.04-Sharpe spread across model class on the *same* features. So the
   correct claim is narrow: *"adding the tested features does not help under
   any learner"* — NOT "the result is model-independent" (the level is
   not).
   - **Ridge A0 +0.56 thread — surfaced and resolved by direct check
     (`ridge_a0_check.json`), not dismissed by assertion:** pooled +0.56 but
     **3/5 groups positive** (g2 −0.33, g4 −0.31), **level CI [−0.50, +1.55]
     includes 0**, corroborating top-K rank readout **−0.71** (negative).
     ⇒ within-noise/indeterminate (most likely Ridge 0-impute shrinkage +
     per-fold standardization), **not a validated portable signal** and not
     a validated null — consistent with the arc's power limit. Not a fundable
     thread on this evidence.

5. **Harness manufactures a non-portable in-universe number.** R1c: the
   in-universe +2.23 is a ~5-effective-name (Herfindahl 0.19) vol-convexity
   bet that *rotates* VVV→AXS→PENDLE under filter_refill (Sharpe decays
   +2.06→+1.89→+1.15). R3c: the full deployable stack on unseen symbols =
   **−0.33 pooled, 2/5 groups** (independently reproduces the prior Test-3
   −0.39). Decisive negative on portability; the IC-selector is known
   noise-dominated (S/N 0.32) so the ≈+0.4 in-universe "drop-selector" lever
   has **0 portable prize**.

## B3 — sized, portability-gated lever menu (the deliverable)

| lever | in-universe Δ | **ports?** | portable prize | basis |
|---|---|---|---|---|
| model / target reframe | ≈0 | n/a | **0** | closed RANK/SEG/CAL (cited) |
| harness: drop IC-selector | ≈+0.4 | **no** | **0** | selector S/N 0.32; R3c |
| feature engineering (tested on-disk cols) | superset Δ<0 | **no detectable** | **0 (not proven; power-limited)** | B★/B★b + IC scan (this arc) |
| **orthogonal data (OI / aggTrade-flow)** | unknown | **being measured** | **GATED — see below** | `orthogonal_oi_flow_2026-05-19` (in flight) |
| paid/alt data (on-chain, options) | unknown | un-refuted | bracket: 0 → (oracle gap), justified only if cohort spread > 11 | reconciled |

**Orthogonal-data row is a gated branch keyed to the in-flight OI/flow arc
(Stage-0 running, full run pending the now-complete 26/26 flow fetch):**
- if full-51 OI/flow portable paired Δ ≥ **+0.5** with CI excluding 0 →
  **fund the orthogonal-data line** (the doc "closures" were stale; we
  measured otherwise).
- if ≤ **+0.2** or CI includes 0 → free-data orthogonal lever is **measured**
  (not cited) as no-detectable-lever; the only money-positive options left
  are paid data clearing the >11 cohort-spread bar, OR the niche Option-B
  meme-convexity bet with the −6,265 bps kill-switch. **Do not deploy the
  current stack; reconcile/retire the K=4 paper bot first** (it ships neither
  the research stack nor anything validated).

## Honest verdict (the exact wording the review permits)

The in-scope **feature / model / harness** space shows **no detectable
*portable* prize**: the deployable stack does not port to unseen symbols
(R3c −0.33, decisive); the tested on-disk feature superset adds no detectable
portable lift under three model classes (all CIs include 0, power-limited,
dimensionality-confounded — *not* a proven ceiling); model reframings and the
IC-selector are closed/zero-portable. The single un-refuted lever is
**orthogonal data**, which is being *measured* right now (OI/flow arc) rather
than asserted from docs. No "stop," no proven ceiling, no model-independence
claim — a measured, power-limited negative on the tested in-scope space, with
the one live lever under active validation.

Artifacts: `B_prefeature_ic.txt`, `B_star_results.json`,
`B_star_b_modelclass.json`, `ridge_a0_check.json`; reconciled
`R1c_concentration_truth.json`, `R3c_portability_proper.json`. Scripts:
`B_star.py`, `B_star_b_modelclass.py`, `ridge_a0_check.py`.
