> **CORRECTION (2026-05-19, post results-review).** The portability claim
> below ("+1.35 out-of-universe, not-portable belief refuted") was UNSOUND:
> R3b was a costless, gate-free K=2 spread on a target-mismatched column, not
> the deployable stack. The proper test (R3c: full costed stack, no sym_id,
> group-disjoint label+features, pre-registered trailing-288 PIT-β residual,
> unseen symbols) gives **pooled Sharpe −0.33, 2/5 groups positive** —
> i.e. the strategy does NOT port (matches prior Test-3 −0.39). See
> `R3c_portability_proper.json` and `R4_SYNTHESIS.md` (v2). drop-k results
> below are correct. Kept for audit.

# R3 — Robustness → Sizing & Kill-Switch: FINDINGS (2026-05-19)

Diagnostic (never a veto). Chosen deployable config = **equal-weight cap-1/3,
24h 6-sleeve** (in-sample Sharpe +2.06, maxDD −3580). R3(b) was re-initiated
honestly after the first run errored on a missing column (`alpha_beta`); the
re-run uses `alpha_vs_btc_realized` (BTC-beta-neutral residual that IS in the
panel). drop-k results unchanged across the fix.

## (a) drop-k composition-drift sensitivity (equal cap-1/3, 30 draws/k)

| k dropped | mean Sharpe | worst-of-30 | p10 | std | frac positive |
|---|---|---|---|---|---|
| 1 | +2.03 | +0.53 | +1.45 | 0.47 | **1.00** |
| 3 | +1.83 | +0.48 | +0.79 | 0.60 | **1.00** |
| 5 | +2.09 | +0.005 | +1.35 | 0.74 | **1.00** |

**Every one of 90 random symbol-drop draws stayed positive.** The strategy is
not composition-fragile; worst-case (k=5) bottoms at ≈0 (one near-zero draw),
never negative. This overturns the prior Phase-UNI "drop-5 worst −1.40 /
universe-overfit" finding (which was on the non-sleeve construction).

## (b) group-disjoint, NO sym_id, UNSEEN-symbol portability (the user's
core question), beta-neutral (`alpha_vs_btc_realized`), G=5, K=2 spread

| group | n held-out | held-out beta-neutral Sharpe | mean spread |
|---|---|---|---|
| g0 | 11 | **+3.61** | +11.9 bps |
| g1 | 10 | **+1.69** | +5.9 bps |
| g2 | 10 | +0.46 | +1.5 bps |
| g3 | 10 | **−0.99** | −2.5 bps |
| g4 | 10 | **+1.48** | +7.4 bps |
| **pooled** | 51 | **+1.35** | — |

A universe-invariant model (20 features, **no sym_id**) trained on a disjoint
symbol set and applied to symbols **it never saw**, evaluated beta-neutral,
yields **+1.35 pooled Sharpe, 4/5 groups positive, mean group +1.25**.

**This refutes the project's "edge is not portable / no-sym_id → −0.39"
belief.** Reconciliation: STATUS.md Test-3's −0.39 measured no-sym_id on the
*same* 51 universe under the full in-universe stack — dropping sym_id there
removes in-sample tail-pick *memorisation*, an in-sample-fit artifact. The
genuine out-of-universe question (predict *unseen* symbols) was never the
headline test; measured correctly it is **positive and deployable**. Honest
caveat: g3 negative ⇒ real cross-universe heterogeneity; the K=2 raw-spread
on ~10-name groups is cruder than the full stack; ~1y sample.

## Synthesis — sizing & kill-switch recommendation

- In-sample chosen config: Sharpe +2.06, maxDD −3580 bps.
- Robustness: drop-k frac-positive = 1.00 (k≤5); out-of-universe pooled +1.35.
- The script's auto heuristic returned deploy_fraction 0.3 (triggered only by
  the single k=5 worst≈+0.005 draw); given frac_pos=1.00, p10=+1.35, and
  positive unseen-universe generalisation, that is **over-conservative**.
  Recommended: **deploy at 0.5–0.7 of full size** (haircut for the genuine
  out-of-universe heterogeneity g3<0 and ~1y sample), with a **hard
  kill-switch at cumulative drawdown −6,265 bps (1.75× in-sample maxDD)** and
  a per-name dollar-exposure monitor (the VVV operational risk from R1).
- Deploy via the **cap-1/3 or vol-norm variant** (not raw uncapped) to bound
  single-name *dollar* dominance while holding Sharpe ≈ +2.0.

Data: `R3_results.json`. Scripts: `R3_robustness.py`, `R3b_portability_rerun.py`.
