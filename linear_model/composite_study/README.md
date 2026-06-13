# Composite Study — pure per-composite effect

Dedicated workspace (created 2026-05-18) to isolate **what each composite
does on its own**, separate from the consensus/voting machinery (Step 102
collapsed/cancelled; Step 103 forensics). Question: *when a composite fires,
what is the forward price performance, in the direction it bets?*

## Pre-registration (LOCKED, descriptive — NOT a gated strategy)

- **Scaling bug fixed:** all magnitude gates use the per-cycle
  cross-sectional z of `s_t` (`sz`) or already-z features (funding_z,
  oi_z, vol_z, obv_z) or scale-free signs. (Steps 102/103 used Z=1 on
  raw-scale `s_t` → P1/V4/R1/X1 fired ~0%; fixed here so they fire.)
- **All 14 composites reported** (no cherry-pick). Each emits a signed
  direction on the β-residual; we measure forward performance *in that
  direction* when it fires.
- **Metrics per composite:** fire-rate; mean signed fwd-4h β-residual
  (`alpha_beta`, the strategy target) + block-bootstrap CI; hit-rate;
  net-of-cost; per-fold sign consistency (k/9); mean signed fwd-24h **raw**
  return (the longer "price performance" view, beta-laden — labelled);
  and a **matched random-timing placebo** (random firing at the same rate,
  same direction rule) → percentile of the real result. That placebo IS
  the "pure composite effect" test: does the composite's *state/timing*
  beat conditioning on nothing?
- **Honest status:** descriptive event study, in-sample. Any composite
  that looks strong here is NOT a finding — per the arc (Step 88 −1.05,
  Step 90 −2.60 below random, Step 102 |V| ρ −0.60) such in-sample
  standouts fail the loop-closed nested-OOS + matched-placebo gate. A
  standout would have to clear that pre-registered gate (separate run)
  before it is anything. No strategy adopted. Production LGBM unaffected.

Script: `per_composite_event_study.py` → `results/`.
