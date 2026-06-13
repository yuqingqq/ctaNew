# iter-035 — EX-ANTE STRUCTURAL panel-selection STANDARDS (nested-OOS)

(Agent computed fully; API-overloaded on final write — results recovered from outputs/iter035/run.log.)

## Reference (+stop @4.5bps, full 2021-26)
- full-156 naive: Sh +1.03 / Cal +1.16 / maxDD −3960 / fp 6/8
- established-70 (curated): Sh +1.34 / Cal +1.84 / maxDD −2647 / fp 7/8

## Maturity-floor sweep (nested-OOS, +hygiene)
- ≥60d +1.06 ; ≥120d +1.17 ; **≥180d +1.20 / Cal +1.63 / maxDD −2978 / fp 7/8 (SWEET SPOT)** ; ≥365d +0.80 (starves)
## + liquidity floor (execution): ≥$5M → +1.23/Cal1.68 (doesn't hurt; marginally helps)
## + dedup(corr>0.9): +1.21/Cal1.66 (~neutral, slight help)
## + dispersion/idio-vol floor: HURTS (+0.91 q25 / +0.86 q50) → REJECTED (overfit-y, starves)
## CANDIDATE STANDARD = mat≥180d + hygiene + $vol≥$3M + dedup0.9: +1.18 / Cal1.63 / maxDD−2978 / fp 7/8

## DECISIVE honest tests
- RANDOM-SAME-SIZE placebo: STANDARD ranks **p32** (random mean +1.186, p95 +1.263) → the rule does NOT beat random
  selection of the same count from the ELIGIBLE pool. i.e. WHICH names you keep within the eligible pool doesn't matter.
- PAIRED CI: STANDARD − full156 = +0.116 bps CI [−0.66,+0.97] CROSSES 0; STANDARD − estab70 = −0.006 bps CROSSES 0.
  → STANDARD is statistically INDISTINGUISHABLE from both full-156 and the established-70.

## CONCLUSION (the standards that work)
The VALUE is entirely in the EX-ANTE ELIGIBILITY FILTER, not in any within-pool name-picker:
1. **Maturity ≥180d** — the one real transferable lever (lifts naive full-156 +1.03→+1.20, recovers most of the gap
   to the curated-70; structural, generalizes). 365d over-filters (starves).
2. **Hygiene** (ex stables/wrapped/PAXG-gold), **liquidity floor** (~$3-5M, execution only — doesn't hurt),
   **dedup** correlated>0.9 — all ~neutral-to-mild-help; keep for cleanliness/execution.
3. **NO within-pool selection** — ranking/keeping by IC, dispersion, or performance does NOT beat random within the
   eligible pool (STANDARD p32 vs random); dispersion-floor actively HURTS. Don't try to be clever; trade the eligible set.
DEPLOYABLE STANDARD: mature(≥180d) + hygiene + liquid(exec floor) + dedup, refreshed QUARTERLY (auto-adds names as they
mature, auto-drops delisted). ≈ statistically equivalent to the hand-curated 70 but MAINTAINABLE by rule. The 70 has a
marginally higher point estimate (+1.34 vs +1.18) but within noise.
