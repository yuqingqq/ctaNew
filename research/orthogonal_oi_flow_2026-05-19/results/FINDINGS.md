# OI + aggTrade-Flow — FINDINGS (2026-05-19, CORRECTED v2) — LINE CLOSED

3-agent results review of the original run = RE-INITIATE (red-team caught the
B3-class duplicate-`time` cartesian join: ~25× n_eff inflation + cross-group
non-lockstep pairing; PLAN parity guards promised-not-coded). Re-initiated
with `oi_flow_test_v2.py`: heavy compute byte-identical, aggregation fixed
(strict within-group pairing, no cartesian, honest n_eff), parity guards
(per-group level-CI, LOFO sign-flip) implemented, per-group artifacts
persisted.

## Corrected result (oi_flow_results_corrected.json)
- honest n_eff ≈ **715** (vs inflated 3682 — confirms ~5× cartesian bug, fixed).
- All 6 (arm×model) cells: **no-portable-lift, underpowered** — every paired
  diff CI includes 0; mean lifts small (A1_OI lgbm −0.12 / ridge +0.29;
  flow ≈0; OI+flow ≈0); MDE ≫ |observed|.
- The previously-waved-away **Ridge-OI positive** (mean_lift +0.29) now
  properly scrutinized at parity with the bottleneck Ridge-A0 thread:
  **3/5 groups positive, LOFO sign-flips present, level/diff CI include 0**
  → within-noise (Ridge 0-impute/standardization artifact), NOT a portable
  signal. Resolved, not buried.
- Leak-guards genuinely clean (|rankIC| 0.032/0.018; coverage→group AUC
  0.56/0.57 ≈0.5; full 51/51; coverage admissibility 93–98% — not
  null-by-construction).

## Verdict
Free Binance OI + aggTrade-flow as model features produce **no detectable
portable lift** at the directly-measured full-51 model-feature form, across
LGBM and Ridge — OUR validation of (not citation of) the prior Phase-P/§5-INT
negatives. Power-limited (~5-group/0.74y) ⇒ "no-detectable / effect-size",
NOT "proven exhausted". Closes the FREE tier of the orthogonal-data lever in
the B3 menu; does NOT justify paid data (free analogues to positioning/flow
came back null; raises pessimism for the paid prior). Honest, decision-useful
negative. Artifacts: `oi_flow_test_v2.py`, `oi_flow_results_corrected.json`,
`reviews/ROUND2_results_review.md`, `_pergroup_*.parquet`.
