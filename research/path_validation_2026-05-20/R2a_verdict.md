# Path (b) verdict — REJECTED (2026-05-20)

## Test
Add `rvol_7d` + `ret_3d` + `btc_rvol_7d` as MODEL features to WINNER_21
production stack. Retrain via R2a_retrain.py (replicates exact production
harness, same 5-seed LGBM ensemble, 10 folds), then run V3.1 6-sleeve replay
via scripts/phase_ah_sleeve_R2a.py.

## Result

| Metric | Production V3.1 | R2a (24 features) | Δ |
|---|---|---|---|
| Sharpe | +2.23 | **+0.39** | **−1.84** |
| totPnL | +9,167 | +1,072 | −8,095 bps |
| maxDD | −4,414 | −2,056 | smaller (PnL collapsed too) |
| Folds positive | 7/9 | 5/9 | −2 |
| Concentration | ~30% | 47% (fold 3 dominant) | worse |
| Sharpe ≥ +0.30 lift | (baseline) | FAIL | |
| Folds positive ≥ 6/9 | (baseline) | FAIL | |
| maxDD ≤ +20% worse | (baseline) | PASS (but only because totPnL ~9× smaller) | |
| Concentration ≤ 40% | (baseline) | FAIL | |

Per-fold Sharpe (R2a): f1 −0.26, f2 +1.82, f3 +3.62, f4 −0.19, f5 +1.72, f6 +0.85, f7 −1.47, f8 −4.17, f9 +0.73. Fold 8 catastrophic (−750 bps).

Pred correlation R2a vs production = **0.50** — half the ranking changed.

## Mechanism / lesson

Cohort attribution `q4-q0 Sharpe spread` is NOT a tradeability predictor for
model features. It measures "if entry is happening, predict its outcome";
it does not measure "if trained on this feature, will the model pick better
entries." The disconnect is now confirmed at THREE signal strengths:

- Phase Q W23 (+8.58 ethbtc_change_24h, +7.18 xs_ret_disp_1d): in-sample +0.18, fold-6-only, paired CI crosses 0
- R2a (+15.77 btc_rvol_7d, +11.32 ret_3d, +9.30 btc_rvol_3d): **−1.84 Sharpe**, fold 3 dominant, fold 8 catastrophic

Stronger cohort signal → worse out-of-sample. The cohort spread predicts
nothing useful about model retrain outcome. **No future paid-data evaluation
should use cohort spread as the gate.**

## Implications for the four-path validation

- (b) closed
- (c) deferral strengthened: cohort threshold ">11" is also wrong — if +15.77
  fails worse than +8.58, paid data justified by "cohort spread >11" has no
  precedent for working
- (a) only feature/model-side direction left; needs target_A diagnostic first
- (d) operational deployment unchanged

Closed: cohort-spread → training-feature as a research methodology.

## Artifacts

- predictions: `research/portable_alpha_2026-05-19/results/_cache/all_predictions_R2a.parquet`
- sleeve output: `outputs/vBTC_sleeve_R2a/` (per_cycle_equal6.csv, production_sleeves.parquet)
- script: `scripts/phase_ah_sleeve_R2a.py`
