# Experiment: bars_since_high_xs_rank over the 94 traded cohort vs full 175 panel (2026-06-06)

**Motivation:** `bars_since_high_xs_rank` (the one cross-sectional rank feature in V0) is ranked over the full
175-symbol panel, but the strategy trades only the 94 low-vol names (exclude top-80 high-vol). Does ranking the
feature over the actual traded cohort help? (Also: would decouple traded features from the 80 untraded names'
klines — the coupling behind the klines-vintage reconciliation.)

**Method:** regenerate BOTH arms' WF preds with identical code (per-symbol RidgeCV, 8 monthly folds, V0+RR long /
V0 short, xs_z target). Treatment recomputes ONLY `bars_since_high_xs_rank` over the low-vol cohort (frozen 5/29
exclude set — same as selection, so A/B is clean); 18 per-symbol feats unchanged. Replay v2 (equal-wt, K=3,
bear K=2, inv-vol, stop-off) on each. Baseline reproduces the known v2 run (+2.268 = 6/5 run.log +2.2686).

**Result — REJECTED:**
| arm | Sharpe | totPnL | maxDD |
|---|---|---|---|
| baseline (175-XS) | **+2.268** | +8,326 | −2,383 |
| treatment (94-XS) | +2.100 | +7,719 | −2,586 |
| lift | **−0.168** | −607 | −203 (worse) |

- treatment wins only **4/9 folds**; preds differ by mean 0.0028 (4% of rows >0.01) — small effect.
- paired block-bootstrap: **−0.41 bps/cyc, CI [−0.92, +0.10], crosses 0** → within noise, point estimate negative.

**Conclusion:** ranking over the traded cohort does not help. The full-175 rank encodes where a calm name sits in
the **whole market's** momentum distribution — marginally more informative than ranking only among similar calm
names. The current 175-XS feature is self-consistent (train==inference) and slightly better. **Keep 175-XS.**
The operational decoupling benefit is real but costs a (non-significant) ~0.17 Sharpe + worse maxDD → not worth it.
Monthly-PIT cohort refinement not pursued (≈10-name membership change, far too small to flip a within-noise result).

Script: `live/exp_xs94_genpreds.py`.
