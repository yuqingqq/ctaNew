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

## ⚠️ CORRECTION (2026-06-06) — first run used the WRONG universe
The first run pre-filtered to the **single frozen 5/29 low-vol set across all 8 months** → baseline +2.268. That is
NOT production-faithful: production **re-ranks the top-80 high-vol exclude at each monthly retrain** (PIT). Applying
a **monthly-PIT rvol exclude** (re-rank per fold) recovers the real number:

| universe | Sharpe | maxDD |
|---|---|---|
| **monthly-PIT (correct / production)** | **+3.89** | −2,793 (matches v2_oos604 exactly) |
| frozen-5/29 (first run — wrong) | +2.27 | −2,383 |
| no-filter (bot trades all incl high-vol) | +1.78 | −4,722 |

Freezing the universe across months cost **−1.6 Sharpe** — it traded names that were calm by 5/29 but wild back
in Oct (and vice-versa). This also explains "why only +2.2": the +3.86/+4.32 v2 headlines were RIGHT
(production-faithful monthly-PIT); the +2.27 was the artifact. (My regen preds are 0.96-corr to v2_oos604's, so
+3.89 vs +4.32 is a minor pred-vintage residual; maxDD and row-count match exactly.)

**XS94 re-run on the CORRECT universe:** 175-XS +3.892 vs 94-XS +3.871, lift **−0.021**, paired CI [−0.54,+0.47]
— null, even cleaner than before. **Keep 175-XS** stands.

**Methodology lesson:** multi-month convexity backtests MUST re-rank the low-vol universe per fold (monthly-PIT).
A single frozen universe understates Sharpe by ~1.6 and is the wrong test. Scripts: `live/exp_xs94_monthlypit.py`.
