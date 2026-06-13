# Linear β-residual — Tested Workflow (reference)

Persisted 2026-05-16. This is the exact validated pipeline run by Step 58b /
59 / 60, and per-iteration by Step 61. Verdict on the signal: **no statistically
real edge once the ~5-symbol meme/microcap tail (SIREN, JELLYJELLY, PIPPIN,
BROCCOLI714, SWARMS ≈ 4.5% of the 110-universe) is removed.** See
`project_vBTC_linear_model` memory for full conclusions.

## A. Per-retrain pipeline (the test harness)

| # | stage | detail | key params / files |
|---|---|---|---|
| 1 | Panel | `panel_btc_only_111.parquet`, 110 syms (BTC excl.), 5m bars, full-PIT features. Drop cumulative meme set + BTCUSDT → N syms | `outputs/vBTC_features_btc_only_111_full_pit/` |
| 2 | Folds | `_multi_oos_splits` → 10 folds: fold 0 train, folds 1–9 expanding-window OOS, embargo + `exit_time` purge | `ml.research.alpha_v4_xs_1d` |
| 3 | Target | β_pit = rolling cov/var(ret, btc_ret), `.shift(49)` strict-PIT; α_β = return_pct − β_pit·btc_ret (4h-fwd β-residual); σ_idio = std(α_β) fold-0 train, frozen (cross-sym median fallback); target_z = clip(α_β/σ_idio, ±5) | `01_build_target.py` logic |
| 4 | Features | V2 = 22: 11 frame-neutral + 3 R3² + 4 BTC-frame + 2 BTC² + return_8h_orth + vol_zscore_4h_over_7d. Per-family preproc on **fold-0 stats only**: heavy-tail→pooled rank+re-z; funding→per-sym rank+re-z; standard→winsor p1/p99+z; U-shapes²; NaN→0 (mask before `searchsorted`) | `58_clean108_train.build_v2_features` |
| 5 | Model | RidgeCV (alpha grid, r²), 5 bootstrap seeds, per fold; train rows `autocorr_pctile_7d ≥ 0.5` → OOS `pred_z` | `58_clean108_train.train_ridge` |
| 6 | Wrapper | trail_ic = rolling-90d IC(pred_z, α_β) per symbol; **pred_B = pred_z × trail_ic** | `compute_trailing_ic` |
| 7 | Universe | `build_rolling_ic_universe`: each cycle rank syms by trailing IC of **pred_z** vs α, take TOP_N=15, PIT-eligible (listed ≥60d). **Must be pred_z, not pred_B** (Step 61 v1 bug) | `phase_ah_sleeve.py` |
| 8 | Protocol | `run_production_protocol_save_sleeves` on **pred_B**: Phase-M K=3/side, persistence-momentum PM_M=2, conv_gate (GATE_LOOKBACK=252), filter_refill → long/short basket per cycle | `phase_ah_sleeve.py` |
| 9 | Sleeve | V3.1: 6 overlapping sleeves, HORIZON_ENTRY=48 (4h cadence), HOLD_BARS=288 (24h hold), equal 1/6 weight | `phase_ah_sleeve.py` |
| 10 | PnL | causal aggregator: gross = Σ tw·α[t]·1e4 (new weights earn fwd 4h α); funding = −Σ tw·funding_settled (UTC-grid, ±2% cap; immaterial); cost = Σ|Δtw|·2.25 bps; net = gross+funding−cost | `59_clean108_funding.aggregate_causal_funding` |
| 11 | Metrics | Sharpe ×√[(288·365)/48]; block-bootstrap CI (block=7, n=1000); folds-positive; per-symbol gross attribution; top-5%-of-cycles concentration | |
| 12 | Placebo | P1 = random picks in liquidity-universe (top-30 by vol); P2 = random picks in rolling-IC universe; 100 seeds each; **gate-consistent** (placebo also gates on pred_B); real must exceed placebo p95 | |

Cost convention: `COST_PER_LEG=4.5` → `COST_PER_UNIT_ABS_DELTA = 0.5·4.5 = 2.25`
bps per unit |Δw|. Full K=3 replacement ≈ 4.0 abs Δ × 2.25 = 9 bps RT.

## B. Step 61 iteration loop (alpha-token search)

Run A (1–12) → take per-symbol gross attribution (11) → add symbols making up
the top ~60% of positive gross to the drop set → repeat. **Stop** when
Sharpe ≤ 0.3, OR CI crosses 0 with no dominant contributor (top-1 < 20%), OR
N < 70. Result curve:

| dropped | N | Sharpe | CI | P1/P2 | top-1 sym |
|---|---|---|---|---|---|
| 2 (SIREN,JELLY) | 108 | +2.34 | [0.58,4.46] | p98 PASS | 38% (PIPPIN) |
| 5 (+PIPPIN,BROCCOLI714,SWARMS) | 105 | +0.66 | [−1.97,+3.00] | p80/p78 FAIL | 17% (none) |

→ entire placebo-real edge = ~5 syms ≈ 4.5% of universe; gone (not shrunk) by 5.

## C. Mean-reversion-exit variant (Step 62/63)

Steps 1–6 + 10–12 reused; steps 7–9 replaced by the event-driven engine
(no rolling-IC filter; rank-pred_B pool over full executable universe;
persist-until-exit). Documented separately in `MEANREV_EXIT_PLAN.md`;
result: REFUTED on blue-chip-44 (nested-OOS −0.67; loses to fixed-hold +0.77
and to random-exit; in-sample ceiling −0.60).
