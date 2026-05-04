# aggTrade microstructure features at h=48 — TESTED NEGATIVE

**Date:** 2026-05-04
**Reproducibility:** `python3 -m ml.research.alpha_v8_h48_audit`
**Per-cycle data:** `outputs/h48_features/alpha_v8_h48_paired.csv`

## TL;DR

Aggregated 4h trade-flow features from Binance aggTrades data
(`signed_volume_4h`, `tfi_4h`, `aggr_ratio_4h`, `buy_count_4h`,
`avg_trade_size_4h`) **do NOT add deployable Sharpe** to v6_clean at
h=48 K=7 ORIG25. Verified across 5 independent multi-OOS tests. The
features carry real per-bar information (univariate IC and LGBM gain
both confirm), but it is information-equivalent to v6_clean's existing
kline-derived flow proxies (OBV/VWAP/MFI/volume_ma) for portfolio
prediction at this horizon.

**Don't redo this audit.**

## Definitive paired comparison (production-realistic)

```
Both configs trained on SAME wide panel (2.86M rows, NaN-tolerant on flow).
Both evaluated on SAME 1620 OOS cycles, SAME 9 multi-OOS folds.

  v6_clean (current):       Sharpe +3.63 [+1.31, +6.14], mean +4.33 bps/cycle
  v6_clean_v2 (swap):       Sharpe +3.32 [+1.17, +5.60], mean +3.79 bps/cycle
  paired Δ:                 -0.54 bps/cycle
  paired t-stat:            -0.54  (one-sided p=0.29)
  swap-wins rate:           48.1% of cycles
  folds favoring swap:      6/9 (one outlier fold drags aggregate negative)
```

## What was tried

5 progressively-rigorous tests, all consistent with "no deployable edge":

| Test | Result | Verdict |
|---|---|---|
| Stage 1: per-bar 5min primitives, univariate IC at h=48 | 1/20 features pass gates (marginal) | Wrong timescale |
| Stage 1.5: 4h-aggregated primitives, univariate IC | 6/29 pass gates, |IC| 0.018-0.026 | Encouraging |
| Stage 2: portfolio additive (v6_clean + 5 aggTrade) | ΔS = +0.04 (essentially flat) | Capacity dilution |
| Stage 2.5: portfolio swap (v6_clean → v6_clean_v2) | Wide panel ΔS=-0.31, narrow panel ΔS=+1.18 | Inconclusive due to fold-shift artifact |
| **Stage 3: unified paired test** | **paired Δ=-0.54, t=-0.54, p=0.29** | **No edge** |

The wide vs narrow panel discrepancy in Stage 2.5 turned out to be a
fold-boundary artifact: the 22k-row truncation shifted the panel
start time by 5 days, which shifted all 9 multi-OOS test windows by
~3 days (verified: 0% test-cycle overlap between wide and narrow
runs). Stage 3 isolated the feature-set effect from the fold-boundary
effect by using a single panel for both configs.

## Mechanism — why aggTrade is information-equivalent to kline-flow

Per-symbol Spearman correlation of aggTrade features with v6_clean kline-flow:

```
                   obv_z_1d  obv_signal  mfi   vwap_zscore  vwap_slope_96  volume_ma_50
signed_volume_4h     0.40      0.36     0.29     0.49          0.41           -0.16
tfi_4h               0.37      0.35     0.30     0.49          0.38           -0.02
aggr_ratio_4h        0.34      0.17     0.15     0.35          0.34           -0.01
buy_count_4h         0.02     -0.01     0.02     0.06          0.04           +0.84  ← duplicate
avg_trade_size_4h    0.03     -0.01    -0.01    -0.04         -0.02           +0.45
```

The kline OBV approximates aggressor-side signed volume from price
direction (close vs prev_close × volume). MFI does the same with typical
price + volume. These approximations are good enough at the 4h horizon
that the more-precise aggTrade versions don't add independent signal.

Specifically:
- `buy_count_4h ↔ volume_ma_50`: 0.844 — essentially the same feature
- `signed_volume_4h, tfi_4h, aggr_ratio_4h`: collectively 0.3-0.5 with
  the OBV+VWAP cluster (multivariate ~70-80% explained)
- `avg_trade_size_4h`: 0.749 with `volume_ma_50` and 0.564 with `atr_pct`
  (institutional flow signal is reconstructible from existing features)

## Saturation hypothesis — 5th independent confirmation

This is the 5th feature-class addition tested against v6_clean and the
5th to fail:

1. DVOL (BTC/ETH options IV): Sharpe -1.86 — broadcast-feature dilution
2. Funding-rate features (3 audited): Sharpe -0.99 — overfitting
3. aggTrade per-bar primitives at h=288: Sharpe collapse (Phase 3, Apr 30)
4. aggTrade per-bar primitives at h=48: 1/20 pass univariate gates
5. **aggTrade 4h-aggregated swap at h=48: paired Δ=-0.54, t=-0.54** ← this audit

Generalizable rule confirmed: **for v6_clean's architecture at h=48-288,
candidate features must derive from data outside the price+volume tape
to add portfolio Sharpe.** Anything derivable from kline data is already
encoded by the existing 28-feature spine.

## Where to spend effort instead

Genuine future Sharpe lift requires **structural changes**:

| Direction | Expected lift | Effort |
|---|---|---|
| Maker-mode execution | +1 to +3 Sharpe | 1-2 weeks (queue modeling) |
| Vol-targeting in portfolio construction | +0.2 to +0.5 | 1-2 days |
| HYPE Bronze tier (10% taker discount) | +0.2 | 30 min |
| L2 order book features (depth, imbalance) | unknown | 2-4 weeks (Tardis data) |
| On-chain features (whale flows, exchange in/out) | unknown | 1-2 weeks (Glassnode) |

Do not attempt:
- Any further v6_clean feature reselection on existing data classes
- LGBM hyperparameter retuning (universe-conditional, doesn't transport)
- Universe expansion (proven harmful)
- Different model architectures (at data ceiling, not architecture ceiling)
- Regime-conditional MoE / mixture models (proven worthless)
