"""Phase 4-5: train LGBM on combined feature set, multi-OOS evaluate with BTC-hedge.

Feature set:
  - 16 v6_clean features that survived Phase 1 IC filter (|IC_btc| > 0.02)
  - 21 new BTC-target-specific features that survived Phase 3 IC filter (|IC| > 0.005)
Total: 37 features.

Configs evaluated:
  A. baseline_basket    — basket-target on v6_clean features (production reference)
  B. btc_xs             — btc_target + cross-sectional book (alt long-short)
  C. btc_long_btc       — btc_target + long top-K alts + short β×BTC
  D. btc_short_btc      — btc_target + short bot-K alts + long β×BTC

Decision point: best variant Sharpe ≥ +1.0 to continue, ideally ≥ +2.0.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_btc_beta_target import evaluate_btc_beta_hedge
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

# Ultra-fast directional mode: 1 seed only. If +Sharpe direction holds we'll
# scale up to full ensemble for proper validation. With 51 symbols × 5.8M
# rows, full 5-seed × 10-fold = 50 LGBMs took ~3hr. 1 × 3 folds = 3 LGBMs ≈
# 10-15 min for go/no-go.
FAST_SEEDS = (42,)
FAST_N_FOLDS = 3   # use 3 representative folds (early/mid/late) for speed

PANEL_PATH = REPO / "outputs/vBTC_features/panel_with_btc_features.parquet"
OUT_DIR = REPO / "outputs/vBTC_train_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
TOP_K = 7
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
# Baseline reference (validated on 25-symbol panel multi-OOS, live-model)
BASKET_BASELINE_SHARPE = 2.47

# Phase 1 KEEP — v6_clean features that survived BTC-target IC filter
# (|IC_btc| > 0.02), MINUS the dedup drop (return_1d_xs_rank, redundant
# with dom_change_288b_vs_bk at ρ=+0.962).
V6_KEEP = [
    "hour_cos", "dom_change_288b_vs_bk", "dom_z_7d_vs_bk",
    "ema_slope_20_1h_xs_rank", "bars_since_high_xs_rank", "obv_z_1d_xs_rank",
    "corr_change_3d_vs_bk", "hour_sin", "mfi", "idio_vol_1d_vs_bk_xs_rank",
    "obv_signal", "atr_pct", "return_1d", "price_volume_corr_10",
    "idio_ret_48b_vs_bk",
]   # 15 features

# Phase 3 KEEP — new BTC-target features that survived IC filter,
# MINUS dedup drops:
#   xs_alpha_dispersion_12b  (redundant with 48b, ρ=+0.877)
#   name_idio_share_1d       (= 1 - factor_loading², ρ=-1.0)
#   btc_ema_slope_4h         (redundant with btc_ret_48b, ρ=+0.868)
#   idio_max_abs_12b         (redundant with idio_vol_to_btc_1h, ρ=+0.950)
NEW_BTC = [
    "btc_realized_vol_1d", "btc_realized_vol_30d", "xs_alpha_iqr_12b",
    "xs_alpha_dispersion_48b", "name_factor_loading_1d",
    "btc_realized_vol_1h", "beta_to_btc_change_5d", "corr_to_btc_1d",
    "btc_ret_288b", "idio_kurt_1d", "idio_vol_to_btc_1d",
    "xs_alpha_mean_48b", "btc_ret_12b", "idio_ret_to_btc_12b",
    "btc_ret_48b", "idio_skew_1d", "idio_vol_to_btc_1h",
]   # 17 features


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print(f"Loading panel from {PANEL_PATH}...")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel.shape[1]} cols, {panel['symbol'].nunique()} syms")

    # Combined feature set
    feat_set = [c for c in V6_KEEP + NEW_BTC if c in panel.columns]
    print(f"  feature set: {len(feat_set)} features ({len([f for f in V6_KEEP if f in feat_set])} v6 + "
          f"{len([f for f in NEW_BTC if f in feat_set])} new BTC)")

    # For baseline: use full v6_clean (28)
    # v6_full would be used for basket-target training; skipped in fast mode
    # (we have +2.47 reference from prior multi-OOS validation runs).

    all_folds = _multi_oos_splits(panel)
    # Subset to representative folds for fast directional test
    n_total = len(all_folds)
    fold_idx = [n_total // 5, n_total // 2, 4 * n_total // 5][:FAST_N_FOLDS]
    folds = [all_folds[i] for i in fold_idx if i < n_total]
    print(f"  multi-OOS folds: {len(folds)} of {n_total} (fast mode, indices {fold_idx})")

    cycles = {"B_btc_xs": [],
              "C_btc_long_btc": [], "D_btc_short_btc": []}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        Xt_btc = tr[feat_set].to_numpy(np.float32)
        Xc_btc = ca[feat_set].to_numpy(np.float32)
        Xtest_btc = test[feat_set].to_numpy(np.float32)

        # === Train btc model only (basket reference is +2.47 from prior multi-OOS) ===
        # Fast mode: 3 seeds (vs 5). Sufficient for go/no-go decision.
        yt_btc = tr["btc_target"].to_numpy(np.float32)
        yc_btc = ca["btc_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt_btc); mask_c = ~np.isnan(yc_btc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue
        models_btc = [_train(Xt_btc[mask_t], yt_btc[mask_t], Xc_btc[mask_c], yc_btc[mask_c], seed=s)
                      for s in FAST_SEEDS]
        pred_btc = np.mean([m.predict(Xtest_btc, num_iteration=m.best_iteration) for m in models_btc], axis=0)

        # === Evaluate ===
        df_B = evaluate_stacked(test, pred_btc, use_conv_gate=True, use_pm_gate=True)
        df_C = evaluate_btc_beta_hedge(test, pred_btc, side="long", use_conv_gate=True, use_pm_gate=True)
        df_D = evaluate_btc_beta_hedge(test, pred_btc, side="short", use_conv_gate=True, use_pm_gate=True)

        for label, df in [("B_btc_xs", df_B),
                            ("C_btc_long_btc", df_C), ("D_btc_short_btc", df_D)]:
            for _, r in df.iterrows():
                cycles[label].append({"fold": fold["fid"], "time": r["time"],
                                       "net": r["net_bps"], "cost": r["cost_bps"],
                                       "skipped": r["skipped"]})

        line = f"  fold {fold['fid']:>2}: "
        for label in cycles:
            df_t = pd.DataFrame(cycles[label])
            if df_t.empty or "fold" not in df_t.columns:
                line += f"{label[:14]}=n/a "
                continue
            df_t = df_t[df_t["fold"] == fold["fid"]]
            if df_t.empty:
                line += f"{label[:14]}=n/a "
                continue
            n = df_t["net"].to_numpy()
            line += f"{label[:14]}={n.mean():+.2f}({_sharpe(n):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print("PHASE 4 — BTC-target on 51-name universe + 37 features (16 v6 keep + 21 new BTC)")
    print("=" * 110)
    print(f"  {'config':<22}  {'n':>4}  {'mean_net':>9}  {'cost':>7}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'max_DD':>8}")
    rows = []
    for label in cycles:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(net)
        cost = df_v["cost"].mean()
        rows.append({"config": label, "n": len(net), "mean_net": net.mean(),
                     "cost": cost, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                     "max_dd": max_dd})
        print(f"  {label:<22}  {len(net):>4}  {net.mean():>+9.2f}  {cost:>+7.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}  {max_dd:>+8.0f}")

    print("\n  Decision point check (vs production basket-target reference Sharpe +2.47):")
    base_sh = BASKET_BASELINE_SHARPE
    for r in rows:
        if r["sharpe"] >= 2.0:
            verdict = "PASS bar (≥+2.0); proceed to Phase 6"
        elif r["sharpe"] >= 1.0:
            verdict = "MARGINAL (≥+1.0); discuss before continuing"
        else:
            verdict = "FAIL (<+1.0); close path"
        print(f"    {r['config']:<22}  Sharpe={r['sharpe']:+.2f}  Δsh vs basket={r['sharpe']-base_sh:+.2f}  → {verdict}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    for label, c in cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"{label}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
