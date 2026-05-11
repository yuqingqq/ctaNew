"""Phase 3: train + multi-OOS evaluate BTC-target model with refined feature set.

Combines:
  - 16 v6_clean features that survived Phase 1 IC ranking (|IC_btc| > 0.02)
  - 11 new BTC-specific features that passed Phase 2 IC threshold (|IC| > 0.005)
Total: 27 features.

Trains LGBM ensemble (5 seeds, basket-tuned hyperparams) on btc_beta_target.
Evaluates with three portfolio structures:
  - cross-sectional (long top-K alts, short bot-K alts) — control
  - BTC-hedge long-side (long top-K alts, short β×BTC)
  - BTC-hedge short-side (short bot-K alts, long β×BTC)

Decision point: if best variant Sharpe < +1.0, exit. If Sharpe ≥ +2.0 → proceed
to Phase 6 discipline gates.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_btc_beta_target import add_btc_beta_target, evaluate_btc_beta_hedge
from ml.research.alpha_v9_btc_features_engineer import compute_btc_specific_features
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
TOP_K = 7
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/btc_pm_hybrid_stack"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1 KEEP+CONSIDER (|IC_btc| > 0.02)
V6_KEEP = [
    "hour_cos", "dom_change_288b_vs_bk", "dom_z_7d_vs_bk", "return_1d_xs_rank",
    "ema_slope_20_1h_xs_rank", "bars_since_high_xs_rank", "obv_z_1d_xs_rank",
    "corr_change_3d_vs_bk", "hour_sin", "mfi", "idio_vol_1d_vs_bk_xs_rank",
    "obv_signal", "atr_pct", "return_1d", "price_volume_corr_10",
    "idio_ret_48b_vs_bk",
]

# Phase 2 KEEP (|IC_btc| > 0.005)
NEW_BTC_FEATURES = [
    "dom_btc_change_288b", "beta_to_btc_change_5d", "dom_btc_z_1d",
    "corr_to_btc_1d", "idio_vol_to_btc_1d", "idio_ret_to_btc_12b",
    "beta_to_btc", "idio_vol_to_btc_1h", "dom_btc_level",
    "dom_btc_change_48b", "corr_to_btc_change_3d",
]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def main():
    print("Building panel + BTC target + BTC features...")
    panel = build_wide_panel()
    panel = add_btc_beta_target(panel)
    panel, _ = compute_btc_specific_features(panel)
    print(f"  panel: {len(panel):,} rows")

    # Combined feature set
    feat_set = [c for c in V6_KEEP + NEW_BTC_FEATURES if c in panel.columns]
    print(f"\n  Feature set: {len(feat_set)} features")
    print(f"    v6_keep:    {[f for f in V6_KEEP if f in feat_set]}")
    print(f"    btc_new:    {[f for f in NEW_BTC_FEATURES if f in feat_set]}")

    folds = _multi_oos_splits(panel)
    cycles = {
        "btc_xs":          [],   # cross-sectional control
        "btc_long_btc":    [],   # long alts + short β×BTC
        "btc_short_btc":   [],   # short alts + long β×BTC
        "baseline_basket": [],   # current production reference
    }

    # For baseline: use v6_clean (28) features
    from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
    v6_full = [c for c in XS_FEATURE_COLS_V6_CLEAN if c in panel.columns]

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        # === BTC-target model (refined features) ===
        Xt_btc = tr[feat_set].to_numpy(np.float32)
        Xc_btc = ca[feat_set].to_numpy(np.float32)
        Xtest_btc = test[feat_set].to_numpy(np.float32)
        yt_btc = tr["btc_beta_target"].to_numpy(np.float32)
        yc_btc = ca["btc_beta_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt_btc); mask_c = ~np.isnan(yc_btc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue
        models_btc = [_train(Xt_btc[mask_t], yt_btc[mask_t], Xc_btc[mask_c], yc_btc[mask_c], seed=s)
                      for s in ENSEMBLE_SEEDS]
        pred_btc = np.mean([m.predict(Xtest_btc, num_iteration=m.best_iteration) for m in models_btc], axis=0)

        # === Baseline basket model (full v6_clean features) ===
        Xt_v6 = tr[v6_full].to_numpy(np.float32)
        Xc_v6 = ca[v6_full].to_numpy(np.float32)
        Xtest_v6 = test[v6_full].to_numpy(np.float32)
        yt_basket = tr["demeaned_target"].to_numpy(np.float32)
        yc_basket = ca["demeaned_target"].to_numpy(np.float32)
        models_basket = [_train(Xt_v6, yt_basket, Xc_v6, yc_basket, seed=s) for s in ENSEMBLE_SEEDS]
        pred_basket = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration) for m in models_basket], axis=0)

        # Evaluate four configurations
        df_xs = evaluate_stacked(test, pred_btc, use_conv_gate=True, use_pm_gate=True)
        df_long = evaluate_btc_beta_hedge(test, pred_btc, side="long",
                                            use_conv_gate=True, use_pm_gate=True)
        df_short = evaluate_btc_beta_hedge(test, pred_btc, side="short",
                                             use_conv_gate=True, use_pm_gate=True)
        df_basket = evaluate_stacked(test, pred_basket, use_conv_gate=True, use_pm_gate=True)

        for label, df in [("btc_xs", df_xs), ("btc_long_btc", df_long),
                            ("btc_short_btc", df_short), ("baseline_basket", df_basket)]:
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
    print("PHASE 3: BTC-TARGET (REFINED FEATURES) MULTI-OOS")
    print("=" * 110)
    print(f"  {'config':<22}  {'n':>4}  {'mean_net':>9}  {'mean_cost':>10}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    rows = []
    for label in cycles:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        cost = df_v["cost"].mean()
        rows.append({"config": label, "n": len(net), "mean_net": net.mean(),
                     "cost": cost, "sharpe": sh, "ci_lo": lo, "ci_hi": hi})
        print(f"  {label:<22}  {len(net):>4}  {net.mean():>+9.2f}  {cost:>+10.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}")

    print("\n  Decision points:")
    base_sh = next((r["sharpe"] for r in rows if r["config"] == "baseline_basket"), 0)
    print(f"    Baseline production: Sharpe {base_sh:+.2f}")
    for r in rows:
        if r["config"] == "baseline_basket": continue
        gap = r["sharpe"] - base_sh
        if r["sharpe"] >= 2.0:
            verdict = "PASS phase-3 bar (≥+2.0)"
        elif r["sharpe"] >= 1.0:
            verdict = "marginal (≥+1.0)"
        else:
            verdict = "FAIL (<+1.0, exit)"
        print(f"    {r['config']:<20}  Δsh vs baseline = {gap:+.2f}  → {verdict}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    for label, c in cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"{label}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
