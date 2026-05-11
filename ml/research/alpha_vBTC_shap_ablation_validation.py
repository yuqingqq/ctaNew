"""SHAP ablation — production-evaluator validation.

Uses phase 6's evaluate_stacked + calibration-based universe (OOS to prod folds).
Variants: baseline_21, no_vol_3, no_vol_no_dom.
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
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_shap_ablation_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
TOP_N = 15
K = 4

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def train_variant(panel, folds_all, feat_set, universe):
    fold_data = {}
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue
        preds = []
        for s in SEEDS:
            m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
            preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        ensemble = np.mean(preds, axis=0)
        keep_mask = test["symbol"].isin(universe).to_numpy()
        fold_data[fid] = (test[keep_mask].copy(), ensemble[keep_mask])
        print(f"    fold {fid}: ({time.time()-t0:.0f}s)", flush=True)
    return fold_data


def evaluate_variant(fold_data):
    cycles = []
    for fid, (test_f, pred_f) in fold_data.items():
        df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
        for _, r in df_eval.iterrows():
            cycles.append({"fold": fid, "time": r["time"],
                            "net": r["net_bps"], "cost": r["cost_bps"]})
    df_v = pd.DataFrame(cycles)
    if df_v.empty: return None
    net = df_v["net"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    per_fold = {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                  for fid in sorted(df_v["fold"].unique())}
    return {"n": len(net), "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "std_bps": net.std(), "max_dd": _max_dd(net),
            "mean_net": net.mean(),
            **{f"sh_f{f}": v for f, v in per_fold.items()}}


def main():
    print(f"Loading panel + calibration universe...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    print(f"  Calibration universe ({len(universe)}): {sorted(universe)}", flush=True)
    folds_all = _multi_oos_splits(panel)

    variants = {
        "baseline_21":   WINNER_21,
        "no_vol_3":      [f for f in WINNER_21 if f not in
                            ("atr_pct", "idio_vol_1d_vs_bk_xs_rank", "idio_vol_to_btc_1h")],
        "no_vol_no_dom": [f for f in WINNER_21 if f not in
                            ("atr_pct", "idio_vol_1d_vs_bk_xs_rank", "idio_vol_to_btc_1h",
                             "dom_change_288b_vs_bk")],
    }

    results = []
    for name, feats in variants.items():
        feat_set = [f for f in feats if f in panel.columns]
        print(f"\n=== {name} ({len(feat_set)} features) ===", flush=True)
        fold_data = train_variant(panel, folds_all, feat_set, universe)
        r = evaluate_variant(fold_data)
        if r is not None:
            r["variant"] = name; r["n_features"] = len(feat_set)
            results.append(r)
            print(f"  Sharpe={r['sharpe']:+.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}], "
                  f"std={r['std_bps']:.1f}, max_DD={r['max_dd']:+.0f}, mean={r['mean_net']:+.2f}",
                  flush=True)

    print(f"\n=== Summary ===", flush=True)
    print(f"  {'variant':<18}  {'n':>3}  {'Sharpe':>10}  {'CI':>17}  {'std':>6}  {'max_DD':>7}",
          flush=True)
    for r in results:
        print(f"  {r['variant']:<18}  {r['n_features']:>3}  {r['sharpe']:>+10.2f}  "
              f"[{r['ci_lo']:>+5.2f},{r['ci_hi']:>+5.2f}]  {r['std_bps']:>6.1f}  {r['max_dd']:>+7.0f}",
              flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'variant':<18}  " + " ".join(f"{'fold' + str(f):>8}" for f in PROD_FOLDS), flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['variant']:<18}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "validation_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
