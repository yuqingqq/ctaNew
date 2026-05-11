"""Loop Phase 6: K-sweep on +btc+fund winner (21 features).

Adding 5 new features may shift optimal K. Test K ∈ {3, 4, 5, 6} on
N=15 universe with the winning 21-feature set.
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
OUT_DIR = REPO / "outputs/vBTC_loop_phase6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
TOP_N = 15
K_VALUES = [3, 4, 5, 6]

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


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    folds_all = _multi_oos_splits(panel)
    folds = [folds_all[i] for i in PROD_FOLDS if i < len(folds_all)]
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"  Winner_21 ({len(feat_set)} features)", flush=True)

    # Train once per fold, reuse predictions across K values
    print(f"\n=== Training 5-seed ensemble per fold ===", flush=True)
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
        print(f"  fold {fid}: trained ({time.time()-t0:.0f}s)", flush=True)

    # Sweep K
    print(f"\n=== K SWEEP ===", flush=True)
    print(f"  {'K':>3} {'n':>4}  {'mean':>6}  {'cost':>5}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}",
          flush=True)
    summary = []
    for k in K_VALUES:
        cycles = []
        for fid, (test_f, pred_f) in fold_data.items():
            df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=k)
            for _, r in df_eval.iterrows():
                cycles.append({"fold": fid, "time": r["time"],
                                "net": r["net_bps"], "cost": r["cost_bps"]})
        df_v = pd.DataFrame(cycles)
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        per_fold_sh = {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                          for fid in sorted(df_v["fold"].unique())}
        summary.append({"K": k, "n": len(net), "mean": net.mean(), "cost": df_v["cost"].mean(),
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          **{f"sh_f{f}": v for f, v in per_fold_sh.items()}})
        print(f"  {k:>3} {len(net):>4}  {net.mean():>+6.2f}  {df_v['cost'].mean():>+5.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}", flush=True)

    print(f"\n  Per-fold Sharpe (winner_21):", flush=True)
    print(f"  {'K':>3}  " + " ".join(f"{'fold' + str(f):>9}" for f in PROD_FOLDS), flush=True)
    for r in summary:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['K']:>3}  " + cells, flush=True)

    pd.DataFrame(summary).to_csv(OUT_DIR / "phase6_k_sweep.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
