"""Loop Phase 9: Design A vs Design B — train on full universe vs train on subset.

Design A (current): train model on all 51 symbols' data → trade selected 15
Design B (new test): train model on only 15 selected symbols → trade those 15

Hypothesis: Design B might be tighter (specialization) or looser (less data).

Plus: test if Design B with rolling universe + retrain on each new universe
helps. This is the most aggressive form of "rolling".

Configurations:
  A_static            — train on 51, trade static 15 (current best, +3.88)
  B_static            — train on calib 15, trade calib 15 (specialized, fixed)
  B_static_full_retr  — train each fold on 51 then SUBSET-RETRAIN on 15

We use 5 seeds for speed (10 seeds was the validation; this is a quick screen).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from collections import deque

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
OUT_DIR = REPO / "outputs/vBTC_loop_phase9"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
K = 4
MIN_OBS_PER_SYM = 100
TARGET_N = 15

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


def train_subset(panel, fold, feat_set, sym_subset=None):
    """Train ensemble; optionally restrict to symbol subset."""
    train, cal, test = _slice(panel, fold)
    if sym_subset is not None:
        train = train[train["symbol"].isin(sym_subset)]
        cal = cal[cal["symbol"].isin(sym_subset)]
        test_full = test  # keep full test for eval consistency
    else:
        test_full = test
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_full[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    iters = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        iters.append(m.best_iteration)
    return test_full, np.mean(preds, axis=0), iters, len(tr)


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    # === Step 1: Determine static calibration universe (using full-51 model on calib folds) ===
    print(f"\n=== Determining calibration universe (top-15 by IC on folds 0-4) ===", flush=True)
    calib_preds = []
    for fid in [0, 1, 2, 3, 4]:
        if fid >= len(folds_all): continue
        test_df, pred, iters, n_train = train_subset(panel, folds_all[fid], feat_set, sym_subset=None)
        if test_df is None: continue
        df = test_df[["symbol", "open_time", "alpha_A"]].copy()
        df["pred"] = pred
        calib_preds.append(df)
    calib_df = pd.concat(calib_preds, ignore_index=True).dropna(subset=["alpha_A"])
    static_ics = calib_df.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = sorted(static_ics.head(TARGET_N).index.tolist())
    print(f"  Static universe ({len(static_universe)}): {static_universe}", flush=True)

    # === Step 2: Train both designs on production folds ===
    print(f"\n=== Training Design A (full 51) and Design B (subset 15) on prod folds ===", flush=True)
    fold_data_A = {}   # train on 51
    fold_data_B = {}   # train on 15 (calibration universe)
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        # Design A
        t0 = time.time()
        test_df_A, pred_A, iters_A, n_A = train_subset(panel, folds_all[fid], feat_set, sym_subset=None)
        if test_df_A is not None:
            keep_mask = test_df_A["symbol"].isin(static_universe).to_numpy()
            fold_data_A[fid] = (test_df_A[keep_mask].copy(), pred_A[keep_mask])
        # Design B
        test_df_B, pred_B, iters_B, n_B = train_subset(panel, folds_all[fid], feat_set,
                                                            sym_subset=static_universe)
        if test_df_B is not None:
            keep_mask = test_df_B["symbol"].isin(static_universe).to_numpy()
            fold_data_B[fid] = (test_df_B[keep_mask].copy(), pred_B[keep_mask])
        print(f"  fold {fid}: A iters={iters_A} (n_train={n_A:,})  "
              f"B iters={iters_B} (n_train={n_B:,})  ({time.time()-t0:.0f}s)", flush=True)

    # === Step 3: Evaluate ===
    print(f"\n=== Evaluation (K={K}, 5-seed ensemble, calib universe of 15) ===", flush=True)
    results = []
    for label, fold_data in [("A_static", fold_data_A), ("B_static", fold_data_B)]:
        cycles = []
        for fid, (test_f, pred_f) in fold_data.items():
            df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
            for _, r in df_eval.iterrows():
                cycles.append({"fold": fid, "time": r["time"],
                                "net": r["net_bps"], "cost": r["cost_bps"]})
        df_v = pd.DataFrame(cycles)
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        per_fold = {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                      for fid in sorted(df_v["fold"].unique())}
        results.append({"label": label, "n_cycles": len(net), "mean": net.mean(),
                          "cost": df_v["cost"].mean(),
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "per_fold": per_fold})
        print(f"  {label:<14}: n={len(net)}  mean={net.mean():+.2f}  Sharpe={sh:+.2f}  "
              f"CI=[{lo:+.2f},{hi:+.2f}]", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    for r in results:
        cells = " ".join(f"{r['per_fold'].get(f, 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['label']:<14}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "phase9_design_compare.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
