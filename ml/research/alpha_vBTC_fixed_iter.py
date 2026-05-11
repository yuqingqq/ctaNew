"""Fix A test: replace early stopping with fixed iteration count.

Compares:
  baseline       — original (early stopping, patience=80)
  fixed_50       — 50 iterations, no early stop
  fixed_100      — 100 iterations
  fixed_200      — 200 iterations
  long_patience  — early stop with patience=500 (very lenient)

Run on N=15 K=3 (best 5-seed config from earlier validation), 5 seeds,
production folds 5-9. Total ~25 LGBMs per config × 5 configs = 125 LGBMs.
~10-15 min.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_fixed_iter"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
TOP_N = 15
K = 3


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def train_lgb(X_train, y_train, X_cal, y_cal, *, seed: int, mode: str):
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dc = lgb.Dataset(X_cal, y_cal, reference=dtr, free_raw_data=False)
    if mode == "baseline":
        return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                         callbacks=[lgb.early_stopping(stopping_rounds=80),
                                    lgb.log_evaluation(period=0)])
    elif mode == "fixed_50":
        return lgb.train(params, dtr, num_boost_round=50, valid_sets=[dc],
                         callbacks=[lgb.log_evaluation(period=0)])
    elif mode == "fixed_100":
        return lgb.train(params, dtr, num_boost_round=100, valid_sets=[dc],
                         callbacks=[lgb.log_evaluation(period=0)])
    elif mode == "fixed_200":
        return lgb.train(params, dtr, num_boost_round=200, valid_sets=[dc],
                         callbacks=[lgb.log_evaluation(period=0)])
    elif mode == "patience_500":
        return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                         callbacks=[lgb.early_stopping(stopping_rounds=500),
                                    lgb.log_evaluation(period=0)])
    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    sorted_syms = df_ic["symbol"].tolist()
    universe = set(sorted_syms[:TOP_N])
    print(f"  Panel: {len(panel):,} rows", flush=True)
    print(f"  Universe (top {TOP_N}): {sorted(universe)}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    modes = ["baseline", "fixed_50", "fixed_100", "fixed_200", "patience_500"]
    summary = []

    for mode in modes:
        print(f"\n=== Mode: {mode} ===", flush=True)
        fold_data = {}
        iter_counts = {}
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
            seeds_iters = []
            preds = []
            for s in SEEDS:
                m = train_lgb(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s, mode=mode)
                seeds_iters.append(m.best_iteration if m.best_iteration > 0 else m.num_trees())
                preds.append(m.predict(Xtest, num_iteration=m.best_iteration if mode != "fixed_50" and mode != "fixed_100" and mode != "fixed_200" else None))
            ensemble = np.mean(preds, axis=0)
            fold_data[fid] = (test.copy(), ensemble)
            iter_counts[fid] = seeds_iters
            print(f"  fold {fid}: iters={seeds_iters}  ({time.time()-t0:.0f}s)", flush=True)

        # Evaluate on N=15 K=3 universe
        cycles = []
        for fid, (test_df, pred) in fold_data.items():
            keep_mask = test_df["symbol"].isin(universe).to_numpy()
            df = evaluate_stacked(test_df[keep_mask].copy(), pred[keep_mask],
                                   use_conv_gate=True, use_pm_gate=True, top_k=K)
            for _, r in df.iterrows():
                cycles.append({"fold": fid, "time": r["time"],
                                "net": r["net_bps"], "cost": r["cost_bps"]})
        df_v = pd.DataFrame(cycles)
        if df_v.empty:
            print(f"  no cycles", flush=True)
            continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        cost = df_v["cost"].mean()
        max_dd = _max_dd(net)
        per_fold_sh = {}
        for fid in sorted(df_v["fold"].unique()):
            n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
            per_fold_sh[fid] = _sharpe(n_f) if len(n_f) >= 3 else 0.0
        avg_iters = float(np.mean([np.mean(its) for its in iter_counts.values()])) if iter_counts else 0.0
        summary.append({"mode": mode, "avg_iters": avg_iters, "n": len(net),
                          "mean_net": net.mean(), "cost": cost,
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd,
                          **{f"sh_f{f}": v for f, v in per_fold_sh.items()}})
        print(f"  Sharpe={sh:+.2f} CI=[{lo:+.2f},{hi:+.2f}] mean={net.mean():+.2f} "
              f"avg_iters={avg_iters:.0f}", flush=True)

    print(f"\n=== Summary (N={TOP_N}, K={K}, 5-seed ensemble) ===", flush=True)
    print(f"  {'mode':<14}  {'avg_iters':>9}  {'mean':>7}  {'Sharpe':>7}  "
          f"{'CI_lo':>7}  {'CI_hi':>7}  fold-by-fold", flush=True)
    for r in summary:
        per_fold = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['mode']:<14}  {r['avg_iters']:>9.0f}  {r['mean_net']:>+7.2f}  "
              f"{r['sharpe']:>+7.2f}  {r['ci_lo']:>+7.2f}  {r['ci_hi']:>+7.2f}  {per_fold}",
              flush=True)

    pd.DataFrame(summary).to_csv(OUT_DIR / "fixed_iter_summary.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
