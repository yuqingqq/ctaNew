"""Direct comparison: Ridge regression vs LGBM ensemble.

Same features (28 v6_clean), same train/cal/test, same N=15 K=3 universe.

Three Ridge variants:
  R0  Ridge(alpha=1.0)    — moderate regularization
  R1  Ridge(alpha=10.0)   — heavy regularization
  R2  RidgeCV             — cross-validated alpha selection

Compared to:
  L0  LGBM single-seed (seed=42)
  L1  LGBM 5-seed ensemble

For each model: per-fold IC, Sharpe, max DD.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_ridge_vs_lgbm"
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


def _per_cycle_ic(test_df: pd.DataFrame, pred: np.ndarray, target_col: str) -> float:
    df = test_df.copy()
    df["pred"] = pred
    ics = []
    for t, g in df.groupby("open_time"):
        sub = g[[target_col, "pred"]].dropna()
        if len(sub) < 5: continue
        ic = sub["pred"].rank().corr(sub[target_col].rank())
        if not pd.isna(ic): ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    print(f"  Universe: {sorted(universe)}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    fold_predictions = {}   # {fid: {model_label: pred (filtered)}}
    print(f"\n=== Training models per fold ===", flush=True)
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        # Combine train+cal for ridge training (no early stopping needed)
        tr_full = pd.concat([tr, ca], axis=0)
        Xtr = tr_full[feat_set].fillna(0).to_numpy(np.float32)
        ytr = tr_full["target_A"].to_numpy(np.float32)
        mask = ~np.isnan(ytr)
        Xtr, ytr = Xtr[mask], ytr[mask]

        Xtest = test[feat_set].fillna(0).to_numpy(np.float32)

        # === Ridge models ===
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xtest_s = scaler.transform(Xtest)

        ridge_preds = {}
        for alpha, label in [(1.0, "R0_ridge_a1"), (10.0, "R1_ridge_a10")]:
            m = Ridge(alpha=alpha).fit(Xtr_s, ytr)
            ridge_preds[label] = m.predict(Xtest_s)

        # RidgeCV: select alpha automatically
        m_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr_s, ytr)
        ridge_preds["R2_ridgecv"] = m_cv.predict(Xtest_s)

        # === LGBM models (using just train, cal for early stopping) ===
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)

        lgbm_preds_per_seed = []
        for s in SEEDS:
            m_lgb = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
            lgbm_preds_per_seed.append(m_lgb.predict(Xtest, num_iteration=m_lgb.best_iteration))
        lgbm_single = lgbm_preds_per_seed[0]
        lgbm_ens = np.mean(lgbm_preds_per_seed, axis=0)

        # Filter universe
        keep_mask = test["symbol"].isin(universe).to_numpy()
        test_f = test[keep_mask].copy()

        fold_predictions[fid] = (test_f, {
            "R0_ridge_a1": ridge_preds["R0_ridge_a1"][keep_mask],
            "R1_ridge_a10": ridge_preds["R1_ridge_a10"][keep_mask],
            "R2_ridgecv": ridge_preds["R2_ridgecv"][keep_mask],
            "L0_lgbm_seed42": lgbm_single[keep_mask],
            "L1_lgbm_5seed": lgbm_ens[keep_mask],
        })
        cv_alpha = m_cv.alpha_
        print(f"  fold {fid}: trained ({time.time()-t0:.0f}s, ridgeCV alpha={cv_alpha})", flush=True)

    # === Evaluate each model ===
    print(f"\n=== Per-fold IC + Sharpe (N={TOP_N}, K={K}) ===", flush=True)
    print(f"  {'fold':>4}  {'model':<18}  {'IC':>8}  {'Sharpe':>7}  {'mean_bps':>8}", flush=True)
    rows = []
    for fid, (test_f, preds) in fold_predictions.items():
        for label in ["R0_ridge_a1", "R1_ridge_a10", "R2_ridgecv", "L0_lgbm_seed42", "L1_lgbm_5seed"]:
            pred = preds[label]
            ic = _per_cycle_ic(test_f, pred, "alpha_A")
            df_eval = evaluate_stacked(test_f.copy(), pred, use_conv_gate=True, use_pm_gate=True, top_k=K)
            if df_eval.empty:
                rows.append({"fold": fid, "model": label, "ic": ic, "sharpe": 0,
                              "mean_net": 0, "n": 0})
                continue
            net = df_eval["net_bps"].to_numpy()
            sh = _sharpe(net)
            rows.append({"fold": fid, "model": label, "ic": ic, "sharpe": sh,
                          "mean_net": net.mean(), "n": len(net)})
            print(f"  {fid:>4}  {label:<18}  {ic:>+8.4f}  {sh:>+7.2f}  {net.mean():>+8.2f}",
                  flush=True)
        print()

    df_out = pd.DataFrame(rows)

    # Aggregate across folds
    print(f"=== Aggregate Sharpe (across all 5 production folds) ===", flush=True)
    print(f"  {'model':<18}  {'avg_IC':>8}  {'agg_Sharpe':>10}  {'CI_lo':>7}  {'CI_hi':>7}  "
          f"{'fold_Sh_range':>17}", flush=True)
    for label in ["R0_ridge_a1", "R1_ridge_a10", "R2_ridgecv", "L0_lgbm_seed42", "L1_lgbm_5seed"]:
        sub = df_out[df_out["model"] == label]
        avg_ic = sub["ic"].mean()
        # Aggregate sharpe by combining all cycles
        all_net = []
        for fid, (test_f, preds) in fold_predictions.items():
            df_eval = evaluate_stacked(test_f.copy(), preds[label],
                                        use_conv_gate=True, use_pm_gate=True, top_k=K)
            if not df_eval.empty:
                all_net.extend(df_eval["net_bps"].tolist())
        net = np.array(all_net)
        if len(net) > 10:
            sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        else:
            sh = lo = hi = 0
        sh_range = f"[{sub['sharpe'].min():+.2f},{sub['sharpe'].max():+.2f}]"
        print(f"  {label:<18}  {avg_ic:>+8.4f}  {sh:>+10.2f}  {lo:>+7.2f}  {hi:>+7.2f}  "
              f"{sh_range:>17}", flush=True)

    df_out.to_csv(OUT_DIR / "ridge_vs_lgbm.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
