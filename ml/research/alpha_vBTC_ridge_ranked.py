"""Path 2: Ridge regression on rank-transformed features.

Cross-sectionally rank each feature per timestamp (pct rank ∈ [0,1]),
then Ridge regression. Compare to:
  - Raw-feature Ridge (R_raw)
  - LGBM 5-seed ensemble (L1)

If Ridge-on-ranks matches or beats LGBM, we have a deterministic
linear model that works as well — eliminating seed variance, early
stopping fragility, and ensembling complexity.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_ridge_ranked"
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


def rank_transform_features(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Cross-sectionally rank each feature per timestamp, return ranks ∈ [0,1]."""
    out = df.copy()
    for f in feat_cols:
        if f == "sym_id": continue
        if f not in df.columns: continue
        # rank(pct=True) gives uniform [0, 1] within each group
        out[f] = df.groupby("open_time")[f].rank(pct=True)
    # Median-fill any NaNs at 0.5 (neutral)
    for f in feat_cols:
        if f in out.columns:
            out[f] = out[f].fillna(0.5)
    return out


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    print(f"  Universe: {sorted(universe)}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    fold_predictions = {}
    print(f"\n=== Training models per fold ===", flush=True)
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        # Rank-transform features per cycle
        tr_ranked = rank_transform_features(tr, feat_set)
        ca_ranked = rank_transform_features(ca, feat_set)
        test_ranked = rank_transform_features(test, feat_set)

        # Combine train+cal for ridge
        tr_full_ranked = pd.concat([tr_ranked, ca_ranked], axis=0)
        Xtr_rk = tr_full_ranked[feat_set].fillna(0.5).to_numpy(np.float32)
        ytr = tr_full_ranked["target_A"].to_numpy(np.float32)
        mask = ~np.isnan(ytr)
        Xtr_rk, ytr = Xtr_rk[mask], ytr[mask]
        Xtest_rk = test_ranked[feat_set].fillna(0.5).to_numpy(np.float32)

        # Ridge on ranks
        m_rk = Ridge(alpha=1.0).fit(Xtr_rk, ytr)
        rk_pred = m_rk.predict(Xtest_rk)

        # Ridge on raw (for comparison)
        Xtr_raw = pd.concat([tr, ca], axis=0)[feat_set].fillna(0).to_numpy(np.float32)
        ytr_raw = pd.concat([tr, ca], axis=0)["target_A"].to_numpy(np.float32)
        mask_raw = ~np.isnan(ytr_raw)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr_raw[mask_raw])
        m_raw = Ridge(alpha=1.0).fit(scaler.transform(Xtr_raw[mask_raw]), ytr_raw[mask_raw])
        Xtest_raw = test[feat_set].fillna(0).to_numpy(np.float32)
        raw_pred = m_raw.predict(scaler.transform(Xtest_raw))

        # LGBM 5-seed ensemble
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest_lgb = test[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        lgbm_preds = []
        for s in SEEDS:
            m_lgb = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
            lgbm_preds.append(m_lgb.predict(Xtest_lgb, num_iteration=m_lgb.best_iteration))
        lgbm_pred = np.mean(lgbm_preds, axis=0)

        # Filter universe
        keep_mask = test["symbol"].isin(universe).to_numpy()
        test_f = test[keep_mask].copy()
        fold_predictions[fid] = (test_f, {
            "Ridge_raw": raw_pred[keep_mask],
            "Ridge_ranked": rk_pred[keep_mask],
            "LGBM_5seed": lgbm_pred[keep_mask],
        })
        # Print Ridge weights for inspection (rank version)
        if fid == PROD_FOLDS[0]:
            print(f"\n  Ridge_ranked feature weights (top |w|):", flush=True)
            weights = list(zip(feat_set, m_rk.coef_))
            weights.sort(key=lambda x: -abs(x[1]))
            for f, w in weights[:15]:
                print(f"    {f:<32}  w={w:+.4f}", flush=True)
        print(f"  fold {fid}: trained ({time.time()-t0:.0f}s)", flush=True)

    # === Evaluate ===
    print(f"\n=== Per-fold IC + Sharpe (N={TOP_N}, K={K}) ===", flush=True)
    print(f"  {'fold':>4}  {'model':<14}  {'IC':>8}  {'Sharpe':>7}  {'mean_bps':>8}", flush=True)
    rows = []
    for fid, (test_f, preds) in fold_predictions.items():
        for label in ["Ridge_raw", "Ridge_ranked", "LGBM_5seed"]:
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
            print(f"  {fid:>4}  {label:<14}  {ic:>+8.4f}  {sh:>+7.2f}  {net.mean():>+8.2f}",
                  flush=True)
        print()

    df_out = pd.DataFrame(rows)

    # Aggregate
    print(f"=== Aggregate ===", flush=True)
    print(f"  {'model':<14}  {'avg_IC':>8}  {'agg_Sharpe':>10}  {'CI_lo':>7}  {'CI_hi':>7}  "
          f"{'fold_Sh_range':>17}", flush=True)
    for label in ["Ridge_raw", "Ridge_ranked", "LGBM_5seed"]:
        sub = df_out[df_out["model"] == label]
        avg_ic = sub["ic"].mean()
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
        print(f"  {label:<14}  {avg_ic:>+8.4f}  {sh:>+10.2f}  {lo:>+7.2f}  {hi:>+7.2f}  "
              f"{sh_range:>17}", flush=True)

    df_out.to_csv(OUT_DIR / "ridge_ranked_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
