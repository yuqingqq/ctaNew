"""V1-V4 robustness validation for hybrid LGBM + Ridge blend.

Headline: w=0.10 blend lifted Sharpe +1.13 → +1.57 (+0.43) on the 10-fold
multi-OOS. Sharp peak (w=0.05 was -0.12, w=0.20 was -0.24) raises selection-
bias concerns. Same V1-V4 protocol applied to conv_gate.

V1. Per-fold consistency at w=0.10 — does lift come from all 10 folds or outliers?
V2. Fine-grained weight plateau — robust or single-point peak?
V3. Hard-split frozen test — train hybrid on early 5 folds, evaluate on late 5
    with no retraining.
V4. Block-bootstrap 95% CI on Δ Sharpe (hybrid vs LGBM-only).

Critical question: does the hybrid help fold 9 (April 2026 loss)?
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio
from ml.research.alpha_v9_positioning_pack import build_panel

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
POS_FEATURES = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
OUT_DIR = REPO / "outputs/h48_hybrid_validate"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def fit_ridge(X_tr, y_tr, alpha=1.0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, y_tr)
    return m, scaler


def predict_ridge(m, scaler, X):
    Xs = scaler.transform(np.nan_to_num(X, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0)
    return m.predict(Xs)


def z(p):
    std = p.std()
    return (p - p.mean()) / (std if std > 1e-8 else 1.0)


def predict_fold(panel, fold, v6_clean, pos_features):
    """Returns (test_df, lgbm_z_pred, ridge_z_pred) for a fold."""
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200:
        return None, None, None

    avail_v6 = [c for c in v6_clean if c in panel.columns]
    avail_pos = [c for c in pos_features if c in panel.columns]
    Xt_v6 = tr[avail_v6].to_numpy(dtype=np.float32)
    yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
    Xc_v6 = ca[avail_v6].to_numpy(dtype=np.float32)
    yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
    Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
    Xt_pos = tr[avail_pos].to_numpy(dtype=np.float64)
    Xc_pos = ca[avail_pos].to_numpy(dtype=np.float64)
    Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)

    lgbm_models = [_train(Xt_v6, yt_, Xc_v6, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                          for m in lgbm_models], axis=0)
    X_full_pos = np.vstack([Xt_pos, Xc_pos])
    y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
    ridge_m, ridge_scaler = fit_ridge(X_full_pos, y_full, alpha=1.0)
    ridge_pred = predict_ridge(ridge_m, ridge_scaler, Xtest_pos)
    return test, z(lgbm_pred), z(ridge_pred)


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    fold_data = {}
    for fold in folds:
        t0 = time.time()
        test, lgbm_z, ridge_z = predict_fold(panel, fold, v6_clean, POS_FEATURES)
        if test is None:
            continue
        fold_data[fold["fid"]] = {"test": test, "lgbm_z": lgbm_z, "ridge_z": ridge_z}
        print(f"  fold {fold['fid']}: trained ({time.time() - t0:.0f}s)")

    # ===== V1. Per-fold consistency at w=0.10 =====
    print("\n" + "=" * 100)
    print("V1. PER-FOLD CONSISTENCY (LGBM-only vs hybrid w=0.10)")
    print("=" * 100)
    print(f"  {'fold':>4} {'period':>30} {'lgbm_net':>9} {'hybrid_net':>11} "
          f"{'Δnet':>7} {'lgbm_Sh':>8} {'hyb_Sh':>8} {'ΔSh':>7}")
    pf_records = []
    for fid, fd in sorted(fold_data.items()):
        test = fd["test"]
        lgbm_z = fd["lgbm_z"]
        ridge_z = fd["ridge_z"]
        # LGBM-only
        df_l = evaluate_portfolio(test, lgbm_z, use_gate=True, gate_pctile=GATE_PCTILE,
                                    use_magweight=False, top_k=TOP_K)
        # Hybrid w=0.10
        hybrid_pred = 0.9 * lgbm_z + 0.1 * ridge_z
        df_h = evaluate_portfolio(test, hybrid_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                    use_magweight=False, top_k=TOP_K)
        l_net = df_l["net_bps"].to_numpy()
        h_net = df_h["net_bps"].to_numpy()
        l_sh = sharpe_est(l_net)
        h_sh = sharpe_est(h_net)
        period = f"{test['open_time'].min().date()} → {test['open_time'].max().date()}"
        pf_records.append({"fold": fid, "lgbm_sh": l_sh, "hybrid_sh": h_sh,
                            "delta_sh": h_sh - l_sh,
                            "delta_net": h_net.mean() - l_net.mean()})
        print(f"  {fid:>4d} {period:>30} {l_net.mean():>+8.2f} {h_net.mean():>+10.2f} "
              f"{h_net.mean() - l_net.mean():>+6.2f} {l_sh:>+7.2f} {h_sh:>+7.2f} "
              f"{h_sh - l_sh:>+6.2f}")
    pf = pd.DataFrame(pf_records)
    print(f"  {'mean':>4} {'':>30} {'':>9} {'':>11} {pf['delta_net'].mean():>+6.2f} "
          f"{pf['lgbm_sh'].mean():>+7.2f} {pf['hybrid_sh'].mean():>+7.2f} "
          f"{pf['delta_sh'].mean():>+6.2f}")
    print(f"\n  folds with positive ΔSharpe: {(pf['delta_sh'] > 0).sum()}/{len(pf)}")
    print(f"  median ΔSharpe: {pf['delta_sh'].median():+.2f}, std {pf['delta_sh'].std():.2f}")

    # ===== V2. Fine-grained weight plateau =====
    print("\n" + "=" * 100)
    print("V2. FINE-GRAINED WEIGHT PLATEAU CHECK")
    print("=" * 100)
    weights = [0.00, 0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    print(f"  {'w':>5} {'gross':>7} {'cost':>6} {'net':>7} {'Sharpe':>7} {'95% CI':>15}")
    plateau = []
    for w in weights:
        all_recs = []
        for fid, fd in fold_data.items():
            blend = (1 - w) * fd["lgbm_z"] + w * fd["ridge_z"]
            df = evaluate_portfolio(fd["test"], blend, use_gate=True, gate_pctile=GATE_PCTILE,
                                     use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                all_recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                                  "skipped": r["skipped"], "gross": r["spread_ret_bps"],
                                  "cost": r["cost_bps"]})
        rdf = pd.DataFrame(all_recs)
        traded = rdf[rdf["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(rdf["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=1500)
        plateau.append({"w": w, "sharpe": sh, "lo": lo, "hi": hi,
                        "net": rdf["net"].mean(),
                        "gross": traded["gross"].mean() if len(traded) > 0 else 0,
                        "cost": traded["cost"].mean() if len(traded) > 0 else 0})
        print(f"  {w:>5.2f} {traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{rdf['net'].mean():>+6.2f}  {sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]")
    pl = pd.DataFrame(plateau)
    best = pl.loc[pl["sharpe"].idxmax()]
    print(f"\n  best w: {best['w']:.2f}, Sharpe {best['sharpe']:+.2f}")
    print(f"  weights within 0.20 of best: "
          f"{sorted(pl[pl['sharpe'] > best['sharpe'] - 0.20]['w'].tolist())}")

    # ===== V3. Hard-split frozen test =====
    print("\n" + "=" * 100)
    print("V3. HARD-SPLIT FROZEN TEST")
    print("=" * 100)
    n_train = max(3, len(fold_data) // 2)
    train_fids = list(fold_data.keys())[:n_train]
    test_fids = list(fold_data.keys())[n_train:]
    print(f"  train folds: {train_fids}, test folds: {test_fids}")

    # Build a combined train set: all rows up to last train fold's test_end
    last_train_end = max(fold_data[fid]["test"]["open_time"].max() for fid in train_fids)
    train_panel = panel[panel["open_time"] <= last_train_end]
    train_panel_filt = train_panel[train_panel["autocorr_pctile_7d"] >= THRESHOLD]
    n_train_rows = len(train_panel_filt)
    split = int(n_train_rows * 0.85)

    avail_v6 = [c for c in v6_clean if c in panel.columns]
    avail_pos = [c for c in POS_FEATURES if c in panel.columns]
    Xt_v6 = train_panel_filt[avail_v6].iloc[:split].to_numpy(dtype=np.float32)
    yt_ = train_panel_filt["demeaned_target"].iloc[:split].to_numpy(dtype=np.float32)
    Xc_v6 = train_panel_filt[avail_v6].iloc[split:].to_numpy(dtype=np.float32)
    yc_ = train_panel_filt["demeaned_target"].iloc[split:].to_numpy(dtype=np.float32)
    Xt_pos = train_panel_filt[avail_pos].iloc[:split].to_numpy(dtype=np.float64)
    Xc_pos = train_panel_filt[avail_pos].iloc[split:].to_numpy(dtype=np.float64)
    print(f"  frozen training: {split:,} train rows + {n_train_rows - split:,} cal rows")

    frozen_lgbm = [_train(Xt_v6, yt_, Xc_v6, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    frozen_ridge_m, frozen_ridge_scaler = fit_ridge(
        np.vstack([Xt_pos, Xc_pos]),
        np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)]),
        alpha=1.0,
    )

    print(f"  {'fold':>4} {'cycles':>7} {'lgbm_net':>9} {'hybrid_net':>11} "
          f"{'lgbm_Sh':>8} {'hyb_Sh':>8} {'ΔSh':>7}")
    hard_l_all, hard_h_all = [], []
    for fid in test_fids:
        test = fold_data[fid]["test"]
        Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
        Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
        lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                              for m in frozen_lgbm], axis=0)
        ridge_pred = predict_ridge(frozen_ridge_m, frozen_ridge_scaler, Xtest_pos)
        lgbm_z_p = z(lgbm_pred)
        ridge_z_p = z(ridge_pred)
        hybrid_p = 0.9 * lgbm_z_p + 0.1 * ridge_z_p
        df_l = evaluate_portfolio(test, lgbm_z_p, use_gate=True, gate_pctile=GATE_PCTILE,
                                    use_magweight=False, top_k=TOP_K)
        df_h = evaluate_portfolio(test, hybrid_p, use_gate=True, gate_pctile=GATE_PCTILE,
                                    use_magweight=False, top_k=TOP_K)
        l_net = df_l["net_bps"].to_numpy()
        h_net = df_h["net_bps"].to_numpy()
        hard_l_all.extend(l_net.tolist())
        hard_h_all.extend(h_net.tolist())
        print(f"  {fid:>4d} {len(l_net):>7d} {l_net.mean():>+8.2f} {h_net.mean():>+10.2f} "
              f"{sharpe_est(l_net):>+7.2f} {sharpe_est(h_net):>+7.2f} "
              f"{sharpe_est(h_net) - sharpe_est(l_net):>+6.2f}")
    l_arr = np.array(hard_l_all); h_arr = np.array(hard_h_all)
    print(f"\n  Hard-split overall (frozen):")
    print(f"    lgbm-only frozen:    Sharpe {sharpe_est(l_arr):+.2f}, net {l_arr.mean():+.2f}")
    print(f"    hybrid w=0.10:       Sharpe {sharpe_est(h_arr):+.2f}, net {h_arr.mean():+.2f}")
    print(f"    delta:               ΔSharpe {sharpe_est(h_arr) - sharpe_est(l_arr):+.2f}")

    # ===== V4. Block-bootstrap CI on Δ Sharpe =====
    print("\n" + "=" * 100)
    print("V4. BLOCK-BOOTSTRAP 95% CI ON Δ SHARPE")
    print("=" * 100)
    l_full, h_full = [], []
    for fid, fd in fold_data.items():
        df_l = evaluate_portfolio(fd["test"], fd["lgbm_z"], use_gate=True,
                                    gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
        hybrid = 0.9 * fd["lgbm_z"] + 0.1 * fd["ridge_z"]
        df_h = evaluate_portfolio(fd["test"], hybrid, use_gate=True,
                                    gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
        l_full.extend(df_l["net_bps"].tolist())
        h_full.extend(df_h["net_bps"].tolist())
    l_arr = np.array(l_full); h_arr = np.array(h_full)
    delta = h_arr - l_arr
    rng = np.random.default_rng(42)
    n = len(l_arr); block = 7; n_boot = 5000
    n_blocks = int(np.ceil(n / block))
    d_boot = np.empty(n_boot)
    l_boot = np.empty(n_boot)
    h_boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        l_boot[i] = sharpe_est(l_arr[idx])
        h_boot[i] = sharpe_est(h_arr[idx])
        d_boot[i] = h_boot[i] - l_boot[i]
    print(f"  LGBM-only Sharpe:    {sharpe_est(l_arr):+.2f}  CI [{np.quantile(l_boot, 0.025):+.2f}, {np.quantile(l_boot, 0.975):+.2f}]")
    print(f"  Hybrid w=0.10:       {sharpe_est(h_arr):+.2f}  CI [{np.quantile(h_boot, 0.025):+.2f}, {np.quantile(h_boot, 0.975):+.2f}]")
    print(f"  ΔSharpe (hyb-lgbm):  {sharpe_est(h_arr) - sharpe_est(l_arr):+.2f}  CI [{np.quantile(d_boot, 0.025):+.2f}, {np.quantile(d_boot, 0.975):+.2f}]")
    print(f"  P(ΔSharpe > 0):       {(d_boot > 0).mean()*100:.1f}%")
    print(f"  P(ΔSharpe > 0.20):    {(d_boot > 0.20).mean()*100:.1f}%")
    print(f"  P(ΔSharpe > 0.40):    {(d_boot > 0.40).mean()*100:.1f}%")

    summary = {
        "v1_per_fold": pf.to_dict("records"),
        "v1_pos_folds": int((pf["delta_sh"] > 0).sum()),
        "v1_total_folds": int(len(pf)),
        "v2_plateau": pl.to_dict("records"),
        "v2_best_w": float(best["w"]),
        "v3_hard_split": {
            "lgbm_sharpe": float(sharpe_est(np.array(hard_l_all))),
            "hybrid_sharpe": float(sharpe_est(np.array(hard_h_all))),
            "delta_sharpe": float(sharpe_est(np.array(hard_h_all)) - sharpe_est(np.array(hard_l_all))),
        },
        "v4_bootstrap": {
            "lgbm_sharpe_pt": float(sharpe_est(l_arr)),
            "hybrid_sharpe_pt": float(sharpe_est(h_arr)),
            "delta_sharpe_pt": float(sharpe_est(h_arr) - sharpe_est(l_arr)),
            "delta_ci": [float(np.quantile(d_boot, 0.025)), float(np.quantile(d_boot, 0.975))],
            "p_delta_positive": float((d_boot > 0).mean() * 100),
            "p_delta_gt_020": float((d_boot > 0.20).mean() * 100),
        },
    }
    with open(OUT_DIR / "alpha_v9_hybrid_validate_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
