"""Hybrid model architecture: LGBM (v6_clean) + Linear (positioning pack).

Motivation: positioning pack has +0.0027 linear orthogonal IC but LGBM
overfits and destroys top-K selection. A separate linear head can extract
small marginal signals without polluting LGBM's tree budget.

Two architectures tested:

  (A) Weighted prediction blend
      lgbm_pred = LGBM(v6_clean) → z-score per fold
      ridge_pred = Ridge(positioning_3) → z-score per fold
      final = (1-w)·lgbm + w·ridge,    w ∈ {0, 0.05, 0.10, 0.20, 0.30}

  (B) Residual stacking
      lgbm_pred = LGBM(v6_clean)
      residual = target - lgbm_pred (on train)
      ridge_residual = Ridge(positioning_3) trained on residual
      final = lgbm_pred + α·ridge_residual,  α ∈ {0, 0.5, 1.0, 2.0}

Key design choice: positioning_3 = funding-z + LS-z + OI-change (the pack
that showed +0.0027 linear IC vs baseline). NOT v6_clean — strict separation
of feature classes between the two heads.

If hybrid works (+0.1-0.3 Sharpe lift), opens path to integrating other
small-signal classes (sentiment, on-chain, calendar) via the linear head
without breaking LGBM.
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
POS_3 = ["funding_z_24h", "ls_ratio_z_24h", "oi_change_24h"]
POS_3_RANK = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
OUT_DIR = REPO / "outputs/h48_hybrid_arch"
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


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    pos_features = POS_3_RANK  # use xs_rank versions (highest linear IC in prior test)
    print(f"Multi-OOS folds: {len(folds)}")
    print(f"v6_clean features: {len(v6_clean)}")
    print(f"Positioning features (linear head): {pos_features}")

    blend_weights = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]   # weight on ridge prediction
    residual_alphas = [0.0, 0.25, 0.5, 1.0, 2.0]            # residual scale factor

    blend_cycles: dict[float, list] = {w: [] for w in blend_weights}
    residual_cycles: dict[float, list] = {a: [] for a in residual_alphas}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        avail_pos = [c for c in pos_features if c in panel.columns]

        Xt_v6 = tr[avail_v6].to_numpy(dtype=np.float32)
        Xt_pos = tr[avail_pos].to_numpy(dtype=np.float64)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)

        Xc_v6 = ca[avail_v6].to_numpy(dtype=np.float32)
        Xc_pos = ca[avail_pos].to_numpy(dtype=np.float64)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)

        Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
        Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)

        # 1. Train LGBM on v6_clean
        lgbm_models = [_train(Xt_v6, yt_, Xc_v6, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                              for m in lgbm_models], axis=0)
        # Also predict on train+cal for residual stacking
        lgbm_pred_tr = np.mean([m.predict(Xt_v6, num_iteration=m.best_iteration)
                                 for m in lgbm_models], axis=0)
        lgbm_pred_ca = np.mean([m.predict(Xc_v6, num_iteration=m.best_iteration)
                                 for m in lgbm_models], axis=0)

        # 2A. Train Ridge on positioning, full target
        X_full_pos = np.vstack([Xt_pos, Xc_pos])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        ridge_target_m, ridge_target_scaler = fit_ridge(X_full_pos, y_full, alpha=1.0)
        ridge_target_pred = predict_ridge(ridge_target_m, ridge_target_scaler, Xtest_pos)

        # 2B. Train Ridge on positioning, LGBM residual
        residual_tr = yt_.astype(np.float64) - lgbm_pred_tr.astype(np.float64)
        residual_ca = yc_.astype(np.float64) - lgbm_pred_ca.astype(np.float64)
        X_full_pos_64 = np.vstack([Xt_pos, Xc_pos])
        residual_full = np.concatenate([residual_tr, residual_ca])
        ridge_resid_m, ridge_resid_scaler = fit_ridge(X_full_pos_64, residual_full, alpha=1.0)
        ridge_resid_pred = predict_ridge(ridge_resid_m, ridge_resid_scaler, Xtest_pos)

        # Z-score each component (within-fold) for stable blending
        def z(p):
            std = p.std()
            return (p - p.mean()) / (std if std > 1e-8 else 1.0)
        lgbm_z = z(lgbm_pred)
        ridge_target_z = z(ridge_target_pred)

        # Architecture A: weighted blend
        for w in blend_weights:
            blended = (1 - w) * lgbm_z + w * ridge_target_z
            df = evaluate_portfolio(test, blended, use_gate=True, gate_pctile=GATE_PCTILE,
                                     use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                blend_cycles[w].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "skipped": r["skipped"],
                })

        # Architecture B: residual stacking
        for a in residual_alphas:
            stacked = lgbm_pred + a * ridge_resid_pred
            df = evaluate_portfolio(test, stacked, use_gate=True, gate_pctile=GATE_PCTILE,
                                     use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                residual_cycles[a].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "skipped": r["skipped"],
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    print("\n" + "=" * 110)
    print("HYBRID ARCHITECTURE TEST")
    print("=" * 110)

    print("\n  --- ARCHITECTURE A: WEIGHTED PREDICTION BLEND ---")
    print(f"  weight (ridge)   gross   cost     net   Sharpe          95% CI   Δ vs lgbm-only")
    base_arr = np.array([r["net"] for r in blend_cycles[0.0]])
    base_sh = sharpe_est(base_arr)
    for w in blend_weights:
        df = pd.DataFrame(blend_cycles[w])
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        print(f"  {w:>6.2f}          "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}   "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]   {sh - base_sh:>+6.2f}")

    print("\n  --- ARCHITECTURE B: RESIDUAL STACKING ---")
    print(f"  alpha (resid)    gross   cost     net   Sharpe          95% CI   Δ vs lgbm-only")
    base_arr = np.array([r["net"] for r in residual_cycles[0.0]])
    base_sh = sharpe_est(base_arr)
    for a in residual_alphas:
        df = pd.DataFrame(residual_cycles[a])
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        print(f"  {a:>6.2f}          "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}   "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]   {sh - base_sh:>+6.2f}")

    summary = {
        "blend": {str(w): {
            "n_cycles": len(blend_cycles[w]),
            "net": float(pd.DataFrame(blend_cycles[w])["net"].mean()),
            "sharpe": float(sharpe_est(np.array([r["net"] for r in blend_cycles[w]]))),
        } for w in blend_weights},
        "residual": {str(a): {
            "n_cycles": len(residual_cycles[a]),
            "net": float(pd.DataFrame(residual_cycles[a])["net"].mean()),
            "sharpe": float(sharpe_est(np.array([r["net"] for r in residual_cycles[a]]))),
        } for a in residual_alphas},
    }
    with open(OUT_DIR / "alpha_v9_hybrid_arch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
