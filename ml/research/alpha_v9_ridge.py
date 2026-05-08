"""Stacked LGBM + ridge ensemble: does adding a different model class help?

The HANDOFF says "linear oracle confirms the model is at the data ceiling,
not the architecture ceiling" — this was a benchmark for upper-bound
assessment, but the linear model was never deployed as part of the
production ensemble.

Test: train ridge regression on the same v6_clean features, blend with
the 5-seed LGBM ensemble. Sweep blend weights.

Hypothesis: trees and linear models capture different non-stationarities.
If they're modestly correlated, ensembling them gets a free Sharpe lift
through bias-variance diversification.

If even modest blending fails, the architecture really is at the
information ceiling — only data improvements would help.
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
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
RIDGE_ALPHA = 1.0
OUT_DIR = REPO / "outputs/h48_ridge"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray) -> tuple[Ridge, StandardScaler]:
    """Fit ridge on standardized features. NaN-fill at 0 after standardization
    (matches LGBM's "missing as separate path" semantics roughly)."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xs, y_train)
    return model, scaler


def predict_ridge(model: Ridge, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(np.nan_to_num(X, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0)
    return model.predict(Xs)


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    blends = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]  # weight on RIDGE in (1-w)*LGBM + w*ridge
    cycles: dict[float, list] = {b: [] for b in blends}
    pred_corrs = []
    pred_correctness = []

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail].to_numpy(dtype=np.float32)

        # LGBM ensemble (current production)
        lgbm_models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in lgbm_models], axis=0)

        # Ridge (single model on stacked train+cal)
        X_full = np.vstack([Xt, Xc])
        y_full = np.concatenate([yt_, yc_])
        ridge_model, scaler = fit_ridge(X_full, y_full)
        ridge_pred = predict_ridge(ridge_model, scaler, Xtest)

        # Standardize predictions to comparable scale (z-score per-fold)
        # so the blend weight is interpretable.
        def _z(p):
            std = p.std()
            return (p - p.mean()) / (std if std > 1e-8 else 1.0)
        lgbm_z = _z(lgbm_pred)
        ridge_z = _z(ridge_pred)

        # Correlation diagnostic
        c = np.corrcoef(lgbm_pred, ridge_pred)[0, 1]
        pred_corrs.append(c)

        for b in blends:
            blend_pred = (1.0 - b) * lgbm_z + b * ridge_z
            r = portfolio_pnl_turnover_aware(
                test, blend_pred, top_frac=TOP_FRAC,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
            )
            for _, row in r["df"].iterrows():
                cycles[b].append({
                    "fold": fold["fid"], "time": row["time"],
                    "gross": row["spread_ret_bps"],
                    "cost": row["cost_bps"],
                    "net": row["net_bps"],
                    "long_turn": row["long_turnover"],
                    "rank_ic": row["rank_ic"],
                })
        print(f"  fold {fold['fid']}: corr(LGBM,ridge)={c:.3f}  "
              f"({time.time() - t0:.0f}s)")

    print("\n" + "=" * 110)
    print(f"BLEND SWEEP — w=ridge_weight in (1-w)·LGBM + w·ridge  "
          f"(h={HORIZON} K={TOP_K}, β-neutral, {COST_PER_LEG} bps/leg)")
    print("=" * 110)
    print(f"  mean correlation between LGBM and ridge predictions: {np.mean(pred_corrs):.3f}  "
          f"(per-fold: {[f'{c:.2f}' for c in pred_corrs]})")
    print(f"\n  {'w':>5} {'n_cyc':>5} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'L_turn':>7} {'IC':>8} {'Sharpe':>7} {'95% CI':>15} {'Δsharp':>8}")

    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
    base_arr = np.array([r["net"] for r in cycles[0.0]])
    base_sh = sharpe_est(base_arr)

    summary = {"pred_corr_mean": float(np.mean(pred_corrs)),
                "pred_corr_per_fold": [float(c) for c in pred_corrs],
                "blends": []}
    for b in blends:
        recs = pd.DataFrame(cycles[b])
        if recs.empty: continue
        sh, lo, hi = block_bootstrap_ci(recs["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d_sh = sh - base_sh
        ic = recs["rank_ic"].mean()
        print(f"  {b:>5.2f} {len(recs):>5d} "
              f"{recs['gross'].mean():>+6.2f}  {recs['cost'].mean():>5.2f}  "
              f"{recs['net'].mean():>+6.2f}  "
              f"{recs['long_turn'].mean():>6.0%}  "
              f"{ic:>+7.4f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_sh:>+7.2f}")
        summary["blends"].append({
            "w_ridge": float(b),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_sharpe_vs_lgbm_only": float(d_sh),
            "gross": float(recs["gross"].mean()),
            "cost": float(recs["cost"].mean()),
            "net": float(recs["net"].mean()),
            "rank_ic": float(ic),
        })
    with open(OUT_DIR / "alpha_v9_ridge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
