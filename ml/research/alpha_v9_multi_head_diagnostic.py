"""Multi-head architecture diagnostic.

Tests whether the hybrid stacking architecture:
  (1) Compounds when packs are split into multiple heads (architecture
      flexibility test)
  (2) Is robust to broadcast features that have no XS variance per bar
      (control test — should fail, validating the diagnosis)
  (3) Is robust to RANDOM features (architecture noise tolerance)
  (4) Combines real + random heads correctly (does noise dilute real signal?)

This answers the question: when you start adding more ridge heads, what's
the failure mode? Is each new head purely additive, or does noise
contamination hurt?
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
POS_3_RANK = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
OUT_DIR = REPO / "outputs/h48_multi_head_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=1.0):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xs_tr = np.nan_to_num(Xs_tr, nan=0.0)
    Xs_te = scaler.transform(np.nan_to_num(X_te, nan=0.0))
    Xs_te = np.nan_to_num(Xs_te, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs_tr, y_tr)
    return m.predict(Xs_te)


def z(p):
    std = p.std()
    return (p - p.mean()) / (std if std > 1e-8 else 1.0)


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    t = pd.to_datetime(p["open_time"])
    p["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24).astype(np.float32)
    p["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24).astype(np.float32)
    p["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7).astype(np.float32)
    p["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7).astype(np.float32)
    return p


def add_random_features(panel: pd.DataFrame, n: int = 3, seed: int = 42) -> pd.DataFrame:
    p = panel.copy()
    rng = np.random.default_rng(seed)
    for i in range(n):
        p[f"random_{i}"] = rng.standard_normal(len(p)).astype(np.float32)
    return p


def main():
    panel = build_panel()
    panel = add_calendar_features(panel)
    panel = add_random_features(panel, n=3, seed=42)
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    RANDOM = ["random_0", "random_1", "random_2"]
    F_FUND = ["funding_z_24h_xs_rank"]
    F_LS = ["ls_ratio_z_24h_xs_rank"]
    F_OI = ["oi_change_24h_xs_rank"]

    # We'll cache LGBM and per-pack ridge predictions per fold
    fold_data = {}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue

        # LGBM on v6_clean
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        # Build prediction stash with ridges for various feature sets
        y_full = np.concatenate([
            tr["demeaned_target"].to_numpy(dtype=np.float64),
            ca["demeaned_target"].to_numpy(dtype=np.float64),
        ])

        def ridge_for(cols):
            avail = [c for c in cols if c in panel.columns]
            X_tr_full = np.vstack([
                tr[avail].to_numpy(dtype=np.float64),
                ca[avail].to_numpy(dtype=np.float64),
            ])
            X_te = test[avail].to_numpy(dtype=np.float64)
            return fit_predict_ridge(X_tr_full, y_full, X_te, alpha=1.0)

        fold_data[fold["fid"]] = {
            "test": test,
            "lgbm_z": z(lgbm_pred),
            "ridge_pos_combined_z": z(ridge_for(POS_3_RANK)),
            "ridge_funding_z": z(ridge_for(F_FUND)),
            "ridge_ls_z": z(ridge_for(F_LS)),
            "ridge_oi_z": z(ridge_for(F_OI)),
            "ridge_calendar_z": z(ridge_for(CALENDAR)),
            "ridge_random_z": z(ridge_for(RANDOM)),
        }
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    def evaluate_blend(blend_dict, label):
        """Evaluate a weighted blend of cached predictions across folds."""
        all_recs = []
        for fid, fd in fold_data.items():
            blended = np.zeros_like(fd["lgbm_z"])
            for key, w in blend_dict.items():
                blended = blended + w * fd[key]
            df = evaluate_portfolio(fd["test"], blended, use_gate=True,
                                     gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                all_recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                                  "skipped": r["skipped"],
                                  "gross": r["spread_ret_bps"], "cost": r["cost_bps"]})
        return pd.DataFrame(all_recs)

    print("\n" + "=" * 110)
    print("MULTI-HEAD ARCHITECTURE DIAGNOSTIC")
    print("=" * 110)
    print(f"  {'variant':<55} {'cycles':>7} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'Sharpe':>7} {'95% CI':>15} {'Δ vs v6':>8}")

    base_arr = np.array([r["net"] for r in evaluate_blend({"lgbm_z": 1.0}, "LGBM-only").to_dict("records")])
    base_sh = sharpe_est(base_arr)

    variants = [
        ("LGBM-only baseline",                                 {"lgbm_z": 1.0}),
        ("LGBM + ridge_pos_combined (w=0.10)",                  {"lgbm_z": 0.9, "ridge_pos_combined_z": 0.1}),
        ("LGBM + 3 split ridges (each w=0.033)",                {"lgbm_z": 0.9, "ridge_funding_z": 0.033,
                                                                   "ridge_ls_z": 0.033, "ridge_oi_z": 0.033}),
        ("LGBM + 3 split ridges (each w=0.05)",                 {"lgbm_z": 0.85, "ridge_funding_z": 0.05,
                                                                   "ridge_ls_z": 0.05, "ridge_oi_z": 0.05}),
        ("LGBM + ridge_calendar (w=0.10) — broadcast control",  {"lgbm_z": 0.9, "ridge_calendar_z": 0.1}),
        ("LGBM + ridge_random (w=0.10) — random control",       {"lgbm_z": 0.9, "ridge_random_z": 0.1}),
        ("LGBM + pos + random (each w=0.05)",                   {"lgbm_z": 0.9, "ridge_pos_combined_z": 0.05,
                                                                   "ridge_random_z": 0.05}),
        ("LGBM + pos + calendar + random (3 heads at w=0.033)", {"lgbm_z": 0.9, "ridge_pos_combined_z": 0.033,
                                                                   "ridge_calendar_z": 0.033, "ridge_random_z": 0.033}),
    ]

    summary = {}
    for label, blend in variants:
        df = evaluate_blend(blend, label)
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d_vs_v6 = sh - base_sh
        print(f"  {label:<55} {len(df):>7d} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_vs_v6:>+7.2f}")
        summary[label] = {
            "n_cycles": int(len(df)), "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_vs_v6": float(d_vs_v6),
        }

    print(f"\n  Reference: hybrid (combined pos pack) baseline = +1.57 Sharpe")
    print(f"  Test 1 (split heads): does B/C beat A?")
    print(f"  Test 2 (broadcast control): is calendar broadcast neutral or harmful?")
    print(f"  Test 3 (random control): does pure noise hurt the portfolio?")
    print(f"  Test 4 (real+noise): does adding noise dilute real-signal lift?")

    with open(OUT_DIR / "alpha_v9_multi_head_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
