"""Minimum viable multi-head test — only the architectural questions that matter.

Q1: Does adding a noise ridge head (pure random) dilute the real signal?
Q2: Does the validated hybrid (LGBM + pos_combined) hold up here?

Just 4 variants. Reuses positioning panel as-is. Single ridge fit per
non-baseline variant.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
POS_3 = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
OUT_DIR = REPO / "outputs/h48_multi_head_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xte = sc.transform(np.nan_to_num(X_te, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, y_tr)
    return m.predict(Xte)


def z(p):
    s = p.std()
    return (p - p.mean()) / (s if s > 1e-8 else 1.0)


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Folds: {len(folds)}", flush=True)

    fold_data = {}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)
        t_lgbm = time.time() - t0

        # Ridge on positioning combined (validated)
        avail_pos = [c for c in POS_3 if c in panel.columns]
        X_full_pos = np.vstack([
            tr[avail_pos].to_numpy(dtype=np.float64),
            ca[avail_pos].to_numpy(dtype=np.float64),
        ])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
        ridge_pos_pred = fit_predict_ridge(X_full_pos, y_full, Xtest_pos)
        t_ridge_pos = time.time() - t0 - t_lgbm

        # Random feature ridge
        rng = np.random.default_rng(42 + fold["fid"])
        X_rand_full = rng.standard_normal((len(X_full_pos), 3))
        X_rand_te = rng.standard_normal((len(Xtest_pos), 3))
        ridge_rand_pred = fit_predict_ridge(X_rand_full, y_full, X_rand_te)
        t_ridge_rand = time.time() - t0 - t_lgbm - t_ridge_pos

        fold_data[fold["fid"]] = {
            "test": test,
            "lgbm_z": z(lgbm_pred),
            "ridge_pos_z": z(ridge_pos_pred),
            "ridge_rand_z": z(ridge_rand_pred),
        }
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s "
              f"(lgbm {t_lgbm:.0f}s, ridge_pos {t_ridge_pos:.0f}s, ridge_rand {t_ridge_rand:.0f}s)", flush=True)

    def evaluate_blend(blend):
        recs = []
        for fid, fd in fold_data.items():
            blended = np.zeros_like(fd["lgbm_z"])
            for k, w in blend.items():
                blended = blended + w * fd[k]
            df = evaluate_portfolio(fd["test"], blended, use_gate=True,
                                     gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                              "skipped": r["skipped"], "gross": r["spread_ret_bps"],
                              "cost": r["cost_bps"]})
        return pd.DataFrame(recs)

    print("\n" + "=" * 110, flush=True)
    print("MULTI-HEAD ARCHITECTURE v3 — robustness & noise tolerance", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'variant':<54} {'gross':>7} {'cost':>6} {'net':>7} {'Sharpe':>7} "
          f"{'95% CI':>15} {'Δ baseline':>12}")

    variants = [
        ("LGBM-only baseline",                                {"lgbm_z": 1.0}),
        ("LGBM + ridge_pos (w=0.10) — validated",              {"lgbm_z": 0.9, "ridge_pos_z": 0.10}),
        ("LGBM + ridge_random (w=0.10) — noise control",       {"lgbm_z": 0.9, "ridge_rand_z": 0.10}),
        ("LGBM + pos + random (w=0.05 each)",                  {"lgbm_z": 0.9, "ridge_pos_z": 0.05,
                                                                  "ridge_rand_z": 0.05}),
    ]
    base_sh = sharpe_est(evaluate_blend({"lgbm_z": 1.0})["net"].values)

    summary = {}
    for label, blend in variants:
        df = evaluate_blend(blend)
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d = sh - base_sh
        print(f"  {label:<54} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d:>+11.2f}", flush=True)
        summary[label] = {
            "n_cycles": int(len(df)), "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_vs_baseline": float(d),
        }

    with open(OUT_DIR / "alpha_v9_multi_head_v3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
