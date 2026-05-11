"""Hybrid test for aggTrade flow features (pack J).

Linear oracle showed:
  standalone IC +0.0317, marginal Δ IC +0.0013, pred_corr +0.181 → ORTHOGONAL.

Prior direct LGBM tests of aggTrade flow showed -0.04 (wash) Sharpe — same
failure mode as positioning pack 1 in direct LGBM (-1.42), which the hybrid
rescued to +0.43.

This test: train Ridge on the 5 aggTrade features, blend with LGBM at
w ∈ {0.05, 0.10, 0.15}. Compare to LGBM-only and to LGBM + Ridge_pos1 baseline.

Also tests COMPOSITION: LGBM + Ridge_pos1 + Ridge_aggTrade (both ridges
active simultaneously).
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
from ml.research.alpha_v9_redundancy_full import build_panel as build_full_panel

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
POS_PACK_1 = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
AGGTRADE_5 = ["signed_volume_4h_xs_rank", "tfi_4h_xs_rank", "aggr_ratio_4h_xs_rank",
                "buy_count_4h_xs_rank", "avg_trade_size_4h_xs_rank"]
OUT_DIR = REPO / "outputs/h48_aggtrade_hybrid"
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
    print("Note: redundancy_full panel (with v6_clean + funding + OI + aggTrade + L/S) "
          "may need positioning pack 1 features merged in.", flush=True)
    panel = build_full_panel()

    # Check if positioning pack 1 features (funding_z_24h_xs_rank etc.) exist
    missing_pos = [c for c in POS_PACK_1 if c not in panel.columns]
    if missing_pos:
        print(f"  Note: pack 1 features {missing_pos} not in panel. "
              f"Cannot run pack 1 + aggTrade composition test.", flush=True)
        print(f"  Will run aggTrade alone vs LGBM-only baseline.", flush=True)

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

        # LGBM on v6_clean
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])

        # Ridge on aggTrade features
        avail_at = [c for c in AGGTRADE_5 if c in panel.columns]
        if len(avail_at) >= 3:
            X_full_at = np.vstack([tr[avail_at].to_numpy(dtype=np.float64),
                                     ca[avail_at].to_numpy(dtype=np.float64)])
            X_te_at = test[avail_at].to_numpy(dtype=np.float64)
            ridge_at_pred = fit_predict_ridge(X_full_at, y_full, X_te_at)
        else:
            ridge_at_pred = np.zeros(len(test))

        # Ridge on positioning pack 1 (if features exist)
        avail_pos = [c for c in POS_PACK_1 if c in panel.columns]
        if len(avail_pos) >= 3:
            X_full_pos = np.vstack([tr[avail_pos].to_numpy(dtype=np.float64),
                                      ca[avail_pos].to_numpy(dtype=np.float64)])
            X_te_pos = test[avail_pos].to_numpy(dtype=np.float64)
            ridge_pos_pred = fit_predict_ridge(X_full_pos, y_full, X_te_pos)
        else:
            ridge_pos_pred = None

        fold_data[fold["fid"]] = {
            "test": test,
            "lgbm_z": z(lgbm_pred),
            "ridge_at_z": z(ridge_at_pred),
            "ridge_pos_z": z(ridge_pos_pred) if ridge_pos_pred is not None else None,
        }
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s", flush=True)

    have_pos = any(fd["ridge_pos_z"] is not None for fd in fold_data.values())

    def evaluate_blend(blend):
        recs = []
        for fid, fd in fold_data.items():
            blended = np.zeros_like(fd["lgbm_z"])
            for k, w in blend.items():
                if k == "ridge_pos_z" and fd[k] is None:
                    continue
                blended = blended + w * fd[k]
            df = evaluate_portfolio(fd["test"], blended, use_gate=True,
                                     gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                              "skipped": r["skipped"], "gross": r["spread_ret_bps"],
                              "cost": r["cost_bps"]})
        return pd.DataFrame(recs)

    print("\n" + "=" * 110, flush=True)
    print("AGGTRADE HYBRID TEST", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'variant':<48} {'gross':>7} {'cost':>6} {'net':>7} {'Sharpe':>7} {'95% CI':>15} {'Δgate':>7}", flush=True)

    base_arr = evaluate_blend({"lgbm_z": 1.0})
    base_sh = sharpe_est(base_arr["net"].values)

    variants = [
        ("LGBM-only baseline",                                  {"lgbm_z": 1.0}),
        ("LGBM + ridge_aggTrade (w=0.05)",                       {"lgbm_z": 0.95, "ridge_at_z": 0.05}),
        ("LGBM + ridge_aggTrade (w=0.10)",                       {"lgbm_z": 0.90, "ridge_at_z": 0.10}),
        ("LGBM + ridge_aggTrade (w=0.15)",                       {"lgbm_z": 0.85, "ridge_at_z": 0.15}),
    ]
    if have_pos:
        variants.extend([
            ("LGBM + ridge_pos (validated, w=0.10)",                {"lgbm_z": 0.90, "ridge_pos_z": 0.10}),
            ("LGBM + ridge_pos + ridge_aggTrade (each w=0.05)",     {"lgbm_z": 0.90, "ridge_pos_z": 0.05, "ridge_at_z": 0.05}),
            ("LGBM + ridge_pos + ridge_aggTrade (pos=0.10, at=0.05)", {"lgbm_z": 0.85, "ridge_pos_z": 0.10, "ridge_at_z": 0.05}),
            ("LGBM + ridge_pos + ridge_aggTrade (pos=0.10, at=0.10)", {"lgbm_z": 0.80, "ridge_pos_z": 0.10, "ridge_at_z": 0.10}),
        ])

    summary = {}
    for label, blend in variants:
        df = evaluate_blend(blend)
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d = sh - base_sh
        print(f"  {label:<48} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d:>+6.2f}", flush=True)
        summary[label] = {
            "n_cycles": int(len(df)),
            "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_vs_baseline": float(d),
        }

    with open(OUT_DIR / "alpha_v9_aggtrade_hybrid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
