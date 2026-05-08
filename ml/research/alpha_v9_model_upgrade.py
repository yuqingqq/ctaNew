"""Test single-model architectures that might natively handle both signal scales.

Hybrid stacking (LGBM + Ridge blend at w=0.10) gave +0.43 Sharpe lift via
SEPARATE models for big and small signals. This test asks: can a single
better-designed model do the same?

Three architectures:

  (A) LGBM with linear_tree=True
      Each leaf fits a linear regression (instead of a constant). Trees
      handle dominant nonlinear signal; linear leaves can pick up small
      marginal signals like positioning. Native single-model integration.

  (B) LGBM with interaction_constraints
      Force v6_clean and positioning features into separate groups; LGBM
      can't combine them in one tree. Prevents positioning-split corruption.

  (C) LGBM with monotone_constraints on positioning
      Restrict positioning features to monotonic relationships only.
      Limits the model's ability to overfit on small signals.

All compared to:
  - LGBM-only on v6_clean (baseline +1.13)
  - LGBM on v6_clean + positioning (failed -3.32)
  - Hybrid stacking at w=0.10 (validated borderline +0.43)
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

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
OUT_DIR = REPO / "outputs/h48_model_upgrade"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def _train_with_extras(X_train, y_train, X_cal, y_cal, *, seed,
                         linear_tree=False,
                         interaction_constraints=None,
                         monotone_constraints=None):
    """Drop-in alternative trainer with optional architecture tweaks."""
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    if linear_tree:
        params["linear_tree"] = True
        params["linear_lambda"] = 0.001  # default ridge on leaf linear regression
    if interaction_constraints is not None:
        params["interaction_constraints"] = interaction_constraints
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dc = lgb.Dataset(X_cal, y_cal, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                      callbacks=[lgb.early_stopping(stopping_rounds=80),
                                  lgb.log_evaluation(period=0)])


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    combined = v6_clean + POS_3_RANK
    n_v6 = len(v6_clean)
    n_pos = len(POS_3_RANK)
    print(f"Multi-OOS folds: {len(folds)}")
    print(f"v6_clean features: {n_v6}, positioning features: {n_pos}, combined: {n_v6+n_pos}")

    # Build interaction constraints: v6_clean is one group, positioning is another
    v6_indices = list(range(n_v6))
    pos_indices = list(range(n_v6, n_v6 + n_pos))
    interaction_groups = [v6_indices, pos_indices]

    # Monotone constraints: 0 for v6_clean, +1 for each positioning feature
    # (this allows LGBM to use them but only in monotonic-increasing way)
    monotone_array = [0] * n_v6 + [1] * n_pos

    variants = [
        ("LGBM-only (v6_clean)",                    "v6", {}),
        ("LGBM (v6+pos) — known fail",               "combined", {}),
        ("LGBM linear_tree=True (v6+pos)",           "combined", {"linear_tree": True}),
        ("LGBM interaction_constraints (v6+pos)",    "combined",
            {"interaction_constraints": interaction_groups}),
        ("LGBM monotone_constraints (v6+pos)",       "combined",
            {"monotone_constraints": monotone_array}),
        ("LGBM linear_tree + interaction_constr",    "combined",
            {"linear_tree": True, "interaction_constraints": interaction_groups}),
    ]
    cycles: dict[str, list] = {v[0]: [] for v in variants}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue

        for tag, feat_set, params in variants:
            cols = v6_clean if feat_set == "v6" else combined
            avail = [c for c in cols if c in panel.columns]
            Xt = tr[avail].to_numpy(dtype=np.float32)
            yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
            Xc = ca[avail].to_numpy(dtype=np.float32)
            yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
            Xtest = test[avail].to_numpy(dtype=np.float32)

            try:
                models = [_train_with_extras(Xt, yt_, Xc, yc_, seed=seed, **params)
                          for seed in ENSEMBLE_SEEDS]
                yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                    for m in models], axis=0)
                df = evaluate_portfolio(test, yt_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                         use_magweight=False, top_k=TOP_K)
                for _, r in df.iterrows():
                    cycles[tag].append({
                        "fold": fold["fid"], "time": r["time"],
                        "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                        "net": r["net_bps"], "skipped": r["skipped"],
                    })
            except Exception as e:
                print(f"  {tag}: {e}")
                continue
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    print("\n" + "=" * 110)
    print("MODEL ARCHITECTURE UPGRADE TEST")
    print("=" * 110)
    print(f"  {'variant':<46} {'cycles':>7} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'Sharpe':>7} {'95% CI':>15} {'Δ vs v6':>8}")

    base_arr = np.array([r["net"] for r in cycles["LGBM-only (v6_clean)"]])
    base_sh = sharpe_est(base_arr)

    summary = {}
    for tag, *_ in variants:
        df = pd.DataFrame(cycles[tag])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d_vs_v6 = sh - base_sh
        print(f"  {tag:<46} {len(df):>7d} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_vs_v6:>+7.2f}")
        summary[tag] = {
            "n_cycles": int(len(df)), "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_vs_v6": float(d_vs_v6),
        }

    print(f"\n  Reference: hybrid stacking (LGBM+Ridge blend w=0.10) achieved ΔSharpe +0.43")
    print(f"  If any single-model upgrade matches +0.43, it's an architectural improvement.")

    with open(OUT_DIR / "alpha_v9_model_upgrade_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
