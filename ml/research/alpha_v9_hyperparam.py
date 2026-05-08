"""LGBM hyperparameter sweep under post-fix cost surface.

Current production (pinned since v3 era):
    num_leaves=63, max_depth=8, min_data_in_leaf=100,
    learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, lambda_l2=3.0

Sweep regularization triple:
    num_leaves     ∈ {31, 63, 127}
    min_data_leaf  ∈ {50, 100, 200}
    lambda_l2      ∈ {1.0, 3.0, 10.0}

= 27 combinations. K=7, n_seeds=5, conv_gate p=0.30 held fixed.
Multi-OOS, post-fix cost. Compare to baseline +1.47 Sharpe.

Hypothesis: the "tails-preserve, smoothing-hurts" pattern from prior
tests suggests current regularization may be slightly too aggressive
or too lax. A grid sweep will find the actual optimum.
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
    _multi_oos_splits, _slice, ENSEMBLE_SEEDS,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
NUM_LEAVES_GRID = [31, 63, 127]
MIN_DATA_LEAF_GRID = [50, 100, 200]
LAMBDA_L2_GRID = [1.0, 3.0, 10.0]
OUT_DIR = REPO / "outputs/h48_hyperparam"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def _train_with_params(X_train, y_train, X_cal, y_cal, *, seed,
                        num_leaves, min_data_in_leaf, lambda_l2):
    """Drop-in replacement for _train with overridable hyperparameters."""
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=num_leaves, max_depth=8,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=lambda_l2, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dc = lgb.Dataset(X_cal, y_cal, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                      callbacks=[lgb.early_stopping(stopping_rounds=80),
                                  lgb.log_evaluation(period=0)])


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    combos = [(nl, ml, l2)
              for nl in NUM_LEAVES_GRID
              for ml in MIN_DATA_LEAF_GRID
              for l2 in LAMBDA_L2_GRID]
    print(f"Hyperparameter combos: {len(combos)}")

    cell_records: dict[tuple, list] = {c: [] for c in combos}
    avg_best_iter: dict[tuple, list] = {c: [] for c in combos}

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

        for nl, ml, l2 in combos:
            models = [_train_with_params(Xt, yt_, Xc, yc_, seed=seed,
                                           num_leaves=nl, min_data_in_leaf=ml,
                                           lambda_l2=l2)
                      for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            avg_best_iter[(nl, ml, l2)].append(np.mean([m.best_iteration for m in models]))
            df = evaluate_portfolio(
                test, yt_pred,
                use_gate=True, gate_pctile=GATE_PCTILE,
                use_magweight=False, top_k=TOP_K,
            )
            for _, r in df.iterrows():
                cell_records[(nl, ml, l2)].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"],
                    "cost": r["cost_bps"], "net": r["net_bps"],
                    "long_turn": r["long_turnover"], "skipped": r["skipped"],
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s ({len(combos)} combos)")

    # Summarize
    print("\n" + "=" * 130)
    print(f"HYPERPARAMETER SWEEP — Sharpe (post-fix cost {COST_PER_LEG}, K={TOP_K}, "
          f"n_seeds=5, conv_gate p={GATE_PCTILE})")
    print("=" * 130)

    # Detailed sorted table
    results = []
    for c in combos:
        recs = pd.DataFrame(cell_records[c])
        if recs.empty:
            results.append({"params": c, "sharpe": np.nan})
            continue
        traded = recs[recs["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(recs["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=1500)
        results.append({
            "params": c, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "gross": traded["gross"].mean() if len(traded) > 0 else 0,
            "cost": traded["cost"].mean() if len(traded) > 0 else 0,
            "net": recs["net"].mean(),
            "long_turn": traded["long_turn"].mean() if len(traded) > 0 else 0,
            "best_iter": np.mean(avg_best_iter[c]),
        })

    results_sorted = sorted(results, key=lambda r: -r["sharpe"] if not np.isnan(r["sharpe"]) else 0)
    print(f"  {'rank':>4} {'num_leaves':>10} {'min_data':>9} {'lambda_l2':>10} "
          f"{'best_iter':>10} {'gross':>7} {'cost':>6} {'net':>7} {'L_turn':>7} "
          f"{'Sharpe':>7} {'95% CI':>15}")
    for i, r in enumerate(results_sorted):
        nl, ml, l2 = r["params"]
        is_baseline = (nl == 63 and ml == 100 and l2 == 3.0)
        marker = " ←baseline" if is_baseline else ""
        print(f"  {i+1:>4d} {nl:>10d} {ml:>9d} {l2:>10.1f} "
              f"{r['best_iter']:>10.1f} "
              f"{r['gross']:>+6.2f}  {r['cost']:>5.2f}  {r['net']:>+6.2f}  "
              f"{r['long_turn']:>6.0%}  {r['sharpe']:>+6.2f}  "
              f"[{r['ci_lo']:>+5.2f},{r['ci_hi']:>+5.2f}]{marker}")

    # Distillation: best by num_leaves, by min_data, by lambda_l2
    print(f"\n  --- MARGINALS ---")
    for axis_name, grid in [("num_leaves", NUM_LEAVES_GRID),
                             ("min_data_in_leaf", MIN_DATA_LEAF_GRID),
                             ("lambda_l2", LAMBDA_L2_GRID)]:
        print(f"  {axis_name}:")
        for v in grid:
            sharpes = [r["sharpe"] for r in results
                       if (axis_name == "num_leaves" and r["params"][0] == v)
                          or (axis_name == "min_data_in_leaf" and r["params"][1] == v)
                          or (axis_name == "lambda_l2" and r["params"][2] == v)]
            if sharpes:
                print(f"    {v:>5}:  mean Sharpe {np.mean(sharpes):+.2f}  "
                      f"max {max(sharpes):+.2f}  min {min(sharpes):+.2f}")

    summary = {
        "results": [{"params": list(r["params"]), **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                                       for k, v in r.items() if k != "params"}}
                     for r in results_sorted],
        "baseline_sharpe": next((r["sharpe"] for r in results
                                  if r["params"] == (63, 100, 3.0)), None),
    }
    with open(OUT_DIR / "alpha_v9_hyperparam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
