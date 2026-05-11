"""Ridge weight sweep with conv+PM gates active.

Tier 1A.1 follow-up: prior Ridge-blend weight (w=0.10) was tuned WITHOUT
the PM gate. With PM active, Ridge perturbs LGBM rankings → PM filters
those perturbations as noise → Ridge becomes counter-productive at w=0.10.

This script sweeps w ∈ {0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20} on the
SAME predictions per fold (LGBM and Ridge fit once per fold, then blended
at multiple w values during evaluation). Output: optimal w under conv+PM.

Plateau check: is the new optimum a sharp peak or a broad shelf?
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_hybrid_validate import predict_fold, POS_FEATURES
from ml.research.alpha_v9_positioning_pack import build_panel
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
CYCLES_PER_YEAR = (288 * 365) / HORIZON
W_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
OUT_DIR = REPO / "outputs/ridge_weight_with_pm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sharpe(x: np.ndarray) -> float:
    if len(x) == 0 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")
    print(f"W grid: {W_GRID}")
    print(f"Gates active: conv_p30 + PM_M2_b1")
    print()

    # Train once per fold, store (test, lgbm_z, ridge_z) for re-blending
    fold_data = {}
    for fold in folds:
        t0 = time.time()
        test, lgbm_z, ridge_z = predict_fold(panel, fold, v6_clean, POS_FEATURES)
        if test is None:
            print(f"  fold {fold['fid']:>2}: skipped"); continue
        fold_data[fold["fid"]] = {"test": test, "lgbm_z": lgbm_z, "ridge_z": ridge_z}
        print(f"  fold {fold['fid']:>2}: trained ({time.time()-t0:.0f}s)")

    print(f"\n  {len(fold_data)} folds with predictions stored. Now evaluating w-grid...")

    # Evaluate each w with conv+PM active
    results: dict[float, list] = {w: [] for w in W_GRID}
    # Also track conv-only and baseline (no gate) at each w for context
    results_baseline: dict[float, list] = {w: [] for w in W_GRID}
    results_convonly: dict[float, list] = {w: [] for w in W_GRID}

    for fid, fd in fold_data.items():
        test = fd["test"]
        for w in W_GRID:
            blended = (1 - w) * fd["lgbm_z"] + w * fd["ridge_z"]
            # baseline
            df_b = evaluate_stacked(test, blended, use_conv_gate=False, use_pm_gate=False)
            results_baseline[w].extend(df_b["net_bps"].tolist())
            # conv only
            df_c = evaluate_stacked(test, blended, use_conv_gate=True, use_pm_gate=False)
            results_convonly[w].extend(df_c["net_bps"].tolist())
            # conv + PM stacked
            df_s = evaluate_stacked(test, blended, use_conv_gate=True, use_pm_gate=True)
            for _, row in df_s.iterrows():
                results[w].append({
                    "fold": fid, "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                })

    # ===== Headline sweep table =====
    print("\n" + "=" * 100)
    print("RIDGE WEIGHT SWEEP under conv+PM  (10 folds, 1800 cycles)")
    print("=" * 100)
    print(f"  {'w':>6}  {'baseline_sh':>11}  {'conv_only_sh':>13}  {'conv+PM_sh':>11}  "
          f"{'conv+PM_net':>11}  {'CI_lo':>7}  {'CI_hi':>7}  {'K_avg':>5}")
    rows = []
    for w in W_GRID:
        b = np.array(results_baseline[w])
        c = np.array(results_convonly[w])
        df_s = pd.DataFrame(results[w])
        s = df_s["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(s, statistic=_sharpe, block_size=7, n_boot=2000)
        K_avg = (df_s["n_long"].mean() + df_s["n_short"].mean()) / 2
        print(f"  {w:>6.3f}  {_sharpe(b):>+11.2f}  {_sharpe(c):>+13.2f}  "
              f"{sh:>+11.2f}  {s.mean():>+11.2f}  {lo:>+7.2f}  {hi:>+7.2f}  {K_avg:>5.2f}")
        rows.append({"w": w, "baseline_sh": _sharpe(b),
                     "conv_only_sh": _sharpe(c), "conv_pm_sh": sh,
                     "conv_pm_net": s.mean(), "ci_lo": lo, "ci_hi": hi,
                     "K_avg": K_avg})
    summary = pd.DataFrame(rows)

    # ===== Best w =====
    best_idx = summary["conv_pm_sh"].idxmax()
    best_w = summary.loc[best_idx, "w"]
    best_sh = summary.loc[best_idx, "conv_pm_sh"]
    print(f"\n  Best w under conv+PM: w={best_w}  Sharpe={best_sh:+.2f}")
    near_best = summary[summary["conv_pm_sh"] >= best_sh - 0.20]["w"].tolist()
    print(f"  W within 0.20 of best (plateau): {near_best}")

    # ===== Paired comparison: each w vs w=0 (Ridge dropped) =====
    print(f"\n  Paired Δ vs w=0.0 (Ridge dropped):")
    base_arr = np.array([r["net"] for r in results[0.0]])
    for w in W_GRID:
        if w == 0.0: continue
        v_arr = np.array([r["net"] for r in results[w]])
        n_min = min(len(base_arr), len(v_arr))
        delta = v_arr[:n_min] - base_arr[:n_min]
        rng = np.random.default_rng(42)
        n_boot = 2000; block = 7
        n_blocks = int(np.ceil(len(delta) / block))
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
            boot_means[i] = delta[idx].mean()
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        d_sh = _sharpe(v_arr[:n_min]) - _sharpe(base_arr[:n_min])
        sig_lo = "✓" if lo > 0 else (" " if hi > 0 else "✗")
        print(f"    w={w:>5.3f}  Δnet={delta.mean():+.3f} bps  CI=[{lo:+.3f}, {hi:+.3f}]{sig_lo}  Δsh={d_sh:+.2f}")

    summary.to_csv(OUT_DIR / "ridge_weight_sweep.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
