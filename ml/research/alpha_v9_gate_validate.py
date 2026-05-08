"""Robustness validation for conviction-gate p=0.30 finding.

The headline result was ΔSharpe +1.85 (p=0.056) vs sharp baseline. Before
deploying, we want to verify:

  V1. Per-fold consistency. Does the lift come from all 9 folds or 1-2 outliers?
  V2. Fine-grained pctile plateau. Is p=0.30 a peak (suggests overfit) or
       part of a smooth plateau (robust)?
  V3. Hard-split frozen test. Train ONCE on early data, freeze model+gate
       state, evaluate on late data. Tests structural robustness — no
       retraining "rescue."
  V4. Block-bootstrap 95% CI on ΔSharpe. Does the lift have a CI that
       excludes zero?
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

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
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
OUT_DIR = REPO / "outputs/h48_gate_validate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def predict_fold(panel, fold, v6_clean):
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200:
        return None, None, None
    avail = [c for c in v6_clean if c in panel.columns]
    Xt = tr[avail].to_numpy(dtype=np.float32)
    yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
    Xc = ca[avail].to_numpy(dtype=np.float32)
    yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
    models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    Xtest = test[avail].to_numpy(dtype=np.float32)
    yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                        for m in models], axis=0)
    return models, test, yt_pred


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    # ===== Cache predictions (5 seeds × 9 folds) once =====
    fold_data = {}
    for fold in folds:
        t0 = time.time()
        models, test, yt_pred = predict_fold(panel, fold, v6_clean)
        if models is None:
            continue
        fold_data[fold["fid"]] = {"test": test, "preds": yt_pred}
        print(f"  fold {fold['fid']}: trained, test={len(test):,} rows ({time.time()-t0:.0f}s)")

    # ===== V1. Per-fold consistency =====
    print("\n" + "=" * 100)
    print("V1. PER-FOLD CONSISTENCY")
    print("=" * 100)
    print(f"  {'fold':>4} {'cycles':>7} {'base_net':>9} {'gate_net':>9} "
          f"{'Δnet':>7} {'base_Sh':>8} {'gate_Sh':>8} {'ΔSh':>7} {'%skip':>6}")

    per_fold_records = []
    for fid, fd in fold_data.items():
        base_df = evaluate_portfolio(fd["test"], fd["preds"], use_gate=False,
                                      gate_pctile=0.30, use_magweight=False)
        gate_df = evaluate_portfolio(fd["test"], fd["preds"], use_gate=True,
                                      gate_pctile=0.30, use_magweight=False)
        base_net = base_df["net_bps"].to_numpy()
        gate_net = gate_df["net_bps"].to_numpy()
        base_sh = sharpe_est(base_net)
        gate_sh = sharpe_est(gate_net)
        d_sh = gate_sh - base_sh
        d_net = gate_net.mean() - base_net.mean()
        pct_skip = 100 * gate_df["skipped"].mean()
        print(f"  {fid:>4d} {len(base_df):>7d} {base_net.mean():>+8.2f} {gate_net.mean():>+8.2f} "
              f"{d_net:>+6.2f} {base_sh:>+7.2f} {gate_sh:>+7.2f} {d_sh:>+6.2f} {pct_skip:>5.1f}%")
        per_fold_records.append({"fold": fid, "base_sh": base_sh, "gate_sh": gate_sh,
                                  "delta_sh": d_sh, "delta_net": d_net,
                                  "pct_skip": pct_skip})
    pf = pd.DataFrame(per_fold_records)
    print(f"  {'mean':>4} {'':>7} {'':>9} {'':>9} {pf['delta_net'].mean():>+6.2f} "
          f"{pf['base_sh'].mean():>+7.2f} {pf['gate_sh'].mean():>+7.2f} "
          f"{pf['delta_sh'].mean():>+6.2f} {pf['pct_skip'].mean():>5.1f}%")
    print(f"\n  folds with positive Δ Sharpe: {(pf['delta_sh'] > 0).sum()}/{len(pf)}")
    print(f"  median ΔSharpe across folds: {pf['delta_sh'].median():+.2f}")
    print(f"  std of ΔSharpe across folds: {pf['delta_sh'].std():.2f}")

    # ===== V2. Fine-grained pctile sweep =====
    print("\n" + "=" * 100)
    print("V2. FINE-GRAINED PCTILE PLATEAU CHECK")
    print("=" * 100)
    pctiles = [0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50]
    print(f"  {'p':>5} {'%trade':>7} {'gross':>7} {'cost':>7} {'net':>7} {'Sharpe':>8} {'ΔSh':>7}")
    base_recs = []
    for fid, fd in fold_data.items():
        base_df = evaluate_portfolio(fd["test"], fd["preds"], use_gate=False,
                                      gate_pctile=0.30, use_magweight=False)
        for _, r in base_df.iterrows():
            base_recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"]})
    base_arr = pd.DataFrame(base_recs)
    base_sh_overall = sharpe_est(base_arr["net"].to_numpy())

    plateau_records = []
    for p in pctiles:
        all_net = []
        traded_gross = []
        traded_cost = []
        traded_count = 0
        total_count = 0
        for fid, fd in fold_data.items():
            df = evaluate_portfolio(fd["test"], fd["preds"], use_gate=True,
                                     gate_pctile=p, use_magweight=False)
            all_net.extend(df["net_bps"].tolist())
            traded = df[df["skipped"] == 0]
            traded_gross.extend(traded["spread_ret_bps"].tolist())
            traded_cost.extend(traded["cost_bps"].tolist())
            traded_count += len(traded)
            total_count += len(df)
        net_arr = np.array(all_net)
        sh = sharpe_est(net_arr)
        pct_trade = 100 * traded_count / total_count
        gross = np.mean(traded_gross) if traded_gross else 0
        cost = np.mean(traded_cost) if traded_cost else 0
        d_sh = sh - base_sh_overall
        plateau_records.append({"p": p, "sharpe": sh, "delta_sh": d_sh,
                                 "pct_trade": pct_trade, "gross": gross, "cost": cost,
                                 "net": net_arr.mean()})
        print(f"  {p:>5.2f} {pct_trade:>6.1f}% {gross:>+6.2f}  {cost:>6.2f}  "
              f"{net_arr.mean():>+6.2f}  {sh:>+7.2f}  {d_sh:>+6.2f}")
    pl = pd.DataFrame(plateau_records)
    best = pl.loc[pl["sharpe"].idxmax()]
    print(f"\n  best p: {best['p']:.2f}, Sharpe {best['sharpe']:+.2f}")
    print(f"  Sharpe within 0.20 of best: p ∈ "
          f"{sorted(pl[pl['sharpe'] > best['sharpe'] - 0.20]['p'].tolist())}")

    # ===== V3. Hard-split frozen test =====
    print("\n" + "=" * 100)
    print("V3. HARD-SPLIT FROZEN TEST (train on early folds, test on late)")
    print("=" * 100)
    n_train_folds = max(3, len(fold_data) // 2)
    train_fids = list(fold_data.keys())[:n_train_folds]
    test_fids = list(fold_data.keys())[n_train_folds:]
    print(f"  train folds (model freeze): {train_fids}")
    print(f"  test folds (frozen evaluation): {test_fids}")

    # Train ONCE on combined train_fids data
    panel_train = panel[panel["open_time"] < fold_data[train_fids[-1]]["test"]["open_time"].max()]
    panel_train_filt = panel_train[panel_train["autocorr_pctile_7d"] >= THRESHOLD]
    avail = [c for c in v6_clean if c in panel.columns]
    n_train = len(panel_train_filt)
    train_split = int(n_train * 0.85)
    Xt = panel_train_filt[avail].iloc[:train_split].to_numpy(dtype=np.float32)
    yt_ = panel_train_filt["demeaned_target"].iloc[:train_split].to_numpy(dtype=np.float32)
    Xc = panel_train_filt[avail].iloc[train_split:].to_numpy(dtype=np.float32)
    yc_ = panel_train_filt["demeaned_target"].iloc[train_split:].to_numpy(dtype=np.float32)
    print(f"  frozen training: {train_split:,} train rows, {n_train - train_split:,} cal rows")
    frozen_models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]

    # Evaluate frozen model on each test fold
    print(f"  {'fold':>4} {'cycles':>7} {'base_net':>9} {'gate_net':>9} {'base_Sh':>8} {'gate_Sh':>8} {'ΔSh':>7}")
    hard_base_all = []
    hard_gate_all = []
    for fid in test_fids:
        test = fold_data[fid]["test"]
        Xtest = test[avail].to_numpy(dtype=np.float32)
        yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in frozen_models], axis=0)
        base_df = evaluate_portfolio(test, yt_pred, use_gate=False,
                                      gate_pctile=0.30, use_magweight=False)
        gate_df = evaluate_portfolio(test, yt_pred, use_gate=True,
                                      gate_pctile=0.30, use_magweight=False)
        b_net = base_df["net_bps"].to_numpy()
        g_net = gate_df["net_bps"].to_numpy()
        hard_base_all.extend(b_net.tolist())
        hard_gate_all.extend(g_net.tolist())
        print(f"  {fid:>4d} {len(b_net):>7d} {b_net.mean():>+8.2f} {g_net.mean():>+8.2f} "
              f"{sharpe_est(b_net):>+7.2f} {sharpe_est(g_net):>+7.2f} "
              f"{sharpe_est(g_net) - sharpe_est(b_net):>+6.2f}")
    hard_b = np.array(hard_base_all)
    hard_g = np.array(hard_gate_all)
    print(f"\n  Hard-split overall (frozen model, no retrain):")
    print(f"    base: net {hard_b.mean():+.2f} bps/cyc, Sharpe {sharpe_est(hard_b):+.2f}")
    print(f"    gate: net {hard_g.mean():+.2f} bps/cyc, Sharpe {sharpe_est(hard_g):+.2f}")
    print(f"    delta: ΔSharpe {sharpe_est(hard_g) - sharpe_est(hard_b):+.2f}, "
          f"Δnet {hard_g.mean() - hard_b.mean():+.2f} bps/cyc")

    # ===== V4. Block-bootstrap CI on ΔSharpe =====
    print("\n" + "=" * 100)
    print("V4. BLOCK-BOOTSTRAP 95% CI ON Δ SHARPE (full multi-OOS)")
    print("=" * 100)
    base_full = []
    gate_full = []
    for fid, fd in fold_data.items():
        base_df = evaluate_portfolio(fd["test"], fd["preds"], use_gate=False,
                                      gate_pctile=0.30, use_magweight=False)
        gate_df = evaluate_portfolio(fd["test"], fd["preds"], use_gate=True,
                                      gate_pctile=0.30, use_magweight=False)
        base_full.extend(base_df["net_bps"].tolist())
        gate_full.extend(gate_df["net_bps"].tolist())
    base_arr = np.array(base_full)
    gate_arr = np.array(gate_full)
    delta = gate_arr - base_arr

    rng = np.random.default_rng(42)
    n = len(delta)
    block = 7
    n_boot = 5000
    n_blocks = int(np.ceil(n / block))
    delta_sh_boot = np.empty(n_boot)
    base_sh_boot = np.empty(n_boot)
    gate_sh_boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        delta_sh_boot[i] = sharpe_est(delta[idx])
        base_sh_boot[i] = sharpe_est(base_arr[idx])
        gate_sh_boot[i] = sharpe_est(gate_arr[idx])

    delta_sh_pt = sharpe_est(delta)
    base_sh_pt = sharpe_est(base_arr)
    gate_sh_pt = sharpe_est(gate_arr)
    ci_d = np.quantile(delta_sh_boot, [0.025, 0.975])
    ci_b = np.quantile(base_sh_boot, [0.025, 0.975])
    ci_g = np.quantile(gate_sh_boot, [0.025, 0.975])
    pos_pct = (delta_sh_boot > 0).mean() * 100

    print(f"  Baseline Sharpe: {base_sh_pt:+.2f}  CI [{ci_b[0]:+.2f}, {ci_b[1]:+.2f}]")
    print(f"  Gate Sharpe:     {gate_sh_pt:+.2f}  CI [{ci_g[0]:+.2f}, {ci_g[1]:+.2f}]")
    print(f"  ΔSharpe:         {delta_sh_pt:+.2f}  CI [{ci_d[0]:+.2f}, {ci_d[1]:+.2f}]")
    print(f"  P(ΔSharpe > 0) by bootstrap: {pos_pct:.1f}%")
    print(f"  P(ΔSharpe > 0.5): {(delta_sh_boot > 0.5).mean()*100:.1f}%")
    print(f"  P(ΔSharpe > 1.0): {(delta_sh_boot > 1.0).mean()*100:.1f}%")

    # Save summary
    summary = {
        "v1_per_fold": pf.to_dict("records"),
        "v1_folds_positive_delta": int((pf["delta_sh"] > 0).sum()),
        "v1_total_folds": int(len(pf)),
        "v1_median_delta_sharpe": float(pf["delta_sh"].median()),
        "v2_plateau": pl.to_dict("records"),
        "v2_best_p": float(best["p"]),
        "v2_best_sharpe": float(best["sharpe"]),
        "v3_hard_split": {
            "train_folds": train_fids, "test_folds": test_fids,
            "base_sharpe": float(sharpe_est(hard_b)),
            "gate_sharpe": float(sharpe_est(hard_g)),
            "delta_sharpe": float(sharpe_est(hard_g) - sharpe_est(hard_b)),
            "delta_net_bps": float(hard_g.mean() - hard_b.mean()),
        },
        "v4_bootstrap": {
            "base_sharpe_pt": float(base_sh_pt), "base_sharpe_ci": ci_b.tolist(),
            "gate_sharpe_pt": float(gate_sh_pt), "gate_sharpe_ci": ci_g.tolist(),
            "delta_sharpe_pt": float(delta_sh_pt), "delta_sharpe_ci": ci_d.tolist(),
            "p_positive_by_bootstrap_pct": float(pos_pct),
        },
    }
    with open(OUT_DIR / "alpha_v9_gate_validate_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
