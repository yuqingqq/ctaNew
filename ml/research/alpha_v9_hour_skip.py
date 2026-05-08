"""Test skipping the dead hours {4, 16} UTC discovered by per-hour diagnostic.

Validates against selection bias via two methods:

  V1. In-sample sweep — skip various hour combinations on full data.
      Establishes upper bound and confirms {4, 16} is the right subset.

  V2. Hard-split validation — discover dead hours on early folds (0-3),
      apply discovered filter to late folds (4-8). Tests whether the
      effect is persistent or sample-specific.
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
from ml.research.alpha_v9_funding_session import (
    add_market_funding, evaluate_with_funding_session,
)

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/h48_hour_skip"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def evaluate_skip_hours(test, yt_pred, *, skip_hours: set):
    """Wrap evaluate_with_funding_session to skip given hours instead of keep."""
    # hour_filter expects KEEP set; convert skip set to keep set
    all_hours = {0, 4, 8, 12, 16, 20}
    keep = all_hours - skip_hours
    return evaluate_with_funding_session(
        test, yt_pred, use_conv_gate=True, use_funding_gate=False,
        hour_filter=keep,
    )


def predict_fold(panel, fold, v6_clean):
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200:
        return None, None
    avail = [c for c in v6_clean if c in panel.columns]
    Xt = tr[avail].to_numpy(dtype=np.float32)
    yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
    Xc = ca[avail].to_numpy(dtype=np.float32)
    yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
    models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    Xtest = test[avail].to_numpy(dtype=np.float32)
    yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                        for m in models], axis=0)
    return test, yt_pred


def main():
    panel = build_wide_panel()
    panel = add_market_funding(panel)
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    fold_data = {}
    for fold in folds:
        t0 = time.time()
        test, yt_pred = predict_fold(panel, fold, v6_clean)
        if test is None: continue
        fold_data[fold["fid"]] = {"test": test, "preds": yt_pred}
        print(f"  fold {fold['fid']}: trained ({time.time()-t0:.0f}s)")

    # ===== V1. In-sample sweep =====
    print("\n" + "=" * 100)
    print("V1. IN-SAMPLE SKIP-HOUR SWEEP")
    print("=" * 100)
    skip_combos = [
        ("conv_gate (production)",  set()),
        ("skip {16}",               {16}),
        ("skip {4}",                {4}),
        ("skip {4, 16}",            {4, 16}),
        ("skip {16, 4, 8}",         {4, 8, 16}),
        ("skip {16, 4, 8, 12}",     {4, 8, 12, 16}),
    ]
    cycles_v1: dict[str, list] = {}
    for label, skip in skip_combos:
        cycles_v1[label] = []
        for fid, fd in fold_data.items():
            if skip:
                df = evaluate_skip_hours(fd["test"], fd["preds"], skip_hours=skip)
            else:
                df = evaluate_with_funding_session(fd["test"], fd["preds"],
                                                    use_conv_gate=True)
            for _, r in df.iterrows():
                cycles_v1[label].append({
                    "fold": fid, "time": r["time"], "net": r["net_bps"],
                    "skipped": r["skipped"], "hour": r["hour"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "long_turn": r["long_turnover"],
                })
    print(f"  {'variant':<26} {'n_cyc':>5} {'%trade':>7} {'gross':>7} {'cost':>6} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>15} {'Δgate':>7}")
    base_recs = pd.DataFrame(cycles_v1["conv_gate (production)"])
    for label, _ in skip_combos:
        df = pd.DataFrame(cycles_v1[label])
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        m = base_recs[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_g = sharpe_est((m["net"] - m["base"]).to_numpy())
        print(f"  {label:<26} {len(df):>5d} {100*len(traded)/len(df):>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_g:>+6.2f}")

    # ===== V2. Hard-split: discover bad hours on early folds, apply to late =====
    print("\n" + "=" * 100)
    print("V2. HARD-SPLIT VALIDATION (discover on folds 0-3, apply to folds 4-8)")
    print("=" * 100)
    train_fids = list(fold_data.keys())[:max(3, len(fold_data) // 2)]
    test_fids = list(fold_data.keys())[max(3, len(fold_data) // 2):]
    print(f"  discovery folds: {train_fids}, test folds: {test_fids}")

    # Discovery: per-hour Sharpe on train_fids
    train_recs = []
    for fid in train_fids:
        df = evaluate_with_funding_session(fold_data[fid]["test"], fold_data[fid]["preds"],
                                            use_conv_gate=True)
        for _, r in df.iterrows():
            train_recs.append({"fold": fid, "hour": r["hour"], "net": r["net_bps"]})
    tdf = pd.DataFrame(train_recs)
    print(f"\n  Per-hour Sharpe on discovery folds (training):")
    discovery = []
    for h in sorted(tdf["hour"].unique()):
        sub = tdf[tdf["hour"] == h]
        sh = sharpe_est(sub["net"].to_numpy())
        discovery.append({"hour": h, "sharpe": sh})
        print(f"    hour {h:>2d}: Sharpe {sh:+.2f}, cycles {len(sub)}")
    bad_hours = {d["hour"] for d in discovery if d["sharpe"] < 0.5}
    print(f"\n  Discovered bad hours (Sharpe < +0.5): {sorted(bad_hours)}")

    # Apply: evaluate test_fids with bad_hours skipped
    print(f"\n  Applying to test folds (no peeking):")
    print(f"  {'fold':>4} {'cycles':>7} {'gate_net':>9} {'skip_net':>9} {'gate_Sh':>8} {'skip_Sh':>8} {'ΔSh':>7}")
    test_g, test_s = [], []
    for fid in test_fids:
        df_g = evaluate_with_funding_session(fold_data[fid]["test"], fold_data[fid]["preds"],
                                               use_conv_gate=True)
        df_s = evaluate_skip_hours(fold_data[fid]["test"], fold_data[fid]["preds"],
                                     skip_hours=bad_hours) if bad_hours else df_g
        g_net = df_g["net_bps"].to_numpy()
        s_net = df_s["net_bps"].to_numpy()
        test_g.extend(g_net.tolist())
        test_s.extend(s_net.tolist())
        print(f"  {fid:>4d} {len(g_net):>7d} {g_net.mean():>+8.2f} {s_net.mean():>+8.2f} "
              f"{sharpe_est(g_net):>+7.2f} {sharpe_est(s_net):>+7.2f} "
              f"{sharpe_est(s_net) - sharpe_est(g_net):>+6.2f}")
    g_arr, s_arr = np.array(test_g), np.array(test_s)
    print(f"\n  Hard-split overall (applying discovery to test folds):")
    print(f"    gate-only:        net {g_arr.mean():+.2f}, Sharpe {sharpe_est(g_arr):+.2f}")
    print(f"    + skip {sorted(bad_hours)}: net {s_arr.mean():+.2f}, Sharpe {sharpe_est(s_arr):+.2f}")
    print(f"    delta:            ΔSharpe {sharpe_est(s_arr) - sharpe_est(g_arr):+.2f}")

    # Bootstrap CI on Δ Sharpe
    delta_arr = s_arr - g_arr
    rng = np.random.default_rng(42)
    n = len(g_arr); block = 7; n_boot = 5000
    n_blocks = int(np.ceil(n / block))
    d_boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        d_boot[i] = sharpe_est(s_arr[idx]) - sharpe_est(g_arr[idx])
    print(f"    bootstrap CI on ΔSharpe: [{np.quantile(d_boot, 0.025):+.2f}, "
          f"{np.quantile(d_boot, 0.975):+.2f}]")
    print(f"    P(ΔSharpe > 0): {(d_boot > 0).mean()*100:.1f}%")
    print(f"    P(ΔSharpe > 0.20): {(d_boot > 0.20).mean()*100:.1f}%")

    summary = {
        "in_sample_sweep": {label: {
            "sharpe": float(sharpe_est(np.array([r["net"] for r in cycles_v1[label]])))
        } for label, _ in skip_combos},
        "hard_split": {
            "discovered_bad_hours": sorted(bad_hours),
            "gate_only_sharpe": float(sharpe_est(g_arr)),
            "skip_sharpe": float(sharpe_est(s_arr)),
            "delta_sharpe": float(sharpe_est(s_arr) - sharpe_est(g_arr)),
            "delta_ci": [float(np.quantile(d_boot, 0.025)), float(np.quantile(d_boot, 0.975))],
            "p_positive": float((d_boot > 0).mean() * 100),
        },
    }
    with open(OUT_DIR / "alpha_v9_hour_skip_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
