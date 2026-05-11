"""Rolling training window + full integrated stack walk-forward.

Question: does a ROLLING 6-month training window combined with the full
production stack (Hybrid LGBM + Ridge_pos + conv_gate p=0.30) outperform
the FULL-history training that's currently default?

Each month: retrain BOTH LGBM and Ridge on a rolling window. Apply
frozen to next month with conv_gate active. Repeat.

This is the production deployment scenario with monthly retraining cadence.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import ENSEMBLE_SEEDS, _train
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
RIDGE_W = 0.10
OUT_DIR = REPO / "outputs/h48_rolling_integrated"
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


def train_and_eval(panel, ts, te, train_window, v6_clean, avail_pos):
    """Train LGBM + Ridge on the given train window; evaluate on test month
    with hybrid blend + conv_gate active. Returns dict with metrics."""
    EMBARGO = pd.Timedelta(days=2)
    CAL_DAYS = pd.Timedelta(days=20)
    train_end = ts - EMBARGO
    cal_start = train_end - CAL_DAYS
    cal = panel[(panel["open_time"] >= cal_start) & (panel["open_time"] < train_end)]
    test = panel[(panel["open_time"] >= ts) & (panel["open_time"] < te)]
    if train_window is None:
        train = panel[panel["open_time"] < cal_start]
    else:
        train = panel[(panel["open_time"] >= cal_start - train_window) & (panel["open_time"] < cal_start)]
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200 or len(test) < 100:
        return None
    avail_v6 = [c for c in v6_clean if c in panel.columns]
    Xt = tr[avail_v6].to_numpy(dtype=np.float32)
    yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
    Xc = ca[avail_v6].to_numpy(dtype=np.float32)
    yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
    Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
    models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
    lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                          for m in models], axis=0)

    # Ridge head — same training window for consistency
    X_pos_full = np.vstack([tr[avail_pos].to_numpy(dtype=np.float64),
                              ca[avail_pos].to_numpy(dtype=np.float64)])
    y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
    Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
    ridge_pred = fit_predict_ridge(X_pos_full, y_full, Xtest_pos)
    hybrid_pred = (1 - RIDGE_W) * z(lgbm_pred) + RIDGE_W * z(ridge_pred)

    # Evaluate four flavors:
    #  V1: LGBM-only no gate
    #  V2: LGBM + conv_gate
    #  V3: Hybrid no gate
    #  V4: Hybrid + conv_gate (production)
    df1 = evaluate_portfolio(test, z(lgbm_pred), use_gate=False, gate_pctile=GATE_PCTILE,
                              use_magweight=False, top_k=TOP_K)
    df2 = evaluate_portfolio(test, z(lgbm_pred), use_gate=True, gate_pctile=GATE_PCTILE,
                              use_magweight=False, top_k=TOP_K)
    df3 = evaluate_portfolio(test, hybrid_pred, use_gate=False, gate_pctile=GATE_PCTILE,
                              use_magweight=False, top_k=TOP_K)
    df4 = evaluate_portfolio(test, hybrid_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                              use_magweight=False, top_k=TOP_K)
    return {
        "n_train_rows": len(tr), "n_cycles": len(df1),
        "lgbm_iters": [int(m.best_iteration) for m in models],
        "V1_lgbm_only_sh": float(sharpe_est(df1["net_bps"].values)),
        "V2_lgbm_gate_sh": float(sharpe_est(df2["net_bps"].values)),
        "V3_hybrid_no_gate_sh": float(sharpe_est(df3["net_bps"].values)),
        "V4_hybrid_gate_sh": float(sharpe_est(df4["net_bps"].values)),
        "df_v1": df1, "df_v4": df4,
    }


def main():
    panel = build_panel()
    panel["open_time"] = pd.to_datetime(panel["open_time"])
    if panel["open_time"].dt.tz is None:
        panel["open_time"] = panel["open_time"].dt.tz_localize("UTC")
    print(f"Panel: {panel['open_time'].min().date()} → {panel['open_time'].max().date()}", flush=True)

    months = [
        (datetime(2025, 12, 1), datetime(2026, 1, 1)),
        (datetime(2026, 1, 1), datetime(2026, 2, 1)),
        (datetime(2026, 2, 1), datetime(2026, 3, 1)),
        (datetime(2026, 3, 1), datetime(2026, 4, 1)),
        (datetime(2026, 4, 1), datetime(2026, 5, 1)),
    ]

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_pos = [c for c in POS_3 if c in panel.columns]
    print(f"Walk-forward months: {len(months)}", flush=True)
    print(f"Positioning features: {avail_pos}", flush=True)

    train_specs = {
        "FULL": None,
        "ROLL_180D": pd.Timedelta(days=180),
        "ROLL_120D": pd.Timedelta(days=120),
        "ROLL_90D":  pd.Timedelta(days=90),
    }

    results_by_spec = {}
    cycles_by_spec = {spec: {"v1": [], "v4": []} for spec in train_specs}

    for spec_name, spec_window in train_specs.items():
        spec_results = []
        print(f"\n--- training spec: {spec_name} ({spec_window}) ---", flush=True)
        for ts_dt, te_dt in months:
            ts = pd.Timestamp(ts_dt, tz="UTC")
            te = pd.Timestamp(te_dt, tz="UTC")
            t0 = time.time()
            r = train_and_eval(panel, ts, te, spec_window, v6_clean, avail_pos)
            if r is None:
                continue
            for _, row in r["df_v1"].iterrows():
                cycles_by_spec[spec_name]["v1"].append({
                    "month": ts.date().isoformat(), "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })
            for _, row in r["df_v4"].iterrows():
                cycles_by_spec[spec_name]["v4"].append({
                    "month": ts.date().isoformat(), "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })
            spec_results.append({
                "test_month": ts.date().isoformat(),
                "n_train_rows": r["n_train_rows"],
                "V1": r["V1_lgbm_only_sh"], "V2": r["V2_lgbm_gate_sh"],
                "V3": r["V3_hybrid_no_gate_sh"], "V4": r["V4_hybrid_gate_sh"],
            })
            print(f"  {ts.date()}: {time.time() - t0:.0f}s  "
                  f"V1={r['V1_lgbm_only_sh']:+.2f} V2={r['V2_lgbm_gate_sh']:+.2f} "
                  f"V3={r['V3_hybrid_no_gate_sh']:+.2f} V4={r['V4_hybrid_gate_sh']:+.2f}  "
                  f"(n_train={r['n_train_rows']//1000}k)", flush=True)
        results_by_spec[spec_name] = spec_results

    print("\n" + "=" * 110, flush=True)
    print("ROLLING-WINDOW WALK-FORWARD WITH INTEGRATED STACK", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'spec':<10} {'cycles':>7} {'V1_lgbm':>9} {'V4_full_stack':>15} {'CI_v4':>20}", flush=True)
    summary = {}
    for spec_name in train_specs:
        v1_arr = pd.DataFrame(cycles_by_spec[spec_name]["v1"])
        v4_arr = pd.DataFrame(cycles_by_spec[spec_name]["v4"])
        if v1_arr.empty: continue
        v1_sh = sharpe_est(v1_arr["net"].values)
        v4_sh = sharpe_est(v4_arr["net"].values)
        v4_sh_b, lo, hi = block_bootstrap_ci(v4_arr["net"].values, statistic=sharpe_est,
                                              block_size=7, n_boot=2000)
        print(f"  {spec_name:<10} {len(v1_arr):>7d} {v1_sh:>+8.2f} {v4_sh:>+14.2f}   [{lo:>+5.2f}, {hi:>+5.2f}]", flush=True)
        summary[spec_name] = {
            "v1_lgbm_only_sharpe": float(v1_sh),
            "v4_full_stack_sharpe": float(v4_sh),
            "v4_ci": [float(lo), float(hi)],
            "n_cycles": int(len(v4_arr)),
        }

    # Per-month V4 by spec
    print(f"\n  PER-MONTH V4 (full integrated stack) Sharpe by training spec:", flush=True)
    print(f"  {'month':>10} " + " ".join(f"{s:>10}" for s in train_specs.keys()), flush=True)
    for i, (ts_dt, te_dt) in enumerate(months):
        line = f"  {ts_dt.strftime('%Y-%m-%d'):>10}  "
        for spec in train_specs:
            r = results_by_spec.get(spec, [])
            if i < len(r):
                line += f" {r[i]['V4']:>+9.2f}"
            else:
                line += f"     N/A "
        print(line, flush=True)

    # Best per spec — Δ vs FULL
    if "FULL" in summary:
        print(f"\n  Δ vs FULL spec (V4 full-stack Sharpe):", flush=True)
        for s in train_specs:
            if s == "FULL": continue
            d = summary[s]["v4_full_stack_sharpe"] - summary["FULL"]["v4_full_stack_sharpe"]
            print(f"    {s}:  Δ Sharpe {d:+.2f}", flush=True)

    with open(OUT_DIR / "alpha_v9_rolling_integrated_summary.json", "w") as f:
        json.dump({"results_by_spec": results_by_spec, "aggregate": summary}, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
