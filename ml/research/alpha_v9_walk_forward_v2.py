"""Walk-forward validation v2 — cold-start gate, per-cycle aggregate.

Fixes from v1:
  - Cold-start gate (empty initial deque, matches multi-OOS and realistic
    production startup) instead of seeding from cal predictions
  - Capture per-cycle net for AGGREGATE Sharpe across all walks
  - Per-month Sharpe is reported as diagnostic but aggregate is the
    deployment-relevant number
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
OUT_DIR = REPO / "outputs/h48_walk_forward_v2"
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
    panel["open_time"] = pd.to_datetime(panel["open_time"])
    if panel["open_time"].dt.tz is None:
        panel["open_time"] = panel["open_time"].dt.tz_localize("UTC")
    print(f"Panel: {panel['open_time'].min().date()} → {panel['open_time'].max().date()}", flush=True)

    months = [
        (datetime(2025, 8, 1), datetime(2025, 9, 1)),
        (datetime(2025, 9, 1), datetime(2025, 10, 1)),
        (datetime(2025, 10, 1), datetime(2025, 11, 1)),
        (datetime(2025, 11, 1), datetime(2025, 12, 1)),
        (datetime(2025, 12, 1), datetime(2026, 1, 1)),
        (datetime(2026, 1, 1), datetime(2026, 2, 1)),
        (datetime(2026, 2, 1), datetime(2026, 3, 1)),
        (datetime(2026, 3, 1), datetime(2026, 4, 1)),
        (datetime(2026, 4, 1), datetime(2026, 5, 1)),
    ]
    print(f"Walk-forward months: {len(months)}", flush=True)

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_v6 = [c for c in v6_clean if c in panel.columns]
    avail_pos = [c for c in POS_3 if c in panel.columns]
    EMBARGO = pd.Timedelta(days=2)
    CAL_DAYS = pd.Timedelta(days=20)

    # Capture per-cycle net for each variant
    all_cycles = {"V1_lgbm_only": [], "V2_lgbm_gate": [],
                   "V3_hybrid_no_gate": [], "V4_hybrid_gate": []}
    monthly = []

    for (test_start, test_end) in months:
        ts = pd.Timestamp(test_start, tz="UTC")
        te = pd.Timestamp(test_end, tz="UTC")
        train_end = ts - EMBARGO
        cal_start = train_end - CAL_DAYS
        train = panel[panel["open_time"] < cal_start]
        cal = panel[(panel["open_time"] >= cal_start) & (panel["open_time"] < train_end)]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] < te)]
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200 or len(test) < 100:
            print(f"  {ts.date()}: skipped", flush=True)
            continue

        t0 = time.time()
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                              for m in models], axis=0)
        X_full_pos = np.vstack([tr[avail_pos].to_numpy(dtype=np.float64),
                                  ca[avail_pos].to_numpy(dtype=np.float64)])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
        ridge_pred = fit_predict_ridge(X_full_pos, y_full, Xtest_pos)

        lgbm_z = z(lgbm_pred)
        hybrid_z = 0.9 * z(lgbm_pred) + 0.1 * z(ridge_pred)

        # All four variants — cold-start gate (empty deque, matches multi-OOS)
        df1 = evaluate_portfolio(test, lgbm_z, use_gate=False, gate_pctile=GATE_PCTILE,
                                  use_magweight=False, top_k=TOP_K)
        df2 = evaluate_portfolio(test, lgbm_z, use_gate=True, gate_pctile=GATE_PCTILE,
                                  use_magweight=False, top_k=TOP_K)
        df3 = evaluate_portfolio(test, hybrid_z, use_gate=False, gate_pctile=GATE_PCTILE,
                                  use_magweight=False, top_k=TOP_K)
        df4 = evaluate_portfolio(test, hybrid_z, use_gate=True, gate_pctile=GATE_PCTILE,
                                  use_magweight=False, top_k=TOP_K)

        for tag, df in [("V1_lgbm_only", df1), ("V2_lgbm_gate", df2),
                          ("V3_hybrid_no_gate", df3), ("V4_hybrid_gate", df4)]:
            for _, r in df.iterrows():
                all_cycles[tag].append({
                    "month": ts.date().isoformat(), "time": r["time"],
                    "net": r["net_bps"], "skipped": r["skipped"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                })

        sh1 = sharpe_est(df1["net_bps"].values)
        sh2 = sharpe_est(df2["net_bps"].values)
        sh3 = sharpe_est(df3["net_bps"].values)
        sh4 = sharpe_est(df4["net_bps"].values)
        skip2 = df2["skipped"].mean() * 100
        skip4 = df4["skipped"].mean() * 100
        monthly.append({
            "test_month": ts.date().isoformat(), "n_cycles": len(df1),
            "sh1": sh1, "sh2": sh2, "sh3": sh3, "sh4": sh4,
            "net1": df1["net_bps"].mean(), "net4": df4["net_bps"].mean(),
            "skip2_pct": skip2, "skip4_pct": skip4,
        })
        print(f"  {ts.date()}: {time.time() - t0:.0f}s  "
              f"V1={sh1:+.2f} V2={sh2:+.2f} V3={sh3:+.2f} V4={sh4:+.2f}  "
              f"skip%={skip2:.0f}/{skip4:.0f}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print("WALK-FORWARD V2 — COLD-START GATE (matches multi-OOS / production cold-start)", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'month':>10} {'cyc':>5} {'V1_lgbm':>9} {'V2_+gate':>9} {'V3_hybrid':>10} "
          f"{'V4_+both':>9} {'skip%':>6} {'Δ(V4-V1)':>10}", flush=True)
    for m in monthly:
        d = m["sh4"] - m["sh1"]
        print(f"  {m['test_month']:>10} {m['n_cycles']:>5d} {m['sh1']:>+8.2f} "
              f"{m['sh2']:>+8.2f} {m['sh3']:>+9.2f} {m['sh4']:>+8.2f} "
              f"{m['skip4_pct']:>5.1f}% {d:>+9.2f}", flush=True)

    # Aggregate Sharpe across all walks (the deployment-relevant number)
    print(f"\n  {'AGGREGATE (across all walks):'}", flush=True)
    summary = {"monthly": monthly, "aggregate": {}}
    for tag in ["V1_lgbm_only", "V2_lgbm_gate", "V3_hybrid_no_gate", "V4_hybrid_gate"]:
        df = pd.DataFrame(all_cycles[tag])
        if df.empty: continue
        nets = df["net"].values
        sh = sharpe_est(nets)
        sh_b, lo, hi = block_bootstrap_ci(nets, statistic=sharpe_est, block_size=7, n_boot=2000)
        net_mean = nets.mean()
        traded = df[df["skipped"] == 0]
        gross = traded["gross"].mean() if len(traded) > 0 else 0
        cost = traded["cost"].mean() if len(traded) > 0 else 0
        pct_trade = 100 * len(traded) / len(df)
        print(f"    {tag:<22} cyc={len(df):>5} %trd={pct_trade:>5.1f}% "
              f"gross={gross:+5.2f} cost={cost:>4.2f} net={net_mean:+5.2f} "
              f"Sharpe={sh:+.2f} CI=[{lo:+.2f},{hi:+.2f}]", flush=True)
        summary["aggregate"][tag] = {
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "gross": float(gross), "cost": float(cost), "net": float(net_mean),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
        }

    # Compare to the V4_hybrid_gate (production candidate) Δ vs V1_lgbm_only
    if "V4_hybrid_gate" in summary["aggregate"] and "V1_lgbm_only" in summary["aggregate"]:
        agg_d = (summary["aggregate"]["V4_hybrid_gate"]["sharpe"]
                  - summary["aggregate"]["V1_lgbm_only"]["sharpe"])
        # paired delta on cycles
        d1 = pd.DataFrame(all_cycles["V1_lgbm_only"])[["month", "time", "net"]].rename(columns={"net": "n1"})
        d4 = pd.DataFrame(all_cycles["V4_hybrid_gate"])[["month", "time", "net"]].rename(columns={"net": "n4"})
        m = d1.merge(d4, on=["month", "time"], how="inner")
        delta = (m["n4"] - m["n1"]).values
        sh_d = sharpe_est(delta)
        sh_d_b, lo_d, hi_d = block_bootstrap_ci(delta, statistic=sharpe_est, block_size=7, n_boot=2000)
        print(f"\n  PAIRED Δ (V4 hybrid+gate vs V1 LGBM-only):", flush=True)
        print(f"    aggregate Δ Sharpe (mean cycle): {sh_d:+.2f}  CI [{lo_d:+.2f}, {hi_d:+.2f}]", flush=True)
        print(f"    P(Δ > 0): {(np.array([sh_d_b])>0).mean()*100:.1f}% (point), CI tells you the band", flush=True)
        summary["aggregate_delta"] = {
            "delta_sharpe_aggregate": float(agg_d),
            "delta_sharpe_paired": float(sh_d),
            "delta_paired_ci": [float(lo_d), float(hi_d)],
        }

    # Recent 3 months
    if len(monthly) >= 3:
        recent = monthly[-3:]
        print(f"\n  RECENT 3 MONTHS:", flush=True)
        for m in recent:
            d = m["sh4"] - m["sh1"]
            print(f"    {m['test_month']}: V1={m['sh1']:+.2f}  V4={m['sh4']:+.2f}  Δ={d:+.2f}", flush=True)
        # Aggregate recent
        recent_v1 = pd.concat([pd.DataFrame(all_cycles["V1_lgbm_only"])
                                  .query(f"month == '{m['test_month']}'") for m in recent])
        recent_v4 = pd.concat([pd.DataFrame(all_cycles["V4_hybrid_gate"])
                                  .query(f"month == '{m['test_month']}'") for m in recent])
        if not recent_v1.empty and not recent_v4.empty:
            sh_r1 = sharpe_est(recent_v1["net"].values)
            sh_r4 = sharpe_est(recent_v4["net"].values)
            print(f"    RECENT-3MO AGGREGATE: V1={sh_r1:+.2f}  V4={sh_r4:+.2f}  Δ={sh_r4 - sh_r1:+.2f}", flush=True)

    with open(OUT_DIR / "alpha_v9_walk_forward_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
