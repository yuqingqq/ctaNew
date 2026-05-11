"""Test: does training on RECENT DATA ONLY help in the current regime?

Hypothesis: long-window features (return_1d, dom_z_7d, etc.) have decayed
in recent months. The model trained on full history (~13 months) heavily
weights these decayed features. A model trained on recent data only
(3 mo or 6 mo) would naturally reweight away from decayed features and
toward currently-working ones (short-window momentum).

Walk-forward test: for each month from Dec 2025 to Apr 2026:
  - Train LGBM on:
      V_FULL:  all available data → cutoff (current production)
      V_6MO:   last 180 days → cutoff
      V_3MO:   last 90 days → cutoff
  - Evaluate frozen on next month with conv_gate p=0.30
  - Compare per-month Sharpe and aggregate

If shorter window beats full-history → deployable change (just shrink
training data). If not → regime is too broken to fit any window cleanly,
and we should wait for vol to recover.
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
OUT_DIR = REPO / "outputs/h48_recent_train"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


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
    print(f"Walk-forward months: {len(months)}", flush=True)

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_v6 = [c for c in v6_clean if c in panel.columns]
    EMBARGO = pd.Timedelta(days=2)
    CAL_DAYS = pd.Timedelta(days=20)

    train_windows = {
        "V_FULL": None,            # use all data before cal_start
        "V_6MO": pd.Timedelta(days=180),
        "V_3MO": pd.Timedelta(days=90),
        "V_2MO": pd.Timedelta(days=60),
    }

    all_cycles = {tag: [] for tag in train_windows}
    monthly = []

    for (test_start, test_end) in months:
        ts = pd.Timestamp(test_start, tz="UTC")
        te = pd.Timestamp(test_end, tz="UTC")
        train_end = ts - EMBARGO
        cal_start = train_end - CAL_DAYS
        cal = panel[(panel["open_time"] >= cal_start) & (panel["open_time"] < train_end)]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] < te)]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(ca) < 200 or len(test) < 100:
            print(f"  {ts.date()}: skipped", flush=True)
            continue

        Xtest = test[avail_v6].to_numpy(dtype=np.float32)
        yt_cal = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)

        m_results = {"test_month": ts.date().isoformat()}
        t_start = time.time()
        for tag, win in train_windows.items():
            if win is None:
                tr_window = panel[panel["open_time"] < cal_start]
            else:
                tr_window = panel[(panel["open_time"] >= cal_start - win) & (panel["open_time"] < cal_start)]
            tr = tr_window[tr_window["autocorr_pctile_7d"] >= THRESHOLD]
            if len(tr) < 1000:
                m_results[tag] = "insufficient"
                continue
            Xt = tr[avail_v6].to_numpy(dtype=np.float32)
            yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yt_cal, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            df = evaluate_portfolio(test, yt_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                     use_magweight=False, top_k=TOP_K)
            sh = sharpe_est(df["net_bps"].values)
            net = df["net_bps"].mean()
            traded = df[df["skipped"] == 0]
            gross = traded["spread_ret_bps"].mean() if len(traded) > 0 else 0
            cost = traded["cost_bps"].mean() if len(traded) > 0 else 0
            n_train_rows = len(tr)
            avg_iter = float(np.mean([m.best_iteration for m in models]))
            m_results[tag] = {
                "sharpe": float(sh), "net": float(net), "gross": float(gross),
                "cost": float(cost), "n_train_rows": int(n_train_rows),
                "avg_best_iter": avg_iter,
            }
            for _, r in df.iterrows():
                all_cycles[tag].append({
                    "month": ts.date().isoformat(), "time": r["time"],
                    "net": r["net_bps"], "skipped": r["skipped"],
                })

        # Print summary
        elapsed = time.time() - t_start
        line = f"  {ts.date()}: {elapsed:.0f}s  "
        for tag in train_windows:
            r = m_results.get(tag)
            if isinstance(r, dict):
                line += f"{tag}={r['sharpe']:+.2f}(n={r['n_train_rows']//1000}k,it={r['avg_best_iter']:.0f}) "
        print(line, flush=True)
        monthly.append(m_results)

    # Aggregate Sharpe per training window across all walks
    print("\n" + "=" * 110, flush=True)
    print("RECENT-DATA TRAINING TEST (walk-forward, conv_gate p=0.30, h=48 K=7)", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'window':<8} {'cycles':>7} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'Sharpe':>7} {'95% CI':>15}", flush=True)

    aggregate = {}
    for tag in train_windows:
        recs = pd.DataFrame(all_cycles[tag])
        if recs.empty:
            print(f"  {tag:<8}  NO DATA")
            continue
        traded = recs[recs["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(recs["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        net = recs["net"].mean()
        # gross/cost per traded cycle (need from per-month details)
        gross_avg = np.mean([m[tag]["gross"] for m in monthly if isinstance(m.get(tag), dict)])
        cost_avg = np.mean([m[tag]["cost"] for m in monthly if isinstance(m.get(tag), dict)])
        print(f"  {tag:<8} {len(recs):>7d} {gross_avg:>+6.2f}  {cost_avg:>5.2f}  "
              f"{net:>+6.2f}  {sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]", flush=True)
        aggregate[tag] = {"n_cycles": int(len(recs)), "net": float(net),
                            "sharpe": float(sh), "ci": [float(lo), float(hi)]}

    # Per-month breakdown
    print(f"\n  Per-month Sharpe:", flush=True)
    print(f"  {'month':>10} " + " ".join(f"{tag:>8}" for tag in train_windows), flush=True)
    for m in monthly:
        line = f"  {m['test_month']:>10} "
        for tag in train_windows:
            r = m.get(tag)
            if isinstance(r, dict):
                line += f" {r['sharpe']:>+7.2f}"
            else:
                line += f" {'N/A':>7}"
        print(line, flush=True)

    # Versus full-window: which beats?
    if "V_FULL" in aggregate:
        print(f"\n  Δ vs V_FULL (Sharpe lift from shorter training):", flush=True)
        for tag in ["V_6MO", "V_3MO", "V_2MO"]:
            if tag in aggregate:
                d = aggregate[tag]["sharpe"] - aggregate["V_FULL"]["sharpe"]
                print(f"    {tag}:  Δ Sharpe {d:+.2f}", flush=True)

    with open(OUT_DIR / "alpha_v9_recent_train_summary.json", "w") as f:
        json.dump({"monthly": monthly, "aggregate": aggregate}, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
