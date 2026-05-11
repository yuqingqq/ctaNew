"""Two-axis sweep: training window length × retrain cadence.

Sweep A (window length, monthly cadence):
  WIN = [90, 120, 180, 270, 365, FULL] days, all retrained monthly.

Sweep B (cadence at 180d window):
  CADENCE = [7, 14, 28, 56] days, all training on trailing 180d.

Both sweeps run on the same 5-month walk-forward (Dec 25 -> Apr 26)
with the production stack: Hybrid LGBM + Ridge_pos w=0.10 + conv_gate p=0.30.

Aggregate Sharpe pooled across all cycles. Compare apples-to-apples.
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
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
POS_3 = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
RIDGE_W = 0.10
EMBARGO = pd.Timedelta(days=2)
CAL_DAYS = pd.Timedelta(days=20)

OUT_DIR = REPO / "outputs/h48_window_cadence_sweep"
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


def train_and_eval_block(panel, ts, te, window, v6_clean, avail_pos):
    """Train with rolling `window` ending at ts-embargo; predict cycles in [ts, te).
    Returns df of cycle-level pnl (for pooling) and per-block metrics."""
    train_end = ts - EMBARGO
    cal_start = train_end - CAL_DAYS
    cal = panel[(panel["open_time"] >= cal_start) & (panel["open_time"] < train_end)]
    test = panel[(panel["open_time"] >= ts) & (panel["open_time"] < te)]
    if window is None:
        train = panel[panel["open_time"] < cal_start]
    else:
        train = panel[(panel["open_time"] >= cal_start - window) & (panel["open_time"] < cal_start)]

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
    lgbm_pred = np.mean(
        [m.predict(Xtest_v6, num_iteration=m.best_iteration) for m in models], axis=0
    )

    X_pos_full = np.vstack([
        tr[avail_pos].to_numpy(dtype=np.float64),
        ca[avail_pos].to_numpy(dtype=np.float64),
    ])
    y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
    Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
    ridge_pred = fit_predict_ridge(X_pos_full, y_full, Xtest_pos)
    hybrid_pred = (1 - RIDGE_W) * z(lgbm_pred) + RIDGE_W * z(ridge_pred)

    df = evaluate_portfolio(
        test, hybrid_pred, use_gate=True, gate_pctile=GATE_PCTILE,
        use_magweight=False, top_k=TOP_K,
    )
    return {
        "n_train_rows": len(tr),
        "df": df,
    }


def sweep_window(panel, v6_clean, avail_pos, months):
    """Sweep A: window length at monthly cadence."""
    print("\n" + "=" * 110, flush=True)
    print("SWEEP A: WINDOW LENGTH @ monthly cadence", flush=True)
    print("=" * 110, flush=True)

    windows = {
        "WIN_90D":  pd.Timedelta(days=90),
        "WIN_120D": pd.Timedelta(days=120),
        "WIN_180D": pd.Timedelta(days=180),
        "WIN_270D": pd.Timedelta(days=270),
        "WIN_365D": pd.Timedelta(days=365),
        "WIN_FULL": None,
    }

    results = {}
    cycles_by = {w: [] for w in windows}
    for wname, win in windows.items():
        per_month = []
        print(f"\n--- {wname} ({win}) ---", flush=True)
        for ts_dt, te_dt in months:
            ts = pd.Timestamp(ts_dt, tz="UTC")
            te = pd.Timestamp(te_dt, tz="UTC")
            t0 = time.time()
            r = train_and_eval_block(panel, ts, te, win, v6_clean, avail_pos)
            if r is None:
                continue
            sh = sharpe_est(r["df"]["net_bps"].values)
            for _, row in r["df"].iterrows():
                cycles_by[wname].append({
                    "month": ts.date().isoformat(), "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })
            per_month.append({
                "test_month": ts.date().isoformat(),
                "n_train_rows": r["n_train_rows"],
                "sharpe": float(sh),
            })
            print(f"  {ts.date()}: {time.time()-t0:.0f}s  Sh={sh:+.2f}  "
                  f"n_train={r['n_train_rows']//1000}k", flush=True)
        results[wname] = per_month

    print(f"\n  {'window':<10} {'cycles':>7} {'Sharpe':>9} {'95% CI':>22}", flush=True)
    summary = {}
    for wname in windows:
        cy = pd.DataFrame(cycles_by[wname])
        if cy.empty: continue
        sh, lo, hi = block_bootstrap_ci(cy["net"].values, statistic=sharpe_est,
                                         block_size=7, n_boot=2000)
        print(f"  {wname:<10} {len(cy):>7d} {sh:>+8.2f}    [{lo:>+5.2f}, {hi:>+5.2f}]",
              flush=True)
        summary[wname] = {"sharpe": float(sh), "ci": [float(lo), float(hi)],
                          "n_cycles": int(len(cy))}

    return results, summary, cycles_by


def sweep_cadence(panel, v6_clean, avail_pos, test_start, test_end):
    """Sweep B: retrain cadence at 180d rolling window.

    For each cadence K days: retrain at t=test_start, predict next K days,
    advance to t+K, retrain, etc., until test_end.
    """
    print("\n" + "=" * 110, flush=True)
    print("SWEEP B: RETRAIN CADENCE @ 180d window", flush=True)
    print("=" * 110, flush=True)

    cadences = {
        "CAD_7D":  pd.Timedelta(days=7),
        "CAD_14D": pd.Timedelta(days=14),
        "CAD_28D": pd.Timedelta(days=28),
        "CAD_56D": pd.Timedelta(days=56),
    }
    window = pd.Timedelta(days=180)
    results = {}
    cycles_by = {c: [] for c in cadences}

    for cname, cad in cadences.items():
        print(f"\n--- {cname} ({cad}) ---", flush=True)
        per_block = []
        cur = pd.Timestamp(test_start, tz="UTC")
        t_end = pd.Timestamp(test_end, tz="UTC")
        block_idx = 0
        while cur < t_end:
            block_end = min(cur + cad, t_end)
            t0 = time.time()
            r = train_and_eval_block(panel, cur, block_end, window, v6_clean, avail_pos)
            if r is None:
                print(f"  block {block_idx} {cur.date()}->{block_end.date()}: SKIPPED",
                      flush=True)
                cur = block_end
                block_idx += 1
                continue
            sh = sharpe_est(r["df"]["net_bps"].values) if len(r["df"]) > 5 else float("nan")
            for _, row in r["df"].iterrows():
                cycles_by[cname].append({
                    "block_start": cur.date().isoformat(), "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })
            per_block.append({
                "block_start": cur.date().isoformat(),
                "block_end": block_end.date().isoformat(),
                "n_train_rows": r["n_train_rows"],
                "n_cycles": len(r["df"]),
                "sharpe": float(sh) if not np.isnan(sh) else None,
            })
            print(f"  block {block_idx} {cur.date()}->{block_end.date()}: "
                  f"{time.time()-t0:.0f}s  Sh={sh:+.2f}  cycles={len(r['df'])}",
                  flush=True)
            cur = block_end
            block_idx += 1
        results[cname] = per_block

    print(f"\n  {'cadence':<10} {'retrain_n':>10} {'cycles':>7} {'Sharpe':>9} {'95% CI':>22}",
          flush=True)
    summary = {}
    for cname in cadences:
        cy = pd.DataFrame(cycles_by[cname])
        if cy.empty: continue
        sh, lo, hi = block_bootstrap_ci(cy["net"].values, statistic=sharpe_est,
                                         block_size=7, n_boot=2000)
        n_retrains = len(results[cname])
        print(f"  {cname:<10} {n_retrains:>10d} {len(cy):>7d} {sh:>+8.2f}    "
              f"[{lo:>+5.2f}, {hi:>+5.2f}]", flush=True)
        summary[cname] = {"sharpe": float(sh), "ci": [float(lo), float(hi)],
                          "n_cycles": int(len(cy)), "n_retrains": n_retrains}

    return results, summary, cycles_by


def main():
    print(f"Building panel...", flush=True)
    panel = build_panel()
    panel["open_time"] = pd.to_datetime(panel["open_time"])
    if panel["open_time"].dt.tz is None:
        panel["open_time"] = panel["open_time"].dt.tz_localize("UTC")
    print(f"Panel: {panel['open_time'].min().date()} -> {panel['open_time'].max().date()}",
          flush=True)

    months = [
        (datetime(2025, 12, 1), datetime(2026, 1, 1)),
        (datetime(2026, 1, 1), datetime(2026, 2, 1)),
        (datetime(2026, 2, 1), datetime(2026, 3, 1)),
        (datetime(2026, 3, 1), datetime(2026, 4, 1)),
        (datetime(2026, 4, 1), datetime(2026, 5, 1)),
    ]
    test_start = datetime(2025, 12, 1)
    test_end = datetime(2026, 5, 1)

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_pos = [c for c in POS_3 if c in panel.columns]

    win_results, win_summary, win_cycles = sweep_window(panel, v6_clean, avail_pos, months)
    cad_results, cad_summary, cad_cycles = sweep_cadence(
        panel, v6_clean, avail_pos, test_start, test_end
    )

    print("\n" + "=" * 110, flush=True)
    print("FINAL: window optimum (monthly cadence)", flush=True)
    print("=" * 110, flush=True)
    if win_summary:
        ranked = sorted(win_summary.items(), key=lambda kv: -kv[1]["sharpe"])
        for k, v in ranked:
            print(f"  {k:<10} Sh={v['sharpe']:>+5.2f}  CI=[{v['ci'][0]:>+5.2f},"
                  f"{v['ci'][1]:>+5.2f}]", flush=True)
        best_win = ranked[0][0]
        print(f"\n  BEST WINDOW: {best_win} (Sh={ranked[0][1]['sharpe']:+.2f})",
              flush=True)

    print("\n" + "=" * 110, flush=True)
    print("FINAL: cadence optimum (180d window)", flush=True)
    print("=" * 110, flush=True)
    if cad_summary:
        ranked = sorted(cad_summary.items(), key=lambda kv: -kv[1]["sharpe"])
        for k, v in ranked:
            print(f"  {k:<10} Sh={v['sharpe']:>+5.2f}  CI=[{v['ci'][0]:>+5.2f},"
                  f"{v['ci'][1]:>+5.2f}]  n_retrain={v['n_retrains']}", flush=True)
        best_cad = ranked[0][0]
        print(f"\n  BEST CADENCE: {best_cad} (Sh={ranked[0][1]['sharpe']:+.2f})",
              flush=True)

    out = {
        "window_sweep": {"results": win_results, "summary": win_summary},
        "cadence_sweep": {"results": cad_results, "summary": cad_summary},
    }
    with open(OUT_DIR / "alpha_v9_window_cadence_sweep_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
