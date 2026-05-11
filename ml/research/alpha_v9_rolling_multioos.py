"""Multi-OOS test: rolling 180d window vs FULL training (anchored expanding).

Question: does rolling-180d beat FULL training on the 10-fold multi-OOS
sample (the long-run validation), or only in the recent 5-month drawdown?

Both specs use the production stack:
  Hybrid (LGBM v6_clean + Ridge_pos w=0.10) + conv_gate p=0.30

Same fold structure as alpha_v9_hybrid_validate.py (10 folds × 30-day OOS).
Aggregate Sharpe pooled over all OOS cycles.
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
from ml.research.alpha_v9_positioning_pack import build_panel

HORIZON = 48
TOP_K = 7
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
POS_3 = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
RIDGE_W = 0.10
ROLL_DAYS = pd.Timedelta(days=180)

OUT_DIR = REPO / "outputs/h48_rolling_multioos"
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


def slice_with_window(panel, fold, window):
    """Like _slice, but trims train to a trailing `window` ending at cal_start.
    If window is None, returns the standard (anchored expanding) slice.
    """
    train, cal, test = _slice(panel, fold)
    if window is not None:
        train_start_floor = fold["cal_start"] - window
        train = train[train["open_time"] >= train_start_floor]
    return train, cal, test


def evaluate_fold(panel, fold, window, v6_clean, avail_pos):
    train, cal, test = slice_with_window(panel, fold, window)
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

    Xt_pos = tr[avail_pos].to_numpy(dtype=np.float64)
    Xc_pos = ca[avail_pos].to_numpy(dtype=np.float64)
    X_full_pos = np.vstack([Xt_pos, Xc_pos])
    y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
    Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
    ridge_pred = fit_predict_ridge(X_full_pos, y_full, Xtest_pos)
    hybrid_pred = (1 - RIDGE_W) * z(lgbm_pred) + RIDGE_W * z(ridge_pred)

    df = evaluate_portfolio(
        test, hybrid_pred, use_gate=True, gate_pctile=GATE_PCTILE,
        use_magweight=False, top_k=TOP_K,
    )
    return {
        "n_train_rows": len(tr),
        "n_cycles": len(df),
        "df": df,
        "lgbm_iters": [int(m.best_iteration) for m in models],
    }


def main():
    print("Building panel...", flush=True)
    panel = build_panel()
    panel["open_time"] = pd.to_datetime(panel["open_time"])
    if panel["open_time"].dt.tz is None:
        panel["open_time"] = panel["open_time"].dt.tz_localize("UTC")
    print(f"Panel: {panel['open_time'].min().date()} -> "
          f"{panel['open_time'].max().date()}", flush=True)

    folds = _multi_oos_splits(panel, min_train_days=60, cal_days=20,
                                test_days=30, embargo_days=2.0)
    print(f"Folds: {len(folds)}", flush=True)
    for f in folds:
        print(f"  fold {f['fid']}: train_end={f['cal_start'].date()}  "
              f"test={f['test_start'].date()}->{f['test_end'].date()}", flush=True)

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_pos = [c for c in POS_3 if c in panel.columns]

    specs = {
        "FULL":      None,        # anchored expanding (current production)
        "ROLL_180D": ROLL_DAYS,
    }

    results = {}
    cycles_by = {s: [] for s in specs}
    per_fold = {s: [] for s in specs}

    for sname, swin in specs.items():
        print(f"\n--- spec: {sname} ({swin}) ---", flush=True)
        for f in folds:
            t0 = time.time()
            r = evaluate_fold(panel, f, swin, v6_clean, avail_pos)
            if r is None:
                print(f"  fold {f['fid']}: SKIPPED (insufficient data)", flush=True)
                continue
            sh = sharpe_est(r["df"]["net_bps"].values)
            for _, row in r["df"].iterrows():
                cycles_by[sname].append({
                    "fold": int(f["fid"]),
                    "test_start": f["test_start"].date().isoformat(),
                    "time": row["time"], "net": row["net_bps"],
                    "skipped": row["skipped"],
                })
            per_fold[sname].append({
                "fid": int(f["fid"]),
                "test_start": f["test_start"].date().isoformat(),
                "n_train_rows": r["n_train_rows"],
                "sharpe": float(sh),
                "n_cycles": r["n_cycles"],
            })
            print(f"  fold {f['fid']} ({f['test_start'].date()}): "
                  f"{time.time()-t0:.0f}s  Sh={sh:+.2f}  "
                  f"n_train={r['n_train_rows']//1000}k  cycles={r['n_cycles']}",
                  flush=True)

    print("\n" + "=" * 110, flush=True)
    print("ROLLING 180D vs FULL — 10-fold multi-OOS (production hybrid stack)",
          flush=True)
    print("=" * 110, flush=True)

    summary = {}
    print(f"  {'spec':<12} {'cycles':>7} {'Sharpe':>8} {'95% CI':>22}", flush=True)
    for sname in specs:
        cy = pd.DataFrame(cycles_by[sname])
        if cy.empty:
            print(f"  {sname:<12} NO DATA")
            continue
        sh, lo, hi = block_bootstrap_ci(cy["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        print(f"  {sname:<12} {len(cy):>7d} {sh:>+7.2f}    "
              f"[{lo:>+5.2f}, {hi:>+5.2f}]", flush=True)
        summary[sname] = {
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "n_cycles": int(len(cy)),
        }

    if "FULL" in summary and "ROLL_180D" in summary:
        d = summary["ROLL_180D"]["sharpe"] - summary["FULL"]["sharpe"]
        print(f"\n  Δ ROLL_180D - FULL = {d:+.2f}", flush=True)

        full_pnl = pd.DataFrame(cycles_by["FULL"])["net"].values
        roll_pnl = pd.DataFrame(cycles_by["ROLL_180D"])["net"].values
        n = min(len(full_pnl), len(roll_pnl))
        delta = roll_pnl[:n] - full_pnl[:n]
        d_sh, d_lo, d_hi = block_bootstrap_ci(delta, statistic=sharpe_est,
                                                block_size=7, n_boot=2000)
        print(f"  Δ Sharpe paired bootstrap CI: [{d_lo:+.2f}, {d_hi:+.2f}]",
              flush=True)
        n_pos = int((delta > 0).sum())
        print(f"  Cycles where ROLL >= FULL: {n_pos}/{n} ({n_pos/n:.1%})",
              flush=True)

    # Per-fold side-by-side
    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'fid':>3} {'test_start':>11} {'FULL':>9} {'ROLL_180D':>11} "
          f"{'Δ':>7}", flush=True)
    fmap_full = {p["fid"]: p for p in per_fold["FULL"]}
    fmap_roll = {p["fid"]: p for p in per_fold["ROLL_180D"]}
    deltas = []
    for fid in sorted(set(list(fmap_full.keys()) + list(fmap_roll.keys()))):
        f_sh = fmap_full.get(fid, {}).get("sharpe", float("nan"))
        r_sh = fmap_roll.get(fid, {}).get("sharpe", float("nan"))
        ts = (fmap_full.get(fid) or fmap_roll.get(fid))["test_start"]
        d = r_sh - f_sh if not (np.isnan(f_sh) or np.isnan(r_sh)) else float("nan")
        deltas.append(d)
        print(f"  {fid:>3} {ts:>11} {f_sh:>+8.2f} {r_sh:>+10.2f} {d:>+7.2f}",
              flush=True)
    valid_d = [d for d in deltas if not np.isnan(d)]
    if valid_d:
        print(f"\n  AVG per-fold Δ = {np.mean(valid_d):+.2f}", flush=True)
        print(f"  Folds where ROLL beats FULL: "
              f"{sum(1 for d in valid_d if d > 0)}/{len(valid_d)}", flush=True)

    out = {"per_fold": per_fold, "summary": summary}
    with open(OUT_DIR / "alpha_v9_rolling_multioos_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
