"""Debug: why does walk-forward give negative Sharpe vs multi-OOS positive?

Hypothesis: the methodology is the same; the discrepancy is purely from
test window alignment (calendar months vs rolling 30-day cursor).

Test:
  1. Run walk-forward CODE using multi-OOS FOLD DATES exactly.
  2. Compare aggregate Sharpe to multi-OOS recent_backtest result (+1.13 for LGBM).
  3. If they match → walk-forward methodology is fine; calendar boundaries
     are the issue.
  4. If they don't → there's a methodological bug in walk-forward code.
"""
from __future__ import annotations
import sys
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
from ml.research.alpha_v4_xs_1d import ENSEMBLE_SEEDS, _train, _multi_oos_splits, _slice
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

    # ===== METHOD A: Multi-OOS framework (using _multi_oos_splits + _slice) =====
    folds = _multi_oos_splits(panel)
    print(f"\nMulti-OOS folds: {len(folds)}", flush=True)
    print(f"Fold dates: ", flush=True)
    for f in folds:
        print(f"  fold {f['fid']}: train_end={f['train_end'].date()}, "
              f"cal=[{f['cal_start'].date()},{f['cal_end'].date()}), "
              f"test=[{f['test_start'].date()},{f['test_end'].date()})", flush=True)

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_v6 = [c for c in v6_clean if c in panel.columns]
    avail_pos = [c for c in POS_3 if c in panel.columns]

    # Aggregate cycle records for METHOD A
    method_a_cycles = []
    print(f"\n--- METHOD A: Multi-OOS framework ---", flush=True)
    print(f"  {'fold':>4} {'date':>12} {'lgbm_sh':>9} {'hybrid_sh':>10} {'%trade':>7} {'n_cyc':>6}", flush=True)
    for fold in folds:
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
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
        hybrid_pred = 0.9 * z(lgbm_pred) + 0.1 * z(ridge_pred)

        df_lgbm = evaluate_portfolio(test, z(lgbm_pred), use_gate=True, gate_pctile=GATE_PCTILE,
                                       use_magweight=False, top_k=TOP_K)
        df_hyb = evaluate_portfolio(test, hybrid_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                       use_magweight=False, top_k=TOP_K)

        sh_lgbm = sharpe_est(df_lgbm["net_bps"].values)
        sh_hyb = sharpe_est(df_hyb["net_bps"].values)
        skip = df_lgbm["skipped"].mean() * 100
        print(f"  {fold['fid']:>4d} {fold['test_start'].date()!s:>12} "
              f"{sh_lgbm:>+8.2f} {sh_hyb:>+9.2f} {100-skip:>6.1f}% {len(df_lgbm):>5d}", flush=True)

        for _, r in df_lgbm.iterrows():
            method_a_cycles.append({
                "method": "A_multi_oos", "fold_or_month": fold["fid"],
                "test_start": fold["test_start"].date().isoformat(),
                "time": r["time"], "net_lgbm": r["net_bps"],
            })
        for i, r in df_hyb.iterrows():
            # Add hybrid net to corresponding cycle record
            if i < len(method_a_cycles) - len(df_lgbm) + len(df_hyb):
                pass  # skip adjustment

    # ===== METHOD A aggregate =====
    cycles_a_df = pd.DataFrame(method_a_cycles)
    if not cycles_a_df.empty:
        sh_a = sharpe_est(cycles_a_df["net_lgbm"].values)
        print(f"\n  METHOD A AGGREGATE: LGBM Sharpe = {sh_a:+.2f} (cycles={len(cycles_a_df)})", flush=True)

    # ===== METHOD B: Walk-forward CODE with multi-OOS fold dates =====
    print(f"\n--- METHOD B: Walk-forward code with multi-OOS fold dates ---", flush=True)
    print(f"  Same dates as A, but using my walk-forward slicing logic", flush=True)
    print(f"  {'fold':>4} {'date':>12} {'lgbm_sh':>9} {'hybrid_sh':>10} {'%trade':>7} {'n_cyc':>6}", flush=True)
    method_b_cycles = []
    EMBARGO = pd.Timedelta(days=2)
    CAL_DAYS = pd.Timedelta(days=20)

    for fold in folds:
        ts = fold["test_start"]
        te = fold["test_end"]
        # Walk-forward logic
        train_end = ts - EMBARGO
        cal_start = train_end - CAL_DAYS
        train = panel[panel["open_time"] < cal_start]
        cal = panel[(panel["open_time"] >= cal_start) & (panel["open_time"] < train_end)]
        test_wf = panel[(panel["open_time"] >= ts) & (panel["open_time"] < te)]
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200 or len(test_wf) < 100:
            continue
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest_v6 = test_wf[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                              for m in models], axis=0)
        X_full_pos = np.vstack([tr[avail_pos].to_numpy(dtype=np.float64),
                                  ca[avail_pos].to_numpy(dtype=np.float64)])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        Xtest_pos = test_wf[avail_pos].to_numpy(dtype=np.float64)
        ridge_pred = fit_predict_ridge(X_full_pos, y_full, Xtest_pos)
        hybrid_pred = 0.9 * z(lgbm_pred) + 0.1 * z(ridge_pred)

        df_lgbm = evaluate_portfolio(test_wf, z(lgbm_pred), use_gate=True, gate_pctile=GATE_PCTILE,
                                       use_magweight=False, top_k=TOP_K)
        df_hyb = evaluate_portfolio(test_wf, hybrid_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                       use_magweight=False, top_k=TOP_K)

        sh_lgbm = sharpe_est(df_lgbm["net_bps"].values)
        sh_hyb = sharpe_est(df_hyb["net_bps"].values)
        skip = df_lgbm["skipped"].mean() * 100
        print(f"  {fold['fid']:>4d} {ts.date()!s:>12} "
              f"{sh_lgbm:>+8.2f} {sh_hyb:>+9.2f} {100-skip:>6.1f}% {len(df_lgbm):>5d}", flush=True)
        for _, r in df_lgbm.iterrows():
            method_b_cycles.append({"method": "B_walkfwd_oos_dates", "fold_or_month": fold["fid"],
                                      "time": r["time"], "net_lgbm": r["net_bps"]})

    cycles_b_df = pd.DataFrame(method_b_cycles)
    if not cycles_b_df.empty:
        sh_b = sharpe_est(cycles_b_df["net_lgbm"].values)
        print(f"\n  METHOD B AGGREGATE: LGBM Sharpe = {sh_b:+.2f} (cycles={len(cycles_b_df)})", flush=True)

    # Compare
    print(f"\n--- COMPARISON ---", flush=True)
    if not cycles_a_df.empty and not cycles_b_df.empty:
        print(f"  Method A (multi-OOS framework):       Sharpe {sh_a:+.2f}, n={len(cycles_a_df)}", flush=True)
        print(f"  Method B (walk-fwd code, OOS dates):  Sharpe {sh_b:+.2f}, n={len(cycles_b_df)}", flush=True)
        # Per-fold comparison
        per_fold_a = cycles_a_df.groupby("fold_or_month")["net_lgbm"].apply(
            lambda x: sharpe_est(x.values))
        per_fold_b = cycles_b_df.groupby("fold_or_month")["net_lgbm"].apply(
            lambda x: sharpe_est(x.values))
        print(f"\n  Per-fold Sharpe comparison (LGBM-only):", flush=True)
        print(f"  {'fold':>4} {'A_multi-OOS':>13} {'B_walk-fwd':>12} {'diff':>8}", flush=True)
        for fid in sorted(per_fold_a.index):
            if fid in per_fold_b.index:
                da, db = per_fold_a[fid], per_fold_b[fid]
                print(f"  {fid:>4d} {da:>+12.2f}  {db:>+11.2f}  {db-da:>+7.2f}", flush=True)


if __name__ == "__main__":
    main()
