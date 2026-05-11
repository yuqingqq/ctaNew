"""Walk-forward validation simulating production deployment cadence.

For each month from August 2025 to April 2026:
  1. Train LGBM ensemble on v6_clean (production model)
  2. Train Ridge on positioning pack (hybrid head)
  3. Train cutoff = 1st of month - 2 day embargo
  4. Cal window = last 20d of training (matches production)
  5. Apply frozen to the month with conv_gate p=0.30
  6. Conv_gate dispersion history seeded from end of training period

Compare four variants per month:
  V1: LGBM-only (current production)
  V2: LGBM + conv_gate (validated execution rule)
  V3: LGBM + ridge_pos hybrid (new ridge head, no gate)
  V4: LGBM + ridge_pos + conv_gate (production candidate)

Report per-month consistency + cumulative + recent-3-months breakdown.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta

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
GATE_LOOKBACK = 252
POS_3 = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
OUT_DIR = REPO / "outputs/h48_walk_forward"
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


def _bn_scales(top_g, bot_g):
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_with_seeded_gate(test_df, yt_pred, *, dispersion_seed: list, top_k: int = TOP_K):
    """Like evaluate_portfolio but with conv_gate dispersion history pre-seeded
    from training period (more realistic for walk-forward)."""
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test_df[cols].copy()
    df["pred"] = yt_pred
    times = sorted(df["open_time"].unique())
    keep = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep)]
    bars = []
    prev_long_w: dict = {}
    prev_short_w: dict = {}
    history = deque(dispersion_seed, maxlen=GATE_LOOKBACK)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        skip = False
        if len(history) >= 30:
            thr = np.quantile(list(history), GATE_PCTILE)
            if dispersion < thr:
                skip = True
        history.append(dispersion)

        if skip:
            bars.append({"time": t, "spread_ret_bps": 0.0, "long_turnover": 0.0,
                          "short_turnover": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                          "skipped": 1})
            continue
        scale_L, scale_S = _bn_scales(top, bot)
        n_l, n_s = len(top), len(bot)
        long_w = {s: scale_L / n_l for s in top["symbol"]}
        short_w = {s: scale_S / n_s for s in bot["symbol"]}
        long_ret = scale_L * top["return_pct"].mean()
        short_ret = scale_S * bot["return_pct"].mean()
        spread_ret = long_ret - short_ret
        if not prev_long_w:
            long_to, short_to = scale_L, scale_S
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        cost_bps = COST_PER_LEG * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - cost_bps
        bars.append({"time": t, "spread_ret_bps": spread_ret * 1e4,
                      "long_turnover": long_to, "short_turnover": short_to,
                      "cost_bps": cost_bps, "net_bps": net_bps, "skipped": 0})
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def main():
    panel = build_panel()
    panel["open_time"] = pd.to_datetime(panel["open_time"])
    if panel["open_time"].dt.tz is None:
        panel["open_time"] = panel["open_time"].dt.tz_localize("UTC")
    print(f"Panel: {panel['open_time'].min().date()} → {panel['open_time'].max().date()}", flush=True)

    # Define monthly walk-forward cutoffs (Aug 2025 → Apr 2026)
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

    monthly = []
    for (test_start, test_end) in months:
        ts = pd.Timestamp(test_start, tz="UTC")
        te = pd.Timestamp(test_end, tz="UTC")
        train_end = ts - EMBARGO
        cal_start = train_end - CAL_DAYS
        train = panel[panel["open_time"] < cal_start]
        cal = panel[(panel["open_time"] >= cal_start) & (panel["open_time"] < train_end)]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] < te)]
        # Filter train and cal by autocorr regime (matches production)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200 or len(test) < 100:
            print(f"  {ts.date()}: skipped (insufficient data: tr={len(tr)} ca={len(ca)} te={len(test)})", flush=True)
            continue

        t0 = time.time()
        # Train LGBM
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest_v6 = test[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest_v6, num_iteration=m.best_iteration)
                              for m in models], axis=0)
        # Train Ridge on positioning
        X_full_pos = np.vstack([tr[avail_pos].to_numpy(dtype=np.float64),
                                  ca[avail_pos].to_numpy(dtype=np.float64)])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        Xtest_pos = test[avail_pos].to_numpy(dtype=np.float64)
        ridge_pred = fit_predict_ridge(X_full_pos, y_full, Xtest_pos)

        # Compute dispersion seed from CAL period (last 252 unique-bar dispersions)
        # Predict on cal window and compute per-bar dispersion
        Xcal_v6 = ca[avail_v6].to_numpy(dtype=np.float32)
        cal_pred = np.mean([m.predict(Xcal_v6, num_iteration=m.best_iteration)
                             for m in models], axis=0)
        cal_df = ca[["open_time", "symbol"]].copy()
        cal_df["pred"] = cal_pred
        # Sample at HORIZON cadence to match test cadence
        cal_times = sorted(cal_df["open_time"].unique())
        cal_keep = set(cal_times[::HORIZON])
        cal_df = cal_df[cal_df["open_time"].isin(cal_keep)]
        dispersion_seed = []
        for tt, gg in cal_df.groupby("open_time"):
            if len(gg) < 2 * TOP_K + 1:
                continue
            sg = gg.sort_values("pred")
            disp = sg.tail(TOP_K)["pred"].mean() - sg.head(TOP_K)["pred"].mean()
            dispersion_seed.append(float(disp))
        # Take last GATE_LOOKBACK
        dispersion_seed = dispersion_seed[-GATE_LOOKBACK:]

        # Evaluate four variants
        results = {}
        # V1: LGBM-only no gate
        df1 = evaluate_with_seeded_gate(test, z(lgbm_pred), dispersion_seed=[])  # empty seed = no gate triggers (need 30+)
        # Actually need to disable gate for V1. Use a separate path:
        from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio
        df1 = evaluate_portfolio(test, z(lgbm_pred), use_gate=False, gate_pctile=GATE_PCTILE,
                                  use_magweight=False, top_k=TOP_K)
        # V2: LGBM + conv_gate (with seeded history)
        df2 = evaluate_with_seeded_gate(test, z(lgbm_pred), dispersion_seed=dispersion_seed)
        # V3: hybrid no gate
        hybrid_pred = 0.9 * z(lgbm_pred) + 0.1 * z(ridge_pred)
        df3 = evaluate_portfolio(test, hybrid_pred, use_gate=False, gate_pctile=GATE_PCTILE,
                                  use_magweight=False, top_k=TOP_K)
        # V4: hybrid + conv_gate (seeded)
        df4 = evaluate_with_seeded_gate(test, hybrid_pred, dispersion_seed=dispersion_seed)

        n_cycles = len(df1)
        sh1 = sharpe_est(df1["net_bps"].values)
        sh2 = sharpe_est(df2["net_bps"].values)
        sh3 = sharpe_est(df3["net_bps"].values)
        sh4 = sharpe_est(df4["net_bps"].values)
        net1 = df1["net_bps"].mean()
        net2 = df2["net_bps"].mean()
        net3 = df3["net_bps"].mean()
        net4 = df4["net_bps"].mean()
        skip2 = df2["skipped"].mean() * 100
        skip4 = df4["skipped"].mean() * 100

        monthly.append({
            "test_month": ts.date().isoformat(),
            "n_cycles": n_cycles,
            "n_train_rows": len(tr) + len(ca),
            "best_iters": [int(m.best_iteration) for m in models],
            "sh_lgbm": sh1, "sh_lgbm_gate": sh2, "sh_hybrid": sh3, "sh_hybrid_gate": sh4,
            "net_lgbm": net1, "net_lgbm_gate": net2, "net_hybrid": net3, "net_hybrid_gate": net4,
            "skip_pct_gate2": skip2, "skip_pct_gate4": skip4,
        })
        print(f"  {ts.date()}: trained ({time.time() - t0:.0f}s)  "
              f"LGBM_only={sh1:+.2f}  +gate={sh2:+.2f}  +hybrid={sh3:+.2f}  "
              f"+both={sh4:+.2f}  skip%(gate)={skip2:.0f}% n={n_cycles}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print("WALK-FORWARD MONTHLY VALIDATION  (production cadence: monthly retrain)", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'month':>10} {'cyc':>5} {'LGBM_only':>10} {'+gate':>9} {'+hybrid':>9} "
          f"{'+both':>9} {'Δ(both-LGBM)':>14}", flush=True)
    for m in monthly:
        d = m["sh_hybrid_gate"] - m["sh_lgbm"]
        print(f"  {m['test_month']:>10} {m['n_cycles']:>5d} {m['sh_lgbm']:>+9.2f} "
              f"{m['sh_lgbm_gate']:>+8.2f} {m['sh_hybrid']:>+8.2f} {m['sh_hybrid_gate']:>+8.2f} "
              f"{d:>+13.2f}", flush=True)

    # Cumulative across months
    if monthly:
        print(f"  {'mean':>10} {'':>5} "
              f"{np.mean([m['sh_lgbm'] for m in monthly]):>+9.2f} "
              f"{np.mean([m['sh_lgbm_gate'] for m in monthly]):>+8.2f} "
              f"{np.mean([m['sh_hybrid'] for m in monthly]):>+8.2f} "
              f"{np.mean([m['sh_hybrid_gate'] for m in monthly]):>+8.2f} "
              f"{np.mean([m['sh_hybrid_gate'] - m['sh_lgbm'] for m in monthly]):>+13.2f}", flush=True)

        # Most recent 3 months
        recent = monthly[-3:]
        print(f"\n  RECENT 3 MONTHS (deployment-relevant):", flush=True)
        for m in recent:
            d = m["sh_hybrid_gate"] - m["sh_lgbm"]
            print(f"    {m['test_month']}: LGBM={m['sh_lgbm']:+.2f}  hybrid+gate={m['sh_hybrid_gate']:+.2f}  Δ={d:+.2f}", flush=True)
        recent_lgbm = np.mean([m['sh_lgbm'] for m in recent])
        recent_full = np.mean([m['sh_hybrid_gate'] for m in recent])
        print(f"  Recent-3-month means: LGBM_only={recent_lgbm:+.2f}, hybrid+gate={recent_full:+.2f}, Δ={recent_full - recent_lgbm:+.2f}", flush=True)

        # Per-month consistency
        deltas = [m["sh_hybrid_gate"] - m["sh_lgbm"] for m in monthly]
        n_pos = sum(1 for d in deltas if d > 0)
        print(f"\n  Per-month: {n_pos}/{len(deltas)} months positive Δ Sharpe (hybrid+gate vs LGBM-only)", flush=True)
        print(f"  Median Δ: {np.median(deltas):+.2f}, std: {np.std(deltas):.2f}", flush=True)

    summary = {
        "monthly": monthly,
        "n_months": len(monthly),
    }
    if monthly:
        summary["overall_mean_lgbm"] = float(np.mean([m['sh_lgbm'] for m in monthly]))
        summary["overall_mean_hybrid_gate"] = float(np.mean([m['sh_hybrid_gate'] for m in monthly]))
        summary["overall_mean_delta"] = float(np.mean([m['sh_hybrid_gate'] - m['sh_lgbm'] for m in monthly]))
        summary["recent_3mo_mean_lgbm"] = float(np.mean([m['sh_lgbm'] for m in monthly[-3:]]))
        summary["recent_3mo_mean_hybrid_gate"] = float(np.mean([m['sh_hybrid_gate'] for m in monthly[-3:]]))
        summary["pos_months_pct"] = float(100 * sum(1 for m in monthly if m['sh_hybrid_gate'] > m['sh_lgbm']) / len(monthly))
    with open(OUT_DIR / "alpha_v9_walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
