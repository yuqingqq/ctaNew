"""Rolling-IC universe validator: production-grade dynamic filter.

For each rebalance time t, the trade universe is determined by recent IC:
  - Look at past N days of (prediction, realized alpha) per symbol
  - Compute Spearman IC per symbol over that window
  - Filter universe: include symbols with IC >= threshold
  - Trade only those names

This simulates what the live system would do at every rebalance.

Compared to the static-calibration filter we've been using (where the universe
is set once from calibration period and never updated), rolling-IC adapts as:
  - Names lose fitness over time → drop
  - New names gain fitness → enter
  - Regime shifts → universe rotates

Expected: roughly comparable Sharpe to static-calib-filter, but more realistic
for production where regime drift is a real risk.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_rolling_ic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)
K = 4
ROLLING_BARS = 288 * 60   # 60 days of 5-min bars
IC_THRESHOLD = 0.02
MIN_OBS = 100
TARGET_N = 15

# Train on all 10 folds, evaluate rolling-IC on folds 5-9
ALL_FOLDS = list(range(10))
PROD_FOLDS = [5, 6, 7, 8, 9]

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def train_fold(panel, fold, feat_set):
    """Train ensemble on one fold's training data, return predictions on test."""
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test.copy(), np.mean(preds, axis=0)


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"  Winner_21 ({len(feat_set)} features), 5 seeds", flush=True)

    folds_all = _multi_oos_splits(panel)

    # === Step 1: Train all folds, save predictions ===
    print(f"\n=== Training all 10 folds (one ensemble per fold) ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        test_df, pred = train_fold(panel, folds_all[fid], feat_set)
        if test_df is None: continue
        df = test_df[["symbol", "open_time", "alpha_A"]].copy()
        df["pred"] = pred
        df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: {len(df):,} test rows ({time.time()-t0:.0f}s)", flush=True)

    all_pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    all_pred_df = all_pred_df.dropna(subset=["alpha_A"])
    print(f"  Total: {len(all_pred_df):,} rows across {all_pred_df['fold'].nunique()} folds", flush=True)

    # === Step 2: Build rolling-IC universe per cycle ===
    print(f"\n=== Computing rolling per-symbol IC (window={ROLLING_BARS // 288}d) ===", flush=True)
    # For each unique open_time, compute per-symbol IC over [t - ROLLING_BARS, t)
    # using all_pred_df. Then filter universe = symbols passing threshold.
    times = sorted(all_pred_df["open_time"].unique())
    print(f"  {len(times):,} unique timestamps", flush=True)

    # Convert open_time to int for fast comparison
    if all_pred_df["open_time"].dtype.kind == "M":
        all_pred_df["t_int"] = all_pred_df["open_time"].astype("int64") // 10**6
    else:
        all_pred_df["t_int"] = all_pred_df["open_time"].astype(np.int64)

    # Subset to production cycles only (sample every HORIZON bars in prod folds)
    prod_pred = all_pred_df[all_pred_df["fold"].isin(PROD_FOLDS)].copy()
    prod_times_all = sorted(prod_pred["open_time"].unique())
    prod_times_sampled = prod_times_all[::HORIZON]
    print(f"  Production rebalance times: {len(prod_times_sampled):,}", flush=True)

    # Build IC at each rebalance time
    print(f"  Computing rolling IC per cycle (this may take a few min)...", flush=True)
    t0 = time.time()
    universe_per_t = {}   # {t: set of symbols passing}
    for i, t in enumerate(prod_times_sampled):
        t_int = int(pd.Timestamp(t).value // 10**6) if hasattr(t, "value") else int(t.timestamp() * 1000)
        # All predictions in window [t - ROLLING_BARS_ms, t)
        window_ms = ROLLING_BARS * 5 * 60 * 1000   # 5-min bars × 5min × 60s × 1000ms
        past = all_pred_df[(all_pred_df["t_int"] >= t_int - window_ms) &
                            (all_pred_df["t_int"] < t_int)]
        if len(past) < 1000:
            universe_per_t[t] = set()
            continue
        # Per-symbol IC
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS else np.nan
        )
        # Take top-N by IC (alternative: threshold-based)
        ics_sorted = ics.dropna().sort_values(ascending=False)
        keep = ics_sorted.head(TARGET_N).index.tolist()
        # Also require IC >= threshold
        keep = [s for s in keep if ics_sorted[s] >= IC_THRESHOLD]
        universe_per_t[t] = set(keep)
        if (i+1) % 50 == 0:
            print(f"    processed {i+1}/{len(prod_times_sampled)} cycles ({time.time()-t0:.0f}s)",
                  flush=True)

    # === Step 3: Evaluate strategy with rolling universe ===
    print(f"\n=== Evaluating strategy with rolling-IC universe ===", flush=True)
    all_pred_df_idx = all_pred_df.set_index("open_time")
    cycles_rolling = []
    cycles_static = []   # comparison: static top-N from all-time IC

    # Static "all-time" universe (using calibration folds 0-4 only — same as before)
    calib = all_pred_df[all_pred_df["fold"].isin([0, 1, 2, 3, 4])]
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = set(static_ics.head(TARGET_N).index.tolist())
    print(f"  Static (calib-only) universe: {sorted(static_universe)}", flush=True)

    # Sample only at HORIZON intervals during production
    for t in prod_times_sampled:
        # Get bar at this time (5-min slice of prod_pred)
        cycle_data = prod_pred[prod_pred["open_time"] == t]
        if len(cycle_data) < 2 * K + 1: continue
        # Construct test_df for evaluator (need return_pct, alpha_realized, etc.)
        # We need to look up additional columns from panel
        cycle_full = panel[(panel["open_time"] == t) &
                            (panel["symbol"].isin(cycle_data["symbol"]))]
        if cycle_full.empty: continue
        cycle_full = cycle_full.merge(cycle_data[["symbol", "pred"]], on="symbol")

        # Rolling universe at this t
        rolling_u = universe_per_t.get(t, set())
        # Static universe
        cycle_rolling = cycle_full[cycle_full["symbol"].isin(rolling_u)].copy()
        cycle_static = cycle_full[cycle_full["symbol"].isin(static_universe)].copy()

        for label, cycle_subset, cycles_list in [
            ("rolling", cycle_rolling, cycles_rolling),
            ("static",  cycle_static,  cycles_static),
        ]:
            if len(cycle_subset) < 2 * K + 1: continue
            # Use evaluate_stacked-like simple K=4 logic
            sym_arr = cycle_subset["symbol"].to_numpy()
            pred_arr = cycle_subset["pred"].to_numpy()
            ret_arr = cycle_subset["return_pct"].to_numpy()
            idx_top = np.argpartition(-pred_arr, K-1)[:K]
            idx_bot = np.argpartition(pred_arr, K-1)[:K]
            long_ret = ret_arr[idx_top].mean()
            short_ret = ret_arr[idx_bot].mean()
            # Simple cost: 24 bps RT (conservative)
            cost_bps = 24.0
            net_bps = (long_ret - short_ret) * 1e4 - cost_bps
            cycles_list.append({"time": t, "net_bps": net_bps, "n": len(cycle_subset)})

    # === Compare ===
    print(f"\n{'=' * 100}", flush=True)
    print(f"ROLLING-IC vs STATIC CALIBRATION (window={ROLLING_BARS // 288}d, K={K}, N={TARGET_N})",
          flush=True)
    print(f"{'=' * 100}", flush=True)
    for label, cycles in [("rolling", cycles_rolling), ("static", cycles_static)]:
        df_v = pd.DataFrame(cycles)
        if df_v.empty: continue
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        print(f"  {label:<10}: n={len(net)}  mean={net.mean():+.2f}  Sharpe={sh:+.2f}  "
              f"CI=[{lo:+.2f}, {hi:+.2f}]", flush=True)

    pd.DataFrame(cycles_rolling).to_csv(OUT_DIR / "rolling_cycles.csv", index=False)
    pd.DataFrame(cycles_static).to_csv(OUT_DIR / "static_cycles.csv", index=False)
    # Save universe at each rebalance for inspection
    universe_df = pd.DataFrame([
        {"time": t, "n_symbols": len(s), "symbols": ",".join(sorted(s))}
        for t, s in sorted(universe_per_t.items())
    ])
    universe_df.to_csv(OUT_DIR / "rolling_universe_history.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
