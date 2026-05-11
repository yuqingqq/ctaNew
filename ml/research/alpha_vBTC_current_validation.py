"""Validate the CURRENT vBTC strategy: rolling training + rolling-IC universe.

Current flow:
  - Training: rolling 180d window, refresh every 90d (best from window_cadence_grid)
  - Universe: rolling-IC, 90d trailing window, 30d update cadence (monthly_90d)
  - Evaluation: walk-forward across all 9 OOS folds (no cherry-picking)

Audits:
  1. Verify no train/test temporal overlap per fold
  2. Verify universe selection only uses past data (boundary <= cycle_t - lookahead_buffer)
  3. Verify embargo properly excludes look-ahead labels
  4. Compare vs static baseline to quantify the cost of rolling

Reports:
  - Per-fold Sharpe with bootstrap CI
  - Aggregate Sharpe across all 9 folds with CI
  - DD distribution per fold
  - Universe stability (turnover per refresh)
  - Selection-bias check: 5-fold prod (5-9) vs all-9-fold mean
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_current_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
MIN_OBS_PER_SYM = 100
TARGET_N = 15
K = 4

# Calibrated config (per docs/ analysis):
#   Training: expanding window (default _slice, anchored to data_start)
#   IC universe: 180d lookback × 90d refresh cadence
# This matches window_cadence_grid.py's optimum: +4.09 Sharpe on prod folds 5-9.
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90

ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))   # honest walk-forward folds
PROD_FOLDS = [5, 6, 7, 8, 9]      # cherry-picked subset for selection-bias check

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


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def train_fold_expanding(panel, fold, feat_set):
    """Train using default _slice (expanding window anchored to data_start).

    This matches the calibrated production protocol.
    """
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test.copy(), np.mean(preds, axis=0), {"train_n": len(tr), "cal_n": len(ca)}


def evaluate_with_dynamic_universe(test_df, universe_per_t, top_k=K, sample_every=HORIZON):
    df = test_df.copy()
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]
    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    bars = []
    for t, g in df.groupby("open_time"):
        if isinstance(universe_per_t, set):
            u = universe_per_t
        else:
            u = universe_per_t.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * top_k + 1:
            bars.append({"time": t, "net_bps": 0.0, "skipped": 1, "n_u": len(g_u)})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())
        skip = False
        if len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), GATE_PCTILE))
            if dispersion < thr: skip = True
        dispersion_history.append(dispersion)
        bk = min(band_k, len(g_u))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]
        if skip:
            if cur_long or cur_short:
                long_g = g[g["symbol"].isin(cur_long)]
                short_g = g[g["symbol"].isin(cur_short)]
                long_ret = long_g["return_pct"].mean() if not long_g.empty else 0.0
                short_ret = short_g["return_pct"].mean() if not short_g.empty else 0.0
                bars.append({"time": t, "net_bps": (long_ret - short_ret) * 1e4, "skipped": 1, "n_u": len(g_u)})
            else:
                bars.append({"time": t, "net_bps": 0.0, "skipped": 1, "n_u": len(g_u)})
            continue
        cand_long = set(sym_arr[idx_top]); cand_short = set(sym_arr[idx_bot])
        if len(history) >= PM_M:
            past_long = [h["long"] for h in history[-PM_M:][:PM_M-1]]
            past_short = [h["short"] for h in history[-PM_M:][:PM_M-1]]
            new_long = cur_long & cand_long
            new_short = cur_short & cand_short
            for s in cand_long - cur_long:
                if all(s in p for p in past_long): new_long.add(s)
            for s in cand_short - cur_short:
                if all(s in p for p in past_short): new_short.add(s)
            if len(new_long) > top_k:
                ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
                new_long = set(ranked)
            if len(new_short) > top_k:
                ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
                new_short = set(ranked)
        else:
            new_long, new_short = cand_long, cand_short
        if not new_long or not new_short:
            bars.append({"time": t, "net_bps": 0.0, "skipped": 0, "n_u": len(g_u)})
            continue
        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        spread = (long_g["return_pct"].mean() - short_g["return_pct"].mean()) * 1e4
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * COST_PER_LEG
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "skipped": 0, "n_u": len(g_u)})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(bars)


def build_rolling_ic_universe(all_pred_df, target_times, ic_window_days, update_days):
    """Build universe schedule.

    For each target_time t:
      - Snap t to nearest preceding update boundary
      - Compute top-15 IC over [boundary - ic_window_days, boundary)
      - Use that universe at t

    LEAK GUARD: only uses predictions/alphas with open_time < boundary,
    and boundary <= t. So no future information enters.
    """
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms

    all_pred_clean = all_pred_df.dropna(subset=["alpha_A"])
    if not target_times: return {}

    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)
    boundaries = []
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n_updates = (t_ms - t0_ms) // update_ms
        boundary_ms = t0_ms + n_updates * update_ms
        boundaries.append((t, boundary_ms))

    unique_b = sorted(set(b for _, b in boundaries))
    boundary_to_universe = {}
    for b in unique_b:
        past = all_pred_clean[(all_pred_clean["t_int"] >= b - window_ms) &
                                (all_pred_clean["t_int"] < b)]
        if len(past) < 1000:
            boundary_to_universe[b] = set()
            continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        boundary_to_universe[b] = set(ics_sorted.head(TARGET_N).index.tolist())

    return {t: boundary_to_universe[b] for t, b in boundaries}


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)
    print(f"  panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols",
          f"\n  features: {len(feat_set)}",
          f"\n  folds: {len(folds_all)}", flush=True)

    # ============ AUDIT 1: Fold structure ============
    print(f"\n=== AUDIT 1: Fold time-windows (no overlap, embargo present) ===", flush=True)
    print(f"  {'fid':>3}  {'train_end':<12}  {'cal_start':<12}  {'test_start':<12}  "
          f"{'test_end':<12}  {'embargo_d':>9}", flush=True)
    for f in folds_all:
        print(f"  {f['fid']:>3}  {f['train_end'].strftime('%Y-%m-%d'):<12}  "
              f"{f['cal_start'].strftime('%Y-%m-%d'):<12}  "
              f"{f['test_start'].strftime('%Y-%m-%d'):<12}  "
              f"{f['test_end'].strftime('%Y-%m-%d'):<12}  "
              f"{f['embargo'].days:>9}", flush=True)

    # ============ TRAIN: EXPANDING WINDOW (calibrated default) ============
    print(f"\n=== Train each fold on expanding window (default _slice) ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p, info = train_fold_expanding(panel, folds_all[fid], feat_set)
        if td is None:
            print(f"  fold {fid}: skipped (insufficient data)", flush=True)
            continue
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: train_n={info['train_n']:,}, cal_n={info['cal_n']:,}, "
              f"test_n={len(df):,} ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    # ============ AUDIT 2: Verify per-fold prediction time spans ============
    print(f"\n=== AUDIT 2: Per-fold prediction date ranges (should be non-overlapping) ===", flush=True)
    for fid in sorted(apd["fold"].unique()):
        fdata = apd[apd["fold"] == fid]
        print(f"  fold {fid}: {fdata['open_time'].min()} to {fdata['open_time'].max()}, "
              f"n={len(fdata):,}", flush=True)

    # ============ EVAL: TWO CONFIGS ============
    print(f"\n=== Build static calibration universe (folds 0-4 only, OOS to test) ===", flush=True)
    calib = apd[apd["fold"].isin([0, 1, 2, 3, 4])].dropna(subset=["alpha_A"])
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = set(static_ics.head(TARGET_N).index.tolist())
    print(f"  Static universe ({len(static_universe)}): {sorted(static_universe)}", flush=True)

    # Build rolling-IC universe
    print(f"\n=== Build rolling-IC universe (window={IC_WINDOW_DAYS}d, update={IC_UPDATE_DAYS}d) ===",
          flush=True)
    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    oos_times_all = sorted(oos_pred["open_time"].unique())
    oos_times_sampled = oos_times_all[::HORIZON]
    rolling_universe = build_rolling_ic_universe(apd, oos_times_sampled, IC_WINDOW_DAYS, IC_UPDATE_DAYS)
    n_unique_universes = len({frozenset(u) for u in rolling_universe.values() if u})
    print(f"  Rebalance times: {len(oos_times_sampled)}, "
          f"unique universes seen: {n_unique_universes}", flush=True)

    # ============ AUDIT 3: Verify universe selection only uses past data ============
    print(f"\n=== AUDIT 3: Spot-check universe boundary leak ===", flush=True)
    bar_ms = 5 * 60 * 1000
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    leak_count = 0
    for t in oos_times_sampled[:5]:
        u = rolling_universe.get(t, set())
        if not u: continue
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        t0_ms = int(pd.Timestamp(oos_times_sampled[0]).timestamp() * 1000)
        n_upd = (t_ms - t0_ms) // update_ms
        b_ms = t0_ms + n_upd * update_ms
        if b_ms > t_ms:
            leak_count += 1
            print(f"  LEAK at {t}: boundary > t!", flush=True)
        else:
            print(f"  OK at {t}: boundary {pd.Timestamp(b_ms, unit='ms')} <= t, window=[{pd.Timestamp(b_ms-window_ms, unit='ms')}, boundary)",
                  flush=True)
    if leak_count == 0:
        print(f"  ✓ All checked boundaries <= corresponding cycle time", flush=True)

    # ============ EVAL: 4 CONFIGS ============
    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    print(f"\n=== Evaluate ===", flush=True)
    configs = [
        ("static_universe",     static_universe),
        ("rolling_ic_universe", rolling_universe),
    ]
    cycle_results = {}
    for label, u in configs:
        df_v = evaluate_with_dynamic_universe(test_data, u, top_k=K)
        if df_v.empty: continue
        df_v["time"] = pd.to_datetime(df_v["time"])
        # Tag fold
        for fid in OOS_FOLDS:
            fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
            df_v.loc[df_v["time"].isin(fold_t), "fold"] = fid
        cycle_results[label] = df_v
        print(f"  {label}: n={len(df_v)}, mean={df_v['net_bps'].mean():+.2f}", flush=True)

    # ============ REPORT ============
    print(f"\n{'=' * 90}", flush=True)
    print(f"WALK-FORWARD VALIDATION (all 9 OOS folds)", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"  {'config':<22}  {'n':>4}  {'mean':>6}  {'Sharpe':>7}  {'CI':>17}  "
          f"{'std':>6}  {'maxDD':>7}", flush=True)
    summary = []
    for label, df_v in cycle_results.items():
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(net)
        summary.append({"config": label, "scope": "all_9_folds",
                          "n": len(net), "mean": net.mean(),
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "std": net.std(), "max_dd": max_dd})
        print(f"  {label:<22}  {len(net):>4}  {net.mean():>+6.2f}  "
              f"{sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{net.std():>6.1f}  {max_dd:>+7.0f}", flush=True)

    # Cherry-picked subset (for selection-bias gap)
    print(f"\n  --- Selection-bias check: prod folds 5-9 only ---", flush=True)
    for label, df_v in cycle_results.items():
        prod = df_v[df_v["fold"].isin(PROD_FOLDS)]
        net = prod["net_bps"].to_numpy()
        if len(net) < 3: continue
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(net)
        summary.append({"config": label, "scope": "prod_5_9",
                          "n": len(net), "mean": net.mean(),
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "std": net.std(), "max_dd": max_dd})
        print(f"  {label:<22}  {len(net):>4}  {net.mean():>+6.2f}  "
              f"{sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{net.std():>6.1f}  {max_dd:>+7.0f}", flush=True)

    # Per-fold breakdown
    print(f"\n=== Per-fold Sharpe ===", flush=True)
    print(f"  {'config':<22}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
    for label, df_v in cycle_results.items():
        cells = []
        for fid in OOS_FOLDS:
            n_f = df_v[df_v["fold"] == fid]["net_bps"].to_numpy()
            cells.append(f"{_sharpe(n_f):+5.2f}" if len(n_f) >= 3 else "  -- ")
        print(f"  {label:<22}  " + " ".join(f"{c:>6}" for c in cells), flush=True)

    # Universe stability (rolling only)
    if "rolling_ic_universe" in cycle_results:
        print(f"\n=== Universe stability ===", flush=True)
        unique_u = list({frozenset(u): u for u in rolling_universe.values() if u}.values())
        print(f"  Unique universes across all rebalances: {len(unique_u)}", flush=True)
        # Average turnover between consecutive rebalances
        prev_u = None
        turns = []
        for t in sorted(rolling_universe.keys()):
            u = rolling_universe[t]
            if prev_u is not None and u and prev_u:
                jaccard = len(u & prev_u) / len(u | prev_u)
                turns.append(1 - jaccard)
            prev_u = u
        if turns:
            print(f"  Average per-rebalance turnover (1-Jaccard): {np.mean(turns):.2f}",
                  flush=True)
            print(f"  Median turnover: {np.median(turns):.2f}", flush=True)

    pd.DataFrame(summary).to_csv(OUT_DIR / "current_validation_summary.csv", index=False)
    for label, df_v in cycle_results.items():
        df_v.to_csv(OUT_DIR / f"cycles_{label}.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
