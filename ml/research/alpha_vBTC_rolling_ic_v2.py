"""Rolling-IC validator v2: full evaluator with per-cycle universe filter.

Mirrors evaluate_stacked logic (conv_gate + PM_M2 + turnover cost) but
allows the trade universe to vary per cycle (driven by rolling-IC).

Two configurations:
  static   — universe fixed by calibration folds 0-4 IC (top-15)
  rolling  — universe at time t = top-15 by trailing 60d IC

Both use K=4, conv_gate p=0.30, PM_M2_b1, turnover-based cost.
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
OUT_DIR = REPO / "outputs/vBTC_rolling_ic_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)
K = 4
COST_PER_LEG = 4.5  # bps; matches evaluate_stacked
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
MIN_OBS_PER_SYM = 100
TARGET_N = 15
ROLLING_BARS_60D = 288 * 60

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


def evaluate_with_dynamic_universe(test_df: pd.DataFrame, universe_per_t: dict | set,
                                      cost_per_leg: float = COST_PER_LEG,
                                      top_k: int = K, sample_every: int = HORIZON) -> pd.DataFrame:
    """Eval cross-sectional long-K/short-K with per-cycle universe.
    universe_per_t: dict[time, set[symbol]] OR a single set (static).
    """
    df = test_df.copy()
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]

    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    prev_scale_L = prev_scale_S = 1.0

    bars = []
    for t, g in df.groupby("open_time"):
        # Universe at this t
        if isinstance(universe_per_t, set):
            u = universe_per_t
        else:
            u = universe_per_t.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * top_k + 1:
            # Skip due to thin universe
            bars.append({"time": t, "net_bps": 0.0, "cost_bps": 0.0, "skipped": 1, "n_u": len(g_u)})
            continue

        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_arr = g_u["return_pct"].to_numpy()
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]

        # Conv gate
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())
        skip = False
        if len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), GATE_PCTILE))
            if dispersion < thr: skip = True
        dispersion_history.append(dispersion)

        # PM_M2_b1 history
        bk = min(band_k, len(g_u))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]

        if skip:
            # Hold-through mark
            if cur_long or cur_short:
                long_g = g[g["symbol"].isin(cur_long)]
                short_g = g[g["symbol"].isin(cur_short)]
                long_ret = long_g["return_pct"].mean() if not long_g.empty else 0.0
                short_ret = short_g["return_pct"].mean() if not short_g.empty else 0.0
                spread = (long_ret - short_ret) * 1e4
                bars.append({"time": t, "net_bps": spread, "cost_bps": 0.0,
                              "skipped": 1, "n_u": len(g_u)})
            else:
                bars.append({"time": t, "net_bps": 0.0, "cost_bps": 0.0, "skipped": 1, "n_u": len(g_u)})
            continue

        # PM_M2 entry persistence
        cand_long = set(sym_arr[idx_top])
        cand_short = set(sym_arr[idx_bot])
        if len(history) >= PM_M:
            past_long = [h["long"] for h in history[-PM_M:][:PM_M-1]]
            past_short = [h["short"] for h in history[-PM_M:][:PM_M-1]]
            new_long = cur_long & cand_long
            new_short = cur_short & cand_short
            for s in cand_long - cur_long:
                if all(s in p for p in past_long): new_long.add(s)
            for s in cand_short - cur_short:
                if all(s in p for p in past_short): new_short.add(s)
            # Cap at top_k
            if len(new_long) > top_k:
                ranked = sorted(new_long, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
                new_long = set(ranked)
            if len(new_short) > top_k:
                ranked = sorted(new_short, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
                new_short = set(ranked)
        else:
            new_long, new_short = cand_long, cand_short

        if not new_long or not new_short:
            bars.append({"time": t, "net_bps": 0.0, "cost_bps": 0.0, "skipped": 0, "n_u": len(g_u)})
            continue

        # PnL
        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        long_ret = long_g["return_pct"].mean()
        short_ret = short_g["return_pct"].mean()
        spread = (long_ret - short_ret) * 1e4
        # Turnover cost
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * cost_per_leg
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "cost_bps": cost, "skipped": 0, "n_u": len(g_u)})
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(bars)


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    print(f"\n=== Train all 10 folds, predict on test slices ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        test_df, pred = train_fold(panel, folds_all[fid], feat_set)
        if test_df is None: continue
        df = test_df[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = pred
        df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
    all_pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])

    # Time as int64 ms for fast windowed lookup
    ts = all_pred_df["open_time"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
            ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            ts_naive = ts
        all_pred_df["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()
    else:
        all_pred_df["t_int"] = ts.astype(np.int64).to_numpy()

    bar_ms = 5 * 60 * 1000
    window_ms = ROLLING_BARS_60D * bar_ms

    # === Build rolling universe at each rebalance time during PROD folds ===
    prod_pred = all_pred_df[all_pred_df["fold"].isin(PROD_FOLDS)].copy()
    prod_times_all = sorted(prod_pred["open_time"].unique())
    rebalance_times = prod_times_all[::HORIZON]
    print(f"\n=== Build rolling universe for {len(rebalance_times)} rebalance times ===",
          flush=True)
    universe_per_t = {}
    t0 = time.time()
    for i, t in enumerate(rebalance_times):
        # Get t as ms consistently — handles both Timestamp and numpy datetime64
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        past = all_pred_df[(all_pred_df["t_int"] >= t_ms - window_ms) &
                            (all_pred_df["t_int"] < t_ms)]
        past_clean = past.dropna(subset=["alpha_A"])
        if len(past_clean) < 1000:
            universe_per_t[t] = set()
            continue
        ics = past_clean.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        keep = ics_sorted.head(TARGET_N).index.tolist()
        universe_per_t[t] = set(keep)
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(rebalance_times)} ({time.time()-t0:.0f}s)", flush=True)
    avg_univ_size = np.mean([len(s) for s in universe_per_t.values()])
    print(f"  Avg rolling universe size: {avg_univ_size:.1f} (target {TARGET_N})", flush=True)

    # === Static universe: from calibration folds 0-4 ===
    calib = all_pred_df[all_pred_df["fold"].isin([0, 1, 2, 3, 4])].dropna(subset=["alpha_A"])
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = set(static_ics.head(TARGET_N).index.tolist())
    print(f"\n  Static universe: {sorted(static_universe)}", flush=True)

    # Test data with returns merged
    print(f"\n=== Evaluate ===", flush=True)
    test_data = prod_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    df_static = evaluate_with_dynamic_universe(test_data, static_universe)
    df_rolling = evaluate_with_dynamic_universe(test_data, universe_per_t)

    print(f"\n{'=' * 100}", flush=True)
    print(f"ROLLING-IC vs STATIC-CALIBRATION (window=60d, K={K}, target_N={TARGET_N})", flush=True)
    print(f"{'=' * 100}", flush=True)
    for label, df_v in [("static", df_static), ("rolling", df_rolling)]:
        if df_v.empty: continue
        # Drop hold-through skip rows for Sharpe computation? No, keep — same as evaluate_stacked
        net = df_v["net_bps"].to_numpy()
        n_skipped = df_v["skipped"].sum()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        avg_n_u = df_v["n_u"].mean()
        print(f"  {label:<10}: n_cycles={len(net)}  skipped={n_skipped}  avg_universe={avg_n_u:.1f}  "
              f"mean={net.mean():+.2f}  Sharpe={sh:+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]", flush=True)

    df_static.to_csv(OUT_DIR / "static_cycles.csv", index=False)
    df_rolling.to_csv(OUT_DIR / "rolling_cycles.csv", index=False)
    pd.DataFrame([
        {"time": t, "n_symbols": len(s), "symbols": ",".join(sorted(s))}
        for t, s in sorted(universe_per_t.items())
    ]).to_csv(OUT_DIR / "rolling_universe_history.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
