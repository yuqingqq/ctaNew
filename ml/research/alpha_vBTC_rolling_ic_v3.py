"""Rolling-IC validator v3: test different update cadences and windows.

Configurations:
  static          — calibration period only, no updates (baseline +3.88)
  weekly_60d      — universe updated weekly, 60-day IC window
  weekly_90d      — universe updated weekly, 90-day IC window
  weekly_120d     — universe updated weekly, 120-day IC window
  monthly_90d     — universe updated monthly, 90-day IC window
  per_cycle_60d   — universe updated per cycle (4h), 60-day window (control: -0.25)

This isolates whether the issue is rotation frequency or IC window size.
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
OUT_DIR = REPO / "outputs/vBTC_rolling_ic_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)
K = 4
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
MIN_OBS_PER_SYM = 100
TARGET_N = 15

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


def evaluate_with_dynamic_universe(test_df, universe_per_t, top_k=K, cost_per_leg=COST_PER_LEG,
                                      sample_every=HORIZON):
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
            bars.append({"time": t, "net_bps": 0.0, "cost_bps": 0.0, "skipped": 1, "n_u": len(g_u)})
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
                spread = (long_ret - short_ret) * 1e4
                bars.append({"time": t, "net_bps": spread, "cost_bps": 0.0,
                              "skipped": 1, "n_u": len(g_u)})
            else:
                bars.append({"time": t, "net_bps": 0.0, "cost_bps": 0.0, "skipped": 1, "n_u": len(g_u)})
            continue

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

        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        long_ret = long_g["return_pct"].mean()
        short_ret = short_g["return_pct"].mean()
        spread = (long_ret - short_ret) * 1e4
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * cost_per_leg
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "cost_bps": cost, "skipped": 0, "n_u": len(g_u)})
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(bars)


def build_universe_schedule(all_pred_df, prod_times_sampled, window_days, update_days):
    """For each rebalance time, look up universe at the most recent update boundary.

    Updates happen every `update_days`. Universe at time t = top-15 by IC over
    past `window_days` (computed at the most recent update boundary <= t).
    """
    bar_ms = 5 * 60 * 1000
    window_ms = window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms

    all_pred_clean = all_pred_df.dropna(subset=["alpha_A"])
    # Find unique update boundaries
    if not prod_times_sampled: return {}
    t0_ms = int(pd.Timestamp(prod_times_sampled[0]).timestamp() * 1000)
    update_boundaries = []
    for t in prod_times_sampled:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        # Snap t to nearest preceding update boundary
        n_updates = (t_ms - t0_ms) // update_ms
        boundary_ms = t0_ms + n_updates * update_ms
        update_boundaries.append((t, boundary_ms))

    # Compute universe at each unique boundary
    unique_boundaries = sorted(set(b for _, b in update_boundaries))
    boundary_to_universe = {}
    for b in unique_boundaries:
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

    # Map each time to its boundary's universe
    return {t: boundary_to_universe[b] for t, b in update_boundaries}


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    print(f"\n=== Train all 10 folds ===", flush=True)
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

    # Time as int64 ms
    ts = all_pred_df["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    all_pred_df["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    prod_pred = all_pred_df[all_pred_df["fold"].isin(PROD_FOLDS)].copy()
    prod_times_all = sorted(prod_pred["open_time"].unique())
    prod_times_sampled = prod_times_all[::HORIZON]
    print(f"\n  Production rebalance times: {len(prod_times_sampled)}", flush=True)

    test_data = prod_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    # === Static universe ===
    calib = all_pred_df[all_pred_df["fold"].isin([0, 1, 2, 3, 4])].dropna(subset=["alpha_A"])
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = set(static_ics.head(TARGET_N).index.tolist())
    print(f"  Static universe: {sorted(static_universe)}", flush=True)

    # === Configurations ===
    configs = [
        ("static",         static_universe),
        ("weekly_60d",     ("weekly", 60, 7)),
        ("weekly_90d",     ("weekly", 90, 7)),
        ("weekly_120d",    ("weekly", 120, 7)),
        ("monthly_90d",    ("monthly", 90, 30)),
        ("per_cycle_60d",  ("per_cycle", 60, None)),
    ]

    print(f"\n=== Running configs ===", flush=True)
    results = []
    for label, spec in configs:
        t0 = time.time()
        if isinstance(spec, set):
            universe = spec
            avg_universe_size = len(spec)
        else:
            kind, window_days, update_days = spec
            if kind == "per_cycle":
                # Per-cycle: just compute IC at every rebalance
                universe = build_universe_schedule(all_pred_df, prod_times_sampled,
                                                     window_days, update_days=1)  # daily updates
                # Actually, want per-cycle. Use update_days=very small. But the function uses
                # update boundaries. For per-cycle, just compute at each t separately.
                window_ms = window_days * 288 * 5 * 60 * 1000
                all_pred_clean = all_pred_df.dropna(subset=["alpha_A"])
                universe = {}
                for t in prod_times_sampled:
                    t_ms = int(pd.Timestamp(t).timestamp() * 1000)
                    past = all_pred_clean[(all_pred_clean["t_int"] >= t_ms - window_ms) &
                                            (all_pred_clean["t_int"] < t_ms)]
                    if len(past) < 1000:
                        universe[t] = set()
                        continue
                    ics = past.groupby("symbol").apply(
                        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
                    )
                    ics_sorted = ics.dropna().sort_values(ascending=False)
                    universe[t] = set(ics_sorted.head(TARGET_N).index.tolist())
            else:
                universe = build_universe_schedule(all_pred_df, prod_times_sampled,
                                                     window_days, update_days)
            avg_universe_size = np.mean([len(s) for s in universe.values()])

        df_v = evaluate_with_dynamic_universe(test_data, universe)
        if df_v.empty: continue
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        results.append({"label": label, "avg_n_u": avg_universe_size,
                          "n_cycles": len(net), "skipped": int(df_v["skipped"].sum()),
                          "mean": net.mean(), "sharpe": sh, "ci_lo": lo, "ci_hi": hi})
        print(f"  {label:<14}: avg_u={avg_universe_size:>4.1f}  Sh={sh:+.2f}  "
              f"CI=[{lo:+.2f},{hi:+.2f}]  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'=' * 100}", flush=True)
    print(f"UPDATE CADENCE × WINDOW SIZE COMPARISON", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'config':<14} {'avg_u':>5}  {'n':>4}  {'skip':>4}  {'mean':>6}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}", flush=True)
    for r in results:
        print(f"  {r['label']:<14} {r['avg_n_u']:>5.1f}  {r['n_cycles']:>4}  "
              f"{r['skipped']:>4}  {r['mean']:>+6.2f}  "
              f"{r['sharpe']:>+7.2f}  {r['ci_lo']:>+7.2f}  {r['ci_hi']:>+7.2f}", flush=True)
    pd.DataFrame(results).to_csv(OUT_DIR / "rolling_cadence_compare.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
