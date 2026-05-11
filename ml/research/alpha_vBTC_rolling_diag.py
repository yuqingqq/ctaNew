"""Diagnostic: why does rolling-IC selection underperform static-calibration?

Core question: does past-60d per-symbol IC predict next-7d (or next-30d) IC?

If YES (high correlation) → rolling SHOULD work. The fact it doesn't would mean
                              there's churn cost, or model differences, or another bug.
If NO  (low correlation)  → rolling-IC is selecting on noise. Static-calibration
                              works because long-window IC is more stable.

Three diagnostics:
  D1. Per-rebalance: past-60d IC vs forward-7d IC per symbol (rank correlation)
  D2. Universe membership turnover (% of names changing per week)
  D3. Per-symbol contribution: do calibration-stable names beat rolling-rotation names?
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

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_rolling_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
MIN_OBS_PER_SYM = 100
TARGET_N = 15
PROD_FOLDS = [5, 6, 7, 8, 9]
ALL_FOLDS = list(range(10))

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
        df = test_df[["symbol", "open_time", "alpha_A"]].copy()
        df["pred"] = pred
        df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: {len(df):,} ({time.time()-t0:.0f}s)", flush=True)
    all_pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    ts = all_pred_df["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    all_pred_df["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    bar_ms = 5 * 60 * 1000
    window_60d_ms = 60 * 288 * bar_ms
    forward_7d_ms = 7 * 288 * bar_ms

    # Calibration universe (folds 0-4)
    calib = all_pred_df[all_pred_df["fold"].isin([0, 1, 2, 3, 4])].dropna(subset=["alpha_A"])
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    static_universe = sorted(static_ics.head(TARGET_N).index.tolist())
    print(f"\n  Static universe ({TARGET_N}): {static_universe}", flush=True)

    # Production weekly rebalance times
    prod_pred = all_pred_df[all_pred_df["fold"].isin(PROD_FOLDS)]
    prod_times = sorted(prod_pred["open_time"].unique())
    weekly_times = prod_times[::7 * 288]   # weekly samples (every 7 days)
    print(f"  Weekly rebalance points in prod: {len(weekly_times)}", flush=True)

    # === D1: past-60d IC vs forward-7d IC per symbol ===
    print(f"\n=== D1: past-60d IC vs next-7d IC (does past predict future?) ===", flush=True)
    print(f"  At each weekly time t, compute past-60d IC and next-7d IC per symbol.", flush=True)
    print(f"  Rank correlation between past and future tells us if rolling-IC works.", flush=True)
    rank_corrs = []
    overlap_top15 = []
    all_pred_clean = all_pred_df.dropna(subset=["alpha_A"])
    for t in weekly_times[:-2]:   # exclude last 2 weeks (no future data)
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        past = all_pred_clean[(all_pred_clean["t_int"] >= t_ms - window_60d_ms) &
                                (all_pred_clean["t_int"] < t_ms)]
        future = all_pred_clean[(all_pred_clean["t_int"] >= t_ms) &
                                  (all_pred_clean["t_int"] < t_ms + forward_7d_ms)]
        past_ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        ).dropna()
        fut_ics = future.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 50 else np.nan
        ).dropna()
        common = past_ics.index.intersection(fut_ics.index)
        if len(common) < 20: continue
        # Spearman correlation between past-rank and future-rank
        rank_corr = past_ics[common].rank().corr(fut_ics[common].rank())
        rank_corrs.append(rank_corr)
        # Overlap: top-15 by past vs top-15 by future
        top15_past = set(past_ics.sort_values(ascending=False).head(TARGET_N).index)
        top15_fut = set(fut_ics.sort_values(ascending=False).head(TARGET_N).index)
        overlap = len(top15_past & top15_fut)
        overlap_top15.append(overlap)

    print(f"  Past-60d IC vs next-7d IC rank correlation: mean={np.mean(rank_corrs):+.3f}  "
          f"std={np.std(rank_corrs):.3f}  median={np.median(rank_corrs):+.3f}", flush=True)
    print(f"  Top-15 overlap (past vs future): mean={np.mean(overlap_top15):.1f} / {TARGET_N}  "
          f"min={np.min(overlap_top15)}  max={np.max(overlap_top15)}", flush=True)

    # === D2: rolling universe turnover ===
    print(f"\n=== D2: weekly universe turnover ===", flush=True)
    weekly_universes = []
    for t in weekly_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        past = all_pred_clean[(all_pred_clean["t_int"] >= t_ms - window_60d_ms) &
                                (all_pred_clean["t_int"] < t_ms)]
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        ).dropna().sort_values(ascending=False)
        weekly_universes.append(set(ics.head(TARGET_N).index.tolist()))

    turnover = []
    for i in range(1, len(weekly_universes)):
        churn = len(weekly_universes[i].symmetric_difference(weekly_universes[i-1]))
        turnover.append(churn)
    print(f"  Symbols changing per week: mean={np.mean(turnover):.1f}/{TARGET_N}  "
          f"max={np.max(turnover)}  min={np.min(turnover)}", flush=True)

    # === D3: rolling vs static universe overlap ===
    print(f"\n=== D3: rolling vs static universe overlap ===", flush=True)
    overlaps_w_static = [len(u & set(static_universe)) for u in weekly_universes]
    print(f"  Rolling universe ∩ static: mean={np.mean(overlaps_w_static):.1f}/{TARGET_N}  "
          f"min={np.min(overlaps_w_static)}  max={np.max(overlaps_w_static)}", flush=True)

    # === D4: full-period IC of static names vs full-period IC of often-rolling-picked names ===
    print(f"\n=== D4: full-period IC of static names vs rolling-picked names ===", flush=True)
    prod_clean = prod_pred.dropna(subset=["alpha_A"])
    full_period_ics = prod_clean.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    print(f"  Per-symbol IC over FULL prod period:", flush=True)
    print(f"    {'symbol':<14} {'IC':>+8}  {'in_static':>9}  {'rolling_freq':>12}", flush=True)
    rolling_pick_count = {}
    for u in weekly_universes:
        for s in u:
            rolling_pick_count[s] = rolling_pick_count.get(s, 0) + 1
    for s in full_period_ics.index[:25]:
        in_static = "YES" if s in static_universe else ""
        roll_freq = rolling_pick_count.get(s, 0) / max(len(weekly_universes), 1)
        print(f"    {s:<14} {full_period_ics[s]:>+8.4f}  {in_static:>9}  {roll_freq:>11.1%}",
              flush=True)

    # === D5: how good are static's symbols' IC over full prod period? ===
    static_full_ic = full_period_ics.reindex(static_universe).dropna()
    rolling_freq_top = sorted(rolling_pick_count.items(), key=lambda x: -x[1])[:15]
    rolling_top_syms = [s for s, _ in rolling_freq_top]
    rolling_full_ic = full_period_ics.reindex(rolling_top_syms).dropna()
    print(f"\n  Avg full-period IC of static-15 symbols: {static_full_ic.mean():+.4f}", flush=True)
    print(f"  Avg full-period IC of top-15-most-rolling-picked symbols: {rolling_full_ic.mean():+.4f}",
          flush=True)
    print(f"  → {'static' if static_full_ic.mean() > rolling_full_ic.mean() else 'rolling'} picks better symbols on avg",
          flush=True)

    # Save
    pd.DataFrame({"weekly_t": weekly_times[:-2],
                   "past_vs_future_rank_corr": rank_corrs + [np.nan] * (len(weekly_times)-2 - len(rank_corrs)),
                   "top15_overlap": overlap_top15 + [np.nan] * (len(weekly_times)-2 - len(overlap_top15))}
                  ).to_csv(OUT_DIR / "past_vs_future_per_week.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
