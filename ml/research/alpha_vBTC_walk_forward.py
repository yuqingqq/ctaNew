"""Walk-forward attribution: does the architecture find alpha across periods?

For each test fold f in {1, 2, ..., 9}:
  - Calibration: top-15 by IC over folds 0..f-1 (only prior data)
  - Evaluate strategy on fold f with that universe
  - Record per-symbol PnL contribution

Output:
  - Per-fold Sharpe over time
  - Top-3 contributors per fold
  - Stability of contributor names across folds

This tests whether the architecture's edge is regime-stable or
regime-specific (different winners per period).
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

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_walk_forward"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718, 99, 777, 123, 456, 789)
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
MIN_OBS_PER_SYM = 100
TARGET_N = 15
K = 3
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


def evaluate_with_attribution(test_df, universe, top_k=K, sample_every=HORIZON):
    df = test_df.copy()
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame(), pd.DataFrame()
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]
    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    bars = []; contributions = []
    for t, g in df.groupby("open_time"):
        g_u = g[g["symbol"].isin(universe)]
        if len(g_u) < 2 * top_k + 1:
            bars.append({"time": t, "net_bps": 0.0, "skipped": 1})
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
                bars.append({"time": t, "net_bps": (long_ret - short_ret) * 1e4, "skipped": 1})
            else:
                bars.append({"time": t, "net_bps": 0.0, "skipped": 1})
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
            bars.append({"time": t, "net_bps": 0.0, "skipped": 0})
            continue
        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        spread = (long_g["return_pct"].mean() - short_g["return_pct"].mean()) * 1e4
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * COST_PER_LEG
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "skipped": 0})
        for s in new_long:
            sub = long_g[long_g["symbol"] == s]
            if not sub.empty:
                contributions.append({"time": t, "symbol": s, "side": "long",
                                      "ret_bps": float(sub["return_pct"].iloc[0]) * 1e4 / len(new_long)})
        for s in new_short:
            sub = short_g[short_g["symbol"] == s]
            if not sub.empty:
                contributions.append({"time": t, "symbol": s, "side": "short",
                                      "ret_bps": -float(sub["return_pct"].iloc[0]) * 1e4 / len(new_short)})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(bars), pd.DataFrame(contributions)


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    print(f"\n=== Train all 10 folds × 10 seeds ===", flush=True)
    fold_data = {}
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p = train_fold(panel, folds_all[fid], feat_set)
        if td is not None: fold_data[fid] = (td, p)
        print(f"  fold {fid}: ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for fid, (td, p) in fold_data.items():
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        rows.append(df)
    apd = pd.concat(rows, ignore_index=True).sort_values(["open_time", "symbol"])

    # === Walk-forward evaluation ===
    # For each test fold f >= 1, calibrate from folds 0..f-1, evaluate on f
    print(f"\n=== Walk-forward attribution (calib = expanding prior folds) ===", flush=True)
    print(f"  {'fold':>4} {'date_range':<25} {'cal_folds':<12} {'top_3_in_universe':<60} "
          f"{'Sharpe':>7}", flush=True)

    fold_results = []
    fold_attributions = []
    for f in range(1, 10):
        if f not in fold_data: continue
        # Calibration: prior folds 0..f-1
        calib_folds = list(range(f))
        calib = apd[apd["fold"].isin(calib_folds)].dropna(subset=["alpha_A"])
        if len(calib) < 1000:
            continue
        ics = calib.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        ).dropna().sort_values(ascending=False)
        universe = set(ics.head(TARGET_N).index.tolist())
        # Evaluate on fold f
        test_data_f = apd[apd["fold"] == f][["symbol", "open_time", "pred", "return_pct", "alpha_A"]]
        df_eval, df_contrib = evaluate_with_attribution(test_data_f, universe, top_k=K)
        if df_eval.empty: continue
        net = df_eval["net_bps"].to_numpy()
        sh = _sharpe(net) if len(net) >= 3 else 0
        # Per-symbol contributions
        sym_attr = df_contrib.groupby("symbol")["ret_bps"].sum().sort_values(ascending=False)
        top3 = ", ".join(f"{s}({v:+.0f})" for s, v in sym_attr.head(3).items())
        # Date range of test fold
        if not test_data_f.empty:
            t_min = pd.Timestamp(test_data_f["open_time"].min()).strftime("%Y-%m-%d")
            t_max = pd.Timestamp(test_data_f["open_time"].max()).strftime("%Y-%m-%d")
            date_range = f"{t_min} to {t_max[5:]}"
        else:
            date_range = "?"
        print(f"  {f:>4} {date_range:<25} {str(calib_folds):<12} {top3:<60} {sh:>+7.2f}",
              flush=True)
        fold_results.append({"fold": f, "calib_folds": str(calib_folds),
                              "n_cycles": len(net), "mean_bps": net.mean(),
                              "sharpe": sh, "universe": ",".join(sorted(universe))})
        for s, v in sym_attr.items():
            fold_attributions.append({"fold": f, "symbol": s, "pnl_bps": v})

    # Cross-fold contributor analysis
    print(f"\n=== Cross-fold top-3 contributor analysis ===", flush=True)
    df_attr = pd.DataFrame(fold_attributions)
    if not df_attr.empty:
        # For each fold, find top 3 contributors
        print(f"  Top-3 contributors per fold:", flush=True)
        for f in sorted(df_attr["fold"].unique()):
            top3 = df_attr[df_attr["fold"] == f].nlargest(3, "pnl_bps")
            print(f"    fold {f}: " + " | ".join(f"{r['symbol']}({r['pnl_bps']:+.0f})"
                                                    for _, r in top3.iterrows()), flush=True)

        # How many unique symbols across all folds' top-3?
        all_top3 = set()
        for f in sorted(df_attr["fold"].unique()):
            top3 = df_attr[df_attr["fold"] == f].nlargest(3, "pnl_bps")["symbol"]
            all_top3.update(top3.tolist())
        print(f"\n  Unique symbols appearing in top-3 across all folds: {len(all_top3)}", flush=True)
        print(f"    {sorted(all_top3)}", flush=True)

        # Symbols appearing in top-3 of multiple folds (= persistent winners)
        sym_appearances = {}
        for f in sorted(df_attr["fold"].unique()):
            top3 = df_attr[df_attr["fold"] == f].nlargest(3, "pnl_bps")["symbol"]
            for s in top3:
                sym_appearances[s] = sym_appearances.get(s, 0) + 1
        persistent = sorted(sym_appearances.items(), key=lambda x: -x[1])
        print(f"\n  Persistence ranking (how often a symbol appears in top-3):", flush=True)
        for s, n in persistent:
            print(f"    {s}: {n} folds", flush=True)

    pd.DataFrame(fold_results).to_csv(OUT_DIR / "walk_forward_results.csv", index=False)
    df_attr.to_csv(OUT_DIR / "walk_forward_attributions.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
