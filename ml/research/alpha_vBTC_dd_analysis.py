"""Drawdown + time-concentration analysis on the final architecture.

Production-critical questions:
  1. When did VVV's +16k bps come from? Concentrated days or consistent?
  2. Max drawdown, longest underwater period
  3. Top 10 best/worst cycles — is PnL driven by tail or by base rate?
  4. Distribution of cycle-level returns
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
OUT_DIR = REPO / "outputs/vBTC_dd_analysis"
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


def evaluate_with_attribution(test_df, universe_per_t, top_k=K, sample_every=HORIZON):
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
        u = universe_per_t.get(t, set()) if not isinstance(universe_per_t, set) else universe_per_t
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
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


def build_schedule(all_pred_clean, prod_times, window_days, cadence_days):
    bar_ms = 5 * 60 * 1000
    window_ms = window_days * 288 * bar_ms
    cadence_ms = cadence_days * 288 * bar_ms
    if not prod_times: return {}
    t0_ms = int(pd.Timestamp(prod_times[0]).timestamp() * 1000)
    schedule = {}
    boundary_to_universe = {}
    for t in prod_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // cadence_ms
        boundary_ms = t0_ms + n * cadence_ms
        if boundary_ms not in boundary_to_universe:
            past = all_pred_clean[(all_pred_clean["t_int"] >= boundary_ms - window_ms) &
                                    (all_pred_clean["t_int"] < boundary_ms)]
            if len(past) < 1000:
                boundary_to_universe[boundary_ms] = set()
            else:
                ics = past.groupby("symbol").apply(
                    lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
                )
                ics_sorted = ics.dropna().sort_values(ascending=False)
                boundary_to_universe[boundary_ms] = set(ics_sorted.head(TARGET_N).index.tolist())
        schedule[t] = boundary_to_universe[boundary_ms]
    return schedule


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
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()
    apd_clean = apd.dropna(subset=["alpha_A"])

    prod_pred = apd[apd["fold"].isin(PROD_FOLDS)].copy()
    prod_times = sorted(prod_pred["open_time"].unique())
    prod_times_sampled = prod_times[::HORIZON]
    test_data = prod_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    schedule_180_90 = build_schedule(apd_clean, prod_times_sampled, 180, 90)

    print(f"\n=== Drawdown analysis (final winner: F21+K=3+180×90) ===", flush=True)
    df_eval, df_contrib = evaluate_with_attribution(test_data, schedule_180_90, top_k=K)
    df_eval = df_eval.sort_values("time").reset_index(drop=True)

    # Cumulative PnL
    df_eval["cum_bps"] = df_eval["net_bps"].cumsum()
    df_eval["peak"] = df_eval["cum_bps"].cummax()
    df_eval["dd"] = df_eval["cum_bps"] - df_eval["peak"]

    print(f"  Total cycles: {len(df_eval)}", flush=True)
    print(f"  Total PnL: {df_eval['net_bps'].sum():+.0f} bps", flush=True)
    print(f"  Mean PnL/cycle: {df_eval['net_bps'].mean():+.2f} bps", flush=True)
    print(f"  Std PnL/cycle: {df_eval['net_bps'].std():.2f} bps", flush=True)
    print(f"  Max drawdown: {df_eval['dd'].min():+.0f} bps", flush=True)
    print(f"  Drawdown / total PnL ratio: {abs(df_eval['dd'].min()) / max(df_eval['net_bps'].sum(), 1):.2%}",
          flush=True)

    # Top 10 best/worst cycles
    print(f"\n=== Top 10 best cycles ===", flush=True)
    top_best = df_eval.nlargest(10, "net_bps")[["time", "net_bps"]]
    print(top_best.to_string(index=False), flush=True)
    print(f"\n=== Top 10 worst cycles ===", flush=True)
    top_worst = df_eval.nsmallest(10, "net_bps")[["time", "net_bps"]]
    print(top_worst.to_string(index=False), flush=True)

    # Top 10 cycles' contribution to total
    sorted_pnl = df_eval["net_bps"].abs().sort_values(ascending=False)
    top10_abs = sorted_pnl.head(10).sum()
    print(f"\n  Top 10 absolute-PnL cycles contribute {top10_abs:+.0f} bps "
          f"({top10_abs/df_eval['net_bps'].abs().sum()*100:.1f}% of total |PnL|)", flush=True)

    # VVV time concentration
    print(f"\n=== VVV time concentration ===", flush=True)
    vvv = df_contrib[df_contrib["symbol"] == "VVVUSDT"].copy()
    vvv = vvv.sort_values("time").reset_index(drop=True)
    vvv["cum"] = vvv["ret_bps"].cumsum()
    print(f"  VVV picks: {len(vvv)}", flush=True)
    print(f"  Total VVV PnL: {vvv['ret_bps'].sum():+.0f} bps", flush=True)
    sorted_vvv = vvv["ret_bps"].abs().sort_values(ascending=False)
    print(f"  Top 5 VVV cycles contribute {sorted_vvv.head(5).sum():+.0f} bps "
          f"({sorted_vvv.head(5).sum()/vvv['ret_bps'].abs().sum()*100:.1f}% of |VVV PnL|)", flush=True)
    print(f"  Top 10 VVV cycles contribute {sorted_vvv.head(10).sum():+.0f} bps "
          f"({sorted_vvv.head(10).sum()/vvv['ret_bps'].abs().sum()*100:.1f}% of |VVV PnL|)", flush=True)

    # Distribution
    print(f"\n=== PnL/cycle distribution ===", flush=True)
    pcts = df_eval["net_bps"].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"  {'pctile':>7}  {'value (bps)':>12}", flush=True)
    for p, v in pcts.items():
        print(f"  {p:>6.2%}  {v:>+12.2f}", flush=True)

    # Skew
    pnl = df_eval["net_bps"]
    print(f"\n  Skew: {pnl.skew():+.2f}  Kurtosis: {pnl.kurtosis():+.1f}", flush=True)
    print(f"  Mean: {pnl.mean():+.2f}  Median: {pnl.median():+.2f}", flush=True)

    # If skew is highly positive and median << mean, alpha is concentrated in winners (lottery-like)
    df_eval.to_csv(OUT_DIR / "cycle_pnl.csv", index=False)
    df_contrib.to_csv(OUT_DIR / "contributions.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
