"""Walk-forward validation of the combined overlay (Test D winner).

For each fold f in 1..9:
  - Calibration: prior folds 0..f-1
  - Apply combined overlay: continuous dispersion sizing + 20% DD deleveraging
  - Evaluate fold f
  - Track Sharpe AND drawdown per fold

Tests whether the +4.71 Sharpe gain is consistent across periods or
period-specific. Compares overlay vs no-overlay for each fold.
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
OUT_DIR = REPO / "outputs/vBTC_overlay_walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718, 99, 777, 123, 456, 789)
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
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


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


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


def evaluate_overlay(test_df, universe, top_k=K, use_overlay=True,
                       dd_threshold=0.20, dd_size_low=0.5,
                       cost_per_leg=COST_PER_LEG, sample_every=HORIZON):
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
    cum_pnl = 0.0
    peak = 0.0
    prev_size = 1.0
    for t, g in df.groupby("open_time"):
        g_u = g[g["symbol"].isin(universe)]
        if len(g_u) < 2 * top_k + 1:
            bars.append({"time": t, "net_bps": 0.0, "size": 0.0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())

        size_disp = 1.0
        if use_overlay and len(dispersion_history) >= 30:
            past_disp = np.array(list(dispersion_history))
            pctile = (past_disp < dispersion).mean()
            size_disp = 1.0 / (1.0 + np.exp(-(pctile - 0.30) * 20))
        dispersion_history.append(dispersion)

        size_dd = 1.0
        if use_overlay and peak > 0:
            dd_pct = (peak - cum_pnl) / peak
            if dd_pct > dd_threshold:
                size_dd = dd_size_low

        size = float(np.clip(size_disp * size_dd, 0.0, 1.0))

        bk = min(band_k, len(g_u))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]

        if size <= 0.001:
            if cur_long or cur_short:
                long_g = g[g["symbol"].isin(cur_long)]
                short_g = g[g["symbol"].isin(cur_short)]
                long_ret = long_g["return_pct"].mean() if not long_g.empty else 0.0
                short_ret = short_g["return_pct"].mean() if not short_g.empty else 0.0
                net = prev_size * (long_ret - short_ret) * 1e4
            else:
                net = 0.0
            bars.append({"time": t, "net_bps": net, "size": 0.0})
            cum_pnl += net; peak = max(peak, cum_pnl)
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
            bars.append({"time": t, "net_bps": 0.0, "size": 0.0})
            continue

        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        spread = (long_g["return_pct"].mean() - short_g["return_pct"].mean()) * 1e4
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * cost_per_leg * size
        net = size * spread - cost
        bars.append({"time": t, "net_bps": net, "size": size})
        cum_pnl += net; peak = max(peak, cum_pnl)
        cur_long, cur_short = new_long, new_short
        prev_size = size

    return pd.DataFrame(bars)


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

    # Walk-forward: for each fold, calibrate from prior, evaluate with/without overlay
    print(f"\n=== Walk-forward overlay validation ===", flush=True)
    print(f"  {'fold':>4}  {'period':<25}  {'no_ovly_Sh':>10}  {'no_ovly_DD':>10}  "
          f"{'ovly_Sh':>8}  {'ovly_DD':>8}  {'d_Sh':>5}  {'DD_red':>7}", flush=True)
    no_overlay_results = []
    overlay_results = []
    for f in range(1, 10):
        if f not in fold_data: continue
        calib_folds = list(range(f))
        calib = apd[apd["fold"].isin(calib_folds)].dropna(subset=["alpha_A"])
        if len(calib) < 1000:
            continue
        ics = calib.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        ).dropna().sort_values(ascending=False)
        universe = set(ics.head(TARGET_N).index.tolist())

        test_data_f = apd[apd["fold"] == f][["symbol", "open_time", "pred", "return_pct", "alpha_A"]]
        df_no = evaluate_overlay(test_data_f, universe, use_overlay=False)
        df_ovly = evaluate_overlay(test_data_f, universe, use_overlay=True)
        if df_no.empty: continue

        net_no = df_no["net_bps"].to_numpy()
        net_ov = df_ovly["net_bps"].to_numpy()
        sh_no = _sharpe(net_no)
        sh_ov = _sharpe(net_ov)
        dd_no = _max_dd(net_no)
        dd_ov = _max_dd(net_ov)
        dd_red = (dd_no - dd_ov) / dd_no * 100 if dd_no != 0 else 0

        if not test_data_f.empty:
            t_min = pd.Timestamp(test_data_f["open_time"].min()).strftime("%Y-%m-%d")
            t_max = pd.Timestamp(test_data_f["open_time"].max()).strftime("%m-%d")
            period = f"{t_min} to {t_max}"
        else:
            period = "?"
        print(f"  {f:>4}  {period:<25}  {sh_no:>+10.2f}  {dd_no:>+10.0f}  "
              f"{sh_ov:>+8.2f}  {dd_ov:>+8.0f}  {sh_ov-sh_no:>+5.2f}  {dd_red:>+6.0f}%",
              flush=True)
        no_overlay_results.append({"fold": f, "sharpe": sh_no, "max_dd": dd_no, "n": len(net_no)})
        overlay_results.append({"fold": f, "sharpe": sh_ov, "max_dd": dd_ov, "n": len(net_ov)})

    print(f"\n=== Aggregate (across all walk-forward folds) ===", flush=True)
    df_no = pd.DataFrame(no_overlay_results)
    df_ov = pd.DataFrame(overlay_results)
    print(f"  Without overlay: mean Sharpe={df_no['sharpe'].mean():+.2f}  "
          f"std={df_no['sharpe'].std():.2f}  worst Sharpe={df_no['sharpe'].min():+.2f}  "
          f"worst DD={df_no['max_dd'].min():+.0f}", flush=True)
    print(f"  With overlay:    mean Sharpe={df_ov['sharpe'].mean():+.2f}  "
          f"std={df_ov['sharpe'].std():.2f}  worst Sharpe={df_ov['sharpe'].min():+.2f}  "
          f"worst DD={df_ov['max_dd'].min():+.0f}", flush=True)
    print(f"\n  Sharpe Δ (overlay - no_overlay): mean=+{(df_ov['sharpe'].mean() - df_no['sharpe'].mean()):.2f}  "
          f"min={(df_ov['sharpe'] - df_no['sharpe']).min():+.2f}  "
          f"max={(df_ov['sharpe'] - df_no['sharpe']).max():+.2f}", flush=True)
    print(f"  Folds where overlay HELPS Sharpe: "
          f"{((df_ov['sharpe'] - df_no['sharpe']) > 0).sum()} of {len(df_ov)}", flush=True)
    print(f"  Folds where overlay REDUCES DD: "
          f"{(df_ov['max_dd'] > df_no['max_dd']).sum()} of {len(df_ov)}", flush=True)

    pd.DataFrame(no_overlay_results).to_csv(OUT_DIR / "no_overlay.csv", index=False)
    pd.DataFrame(overlay_results).to_csv(OUT_DIR / "overlay.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
