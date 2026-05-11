"""20-seed ensemble validation on the production stack.

Stack: flat_real skip mode + Test C overlay on rolling-IC base.
Goal: tighten CI from current 5-seed [+1.31, +6.04] toward something narrower.

5-seed Sharpe was +3.57 [+1.31, +6.04]. Expectation: 20-seed gives ~10-30%
narrower CI and slightly higher mean (better signal averaging).
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
OUT_DIR = REPO / "outputs/vBTC_20seed_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS_5 = (42, 1337, 7, 19, 2718)
SEEDS_20 = (42, 1337, 7, 19, 2718,
              99, 777, 123, 456, 789,
              314, 628, 1010, 1234, 4321,
              5555, 6789, 8765, 9999, 11111)
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
MIN_OBS_PER_SYM = 100
TARGET_N = 15
K = 4
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
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


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def train_fold(panel, fold, feat_set, seeds):
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
    for s in seeds:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test.copy(), np.mean(preds, axis=0)


def evaluate_flat_real(test_df, universe_per_t, top_k=K, sample_every=HORIZON):
    df = test_df.copy()
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]
    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    is_flat = False
    bars = []
    for t, g in df.groupby("open_time"):
        if isinstance(universe_per_t, set):
            u = universe_per_t
        else:
            u = universe_per_t.get(t, set())
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
            if not is_flat and (cur_long or cur_short):
                bars.append({"time": t, "net_bps": -2 * COST_PER_LEG, "skipped": 1})
                is_flat = True
                cur_long, cur_short = set(), set()
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
        if is_flat:
            cost = 2 * COST_PER_LEG
            is_flat = False
        else:
            churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
            churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
            cost = (churn_long + churn_short) * COST_PER_LEG
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "skipped": 0})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(bars)


def build_rolling_ic_universe(all_pred_df, target_times, ic_window_days, update_days):
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


def apply_dd_overlay(net_bps, threshold_dd=0.20, size_drawdown=0.3):
    net = np.asarray(net_bps, dtype=float)
    sizes = np.ones_like(net)
    cum = np.cumsum(net)
    peak = -np.inf
    for i in range(len(net)):
        peak = max(peak, cum[i] if i > 0 else 0)
        if peak > 0:
            dd_pct = (peak - cum[i]) / peak
            sizes[i] = size_drawdown if dd_pct > threshold_dd else 1.0
    return sizes * net, sizes


def run_validation(panel, feat_set, folds_all, seeds, label):
    print(f"\n=== Train {label} ({len(seeds)} seeds) ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p = train_fold(panel, folds_all[fid], feat_set, seeds)
        if td is None: continue
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()
    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    oos_times_all = sorted(oos_pred["open_time"].unique())
    oos_times_sampled = oos_times_all[::HORIZON]
    rolling_universe = build_rolling_ic_universe(apd, oos_times_sampled, IC_WINDOW_DAYS, IC_UPDATE_DAYS)
    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    df_v = evaluate_flat_real(test_data, rolling_universe)
    df_v["time"] = pd.to_datetime(df_v["time"])
    for fid in OOS_FOLDS:
        fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
        df_v.loc[df_v["time"].isin(fold_t), "fold"] = fid

    net = df_v["net_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    max_dd = _max_dd(net)

    overlay_net, _ = apply_dd_overlay(net)
    sh_c, lo_c, hi_c = block_bootstrap_ci(overlay_net, statistic=_sharpe,
                                            block_size=7, n_boot=2000)
    max_dd_c = _max_dd(overlay_net)

    df_v["net_with_overlay"] = overlay_net
    df_v.to_csv(OUT_DIR / f"cycles_{label}.csv", index=False)

    return {
        "label": label, "n_seeds": len(seeds),
        "no_overlay": {"sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd, "mean": net.mean()},
        "with_overlay": {"sharpe": sh_c, "ci_lo": lo_c, "ci_hi": hi_c, "max_dd": max_dd_c, "mean": overlay_net.mean()},
    }


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    results = []
    results.append(run_validation(panel, feat_set, folds_all, SEEDS_5, "5seed"))
    results.append(run_validation(panel, feat_set, folds_all, SEEDS_20, "20seed"))

    print(f"\n{'=' * 90}", flush=True)
    print(f"SEED-COUNT COMPARISON (flat_real + Test C overlay)", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"  {'config':<10}  {'overlay':<12}  {'Sharpe':>7}  {'CI':>17}  {'CI_width':>9}  {'maxDD':>7}",
          flush=True)
    for r in results:
        for ovl in ["no_overlay", "with_overlay"]:
            d = r[ovl]
            ci_w = d["ci_hi"] - d["ci_lo"]
            print(f"  {r['label']:<10}  {ovl:<12}  {d['sharpe']:>+7.2f}  "
                  f"[{d['ci_lo']:>+5.2f},{d['ci_hi']:>+5.2f}]  {ci_w:>9.2f}  "
                  f"{d['max_dd']:>+7.0f}", flush=True)

    summary = []
    for r in results:
        for ovl in ["no_overlay", "with_overlay"]:
            d = r[ovl]
            summary.append({"label": r["label"], "n_seeds": r["n_seeds"], "overlay": ovl,
                              **d, "ci_width": d["ci_hi"] - d["ci_lo"]})
    pd.DataFrame(summary).to_csv(OUT_DIR / "seed_comparison.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
