"""Test concentration risk: how much of Sharpe is driven by VVV?

For each (K, exclude_VVV) combination:
  - Same trained model (F21 features, 10-seed ensemble)
  - Same 180×90 schedule
  - K=3, 4, 5
  - With and without VVV in trade universe

Tells us:
  - How much Sharpe drops when VVV is removed
  - Whether higher K provides diversification
  - The honest "no-single-name" Sharpe
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
OUT_DIR = REPO / "outputs/vBTC_concentration_test"
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


def evaluate_with_dynamic_universe(test_df, universe_per_t, top_k, cost_per_leg=COST_PER_LEG,
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
        cost = (churn_long + churn_short) * cost_per_leg
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "skipped": 0})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(bars)


def build_schedule(all_pred_clean, prod_times, window_days, cadence_days, exclude=None):
    bar_ms = 5 * 60 * 1000
    window_ms = window_days * 288 * bar_ms
    cadence_ms = cadence_days * 288 * bar_ms
    exclude = exclude or set()
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
            past = past[~past["symbol"].isin(exclude)]
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

    schedule_full = build_schedule(apd_clean, prod_times_sampled, 180, 90)
    schedule_no_vvv = build_schedule(apd_clean, prod_times_sampled, 180, 90, exclude={"VVVUSDT"})
    schedule_no_top2 = build_schedule(apd_clean, prod_times_sampled, 180, 90, exclude={"VVVUSDT", "ORDIUSDT"})

    # Filter test data to exclude VVV (and ORDI for the strictest test)
    test_no_vvv = test_data[~test_data["symbol"].isin(["VVVUSDT"])].copy()
    test_no_top2 = test_data[~test_data["symbol"].isin(["VVVUSDT", "ORDIUSDT"])].copy()

    print(f"\n=== K × Exclusion test (10-seed F21+180×90) ===", flush=True)
    print(f"  {'config':<30}  {'n':>4}  {'mean':>6}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}",
          flush=True)
    results = []
    for label, td, sched, K in [
        ("baseline_full_K3",        test_data, schedule_full, 3),
        ("baseline_full_K4",        test_data, schedule_full, 4),
        ("baseline_full_K5",        test_data, schedule_full, 5),
        ("no_VVV_K3",               test_no_vvv, schedule_no_vvv, 3),
        ("no_VVV_K4",               test_no_vvv, schedule_no_vvv, 4),
        ("no_VVV_K5",               test_no_vvv, schedule_no_vvv, 5),
        ("no_top2_K3",              test_no_top2, schedule_no_top2, 3),
        ("no_top2_K4",              test_no_top2, schedule_no_top2, 4),
        ("no_top2_K5",              test_no_top2, schedule_no_top2, 5),
    ]:
        df_eval = evaluate_with_dynamic_universe(td, sched, top_k=K)
        net = df_eval["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        per_fold = {}
        for fid in PROD_FOLDS:
            fold_t = set(prod_pred[prod_pred["fold"] == fid]["open_time"].unique())
            mask = df_eval["time"].isin(fold_t)
            n_f = df_eval.loc[mask, "net_bps"].to_numpy()
            if len(n_f) >= 3:
                per_fold[fid] = _sharpe(n_f)
        results.append({"label": label, "n": len(net), "mean": net.mean(),
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        print(f"  {label:<30}  {len(net):>4}  {net.mean():>+6.2f}  {sh:>+7.2f}  "
              f"{lo:>+7.2f}  {hi:>+7.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'config':<30}  " + " ".join(f"{'fold' + str(f):>9}" for f in PROD_FOLDS), flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['label']:<30}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "concentration_test.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
