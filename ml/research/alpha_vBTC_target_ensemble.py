"""Target-ensemble test: train on target_A and target_D separately, average.

target_A and target_D are different basket-residual definitions (corr ~0.86).
If ensembling helps, it's directional evidence that multi-horizon would too.

Variants:
  target_A_only     : current production
  target_D_only     : alternative target
  target_AD_avg     : average of A and D model predictions

Stack: rolling-IC 180/90, expanding train, K=4, N=15, WINNER_21,
flat_real, Test C overlay.
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
OUT_DIR = REPO / "outputs/vBTC_target_ensemble"
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
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))

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


def train_fold_on_target(panel, fold, feat_set, target_col):
    """Train ensemble on a specific target column."""
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test[feat_set].to_numpy(np.float32)
    yt = tr[target_col].to_numpy(np.float32)
    yc = ca[target_col].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
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
        n = (t_ms - t0_ms) // update_ms
        b = t0_ms + n * update_ms
        boundaries.append((t, b))
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


def run_with_predictions(panel, all_preds_df, label):
    """Eval a prediction set with rolling-IC universe + flat_real + Test C."""
    apd = all_preds_df.sort_values(["open_time", "symbol"]).copy()
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    oos_times_all = sorted(oos_pred["open_time"].unique())
    oos_times_sampled = oos_times_all[::HORIZON]
    rolling_universe = build_rolling_ic_universe(apd, oos_times_sampled,
                                                    IC_WINDOW_DAYS, IC_UPDATE_DAYS)
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
    sh_o, lo_o, hi_o = block_bootstrap_ci(overlay_net, statistic=_sharpe,
                                            block_size=7, n_boot=2000)
    max_dd_o = _max_dd(overlay_net)
    per_fold_o = {}
    df_temp = df_v.copy(); df_temp["scaled"] = overlay_net
    for fid in OOS_FOLDS:
        fdat = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
        if len(fdat) >= 3:
            per_fold_o[fid] = _sharpe(fdat)

    return {
        "label": label,
        "no_overlay": {"sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd, "mean": net.mean()},
        "with_overlay": {"sharpe": sh_o, "ci_lo": lo_o, "ci_hi": hi_o, "max_dd": max_dd_o, "mean": overlay_net.mean()},
        "per_fold_overlay": per_fold_o,
    }


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    # Train each target separately
    print(f"\n=== Train per-target ensembles ===", flush=True)
    preds_by_target = {}  # target_X -> dict[fid] = (test_df, preds)
    for target_col in ["target_A", "target_D"]:
        print(f"\n  --- {target_col} ---", flush=True)
        per_fold = {}
        for fid in ALL_FOLDS:
            if fid >= len(folds_all): continue
            t0 = time.time()
            td, p = train_fold_on_target(panel, folds_all[fid], feat_set, target_col)
            if td is None: continue
            per_fold[fid] = (td, p)
            print(f"    fold {fid}: ({time.time()-t0:.0f}s)", flush=True)
        preds_by_target[target_col] = per_fold

    # Build prediction DataFrames per target
    def build_pred_df(per_fold_data):
        rows = []
        for fid, (td, p) in per_fold_data.items():
            df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            df["pred"] = p
            df["fold"] = fid
            rows.append(df)
        return pd.concat(rows, ignore_index=True)

    pred_A = build_pred_df(preds_by_target["target_A"])
    pred_D = build_pred_df(preds_by_target["target_D"])

    # Average: align by (symbol, open_time)
    print(f"\n  Building ensemble (A + D average)...", flush=True)
    df_merge = pred_A.merge(pred_D[["symbol", "open_time", "pred"]],
                              on=["symbol", "open_time"], suffixes=("_A", "_D"))
    df_merge["pred"] = (df_merge["pred_A"] + df_merge["pred_D"]) / 2
    df_merge = df_merge.drop(columns=["pred_A", "pred_D"])

    # Run all 3 variants
    print(f"\n=== Variant comparison ===", flush=True)
    variants = [
        ("target_A_only", pred_A),
        ("target_D_only", pred_D),
        ("target_AD_avg", df_merge),
    ]
    results = []
    print(f"  {'variant':<20}  {'overlay':<14}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'mean':>6}",
          flush=True)
    for label, pdf in variants:
        r = run_with_predictions(panel, pdf, label)
        results.append(r)
        d = r["no_overlay"]
        print(f"  {label:<20}  {'no_overlay':<14}  {d['sharpe']:>+7.2f}  "
              f"[{d['ci_lo']:>+5.2f},{d['ci_hi']:>+5.2f}]  {d['max_dd']:>+7.0f}  {d['mean']:>+6.2f}",
              flush=True)
        d = r["with_overlay"]
        print(f"  {label:<20}  {'+ Test C':<14}  {d['sharpe']:>+7.2f}  "
              f"[{d['ci_lo']:>+5.2f},{d['ci_hi']:>+5.2f}]  {d['max_dd']:>+7.0f}  {d['mean']:>+6.2f}",
              flush=True)

    print(f"\n  Per-fold (with overlay):", flush=True)
    print(f"  {'variant':<20}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
    for r in results:
        cells = " ".join(f"{r['per_fold_overlay'].get(f, 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['label']:<20}  " + cells, flush=True)

    # Save
    summary = []
    for r in results:
        for ovl in ["no_overlay", "with_overlay"]:
            d = r[ovl]
            summary.append({"variant": r["label"], "overlay": ovl, **d})
    pd.DataFrame(summary).to_csv(OUT_DIR / "target_ensemble_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
