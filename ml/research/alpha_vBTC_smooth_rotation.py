"""Smooth-rotation universe test.

Replaces the 90-day discrete cliff with weekly evaluation + at most 1 name
swap per week. Goal: avoid discrete refresh shocks; deployment-time-independent.

Variants:
  baseline_90d_cliff   : current production (rolling-IC 180/90)
  smooth_swap1_week    : evaluate weekly, swap up to 1 name
  smooth_swap1_2week   : evaluate biweekly, swap up to 1 name
  smooth_swap2_week    : evaluate weekly, swap up to 2 names

Eval base: expanding train, K=4, N=15, WINNER_21, flat_real, Test C overlay.
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
OUT_DIR = REPO / "outputs/vBTC_smooth_rotation"
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


def compute_ic_at_boundary(all_pred_clean, boundary_ms, ic_window_ms):
    """Return Series of IC per symbol, sorted descending."""
    past = all_pred_clean[(all_pred_clean["t_int"] >= boundary_ms - ic_window_ms) &
                            (all_pred_clean["t_int"] < boundary_ms)]
    if len(past) < 1000:
        return pd.Series(dtype=float)
    ics = past.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    )
    return ics.dropna().sort_values(ascending=False)


def build_cliff_universe(all_pred_clean, target_times, ic_window_days, update_days):
    """Original 90d cliff universe."""
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms
    if not target_times: return {}
    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)
    boundaries = []
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b_ms = t0_ms + n * update_ms
        boundaries.append((t, b_ms))
    unique_b = sorted(set(b for _, b in boundaries))
    boundary_to_universe = {}
    for b in unique_b:
        ics = compute_ic_at_boundary(all_pred_clean, b, window_ms)
        boundary_to_universe[b] = set(ics.head(TARGET_N).index.tolist()) if not ics.empty else set()
    return {t: boundary_to_universe[b] for t, b in boundaries}


def build_smooth_universe(all_pred_clean, target_times, ic_window_days,
                            check_interval_days=7, max_swaps_per_check=1):
    """Smooth-rotation universe.

    Every check_interval_days, recompute IC ranking, allow up to N swaps to
    move toward the new top-15. Initial universe = standard top-15 at first
    boundary. From then on, each check evaluates: which currently-in-universe
    name has the lowest current IC rank? If it's > N+5 (clearly fallen out),
    swap with the highest-IC not-in-universe (if rank <= N).
    """
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    check_ms = check_interval_days * 288 * bar_ms
    if not target_times: return {}
    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)

    # Generate check boundaries from t0 to last target time
    last_t_ms = int(pd.Timestamp(target_times[-1]).timestamp() * 1000)
    checks = []
    b = t0_ms
    while b <= last_t_ms:
        checks.append(b)
        b += check_ms

    # Initial universe at first check
    ics0 = compute_ic_at_boundary(all_pred_clean, checks[0], window_ms)
    if ics0.empty:
        return {}
    cur_universe = set(ics0.head(TARGET_N).index.tolist())
    universe_at_check = {checks[0]: set(cur_universe)}

    # Iterate checks
    for b in checks[1:]:
        ics = compute_ic_at_boundary(all_pred_clean, b, window_ms)
        if ics.empty:
            universe_at_check[b] = set(cur_universe)
            continue
        # Build full ranking
        ranking = list(ics.index)  # descending IC
        rank_of = {s: i + 1 for i, s in enumerate(ranking)}

        for _ in range(max_swaps_per_check):
            # Candidates to swap OUT: in universe but rank > TARGET_N (i.e., fell out)
            in_u_out_of_top = [s for s in cur_universe
                                if rank_of.get(s, len(ranking) + 1) > TARGET_N]
            if not in_u_out_of_top:
                break
            # Worst rank in universe (highest rank number = lowest IC)
            worst_in = max(in_u_out_of_top, key=lambda s: rank_of[s])

            # Candidates to swap IN: not in universe and rank <= TARGET_N
            top_n_candidates = ranking[:TARGET_N]
            best_out_of_u = next((s for s in top_n_candidates if s not in cur_universe), None)
            if best_out_of_u is None:
                break

            # Swap
            cur_universe.discard(worst_in)
            cur_universe.add(best_out_of_u)

        universe_at_check[b] = set(cur_universe)

    # Map each target_time to the most recent check's universe
    sorted_checks = sorted(universe_at_check.keys())
    out = {}
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        # Find latest check <= t
        idx = np.searchsorted(sorted_checks, t_ms, side="right") - 1
        if idx < 0: idx = 0
        out[t] = universe_at_check[sorted_checks[idx]]
    return out


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
        td, p = train_fold(panel, folds_all[fid], feat_set)
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
    apd_clean = apd.dropna(subset=["alpha_A"])

    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    oos_times_all = sorted(oos_pred["open_time"].unique())
    oos_times_sampled = oos_times_all[::HORIZON]
    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    variants = [
        ("baseline_90d_cliff",     "cliff", {"ic_window_days": 180, "update_days": 90}),
        ("smooth_swap1_per_week",  "smooth", {"ic_window_days": 180, "check_interval_days": 7, "max_swaps_per_check": 1}),
        ("smooth_swap1_per_2week", "smooth", {"ic_window_days": 180, "check_interval_days": 14, "max_swaps_per_check": 1}),
        ("smooth_swap2_per_week",  "smooth", {"ic_window_days": 180, "check_interval_days": 7, "max_swaps_per_check": 2}),
    ]

    print(f"\n=== Variant comparison ===", flush=True)
    print(f"  {'variant':<28}  {'overlay':<14}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  "
          f"{'mean':>6}  {'turnover':>9}", flush=True)
    results = []
    for label, kind, params in variants:
        if kind == "cliff":
            u_per_t = build_cliff_universe(apd_clean, oos_times_sampled, **params)
        else:
            u_per_t = build_smooth_universe(apd_clean, oos_times_sampled, **params)
        # Compute average per-cycle universe turnover (Jaccard distance from prior)
        sorted_times = sorted(u_per_t.keys())
        jacs = []
        for i in range(1, len(sorted_times)):
            u_prev = u_per_t[sorted_times[i-1]]
            u_cur = u_per_t[sorted_times[i]]
            if u_prev and u_cur:
                jacs.append(1 - len(u_prev & u_cur) / len(u_prev | u_cur))
        avg_turnover = np.mean(jacs) if jacs else 0.0

        df_v = evaluate_flat_real(test_data, u_per_t)
        df_v["time"] = pd.to_datetime(df_v["time"])
        for fid in OOS_FOLDS:
            fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
            df_v.loc[df_v["time"].isin(fold_t), "fold"] = fid

        for ovl_label, apply_ovl in [("none", False), ("dd>20%_size=0.3", True)]:
            net = df_v["net_bps"].to_numpy()
            if apply_ovl:
                net, _ = apply_dd_overlay(net)
            sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
            max_dd = _max_dd(net)
            per_fold = {}
            df_temp = df_v.copy(); df_temp["scaled"] = net
            for fid in OOS_FOLDS:
                fdat = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
                if len(fdat) >= 3:
                    per_fold[fid] = _sharpe(fdat)
            results.append({"variant": label, "overlay": ovl_label,
                              "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                              "max_dd": max_dd, "mean": net.mean(),
                              "avg_turnover": avg_turnover,
                              **{f"sh_f{f}": v for f, v in per_fold.items()}})
            print(f"  {label:<28}  {ovl_label:<14}  {sh:>+7.2f}  "
                  f"[{lo:>+5.2f},{hi:>+5.2f}]  {max_dd:>+7.0f}  "
                  f"{net.mean():>+6.2f}  {avg_turnover:>9.4f}", flush=True)

    print(f"\n  Per-fold Sharpe (with overlay):", flush=True)
    print(f"  {'variant':<28}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
    for r in results:
        if r["overlay"] != "dd>20%_size=0.3": continue
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<28}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "smooth_rotation_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
