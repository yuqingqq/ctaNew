"""Final production-stack simulation + integrity audit + real PnL growth.

Production stack:
  - WINNER_21 features, LGBM 5-seed ensemble
  - Expanding-window training, per-fold (every 30 days)
  - PIT eligibility: listing_date + 60d ≤ T
  - Rolling-IC universe: 180d lookback × 90d refresh, top-15
  - K=4 longs / K=4 shorts
  - conv_gate: skip cycle if dispersion < 30th-pctile (252-cycle history)
  - PM_M2_b1 persistence gate
  - flat_real skip mode (close on gate, re-open on clear with 2-leg cost)
  - dd_tier_aggressive overlay: dd>10%→0.6, dd>20%→0.3, dd>30%→0.1

Outputs:
  1. Pipeline integrity audits (fold time separation, no-leak universe, embargo)
  2. Per-cycle PnL CSV (with overlay)
  3. Cumulative growth curve numbers (by week/month)
  4. Per-fold PnL/Sharpe/maxDD
  5. Drawdown episodes (full list)
  6. Realized return distribution
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
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_final_simulation"
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
MIN_HISTORY_DAYS = 60
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


def _all_dd_episodes(net):
    """Return list of (start_idx, trough_idx, end_idx, magnitude_bps)."""
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd_series = cum - peak
    episodes = []
    in_ep = False
    start = trough = 0
    trough_val = 0
    for i in range(len(net)):
        if dd_series[i] < 0:
            if not in_ep:
                start = i; trough = i; trough_val = dd_series[i]
                in_ep = True
            elif dd_series[i] < trough_val:
                trough = i; trough_val = dd_series[i]
        else:
            if in_ep:
                episodes.append((start, trough, i - 1, trough_val))
                in_ep = False
    if in_ep:
        episodes.append((start, trough, len(net) - 1, trough_val))
    return episodes


def get_listing_dates_from_klines():
    listings = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        sym = sym_dir.name
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            ts = pd.Timestamp(files[0].stem, tz="UTC")
            listings[sym] = ts
        except Exception:
            continue
    return listings


def train_fold_restricted(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) &
                 (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) &
                (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


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
            bars.append({"time": t, "spread_bps": 0.0, "cost_bps": 0.0,
                          "net_raw_bps": 0.0, "skipped": 1,
                          "n_eligible": len(g_u), "long_syms": "", "short_syms": ""})
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
                bars.append({"time": t, "spread_bps": 0.0, "cost_bps": 2 * COST_PER_LEG,
                              "net_raw_bps": -2 * COST_PER_LEG, "skipped": 1,
                              "n_eligible": len(g_u), "long_syms": "", "short_syms": ""})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                bars.append({"time": t, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_raw_bps": 0.0, "skipped": 1,
                              "n_eligible": len(g_u), "long_syms": "", "short_syms": ""})
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
            bars.append({"time": t, "spread_bps": 0.0, "cost_bps": 0.0,
                          "net_raw_bps": 0.0, "skipped": 0,
                          "n_eligible": len(g_u), "long_syms": "", "short_syms": ""})
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
        bars.append({"time": t, "spread_bps": spread, "cost_bps": cost,
                      "net_raw_bps": net, "skipped": 0,
                      "n_eligible": len(g_u),
                      "long_syms": ",".join(sorted(new_long)),
                      "short_syms": ",".join(sorted(new_short))})
        cur_long, cur_short = new_long, new_short
    return pd.DataFrame(bars)


def build_rolling_ic_universe_pit(all_pred_df, target_times, ic_window_days, update_days,
                                      eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms
    all_pred_clean = all_pred_df.dropna(subset=["alpha_A"])
    if not target_times: return {}, {}
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
        eligible = eligibility_at_t(b)
        past = all_pred_clean[(all_pred_clean["t_int"] >= b - window_ms) &
                                (all_pred_clean["t_int"] < b) &
                                (all_pred_clean["symbol"].isin(eligible))]
        if len(past) < 1000:
            boundary_to_universe[b] = set()
            continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        boundary_to_universe[b] = set(ics_sorted.head(TARGET_N).index.tolist())
    return {t: boundary_to_universe[b] for t, b in boundaries}, boundary_to_universe


def apply_dd_tier_aggressive(net):
    """dd>10%→0.6, dd>20%→0.3, dd>30%→0.1"""
    n = len(net); sizes = np.ones(n)
    cum = np.cumsum(net); peak = -np.inf
    for i in range(n):
        peak = max(peak, cum[i] if i > 0 else 0)
        if peak > 0:
            dd_pct = (peak - cum[i]) / peak
            if dd_pct > 0.30: sizes[i] = 0.1
            elif dd_pct > 0.20: sizes[i] = 0.3
            elif dd_pct > 0.10: sizes[i] = 0.6
            else: sizes[i] = 1.0
        else:
            sizes[i] = 1.0
    return sizes * net, sizes


def main():
    print(f"\n{'=' * 90}", flush=True)
    print(f"  vBTC PRODUCTION STACK — END-TO-END SIMULATION + AUDIT", flush=True)
    print(f"{'=' * 90}\n", flush=True)

    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)
    listings = get_listing_dates_from_klines()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    print(f"Panel: {panel['symbol'].nunique()} symbols, {len(panel):,} rows, "
          f"period {panel['open_time'].min()} to {panel['open_time'].max()}", flush=True)
    print(f"Features: {len(feat_set)}", flush=True)
    print(f"Folds: {len(folds_all)} (30-day OOS each, 2-day embargo)", flush=True)
    print(f"Seeds per fold: {len(SEEDS)}", flush=True)

    # === AUDIT 1: fold time integrity ===
    print(f"\n=== AUDIT 1: Fold time-windows ===", flush=True)
    print(f"  {'fid':>3}  {'train_end':<12}  {'cal_window':<24}  {'test_window':<24}  {'embargo':>8}",
          flush=True)
    last_test_end = None
    overlap_detected = False
    for f in folds_all:
        cal_win = f"{f['cal_start'].strftime('%Y-%m-%d')}→{f['cal_end'].strftime('%Y-%m-%d')}"
        test_win = f"{f['test_start'].strftime('%Y-%m-%d')}→{f['test_end'].strftime('%Y-%m-%d')}"
        print(f"  {f['fid']:>3}  {f['train_end'].strftime('%Y-%m-%d'):<12}  {cal_win:<24}  {test_win:<24}  "
              f"{f['embargo'].days:>5}d", flush=True)
        if last_test_end is not None and f['test_start'] < last_test_end:
            overlap_detected = True
        last_test_end = f['test_end']
    print(f"  Overlap detected: {'YES — LEAK!' if overlap_detected else 'no'}", flush=True)

    panel_syms = set(panel["symbol"].unique())
    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    # === Train ===
    print(f"\n=== TRAINING (10 folds × 5 seeds, expanding window, PIT eligibility) ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold_restricted(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: train_eligible={len(eligible)}, n_test={len(td):,} ({time.time()-t0:.0f}s)",
              flush=True)
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
    rolling_universe, b2u = build_rolling_ic_universe_pit(apd, oos_times_sampled,
                                                              IC_WINDOW_DAYS, IC_UPDATE_DAYS,
                                                              eligibility_at)

    # === AUDIT 2: universe rebalances ===
    print(f"\n=== AUDIT 2: Rolling-IC universe boundaries ===", flush=True)
    bar_ms = 5 * 60 * 1000
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    for b in sorted(b2u.keys()):
        u = b2u[b]
        b_ts = pd.Timestamp(b, unit='ms')
        lookback_start = pd.Timestamp(b - window_ms, unit='ms')
        print(f"  {b_ts.strftime('%Y-%m-%d')}: IC window [{lookback_start.strftime('%Y-%m-%d')}, "
              f"boundary), |U|={len(u)}", flush=True)
        print(f"    {sorted(u)}", flush=True)

    # === Evaluate ===
    print(f"\n=== EVALUATION (flat_real + dd_tier_aggressive) ===", flush=True)
    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()
    df_v = evaluate_flat_real(test_data, rolling_universe)
    df_v["time"] = pd.to_datetime(df_v["time"])
    for fid in OOS_FOLDS:
        fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
        df_v.loc[df_v["time"].isin(fold_t), "fold"] = fid

    net_raw = df_v["net_raw_bps"].to_numpy()
    net_with_overlay, sizes = apply_dd_tier_aggressive(net_raw)
    df_v["net_with_overlay_bps"] = net_with_overlay
    df_v["size_multiplier"] = sizes
    df_v["cum_pnl_raw"] = np.cumsum(net_raw)
    df_v["cum_pnl_overlay"] = np.cumsum(net_with_overlay)

    # === Headline numbers ===
    print(f"\n{'=' * 90}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'=' * 90}\n", flush=True)

    print(f"WALK-FORWARD (9 OOS folds, ~9 months 2025-07-19 to 2026-04-30):", flush=True)
    for label, arr in [("Without overlay", net_raw), ("With dd_tier_aggressive", net_with_overlay)]:
        sh, lo, hi = block_bootstrap_ci(arr, statistic=_sharpe, block_size=7, n_boot=2000)
        print(f"\n  {label}:", flush=True)
        print(f"    n cycles            : {len(arr)}", flush=True)
        print(f"    mean per cycle      : {arr.mean():+.2f} bps", flush=True)
        print(f"    std per cycle       : {arr.std():.1f} bps", flush=True)
        print(f"    annualized Sharpe   : {sh:+.2f}", flush=True)
        print(f"    Sharpe 95% CI       : [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"    max drawdown        : {_max_dd(arr):+.0f} bps", flush=True)
        print(f"    total period PnL    : {arr.sum():+.0f} bps", flush=True)
        cum = np.cumsum(arr)
        print(f"    ending cum PnL      : {cum[-1]:+.0f} bps", flush=True)
        # Annualized return
        cycles_per_year = (288 * 365) / HORIZON
        annual_ret_pct = arr.mean() * cycles_per_year / 100  # bps to %
        print(f"    annualized return   : {annual_ret_pct:+.1f}% (at 1× leverage)", flush=True)

    print(f"\n=== PER-FOLD BREAKDOWN (with overlay) ===", flush=True)
    print(f"  {'fold':>5}  {'date_range':<25}  {'n':>4}  {'mean':>6}  {'std':>6}  "
          f"{'Sharpe':>7}  {'maxDD':>7}  {'cum_PnL':>9}", flush=True)
    for fid in OOS_FOLDS:
        fdat = df_v[df_v["fold"] == fid]
        if fdat.empty: continue
        arr = fdat["net_with_overlay_bps"].to_numpy()
        date_range = f"{fdat['time'].min().strftime('%Y-%m-%d')} to {fdat['time'].max().strftime('%m-%d')}"
        print(f"  {int(fid):>5}  {date_range:<25}  {len(arr):>4}  {arr.mean():>+6.2f}  "
              f"{arr.std():>6.1f}  {_sharpe(arr):>+7.2f}  {_max_dd(arr):>+7.0f}  "
              f"{arr.sum():>+9.0f}", flush=True)

    # Monthly breakdown
    print(f"\n=== MONTHLY PnL GROWTH ===", flush=True)
    df_v["month"] = df_v["time"].dt.to_period("M")
    monthly = df_v.groupby("month").agg(
        n=("net_with_overlay_bps", "size"),
        mean=("net_with_overlay_bps", "mean"),
        sum=("net_with_overlay_bps", "sum"),
        std=("net_with_overlay_bps", "std"),
        avg_size=("size_multiplier", "mean"),
    )
    monthly["cum_pnl"] = monthly["sum"].cumsum()
    monthly["sharpe"] = monthly.apply(lambda r: r["mean"] / r["std"] * np.sqrt(CYCLES_PER_YEAR) if r["std"] > 0 else 0,
                                        axis=1)
    print(f"  {'month':<8}  {'cycles':>6}  {'mean_bps':>8}  {'sum_bps':>8}  {'Sharpe':>7}  "
          f"{'avg_size':>9}  {'cum_pnl':>9}", flush=True)
    for m, row in monthly.iterrows():
        print(f"  {str(m):<8}  {int(row['n']):>6}  {row['mean']:>+8.2f}  {row['sum']:>+8.0f}  "
              f"{row['sharpe']:>+7.2f}  {row['avg_size']:>9.2f}  {row['cum_pnl']:>+9.0f}", flush=True)

    # Drawdown episodes
    print(f"\n=== ALL DRAWDOWN EPISODES (with overlay) ===", flush=True)
    eps = _all_dd_episodes(net_with_overlay)
    eps.sort(key=lambda e: e[3])  # sort by trough magnitude ascending
    print(f"  Total episodes: {len(eps)}. Top-10 by magnitude:", flush=True)
    print(f"  {'rank':>4}  {'start':<11}  {'trough':<11}  {'end':<11}  {'duration':>9}  {'magnitude':>10}",
          flush=True)
    for rank, (s, tr, e, mag) in enumerate(eps[:10], 1):
        s_t = df_v.iloc[s]["time"].strftime('%Y-%m-%d')
        tr_t = df_v.iloc[tr]["time"].strftime('%Y-%m-%d')
        e_t = df_v.iloc[e]["time"].strftime('%Y-%m-%d')
        dur = e - s + 1
        print(f"  {rank:>4}  {s_t:<11}  {tr_t:<11}  {e_t:<11}  {dur:>6}cyc  {mag:>+10.0f}",
              flush=True)

    # Return distribution
    print(f"\n=== RETURN DISTRIBUTION (per-cycle bps, with overlay) ===", flush=True)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:>2}: {np.percentile(net_with_overlay, p):+8.1f}", flush=True)

    # Save outputs
    df_v.to_csv(OUT_DIR / "per_cycle_pnl.csv", index=False)
    monthly.to_csv(OUT_DIR / "monthly_growth.csv")
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
