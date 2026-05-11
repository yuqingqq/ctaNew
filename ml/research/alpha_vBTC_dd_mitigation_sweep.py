"""DD mitigation sweep on the validated production stack (PIT 60d + flat_real).

Tests multiple overlay variants to find a Sharpe-preserving / DD-reducing
combination. Post-process on per-cycle PnL of the production stack — fast.

Variants:
  baseline                   : no overlay
  dd>10%_s0.5                : tighter trigger
  dd>15%_s0.4                : middle
  dd>20%_s0.3 (current)      : current production
  dd>25%_s0.3                : looser trigger
  dd_tier_simple             : 10%→0.7 / 20%→0.4 / 30%→0.2
  dd_tier_aggressive         : 10%→0.6 / 20%→0.3 / 30%→0.1
  dd>20%_s0.3 + hysteresis5% : require recovery to peak−5% before re-lever
  dd>20%_s0.3 + trail_vol    : combined with vol-targeting
  vol_target_only            : scale by target_vol / trailing_std
  const_leverage_50pct       : just scale all positions by 0.5
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
OUT_DIR = REPO / "outputs/vBTC_dd_mitigation_sweep"
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
PROD_FOLDS = [5, 6, 7, 8, 9]
MIN_HISTORY_DAYS = 60

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


def get_listing_dates_from_klines():
    listings = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        sym = sym_dir.name
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        first_date = files[0].stem
        try:
            ts = pd.Timestamp(first_date, tz="UTC")
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


def build_rolling_ic_universe_pit(all_pred_df, target_times, ic_window_days, update_days,
                                      eligibility_at_t):
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
    return {t: boundary_to_universe[b] for t, b in boundaries}


# ===== OVERLAY VARIANTS =====

def overlay_dd_binary(net, threshold_dd, size_drawdown):
    n = len(net); sizes = np.ones(n)
    cum = np.cumsum(net); peak = -np.inf
    for i in range(n):
        peak = max(peak, cum[i] if i > 0 else 0)
        if peak > 0:
            dd_pct = (peak - cum[i]) / peak
            sizes[i] = size_drawdown if dd_pct > threshold_dd else 1.0
    return sizes * net, sizes


def overlay_dd_tier(net, tiers):
    """tiers = [(threshold_dd, size_at_dd)] sorted ascending threshold."""
    n = len(net); sizes = np.ones(n)
    cum = np.cumsum(net); peak = -np.inf
    for i in range(n):
        peak = max(peak, cum[i] if i > 0 else 0)
        s = 1.0
        if peak > 0:
            dd_pct = (peak - cum[i]) / peak
            for thr, size in tiers:
                if dd_pct > thr:
                    s = size
        sizes[i] = s
    return sizes * net, sizes


def overlay_dd_hysteresis(net, threshold_dd, size_drawdown, hyst_pct):
    """Trigger at dd>threshold; only release back when cum recovers to peak * (1 - hyst_pct)."""
    n = len(net); sizes = np.ones(n)
    cum = np.cumsum(net); peak = -np.inf
    in_dd = False
    for i in range(n):
        peak = max(peak, cum[i] if i > 0 else 0)
        if peak > 0:
            dd_pct = (peak - cum[i]) / peak
            if not in_dd and dd_pct > threshold_dd:
                in_dd = True
            elif in_dd and dd_pct < hyst_pct:
                in_dd = False
        sizes[i] = size_drawdown if in_dd else 1.0
    return sizes * net, sizes


def overlay_vol_target(net, target_std, window=30):
    """Scale by target_std / trailing window std."""
    n = len(net); sizes = np.ones(n)
    for i in range(n):
        if i >= window:
            past = net[max(0, i-window):i]
            if past.std() > 0:
                sizes[i] = min(target_std / past.std(), 1.5)  # cap at 1.5
    return sizes * net, sizes


def main():
    print(f"Loading panel + listing dates...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)
    listings = get_listing_dates_from_klines()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print(f"\n=== Train production stack with PIT 60d ===", flush=True)
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
    rolling_universe = build_rolling_ic_universe_pit(apd, oos_times_sampled,
                                                        IC_WINDOW_DAYS, IC_UPDATE_DAYS,
                                                        eligibility_at)
    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()
    df_v = evaluate_flat_real(test_data, rolling_universe)
    df_v["time"] = pd.to_datetime(df_v["time"])
    for fid in OOS_FOLDS:
        fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
        df_v.loc[df_v["time"].isin(fold_t), "fold"] = fid
    base_net = df_v["net_bps"].to_numpy()
    base_std = base_net.std()
    print(f"\n  Base PnL: n={len(base_net)}, mean={base_net.mean():+.2f}, "
          f"std={base_std:.1f}, base_Sharpe={_sharpe(base_net):+.2f}", flush=True)

    # Variants
    variants = [
        ("baseline_no_overlay",       lambda x: (x.copy(), np.ones_like(x))),
        ("dd>10%_s0.5",               lambda x: overlay_dd_binary(x, 0.10, 0.5)),
        ("dd>15%_s0.4",               lambda x: overlay_dd_binary(x, 0.15, 0.4)),
        ("dd>20%_s0.3 (current)",     lambda x: overlay_dd_binary(x, 0.20, 0.3)),
        ("dd>25%_s0.3",               lambda x: overlay_dd_binary(x, 0.25, 0.3)),
        ("dd_tier_simple",            lambda x: overlay_dd_tier(x, [(0.10, 0.7), (0.20, 0.4), (0.30, 0.2)])),
        ("dd_tier_aggressive",        lambda x: overlay_dd_tier(x, [(0.10, 0.6), (0.20, 0.3), (0.30, 0.1)])),
        ("dd_tier_smooth",            lambda x: overlay_dd_tier(x, [(0.05, 0.85), (0.10, 0.7), (0.15, 0.55), (0.20, 0.4), (0.25, 0.3)])),
        ("dd>20%_hyst5%",             lambda x: overlay_dd_hysteresis(x, 0.20, 0.3, 0.05)),
        ("dd>20%_hyst10%",            lambda x: overlay_dd_hysteresis(x, 0.20, 0.3, 0.10)),
        ("vol_target_75bps",          lambda x: overlay_vol_target(x, 75.0, window=30)),
        ("vol_target_100bps",         lambda x: overlay_vol_target(x, 100.0, window=30)),
        ("const_50pct",               lambda x: (x * 0.5, np.full_like(x, 0.5))),
    ]

    print(f"\n=== DD MITIGATION SWEEP ===", flush=True)
    print(f"  {'variant':<28}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'std':>6}  "
          f"{'mean':>6}  {'mean_size':>9}", flush=True)
    results = []
    for label, fn in variants:
        scaled, sizes = fn(base_net)
        sh, lo, hi = block_bootstrap_ci(scaled, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(scaled)
        per_fold = {}
        df_temp = df_v.copy(); df_temp["scaled"] = scaled
        for fid in OOS_FOLDS:
            fdat = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
            if len(fdat) >= 3:
                per_fold[fid] = _sharpe(fdat)
        results.append({"variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "max_dd": max_dd, "std": scaled.std(), "mean": scaled.mean(),
                          "mean_size": sizes.mean(),
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        print(f"  {label:<28}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{max_dd:>+7.0f}  {scaled.std():>6.1f}  {scaled.mean():>+6.2f}  "
              f"{sizes.mean():>9.2f}", flush=True)

    print(f"\n  Per-fold (with overlay):", flush=True)
    print(f"  {'variant':<28}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<28}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "dd_mitigation_sweep.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
