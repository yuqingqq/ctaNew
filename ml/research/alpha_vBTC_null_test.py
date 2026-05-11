"""Random-target null test: gold-standard leakage detection.

Procedure:
  1. Shuffle target_A WITHIN each open_time (preserves cross-sectional
     distribution at each time but breaks the features→target relationship)
  2. Re-train all 10 folds with shuffled target
  3. Generate predictions
  4. Run through SAME production evaluator + DD overlay
  5. Report Sharpe — should be ≈ 0 if pipeline is leak-free

Multiple random shuffles for robustness. Reports distribution of null-Sharpe.

Compares to real-target Sharpe from the production run (+4.46).
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
OUT_DIR = REPO / "outputs/vBTC_null_test"
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

NULL_SEEDS = [12345, 99999]  # 2 different shuffles (real reference cached)

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
        try:
            ts = pd.Timestamp(files[0].stem, tz="UTC")
            listings[sym] = ts
        except Exception:
            continue
    return listings


def shuffle_target_within_time(panel, target_col, seed):
    """Shuffle target_col values within each open_time group.

    Preserves cross-sectional distribution per time; breaks features→target.
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    targets = out[target_col].to_numpy().copy()
    # Group indices by open_time
    out_sorted = out.sort_values("open_time").reset_index(drop=False)
    times = out_sorted["open_time"].to_numpy()
    orig_indices = out_sorted["index"].to_numpy()
    target_sorted = out_sorted[target_col].to_numpy().copy()  # writable copy

    # Find group boundaries
    n = len(times)
    i = 0
    while i < n:
        j = i
        while j < n and times[j] == times[i]:
            j += 1
        # Shuffle target values in [i, j)
        idx = np.arange(i, j)
        rng.shuffle(idx)
        target_sorted[i:j] = target_sorted[idx]
        i = j

    # Map back to original row order
    targets_new = np.empty_like(targets)
    targets_new[orig_indices] = target_sorted
    out[target_col] = targets_new
    return out


def train_fold_restricted(panel, fold, feat_set, eligible_syms, target_col="target_A"):
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
    yt = tr[target_col].to_numpy(np.float32)
    yc = ca[target_col].to_numpy(np.float32)
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


def apply_dd_tier_aggressive(net):
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


def run_pipeline_with_target(panel_input, listings, label):
    """Run full pipeline; returns (net_no_overlay, net_with_overlay, sharpe_overlay)."""
    feat_set = [f for f in WINNER_21 if f in panel_input.columns]
    folds_all = _multi_oos_splits(panel_input)
    panel_syms = set(panel_input["symbol"].unique())

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold_restricted(panel_input, folds_all[fid], feat_set, eligible)
        if td is None: continue
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        all_preds.append(df)
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
    net_raw = df_v["net_bps"].to_numpy()
    net_overlay, _ = apply_dd_tier_aggressive(net_raw)
    sh_raw = _sharpe(net_raw)
    sh_overlay = _sharpe(net_overlay)
    return net_raw, net_overlay, sh_raw, sh_overlay


def main():
    print(f"\n{'=' * 90}", flush=True)
    print(f"  RANDOM-TARGET NULL TEST", flush=True)
    print(f"{'=' * 90}\n", flush=True)
    print(f"Method: shuffle target_A within each open_time → re-train → re-evaluate.", flush=True)
    print(f"Expected: if pipeline is leak-free, Sharpe should drop to ≈ 0.", flush=True)
    print(f"Production reference (real target): Sharpe +4.46 (with overlay).", flush=True)
    print(f"\nUsing {len(NULL_SEEDS)} different shuffle seeds for robustness.\n", flush=True)

    panel = pd.read_parquet(PANEL_PATH)
    listings = get_listing_dates_from_klines()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    # Skip real-target rerun; reference Sharpe is cached as +4.46 (no_overlay +1.56)
    sh_real_raw = 1.56
    sh_real_overlay = 4.46
    print(f"=== Real target (cached reference) ===", flush=True)
    print(f"  Sharpe (no overlay): {sh_real_raw:+.2f}", flush=True)
    print(f"  Sharpe (+ overlay):  {sh_real_overlay:+.2f}", flush=True)

    null_results = []
    for null_seed in NULL_SEEDS:
        print(f"\n=== Null shuffle (seed={null_seed}) ===", flush=True)
        t0 = time.time()
        panel_null = shuffle_target_within_time(panel, "target_A", null_seed)
        # Sanity check: total target_A sum should be unchanged (just permuted)
        sum_real = panel["target_A"].sum()
        sum_null = panel_null["target_A"].sum()
        print(f"  shuffle sanity: sum_real={sum_real:.4f} sum_null={sum_null:.4f} "
              f"(should match)", flush=True)
        net_null_raw, net_null_overlay, sh_null_raw, sh_null_overlay = run_pipeline_with_target(
            panel_null, listings, f"null_{null_seed}")
        null_results.append({
            "seed": null_seed,
            "sharpe_no_overlay": sh_null_raw,
            "sharpe_with_overlay": sh_null_overlay,
            "max_dd_overlay": _max_dd(net_null_overlay),
            "mean_net": net_null_raw.mean(),
        })
        print(f"  Sharpe (no overlay): {sh_null_raw:+.2f}", flush=True)
        print(f"  Sharpe (+ overlay):  {sh_null_overlay:+.2f}", flush=True)
        print(f"  Max DD (overlay):    {_max_dd(net_null_overlay):+.0f} bps", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'=' * 90}", flush=True)
    print(f"  NULL TEST SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"\n  Real target:  Sharpe (no overlay) = {sh_real_raw:+.2f}  "
          f"(+ overlay) = {sh_real_overlay:+.2f}", flush=True)
    print(f"\n  Null shuffles:", flush=True)
    print(f"  {'seed':>6}  {'Sh_no_ovl':>10}  {'Sh_with_ovl':>11}  {'max_DD':>7}",
          flush=True)
    for r in null_results:
        print(f"  {r['seed']:>6}  {r['sharpe_no_overlay']:>+10.2f}  "
              f"{r['sharpe_with_overlay']:>+11.2f}  {r['max_dd_overlay']:>+7.0f}",
              flush=True)
    null_sh_raw = [r["sharpe_no_overlay"] for r in null_results]
    null_sh_overlay = [r["sharpe_with_overlay"] for r in null_results]
    print(f"\n  Null mean (no_overlay):     {np.mean(null_sh_raw):+.2f}", flush=True)
    print(f"  Null std (no_overlay):      {np.std(null_sh_raw):.2f}", flush=True)
    print(f"  Null mean (with_overlay):   {np.mean(null_sh_overlay):+.2f}", flush=True)
    print(f"  Null std (with_overlay):    {np.std(null_sh_overlay):.2f}", flush=True)

    print(f"\n  Verdict:", flush=True)
    if abs(np.mean(null_sh_overlay)) < 1.0:
        print(f"  ✓ Null Sharpe ≈ 0 (mean {np.mean(null_sh_overlay):+.2f}). "
              f"No detectable leakage.", flush=True)
        print(f"    Real Sharpe {sh_real_overlay:+.2f} reflects genuine alpha.", flush=True)
    else:
        print(f"  ⚠ Null Sharpe is non-trivially nonzero ({np.mean(null_sh_overlay):+.2f}).", flush=True)
        print(f"    Possible explanations:", flush=True)
        print(f"    1. Conv_gate / overlay extracts signal even from random preds (path-dependent)", flush=True)
        print(f"    2. Hidden leakage in features", flush=True)
        print(f"    3. Strategy mechanics exploit cycle/temporal structure", flush=True)

    pd.DataFrame(null_results).to_csv(OUT_DIR / "null_test_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
