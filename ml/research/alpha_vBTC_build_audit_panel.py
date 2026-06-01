"""Phase 0: build per-cycle × per-symbol OOS audit panel.

Re-runs the corrected pipeline (PIT timing fixes from alpha_vBTC_final_simulation.py)
but saves the granular auditable panel instead of just aggregate cycle PnL.

Columns saved (one row per (cycle_time, candidate_symbol)):
  time, fold, symbol, pred, rank_long, rank_short, return_pct, alpha_A,
  exit_time, in_universe, eligible_pit,
  long_contrib_bps_if_picked, short_contrib_bps_if_picked,
  picked_long, picked_short

This panel lets us:
  1. Compute trailing per-(symbol, side) metrics PIT (only rows with exit_time <= t)
  2. Test symbol-side filters/sizing in pure post-processing
  3. Independently verify the headline numbers from final_simulation.py

Saved to outputs/vBTC_audit_panel/audit_panel.parquet
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
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_audit_panel"
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


def get_listing_dates_from_klines():
    listings = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            listings[sym_dir.name] = pd.Timestamp(files[0].stem, tz="UTC")
        except Exception:
            continue
    return listings


def train_fold_restricted(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100:
        return None, None
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


def build_rolling_ic_universe_pit(all_pred_df, target_times, ic_window_days, update_days,
                                      eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms
    all_pred_clean = all_pred_df.dropna(subset=["alpha_A"]).copy()
    if "exit_time" not in all_pred_clean.columns:
        all_pred_clean["exit_time"] = all_pred_clean["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    exit_ts = all_pred_clean["exit_time"]
    if hasattr(exit_ts.dtype, "tz") and exit_ts.dtype.tz is not None:
        exit_ts_naive = exit_ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        exit_ts_naive = exit_ts
    all_pred_clean["exit_t_int"] = exit_ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()
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
                                (all_pred_clean["exit_t_int"] <= b) &
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


def run_base_evaluator_recording_picks(test_df, universe_per_t, top_k=K, sample_every=HORIZON):
    """Run base evaluator (flat_real + conv_gate) and record per-(cycle, symbol) picks."""
    df = test_df.copy()
    times = sorted(df["open_time"].unique())
    if not times: return [], []
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]
    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    is_flat = False
    picks = []
    meta = []
    for t, g in df.groupby("open_time"):
        u = universe_per_t.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * top_k + 1:
            meta.append({"time": t, "skipped": 1, "n_universe": len(g_u),
                          "spread_bps": 0.0, "cost_bps": 0.0})
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
                meta.append({"time": t, "skipped": 1, "n_universe": len(g_u),
                              "spread_bps": 0.0, "cost_bps": 2 * COST_PER_LEG})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                meta.append({"time": t, "skipped": 1, "n_universe": len(g_u),
                              "spread_bps": 0.0, "cost_bps": 0.0})
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
            meta.append({"time": t, "skipped": 0, "n_universe": len(g_u),
                          "spread_bps": 0.0, "cost_bps": 0.0})
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
        meta.append({"time": t, "skipped": 0, "n_universe": len(g_u),
                      "spread_bps": spread, "cost_bps": cost})
        for s in new_long:
            picks.append((t, s, 1, 0))
        for s in new_short:
            picks.append((t, s, 0, 1))
        cur_long, cur_short = new_long, new_short
    return picks, meta


def main():
    print(f"=== Phase 0: Build OOS audit panel ===", flush=True)
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

    print(f"\n--- Train all 10 folds with PIT 60d ---", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold_restricted(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols_to_save = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns:
            cols_to_save.append("exit_time")
        df = td[cols_to_save].copy()
        df["pred"] = p; df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
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

    print(f"\n--- Build PIT rolling-IC universe ---", flush=True)
    rolling_universe, _ = build_rolling_ic_universe_pit(apd, oos_times_sampled,
                                                            IC_WINDOW_DAYS, IC_UPDATE_DAYS,
                                                            eligibility_at)

    sampled_times = set(oos_times_sampled)
    oos_at_rebal = oos_pred[oos_pred["open_time"].isin(sampled_times)].copy()
    print(f"  OOS rebalance rows: {len(oos_at_rebal):,}", flush=True)

    print(f"\n--- Build audit panel ---", flush=True)
    audit_rows = []
    for t, g in oos_at_rebal.groupby("open_time"):
        u_t = rolling_universe.get(t, set())
        elig_t = eligibility_at(t)
        preds = g["pred"].to_numpy()
        # 1-indexed ranks
        rank_long = (-preds).argsort().argsort() + 1
        rank_short = preds.argsort().argsort() + 1
        for i, (_, row) in enumerate(g.iterrows()):
            sym = row["symbol"]
            r_pct = row["return_pct"]
            audit_rows.append({
                "time": t,
                "fold": int(row["fold"]),
                "symbol": sym,
                "pred": float(row["pred"]),
                "rank_long": int(rank_long[i]),
                "rank_short": int(rank_short[i]),
                "return_pct": float(r_pct),
                "alpha_A": float(row["alpha_A"]) if pd.notna(row["alpha_A"]) else np.nan,
                "exit_time": row["exit_time"],
                "in_universe": 1 if sym in u_t else 0,
                "eligible_pit": 1 if sym in elig_t else 0,
                "long_contrib_bps_if_picked": float(r_pct) * 1e4 / K,
                "short_contrib_bps_if_picked": -float(r_pct) * 1e4 / K,
            })
    audit_df = pd.DataFrame(audit_rows)
    print(f"  audit panel rows: {len(audit_df):,}", flush=True)

    print(f"\n--- Run base evaluator to record picked_long / picked_short ---", flush=True)
    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()
    picks, meta = run_base_evaluator_recording_picks(test_data, rolling_universe)
    pick_df = pd.DataFrame(picks, columns=["time", "symbol", "picked_long", "picked_short"])
    print(f"  base run picks: {len(pick_df):,}", flush=True)

    audit_df = audit_df.merge(pick_df, on=["time", "symbol"], how="left")
    audit_df["picked_long"] = audit_df["picked_long"].fillna(0).astype(int)
    audit_df["picked_short"] = audit_df["picked_short"].fillna(0).astype(int)

    # Per-cycle actual K_long / K_short (after PM persistence + universe).
    # These differ from the nominal K=4 because PM persistence + universe
    # constraints often leave fewer than K names per side.
    per_cycle = audit_df.groupby("time").agg(
        n_long_picked=("picked_long", "sum"),
        n_short_picked=("picked_short", "sum"),
    ).reset_index()
    audit_df = audit_df.merge(per_cycle, on="time", how="left")

    # Actual per-pick contribution in bps — uses ACTUAL per-cycle K, not nominal.
    # For picked rows: contribution = ret_pct × 1e4 / actual_K_side
    # For non-picked rows: hypothetical contribution at nominal K=4 (for filter
    # simulation that needs trailing per-(sym,side) metric of "what if we had picked")
    audit_df["long_contrib_bps_actual"] = np.where(
        (audit_df["picked_long"] == 1) & (audit_df["n_long_picked"] > 0),
        audit_df["return_pct"] * 1e4 / audit_df["n_long_picked"].replace(0, np.nan),
        audit_df["return_pct"] * 1e4 / K,
    )
    audit_df["short_contrib_bps_actual"] = np.where(
        (audit_df["picked_short"] == 1) & (audit_df["n_short_picked"] > 0),
        -audit_df["return_pct"] * 1e4 / audit_df["n_short_picked"].replace(0, np.nan),
        -audit_df["return_pct"] * 1e4 / K,
    )

    audit_df.to_parquet(OUT_DIR / "audit_panel.parquet", index=False)
    meta_df = pd.DataFrame(meta)
    meta_df.to_parquet(OUT_DIR / "cycle_meta.parquet", index=False)

    # Also save FULL prediction panel (all 10 folds) so downstream scripts can
    # reconstruct PIT rolling-IC universe with the same data the build script used.
    # The audit panel only contains OOS folds 1-9 + their pick/universe flags.
    # all_predictions includes fold 0 too — needed to reproduce the same universe
    # selection in Phase 2a-style universe rebuilds.
    all_pred_save = apd[["symbol", "open_time", "alpha_A", "return_pct", "pred", "fold"]].copy()
    all_pred_save["exit_time"] = (all_pred_save["open_time"]
                                  + pd.Timedelta(minutes=HORIZON * 5))
    all_pred_save.to_parquet(OUT_DIR / "all_predictions.parquet", index=False)
    print(f"\n  all_predictions saved: {len(all_pred_save):,} rows "
          f"(folds {sorted(all_pred_save['fold'].unique())})", flush=True)

    print(f"\n=== Audit panel summary ===", flush=True)
    print(f"  Total rows: {len(audit_df):,}", flush=True)
    print(f"  Unique cycles: {audit_df['time'].nunique()}", flush=True)
    print(f"  Unique symbols: {audit_df['symbol'].nunique()}", flush=True)
    print(f"  In universe: {audit_df['in_universe'].sum():,}", flush=True)
    print(f"  PIT eligible: {audit_df['eligible_pit'].sum():,}", flush=True)
    print(f"  Picked long: {audit_df['picked_long'].sum():,}", flush=True)
    print(f"  Picked short: {audit_df['picked_short'].sum():,}", flush=True)

    picked_long_contrib = audit_df.loc[audit_df["picked_long"] == 1, "long_contrib_bps_if_picked"].sum()
    picked_short_contrib = audit_df.loc[audit_df["picked_short"] == 1, "short_contrib_bps_if_picked"].sum()
    print(f"\n  Spread reconstruction:", flush=True)
    print(f"    sum picked long contributions:  {picked_long_contrib:+.0f} bps", flush=True)
    print(f"    sum picked short contributions: {picked_short_contrib:+.0f} bps", flush=True)
    print(f"    total gross spread:             {picked_long_contrib + picked_short_contrib:+.0f} bps", flush=True)
    print(f"    cycle_meta sum spread_bps:      {meta_df['spread_bps'].sum():+.0f} bps", flush=True)

    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
