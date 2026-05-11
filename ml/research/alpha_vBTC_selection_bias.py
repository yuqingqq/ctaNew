"""Selection-bias correction via leave-one-fold-out validation.

For each fold f in {5,6,7,8,9}:
  1. Pretend f is unseen
  2. Use other 4 folds to "select" between competing architecture choices
  3. Apply the selected architecture to fold f
  4. Record fold-f Sharpe under the selected architecture

Aggregate of the 5 fold-f Sharpes = bias-corrected estimate.
The 3 architecture decisions we test:
  D1. Features: winner_21 vs baseline_28
  D2. K: 3 vs 4
  D3. Schedule: static vs 180d×90d quarterly

For each LOFO iteration, decide each axis based on majority/best on the
other 4 folds, then evaluate on the held-out fold.

Comparisons:
  A: "biased_optimal" — best on all 5 folds (= winner_21 + K=4 + 180×90)
  B: "lofo_corrected" — chosen per LOFO, evaluated honestly on held-out
  Difference = selection bias magnitude
"""
from __future__ import annotations
import sys, time, warnings, itertools
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
OUT_DIR = REPO / "outputs/vBTC_selection_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)   # 5 seeds for compute speed
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
BASELINE_28 = list(V6_CLEAN_28)


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


def get_static_universe(all_pred_df):
    calib = all_pred_df[all_pred_df["fold"].isin([0, 1, 2, 3, 4])].dropna(subset=["alpha_A"])
    static_ics = calib.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    ).dropna().sort_values(ascending=False)
    return set(static_ics.head(TARGET_N).index.tolist())


def add_t_int(df):
    ts = df["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    df = df.copy()
    df["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()
    return df


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    folds_all = _multi_oos_splits(panel)

    # Train both feature sets on all folds
    print(f"\n=== Training all 10 folds for both feature sets ===", flush=True)
    fold_data_21 = {}; fold_data_28 = {}
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td_21, p_21 = train_fold(panel, folds_all[fid], WINNER_21)
        if td_21 is not None: fold_data_21[fid] = (td_21, p_21)
        td_28, p_28 = train_fold(panel, folds_all[fid], BASELINE_28)
        if td_28 is not None: fold_data_28[fid] = (td_28, p_28)
        print(f"  fold {fid}: ({time.time()-t0:.0f}s)", flush=True)

    # Build all_pred_df for each feature set
    def build_apd(fold_data):
        rows = []
        for fid, (td, p) in fold_data.items():
            df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
            df["pred"] = p; df["fold"] = fid
            rows.append(df)
        apd = pd.concat(rows, ignore_index=True).sort_values(["open_time", "symbol"])
        return add_t_int(apd)

    apd_21 = build_apd(fold_data_21)
    apd_28 = build_apd(fold_data_28)

    # Pre-build prod-folds aggregates for evaluation
    def get_test_data(apd):
        prod = apd[apd["fold"].isin(PROD_FOLDS)]
        return prod[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    test_data_21 = get_test_data(apd_21)
    test_data_28 = get_test_data(apd_28)

    # Common prod times (sampled)
    prod_pred = apd_21[apd_21["fold"].isin(PROD_FOLDS)]
    prod_times = sorted(prod_pred["open_time"].unique())
    prod_times_sampled = prod_times[::HORIZON]
    apd_21_clean = apd_21.dropna(subset=["alpha_A"])

    # Static universes (from calib folds 0-4 of the respective model)
    static_u_21 = get_static_universe(apd_21)
    static_u_28 = get_static_universe(apd_28)
    schedule_180_90_21 = build_schedule(apd_21_clean, prod_times_sampled, 180, 90)

    # === Evaluate all 8 architecture combinations across all 5 folds ===
    # 8 = 2 features × 2 K × 2 schedules
    print(f"\n=== Evaluating 8 architecture combinations on each fold ===", flush=True)
    print(f"  Features × K × Schedule = 2 × 2 × 2", flush=True)

    fold_pnl = {fid: {} for fid in PROD_FOLDS}   # {fid: {arch_name: list_of_net_bps}}
    arch_options = [
        ("F21", WINNER_21, test_data_21, static_u_21, schedule_180_90_21),
        ("F28", BASELINE_28, test_data_28, static_u_28, None),  # F28 with 180×90 needs different schedule; skip
    ]
    # Build schedule for F28 too
    apd_28_clean = apd_28.dropna(subset=["alpha_A"])
    schedule_180_90_28 = build_schedule(apd_28_clean, prod_times_sampled, 180, 90)

    configs = []
    for f_label, _, test_data, static_u, _ in arch_options:
        sched = schedule_180_90_21 if f_label == "F21" else schedule_180_90_28
        for K in [3, 4]:
            for sch_label, sch_obj in [("static", static_u), ("180x90", sched)]:
                configs.append((f"{f_label}_K{K}_{sch_label}", test_data, sch_obj, K))

    for label, test_data, universe, K in configs:
        df_eval = evaluate_with_dynamic_universe(test_data, universe, top_k=K)
        for fid in PROD_FOLDS:
            fold_t = set(test_data[test_data["open_time"].isin(
                prod_pred[prod_pred["fold"] == fid]["open_time"].unique())]["open_time"].unique())
            mask = df_eval["time"].isin(fold_t)
            net_f = df_eval.loc[mask, "net_bps"].to_numpy()
            fold_pnl[fid][label] = net_f
        print(f"  {label}: aggregate Sharpe = {_sharpe(df_eval['net_bps'].to_numpy()):+.2f}",
              flush=True)

    # === Selection-bias-corrected analysis ===
    print(f"\n{'='*100}", flush=True)
    print(f"SELECTION-BIAS CORRECTION via LEAVE-ONE-FOLD-OUT", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  For each held-out fold, pick best config based on OTHER 4 folds, eval on held-out.",
          flush=True)

    arch_names = [c[0] for c in configs]
    held_out_pnls = []
    biased_pnls = []
    for held_out in PROD_FOLDS:
        other = [f for f in PROD_FOLDS if f != held_out]
        # For each config, compute Sharpe on the 4 other folds
        best_arch = None; best_sh = -1e9
        for ar in arch_names:
            other_pnl = np.concatenate([fold_pnl[f][ar] for f in other])
            if len(other_pnl) < 10: continue
            sh = _sharpe(other_pnl)
            if sh > best_sh:
                best_sh = sh; best_arch = ar
        held_out_net = fold_pnl[held_out][best_arch]
        held_out_sh = _sharpe(held_out_net) if len(held_out_net) >= 3 else 0
        biased_arch = "F21_K4_180x90"   # the "claimed winner"
        biased_net = fold_pnl[held_out][biased_arch]
        biased_sh = _sharpe(biased_net) if len(biased_net) >= 3 else 0
        held_out_pnls.append(held_out_net)
        biased_pnls.append(biased_net)
        print(f"  held-out fold {held_out}: LOFO-selected = {best_arch} (other-folds Sh={best_sh:+.2f}) "
              f"→ held-out Sh={held_out_sh:+.2f}  |  biased {biased_arch}: held-out Sh={biased_sh:+.2f}",
              flush=True)

    # Aggregate across LOFO held-outs
    lofo_aggregate = np.concatenate(held_out_pnls)
    biased_aggregate = np.concatenate(biased_pnls)
    sh_lofo = _sharpe(lofo_aggregate)
    sh_biased = _sharpe(biased_aggregate)
    print(f"\n  AGGREGATE over all held-out folds:", flush=True)
    print(f"    LOFO-corrected (architecture chosen without seeing held-out): Sharpe = {sh_lofo:+.2f}",
          flush=True)
    print(f"    Biased (always picks claimed winner F21_K4_180x90):           Sharpe = {sh_biased:+.2f}",
          flush=True)
    print(f"    Selection bias estimate: {sh_biased - sh_lofo:+.2f}", flush=True)

    # Save details
    pd.DataFrame([{"fold": f, "arch": ar, "n": len(fold_pnl[f][ar]),
                    "mean_net": fold_pnl[f][ar].mean() if len(fold_pnl[f][ar]) else 0,
                    "sharpe": _sharpe(fold_pnl[f][ar]) if len(fold_pnl[f][ar]) >= 3 else 0}
                   for f in PROD_FOLDS for ar in arch_names]
                  ).to_csv(OUT_DIR / "per_fold_per_arch.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
