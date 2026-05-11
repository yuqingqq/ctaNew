"""Validate Test B (dispersion sizing) and B+C (combined) on corrected pipeline.

Setup:
  - Expanding training (default _slice)
  - Rolling-IC universe (180d / 90d) — calibrated config
  - Walk-forward across all 9 OOS folds

Variants:
  baseline           : binary conv_gate at 30th-pctile (current default)
  test_B_steep       : continuous sigmoid sizing on dispersion percentile (steep slope)
  test_B_smooth      : continuous sigmoid sizing (smoother)
  test_C_dd20_s30    : baseline + DD overlay (dd>20% → size=0.3)  [validated winner from Test C]
  test_BC_steep      : test_B_steep + DD overlay
  test_BC_smooth     : test_B_smooth + DD overlay

Sigmoid sizing: size = 1 / (1 + exp(-(pctile - 0.30) * slope))
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
OUT_DIR = REPO / "outputs/vBTC_test_BC_validation"
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


def train_fold_expanding(panel, fold, feat_set):
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


def evaluate_full(test_df, universe_per_t, top_k=K, sample_every=HORIZON):
    """Return per-cycle: time, fold, dispersion, dispersion_pctile, spread_bps, cost_bps, skipped_binary.

    With these, we can post-process any sizing logic.
    """
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
            bars.append({"time": t, "spread_bps": 0.0, "cost_bps": 0.0,
                          "dispersion": 0.0, "dispersion_pctile": np.nan, "skipped_binary": 1,
                          "n_u": len(g_u), "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())

        if len(dispersion_history) >= 30:
            past = np.asarray(list(dispersion_history))
            dispersion_pctile = float((past <= dispersion).mean())
            thr = float(np.quantile(past, GATE_PCTILE))
            skipped_binary = 1 if dispersion < thr else 0
        else:
            dispersion_pctile = np.nan
            skipped_binary = 0
        dispersion_history.append(dispersion)

        bk = min(band_k, len(g_u))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]

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
                          "dispersion": dispersion, "dispersion_pctile": dispersion_pctile,
                          "skipped_binary": skipped_binary,
                          "n_u": len(g_u), "n_long": 0, "n_short": 0})
            continue

        long_g = g_u[g_u["symbol"].isin(new_long)]
        short_g = g_u[g_u["symbol"].isin(new_short)]
        spread_bps = float((long_g["return_pct"].mean() - short_g["return_pct"].mean()) * 1e4)
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost_bps = float((churn_long + churn_short) * COST_PER_LEG)

        bars.append({"time": t, "spread_bps": spread_bps, "cost_bps": cost_bps,
                      "dispersion": dispersion, "dispersion_pctile": dispersion_pctile,
                      "skipped_binary": skipped_binary,
                      "n_u": len(g_u), "n_long": len(new_long), "n_short": len(new_short)})
        if not skipped_binary:
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


# -------------- OVERLAY APPLICATION --------------

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def apply_overlay(df_eval, b_mode, c_mode):
    """Apply test B (dispersion sizing) and test C (DD deleveraging) on per-cycle data.

    b_mode: 'binary' (no size, binary skip), 'steep' (sigmoid slope=20), 'smooth' (slope=10), 'off'
    c_mode: 'off' or dict like {'mode': 'cum_dd_pct', 'threshold_dd': 0.20, 'size_drawdown': 0.3}
    """
    df = df_eval.copy().reset_index(drop=True)
    spread = df["spread_bps"].to_numpy()
    cost = df["cost_bps"].to_numpy()
    pctile = df["dispersion_pctile"].to_numpy()
    skipped_binary = df["skipped_binary"].to_numpy()

    # Test B size: depends on dispersion percentile
    n = len(df)
    size_b = np.ones(n)
    if b_mode == "binary":
        # Binary skip below 30th pctile: size=0 (skip), size=1 otherwise
        size_b = np.where(skipped_binary == 1, 0.0, 1.0)
        # In skipped cycles, the original evaluator HOLDS prior position, capturing spread without cost.
        # Approximate: skipped cycles have spread_bps from held position, cost_bps = 0.
        # That's how the original evaluator wrote the data (skipped cycles had cost_bps=0).
        # So baseline net = spread_bps - cost_bps already (cost is 0 on skip).
    elif b_mode == "steep":
        # sigmoid slope=20 around 0.30 pctile
        valid = ~np.isnan(pctile)
        size_b = np.zeros(n)
        size_b[valid] = sigmoid((pctile[valid] - 0.30) * 20)
        size_b[~valid] = 1.0  # before 30 cycles of history, no sizing
    elif b_mode == "smooth":
        valid = ~np.isnan(pctile)
        size_b = np.zeros(n)
        size_b[valid] = sigmoid((pctile[valid] - 0.30) * 10)
        size_b[~valid] = 1.0
    elif b_mode == "off":
        size_b = np.ones(n)
    else:
        raise ValueError(f"unknown b_mode {b_mode}")

    # Apply size_b: net = size_b * (spread - cost). Skipped cycles in binary mode: spread is held position's spread, no entry cost.
    net_b = size_b * (spread - cost)

    # Test C: trailing-DD deleveraging applied AFTER test B
    size_c = np.ones(n)
    if c_mode != "off":
        c = c_mode
        cum = np.cumsum(net_b)
        peak = -np.inf
        for i in range(n):
            peak = max(peak, cum[i] if i > 0 else 0)
            if peak > 0:
                dd_pct = (peak - cum[i]) / peak
                size_c[i] = c["size_drawdown"] if dd_pct > c["threshold_dd"] else 1.0
            else:
                size_c[i] = 1.0

    net_final = size_c * net_b
    df["size_b"] = size_b
    df["size_c"] = size_c
    df["size_combined"] = size_b * size_c
    df["net_bps"] = net_final
    return df


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    print(f"\n=== Train all 10 folds (expanding window) ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p = train_fold_expanding(panel, folds_all[fid], feat_set)
        if td is None: continue
        df = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        df["pred"] = p; df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: {len(df):,} rows ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    # Rolling-IC universe (180/90)
    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    oos_times_all = sorted(oos_pred["open_time"].unique())
    oos_times_sampled = oos_times_all[::HORIZON]
    rolling_universe = build_rolling_ic_universe(apd, oos_times_sampled, IC_WINDOW_DAYS, IC_UPDATE_DAYS)

    test_data = oos_pred[["symbol", "open_time", "pred", "return_pct", "alpha_A"]].copy()

    # Run evaluator once → get per-cycle dispersion + spread + cost data
    print(f"\n=== Running base evaluator (saves per-cycle dispersion + spread + cost) ===", flush=True)
    df_full = evaluate_full(test_data, rolling_universe)
    df_full["time"] = pd.to_datetime(df_full["time"])
    for fid in OOS_FOLDS:
        fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
        df_full.loc[df_full["time"].isin(fold_t), "fold"] = fid
    df_full.to_csv(OUT_DIR / "per_cycle_full.csv", index=False)
    print(f"  Saved per-cycle data: n={len(df_full)}", flush=True)
    print(f"  spread_bps: mean={df_full['spread_bps'].mean():+.2f}, "
          f"std={df_full['spread_bps'].std():.1f}", flush=True)
    print(f"  cost_bps: mean={df_full['cost_bps'].mean():.2f}", flush=True)
    print(f"  dispersion_pctile: mean={df_full['dispersion_pctile'].mean():.2f} "
          f"(NaN count: {df_full['dispersion_pctile'].isna().sum()})", flush=True)
    print(f"  skipped_binary: {df_full['skipped_binary'].sum()} of {len(df_full)} cycles", flush=True)

    # Variants
    DD_OVERLAY = {"mode": "cum_dd_pct", "threshold_dd": 0.20, "size_drawdown": 0.3}

    variants = [
        ("baseline_binary",   "binary", "off"),
        ("test_B_steep",      "steep",  "off"),
        ("test_B_smooth",     "smooth", "off"),
        ("test_B_off",        "off",    "off"),  # no gate at all (full size always)
        ("test_C_only",       "binary", DD_OVERLAY),
        ("test_BC_steep",     "steep",  DD_OVERLAY),
        ("test_BC_smooth",    "smooth", DD_OVERLAY),
    ]

    print(f"\n{'=' * 100}", flush=True)
    print(f"OVERLAY VALIDATION — base = expanding train + rolling-IC (180/90)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'variant':<22}  {'Sharpe':>7}  {'CI':>17}  {'mean':>6}  {'std':>6}  "
          f"{'maxDD':>7}  {'sizeB':>5}  {'sizeC':>5}", flush=True)

    results = []
    cycles_by_variant = {}
    for label, b_mode, c_mode in variants:
        df_v = apply_overlay(df_full, b_mode, c_mode)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(net)
        size_b_mean = df_v["size_b"].mean()
        size_c_mean = df_v["size_c"].mean()
        per_fold = {}
        for fid in OOS_FOLDS:
            fold_n = df_v[df_v["fold"] == fid]["net_bps"].to_numpy()
            if len(fold_n) >= 3:
                per_fold[fid] = _sharpe(fold_n)
        prod_n = df_v[df_v["fold"].isin(PROD_FOLDS)]["net_bps"].to_numpy()
        prod_sh = _sharpe(prod_n) if len(prod_n) >= 3 else 0
        results.append({"variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "mean": net.mean(), "std": net.std(), "max_dd": max_dd,
                          "size_b": size_b_mean, "size_c": size_c_mean,
                          "prod_sharpe": prod_sh,
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        cycles_by_variant[label] = df_v
        print(f"  {label:<22}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{net.mean():>+6.2f}  {net.std():>6.1f}  {max_dd:>+7.0f}  "
              f"{size_b_mean:>5.2f}  {size_c_mean:>5.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'variant':<22}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<22}  " + cells, flush=True)

    print(f"\n  Prod folds 5-9 isolation:", flush=True)
    for r in results:
        base = next(rr for rr in results if rr['variant'] == 'baseline_binary')
        delta = r['prod_sharpe'] - base['prod_sharpe']
        print(f"  {r['variant']:<22}  prod_Sh={r['prod_sharpe']:+.2f}  "
              f"(Δ vs baseline: {delta:+.2f})", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "test_BC_validation_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
