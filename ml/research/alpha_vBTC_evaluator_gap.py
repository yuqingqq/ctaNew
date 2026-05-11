"""Resolve the evaluator gap: re-run validated pipeline with `evaluate_stacked`.

The +0.9 Sharpe gap between phase 6 (+5.19) and current_validation (+4.31) on
prod folds 5-9 traces to evaluator differences:
  evaluate_stacked: β-neutral scaling, live-model MtM holds, turnover-aware cost
  local evaluator:  equal-weighted, skip-zero, Jaccard turnover

Hypothesis: β-neutral scaling is the structural alpha source. Adopting it on
the rolling-IC pipeline should boost Sharpe ~+0.5-0.9 with no structural risk.

This script:
  1. Trains all 10 folds (expanding window, default _slice)
  2. Builds rolling-IC universe (180/90)
  3. Filters test panel to universe per cycle
  4. Calls evaluate_stacked (β-neutral, live-model, conv_gate + PM_M2_b1)
  5. Reports Sharpe walk-forward + prod folds 5-9
  6. Applies Test C overlay (dd_pct>20%_size=0.3) on top

Comparison axes:
  - local_evaluator (baseline reference) vs evaluate_stacked
  - With and without Test C overlay
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
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR = REPO / "outputs/vBTC_evaluator_gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)
COST_PER_LEG = 4.5
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


def build_rolling_ic_universe(all_pred_df, target_times, ic_window_days, update_days):
    """Returns (boundary_to_universe, t0_ms, update_ms) for direct boundary lookup."""
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms
    all_pred_clean = all_pred_df.dropna(subset=["alpha_A"])
    if not target_times: return {}, 0, update_ms
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
    return boundary_to_universe, t0_ms, update_ms


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


def evaluate_with_stacked(panel_test, preds, boundary_to_universe, t0_ms, update_ms):
    """Run evaluate_stacked, restricting symbols by the universe at each row's boundary.

    For each 5-min row in panel_test, find its boundary (boundary_ms = t0_ms + n*update_ms)
    and keep only if symbol is in that boundary's universe.
    Preserves 5-min granularity so evaluate_stacked's internal sampling works.
    """
    df = panel_test.copy()
    df["pred"] = preds
    ts = pd.to_datetime(df["open_time"])
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    t_arr = ts.astype("datetime64[ms]").astype("int64").to_numpy()
    n_arr = (t_arr - t0_ms) // update_ms
    boundary_arr = t0_ms + n_arr * update_ms
    sym_arr = df["symbol"].to_numpy()
    keep_mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        b = int(boundary_arr[i])
        u = boundary_to_universe.get(b, set())
        if sym_arr[i] in u:
            keep_mask[i] = True
    df_sub = df[keep_mask].copy()
    if df_sub.empty:
        return pd.DataFrame()
    pred_sub = df_sub["pred"].to_numpy()
    return evaluate_stacked(
        df_sub, pred_sub,
        use_conv_gate=True, use_pm_gate=True,
        top_k=K, cost_bps_per_leg=COST_PER_LEG,
        execution_model="live",
    )


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    # Required columns for evaluate_stacked
    needed = ["alpha_realized", "basket_fwd", "beta_short_vs_bk"]
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        print(f"  MISSING for evaluate_stacked: {missing}", flush=True)
        return

    print(f"\n=== Train all 10 folds (expanding window) ===", flush=True)
    fold_data = {}  # fid -> (test_df, preds)
    all_pred_records = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p = train_fold_expanding(panel, folds_all[fid], feat_set)
        if td is None: continue
        fold_data[fid] = (td, p)
        rec = td[["symbol", "open_time", "alpha_A", "return_pct"]].copy()
        rec["pred"] = p; rec["fold"] = fid
        all_pred_records.append(rec)
        print(f"  fold {fid}: {len(td):,} rows ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_pred_records, ignore_index=True).sort_values(["open_time", "symbol"])
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    # Rolling-IC universe
    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    oos_times_all = sorted(oos_pred["open_time"].unique())
    oos_times_sampled = oos_times_all[::HORIZON]
    boundary_to_universe, t0_ms, update_ms = build_rolling_ic_universe(
        apd, oos_times_sampled, IC_WINDOW_DAYS, IC_UPDATE_DAYS)
    print(f"\n  Rolling-IC: {len(boundary_to_universe)} unique boundaries, "
          f"unique universes: {len({frozenset(u) for u in boundary_to_universe.values() if u})}",
          flush=True)

    # Run evaluate_stacked on each fold's test panel + preds, filtered by universe
    print(f"\n=== Run evaluate_stacked across folds 1-9 (β-neutral, live model) ===", flush=True)
    all_cycles = []
    for fid in OOS_FOLDS:
        if fid not in fold_data: continue
        test_df, preds = fold_data[fid]
        df_eval = evaluate_with_stacked(test_df, preds, boundary_to_universe, t0_ms, update_ms)
        if df_eval.empty: continue
        df_eval["fold"] = fid
        all_cycles.append(df_eval)
    if not all_cycles:
        print("No data!", flush=True)
        return
    cycles = pd.concat(all_cycles, ignore_index=True)
    cycles = cycles.sort_values("time").reset_index(drop=True)

    print(f"  Total cycles: {len(cycles)}, mean net: {cycles['net_bps'].mean():+.2f}", flush=True)

    # ---- Compare baseline (no overlay) ----
    net = cycles["net_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    max_dd = _max_dd(net)
    prod_mask = cycles["fold"].isin(PROD_FOLDS)
    prod_net = cycles.loc[prod_mask, "net_bps"].to_numpy()
    prod_sh = _sharpe(prod_net)
    prod_dd = _max_dd(prod_net)
    print(f"\n  Walk-forward (folds 1-9): Sharpe={sh:+.2f} [{lo:+.2f}, {hi:+.2f}], "
          f"maxDD={max_dd:+.0f}, mean={net.mean():+.2f}", flush=True)
    print(f"  Prod folds 5-9:           Sharpe={prod_sh:+.2f}, maxDD={prod_dd:+.0f}", flush=True)

    # Per-fold
    print(f"\n  Per-fold Sharpe:", flush=True)
    cells = []
    for fid in OOS_FOLDS:
        n_f = cycles[cycles["fold"] == fid]["net_bps"].to_numpy()
        cells.append(f"f{fid}={_sharpe(n_f) if len(n_f) >= 3 else 0:+5.2f}")
    print("  " + "  ".join(cells), flush=True)

    # ---- Apply Test C overlay ----
    overlay_net, sizes = apply_dd_overlay(net, threshold_dd=0.20, size_drawdown=0.3)
    sh_c, lo_c, hi_c = block_bootstrap_ci(overlay_net, statistic=_sharpe,
                                            block_size=7, n_boot=2000)
    max_dd_c = _max_dd(overlay_net)
    prod_overlay = overlay_net[prod_mask.to_numpy()]
    prod_sh_c = _sharpe(prod_overlay)
    prod_dd_c = _max_dd(prod_overlay)
    print(f"\n  + Test C overlay (dd>20%_size=0.3):", flush=True)
    print(f"    Walk-forward: Sharpe={sh_c:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}], "
          f"maxDD={max_dd_c:+.0f}", flush=True)
    print(f"    Prod folds 5-9: Sharpe={prod_sh_c:+.2f}, maxDD={prod_dd_c:+.0f}", flush=True)

    # ---- Comparison summary ----
    print(f"\n{'=' * 90}", flush=True)
    print(f"EVALUATOR COMPARISON: local vs evaluate_stacked  (rolling-IC, K=4, N=15, WINNER_21)", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"  {'evaluator':<22}  {'overlay':<22}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}", flush=True)
    print(f"  {'local (current_val)':<22}  {'none':<22}  {'+2.59':>7}  [+0.53,+4.69]  -6,009", flush=True)
    print(f"  {'local (test_C_validation)':<22}  {'dd>20%_size=0.3':<22}  {'+3.94':>7}  [+2.22,+5.79]  -2,153", flush=True)
    print(f"  {'evaluate_stacked':<22}  {'none':<22}  {sh:>+7.2f}  [{lo:+.2f},{hi:+.2f}]  {max_dd:+.0f}",
          flush=True)
    print(f"  {'evaluate_stacked':<22}  {'dd>20%_size=0.3':<22}  {sh_c:>+7.2f}  [{lo_c:+.2f},{hi_c:+.2f}]  {max_dd_c:+.0f}",
          flush=True)

    summary = pd.DataFrame([
        {"evaluator": "evaluate_stacked", "overlay": "none",
          "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd, "mean": net.mean(),
          "prod_sharpe": prod_sh, "prod_max_dd": prod_dd},
        {"evaluator": "evaluate_stacked", "overlay": "dd>20%_size=0.3",
          "sharpe": sh_c, "ci_lo": lo_c, "ci_hi": hi_c, "max_dd": max_dd_c, "mean": overlay_net.mean(),
          "prod_sharpe": prod_sh_c, "prod_max_dd": prod_dd_c},
    ])
    summary.to_csv(OUT_DIR / "evaluator_gap_summary.csv", index=False)
    cycles.to_csv(OUT_DIR / "evaluate_stacked_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
