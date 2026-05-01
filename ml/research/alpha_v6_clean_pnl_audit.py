"""PnL + model-level leakage audit for v6_clean.

Three orthogonal checks the prior audits did NOT cover:

1. **Model-level shift test.** Per-feature shift test (alpha_v6_leakage_check.py)
   verifies each feature individually — but the ensemble could in principle
   aggregate tiny per-feature look-aheads into a larger model-level leak.
   Test: train once, predict on TEST features at bar t (baseline) vs TEST
   features at bar t-1 (lagged). If lagged Sharpe ≈ baseline Sharpe (drop
   < 10%), no model-level leakage.

2. **β-neutral consistency.** Under perfect β-neutral execution,
   spread_ret_BN should equal spread_alpha_BN to within numerical noise.
   The multi-OOS run reported +30.70 vs +30.20 (0.5 bp gap). This script
   re-confirms on the holdout fold and prints per-cycle diff distribution.

3. **Cost-stack reconstruction.** Recompute net per cycle from raw
   spread_ret and turnover, verify match with portfolio_pnl_turnover_aware
   internal accounting.
"""
from __future__ import annotations

import gc
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_BASE_FEATURES, XS_CROSS_FEATURES, XS_FLOW_FEATURES, XS_RANK_FEATURES,
    XS_FEATURE_COLS_V6, XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train, _holdout_split, _slice,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_SET = os.environ.get("FEATURE_SET", "v6_clean").lower()
ACTIVE_COLS = XS_FEATURE_COLS_V6_CLEAN if FEATURE_SET == "v6_clean" else XS_FEATURE_COLS_V6


def _build_panel():
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))
    feats_by_sym = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}
    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols = [c for c in ACTIVE_COLS if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(ACTIVE_COLS) + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols) - set(rank_cols))
    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
        del f
    del enriched, feats_by_sym
    gc.collect()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames, labels
    gc.collect()
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel[rank_cols] = panel[rank_cols].astype("float32")
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    return panel


def _per_bar_xs_ic(test_df, pred_arr):
    df = test_df[["open_time", "alpha_realized"]].copy()
    df["pred"] = pred_arr
    bar_ics = []
    for t, g in df.groupby("open_time"):
        if len(g) < 5:
            continue
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())
        if not np.isnan(ic):
            bar_ics.append(ic)
    return np.mean(bar_ics) if bar_ics else np.nan, len(bar_ics)


def main():
    panel = _build_panel()
    log.info("panel: %d rows", len(panel))

    fold = _holdout_split(panel)[0]
    train, cal, test = _slice(panel, fold)
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    test_f = test
    log.info("train: %d, cal: %d, test: %d", len(train_f), len(cal_f), len(test_f))

    feat_cols = list(ACTIVE_COLS)
    X_train = train_f[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_f["demeaned_target"].to_numpy(dtype=np.float32)
    X_cal = cal_f[feat_cols].to_numpy(dtype=np.float32)
    y_cal = cal_f["demeaned_target"].to_numpy(dtype=np.float32)

    log.info("training v6_clean ensemble (5 seeds)...")
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
        log.info("  seed %d trained, best_iter=%d", seed, m.best_iteration)
        models.append(m)
    del X_train, y_train, X_cal, y_cal, train, cal, train_f, cal_f
    gc.collect()

    # ====================================================================
    # CHECK 1: Model-level shift test
    # ====================================================================
    print("\n" + "=" * 100)
    print("CHECK 1: Model-level shift test")
    print("=" * 100)
    print("\nIf v6_clean has any aggregated look-ahead, predicting with features")
    print("lagged 1 bar (per-symbol) will produce a clearly-degraded Sharpe.")
    print("If features are PIT, lagged predictions should be ~same quality.\n")

    # Baseline: predict on test with original features
    X_test = test_f[feat_cols].to_numpy(dtype=np.float32)
    yt_baseline = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0)
    baseline_ic, n_bars_b = _per_bar_xs_ic(test_f, yt_baseline)
    print(f"  Baseline OOS XS IC: {baseline_ic:+.4f}  (n_bars={n_bars_b})")

    # Lagged: shift features +1 per symbol BEFORE predict
    test_f_lag = test_f.copy()
    test_f_lag = test_f_lag.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    for c in feat_cols:
        if c == "sym_id":
            continue  # don't lag the categorical
        test_f_lag[c] = test_f_lag.groupby("symbol")[c].shift(1)
    test_f_lag = test_f_lag.dropna(subset=[c for c in feat_cols if c != "sym_id"])
    X_test_lag = test_f_lag[feat_cols].to_numpy(dtype=np.float32)
    yt_lag = np.mean([m.predict(X_test_lag, num_iteration=m.best_iteration) for m in models], axis=0)
    lag_ic, n_bars_l = _per_bar_xs_ic(test_f_lag, yt_lag)
    print(f"  Lagged   OOS XS IC: {lag_ic:+.4f}  (n_bars={n_bars_l})")
    print(f"  Δ = {(lag_ic - baseline_ic):+.4f}  ({100 * (lag_ic - baseline_ic) / baseline_ic:+.1f}% of baseline)")

    # Also lag +2, +3 to see decay pattern
    print(f"\n  Decay pattern (multi-bar lag):")
    test_sorted = test_f.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    for k in (1, 2, 3, 5, 10):
        tlag = test_sorted.copy()
        for c in feat_cols:
            if c == "sym_id":
                continue
            tlag[c] = tlag.groupby("symbol")[c].shift(k)
        tlag = tlag.dropna(subset=[c for c in feat_cols if c != "sym_id"])
        X = tlag[feat_cols].to_numpy(dtype=np.float32)
        yt = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
        ic, n = _per_bar_xs_ic(tlag, yt)
        print(f"    lag +{k:>2} bars: OOS XS IC = {ic:+.4f}  ({100 * ic / baseline_ic:+.0f}% of baseline)")

    print("\n  Verdict:")
    if abs(lag_ic - baseline_ic) > 0.5 * baseline_ic:
        print("  ⚠️  LARGE drop with 1-bar lag — possible model-level look-ahead")
    elif abs(lag_ic - baseline_ic) > 0.2 * baseline_ic:
        print("  ⚠️  Moderate drop with 1-bar lag — investigate")
    else:
        print("  ✓  Lag drop within expected range (~5-15% per bar at h=288)")

    # ====================================================================
    # CHECK 2: β-neutral consistency
    # ====================================================================
    print("\n" + "=" * 100)
    print("CHECK 2: β-neutral consistency (ret_BN should ≈ alpha_BN)")
    print("=" * 100)
    print("\nUnder perfect β-stripping, spread_ret_BN = spread_alpha_BN per cycle.")
    print("Any persistent gap reveals β-mis-scaling or implicit market exposure.\n")

    result_bn = portfolio_pnl_turnover_aware(test_f, yt_baseline, top_frac=0.2,
                                              sample_every=HORIZON, beta_neutral=True)
    bdf = result_bn["df"]
    diff = bdf["spread_ret_bps"] - bdf["spread_alpha_bps"]
    print(f"  n_cycles: {len(bdf)}")
    print(f"  mean spread_ret_BN:    {bdf['spread_ret_bps'].mean():+.3f} bps")
    print(f"  mean spread_alpha_BN:  {bdf['spread_alpha_bps'].mean():+.3f} bps")
    print(f"  mean diff (ret - alpha): {diff.mean():+.3f} bps")
    print(f"  std diff:               {diff.std():.3f} bps")
    print(f"  max |diff|:             {diff.abs().max():.3f} bps")
    print(f"  diff > 5 bps cycles:    {(diff.abs() > 5).sum()} / {len(bdf)}")
    print(f"  scale_L mean / range:   {bdf['scale_L'].mean():.3f}  [{bdf['scale_L'].min():.2f}, {bdf['scale_L'].max():.2f}]")
    print(f"  scale_S mean / range:   {bdf['scale_S'].mean():.3f}  [{bdf['scale_S'].min():.2f}, {bdf['scale_S'].max():.2f}]")
    print(f"  gross exposure mean:    {bdf['gross_exposure'].mean():.3f}")
    print(f"  degenerate β cycles:    {bdf['degen_beta'].sum()} / {len(bdf)}")
    if abs(diff.mean()) > 2:
        print(f"\n  ⚠️  Mean diff > 2 bps — β-neutral not stripping cleanly")
    else:
        print(f"\n  ✓  Mean diff small (<2 bps) — β-stripping working")

    # ====================================================================
    # CHECK 3: Cost-stack reconstruction
    # ====================================================================
    print("\n" + "=" * 100)
    print("CHECK 3: Cost-stack reconstruction (manual recomputation matches PnL fn)")
    print("=" * 100)

    # Manual cost recomputation
    cost_per_leg = 12.0  # bps RT, from NAKED_COST_BPS_PER_LEG
    expected_cost = cost_per_leg * (bdf["long_turnover"] + bdf["short_turnover"])
    diff_cost = (bdf["cost_bps"] - expected_cost).abs()
    print(f"\n  cost = cost_per_leg × (long_to + short_to)")
    print(f"  manual recomputation matches stored cost_bps: max diff = {diff_cost.max():.6f}")
    if diff_cost.max() < 1e-3:
        print(f"  ✓  Cost formula matches")
    else:
        print(f"  ⚠️  Cost mismatch detected")

    # Verify net = gross_ret_bps - cost_bps
    expected_net = bdf["spread_ret_bps"] - bdf["cost_bps"]
    diff_net = (bdf["net_bps"] - expected_net).abs()
    print(f"\n  net = spread_ret_bps - cost_bps")
    print(f"  max diff: {diff_net.max():.6f}")
    if diff_net.max() < 1e-3:
        print(f"  ✓  Net formula matches")
    else:
        print(f"  ⚠️  Net mismatch detected")

    # First trade vs subsequent trades cost pattern
    print(f"\n  First trade cost: {bdf['cost_bps'].iloc[0]:.2f} bps  (should be ~24 = full entry × 2 legs)")
    print(f"  Subsequent mean: {bdf['cost_bps'].iloc[1:].mean():.2f} bps")
    print(f"  Subsequent range: [{bdf['cost_bps'].iloc[1:].min():.2f}, {bdf['cost_bps'].iloc[1:].max():.2f}]")
    print(f"  long_turnover stats:  mean={bdf['long_turnover'].mean():.3f}, std={bdf['long_turnover'].std():.3f}")
    print(f"  short_turnover stats: mean={bdf['short_turnover'].mean():.3f}, std={bdf['short_turnover'].std():.3f}")

    # ====================================================================
    # CHECK 4: Sample-cadence verification (non-overlapping cycles)
    # ====================================================================
    print("\n" + "=" * 100)
    print("CHECK 4: Sample-cadence verification (non-overlapping forward windows)")
    print("=" * 100)
    cycle_times = pd.to_datetime(bdf["time"])
    deltas = cycle_times.diff().dt.total_seconds() / 60  # minutes
    expected_delta_minutes = HORIZON * 5  # 5-min bars
    print(f"\n  Expected cycle gap: {expected_delta_minutes} minutes (h={HORIZON} × 5min)")
    print(f"  Actual gap stats: mean={deltas.mean():.1f}, std={deltas.std():.1f}, min={deltas.min():.1f}, max={deltas.max():.1f}")
    consistent = (deltas.dropna() == expected_delta_minutes).mean()
    print(f"  Fraction of cycles with exact h-bar gap: {100 * consistent:.1f}%")
    if consistent > 0.95:
        print(f"  ✓  Cycles are non-overlapping h=288")
    else:
        print(f"  ⚠️  Cycle cadence inconsistent")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Baseline OOS XS IC:    {baseline_ic:+.4f}")
    print(f"  Lag-1  OOS XS IC:      {lag_ic:+.4f}  ({100 * lag_ic / baseline_ic:+.0f}% of baseline)")
    print(f"  β-neutral diff (mean): {diff.mean():+.3f} bps  (gap >2 = bad)")
    print(f"  Cost recomputation:    max diff {diff_cost.max():.6f}  (< 1e-3 = ok)")
    print(f"  Cycle cadence:         {100 * consistent:.0f}% on h={HORIZON} grid")


if __name__ == "__main__":
    main()
