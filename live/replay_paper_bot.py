"""Replay-mode validator for live/paper_bot.py.

Validates that the live PANEL-BUILD path produces predictions matching
the canonical backtest (alpha_v6_clean_pnl_audit.py) when given the same
training data. To do so apples-to-apples we:

  1. Build the panel via the LIVE code path (build_panel_for_inference)
  2. Train a FRESH model on the holdout fold's pre-test region (same as
     backtest does — the saved live artifact has seen the fold and would
     leak)
  3. Predict on the fold via live path + fresh model
  4. Compare portfolio P&L / IC to the backtest reference

Backtest reference (alpha_v6_clean_pnl_audit.py on holdout fold):
  spread_ret_BN: +41.47 bps/cycle
  baseline OOS XS IC: +0.0519
  n_cycles: 90

Outputs:
  outputs/replay_paper_bot.csv     per-cycle replay P&L
  outputs/replay_paper_bot.log     numeric comparison
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _holdout_split, _slice, _train,
)
from live.paper_bot import (
    build_panel_for_inference, build_kline_features_inmem,
    select_portfolio, MODEL_DIR,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("replay")


def _load_klines_from_disk(universe):
    """Load historical 5m klines for each symbol via the canonical cache
    that `build_kline_features` already uses."""
    out = {}
    for s in universe:
        f = build_kline_features(s)
        if f.empty:
            continue
        # build_kline_features already computed features; but
        # build_panel_for_inference re-runs kline_features_inmem on raw
        # klines. We pass the OHLCV columns through.
        ohlcv = f[["open", "high", "low", "close", "volume"]].copy()
        if "trades" in f.columns:
            ohlcv["trades"] = f["trades"]
        if "quote_volume" in f.columns:
            ohlcv["quote_volume"] = f["quote_volume"]
        out[s] = ohlcv
    return out


def main():
    feat_cols = list(XS_FEATURE_COLS_V6_CLEAN)
    universe = list_universe(min_days=200)
    sym_to_id = {s: i for i, s in enumerate(sorted(universe))}

    log.info("Loading historical klines for %d symbols...", len(universe))
    klines_by_sym = _load_klines_from_disk(universe)
    log.info("Got klines for %d/%d symbols", len(klines_by_sym), len(universe))

    # Build panel via the LIVE PATH (not the alpha_v4_xs_1d backtest path)
    log.info("Building panel via live code path (build_panel_for_inference)...")
    panel = build_panel_for_inference(klines_by_sym, sym_to_id)
    log.info("panel: %d rows, time %s -> %s", len(panel),
              panel["open_time"].min(), panel["open_time"].max())

    # We need realized labels for replay evaluation. To match the canonical
    # backtest, use the same `make_xs_alpha_labels` chain — which computes
    # demeaned_target on each symbol's ORIGINAL index BEFORE reindex/join.
    # Computing it inline on the post-merge panel produces slightly different
    # rolling stats and breaks reproducibility.
    log.info("Building labels via make_xs_alpha_labels (matches backtest)...")
    closes_df = pd.DataFrame({s: kl["close"] for s, kl in klines_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes_df)
    # Build enriched per-symbol frames so make_xs_alpha_labels has the
    # `beta_short_vs_bk` it needs.
    enriched = {}
    for s, kl in klines_by_sym.items():
        f = build_kline_features_inmem(kl)
        f = f.reindex(closes_df.index)
        f = add_basket_features(f, basket_close, basket_ret)
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    label_frames = []
    for s, lab in labels.items():
        lab2 = lab.copy().reset_index().rename(columns={"index": "open_time"})
        lab2["symbol"] = s
        label_frames.append(lab2)
    labels_df = pd.concat(label_frames, ignore_index=True)
    panel = panel.merge(labels_df, on=["open_time", "symbol"], how="left")

    # Restrict to the holdout fold for an apples-to-apples comparison with
    # alpha_v4_xs_1d's reported numbers. Match the backtest exactly: do NOT
    # dropna on return_pct here — alpha_v4_xs_1d's `_slice` is purely
    # time-based and `portfolio_pnl_turnover_aware` samples cycles by
    # `times[::sample_every]`, so dropping last-horizon bars here would
    # shift cycle boundaries by 287 bars and change the sampled cycles.
    fold = _holdout_split(panel)[0]
    test = panel[(panel["open_time"] >= fold["test_start"])
                  & (panel["open_time"] < fold["test_end"])].copy()
    log.info("Holdout fold test rows: %d, time %s -> %s",
              len(test), fold["test_start"], fold["test_end"])

    # Train a FRESH model on the fold's pre-test region (so the test fold is
    # genuinely OOS). Mirrors alpha_v6_clean_pnl_audit / alpha_v4_xs_1d
    # exactly: no dropna here — match the audit's `_slice + regime filter`
    # protocol. The audit's `_stack_xs_panel` already dropped rows where
    # source feature cols (cols+demeaned_target+return_pct+basket_fwd) were
    # NaN — replicate that filtering at panel-build time below if needed.
    log.info("Training fresh model on pre-test region (matching backtest)...")
    train = panel[panel["open_time"] < fold["cal_start"]]
    cal = panel[(panel["open_time"] >= fold["cal_start"])
                  & (panel["open_time"] < fold["cal_end"])]
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    log.info("train: %d (raw), cal: %d (raw)", len(train_f), len(cal_f))

    X_train = train_f[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_f["demeaned_target"].to_numpy(dtype=np.float32)
    X_cal = cal_f[feat_cols].to_numpy(dtype=np.float32)
    y_cal = cal_f["demeaned_target"].to_numpy(dtype=np.float32)
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
        log.info("  seed %d trained, best_iter=%d", seed, m.best_iteration)
        models.append(m)

    X = test[feat_cols].to_numpy(dtype=np.float32)
    yt = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)

    # Run portfolio P&L
    log.info("Running portfolio P&L (β-neutral, K=5, h=288 cadence)...")
    res = portfolio_pnl_turnover_aware(test, yt, top_frac=0.20,
                                        sample_every=HORIZON, beta_neutral=True)
    bdf = res["df"]
    log.info("n_cycles=%d", len(bdf))

    # Aggregate stats
    cycles_per_year = 365.0
    def _sharpe_yr(arr):
        if arr.std() == 0:
            return 0.0
        return (arr.mean() / arr.std()) * np.sqrt(cycles_per_year)
    s_gross, sg_lo, sg_hi = block_bootstrap_ci(bdf["spread_ret_bps"].to_numpy(),
                                                 statistic=_sharpe_yr, block_size=7)
    s_net, sn_lo, sn_hi = block_bootstrap_ci(bdf["net_bps"].to_numpy(),
                                               statistic=_sharpe_yr, block_size=7)

    print("\n" + "=" * 100)
    print("REPLAY VALIDATION: live code path on holdout fold")
    print("=" * 100)
    print(f"  fold: {fold['test_start']} -> {fold['test_end']}")
    print(f"  n_cycles: {len(bdf)}")
    print(f"  spread_ret_bps mean:   {bdf['spread_ret_bps'].mean():+.3f}")
    print(f"  spread_alpha_bps mean: {bdf['spread_alpha_bps'].mean():+.3f}")
    print(f"  cost_bps mean (12bps/leg backtest): {bdf['cost_bps'].mean():.3f}")
    print(f"  net_bps mean: {bdf['net_bps'].mean():+.3f}")
    print(f"  rank_ic mean: {bdf['rank_ic'].mean():+.4f}")
    print(f"  long_to mean: {bdf['long_turnover'].mean():.3f}")
    print(f"  Sharpe gross (annualized): {s_gross:+.2f}  CI [{sg_lo:+.2f}, {sg_hi:+.2f}]")
    print(f"  Sharpe net   (annualized): {s_net:+.2f}  CI [{sn_lo:+.2f}, {sn_hi:+.2f}]")

    # Reference numbers from prior backtest (committed in 47b9a3e):
    # holdout fold (alpha_v6_clean_pnl_audit results):
    #   spread_ret_BN +41.465 bps/cycle, baseline OOS XS IC +0.0519
    print("\n  Reference (alpha_v6_clean_pnl_audit on same fold, fresh-trained model):")
    print(f"    spread_ret_BN: +41.47 bps/cycle  (replay: {bdf['spread_ret_bps'].mean():+.2f})")
    print(f"    baseline IC:   +0.0519           (replay rank_ic: {bdf['rank_ic'].mean():+.4f})")

    delta_spread = abs(bdf["spread_ret_bps"].mean() - 41.47)
    delta_ic = abs(bdf["rank_ic"].mean() - 0.0519)
    print(f"\n  |Δ spread|: {delta_spread:.3f} bps  (tolerance: 5 bps for model retrain noise)")
    print(f"  |Δ ic|:     {delta_ic:.4f}      (tolerance: 0.005)")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    bdf.to_csv(out / "replay_paper_bot.csv", index=False)

    if delta_spread < 5 and delta_ic < 0.005:
        print("\n  ✓  Replay matches backtest within tolerance — live path is consistent.")
        return 0
    else:
        print("\n  ⚠️  Replay deviates from backtest reference by more than tolerance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
