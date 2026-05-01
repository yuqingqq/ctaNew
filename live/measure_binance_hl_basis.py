"""Phase 0b: Measure Binance↔Hyperliquid basis risk for v6_clean's universe.

Hypothesis: v6_clean predicts cross-sectional alpha computed from Binance
prices. If we execute on Hyperliquid, realized P&L uses HL prices. Any
persistent price divergence (basis) eats into the alpha capture.

Method:
  1. Pull HL 5-minute kline data for all 25 symbols, ~180 days back.
  2. Align Binance & HL on shared timestamps.
  3. Compute per-symbol: 5min return correlation, daily (h=288) return
     correlation, mean bias bps, std diff bps.
  4. Compute basket-level basis: HL-basket return vs Binance-basket return.
  5. Re-run v6_clean's portfolio_pnl on HL-prices' realized returns vs
     Binance-prices' realized returns — direct test of how much Sharpe
     transfers.

Output: outputs/binance_hl_basis_*.csv + summary table to stdout.
"""
from __future__ import annotations

import gc
import logging
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_collectors.hl_data_fetcher import HyperliquidDataFetcher
from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES, add_basket_features,
    add_engineered_flow_features, add_xs_rank_features, build_basket,
    build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train, _holdout_split, _slice,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ml/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _binance_to_hl_coin(symbol: str) -> str:
    sym = symbol.upper()
    for suffix in ("USDT", "USDC", "USD", "PERP"):
        if sym.endswith(suffix):
            return sym[: -len(suffix)]
    return sym


def fetch_hl_klines(symbol: str, start: datetime, end: datetime,
                     interval: str = "5m") -> pd.DataFrame:
    """Fetch HL klines, cache to data/ml/cache/hl_klines_{symbol}.parquet.

    Cache covers a [start, end] window. If cache exists and covers the
    requested range, reuse it; otherwise refetch the full range."""
    cache_path = CACHE_DIR / f"hl_klines_{symbol}_{interval}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if not df.empty and df.index.min() <= start and df.index.max() >= end - timedelta(hours=1):
            return df.loc[start:end]

    fetcher = HyperliquidDataFetcher()
    df = fetcher.fetch_range(symbol=symbol, interval=interval,
                              start_time=start, end_time=end)
    if df.empty:
        log.warning("HL %s: empty kline response", symbol)
        return df
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif "open_time" in df.columns:
        df = df.set_index("open_time")
    df = df.sort_index()
    df.to_parquet(cache_path, compression="zstd")
    log.info("HL %s cached %d rows (%s → %s)", symbol, len(df),
              df.index.min(), df.index.max())
    return df


def per_symbol_basis(binance_close: pd.Series, hl_close: pd.Series,
                      symbol: str) -> dict:
    """Align two series, compute basis statistics on aligned 5min returns.

    Returns dict of stats. NaN-handled per-bar.
    """
    aligned = pd.DataFrame({"b": binance_close, "h": hl_close}).dropna()
    if len(aligned) < 1000:
        return {"symbol": symbol, "n_aligned_bars": len(aligned), "insufficient": True}

    b_ret = aligned["b"].pct_change()
    h_ret = aligned["h"].pct_change()
    valid = b_ret.notna() & h_ret.notna()
    b_ret = b_ret[valid]
    h_ret = h_ret[valid]

    # 5min return stats
    corr_5m = b_ret.corr(h_ret)
    diff_5m = h_ret - b_ret
    mean_diff_5m_bps = diff_5m.mean() * 1e4
    std_diff_5m_bps = diff_5m.std() * 1e4

    # Daily (h=288) return stats — directly the v6_clean horizon
    b_d = aligned["b"].pct_change(HORIZON)
    h_d = aligned["h"].pct_change(HORIZON)
    valid_d = b_d.notna() & h_d.notna()
    corr_d = b_d[valid_d].corr(h_d[valid_d])
    diff_d_bps = (h_d - b_d)[valid_d] * 1e4
    mean_diff_d_bps = diff_d_bps.mean()
    std_diff_d_bps = diff_d_bps.std()
    rmse_d_bps = np.sqrt(((h_d - b_d) ** 2)[valid_d].mean()) * 1e4

    # Price spread (% diff)
    spread_pct = ((aligned["h"] / aligned["b"] - 1.0) * 100).abs()

    return {
        "symbol": symbol,
        "n_aligned_bars": len(aligned),
        "corr_5m_returns": corr_5m,
        "corr_daily_returns": corr_d,
        "mean_diff_daily_bps": mean_diff_d_bps,
        "std_diff_daily_bps": std_diff_d_bps,
        "rmse_daily_bps": rmse_d_bps,
        "abs_price_spread_mean_pct": spread_pct.mean(),
        "abs_price_spread_max_pct": spread_pct.max(),
        "insufficient": False,
    }


def main():
    universe = list_universe(min_days=200)
    log.info("Universe: %d symbols", len(universe))

    # Determine date range from Binance cache: use last ~6 months
    binance_feats = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            binance_feats[s] = f
    end = max(f.index.max() for f in binance_feats.values())
    start = end - timedelta(days=180)
    log.info("Pulling HL klines for date range %s → %s", start.date(), end.date())

    # Fetch HL klines (with cache)
    hl_closes = {}
    for i, s in enumerate(sorted(binance_feats.keys())):
        log.info("[%d/%d] Fetching HL %s...", i + 1, len(binance_feats), s)
        try:
            df = fetch_hl_klines(s, start=start, end=end, interval="5m")
            if not df.empty and "close" in df.columns:
                hl_closes[s] = df["close"].astype(float)
        except Exception as e:
            log.error("HL %s fetch failed: %s", s, e)
        time.sleep(0.3)  # be polite to HL API

    log.info("Got HL klines for %d/%d symbols", len(hl_closes), len(binance_feats))

    # ====================================================================
    # CHECK A: Per-symbol basis statistics
    # ====================================================================
    print("\n" + "=" * 100)
    print("A. PER-SYMBOL BASIS (Binance↔Hyperliquid, 5min closes, ~180d)")
    print("=" * 100)
    rows = []
    for s in sorted(hl_closes.keys()):
        b_close = binance_feats[s]["close"]
        h_close = hl_closes[s]
        # Make timezones consistent — Binance uses UTC tz-aware
        if h_close.index.tz is None:
            h_close.index = h_close.index.tz_localize("UTC")
        if b_close.index.tz is None:
            b_close.index = b_close.index.tz_localize("UTC")
        stats = per_symbol_basis(b_close, h_close, s)
        rows.append(stats)
    df_stats = pd.DataFrame(rows)
    df_stats = df_stats.sort_values("rmse_daily_bps", ascending=True, na_position="last")
    print(df_stats.round(4).to_string(index=False))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(out / "binance_hl_basis_per_symbol.csv", index=False)

    # ====================================================================
    # CHECK B: Basket-level basis
    # ====================================================================
    print("\n" + "=" * 100)
    print("B. BASKET-LEVEL BASIS (equal-weight 25-symbol basket)")
    print("=" * 100)
    # Align all symbols' closes on common timestamps
    common_idx = None
    for s in sorted(hl_closes.keys()):
        b = binance_feats[s]["close"]
        h = hl_closes[s]
        if h.index.tz is None:
            h.index = h.index.tz_localize("UTC")
        if b.index.tz is None:
            b.index = b.index.tz_localize("UTC")
        idx = b.index.intersection(h.index)
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    log.info("Common timestamp grid: %d bars", len(common_idx))

    b_closes = pd.DataFrame({s: binance_feats[s]["close"].reindex(common_idx)
                              for s in sorted(hl_closes.keys())})
    h_closes = pd.DataFrame({s: hl_closes[s].reindex(common_idx)
                              for s in sorted(hl_closes.keys())})
    b_basket_ret = b_closes.pct_change().mean(axis=1)
    h_basket_ret = h_closes.pct_change().mean(axis=1)
    aligned = pd.DataFrame({"b": b_basket_ret, "h": h_basket_ret}).dropna()
    corr = aligned["b"].corr(aligned["h"])
    diff = (aligned["h"] - aligned["b"]) * 1e4
    print(f"  Basket return correlation (5min): {corr:.5f}")
    print(f"  Basket return diff (HL - Binance):")
    print(f"    mean    = {diff.mean():+.3f} bps/5min")
    print(f"    std     = {diff.std():.3f} bps/5min")
    print(f"    Daily aggregated mean: {diff.mean() * 288:+.2f} bps/day")
    print(f"    Daily aggregated std:  {diff.std() * np.sqrt(288):.2f} bps/day")

    # ====================================================================
    # CHECK C: P&L on HL prices (realized) using v6_clean predictions
    #          built from Binance features.
    # ====================================================================
    print("\n" + "=" * 100)
    print("C. v6_clean PORTFOLIO P&L on HL realized returns vs Binance realized returns")
    print("=" * 100)
    print("\nBuild v6_clean panel from BINANCE features and labels (training-side).")
    print("For each test cycle, also compute HL-realized return from HL closes.")
    print("Recompute portfolio P&L using HL returns. Diff = basis-attributable PnL.\n")

    # Build the training panel like alpha_v6_clean_pnl_audit does
    closes_df = pd.DataFrame({s: f["close"] for s, f in binance_feats.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes_df)
    sym_to_id = {s: i for i, s in enumerate(sorted(binance_feats.keys()))}
    enriched = {}
    for s, f in binance_feats.items():
        f = f.reindex(closes_df.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN) + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
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
    del enriched, binance_feats
    gc.collect()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames, labels
    gc.collect()
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel[rank_cols] = panel[rank_cols].astype("float32")
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    log.info("panel: %d rows", len(panel))

    # Train on holdout fold
    fold = _holdout_split(panel)[0]
    train, cal, test = _slice(panel, fold)
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    test_f = test
    log.info("train: %d, cal: %d, test: %d", len(train_f), len(cal_f), len(test_f))

    feat_cols = list(XS_FEATURE_COLS_V6_CLEAN)
    X_train = train_f[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_f["demeaned_target"].to_numpy(dtype=np.float32)
    X_cal = cal_f[feat_cols].to_numpy(dtype=np.float32)
    y_cal = cal_f["demeaned_target"].to_numpy(dtype=np.float32)
    log.info("training v6_clean ensemble (5 seeds)...")
    models = [_train(X_train, y_train, X_cal, y_cal, seed=seed)
              for seed in ENSEMBLE_SEEDS]

    # Predict on test
    X_test = test_f[feat_cols].to_numpy(dtype=np.float32)
    yt = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0)

    # Compute HL-realized return per (symbol, open_time) for test rows
    log.info("Computing HL-realized forward returns for test bars...")
    test_f = test_f.copy()
    test_f["return_pct_hl"] = np.nan
    for s, g in test_f.groupby("symbol"):
        if s not in hl_closes:
            continue
        h = hl_closes[s]
        if h.index.tz is None:
            h.index = h.index.tz_localize("UTC")
        # forward return: (close[t+h] - close[t]) / close[t]
        h_fwd = h.pct_change(HORIZON).shift(-HORIZON)
        # match by open_time
        mapped = g["open_time"].map(h_fwd)
        test_f.loc[g.index, "return_pct_hl"] = mapped.values

    coverage = test_f["return_pct_hl"].notna().mean()
    log.info("HL realized return coverage on test: %.1f%%", 100 * coverage)

    # Run portfolio_pnl_turnover_aware twice: once with Binance returns,
    # once with HL returns (swapped into return_pct).
    test_binance = test_f.copy()
    test_hl = test_f.copy()
    test_hl["return_pct"] = test_hl["return_pct_hl"]
    test_hl = test_hl.dropna(subset=["return_pct"])

    res_b = portfolio_pnl_turnover_aware(test_binance, yt, top_frac=0.2,
                                          sample_every=HORIZON, beta_neutral=True)
    # Re-predict on HL-aligned test (subset) with same model
    yt_hl = np.mean([m.predict(test_hl[feat_cols].to_numpy(dtype=np.float32),
                                num_iteration=m.best_iteration) for m in models], axis=0)
    res_h = portfolio_pnl_turnover_aware(test_hl, yt_hl, top_frac=0.2,
                                          sample_every=HORIZON, beta_neutral=True)
    print(f"  Binance realized (test, β-neutral, K=5):")
    print(f"    n_cycles:        {res_b['n_bars']}")
    print(f"    spread_ret:      {res_b['spread_ret_bps_mean']:+.2f} bps/cycle")
    print(f"    cost @ 12 bps:   {res_b['cost_bps_mean']:.2f} bps/cycle")
    print(f"    net @ VIP-0:     {res_b['net_bps_mean']:+.2f} bps/cycle")
    print(f"    rank_ic_mean:    {res_b['rank_ic_mean']:+.4f}")
    print(f"\n  HL realized (test, β-neutral, K=5):")
    print(f"    n_cycles:        {res_h['n_bars']}")
    print(f"    spread_ret:      {res_h['spread_ret_bps_mean']:+.2f} bps/cycle")
    print(f"    cost @ 12 bps:   {res_h['cost_bps_mean']:.2f} bps/cycle")
    print(f"    net @ VIP-0:     {res_h['net_bps_mean']:+.2f} bps/cycle")
    print(f"    rank_ic_mean:    {res_h['rank_ic_mean']:+.4f}")
    print(f"\n  Δ (HL - Binance):")
    print(f"    spread_ret:    {res_h['spread_ret_bps_mean'] - res_b['spread_ret_bps_mean']:+.2f} bps/cycle  (basis-attributable alpha loss)")
    print(f"    rank_ic:       {res_h['rank_ic_mean'] - res_b['rank_ic_mean']:+.4f}  (basis-attributable IC loss)")

    # Save merged comparison
    summary = {
        "n_cycles_b": res_b["n_bars"],
        "n_cycles_hl": res_h["n_bars"],
        "spread_ret_b": res_b["spread_ret_bps_mean"],
        "spread_ret_hl": res_h["spread_ret_bps_mean"],
        "cost_b": res_b["cost_bps_mean"],
        "cost_hl": res_h["cost_bps_mean"],
        "net_b": res_b["net_bps_mean"],
        "net_hl": res_h["net_bps_mean"],
        "ic_b": res_b["rank_ic_mean"],
        "ic_hl": res_h["rank_ic_mean"],
        "delta_spread_bps": res_h["spread_ret_bps_mean"] - res_b["spread_ret_bps_mean"],
        "delta_ic": res_h["rank_ic_mean"] - res_b["rank_ic_mean"],
    }
    pd.DataFrame([summary]).to_csv(out / "binance_hl_basis_pnl.csv", index=False)
    print(f"\n  Output: outputs/binance_hl_basis_*.csv")


if __name__ == "__main__":
    main()
