"""Phase 0b (1h variant): Basis check at hourly resolution.

HL only stores ~15 days of 5min kline history but 180+ days at 1h. Since
v6_clean's holding horizon is 1 day (h=288 5min bars), 1h granularity is
plenty for measuring basis-attributable Sharpe loss.

Method:
  1. Pull 1h HL klines for all 25 symbols, 180 days back.
  2. For each v6_clean test cycle (at 23:55 UTC on day t), find:
     - HL close at hour h_t  (rounded down to top of hour at 23:00 UTC)
     - HL close at h_t + 24h
     - HL_ret = (HL_close[h_t + 24h] / HL_close[h_t]) - 1
  3. Substitute HL_ret for return_pct in v6_clean test panel.
  4. Re-run portfolio_pnl_turnover_aware with HL returns.
  5. Compare Sharpe / IC / spread to Binance baseline.

The 5-minute alignment slippage at cycle entry/exit (max 60 min) is small
relative to a 24-hour holding period.
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
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v4_xs_1d import (
    HORIZON, ENSEMBLE_SEEDS, REGIME_CUTOFF, _train, _holdout_split, _slice,
    _multi_oos_splits,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ml/cache")


def fetch_hl_klines_1h(symbol: str, days: int = 180) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"hl_klines_{symbol}_1h.parquet"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if not df.empty and df.index.min() <= start + timedelta(days=1) and df.index.max() >= end - timedelta(hours=2):
            return df.loc[start:end]
    fetcher = HyperliquidDataFetcher()
    df = fetcher.fetch_range(symbol=symbol, interval="1h", start_time=start, end_time=end)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif "open_time" in df.columns:
        df = df.set_index("open_time")
    df = df.sort_index()
    df.to_parquet(cache_path, compression="zstd")
    log.info("HL %s 1h cached %d rows (%s -> %s)", symbol, len(df),
              df.index.min(), df.index.max())
    return df


def _align_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    """Round down to top of hour."""
    return ts.floor("1h")


def main():
    universe = list_universe(min_days=200)
    log.info("Universe: %d symbols", len(universe))

    binance_feats = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            binance_feats[s] = f

    # Pull HL 1h klines
    hl_closes = {}
    for i, s in enumerate(sorted(binance_feats.keys())):
        log.info("[%d/%d] HL 1h %s...", i + 1, len(binance_feats), s)
        try:
            df = fetch_hl_klines_1h(s, days=180)
            if not df.empty and "close" in df.columns:
                c = df["close"].astype(float)
                if c.index.tz is None:
                    c.index = c.index.tz_localize("UTC")
                hl_closes[s] = c
        except Exception as e:
            log.error("HL %s fetch failed: %s", s, e)
        time.sleep(0.3)
    log.info("Got HL 1h klines for %d/%d symbols", len(hl_closes), len(binance_feats))

    # Build v6_clean panel
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

    X_test = test_f[feat_cols].to_numpy(dtype=np.float32)
    yt = np.mean([m.predict(X_test, num_iteration=m.best_iteration) for m in models], axis=0)

    # Compute HL-realized forward returns at hour-aligned timestamps
    log.info("Computing HL-realized forward returns (1h-aligned)...")
    test_f = test_f.copy()
    test_f["return_pct_hl"] = np.nan
    for s, g in test_f.groupby("symbol"):
        if s not in hl_closes:
            continue
        h = hl_closes[s]
        # For each cycle's open_time t, find HL close at floor_hour(t),
        # and HL close at floor_hour(t + horizon * 5min)
        entry_t = pd.DatetimeIndex(g["open_time"]).floor("1h")
        exit_t = (pd.DatetimeIndex(g["open_time"]) + pd.Timedelta(minutes=HORIZON * 5)).floor("1h")
        # Reindex h to entry/exit times via nearest forward fill
        entry_p = h.reindex(entry_t, method="nearest", tolerance=pd.Timedelta("2h"))
        exit_p = h.reindex(exit_t, method="nearest", tolerance=pd.Timedelta("2h"))
        ret = (exit_p.values / entry_p.values) - 1.0
        test_f.loc[g.index, "return_pct_hl"] = ret

    coverage = test_f["return_pct_hl"].notna().mean()
    log.info("HL realized return coverage on test: %.1f%%", 100 * coverage)
    n_test = test_f["return_pct_hl"].notna().sum()
    log.info("HL realized rows: %d / %d", n_test, len(test_f))

    # Run portfolio_pnl with Binance returns and with HL returns
    test_b = test_f.copy()
    test_h = test_f.copy()
    test_h["return_pct"] = test_h["return_pct_hl"]
    test_h = test_h.dropna(subset=["return_pct"])
    yt_h = np.mean([m.predict(test_h[feat_cols].to_numpy(dtype=np.float32),
                                num_iteration=m.best_iteration) for m in models], axis=0)

    res_b = portfolio_pnl_turnover_aware(test_b, yt, top_frac=0.2,
                                          sample_every=HORIZON, beta_neutral=True)
    res_h = portfolio_pnl_turnover_aware(test_h, yt_h, top_frac=0.2,
                                          sample_every=HORIZON, beta_neutral=True)

    # Bootstrap CI on each
    bdf_b = res_b["df"]
    bdf_h = res_h["df"]
    cycles_per_year = 365.0  # h=288 = 1d
    def _sharpe_yr(arr):
        if arr.std() == 0:
            return 0.0
        return (arr.mean() / arr.std()) * np.sqrt(cycles_per_year)

    sb, sb_lo, sb_hi = block_bootstrap_ci(bdf_b["spread_ret_bps"].to_numpy(),
                                            statistic=_sharpe_yr, block_size=7)
    sh, sh_lo, sh_hi = block_bootstrap_ci(bdf_h["spread_ret_bps"].to_numpy(),
                                            statistic=_sharpe_yr, block_size=7)

    print("\n" + "=" * 100)
    print("BASIS-ADJUSTED P&L: Binance prices vs Hyperliquid prices")
    print("(model trained on Binance, predictions identical, returns swapped)")
    print("=" * 100)
    print()
    print(f"  Holdout fold: {fold['test_start']:%Y-%m-%d} -> {fold['test_end']:%Y-%m-%d}")
    print(f"  HL data window: 180 days at 1h resolution")
    print(f"  Test cycles available on HL: {res_h['n_bars']} / {res_b['n_bars']}")
    print()
    print(f"  Binance realized (β-neutral, K=5):")
    print(f"    n_cycles:      {res_b['n_bars']}")
    print(f"    spread_ret:    {res_b['spread_ret_bps_mean']:+.2f} bps/cycle (gross)")
    print(f"    rank_ic_mean:  {res_b['rank_ic_mean']:+.4f}")
    print(f"    Sharpe (gross): {sb:+.2f}  CI [{sb_lo:+.2f}, {sb_hi:+.2f}]")
    print()
    print(f"  HL realized (β-neutral, K=5):")
    print(f"    n_cycles:      {res_h['n_bars']}")
    print(f"    spread_ret:    {res_h['spread_ret_bps_mean']:+.2f} bps/cycle (gross)")
    print(f"    rank_ic_mean:  {res_h['rank_ic_mean']:+.4f}")
    print(f"    Sharpe (gross): {sh:+.2f}  CI [{sh_lo:+.2f}, {sh_hi:+.2f}]")
    print()
    print(f"  Δ (HL - Binance):")
    delta_spread = res_h['spread_ret_bps_mean'] - res_b['spread_ret_bps_mean']
    delta_ic = res_h['rank_ic_mean'] - res_b['rank_ic_mean']
    print(f"    spread_ret:    {delta_spread:+.2f} bps/cycle")
    print(f"    rank_ic:       {delta_ic:+.4f}")
    print(f"    Sharpe:        {sh - sb:+.2f}")

    # Per-symbol IC on HL vs Binance
    print()
    print(f"  Per-symbol HL realized return statistics (1h-aligned):")
    rows = []
    for s in sorted(hl_closes.keys()):
        g = test_f[test_f["symbol"] == s]
        if g["return_pct_hl"].notna().sum() < 30:
            continue
        b_ret = g["return_pct"].dropna()
        h_ret = g["return_pct_hl"].dropna()
        common = b_ret.index.intersection(h_ret.index)
        if len(common) < 30:
            continue
        b = b_ret.loc[common]
        h = h_ret.loc[common]
        rows.append({
            "symbol": s,
            "n_cycles": len(common),
            "binance_mean_bps": b.mean() * 1e4,
            "hl_mean_bps": h.mean() * 1e4,
            "diff_mean_bps": (h - b).mean() * 1e4,
            "diff_std_bps": (h - b).std() * 1e4,
            "corr": b.corr(h),
        })
    df_per = pd.DataFrame(rows).sort_values("diff_std_bps")
    print(df_per.round(3).to_string(index=False))

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df_per.to_csv(out / "binance_hl_basis_1h_per_symbol.csv", index=False)
    summary = {
        "n_cycles_b": res_b["n_bars"],
        "n_cycles_hl": res_h["n_bars"],
        "spread_ret_b": res_b["spread_ret_bps_mean"],
        "spread_ret_hl": res_h["spread_ret_bps_mean"],
        "ic_b": res_b["rank_ic_mean"],
        "ic_hl": res_h["rank_ic_mean"],
        "sharpe_b": sb, "sharpe_b_lo": sb_lo, "sharpe_b_hi": sb_hi,
        "sharpe_hl": sh, "sharpe_hl_lo": sh_lo, "sharpe_hl_hi": sh_hi,
    }
    pd.DataFrame([summary]).to_csv(out / "binance_hl_basis_1h_pnl.csv", index=False)
    print(f"\n  Output: outputs/binance_hl_basis_1h_*.csv")


if __name__ == "__main__":
    main()
