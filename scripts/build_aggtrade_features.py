"""Compute 5min trade-flow features from cached aggTrades parquets.

Phase 3 of the signal-quality plan. After scripts/pull_aggtrades.py has
pulled the daily aggTrades parquets, this script:

1. For each symbol, streams the daily parquets through
   `features_ml.trade_flow.aggregate_trades_streaming` to produce a
   bar-level feature DataFrame.
2. Caches the features as `data/ml/cache/flow_<SYMBOL>.parquet`.

Output features per 5min bar:
  - buy_volume, sell_volume, signed_volume, tfi
  - buy_count, sell_count, aggressor_count_ratio
  - avg_trade_size, max_trade_size, large_trade_volume, large_trade_count
  - vwap, vwap_dev_bps, kyle_lambda
  - vpin (rolling 50-bucket VPIN)
  - tfi_smooth_12 (EMA-12 of tfi)
  - signed_volume_z (12-bar z-score of signed_volume)

Run from repo root:
    python3 -m scripts.build_aggtrade_features

Resumable: skips symbols whose flow_<SYMBOL>.parquet already exists. To
force rebuild, delete the cache file or pass --force.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from features_ml.trade_flow import TradeFlowConfig, aggregate_trades_streaming

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/ml/test/parquet")
CACHE_DIR = Path("data/ml/cache")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    p.add_argument("--symbols", nargs="*", default=SYMBOLS,
                    help="override symbol list")
    p.add_argument("--no-kyle", action="store_true",
                    help="skip Kyle's λ (per-bar regression — slow)")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TradeFlowConfig(bar_interval="5min",
                            compute_kyle_lambda=not args.no_kyle)

    for sym in args.symbols:
        cache = CACHE_DIR / f"flow_{sym}.parquet"
        if cache.exists() and not args.force:
            log.info("[%s] cached %s; skipping (--force to rebuild)", sym, cache)
            continue
        sym_dir = DATA_DIR / "aggTrades" / sym
        if not sym_dir.exists():
            log.warning("[%s] no aggTrades dir at %s; skipping", sym, sym_dir)
            continue
        paths = sorted(sym_dir.glob("*.parquet"))
        if not paths:
            log.warning("[%s] no daily parquets in %s", sym, sym_dir)
            continue
        log.info("[%s] processing %d daily parquets...", sym, len(paths))
        try:
            feats = aggregate_trades_streaming(paths, cfg)
        except Exception as e:
            log.error("[%s] failed: %s", sym, e)
            continue
        # Drop the rolling-window NaN warmup
        feats = feats.dropna(subset=["vpin"], how="all")
        log.info("[%s] features: %d rows from %s to %s",
                  sym, len(feats),
                  feats.index.min() if len(feats) else "EMPTY",
                  feats.index.max() if len(feats) else "EMPTY")
        feats.to_parquet(cache, compression="zstd")
        log.info("[%s] cached → %s (%d MB)",
                  sym, cache, cache.stat().st_size // (1024 * 1024))


if __name__ == "__main__":
    main()
