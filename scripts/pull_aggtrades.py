"""Download Binance Vision aggTrades for the top-10 most-liquid USDM perps.

Phase 3 of the signal-quality plan: enables true microstructure features
(TFI, VPIN, Kyle's λ, signed-volume z-score, Roll's effective spread) at
the 5min bar level.

Run from repo root:
    python3 -m scripts.pull_aggtrades

Estimated runtime: ~5-8 hours with 8 parallel workers per symbol.
Estimated raw disk: ~50-80 GB across all symbols.

Once raw daily parquets are pulled, run:
    python3 -m scripts.build_aggtrade_features
to compute 5min bar features and (optionally) drop the raw files.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from data_collectors.binance_vision_loader import LoaderConfig, list_aggtrade_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Top 10 most-liquid USDM perps that overlap with the v6 universe.
# These have the highest trade frequency, so trade-flow features
# will be most reliable.
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]

# Date range: matches kline coverage so feature alignment is straightforward.
# Trim slightly tighter to avoid pulling days where we don't have klines.
START = datetime(2025, 3, 23).date()
END = datetime(2026, 4, 28).date()
OUT_DIR = Path("data/ml/test")


def main():
    log.info("Pulling aggTrades for %d symbols, %s to %s", len(SYMBOLS), START, END)
    log.info("Output: %s/parquet/aggTrades/<SYMBOL>/", OUT_DIR)
    for sym in SYMBOLS:
        log.info("=== %s ===", sym)
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=8)
        try:
            paths = list_aggtrade_paths(START, END, cfg=cfg)
            log.info("[%s] %d daily parquets ready at %s",
                      sym, len(paths), OUT_DIR / "parquet/aggTrades" / sym)
        except Exception as e:
            log.error("[%s] FAILED: %s", sym, e)


if __name__ == "__main__":
    main()
