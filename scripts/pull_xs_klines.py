"""Download Binance Vision 5m klines for the cross-sectional universe.

Run from repo root:
    python3 scripts/pull_xs_klines.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from data_collectors.binance_vision_loader import LoaderConfig, fetch_klines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Top liquid Binance USDM perps that aren't already pulled.
SYMBOLS = [
    "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "NEARUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT",
    "ARBUSDT", "OPUSDT", "APTUSDT", "INJUSDT",
    "SUIUSDT", "SEIUSDT", "TIAUSDT", "RUNEUSDT", "WLDUSDT",
]

START = datetime(2025, 3, 23).date()
END = datetime(2026, 4, 28).date()
OUT_DIR = Path("data/ml/test")


def main():
    for sym in SYMBOLS:
        log.info("=== %s ===", sym)
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=8)
        try:
            df = fetch_klines(START, END, interval="5m", cfg=cfg)
            log.info("[%s] %d rows %s -> %s", sym, len(df),
                      df["open_time"].min() if not df.empty else "EMPTY",
                      df["open_time"].max() if not df.empty else "EMPTY")
        except Exception as e:
            log.error("[%s] FAILED: %s", sym, e)


if __name__ == "__main__":
    main()
