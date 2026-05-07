"""Download Binance Vision 5m klines for the cross-sectional universe.

Run from repo root:
    python3 scripts/pull_xs_klines.py
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_collectors.binance_vision_loader import LoaderConfig, fetch_klines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ORIG25 — the 25 BNF perps that v6_clean was selected on.
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "NEARUSDT", "UNIUSDT", "ATOMUSDT", "FILUSDT",
    "ARBUSDT", "OPUSDT", "APTUSDT", "INJUSDT",
    "SUIUSDT", "SEIUSDT", "TIAUSDT", "RUNEUSDT", "WLDUSDT",
]

START = datetime(2025, 3, 23).date()
# END = yesterday UTC. Vision archives publish with ~1-day lag; today's file
# is usually missing. Dynamic so the weekly cron picks up new bars each run.
END = (datetime.now(timezone.utc) - timedelta(days=1)).date()
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
