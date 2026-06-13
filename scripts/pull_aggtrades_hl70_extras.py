"""Pull Binance Vision aggTrades for the 20 HL-70-extra syms missing them.

These are in HL-70 (HL-tradeable) but were not in the 51-panel originally.
After download: run build_aggtrade_features to compute flow_<SYM>.parquet.
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

from data_collectors.binance_vision_loader import LoaderConfig, list_aggtrade_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MISSING_HL70_EXTRAS = [
    "AEROUSDT", "AIXBTUSDT", "ANIMEUSDT", "BERAUSDT", "FARTCOINUSDT",
    "GRIFFAINUSDT", "IPUSDT", "KAITOUSDT", "LAYERUSDT", "MELANIAUSDT",
    "MEUSDT", "MOVEUSDT", "NILUSDT", "PAXGUSDT", "SPXUSDT",
    "SUSDT", "TRUMPUSDT", "TSTUSDT", "USUALUSDT", "VINEUSDT",
]

START = datetime(2025, 3, 23).date()
END = datetime(2026, 5, 6).date()
OUT_DIR = Path("/home/yuqing/ctaNew/data/ml/test")


def main():
    log.info("Pulling aggTrades for %d HL-70 EXTRA syms (%s to %s)",
             len(MISSING_HL70_EXTRAS), START, END)
    for i, sym in enumerate(MISSING_HL70_EXTRAS, 1):
        log.info("=== [%d/%d] %s ===", i, len(MISSING_HL70_EXTRAS), sym)
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=8)
        try:
            paths = list_aggtrade_paths(START, END, cfg=cfg)
            log.info("[%s] %d daily parquets ready", sym, len(paths))
        except Exception as e:
            log.error("[%s] FAILED: %s", sym, e)


if __name__ == "__main__":
    main()
