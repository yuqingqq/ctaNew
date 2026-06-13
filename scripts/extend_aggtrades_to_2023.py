"""Extend Binance PERP aggTrades back to 2023-01 via Vision daily archives,
then build flow features over the full range.

This is the big download (aggTrades are large). Uses the existing
binance_vision_loader (futures/um/daily aggTrades).
"""
from __future__ import annotations
import sys, logging
from datetime import date
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from data_collectors.binance_vision_loader import LoaderConfig, list_aggtrade_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

START = date(2023, 1, 1)
END = date(2025, 3, 22)   # pre-period; recent already cached
OUT_DIR = REPO/"data/ml/test"

CANDIDATES = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","LINKUSDT","DOTUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","NEARUSDT",
    "UNIUSDT","TIAUSDT","SUIUSDT","SEIUSDT","INJUSDT","ARBUSDT","APTUSDT","OPUSDT",
    "AAVEUSDT","AXSUSDT","FILUSDT","ETCUSDT","TRBUSDT","WLDUSDT","ICPUSDT","ONDOUSDT",
    "PENDLEUSDT","LDOUSDT","JTOUSDT","ENAUSDT","HBARUSDT","TONUSDT","STRKUSDT",
    "WIFUSDT","ORDIUSDT","JUPUSDT","GMXUSDT","TAOUSDT","RUNEUSDT","SUSDT","ZECUSDT",
]


def main():
    log.info("Extending aggTrades to 2023-01 for %d syms (%s → %s)", len(CANDIDATES), START, END)
    for i, sym in enumerate(CANDIDATES, 1):
        log.info("=== [%d/%d] %s ===", i, len(CANDIDATES), sym)
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=8)
        try:
            paths = list_aggtrade_paths(START, END, cfg=cfg)
            log.info("[%s] %d daily aggtrade parquets ready", sym, len(paths))
        except Exception as e:
            log.error("[%s] FAILED: %s", sym, e)
    log.info("Done aggTrades download. Run build_aggtrade_features next.")


if __name__ == "__main__":
    main()
