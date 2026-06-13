"""Pull Binance Vision aggTrades for the 26 syms missing from the panel.

These syms are in the 51-panel/HL-50 universe but have no aggTrades data,
causing 51% NaN rate in aggT features and hurting Pool+symid model performance.

Estimated raw disk: ~15-30 GB (smaller-cap syms have fewer trades per day).
Estimated runtime: 1-2 hours with 8 parallel workers per symbol.

After this completes:
1. Run: python3 -m scripts.build_aggtrade_features --symbols <missing 26 syms>
2. Rebuild panel with new flow_<sym>.parquet files
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

from data_collectors.binance_vision_loader import LoaderConfig, list_aggtrade_paths

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# The 26 syms in the 51-panel that have ZERO aggT coverage currently
MISSING_SYMBOLS = [
    "AAVEUSDT", "ASTERUSDT", "AXSUSDT", "BIOUSDT", "ENAUSDT",
    "ETCUSDT", "GMXUSDT", "HBARUSDT", "HYPEUSDT", "ICPUSDT",
    "JTOUSDT", "JUPUSDT", "LDOUSDT", "ONDOUSDT", "ORDIUSDT",
    "PENDLEUSDT", "TAOUSDT", "TONUSDT", "TRBUSDT", "VIRTUALUSDT",
    "ARBUSDT_check",  # placeholder removed below if already present
]
# Actually check which 26 are in panel without aggT (re-derive at runtime)


def get_missing_syms():
    """Re-derive missing syms by checking the existing aggTrades dir + panel."""
    import pandas as pd
    REPO = Path("/home/yuqing/ctaNew")
    aggT_dir = REPO / "data/ml/test/parquet/aggTrades"
    existing = set(p.name for p in aggT_dir.iterdir() if p.is_dir())

    panel_syms_file = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
    syms_in_panel = set(pd.read_parquet(panel_syms_file, columns=["symbol"])["symbol"].unique())

    missing = sorted(syms_in_panel - existing - {"BTCUSDT"})  # BTC always present
    log.info("Panel syms: %d, existing aggT: %d, missing: %d",
             len(syms_in_panel), len(existing), len(missing))
    log.info("Missing: %s", missing)
    return missing


# Date range
START = datetime(2025, 3, 23).date()
END = datetime(2026, 5, 6).date()
OUT_DIR = Path("/home/yuqing/ctaNew/data/ml/test")


def main():
    missing = get_missing_syms()
    log.info("Pulling aggTrades for %d MISSING syms, %s to %s",
             len(missing), START, END)
    log.info("Output: %s/parquet/aggTrades/<SYMBOL>/", OUT_DIR)
    for i, sym in enumerate(missing, 1):
        log.info("=== [%d/%d] %s ===", i, len(missing), sym)
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=8)
        try:
            paths = list_aggtrade_paths(START, END, cfg=cfg)
            log.info("[%s] %d daily parquets ready", sym, len(paths))
        except Exception as e:
            log.error("[%s] FAILED: %s", sym, e)


if __name__ == "__main__":
    main()
