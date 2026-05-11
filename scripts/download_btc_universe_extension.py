"""Download 5-min klines for the 17 new symbols added to the alts-vs-BTC pool.

Targets historical range 2025-03-23 → 2026-05-06 (matches existing local data
span). For symbols listed later than 2025-03-23, missing days will 404 and
get skipped — that's fine, we capture whatever's available.

Output: data/ml/test/parquet/klines/{SYMBOL}/5m/{DATE}.parquet (matches existing
local data layout used by features_ml.cross_sectional.list_universe).
"""
from __future__ import annotations
import sys, time, logging
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_collectors.binance_vision_loader import LoaderConfig, fetch_klines

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEW_SYMBOLS = [
    "ZECUSDT", "ONDOUSDT", "STRKUSDT", "CHIPUSDT", "HYPEUSDT", "TAOUSDT",
    "ENAUSDT", "JTOUSDT", "ASTERUSDT", "PUMPUSDT", "VVVUSDT", "BIOUSDT",
    "JUPUSDT", "PENGUUSDT", "VIRTUALUSDT", "MEGAUSDT", "PENDLEUSDT",
]

START = date(2025, 3, 23)
END = date(2026, 5, 6)
OUT_DIR = REPO / "data/ml/test"


def main():
    print(f"Downloading 5-min klines for {len(NEW_SYMBOLS)} new symbols")
    print(f"  Range: {START} → {END} ({(END - START).days + 1} days)")
    print(f"  Output: {OUT_DIR}")
    print()

    for i, sym in enumerate(NEW_SYMBOLS, 1):
        t0 = time.time()
        cfg = LoaderConfig(symbol=sym, out_dir=OUT_DIR, max_workers=8)
        try:
            df = fetch_klines(START, END, interval="5m", cfg=cfg)
            n_days = df["open_time"].dt.normalize().nunique() if not df.empty else 0
            elapsed = time.time() - t0
            print(f"  [{i:>2}/{len(NEW_SYMBOLS)}] {sym:<14}  "
                  f"rows={len(df):>8,}  days={n_days:>4}  ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  [{i:>2}/{len(NEW_SYMBOLS)}] {sym:<14}  FAILED: {e}")

    # Verify all files landed
    print("\n  Verification — file counts in data dir:")
    for sym in NEW_SYMBOLS:
        d = OUT_DIR / "parquet/klines" / sym / "5m"
        n = len(list(d.glob("*.parquet"))) if d.exists() else 0
        print(f"    {sym:<14}  files={n}")


if __name__ == "__main__":
    main()
