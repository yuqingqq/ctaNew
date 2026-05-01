"""Phase 0a: Verify Hyperliquid lists all 25 symbols in v6_clean's universe.

Hyperliquid uses coin names (BTC, ETH, ADA) instead of Binance pair names
(BTCUSDT etc.). Their info API returns the full perp universe via
POST /info {"type": "meta"} → universe[].name.

We pull that list, map v6_clean's 25 symbols to HL coin names, and
report:
  - Symbols present on HL (good)
  - Symbols missing on HL (must drop from universe OR retrain v6_clean)
  - Per-symbol max leverage on HL (info for execution sizing)

Output: outputs/hl_universe_check.csv with one row per v6_clean symbol.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from features_ml.cross_sectional import list_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _binance_to_hl_coin(symbol: str) -> str:
    """ETHUSDT -> ETH. Strip USDT/USDC/USD/PERP suffix."""
    sym = symbol.upper()
    for suffix in ("USDT", "USDC", "USD", "PERP"):
        if sym.endswith(suffix):
            return sym[: -len(suffix)]
    return sym


def fetch_hl_perp_universe() -> pd.DataFrame:
    """Hyperliquid info.meta() returns the full perp universe."""
    r = requests.post(HL_INFO_URL, json={"type": "meta"}, timeout=15)
    r.raise_for_status()
    payload = r.json()
    rows = []
    for entry in payload["universe"]:
        rows.append({
            "hl_coin": entry["name"],
            "hl_max_leverage": entry.get("maxLeverage"),
            "hl_only_isolated": entry.get("onlyIsolated", False),
            "hl_sz_decimals": entry.get("szDecimals"),
        })
    return pd.DataFrame(rows)


def main():
    log.info("Fetching Hyperliquid perp universe via info.meta()...")
    hl = fetch_hl_perp_universe()
    log.info("HL lists %d perpetual coins", len(hl))

    universe = list_universe(min_days=200)
    log.info("v6_clean Binance universe: %d symbols", len(universe))

    rows = []
    for binance_sym in universe:
        hl_coin = _binance_to_hl_coin(binance_sym)
        match = hl[hl["hl_coin"] == hl_coin]
        if match.empty:
            rows.append({
                "binance_symbol": binance_sym,
                "hl_coin": hl_coin,
                "on_hl": False,
                "hl_max_leverage": None,
                "hl_sz_decimals": None,
            })
        else:
            r = match.iloc[0]
            rows.append({
                "binance_symbol": binance_sym,
                "hl_coin": hl_coin,
                "on_hl": True,
                "hl_max_leverage": r["hl_max_leverage"],
                "hl_sz_decimals": r["hl_sz_decimals"],
            })

    df = pd.DataFrame(rows).sort_values("on_hl", ascending=False)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hl_universe_check.csv", index=False)

    print("\n" + "=" * 80)
    print("HYPERLIQUID UNIVERSE COVERAGE CHECK (v6_clean's 25 Binance symbols)")
    print("=" * 80)
    print(df.to_string(index=False))

    on_hl = df[df["on_hl"]]
    missing = df[~df["on_hl"]]
    print(f"\n  Coverage: {len(on_hl)}/{len(df)} symbols available on Hyperliquid")
    if len(missing):
        print(f"  Missing: {sorted(missing['binance_symbol'].tolist())}")
        print(f"\n  ⚠️  v6_clean cannot trade these on HL. Options:")
        print(f"     - Drop from universe and retrain (changes basket; need re-validation)")
        print(f"     - Trade them on Binance, others on HL (split execution)")
        print(f"     - Skip these names entirely at execution (unactionable predictions)")
    else:
        print(f"  ✓  All 25 symbols available on Hyperliquid")

    print(f"\n  Output: outputs/hl_universe_check.csv")


if __name__ == "__main__":
    sys.exit(main())
