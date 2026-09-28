"""Venue registry. `venues:` in config/movewatch.yaml enables/disables each feed."""
from __future__ import annotations

import logging

from .base import http_json
from .binance import BinanceSpot, BinanceUsdm
from .bitget import BitgetFut, BitgetSpot
from .bybit import BybitLinear, BybitSpot, linear_archive_candidates
from .coinbase import Coinbase
from .hyperliquid import Hyperliquid
from .korea import Bithumb, Upbit
from .okx import OkxSpot, OkxSwap

log = logging.getLogger("movewatch.venues")

SIMPLE = {
    "binance_usdm": BinanceUsdm, "binance_spot": BinanceSpot,
    "okx_swap": OkxSwap, "okx_spot": OkxSpot, "coinbase": Coinbase,
    "upbit": Upbit, "bithumb": Bithumb, "bitget_fut": BitgetFut, "bitget_spot": BitgetSpot,
    "hyperliquid": Hyperliquid,
}
ALL = list(SIMPLE) + ["bybit_linear", "bybit_spot"]


async def build_feeds(cfg: dict, store, session) -> list:
    vcfg = cfg.get("venues", {})
    only = cfg.get("only_venues")
    enabled = [v for v in ALL if vcfg.get(v, {}).get("enabled", True) and (not only or v in only)]
    feeds = []
    for v in enabled:
        if v in SIMPLE:
            feeds.append(SIMPLE[v](store, vcfg.get(v, {})))
    if "bybit_linear" in enabled:
        try:
            cands = await linear_archive_candidates(session)
        except Exception as e:  # noqa: BLE001
            log.warning("bybit linear candidates: %s", e)
            cands = []
        feeds.append(BybitLinear(store, vcfg.get("bybit_linear", {}), candidates=cands))
    if "bybit_spot" in enabled:
        try:
            px = await http_json(session, "GET", "https://data-api.binance.vision/api/v3/ticker/price")
            cands = [p["symbol"] for p in px if p["symbol"].endswith("USDT")]
        except Exception as e:  # noqa: BLE001
            log.warning("bybit spot candidates: %s", e)
            cands = []
        feeds.append(BybitSpot(store, vcfg.get("bybit_spot", {}), candidates=cands))
    return feeds
