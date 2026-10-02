"""Stock / ETF / commodity perps ("TradFi") are not crypto assets: exclude them from detection,
the market median and breadth (the research excluded them too; research/common.py TRADFI).

Venues label them differently:
  - Bitget futures: contract field isRwa == "YES" (339 of 804 contracts, 2026-09-28)
  - OKX: instCategory 3 (stocks) / 4 (commodities); the feed already keeps only category 1
  - Binance USDM / Bybit: no metadata reachable from this box
So the rule is price-based, reusing the same-ticker price clusters: an asset cluster is TradFi
if a LABELLED instrument with the same base trades within 25% (log 0.25) of the cluster's USD
price. Ticker collisions stay crypto (OKX's QNT stock ~$48 vs Quant ~$230; Upbit's Metadium
"META" ~$0.009 vs the Meta stock perp ~$729). Binance spot lists tokenized stocks as the
stock ticker + "B" (NVDAB, TSLAB, MSTRB: 71 pairs on 2026-10-02, every one within 1% of its
stock); for clusters with a binance_spot instrument the ticker without the "B" is matched too,
by price only. Fallback: the Binance tickers verified by price
level in research/common.py, applied only to clusters that contain a binance_usdm instrument.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger("movewatch.tradfi")

STATIC_BINANCE = {"AAPL", "AMD", "AMZN", "ARM", "AVGO", "BABA", "CL", "COIN", "COPPER", "CRCL", "CRM", "DELL",
                  "GOOGL", "HOOD", "IBM", "INTC", "META", "MSFT", "MSTR", "MU", "NATGAS", "NVDA", "ORCL", "PLTR",
                  "QCOM", "QQQ", "SMCI", "SNOW", "SOXL", "SPY", "TQQQ", "TSLA", "TSM", "WMT", "XAG", "XPD", "XPT"}
MAX_LOG_DEV = 0.25


async def fetch_reference(session) -> dict[str, list[float]]:
    """{base: [USD prices of instruments a venue LABELS as stock/commodity]}."""
    from .venues.base import http_json
    ref: dict[str, list[float]] = {}

    def add(base, px):
        try:
            px = float(px)
        except (TypeError, ValueError):
            return
        if px > 0:
            ref.setdefault(base.upper(), []).append(px)
    try:
        cs = (await http_json(session, "GET", "https://api.bitget.com/api/v2/mix/market/contracts",
                              params={"productType": "USDT-FUTURES"}))["data"]
        tk = (await http_json(session, "GET", "https://api.bitget.com/api/v2/mix/market/tickers",
                              params={"productType": "USDT-FUTURES"}))["data"]
        last = {t["symbol"]: t.get("lastPr") for t in tk}
        for c in cs:
            if c.get("isRwa") == "YES":
                add(c["baseCoin"], last.get(c["symbol"]))
    except Exception as e:  # noqa: BLE001 -- a missing reference is logged; the static list still applies
        log.warning("tradfi ref (bitget): %s", e)
    for inst_type in ("SWAP", "SPOT"):
        try:
            ins = (await http_json(session, "GET", "https://www.okx.com/api/v5/public/instruments",
                                   params={"instType": inst_type}))["data"]
            tk = (await http_json(session, "GET", "https://www.okx.com/api/v5/market/tickers",
                                  params={"instType": inst_type}))["data"]
            last = {t["instId"]: t.get("last") for t in tk}
            for x in ins:
                if x.get("instCategory") in ("3", "4"):
                    base = x["instFamily"].split("-")[0] if inst_type == "SWAP" else x.get("baseCcy", "")
                    add(base, last.get(x["instId"]))
        except Exception as e:  # noqa: BLE001
            log.warning("tradfi ref (okx %s): %s", inst_type, e)
    return ref


def is_tradfi(base: str, usd_px: float, ref: dict[str, list[float]], has_binance_usdm: bool,
              has_binance_spot: bool = False) -> bool:
    """base: the instrument ticker base (not the cluster name); usd_px: the cluster's USD price."""
    b = base.upper()
    cands = [b] + ([b[:-1]] if has_binance_spot and b.endswith("B") and len(b) > 2 else [])
    if usd_px == usd_px and usd_px > 0:
        for c in cands:
            prices = ref.get(c)
            if prices and min(abs(math.log(usd_px / p)) for p in prices) <= MAX_LOG_DEV:
                return True
    return has_binance_usdm and b in STATIC_BINANCE and not ref.get(b)
