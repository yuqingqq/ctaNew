"""Instrument identity across venues.

An Instrument is one tradable market on one venue. Detection runs per *asset*
(normalized base, e.g. "PEPE"), so every venue symbol must map to a base and a
price multiplier (1000PEPEUSDT quotes 1000 PEPE; Hyperliquid kPEPE likewise).

Same ticker is NOT same asset. Verified 2026-09-28: OKX QNT-USDT-SWAP is an
equity perp (instCategory 3) trading ~$48 while Quant (QNT) was ~$220. Two
guards: venue metadata filters (OKX instCategory == 1) and a runtime
cross-venue price-consistency check (store.consistency) that marks an
instrument ALIAS_MISMATCH when its USD price sits far from the asset's
cross-venue median. Exclusions are statuses, never silent drops.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# USD-like quote currencies treated as 1:1 with USD for volume weights and the
# price-consistency check (never for P&L).
USD_QUOTES = {"USDT", "USD", "USDC", "FDUSD"}
FIAT_QUOTES = {"KRW"}  # converted through a live KRW/USDT rate

# Bases never alerted by the pump/dump detector (a depeg monitor would need its
# own thresholds: 1% on a stable is an extreme move).
STABLE_BASES = {
    "USDT", "USDC", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "USDD", "PYUSD", "USDE", "USDS",
    "EUR", "EURI", "EURC", "AEUR", "TRY", "BRL", "GBP", "JPY", "USD1", "RLUSD", "USDG", "USDQ",
    "XUSD", "BFUSD", "FRAX", "LUSD", "GUSD", "SUSD", "USTC", "UST", "PAXG", "XAUT",
}

_MULT_PREFIX = re.compile(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])")
_MULT_SUFFIX = re.compile(r"(?<=[A-Z])(1000000|100000|10000|1000)$")
_MULT = {"1000000": 1e6, "100000": 1e5, "10000": 1e4, "1000": 1e3, "1M": 1e6}

# Bases whose ticker legitimately starts with digits (never strip these).
_DIGIT_BASES = {"1INCH", "1000SATS", "1MBABYDOGE"}


@dataclass(frozen=True)
class Instrument:
    venue: str          # binance_spot, binance_usdm, bybit_linear, bybit_spot, okx_spot, okx_swap,
                        # coinbase, upbit, bithumb, bitget_spot, bitget_fut, hyperliquid
    symbol: str         # venue-native id used on the wire
    base: str           # normalized asset, e.g. "PEPE"
    quote: str          # USDT / USD / USDC / FDUSD / KRW
    kind: str           # "spot" | "perp"
    mult: float = 1.0   # native price = mult * price of one base unit

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.symbol}"


def split_multiplier(raw_base: str) -> tuple[str, float]:
    """'1000PEPE' -> ('PEPE', 1000); 'SHIB1000' -> ('SHIB', 1000); 'kPEPE' -> ('PEPE', 1000)."""
    if raw_base in _DIGIT_BASES:
        if raw_base == "1000SATS":
            return "SATS", 1e3
        if raw_base == "1MBABYDOGE":
            return "BABYDOGE", 1e6
        return raw_base, 1.0
    if len(raw_base) > 1 and raw_base[0] == "k" and raw_base[1:].isupper():  # Hyperliquid kPEPE
        return raw_base[1:], 1e3
    m = _MULT_PREFIX.match(raw_base)
    if m:
        return raw_base[m.end():], _MULT[m.group(1)]
    m = _MULT_SUFFIX.search(raw_base)
    if m:
        return raw_base[: m.start()], _MULT[m.group(1)]
    return raw_base, 1.0


def split_concat_symbol(sym: str, quotes=("USDT", "USDC", "FDUSD", "USD")) -> tuple[str, str] | None:
    """'QNTUSDT' -> ('QNT', 'USDT'). Returns None for dated futures / unknown quotes."""
    if "_" in sym or "-" in sym:
        return None
    for q in quotes:
        if sym.endswith(q) and len(sym) > len(q):
            return sym[: -len(q)], q
    return None


def make(venue: str, symbol: str, raw_base: str, quote: str, kind: str) -> Instrument | None:
    """Build a normalized Instrument, or None if it is out of scope (stable base,
    unsupported quote)."""
    quote = quote.upper()
    if quote not in USD_QUOTES and quote not in FIAT_QUOTES:
        return None
    base, mult = split_multiplier(raw_base)
    base = base.upper()
    if not base or base in STABLE_BASES:
        return None
    return Instrument(venue=venue, symbol=symbol, base=base, quote=quote, kind=kind, mult=mult)


@dataclass
class FX:
    """Quote-currency -> USD conversion. KRW is updated live from Upbit/Bithumb
    KRW-USDT trades; until the first print a conservative static rate is used
    (flagged in health output)."""
    krw_per_usd: float = 1400.0
    krw_live: bool = False
    rates: dict = field(default_factory=dict)

    def to_usd(self, quote: str) -> float:
        if quote in USD_QUOTES:
            return 1.0
        if quote == "KRW":
            return 1.0 / self.krw_per_usd
        return float("nan")

    def set_krw(self, krw_per_usdt: float) -> None:
        if krw_per_usdt > 0:
            self.krw_per_usd = krw_per_usdt
            self.krw_live = True
