"""Falsifiers for the TradFi filter: stocks out, same-ticker crypto in."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movewatch.tradfi import is_tradfi  # noqa: E402

REF = {"AAPL": [317.0], "QNT": [48.5], "META": [729.0], "UVXY": [17.4]}   # venue-labelled stock prices


def test_labelled_stock_at_the_same_price_is_tradfi():
    assert is_tradfi("AAPL", 318.0, REF, has_binance_usdm=True)          # Binance AAPL matches Bitget's label
    assert is_tradfi("UVXY", 17.2, REF, has_binance_usdm=False)          # Bybit/Bitget-only ETF perp


def test_same_ticker_crypto_is_not_tradfi():
    assert not is_tradfi("QNT", 230.0, REF, has_binance_usdm=True)       # Quant vs OKX's QNT stock
    assert not is_tradfi("META", 0.0086, REF, has_binance_usdm=False)    # Metadium (Upbit) vs Meta stock
    assert not is_tradfi("BTC", 82000.0, REF, has_binance_usdm=True)


def test_static_binance_list_is_a_fallback_only_without_a_reference():
    assert is_tradfi("MSTR", 127.0, {}, has_binance_usdm=True)           # verified Binance stock perp
    assert not is_tradfi("MSTR", 127.0, {}, has_binance_usdm=False)      # no Binance instrument: no fallback
    assert not is_tradfi("META", 0.0086, REF, has_binance_usdm=True)     # a reference exists and disagrees
