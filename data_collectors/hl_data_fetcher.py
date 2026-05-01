"""
Hyperliquid Data Fetcher
Fetches historical OHLCV data from Hyperliquid API
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from pathlib import Path


class HyperliquidDataFetcher:
    """Fetch historical candle data from Hyperliquid"""

    def __init__(self, base_url: str = "https://api.hyperliquid.xyz"):
        self.base_url = base_url
        self.max_candles_per_request = 5000  # Hyperliquid limit

    def _convert_symbol(self, symbol: str) -> str:
        """
        Convert Binance-style symbol to Hyperliquid coin format.
        e.g., 'ETHUSDT' -> 'ETH', 'BTCUSDT' -> 'BTC', 'SOLUSDT' -> 'SOL'
        """
        # Remove common quote currencies
        for suffix in ['USDT', 'USDC', 'USD', 'PERP']:
            if symbol.upper().endswith(suffix):
                return symbol.upper()[:-len(suffix)]
        return symbol.upper()

    def fetch_candles(
        self,
        symbol: str,
        interval: str = '5m',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch candle data from Hyperliquid

        Args:
            symbol: Trading pair (e.g., 'ETHUSDT' or 'ETH')
            interval: Timeframe ('1m', '5m', '15m', '1h', etc.)
            start_time: Start datetime (None = fetch most recent)
            end_time: End datetime (None = now)

        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/info"
        coin = self._convert_symbol(symbol)

        # Default to now if no end_time
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        # Default start_time based on interval and max candles
        if start_time is None:
            interval_minutes = self._interval_to_minutes(interval)
            start_time = end_time - timedelta(minutes=interval_minutes * self.max_candles_per_request)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            # Parse response - Hyperliquid returns array of candle objects
            records = []
            for candle in data:
                records.append({
                    'timestamp': pd.to_datetime(candle['t'], unit='ms', utc=True),
                    'open': float(candle['o']),
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'close': float(candle['c']),
                    'volume': float(candle['v']),
                })

            df = pd.DataFrame(records)
            if df.empty:
                return df

            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            return df

        except Exception as e:
            print(f"Warning: error fetching {symbol} {interval} from Hyperliquid: {e}")
            return pd.DataFrame()

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = '5m',
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical data by making multiple API calls if needed.
        Note: Hyperliquid only provides the most recent 5000 candles per request.

        Args:
            symbol: Trading pair (e.g., 'ETHUSDT' or 'ETH')
            interval: Timeframe
            days: Number of days of history

        Returns:
            DataFrame with historical data
        """
        print(f"Fetching {days} days of {interval} data for {symbol} from Hyperliquid...")

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        interval_minutes = self._interval_to_minutes(interval)
        candles_needed = (days * 1440) // interval_minutes

        # Check if we can get all data in one request
        if candles_needed <= self.max_candles_per_request:
            df = self.fetch_candles(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            if not df.empty:
                print(f"Fetched {len(df)} candles for {symbol}")
                print(f"  Time range: {df.index[0]} to {df.index[-1]}")
            return df

        # Need multiple requests - fetch in chunks going backwards
        print(f"  Need {candles_needed} candles, fetching in chunks...")
        all_data = []
        current_end = end_time
        request_count = 0

        while current_end > start_time:
            # Calculate chunk start (go back max_candles worth of time)
            chunk_minutes = interval_minutes * self.max_candles_per_request
            chunk_start = max(start_time, current_end - timedelta(minutes=chunk_minutes))

            df_chunk = self.fetch_candles(
                symbol=symbol,
                interval=interval,
                start_time=chunk_start,
                end_time=current_end
            )

            if df_chunk.empty:
                print(f"\n  Warning: No data returned for chunk ending at {current_end}")
                break

            all_data.append(df_chunk)
            request_count += 1

            # Move to next chunk (go earlier in time)
            current_end = df_chunk.index[0] - timedelta(minutes=interval_minutes)

            # Rate limiting
            if request_count % 5 == 0:
                time.sleep(1)

            # Progress indicator
            progress = (end_time - current_end) / (end_time - start_time) * 100
            progress = min(100, progress)
            print(f"  Progress: {progress:.1f}% ({request_count} requests)", end='\r')

        if not all_data:
            print(f"\nNo data fetched for {symbol}")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Trim to requested date range
        df = df[df.index >= start_time]

        print(f"\nFetched {len(df)} candles for {symbol} from Hyperliquid")
        print(f"  Time range: {df.index[0]} to {df.index[-1]}")

        return df

    def fetch_range(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical data between two datetimes.
        """
        if start_time >= end_time:
            return pd.DataFrame()

        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        interval_minutes = self._interval_to_minutes(interval)
        candles_needed = max(
            0,
            int((end_time - start_time).total_seconds() / 60 / interval_minutes) + 1
        )

        if candles_needed <= self.max_candles_per_request:
            df = self.fetch_candles(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            if df.empty:
                return df
            return df[df.index >= start_time]

        all_data = []
        current_end = end_time
        request_count = 0
        while current_end > start_time:
            chunk_minutes = interval_minutes * self.max_candles_per_request
            chunk_start = max(start_time, current_end - timedelta(minutes=chunk_minutes))

            df_chunk = self.fetch_candles(
                symbol=symbol,
                interval=interval,
                start_time=chunk_start,
                end_time=current_end
            )

            if df_chunk.empty:
                break

            all_data.append(df_chunk)
            request_count += 1
            current_end = df_chunk.index[0] - timedelta(minutes=interval_minutes)

            if request_count % 5 == 0:
                time.sleep(1)

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df[df.index >= start_time]

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        unit = interval[-1].lower()
        value = int(interval[:-1])

        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        elif unit == 'w':
            return value * 10080
        else:
            raise ValueError(f"Unknown interval: {interval}")

    def get_multiple_symbols(
        self,
        symbols: list,
        interval: str = '5m',
        days: int = 30
    ) -> dict:
        """
        Fetch data for multiple symbols

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}

        print(f"\nFetching Hyperliquid data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {symbol}")
            df = self.fetch_historical_data(symbol, interval, days)

            if not df.empty:
                data[symbol] = df
            else:
                print(f"  Skipped {symbol} (no data)")

            # Rate limiting
            time.sleep(0.5)

        print(f"\nSuccessfully fetched {len(data)}/{len(symbols)} symbols")

        return data


if __name__ == "__main__":
    # Example usage
    fetcher = HyperliquidDataFetcher()

    # Test with ETH
    symbol = 'ETH'
    fetch_days = 60  # Hyperliquid has limited history
    base_dir = Path(__file__).parent

    try:
        df = fetcher.fetch_historical_data(
            symbol,
            interval='5m',
            days=fetch_days
        )
        if not df.empty:
            # Save with hyperliquid marker
            path = base_dir / f'cached_{symbol.upper()}USDT_5m_hyperliquid.csv'
            df.to_csv(path)
            print(f"Saved {len(df)} records to {path.name}")
            print(f"  Range: {df.index[0]} to {df.index[-1]}")

            print(f"\nData Summary for {symbol}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"\nFirst few rows:")
            print(df.head())
            print(f"\nLast few rows:")
            print(df.tail())
        else:
            print(f"Failed to fetch data from Hyperliquid")
    except Exception as e:
        print(f"Error: {e}")
