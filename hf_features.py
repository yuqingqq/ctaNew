"""
High-Frequency Feature Engineering
Extracts technical indicators optimized for 1-5 minute trading
"""

import pandas as pd
import numpy as np
from typing import Tuple


class HFFeatureEngine:
    """High-frequency feature extraction for ML model"""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive features for high-frequency trading

        Features include:
        - Price momentum (multiple timeframes)
        - Volume indicators
        - Volatility measures
        - Trend strength
        - Order flow proxies
        - Microstructure indicators
        """
        df = df.copy()

        # ===== PRICE FEATURES =====
        # Returns across multiple windows
        for period in [1, 2, 3, 5, 10, 15, 20, 30]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'return_{period}'] = df[f'return_{period}'].fillna(0.0)

        # Price position in recent range
        for window in [10, 20, 50]:
            rolling_min = df['close'].rolling(window).min()
            rolling_max = df['close'].rolling(window).max()
            range_ = (rolling_max - rolling_min).replace(0, np.nan)
            price_pct = (df['close'] - rolling_min) / range_
            if window >= 20:
                price_pct = price_pct.clip(0.0, 1.0)
            df[f'price_pct_{window}'] = price_pct

        # ===== MOMENTUM INDICATORS =====
        # RSI at multiple periods
        for period in [5, 10, 14, 21]:
            df[f'rsi_{period}'] = HFFeatureEngine._calculate_rsi(df['close'], period)

        # Multi-timeframe RSI (4H) for signal filtering
        # Experiment showed 4H RSI 30-70 filter improves Sharpe from 1.92 to 2.09
        df['rsi_4h'] = HFFeatureEngine._calculate_mtf_rsi(df, timeframe='4h', period=14)

        # MACD components
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                   df['close'].shift(period) * 100).fillna(0.0)

        # ===== TREND INDICATORS =====
        # EMAs
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

        # EMA slopes (relative change over 5 bars)
        df['ema_slope_20'] = (df['ema_20'] / df['ema_20'].shift(5) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['ema_slope_50'] = (df['ema_50'] / df['ema_50'].shift(5) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ===== DIRECTIONAL EFFICIENCY (Kaufman Efficiency Ratio style) =====
        # Captures how directional vs choppy price action is over a window.
        # 0 = pure chop, 1 = perfectly directional.
        try:
            window = 96  # 8 hours of 5m bars
            net_move = (df['close'] - df['close'].shift(window)).abs()
            path_move = df['close'].diff().abs().rolling(window, min_periods=window).sum()
            eff = (net_move / path_move.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[f'efficiency_{window}'] = eff.clip(0.0, 1.0)
        except Exception:
            df['efficiency_96'] = 0.0

        # EMA alignment score
        df['ema_alignment'] = (
            (df['ema_5'] > df['ema_10']).astype(int) +
            (df['ema_10'] > df['ema_20']).astype(int) +
            (df['ema_20'] > df['ema_50']).astype(int) +
            (df['ema_50'] > df['ema_100']).astype(int)
        )

        # ADX for trend strength
        df['adx'] = HFFeatureEngine._calculate_adx(df, 14).fillna(0.0)
        df['adx_20'] = HFFeatureEngine._calculate_adx(df, 20).fillna(0.0)
        # ADX slope (trend acceleration proxy)
        df['adx_slope_5'] = df['adx'].diff(5).fillna(0.0)

        # ===== SUPERTREND (ATR-based trailing) =====
        try:
            st_upper, st_lower, st_dir = HFFeatureEngine._calculate_supertrend(df, period=10, multiplier=3.0)
            df['supertrend_upper'] = st_upper
            df['supertrend_lower'] = st_lower
            df['supertrend_dir'] = st_dir
        except Exception:
            df['supertrend_upper'] = np.nan
            df['supertrend_lower'] = np.nan
            df['supertrend_dir'] = 0

        # ===== KELTNER CHANNELS (EMA20 +/- ATR * 1.5) =====
        try:
            ema20 = df['ema_20'] if 'ema_20' in df.columns else df['close'].ewm(span=20, adjust=False).mean()
            kc_mult = 1.5
            atr14 = df['atr_14'] if 'atr_14' in df.columns else HFFeatureEngine._calculate_atr(df, 14)
            df['keltner_upper'] = ema20 + kc_mult * atr14
            df['keltner_lower'] = ema20 - kc_mult * atr14
        except Exception:
            df['keltner_upper'] = np.nan
            df['keltner_lower'] = np.nan

        # ===== HEIKIN-ASHI DIRECTION =====
        try:
            df['ha_dir'] = HFFeatureEngine._calculate_heikin_ashi_dir(df)
        except Exception:
            df['ha_dir'] = 0

        # ===== DONCHIAN WIDTH (20) =====
        try:
            dc_high = df['high'].rolling(20).max()
            dc_low = df['low'].rolling(20).min()
            df['donchian_width_20'] = HFFeatureEngine._safe_divide(dc_high - dc_low, df['close'], default=0.0)
        except Exception:
            df['donchian_width_20'] = 0.0

        index_is_close = bool(df.attrs.get("index_is_close", False))
        base_interval = df.attrs.get("base_interval") or HFFeatureEngine._infer_base_interval(df.index)

        # ===== MULTI-TIMEFRAME CONFIRMATION (15m) =====
        try:
            ohlcv = df[['open', 'high', 'low', 'close', 'volume']]
            close_index = df.index if index_is_close else df.index + base_interval
            ohlcv_close = ohlcv.copy()
            ohlcv_close.index = close_index
            df15 = ohlcv_close.resample(
                '15min',
                label='right',
                closed='right'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            # 15m EMA and slope (use last completed 15m bar at each 5m close)
            ema20_15 = df15['close'].ewm(span=20, adjust=False).mean()
            ema_slope_20_15 = (ema20_15 / ema20_15.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
            # 15m ADX
            adx_15 = HFFeatureEngine._calculate_adx(df15, 14)
            # Map back to 5m close index, then align to 5m bar start index
            ema_slope_20_15m = ema_slope_20_15.reindex(close_index, method='ffill')
            adx_15m = adx_15.reindex(close_index, method='ffill')
            df['ema_slope_20_15m'] = pd.Series(ema_slope_20_15m.values, index=df.index)
            df['adx_15m'] = pd.Series(adx_15m.values, index=df.index)
        except Exception:
            df['ema_slope_20_15m'] = 0.0
            df['adx_15m'] = 0.0

        # ===== MULTI-TIMEFRAME CONFIRMATION (1h) =====
        try:
            ohlcv = df[['open', 'high', 'low', 'close', 'volume']]
            close_index = df.index if index_is_close else df.index + base_interval
            ohlcv_close = ohlcv.copy()
            ohlcv_close.index = close_index
            df60 = ohlcv_close.resample(
                '60min',
                label='right',
                closed='right'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            ema20_60 = df60['close'].ewm(span=20, adjust=False).mean()
            ema_slope_20_60 = (ema20_60 / ema20_60.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
            adx_60 = HFFeatureEngine._calculate_adx(df60, 14)
            ema_slope_20_1h = ema_slope_20_60.reindex(close_index, method='ffill')
            adx_1h = adx_60.reindex(close_index, method='ffill')
            df['ema_slope_20_1h'] = pd.Series(ema_slope_20_1h.values, index=df.index)
            df['adx_1h'] = pd.Series(adx_1h.values, index=df.index)
        except Exception:
            df['ema_slope_20_1h'] = 0.0
            df['adx_1h'] = 0.0

        # ===== VWAP (rolling) =====
        try:
            pv = (df['close'] * df['volume']).rolling(96, min_periods=20).sum()
            vv = df['volume'].rolling(96, min_periods=20).sum()
            vwap_96 = HFFeatureEngine._safe_divide(pv, vv, default=0.0)
            df['vwap_96'] = vwap_96
            df['vwap_slope_96'] = (vwap_96 / vwap_96.shift(5) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            df['vwap_96'] = 0.0
            df['vwap_slope_96'] = 0.0
        # ADX slope (trend acceleration proxy)
        df['adx_slope_5'] = df['adx'].diff(5)

        # ===== VOLATILITY FEATURES =====
        # ATR
        df['atr_5'] = HFFeatureEngine._calculate_atr(df, 5)
        df['atr_14'] = HFFeatureEngine._calculate_atr(df, 14)
        df['atr_pct'] = df['atr_14'] / df['close']

        # Crash filter features: 7-day momentum and daily volatility
        # Used to block LONG signals during severe downtrends with high volatility
        crash_momentum_bars = 2016  # ~7 days at 5m
        df['crash_ret_7d'] = df['close'].pct_change(crash_momentum_bars) * 100
        daily_vol_bars = 288  # 1 day at 5m
        df['crash_daily_vol'] = df['close'].pct_change().rolling(daily_vol_bars).std() * np.sqrt(daily_vol_bars) * 100

        # Bollinger Bands
        for period in [10, 20]:
            bb_mid = df['close'].rolling(period).mean()
            bb_std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_mid + (bb_std * 2)
            df[f'bb_lower_{period}'] = bb_mid - (bb_std * 2)
            df[f'bb_width_{period}'] = HFFeatureEngine._safe_divide(
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'],
                bb_mid,
                default=0.0
            )
            df[f'bb_position_{period}'] = HFFeatureEngine._safe_divide(
                df['close'] - df[f'bb_lower_{period}'],
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'],
                default=0.5
            )

        # ===== SQUEEZE (BB width contraction) =====
        try:
            bbw = df['bb_width_20']
            roll = bbw.rolling(120, min_periods=20)
            pctl = roll.rank(pct=True)
            df['bb_width_pctl_120'] = pctl
            # Squeeze flag when width below 20th percentile
            df['bb_squeeze_20'] = (pctl < 0.2).astype(int)
        except Exception:
            df['bb_width_pctl_120'] = 0.5
            df['bb_squeeze_20'] = 0

        # Historical volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['return_1'].rolling(period).std()

        # ===== VOLUME FEATURES =====
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = HFFeatureEngine._safe_divide(
                df['volume'],
                df[f'volume_ma_{period}'],
                default=1.0
            )

        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5)

        # Price-volume correlation
        for period in [10, 20]:
            df[f'price_volume_corr_{period}'] = df['close'].rolling(period).corr(df['volume'])

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        df['obv_signal'] = df['obv'] - df['obv_ema']

        # ===== CANDLE FEATURES =====
        # Candle body and wick sizes
        df['body_size'] = HFFeatureEngine._safe_divide(
            abs(df['close'] - df['open']),
            df['open'],
            default=0.0
        )
        high_close = df[['open', 'close']].max(axis=1)
        low_close = df[['open', 'close']].min(axis=1)
        df['upper_wick'] = HFFeatureEngine._safe_divide(
            df['high'] - high_close,
            df['open'],
            default=0.0
        )
        df['lower_wick'] = HFFeatureEngine._safe_divide(
            low_close - df['low'],
            df['open'],
            default=0.0
        )
        df['candle_range'] = HFFeatureEngine._safe_divide(
            df['high'] - df['low'],
            df['open'],
            default=0.0
        )

        # Wick ratio over range
        df['wick_ratio'] = HFFeatureEngine._safe_divide(
            (df['upper_wick'].fillna(0.0) + df['lower_wick'].fillna(0.0)),
            (df['candle_range'].replace(0, np.nan)),
            default=0.0
        )

        # Candle patterns
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)

        # ===== MICROSTRUCTURE FEATURES =====
        # High-low spread
        df['hl_spread'] = HFFeatureEngine._safe_divide(
            df['high'] - df['low'],
            df['close'],
            default=0.0
        )

        # Close position in candle
        range_ = df['high'] - df['low']
        df['close_position'] = HFFeatureEngine._safe_divide(
            df['close'] - df['low'],
            range_,
            default=0.5
        )

        # Realized volatility (Parkinson estimator)
        df['parkinson_vol'] = np.sqrt(1 / (4 * np.log(2)) *
                                      np.log(df['high'] / df['low']) ** 2)

        # ===== ORDER FLOW PROXIES =====
        # Buying/selling pressure
        df['buy_pressure'] = HFFeatureEngine._safe_divide(
            df['close'] - df['low'],
            range_,
            default=0.5
        )
        df['sell_pressure'] = HFFeatureEngine._safe_divide(
            df['high'] - df['close'],
            range_,
            default=0.5
        )

        # Money Flow Index
        df['mfi'] = HFFeatureEngine._calculate_mfi(df, 14)

        # ===== TIME-BASED FEATURES =====
        time_features = {
            "hour": df.index.hour,
            "minute": df.index.minute,
            "day_of_week": df.index.dayofweek,
        }
        df = df.assign(**time_features)

        # Cyclical encoding for time
        cyclical_features = {
            "hour_sin": np.sin(2 * np.pi * df["hour"] / 24),
            "hour_cos": np.cos(2 * np.pi * df["hour"] / 24),
            "minute_sin": np.sin(2 * np.pi * df["minute"] / 60),
            "minute_cos": np.cos(2 * np.pi * df["minute"] / 60),
        }
        df = df.assign(**cyclical_features)

        # ===== PRICE ACTION FEATURES =====
        # Higher highs / lower lows
        price_action = {
            "higher_high": (df["high"] > df["high"].shift(1)).astype(int),
            "lower_low": (df["low"] < df["low"].shift(1)).astype(int),
            "consecutive_up": (df["close"] > df["close"].shift(1)).astype(int),
            "consecutive_down": (df["close"] < df["close"].shift(1)).astype(int),
        }
        df = df.assign(**price_action)

        # Support/Resistance levels
        sr_features = {}
        for window in [20, 50]:
            resistance = df["high"].rolling(window).max()
            support = df["low"].rolling(window).min()
            sr_features[f"resistance_{window}"] = resistance
            sr_features[f"support_{window}"] = support
            sr_features[f"dist_resistance_{window}"] = HFFeatureEngine._safe_divide(
                resistance - df["close"],
                df["close"],
                default=0.0,
            )
            sr_features[f"dist_support_{window}"] = HFFeatureEngine._safe_divide(
                df["close"] - support,
                df["close"],
                default=0.0,
            )
        df = df.assign(**sr_features)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill()

        # ===== EXHAUSTION FEATURES =====
        df = HFFeatureEngine._calculate_exhaustion_features(df)

        # ===== WYCKOFF PATTERN FEATURES =====
        df = HFFeatureEngine._calculate_wyckoff_features(df)

        return df

    @staticmethod
    def _calculate_exhaustion_features(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
        """
        Calculate exhaustion-specific features for reversal detection.
        All thresholds use rolling quantiles for adaptive behavior.

        Features:
        - close_position_at_high: Where price closed relative to range when making new high
        - close_position_at_low: Where price closed relative to range when making new low
        - rsi_divergence: Detects momentum divergence (price HH but RSI LH, or vice versa)
        - volume_trend_at_highs: Whether volume is declining on successive highs
        - effort_result_ratio: High volume but low price movement = absorption
        - upper_wick_pct / lower_wick_pct: Wick size as percentage of price
        """
        # Close position when making new highs/lows (failure to close near extreme)
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        candle_range = (df['high'] - df['low']).replace(0, np.nan)

        # When we make a new 20-bar high, where did we close?
        # Low value = failure to close near high (bearish exhaustion)
        at_new_high = df['high'] >= high_20
        at_new_low = df['low'] <= low_20

        close_pos = (df['close'] - df['low']) / candle_range
        df['close_position_at_high'] = np.where(at_new_high, close_pos, np.nan)
        df['close_position_at_low'] = np.where(at_new_low, close_pos, np.nan)

        # Forward fill for recent exhaustion context, then fill remaining with neutral 0.5
        df['close_position_at_high'] = df['close_position_at_high'].ffill(limit=5).fillna(0.5)
        df['close_position_at_low'] = df['close_position_at_low'].ffill(limit=5).fillna(0.5)

        # Rolling quantiles for close position thresholds
        df['close_pos_q20'] = close_pos.rolling(window, min_periods=20).quantile(0.20)
        df['close_pos_q80'] = close_pos.rolling(window, min_periods=20).quantile(0.80)

        # RSI Divergence Detection
        # Look for price making higher highs but RSI making lower highs (bearish divergence)
        # or price making lower lows but RSI making higher lows (bullish divergence)
        rsi = df.get('rsi_14', df.get('rsi_10', pd.Series(50.0, index=df.index)))
        lookback = 20

        # Rolling max/min of price and RSI for divergence
        price_roll_high = df['high'].rolling(lookback, min_periods=5).max()
        price_roll_low = df['low'].rolling(lookback, min_periods=5).min()
        rsi_roll_high = rsi.rolling(lookback, min_periods=5).max()
        rsi_roll_low = rsi.rolling(lookback, min_periods=5).min()

        # Bearish divergence: price at/near rolling high, RSI below its rolling high
        price_near_high = df['high'] >= price_roll_high * 0.998
        rsi_below_high = rsi < rsi_roll_high * 0.95
        bearish_div = price_near_high & rsi_below_high & (rsi < 70)

        # Bullish divergence: price at/near rolling low, RSI above its rolling low
        price_near_low = df['low'] <= price_roll_low * 1.002
        rsi_above_low = rsi > rsi_roll_low * 1.05
        bullish_div = price_near_low & rsi_above_low & (rsi > 30)

        # 1 = bearish divergence (short signal), -1 = bullish divergence (long signal)
        df['rsi_divergence'] = np.where(bearish_div, 1, np.where(bullish_div, -1, 0))

        # Volume trend at highs - declining volume on successive highs
        # Look at volume when price makes new highs
        vol_at_high = np.where(at_new_high, df['volume'], np.nan)
        vol_at_high_series = pd.Series(vol_at_high, index=df.index)
        vol_at_high_ma = vol_at_high_series.ffill().rolling(5, min_periods=1).mean()
        vol_at_high_prev = vol_at_high_series.ffill().shift(1).rolling(5, min_periods=1).mean()
        df['volume_trend_at_highs'] = np.where(
            vol_at_high_prev > 0,
            (vol_at_high_ma - vol_at_high_prev) / vol_at_high_prev,
            0
        )
        df['volume_trend_at_highs'] = df['volume_trend_at_highs'].fillna(0)

        # Effort vs Result Ratio
        # High volume (effort) but low price movement (result) = potential absorption
        vol_ratio = df.get('volume_ratio_20', df['volume'] / df['volume'].rolling(20, min_periods=5).mean())
        atr_ratio = df.get('atr_14', HFFeatureEngine._calculate_atr(df, 14)) / \
                    df.get('atr_14', HFFeatureEngine._calculate_atr(df, 14)).rolling(20, min_periods=5).mean()
        atr_ratio = atr_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # High effort/result ratio = lots of volume but price not moving much
        df['effort_result_ratio'] = (vol_ratio / atr_ratio.replace(0, np.nan)).fillna(1.0)
        df['effort_result_ratio'] = df['effort_result_ratio'].clip(0.1, 10.0)

        # Rolling quantile for effort/result threshold
        df['effort_result_q85'] = df['effort_result_ratio'].rolling(window, min_periods=20).quantile(0.85)

        # Wick ratios as percentage of price (for exhaustion candle detection)
        high_body = df[['open', 'close']].max(axis=1)
        low_body = df[['open', 'close']].min(axis=1)
        df['upper_wick_pct'] = HFFeatureEngine._safe_divide(df['high'] - high_body, df['close'], default=0.0)
        df['lower_wick_pct'] = HFFeatureEngine._safe_divide(low_body - df['low'], df['close'], default=0.0)

        # Rolling quantiles for wick thresholds
        df['upper_wick_q80'] = df['upper_wick_pct'].rolling(window, min_periods=20).quantile(0.80)
        df['lower_wick_q80'] = df['lower_wick_pct'].rolling(window, min_periods=20).quantile(0.80)

        # VWAP z-score for structure alignment
        vwap = df.get('vwap_96')
        if vwap is not None:
            atr = df.get('atr_14', HFFeatureEngine._calculate_atr(df, 14))
            df['vwap_zscore'] = HFFeatureEngine._safe_divide(df['close'] - vwap, atr, default=0.0)
            df['vwap_zscore_q90'] = df['vwap_zscore'].abs().rolling(window, min_periods=20).quantile(0.90)
        else:
            df['vwap_zscore'] = 0.0
            df['vwap_zscore_q90'] = 1.5

        # ADX declining (momentum weakening)
        adx = df.get('adx', pd.Series(25.0, index=df.index))
        df['adx_declining'] = (adx < adx.shift(3)).astype(int)

        # RSI extreme quantiles
        df['rsi_q90'] = rsi.rolling(window, min_periods=20).quantile(0.90)
        df['rsi_q10'] = rsi.rolling(window, min_periods=20).quantile(0.10)

        return df

    @staticmethod
    def _calculate_wyckoff_features(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
        """
        Calculate Wyckoff pattern features for reversal detection.

        Features for three high-priority patterns:
        1. Upthrust (UT) - Failed breakout above resistance (bearish)
        2. Spring - Failed breakdown below support (bullish)
        3. Shooting Star - Long upper wick rejection at highs (bearish)

        All thresholds use rolling quantiles for adaptive behavior.
        """
        # === COMMON COMPONENTS ===
        # Support/Resistance levels (use existing if available)
        res_20 = df.get('resistance_20', df['high'].rolling(20).max())
        sup_20 = df.get('support_20', df['low'].rolling(20).min())

        # Volume ratio (use existing if available)
        vol_ratio = df.get('volume_ratio_20')
        if vol_ratio is None:
            vol_ma_20 = df['volume'].rolling(20, min_periods=5).mean()
            vol_ratio = HFFeatureEngine._safe_divide(df['volume'], vol_ma_20, default=1.0)

        # Candle components
        candle_range = (df['high'] - df['low']).replace(0, np.nan)
        body_high = df[['open', 'close']].max(axis=1)
        body_low = df[['open', 'close']].min(axis=1)
        body_size = (body_high - body_low).abs()
        upper_wick = df['high'] - body_high
        lower_wick = body_low - df['low']

        # Close position within candle (0 = at low, 1 = at high)
        close_pos = HFFeatureEngine._safe_divide(df['close'] - df['low'], candle_range, default=0.5)

        # === UPTHRUST FEATURES ===
        # 1. Breakout above resistance (high breaks above prior resistance)
        ut_breakout = df['high'] > res_20.shift(1)

        # 2. Failure to hold - close back below resistance
        ut_close_below = df['close'] < res_20.shift(1)

        # 3. Close in lower portion of candle (failure to close near high)
        close_pos_q30 = close_pos.rolling(window, min_periods=20).quantile(0.30)
        ut_close_failure = close_pos < close_pos_q30

        # 4. Volume spike on the failed breakout
        vol_q75 = vol_ratio.rolling(window, min_periods=20).quantile(0.75)
        ut_volume_spike = vol_ratio >= vol_q75

        # Upthrust composite boolean (all conditions met)
        df['wyckoff_upthrust'] = (ut_breakout & ut_close_below & ut_volume_spike).astype(int)

        # Store individual components for scoring
        df['ut_breakout_above'] = ut_breakout.astype(int)
        df['ut_close_below_res'] = ut_close_below.astype(int)
        df['ut_close_failure'] = ut_close_failure.astype(int)
        df['ut_volume_spike'] = ut_volume_spike.astype(int)

        # === SPRING FEATURES ===
        # 1. Breakdown below support (low breaks below prior support)
        spring_breakdown = df['low'] < sup_20.shift(1)

        # 2. Recovery - close back above support
        spring_close_above = df['close'] > sup_20.shift(1)

        # 3. Close in upper portion of candle (recovery/absorption)
        close_pos_q70 = close_pos.rolling(window, min_periods=20).quantile(0.70)
        spring_close_success = close_pos > close_pos_q70

        # 4. Volume spike on the spring
        spring_volume_spike = vol_ratio >= vol_q75

        # Spring composite boolean (all conditions met)
        df['wyckoff_spring'] = (spring_breakdown & spring_close_above & spring_volume_spike).astype(int)

        # Store individual components for scoring
        df['spring_breakdown_below'] = spring_breakdown.astype(int)
        df['spring_close_above_sup'] = spring_close_above.astype(int)
        df['spring_close_success'] = spring_close_success.astype(int)
        df['spring_volume_spike'] = spring_volume_spike.astype(int)

        # === SHOOTING STAR FEATURES ===
        # 1. Upper wick >= 2x body (long upper shadow)
        upper_wick_ratio = HFFeatureEngine._safe_divide(upper_wick, body_size.replace(0, np.nan), default=0.0)
        ss_long_upper_wick = upper_wick_ratio >= 2.0

        # 2. Lower wick is small (< 10% of range)
        lower_wick_pct = HFFeatureEngine._safe_divide(lower_wick, candle_range, default=0.0)
        ss_small_lower_wick = lower_wick_pct <= 0.10

        # 3. Close near low (bearish close)
        ss_close_near_low = close_pos <= 0.35

        # 4. At or near resistance (within 0.2% of resistance)
        res_proximity = HFFeatureEngine._safe_divide(df['high'] - res_20.shift(1), df['close'], default=1.0).abs()
        ss_at_resistance = res_proximity <= 0.002

        # Shooting star composite boolean
        df['wyckoff_shooting_star'] = (
            ss_long_upper_wick & ss_small_lower_wick & ss_close_near_low & ss_at_resistance
        ).astype(int)

        # Store individual components for scoring
        df['ss_long_upper_wick'] = ss_long_upper_wick.astype(int)
        df['ss_small_lower_wick'] = ss_small_lower_wick.astype(int)
        df['ss_close_near_low'] = ss_close_near_low.astype(int)
        df['ss_at_resistance'] = ss_at_resistance.astype(int)

        # === INVERSE HAMMER (Bullish counterpart to Shooting Star) ===
        # 1. Lower wick >= 2x body (long lower shadow)
        lower_wick_ratio = HFFeatureEngine._safe_divide(lower_wick, body_size.replace(0, np.nan), default=0.0)
        ih_long_lower_wick = lower_wick_ratio >= 2.0

        # 2. Upper wick is small
        upper_wick_pct = HFFeatureEngine._safe_divide(upper_wick, candle_range, default=0.0)
        ih_small_upper_wick = upper_wick_pct <= 0.10

        # 3. Close near high (bullish close)
        ih_close_near_high = close_pos >= 0.65

        # 4. At or near support
        sup_proximity = HFFeatureEngine._safe_divide(sup_20.shift(1) - df['low'], df['close'], default=1.0).abs()
        ih_at_support = sup_proximity <= 0.002

        # Inverse hammer composite boolean
        df['wyckoff_inv_hammer'] = (
            ih_long_lower_wick & ih_small_upper_wick & ih_close_near_high & ih_at_support
        ).astype(int)

        # Store individual components
        df['ih_long_lower_wick'] = ih_long_lower_wick.astype(int)
        df['ih_small_upper_wick'] = ih_small_upper_wick.astype(int)
        df['ih_close_near_high'] = ih_close_near_high.astype(int)
        df['ih_at_support'] = ih_at_support.astype(int)

        # === QUANTILE THRESHOLDS FOR SCORING ===
        # Volume spike threshold
        df['wyckoff_vol_spike_q75'] = vol_q75

        # Upper/lower wick quantiles
        upper_wick_pct_series = HFFeatureEngine._safe_divide(upper_wick, df['close'], default=0.0)
        lower_wick_pct_series = HFFeatureEngine._safe_divide(lower_wick, df['close'], default=0.0)
        df['upper_wick_pct_q80'] = upper_wick_pct_series.rolling(window, min_periods=20).quantile(0.80)
        df['lower_wick_pct_q80'] = lower_wick_pct_series.rolling(window, min_periods=20).quantile(0.80)

        # Close position quantiles
        df['close_pos_q30'] = close_pos_q30
        df['close_pos_q70'] = close_pos_q70

        return df

    @staticmethod
    def _calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Supertrend using numpy arrays to avoid pandas-loop overhead."""
        n = len(df)
        if n == 0:
            empty = pd.Series(dtype=float, index=df.index)
            return empty, empty, pd.Series(dtype=int, index=df.index)

        atr = HFFeatureEngine._calculate_atr(df, period)
        hl2 = (df['high'] + df['low']) / 2
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        close = df['close'].to_numpy()
        bu = basic_upper.to_numpy()
        bl = basic_lower.to_numpy()

        final_upper = bu.copy()
        final_lower = bl.copy()

        for i in range(1, n):
            prev_upper = final_upper[i - 1]
            prev_close = close[i - 1]
            if not np.isnan(prev_upper) and prev_close > prev_upper:
                final_upper[i] = bu[i]
            else:
                final_upper[i] = bu[i] if np.isnan(prev_upper) else min(bu[i], prev_upper)

            prev_lower = final_lower[i - 1]
            if not np.isnan(prev_lower) and prev_close < prev_lower:
                final_lower[i] = bl[i]
            else:
                final_lower[i] = bl[i] if np.isnan(prev_lower) else max(bl[i], prev_lower)

        st_dir = np.ones(n, dtype=np.int8)
        for i in range(1, n):
            if close[i] > final_upper[i - 1]:
                st_dir[i] = 1
            elif close[i] < final_lower[i - 1]:
                st_dir[i] = -1
            else:
                st_dir[i] = st_dir[i - 1]

        index = df.index
        return (
            pd.Series(final_upper, index=index, name='supertrend_upper'),
            pd.Series(final_lower, index=index, name='supertrend_lower'),
            pd.Series(st_dir, index=index, name='supertrend_dir')
        )

    @staticmethod
    def _infer_base_interval(index: pd.Index) -> pd.Timedelta:
        """Infer the base bar interval from the index."""
        try:
            diffs = index.to_series().diff().dropna()
            if diffs.empty:
                return pd.Timedelta(minutes=5)
            mode = diffs.mode()
            delta = mode.iloc[0] if not mode.empty else diffs.median()
            if pd.isna(delta) or delta <= pd.Timedelta(0):
                return pd.Timedelta(minutes=5)
            return delta
        except Exception:
            return pd.Timedelta(minutes=5)

    @staticmethod
    def _calculate_heikin_ashi_dir(df: pd.DataFrame) -> pd.Series:
        """Heikin-Ashi direction with a numpy loop to avoid pandas indexing cost."""
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float, index=df.index)

        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = np.empty(n, dtype=float)
        ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2

        ha_close_np = ha_close.to_numpy()
        for i in range(1, n):
            ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close_np[i - 1])

        ha_dir = np.sign(ha_close_np - ha_open)
        return pd.Series(ha_dir, index=df.index, name='ha_dir')

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_mtf_rsi(df: pd.DataFrame, timeframe: str = '4h', period: int = 14) -> pd.Series:
        """
        Calculate RSI on a higher timeframe and forward-fill to original resolution.

        Args:
            df: DataFrame with 'close' column and DatetimeIndex
            timeframe: Target timeframe ('1h', '4h', etc.)
            period: RSI period on the higher timeframe

        Returns:
            RSI series aligned to original DataFrame index
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(50.0, index=df.index, name=f'rsi_{timeframe}')

        try:
            # Resample close to higher timeframe (right-closed to avoid forward-looking buckets)
            close_htf = df['close'].resample(timeframe, label='right', closed='right').last().dropna()

            if len(close_htf) < period + 1:
                return pd.Series(50.0, index=df.index, name=f'rsi_{timeframe}')

            # Calculate RSI on higher timeframe
            delta = close_htf.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_htf = 100 - (100 / (1 + rs))

            # Forward-fill back to original resolution
            rsi_aligned = rsi_htf.reindex(df.index, method='ffill')
            return rsi_aligned.fillna(50.0)
        except Exception:
            return pd.Series(50.0, index=df.index, name=f'rsi_{timeframe}')

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR using EWM (matches backtest logic)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return pd.Series(true_range, index=df.index).ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX using EWM (matches backtest logic)"""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range for ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # EWM smoothing (matches backtest)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    @staticmethod
    def _calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
        """Safely divide two series, substituting defaults when denominator is zero"""
        denominator = denominator.replace(0, np.nan)
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)

    @staticmethod
    def get_feature_names() -> list:
        """Get list of all feature column names (for reference)"""
        # This is a helper method to identify which columns are features
        # Actual feature selection will be done dynamically based on what's in the dataframe
        base_features = [
            'close', 'volume', 'open', 'high', 'low'
        ]

        # Pattern-based features (will be matched dynamically)
        feature_patterns = [
            'return_', 'price_pct_', 'rsi_', 'roc_', 'ema_', 'dist_ema_',
            'bb_', 'volatility_', 'volume_', 'obv', 'atr_', 'adx',
            'macd', 'body_size', 'wick', 'candle_', 'hl_spread', 'close_position',
            'parkinson_vol', 'pressure', 'mfi', 'hour', 'minute', 'day_of_week',
            'sin', 'cos', 'higher_', 'lower_', 'consecutive_', 'resistance_',
            'support_', 'dist_resistance_', 'dist_support_', 'ema_alignment',
            'price_volume_corr_'
        ]

        return feature_patterns
