import numpy as np
import pandas as pd
import pytest

from hf_features import HFFeatureEngine


def _make_price_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01", periods=n, freq="5min")

    close = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    open_ = pd.Series(close, index=index).shift(1).fillna(close[0]) + rng.normal(scale=0.2, size=n)
    close_series = pd.Series(close, index=index)
    high = np.maximum(open_, close_series) + rng.uniform(0.0, 0.3, size=n)
    low = np.minimum(open_, close_series) - rng.uniform(0.0, 0.3, size=n)
    volume = rng.integers(1_000, 5_000, size=n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close_series,
            "volume": volume,
        },
        index=index,
    )


def _make_price_frame_with_closes(closes, start="2024-01-01 00:00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="5min")
    close_series = pd.Series(closes, index=index, dtype=float)
    open_series = close_series.shift(1).fillna(close_series.iloc[0])
    high = pd.concat([open_series, close_series], axis=1).max(axis=1) + 0.1
    low = pd.concat([open_series, close_series], axis=1).min(axis=1) - 0.1
    volume = pd.Series(1_000.0, index=index)

    return pd.DataFrame(
        {
            "open": open_series,
            "high": high,
            "low": low,
            "close": close_series,
            "volume": volume,
        },
        index=index,
    )


def _supertrend_reference(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr = HFFeatureEngine._calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        if not pd.isna(final_upper.iloc[i - 1]) and df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = (
                min(basic_upper.iloc[i], final_upper.iloc[i - 1])
                if not pd.isna(final_upper.iloc[i - 1])
                else basic_upper.iloc[i]
            )

        if not pd.isna(final_lower.iloc[i - 1]) and df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = (
                max(basic_lower.iloc[i], final_lower.iloc[i - 1])
                if not pd.isna(final_lower.iloc[i - 1])
                else basic_lower.iloc[i]
            )

    st_dir = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > final_upper.iloc[i - 1]:
            st_dir.iloc[i] = 1
        elif df["close"].iloc[i] < final_lower.iloc[i - 1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i - 1]

    return final_upper, final_lower, st_dir


def _heikin_ashi_dir_reference(df: pd.DataFrame) -> pd.Series:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    return np.sign(ha_close - ha_open)


def test_supertrend_matches_previous_logic():
    df_input = _make_price_frame()
    features = HFFeatureEngine.calculate_features(df_input)
    expected_upper, expected_lower, expected_dir = _supertrend_reference(df_input)

    pd.testing.assert_series_equal(
        features["supertrend_upper"],
        expected_upper,
        check_names=False,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        features["supertrend_lower"],
        expected_lower,
        check_names=False,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        features["supertrend_dir"],
        expected_dir,
        check_names=False,
        check_dtype=False,
    )


def test_heikin_ashi_dir_matches_previous_logic():
    df_input = _make_price_frame()
    features = HFFeatureEngine.calculate_features(df_input)
    expected_dir = _heikin_ashi_dir_reference(df_input)

    pd.testing.assert_series_equal(
        features["ha_dir"],
        expected_dir,
        check_names=False,
        check_dtype=False,
    )


def test_mtf_15m_boundary_uses_last_closed_bar():
    closes = [100, 100, 100, 110, 110, 110, 90, 90, 90]
    df_input = _make_price_frame_with_closes(closes)
    features = HFFeatureEngine.calculate_features(df_input)

    close_index = df_input.index + pd.Timedelta(minutes=5)
    ohlcv_close = df_input.copy()
    ohlcv_close.index = close_index
    df15 = ohlcv_close.resample(
        "15min",
        label="right",
        closed="right",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    ema20_15 = df15["close"].ewm(span=20, adjust=False).mean()
    ema_slope_20_15 = ema20_15 / ema20_15.shift(1) - 1

    expected_0030 = ema_slope_20_15.loc[pd.Timestamp("2024-01-01 00:30:00")]
    expected_0045 = ema_slope_20_15.loc[pd.Timestamp("2024-01-01 00:45:00")]

    assert np.isfinite(expected_0030)
    assert np.isfinite(expected_0045)
    assert features.loc["2024-01-01 00:25:00", "ema_slope_20_15m"] == pytest.approx(expected_0030)
    assert features.loc["2024-01-01 00:40:00", "ema_slope_20_15m"] == pytest.approx(expected_0045)


def test_mtf_1h_boundary_uses_last_closed_bar():
    closes = [100] * 12 + [120] * 12
    df_input = _make_price_frame_with_closes(closes)
    features = HFFeatureEngine.calculate_features(df_input)

    close_index = df_input.index + pd.Timedelta(minutes=5)
    ohlcv_close = df_input.copy()
    ohlcv_close.index = close_index
    df60 = ohlcv_close.resample(
        "60min",
        label="right",
        closed="right",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    ema20_60 = df60["close"].ewm(span=20, adjust=False).mean()
    ema_slope_20_60 = ema20_60 / ema20_60.shift(1) - 1

    expected_0200 = ema_slope_20_60.loc[pd.Timestamp("2024-01-01 02:00:00")]
    assert np.isfinite(expected_0200)
    assert features.loc["2024-01-01 01:55:00", "ema_slope_20_1h"] == pytest.approx(expected_0200)


def test_mtf_adx_boundary_alignment():
    df_input = _make_price_frame(n=240)
    features = HFFeatureEngine.calculate_features(df_input)

    base_interval = HFFeatureEngine._infer_base_interval(df_input.index)
    close_index = df_input.index + base_interval
    ohlcv_close = df_input.copy()
    ohlcv_close.index = close_index

    df15 = ohlcv_close.resample(
        "15min",
        label="right",
        closed="right",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    adx_15 = HFFeatureEngine._calculate_adx(df15, 14)
    adx_15m = adx_15.reindex(close_index, method="ffill")
    adx_15m_series = pd.Series(adx_15m.values, index=df_input.index)

    df60 = ohlcv_close.resample(
        "60min",
        label="right",
        closed="right",
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    adx_60 = HFFeatureEngine._calculate_adx(df60, 14)
    adx_1h = adx_60.reindex(close_index, method="ffill")
    adx_1h_series = pd.Series(adx_1h.values, index=df_input.index)

    boundary_15 = df_input.index[close_index.minute % 15 == 0]
    target_15 = next(
        (idx for idx in reversed(boundary_15) if pd.notna(adx_15m_series.loc[idx])),
        None,
    )
    if target_15 is None:
        pytest.skip("No finite 15m ADX boundary value available for test.")

    boundary_1h = df_input.index[close_index.minute == 0]
    target_1h = next(
        (idx for idx in reversed(boundary_1h) if pd.notna(adx_1h_series.loc[idx])),
        None,
    )
    if target_1h is None:
        pytest.skip("No finite 1h ADX boundary value available for test.")

    assert features.loc[target_15, "adx_15m"] == pytest.approx(adx_15m_series.loc[target_15])
    assert features.loc[target_1h, "adx_1h"] == pytest.approx(adx_1h_series.loc[target_1h])
