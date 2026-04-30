"""Bar-aggregated trade-flow features from Binance Vision aggTrades.

Maps a stream of aggregated trades onto a target bar grid (e.g. 5m klines) and
produces per-bar microstructure features that capture order-flow asymmetry.

Conventions:
- `is_buyer_maker == True`  → the seller was the aggressor (taker-sell)
- `is_buyer_maker == False` → the buyer was the aggressor (taker-buy)
- Signed volume convention: + for taker-buy, - for taker-sell.

Features per bar:
- buy_volume, sell_volume          : aggressor-sided volumes
- signed_volume                    : buy_volume - sell_volume
- tfi                              : trade-flow imbalance, (buy - sell) / (buy + sell)
- buy_count, sell_count            : aggressor-sided trade counts
- aggressor_count_ratio            : buy_count / (buy_count + sell_count)
- avg_trade_size                   : volume / count
- max_trade_size                   : largest single aggregated trade
- large_trade_volume               : sum of trades above the rolling 95th-pct size
- large_trade_count                : count of those trades
- vwap                             : volume-weighted average price within the bar
- vwap_dev_bps                     : 1e4 * (last_price - vwap) / vwap
- kyle_lambda                      : per-bar slope of price-tick vs signed_volume

Rolling features (computed on top of per-bar series):
- vpin_K                           : Easley/López de Prado VPIN over K volume buckets
- tfi_smooth_N                     : EMA(N) of tfi
- signed_volume_z_N                : (signed_volume - rolling_mean_N) / rolling_std_N

All features are point-in-time as long as the input aggTrades and bar grid
align by timestamp (no lookahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeFlowConfig:
    bar_interval: str = "5min"        # pandas resample alias
    large_trade_pctile: float = 0.95  # rolling pctile for large-trade flag
    large_trade_window: int = 288     # bars over which to estimate the pctile
    vpin_buckets: int = 50            # VPIN volume-bucket count
    smooth_n: int = 12                # EMA span for tfi_smooth, z-score window
    compute_kyle_lambda: bool = True  # set False for big windows; per-bar regression is slow


def per_bar_features(
    trades: pd.DataFrame,
    config: TradeFlowConfig = TradeFlowConfig(),
) -> pd.DataFrame:
    """Pure per-bar aggregation. Independent across days — safe to call per-file
    and concatenate. Excludes rolling features (those go in `add_rolling_features`).
    """
    if trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df["taker_buy"] = ~df["is_buyer_maker"]
    df["signed_qty"] = np.where(df["taker_buy"], df["quantity"], -df["quantity"])
    df["buy_qty"] = np.where(df["taker_buy"], df["quantity"], 0.0)
    df["sell_qty"] = np.where(df["taker_buy"], 0.0, df["quantity"])
    df["buy_cnt"] = df["taker_buy"].astype(int)
    df["sell_cnt"] = (~df["taker_buy"]).astype(int)
    df["pq"] = df["price"] * df["quantity"]

    g = df.set_index("transact_time").resample(config.bar_interval, label="left", closed="left")

    out = pd.DataFrame({
        "buy_volume": g["buy_qty"].sum(),
        "sell_volume": g["sell_qty"].sum(),
        "buy_count": g["buy_cnt"].sum(),
        "sell_count": g["sell_cnt"].sum(),
        "trade_count": g["quantity"].count(),
        "total_volume": g["quantity"].sum(),
        "vwap": g["pq"].sum() / g["quantity"].sum().replace(0, np.nan),
        "last_price": g["price"].last(),
        "max_trade_size": g["quantity"].max(),
    })
    out["signed_volume"] = out["buy_volume"] - out["sell_volume"]
    denom = (out["buy_volume"] + out["sell_volume"]).replace(0, np.nan)
    out["tfi"] = out["signed_volume"] / denom
    out["aggressor_count_ratio"] = out["buy_count"] / (out["buy_count"] + out["sell_count"]).replace(0, np.nan)
    out["avg_trade_size"] = out["total_volume"] / out["trade_count"].replace(0, np.nan)
    out["vwap_dev_bps"] = 1e4 * (out["last_price"] - out["vwap"]) / out["vwap"]

    if config.compute_kyle_lambda:
        df["bar_open"] = df["transact_time"].dt.floor(config.bar_interval)
        out["kyle_lambda"] = _per_bar_kyle_lambda(df, config.bar_interval)

    out.index.name = "open_time"
    return out


def add_rolling_features(
    bars: pd.DataFrame,
    config: TradeFlowConfig = TradeFlowConfig(),
) -> pd.DataFrame:
    """Add rolling / cross-day features on top of `per_bar_features` output.

    Must be applied to the FULL concatenated bar history (not per-day) because
    rolling windows span day boundaries.
    """
    out = bars.copy()

    # Large-trade detection — rolling 95th pctile of max_trade_size, shifted +1
    # so it doesn't peek at the current bar.
    threshold = (
        out["max_trade_size"]
        .rolling(config.large_trade_window, min_periods=config.large_trade_window // 4)
        .quantile(config.large_trade_pctile)
        .shift(1)
    )
    # Approximation: take large-trade volume = max_trade_size when current bar's
    # max exceeds the threshold. Exact streaming reconstruction would need to
    # rejoin per-trade data; for the IC triage this approximation is sufficient.
    is_large = out["max_trade_size"] >= threshold.fillna(np.inf)
    out["large_trade_volume"] = np.where(is_large, out["max_trade_size"], 0.0)
    out["large_trade_count"] = is_large.astype(int)

    # Smoothing and normalization.
    out["tfi_smooth"] = out["tfi"].ewm(span=config.smooth_n, adjust=False).mean()
    rmean = out["signed_volume"].rolling(config.smooth_n, min_periods=config.smooth_n).mean()
    rstd = out["signed_volume"].rolling(config.smooth_n, min_periods=config.smooth_n).std()
    out["signed_volume_z"] = (out["signed_volume"] - rmean) / rstd.replace(0, np.nan)

    # VPIN.
    out["vpin"] = _vpin(out, n_buckets=config.vpin_buckets)

    return out


def aggregate_trades_to_bars(
    trades: pd.DataFrame,
    config: TradeFlowConfig = TradeFlowConfig(),
) -> pd.DataFrame:
    """All-in-one entry point — backwards compatible with prior API. Use
    `aggregate_trades_streaming` for windows that don't fit in memory."""
    bars = per_bar_features(trades, config)
    if bars.empty:
        return bars
    return add_rolling_features(bars, config)


def aggregate_trades_streaming(
    parquet_paths: list,
    config: TradeFlowConfig = TradeFlowConfig(),
) -> pd.DataFrame:
    """Memory-efficient: process per-day parquets one at a time, then apply
    cross-day rolling features.

    Parameters
    ----------
    parquet_paths : iterable of Path / str
        Daily aggTrade parquet files (sorted by date).

    Returns
    -------
    DataFrame indexed by bar `open_time` with all per-bar + rolling features.
    """
    chunks = []
    for path in sorted(parquet_paths):
        day = pd.read_parquet(path)
        chunk = per_bar_features(day, config)
        chunks.append(chunk)
        del day  # free memory before the next iteration
    if not chunks:
        return pd.DataFrame()
    bars = pd.concat(chunks).sort_index()
    # In case of any timestamp overlap (shouldn't happen with daily files), keep last.
    bars = bars[~bars.index.duplicated(keep="last")]
    return add_rolling_features(bars, config)


def _per_bar_kyle_lambda(trades_with_bar: pd.DataFrame, bar_interval: str) -> pd.Series:
    """Per-bar slope of cumulative price-tick on cumulative signed-volume.

    Returns NaN for sparse bars (< 10 trades).
    """
    g = trades_with_bar.groupby("bar_open")
    out = {}
    for bar, sub in g:
        if len(sub) < 10:
            out[bar] = np.nan
            continue
        sub = sub.sort_values("transact_time")
        ticks = sub["price"].diff().fillna(0.0).cumsum().to_numpy()
        sv = sub["signed_qty"].cumsum().to_numpy()
        # Avoid division by zero variance.
        var = sv.var()
        if var <= 0:
            out[bar] = np.nan
            continue
        cov = np.cov(ticks, sv, ddof=0)[0, 1]
        out[bar] = cov / var
    return pd.Series(out, name="kyle_lambda")


def _vpin(
    bars: pd.DataFrame,
    n_buckets: int = 50,
    *,
    lookback_bars: int = 2016,
) -> pd.Series:
    """Volume-synchronized PIN — POINT-IN-TIME version.

    The original implementation used `total_vol.iloc[-1]` to size buckets,
    which leaks the entire dataset's volume into every bar's VPIN — the
    bucketing depends on future data. This version sizes buckets using a
    trailing window of past data only.

    For each bar t:
      1. Look at the last `lookback_bars` bars (default ~7 days at 5m).
      2. Compute total volume in that window. Set bucket_size = total / n_buckets.
      3. Assign each bar in the window to its bucket based on cumulative
         volume from the window start.
      4. For each bucket, compute |buy - sell| / (buy + sell).
      5. VPIN at bar t = mean of last `n_buckets // 5` buckets' imbalances.

    Output is point-in-time: VPIN at bar t uses only data from bars before t.
    """
    n_buckets = int(n_buckets)
    window_buckets = max(2, n_buckets // 5)
    out = pd.Series(np.nan, index=bars.index)
    if len(bars) < lookback_bars:
        return out
    total_vol = bars["total_volume"].to_numpy()
    buy_vol = bars["buy_volume"].to_numpy()
    sell_vol = bars["sell_volume"].to_numpy()

    for i in range(lookback_bars, len(bars)):
        # Window: bars[i - lookback_bars : i] — strictly before bar i
        window_start = i - lookback_bars
        win_total = total_vol[window_start:i]
        win_buy = buy_vol[window_start:i]
        win_sell = sell_vol[window_start:i]
        wsum = win_total.sum()
        if wsum <= 0:
            continue
        bucket_size = wsum / n_buckets
        # Cumulative volume within the window
        cum = np.cumsum(win_total)
        bucket_idx = np.minimum((cum / bucket_size).astype(int), n_buckets - 1)
        # Per-bucket buy and sell sums
        b_buy = np.bincount(bucket_idx, weights=win_buy, minlength=n_buckets)
        b_sell = np.bincount(bucket_idx, weights=win_sell, minlength=n_buckets)
        denom = b_buy + b_sell
        imb = np.where(denom > 0, np.abs(b_buy - b_sell) / denom, 0.0)
        # Mean of last `window_buckets` bucket imbalances
        out.iloc[i] = imb[-window_buckets:].mean()
    return out
