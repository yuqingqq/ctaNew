"""Exact endpoint-to-endpoint book-depth reaction features.

This module is the gap-recovery counterpart to
``live.bookdepth_flow_dynamics``.  A five-minute reaction only needs three
observed ingredients:

* a real book snapshot at the start of the window;
* every aggTrade in ``(start_snapshot, end_snapshot]``; and
* a real book snapshot at the end of the window.

Missing *intermediate* book snapshots therefore do not require interpolation.
They remain visible in the diagnostics, while the five-minute flow is summed
directly from aggTrades instead of from snapshot intervals.  Windows with a
stale or missing endpoint remain invalid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_book_snapshots(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine raw ``_fetch_day`` frames and reconstruct exact +/-1% sides."""
    frames = [x for x in parts if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()
    b = pd.concat(frames).sort_index()
    b.index = pd.DatetimeIndex(b.index).as_unit("ns")
    b = b[~b.index.duplicated(keep="last")]

    liq1 = np.exp(b["liq1"])
    b["bid1"] = liq1 * (1.0 + b["imb1"]) / 2.0
    b["ask1"] = liq1 * (1.0 - b["imb1"]) / 2.0

    liq02 = liq1 * b["touch"]
    b["bid02"] = liq02 * (1.0 + b["imb02"]) / 2.0
    b["ask02"] = liq02 * (1.0 - b["imb02"]) / 2.0
    return b[["bid1", "ask1", "bid02", "ask02", "imb1", "imb02"]]


def _window_start_positions(
    index: pd.DatetimeIndex,
    window: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return last snapshot <= ``end-window`` and its timing diagnostics."""
    idx_ns = index.as_unit("ns").asi8
    target_ns = idx_ns - window.value
    pos = np.searchsorted(idx_ns, target_ns, side="right") - 1
    valid = pos >= 0
    start_ns = np.full(len(index), np.iinfo(np.int64).min, dtype=np.int64)
    start_ns[valid] = idx_ns[pos[valid]]
    stale_seconds = np.full(len(index), np.nan)
    elapsed_seconds = np.full(len(index), np.nan)
    stale_seconds[valid] = (target_ns[valid] - start_ns[valid]) / 1e9
    elapsed_seconds[valid] = (idx_ns[valid] - start_ns[valid]) / 1e9
    return pos, stale_seconds, elapsed_seconds


def _exact_trade_windows(
    trades: pd.DataFrame,
    start_ns: np.ndarray,
    end_ns: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sum every aggTrade in each exact ``(start, end]`` interval."""
    result = {
        name: np.full(len(end_ns), np.nan)
        for name in ["buy_quote", "sell_quote", "buy_count", "sell_count"]
    }
    if trades.empty or not valid.any():
        return result

    t = trades
    trade_ns = pd.DatetimeIndex(t["transact_time"]).as_unit("ns").asi8
    buy = t["buy_quote"].to_numpy(dtype=float)
    sell = t["sell_quote"].to_numpy(dtype=float)
    values = {
        "buy_quote": buy,
        "sell_quote": sell,
        "buy_count": (buy > 0).astype(float),
        "sell_count": (sell > 0).astype(float),
    }
    ii = np.flatnonzero(valid)
    left = np.searchsorted(trade_ns, start_ns[ii], side="right")
    right = np.searchsorted(trade_ns, end_ns[ii], side="right")
    for name, x in values.items():
        prefix = np.empty(len(x) + 1, dtype=float)
        prefix[0] = 0.0
        np.cumsum(x, out=prefix[1:])
        result[name][ii] = prefix[right] - prefix[left]
    return result


def build_recovered_dynamics(
    symbol: str,
    book: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    window: str = "5min",
    max_endpoint_staleness_seconds: float = 90.0,
) -> pd.DataFrame:
    """Build interval diagnostics and exact endpoint reaction features."""
    if book.empty or trades.empty:
        return pd.DataFrame()

    from live.bookdepth_flow_dynamics import _align_intervals

    d = _align_intervals(book, trades)
    if d.empty:
        return d
    d["symbol"] = symbol

    # Keep the 30-second diagnostics compatible with v2.  A long interval is
    # still invalid as a 30-second reaction, even when the endpoint-level
    # five-minute reaction can be recovered exactly.
    d["return"] = d["price"].pct_change(fill_method=None)
    prev_bid = d["bid1"].shift(1).where(d["bid1"].shift(1).gt(0))
    prev_ask = d["ask1"].shift(1).where(d["ask1"].shift(1).gt(0))
    d["bid_change"] = d["bid1"] / prev_bid - 1.0
    d["ask_change"] = d["ask1"] / prev_ask - 1.0
    d["ask_bid_ratio"] = d["ask1"] / d["bid1"].where(d["bid1"].gt(0))
    d["ask_bid_ratio_change"] = d["ask_bid_ratio"] / d["ask_bid_ratio"].shift(1) - 1.0
    d["imb_change"] = d["imb1"].diff()
    d["buy_to_ask"] = d["buy_quote"] / prev_ask
    d["sell_to_bid"] = d["sell_quote"] / prev_bid
    d["signed_pressure"] = d["buy_to_ask"] - d["sell_to_bid"]
    d["ask_depth_residual"] = (d["ask1"] - d["ask1"].shift(1) + d["buy_quote"]) / prev_ask
    d["bid_depth_residual"] = (d["bid1"] - d["bid1"].shift(1) + d["sell_quote"]) / prev_bid
    d["return_bps"] = d["return"] * 1e4
    d["impact_bps_per_pressure"] = d["return_bps"] / d["signed_pressure"].where(
        d["signed_pressure"].abs() >= 1e-4
    )
    bad_interval = d["interval_seconds"].gt(90.0)
    d["gap_interval"] = bad_interval
    d.loc[bad_interval, [
        "return", "bid_change", "ask_change", "ask_bid_ratio_change", "imb_change",
        "buy_to_ask", "sell_to_bid", "signed_pressure", "ask_depth_residual",
        "bid_depth_residual", "return_bps", "impact_bps_per_pressure",
    ]] = np.nan

    w = pd.Timedelta(window)
    suffix = window
    d[f"snapshot_count_{suffix}"] = d["interval_seconds"].rolling(w, min_periods=1).count()
    d[f"gap_count_{suffix}"] = d["gap_interval"].rolling(w, min_periods=1).sum()
    d[f"max_interval_seconds_{suffix}"] = d["interval_seconds"].rolling(w, min_periods=1).max()
    d[f"any_raw_gap_{suffix}"] = d[f"gap_count_{suffix}"].gt(0)

    pos, start_stale, elapsed = _window_start_positions(d.index, w)
    has_start = pos >= 0
    idx_ns = d.index.as_unit("ns").asi8
    start_ns = np.full(len(d), np.iinfo(np.int64).min, dtype=np.int64)
    start_ns[has_start] = idx_ns[pos[has_start]]
    start_time = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    if has_start.any():
        start_time.iloc[np.flatnonzero(has_start)] = pd.to_datetime(
            start_ns[has_start], utc=True
        ).to_numpy()
    d[f"window_start_snapshot_time_{suffix}"] = start_time
    d[f"window_start_staleness_seconds_{suffix}"] = start_stale
    d[f"window_elapsed_seconds_{suffix}"] = elapsed
    d[f"start_endpoint_fresh_{suffix}"] = (
        has_start
        & (start_stale >= 0.0)
        & (start_stale <= max_endpoint_staleness_seconds)
    )

    start_fields = ["bid1", "ask1", "price", "imb1", "ask_bid_ratio"]
    for side in start_fields:
        values = np.full(len(d), np.nan)
        values[has_start] = d[side].to_numpy(dtype=float)[pos[has_start]]
        d[f"{side}_start_{suffix}"] = values

    flow = _exact_trade_windows(trades, start_ns, idx_ns, has_start)
    d[f"buy_quote_{suffix}"] = flow["buy_quote"]
    d[f"sell_quote_{suffix}"] = flow["sell_quote"]
    d[f"buy_count_{suffix}"] = flow["buy_count"]
    d[f"sell_count_{suffix}"] = flow["sell_count"]
    d[f"flow_exact_{suffix}"] = has_start

    d["extreme_imbalance_1pct"] = d["imb1"].abs().gt(0.999)
    d[f"extreme_imbalance_{suffix}"] = (
        d["extreme_imbalance_1pct"]
        | d[f"imb1_start_{suffix}"].abs().gt(0.999)
    )
    bid0 = d[f"bid1_start_{suffix}"].where(d[f"bid1_start_{suffix}"].gt(0))
    ask0 = d[f"ask1_start_{suffix}"].where(d[f"ask1_start_{suffix}"].gt(0))
    price0 = d[f"price_start_{suffix}"]
    buys = d[f"buy_quote_{suffix}"]
    sells = d[f"sell_quote_{suffix}"]
    d[f"return_{suffix}"] = d["price"] / price0 - 1.0
    d[f"bid_change_{suffix}"] = d["bid1"] / bid0 - 1.0
    d[f"ask_change_{suffix}"] = d["ask1"] / ask0 - 1.0
    d[f"ask_bid_ratio_change_{suffix}"] = (
        d["ask_bid_ratio"] / d[f"ask_bid_ratio_start_{suffix}"] - 1.0
    )
    d[f"imb_change_{suffix}"] = d["imb1"] - d[f"imb1_start_{suffix}"]
    d[f"buy_to_ask_{suffix}"] = buys / ask0
    d[f"sell_to_bid_{suffix}"] = sells / bid0
    d[f"signed_pressure_{suffix}"] = d[f"buy_to_ask_{suffix}"] - d[f"sell_to_bid_{suffix}"]
    d[f"ask_depth_residual_{suffix}"] = (d["ask1"] - ask0 + buys) / ask0
    d[f"bid_depth_residual_{suffix}"] = (d["bid1"] - bid0 + sells) / bid0
    d[f"impact_bps_per_pressure_{suffix}"] = (
        d[f"return_{suffix}"] * 1e4
        / d[f"signed_pressure_{suffix}"].where(
            d[f"signed_pressure_{suffix}"].abs() >= 1e-3
        )
    )

    core = [
        f"return_{suffix}", f"buy_to_ask_{suffix}", f"sell_to_bid_{suffix}",
        f"ask_depth_residual_{suffix}", f"bid_depth_residual_{suffix}",
    ]
    d[f"window_data_valid_{suffix}"] = (
        d[core].notna().all(axis=1)
        & d[f"start_endpoint_fresh_{suffix}"]
        & ~d[f"extreme_imbalance_{suffix}"]
    )
    d[f"ask_absorption_candidate_{suffix}"] = (
        d[f"window_data_valid_{suffix}"]
        & (d[f"signed_pressure_{suffix}"] >= 0.25)
        & (d[f"ask_depth_residual_{suffix}"] > 0.0)
        & (d[f"return_{suffix}"] <= 0.0)
    )
    d[f"bid_absorption_candidate_{suffix}"] = (
        d[f"window_data_valid_{suffix}"]
        & (d[f"signed_pressure_{suffix}"] <= -0.25)
        & (d[f"bid_depth_residual_{suffix}"] > 0.0)
        & (d[f"return_{suffix}"] >= 0.0)
    )
    return d


def finalize_recovered_day(
    dynamics: pd.DataFrame,
    day: pd.Timestamp,
    *,
    window: str = "5min",
    resolution: str = "5min",
    max_endpoint_staleness_seconds: float = 90.0,
) -> pd.DataFrame:
    """Select a UTC day, downsample, and apply endpoint-only quality gates."""
    start = day.floor("1D")
    d = dynamics[(dynamics.index >= start) & (dynamics.index < start + pd.Timedelta("1D"))].copy()
    if d.empty:
        return d
    source_snapshot_count_day = len(d)
    if resolution != "30s":
        d = d.groupby(d.index.floor(resolution), sort=True).tail(1)
    if d.empty:
        return d

    suffix = window
    source_day_bar_count = len(d)
    complete_threshold = 280 if resolution == "5min" else 2700
    d["source_snapshot_count_day"] = source_snapshot_count_day
    d["source_day_bar_count"] = source_day_bar_count
    d["source_day_complete"] = source_day_bar_count >= complete_threshold

    bar_time = d.index.floor(resolution)
    bar_end = bar_time + pd.Timedelta(resolution)
    end_stale = (bar_end - d.index).total_seconds()
    d[f"bar_end_staleness_seconds_{suffix}"] = end_stale
    d[f"end_endpoint_fresh_{suffix}"] = (
        (end_stale >= 0.0) & (end_stale <= max_endpoint_staleness_seconds)
    )
    d[f"endpoint_time_valid_{suffix}"] = (
        d[f"start_endpoint_fresh_{suffix}"] & d[f"end_endpoint_fresh_{suffix}"]
    )
    d[f"window_data_valid_{suffix}"] = (
        d[f"window_data_valid_{suffix}"] & d[f"end_endpoint_fresh_{suffix}"]
    )
    d[f"quality_valid_{suffix}"] = d[f"window_data_valid_{suffix}"]
    d[f"recovered_internal_gap_{suffix}"] = (
        d[f"quality_valid_{suffix}"] & d[f"any_raw_gap_{suffix}"]
    )
    start_time = d[f"window_start_snapshot_time_{suffix}"]
    d[f"recovered_cross_day_{suffix}"] = (
        d[f"quality_valid_{suffix}"]
        & start_time.notna()
        & (start_time.dt.floor("1D") < d.index.floor("1D"))
    )
    for side in ["ask", "bid"]:
        flag = f"{side}_absorption_candidate_{suffix}"
        d[flag] = d[flag] & d[f"quality_valid_{suffix}"]

    out = d.reset_index(names="snapshot_time")
    out.insert(1, "bar_time", out["snapshot_time"].dt.floor(resolution))
    return out
