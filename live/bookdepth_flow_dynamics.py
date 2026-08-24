"""Research extraction of order-book response dynamics from public snapshots.

Align consecutive Binance Vision 30-second ``bookDepth`` snapshots with every
``aggTrade`` that occurred between them.  The output keeps the ingredients of
an absorption hypothesis separate:

* displayed bid/ask notional and their changes;
* aggressive buy/sell quote notional;
* flow normalized by the depth it met;
* a residual-depth proxy after subtracting aggressive consumption; and
* price response per unit of normalized aggressive flow.

The residual-depth fields are *proxies*, not queue reconstruction.  A snapshot
change also contains cancellations and the effect of percentage bands moving
with price.  Consequently this module calculates auditable components and
candidate flags rather than declaring a fitted alpha score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from live.bookdepth_loader import _fetch_day


REPO = Path("/home/yuqing/ctaNew")
AGG_ROOT = REPO / "data/ml/test/parquet/aggTrades"


def _load_book(symbol: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    parts = []
    for day in days:
        d = _fetch_day(symbol, day)
        if d is not None and not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    b = pd.concat(parts).sort_index()
    b.index = pd.DatetimeIndex(b.index).as_unit("ns")
    b = b[~b.index.duplicated(keep="last")]

    liq1 = np.exp(b["liq1"])
    imb1 = b["imb1"]
    b["bid1"] = liq1 * (1.0 + imb1) / 2.0
    b["ask1"] = liq1 * (1.0 - imb1) / 2.0

    # The +/-0.2% band is missing from much of the older archive.  Preserve it
    # when available but use +/-1% as the cross-era primary measurement.
    liq02 = liq1 * b["touch"]
    imb02 = b["imb02"]
    b["bid02"] = liq02 * (1.0 + imb02) / 2.0
    b["ask02"] = liq02 * (1.0 - imb02) / 2.0
    return b[["bid1", "ask1", "bid02", "ask02", "imb1", "imb02"]]


def _load_trades(symbol: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    parts = []
    for day in days:
        path = AGG_ROOT / symbol / f"{day:%Y-%m-%d}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(
            path,
            columns=[
                "agg_trade_id", "price", "quantity", "transact_time",
                "is_buyer_maker",
            ],
        )
        if not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    t["transact_time"] = pd.to_datetime(t["transact_time"], utc=True).dt.as_unit("ns")
    # Daily files do not overlap. Do not deduplicate on time/price/size:
    # distinct aggregate trades can legitimately share those attributes.
    # Multiple aggregate trades can share a millisecond.  Exchange sequence is
    # then determined by agg_trade_id; sorting only by time makes the as-of
    # price depend on the input/concat order.
    if not (
        t["transact_time"].is_monotonic_increasing
        and t["agg_trade_id"].is_monotonic_increasing
    ):
        t = t.sort_values(["transact_time", "agg_trade_id"], kind="mergesort")
    t["quote"] = t["price"].astype(float) * t["quantity"].astype(float)
    t["buy_quote"] = np.where(~t["is_buyer_maker"], t["quote"], 0.0)
    t["sell_quote"] = np.where(t["is_buyer_maker"], t["quote"], 0.0)
    return t


def _align_intervals(book: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Assign trades in ``(snapshot[i-1], snapshot[i]]`` to snapshot ``i``."""
    if book.empty or trades.empty:
        return pd.DataFrame()
    out = book.copy()
    snap_ns = out.index.as_unit("ns").asi8
    trade_ns = trades["transact_time"].array.asi8

    # First snapshot at or after each trade closes that trade's observation
    # interval.  This avoids putting trades that happened after a snapshot into
    # the depth change ending at that snapshot.
    pos = np.searchsorted(snap_ns, trade_ns, side="left")
    valid = (pos > 0) & (pos < len(out))
    flow = trades.loc[valid, ["buy_quote", "sell_quote"]].copy()
    flow["_pos"] = pos[valid]
    flow["buy_count"] = (flow["buy_quote"] > 0).astype(int)
    flow["sell_count"] = (flow["sell_quote"] > 0).astype(int)
    agg = flow.groupby("_pos").agg(
        buy_quote=("buy_quote", "sum"),
        sell_quote=("sell_quote", "sum"),
        buy_count=("buy_count", "sum"),
        sell_count=("sell_count", "sum"),
    )
    for col in ["buy_quote", "sell_quote", "buy_count", "sell_count"]:
        values = np.zeros(len(out), dtype=float)
        values[agg.index.to_numpy(dtype=int)] = agg[col].to_numpy(dtype=float)
        out[col] = values

    # Last traded price known at each snapshot, strictly using trades at or
    # before the snapshot timestamp.
    last_pos = np.searchsorted(trade_ns, snap_ns, side="right") - 1
    price = np.full(len(out), np.nan)
    has_price = last_pos >= 0
    price[has_price] = trades["price"].to_numpy(dtype=float)[last_pos[has_price]]
    out["price"] = price
    out["interval_seconds"] = out.index.to_series().diff().dt.total_seconds()

    # Do not interpret a multi-minute archive gap as one normal 30-second
    # depletion/replenishment interval.
    gap = out["interval_seconds"].gt(90.0)
    out.loc[gap, ["buy_quote", "sell_quote", "buy_count", "sell_count"]] = np.nan
    return out


def _lag_at_or_before(frame: pd.DataFrame, column: str, window: pd.Timedelta) -> pd.Series:
    idx_ns = frame.index.as_unit("ns").asi8
    target = idx_ns - window.value
    pos = np.searchsorted(idx_ns, target, side="right") - 1
    values = np.full(len(frame), np.nan)
    valid = pos >= 0
    values[valid] = frame[column].to_numpy(dtype=float)[pos[valid]]
    return pd.Series(values, index=frame.index)


def build_dynamics(
    symbol: str,
    days: pd.DatetimeIndex,
    *,
    window: str = "5min",
) -> pd.DataFrame:
    """Build snapshot-interval and trailing-window response components."""
    book = _load_book(symbol, days)
    trades = _load_trades(symbol, days)
    d = _align_intervals(book, trades)
    if d.empty:
        return d

    d["symbol"] = symbol
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

    # Displayed-depth balance: A_t = A_{t-1} - aggressive_buys + residual.
    # Positive residual means net replenishment after executions. It still
    # includes cancellations and moving-band composition.
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
        "buy_to_ask", "sell_to_bid",
        "signed_pressure", "ask_depth_residual", "bid_depth_residual",
        "return_bps", "impact_bps_per_pressure",
    ]] = np.nan

    w = pd.Timedelta(window)
    min_periods = max(2, int(w / pd.Timedelta("30s") * 0.6))
    d[f"buy_quote_{window}"] = d["buy_quote"].rolling(window, min_periods=min_periods).sum()
    d[f"sell_quote_{window}"] = d["sell_quote"].rolling(window, min_periods=min_periods).sum()
    d[f"snapshot_count_{window}"] = d["interval_seconds"].rolling(window, min_periods=1).count()
    d[f"gap_count_{window}"] = d["gap_interval"].rolling(window, min_periods=1).sum()
    d[f"max_interval_seconds_{window}"] = d["interval_seconds"].rolling(window, min_periods=1).max()
    d[f"any_raw_gap_{window}"] = d[f"gap_count_{window}"].gt(0)
    d["extreme_imbalance_1pct"] = d["imb1"].abs().gt(0.999)
    for side in ["bid1", "ask1", "price", "imb1", "ask_bid_ratio"]:
        d[f"{side}_start_{window}"] = _lag_at_or_before(d, side, w)
    d[f"extreme_imbalance_{window}"] = (
        d["extreme_imbalance_1pct"]
        | d[f"imb1_start_{window}"].abs().gt(0.999)
    )

    bid0 = d[f"bid1_start_{window}"].where(d[f"bid1_start_{window}"].gt(0))
    ask0 = d[f"ask1_start_{window}"].where(d[f"ask1_start_{window}"].gt(0))
    price0 = d[f"price_start_{window}"]
    buys = d[f"buy_quote_{window}"]
    sells = d[f"sell_quote_{window}"]
    suffix = window
    d[f"return_{suffix}"] = d["price"] / price0 - 1.0
    d[f"bid_change_{suffix}"] = d["bid1"] / bid0 - 1.0
    d[f"ask_change_{suffix}"] = d["ask1"] / ask0 - 1.0
    d[f"ask_bid_ratio_change_{suffix}"] = (
        d["ask_bid_ratio"] / d[f"ask_bid_ratio_start_{window}"] - 1.0
    )
    d[f"imb_change_{suffix}"] = d["imb1"] - d[f"imb1_start_{window}"]
    d[f"buy_to_ask_{suffix}"] = buys / ask0
    d[f"sell_to_bid_{suffix}"] = sells / bid0
    d[f"signed_pressure_{suffix}"] = d[f"buy_to_ask_{suffix}"] - d[f"sell_to_bid_{suffix}"]
    d[f"ask_depth_residual_{suffix}"] = (d["ask1"] - ask0 + buys) / ask0
    d[f"bid_depth_residual_{suffix}"] = (d["bid1"] - bid0 + sells) / bid0
    d[f"impact_bps_per_pressure_{suffix}"] = (
        d[f"return_{suffix}"] * 1e4
        / d[f"signed_pressure_{suffix}"].where(d[f"signed_pressure_{suffix}"].abs() >= 1e-3)
    )

    # Window-quality flags are kept separate from the raw components so tests
    # can enforce identical data-quality rules without discarding diagnostics.
    core = [
        f"return_{suffix}", f"buy_to_ask_{suffix}", f"sell_to_bid_{suffix}",
        f"ask_depth_residual_{suffix}", f"bid_depth_residual_{suffix}",
    ]
    d[f"window_data_valid_{suffix}"] = (
        d[core].notna().all(axis=1)
        & ~d[f"any_raw_gap_{suffix}"]
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


def _print_audit(d: pd.DataFrame, window: str, top: int) -> None:
    print(
        f"rows {len(d)} | {d.index.min()} .. {d.index.max()} | "
        f"cadence median {d.interval_seconds.median():.0f}s p99 {d.interval_seconds.quantile(.99):.0f}s"
    )
    cols = [
        f"return_{window}", f"buy_to_ask_{window}", f"sell_to_bid_{window}",
        f"ask_depth_residual_{window}", f"bid_depth_residual_{window}",
        f"impact_bps_per_pressure_{window}",
    ]
    print("\nComponent distribution:")
    print(d[cols].quantile([0.1, 0.5, 0.9]).T.round(4).to_string())
    for side in ["ask", "bid"]:
        flag = f"{side}_absorption_candidate_{window}"
        sub = d[d[flag]].copy()
        print(f"\n{flag}: {len(sub)} rows")
        if sub.empty:
            continue
        pressure = f"buy_to_ask_{window}" if side == "ask" else f"sell_to_bid_{window}"
        residual = f"{side}_depth_residual_{window}"
        show = sub.nlargest(top, pressure)[
            ["symbol", "price", f"return_{window}", pressure, residual,
             f"impact_bps_per_pressure_{window}"]
        ]
        print(show.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--start", default="2025-09-15")
    ap.add_argument("--end", default=None)
    ap.add_argument("--window", default="5min")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional parquet output path")
    args = ap.parse_args()
    end = args.end or args.start
    days = pd.date_range(args.start, end, freq="D", tz="UTC")
    d = build_dynamics(args.symbol.upper(), days, window=args.window)
    if d.empty:
        raise SystemExit("no overlapping bookDepth and aggTrades")
    _print_audit(d, args.window, args.top)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        d.reset_index(names="snapshot_time").to_parquet(path, index=False)
        print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("\nFLOWDYNDONE")


if __name__ == "__main__":
    main()
