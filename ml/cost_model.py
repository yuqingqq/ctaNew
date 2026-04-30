"""Cost model for net-of-cost backtest evaluation.

Components:
    Fees      — maker/taker, per-side
    Spread    — 0.5 × bid-ask spread, per-side. Without L2 we use Roll's (1984)
                effective-spread estimator from trade prices alone.
    Slippage  — λ · √(size / depth). Without depth we fall back to a flat
                bps floor; flagged in the report as a conservative shortcut.
    Funding   — perpetual funding payments accrued over holding bars.

For the gate (i) probe (kline+tape only, no L2), `effective_spread_roll` gives
a reasonable spread proxy from aggTrades alone. For gate (ii) we will swap in
real L2-snapshot spreads from Tardis.

All return values are in fractional units (not bps) unless otherwise noted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    fee_maker: float = 0.0002       # 0.02% — Binance Futures USDM VIP-0
    fee_taker: float = 0.0005       # 0.05% — Binance Futures USDM VIP-0
    is_taker_entry: bool = True
    is_taker_exit: bool = True
    flat_slippage_bps: float = 1.0  # fallback when L2 depth unknown; 1 bp per side
    use_roll_spread: bool = True    # add Roll spread proxy on top of slippage
    funding_rate_per_bar: float = 0.0  # placeholder; pass realized series via apply_costs


def fee_per_side(config: CostConfig, *, taker: bool) -> float:
    return config.fee_taker if taker else config.fee_maker


def effective_spread_roll(
    trade_prices: pd.Series,
    bar_interval: str = "5min",
) -> pd.Series:
    """Roll (1984) effective-spread estimator at the bar level.

    For trade-price changes Δp_t, the serial autocovariance under bid-ask
    bounce is cov(Δp_t, Δp_{t-1}) = -s²/4 where s is the effective spread.
    Per bar we estimate s = 2 · √(-cov) when cov < 0, else 0.

    Returns a Series indexed by bar `open_time` with effective spread in the
    same price units as `trade_prices`.
    """
    trades = trade_prices.copy()
    trades.index.name = "transact_time"
    bar_open = trades.index.floor(bar_interval)
    df = pd.DataFrame({"price": trades.values, "bar": bar_open}, index=trades.index)
    df["dp"] = df["price"].diff()

    out = {}
    for bar, sub in df.groupby("bar", sort=True):
        dp = sub["dp"].dropna()
        if len(dp) < 20:
            out[bar] = np.nan
            continue
        cov = float(dp.cov(dp.shift(1)))
        if not np.isfinite(cov) or cov >= 0:
            out[bar] = 0.0
        else:
            out[bar] = 2.0 * np.sqrt(-cov)
    s = pd.Series(out, name="roll_spread")
    s.index.name = "open_time"
    return s


def round_trip_cost_bps(
    config: CostConfig,
    spread_at_entry_bps: float,
    spread_at_exit_bps: float,
    funding_total: float = 0.0,
) -> float:
    """Total round-trip cost in basis points.

    spread_at_*_bps are 1e4 × spread/price ratios at entry/exit times.
    `funding_total` is the cumulative funding payment over the holding period
    (positive means cost; negative means receipt) in fractional units.
    """
    fee = fee_per_side(config, taker=config.is_taker_entry) + fee_per_side(config, taker=config.is_taker_exit)
    slip = 2 * config.flat_slippage_bps / 1e4
    spread_term = 0.0
    if config.use_roll_spread:
        spread_term = 0.5 * (spread_at_entry_bps + spread_at_exit_bps) / 1e4
    return 1e4 * (fee + slip + spread_term + funding_total)


def apply_costs_to_returns(
    returns: pd.Series,
    spread_bps_at_entry: pd.Series,
    spread_bps_at_exit: pd.Series,
    config: CostConfig = CostConfig(),
    funding_total: pd.Series | None = None,
) -> pd.Series:
    """Subtract round-trip costs from gross returns.

    Parameters
    ----------
    returns : Series of gross fractional returns indexed by entry_time.
    spread_bps_at_entry, spread_bps_at_exit : Series indexed by entry_time
        with bar-level spread proxy in bps. Use `effective_spread_roll`-derived
        series. NaN spreads are treated as 0 (assumes the bar's spread was
        unknown — costs will then come from fees + flat slippage only).
    funding_total : optional Series of cumulative funding paid per trade.
    """
    fee = fee_per_side(config, taker=config.is_taker_entry) + fee_per_side(config, taker=config.is_taker_exit)
    slip = 2 * config.flat_slippage_bps / 1e4
    spread_term = pd.Series(0.0, index=returns.index)
    if config.use_roll_spread:
        sp_entry = spread_bps_at_entry.reindex(returns.index).fillna(0.0)
        sp_exit = spread_bps_at_exit.reindex(returns.index).fillna(0.0)
        spread_term = 0.5 * (sp_entry + sp_exit) / 1e4
    funding_term = pd.Series(0.0, index=returns.index)
    if funding_total is not None:
        funding_term = funding_total.reindex(returns.index).fillna(0.0)
    return returns - fee - slip - spread_term - funding_term
