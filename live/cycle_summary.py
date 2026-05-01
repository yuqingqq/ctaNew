"""Read live/state/cycles.csv and print a forward-test PnL summary.

Daily one-liner:
  python -m live.cycle_summary

Headers tell you:
  - cumulative PnL (both cost models)
  - per-cycle mean / std / annualized Sharpe + bootstrap CI when N>=30
  - rolling 7-day and 30-day Sharpe
  - hit rate (% positive cycles)
  - turnover stats
  - recent cycle table

When N (number of completed cycles) is small the CI will be wide; that's
expected. The point estimate stabilizes around N≈30-60 cycles.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml.research.alpha_v4_xs import block_bootstrap_ci

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("summary")

CYCLES_PATH = Path("live/state/cycles.csv")
CYCLES_PER_YEAR = 365.0


def _sharpe_yr(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return (arr.mean() / arr.std()) * np.sqrt(CYCLES_PER_YEAR)


def _print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=10,
                    help="rows of recent cycle table to print (default 10)")
    ap.add_argument("--cycles-path", default=str(CYCLES_PATH),
                    help=f"path to cycles.csv (default {CYCLES_PATH})")
    args = ap.parse_args()

    p = Path(args.cycles_path)
    if not p.exists():
        print(f"No cycles.csv found at {p}. Bot hasn't run yet.")
        return 1

    df = pd.read_csv(p)
    if df.empty:
        print("cycles.csv is empty.")
        return 1

    df["decision_time_utc"] = pd.to_datetime(df["decision_time_utc"], utc=True)
    df = df.sort_values("decision_time_utc").reset_index(drop=True)

    # First cycle has no realized prior PnL (no positions to mark before it).
    if "had_prev_positions" in df.columns:
        realized = df[df["had_prev_positions"] == 1].copy()
    else:
        # Legacy schema (pre Phase 2.1): use prior_n_long as proxy
        realized = df[df.get("prior_n_long", 0) > 0].copy()
    n = len(realized)

    _print_section(f"v6_clean paper-trade forward summary  (N={n} realized cycles)")
    if n == 0:
        print("\nNo realized cycles yet — bot has only opened initial positions.")
        print(f"  First cycle decision: {df['decision_time_utc'].iloc[0]}")
        print(f"  Open positions: see `python -m live.paper_bot --check-state`")
        return 0

    print(f"  First realized: {realized['decision_time_utc'].iloc[0]}")
    print(f"  Last realized:  {realized['decision_time_utc'].iloc[-1]}")
    span_days = (realized["decision_time_utc"].iloc[-1]
                  - realized["decision_time_utc"].iloc[0]).total_seconds() / 86400
    print(f"  Span:           {span_days:.1f} days")

    # ----- Per-cycle PnL stats -----
    _print_section("Per-cycle net PnL (bps, turnover-aware)")
    arr = realized["net_bps"].dropna().to_numpy()
    if len(arr):
        mean = arr.mean()
        std = arr.std()
        cum = arr.sum()
        sharpe = _sharpe_yr(arr)
        hit_rate = (arr > 0).mean()
        print(f"  mean / cycle:     {mean:+.2f} bps")
        print(f"  std / cycle:      {std:.2f} bps")
        print(f"  cumulative:       {cum:+.2f} bps  ({cum / 100:+.2f}%)")
        print(f"  hit rate:         {100 * hit_rate:.1f}%  ({int((arr > 0).sum())}/{len(arr)})")
        print(f"  Sharpe (annual):  {sharpe:+.2f}")
        if len(arr) >= 30:
            s, lo, hi = block_bootstrap_ci(arr, statistic=_sharpe_yr,
                                             block_size=min(7, max(2, len(arr) // 4)))
            print(f"  Sharpe 95% CI:    [{lo:+.2f}, {hi:+.2f}]  (block-bootstrap)")
        else:
            print(f"  Sharpe 95% CI:    (need N>=30 for bootstrap)")

    # ----- Cost decomposition -----
    _print_section("Per-cycle PnL decomposition (means, bps)")
    fund_col = realized.get("funding_bps", pd.Series([0.0] * len(realized)))
    fund_usd_col = realized.get("funding_usd", pd.Series([0.0] * len(realized)))
    print(f"  gross MtM PnL:             {realized['gross_pnl_bps'].mean():+.2f}")
    print(f"  fees (taker, delta-only):  {realized['fees_bps'].mean():.2f}")
    print(f"  slippage (L2-walk):        {realized['slippage_bps'].mean():.2f}")
    print(f"  funding (HL hourly):       {fund_col.mean():+.2f}     "
           f"(${fund_usd_col.mean():+.2f}/cycle in USD)")
    print(f"  net:                       {realized['net_bps'].mean():+.2f}")
    print(f"  trades / cycle (mean):     {realized['n_trades'].mean():.1f}")
    print(f"  trade notional / cycle:    ${realized['trade_notional_usd'].mean():,.0f}")
    # Funding is often the biggest persistent drag; flag if it's a meaningful
    # share of gross.
    gross_mean = realized["gross_pnl_bps"].mean()
    fund_mean = fund_col.mean()
    if abs(gross_mean) > 0.01:
        share = 100 * fund_mean / abs(gross_mean)
        if abs(share) > 20:
            print(f"  ⚠️ funding is {share:+.0f}% of |gross PnL| — meaningful drag")

    # ----- Rolling Sharpe -----
    _print_section("Rolling Sharpe (turnover-aware net_bps)")
    for window in (7, 30):
        if n < window + 1:
            print(f"  {window}d rolling: need {window + 1}+ cycles, have {n}")
            continue
        rolled = realized["net_bps"].rolling(window).apply(_sharpe_yr, raw=True)
        latest = rolled.iloc[-1]
        worst = rolled.min()
        best = rolled.max()
        print(f"  {window}d:  latest={latest:+.2f}  best={best:+.2f}  worst={worst:+.2f}")

    # ----- Recent cycles -----
    _print_section(f"Last {min(args.last, n)} cycles")
    cols_show = ["decision_time_utc", "long_symbols", "short_symbols",
                 "gross_pnl_bps", "fees_bps", "slippage_bps", "net_bps",
                 "n_trades", "trade_notional_usd"]
    cols_show = [c for c in cols_show if c in realized.columns]
    print(realized[cols_show].tail(args.last).to_string(index=False))

    # ----- Comparison to backtest expectation -----
    _print_section("vs backtest expectation")
    print(f"  v6_clean multi-OOS Sharpe (backtest, K=5+VIP-3+maker): +2.95")
    print(f"  v6_clean multi-OOS net/cycle (backtest):              +26.7 bps")
    print(f"  v6_clean multi-OOS spread/cycle gross:                +30.7 bps")
    print()
    fwd_gross = realized["gross_pnl_bps"].mean()
    fwd_net = realized["net_bps"].mean()
    fwd_sharpe = _sharpe_yr(realized["net_bps"].dropna().to_numpy())
    print(f"  forward gross MtM/cycle:       {fwd_gross:+.2f} bps  "
           f"(backtest gross: +30.7, Δ {fwd_gross - 30.7:+.2f})")
    print(f"  forward net/cycle:             {fwd_net:+.2f} bps  "
           f"(backtest net: +26.7, Δ {fwd_net - 26.7:+.2f})")
    print(f"  forward Sharpe:                {fwd_sharpe:+.2f}     "
           f"(backtest: +2.95, Δ {fwd_sharpe - 2.95:+.2f})")

    if n >= 30:
        if abs(fwd_sharpe - 2.95) < 1.5:
            print(f"\n  ✓ Forward Sharpe is consistent with backtest expectation.")
        elif fwd_sharpe > 2.95 - 3.0:
            print(f"\n  ~ Forward Sharpe within wide CI of backtest. Keep running.")
        else:
            print(f"\n  ⚠️  Forward Sharpe materially below backtest. Investigate.")
    else:
        print(f"\n  (need N>=30 for meaningful comparison; have {n})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
