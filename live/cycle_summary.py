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

    # First cycle has no realized prior P&L — drop for stats.
    realized = df[df["prior_n_long"] > 0].copy()
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

    # ----- Per-cycle stats: two cost models -----
    _print_section("Per-cycle PnL (bps), two cost models")
    for label, col in [("close-all + reopen-all (conservative)", "net_bps"),
                        ("turnover-aware (matches backtest)", "tt_net_bps")]:
        arr = realized[col].dropna().to_numpy()
        if len(arr) == 0:
            continue
        mean = arr.mean()
        std = arr.std()
        cum = arr.sum()
        sharpe = _sharpe_yr(arr)
        hit_rate = (arr > 0).mean()
        print(f"\n  [{label}]")
        print(f"    mean / cycle:     {mean:+.2f} bps")
        print(f"    std / cycle:      {std:.2f} bps")
        print(f"    cumulative:       {cum:+.2f} bps  ({cum / 100:+.2f}%)")
        print(f"    hit rate:         {100 * hit_rate:.1f}%  ({int((arr > 0).sum())}/{len(arr)})")
        print(f"    Sharpe (annual):  {sharpe:+.2f}")
        if len(arr) >= 30:
            s, lo, hi = block_bootstrap_ci(arr, statistic=_sharpe_yr,
                                             block_size=min(7, max(2, len(arr) // 4)))
            print(f"    Sharpe 95% CI:    [{lo:+.2f}, {hi:+.2f}]  (block-bootstrap)")
        else:
            print(f"    Sharpe 95% CI:    (need N>=30 for bootstrap)")

    # ----- Spread / slippage / fee decomposition -----
    _print_section("Cost decomposition (means, bps)")
    print(f"  spread_ret (gross):        {realized['prior_spread_ret_bps'].mean():+.2f}")
    print(f"  entry slippage:            {realized['prior_entry_slip_bps_mean'].mean():+.2f}")
    print(f"  exit slippage:             {realized['prior_exit_slip_bps_mean'].mean():+.2f}")
    print(f"  fees (close-all):          {realized['prior_fees_bps'].mean():.2f}")
    print(f"  fees (turnover-aware):     {realized['tt_fees_bps'].mean():.2f}")
    print(f"  long turnover (mean):      {realized['tt_long_turnover'].mean():.3f}")
    print(f"  short turnover (mean):     {realized['tt_short_turnover'].mean():.3f}")

    # ----- Rolling Sharpe -----
    _print_section("Rolling Sharpe (turnover-aware net_bps)")
    for window in (7, 30):
        if n < window + 1:
            print(f"  {window}d rolling: need {window + 1}+ cycles, have {n}")
            continue
        rolled = realized["tt_net_bps"].rolling(window).apply(_sharpe_yr, raw=True)
        latest = rolled.iloc[-1]
        worst = rolled.min()
        best = rolled.max()
        print(f"  {window}d:  latest={latest:+.2f}  best={best:+.2f}  worst={worst:+.2f}")

    # ----- Recent cycles -----
    _print_section(f"Last {min(args.last, n)} cycles")
    cols_show = ["decision_time_utc", "long_symbols", "short_symbols",
                 "prior_spread_ret_bps", "tt_net_bps", "net_bps",
                 "tt_long_turnover", "tt_short_turnover"]
    cols_show = [c for c in cols_show if c in realized.columns]
    print(realized[cols_show].tail(args.last).to_string(index=False))

    # ----- Comparison to backtest expectation -----
    _print_section("vs backtest expectation")
    print(f"  v6_clean multi-OOS Sharpe (backtest, K=5+VIP-3+maker): +2.95")
    print(f"  v6_clean multi-OOS net/cycle (backtest):              +26.7 bps")
    print(f"  v6_clean multi-OOS spread/cycle gross:                +30.7 bps")
    print()
    fwd_spread = realized["prior_spread_ret_bps"].mean()
    fwd_tt_net = realized["tt_net_bps"].mean()
    fwd_tt_sharpe = _sharpe_yr(realized["tt_net_bps"].dropna().to_numpy())
    print(f"  forward spread/cycle gross:    {fwd_spread:+.2f} bps  "
           f"(backtest: +30.7, Δ {fwd_spread - 30.7:+.2f})")
    print(f"  forward net/cycle (TT-aware):  {fwd_tt_net:+.2f} bps  "
           f"(backtest: +26.7, Δ {fwd_tt_net - 26.7:+.2f})")
    print(f"  forward Sharpe (TT-aware):     {fwd_tt_sharpe:+.2f}     "
           f"(backtest: +2.95, Δ {fwd_tt_sharpe - 2.95:+.2f})")

    if n >= 30:
        if abs(fwd_tt_sharpe - 2.95) < 1.5:
            print(f"\n  ✓ Forward Sharpe is consistent with backtest expectation.")
        elif fwd_tt_sharpe > 2.95 - 3.0:
            print(f"\n  ~ Forward Sharpe within wide CI of backtest. Keep running.")
        else:
            print(f"\n  ⚠️  Forward Sharpe materially below backtest. Investigate.")
    else:
        print(f"\n  (need N>=30 for meaningful comparison; have {n})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
