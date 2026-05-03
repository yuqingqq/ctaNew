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
import json
import logging
import os
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
MODEL_DIR = Path("models")


def _resolve_horizon_bars() -> int:
    """Resolve horizon to annualize Sharpe correctly.

    Priority: HORIZON_BARS env (same convention as paper_bot.py) > legacy
    artifact's meta (preserves the running bot's view) > newest suffixed
    artifact > 288. The legacy meta wins by default so the production h=288
    bot's cycles are summarized correctly even when an h=48 artifact exists
    alongside it (e.g. mid-migration).
    """
    env_val = os.environ.get("HORIZON_BARS")
    if env_val is not None:
        return int(env_val)
    legacy = MODEL_DIR / "v6_clean_meta.json"
    if legacy.exists():
        try:
            with legacy.open() as fh:
                return int(json.load(fh).get("horizon_bars", 288))
        except Exception:
            pass
    for meta_path in sorted(MODEL_DIR.glob("v6_clean_h*_meta.json")):
        try:
            with meta_path.open() as fh:
                return int(json.load(fh).get("horizon_bars", 288))
        except Exception:
            pass
    return 288


HORIZON_BARS = _resolve_horizon_bars()
CYCLES_PER_YEAR = 365.0 * (288.0 / HORIZON_BARS)  # h=288 → 365; h=48 → 2190


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
    # Reference numbers from multi-OOS turnover-aware tests on ORIG25, K matching:
    #   h=288 K=5 (legacy production):  Sharpe +3.30, net +26.7 bps, spread +30.7 bps
    #   h=48  K=7 (4h cadence, ORIG25): Sharpe +3.63, net +4.33 bps, spread +7.90 bps
    if HORIZON_BARS == 48:
        ref_sharpe, ref_net, ref_gross = 3.63, 4.33, 7.90
        ref_label = "h=48 K=7 ORIG25 multi-OOS @ 4.5 bps/leg taker"
    else:
        ref_sharpe, ref_net, ref_gross = 3.30, 26.7, 30.7
        ref_label = f"h={HORIZON_BARS} K=5 multi-OOS"
    _print_section(f"vs backtest expectation ({ref_label})")
    print(f"  backtest Sharpe:           {ref_sharpe:+.2f}")
    print(f"  backtest net/cycle:        {ref_net:+.2f} bps")
    print(f"  backtest spread/cycle:     {ref_gross:+.2f} bps")
    print()
    fwd_gross = realized["gross_pnl_bps"].mean()
    fwd_net = realized["net_bps"].mean()
    fwd_sharpe = _sharpe_yr(realized["net_bps"].dropna().to_numpy())
    print(f"  forward gross MtM/cycle:       {fwd_gross:+.2f} bps  "
           f"(Δ {fwd_gross - ref_gross:+.2f} vs backtest)")
    print(f"  forward net/cycle:             {fwd_net:+.2f} bps  "
           f"(Δ {fwd_net - ref_net:+.2f})")
    print(f"  forward Sharpe:                {fwd_sharpe:+.2f}     "
           f"(Δ {fwd_sharpe - ref_sharpe:+.2f})")

    if n >= 30:
        if abs(fwd_sharpe - ref_sharpe) < 1.5:
            print(f"\n  ✓ Forward Sharpe is consistent with backtest expectation.")
        elif fwd_sharpe > ref_sharpe - 3.0:
            print(f"\n  ~ Forward Sharpe within wide CI of backtest. Keep running.")
        else:
            print(f"\n  ⚠️  Forward Sharpe materially below backtest. Investigate.")
    else:
        print(f"\n  (need N>=30 for meaningful comparison; have {n})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
