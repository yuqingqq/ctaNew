"""Regression test: cycle gross_pnl_bps must reference last_cycle_mid, not
last_marked_mid. Catches the bug where hourly_monitor mutates
last_marked_mid between cycles, causing the next cycle's reported gross
MtM to under-state the holding-window PnL by O(100 bps).

Run: `python -m live.test_cycle_isolation` — exits non-zero on failure.

This complements replay_paper_bot.py (which validates the prediction path
end-to-end but skips hourly_monitor and so cannot exercise this issue).
"""
from __future__ import annotations

import sys

import numpy as np

from live.paper_bot import (
    LegPosition, execute_cycle_turnover_aware, INITIAL_EQUITY_USD,
    _binance_to_hl_coin,
)


def _book(bid: float, ask: float) -> dict:
    """Synthetic L2 book: single level each side, deep enough that any
    realistic taker fill walks one level only."""
    return {"bids": [[bid, 1e9]], "asks": [[ask, 1e9]]}


def _mid_book(mid: float) -> dict:
    return _book(mid * (1 - 1e-5), mid * (1 + 1e-5))


def main() -> int:
    sym_long = "BTCUSDT"
    sym_short = "ETHUSDT"

    # Cycle N: opened both legs at price=100. last_cycle_mid recorded at 100.
    cycle_basis = 100.0
    prev = [
        LegPosition(
            symbol=sym_long, side="L", weight=+0.5,
            entry_price_hl=cycle_basis, entry_mid_hl=cycle_basis,
            entry_notional_usd=0.5 * INITIAL_EQUITY_USD,
            entry_slippage_bps=0.0, entry_time="2026-05-01T00:00:00+00:00",
            last_marked_mid=cycle_basis, last_cycle_mid=cycle_basis,
        ),
        LegPosition(
            symbol=sym_short, side="S", weight=-0.5,
            entry_price_hl=cycle_basis, entry_mid_hl=cycle_basis,
            entry_notional_usd=0.5 * INITIAL_EQUITY_USD,
            entry_slippage_bps=0.0, entry_time="2026-05-01T00:00:00+00:00",
            last_marked_mid=cycle_basis, last_cycle_mid=cycle_basis,
        ),
    ]

    # Between cycle N and cycle N+1, hourly_monitor runs many times and
    # mutates last_marked_mid (but NOT last_cycle_mid). Drift the two legs
    # ASYMMETRICALLY so β-neutral cancellation can't hide a bug regression.
    long_marked = 110.0   # +10% mark drift
    short_marked = 95.0   # -5%  mark drift
    prev[0].last_marked_mid = long_marked
    prev[1].last_marked_mid = short_marked

    # Cycle N+1: actual mid is 105 for both. Target == prev (no trades).
    mid_now = 105.0
    target_weights = {p.symbol: p.weight for p in prev}
    books = {
        _binance_to_hl_coin(sym_long): _mid_book(mid_now),
        _binance_to_hl_coin(sym_short): _mid_book(mid_now),
    }

    res = execute_cycle_turnover_aware(
        prev, target_weights, books,
        now_iso="2026-05-02T00:00:00+00:00",
        equity_usd=INITIAL_EQUITY_USD,
    )

    # Expected: gross PnL measured from last_cycle_mid=100 for both legs.
    expected = ((mid_now / cycle_basis - 1) * 0.5
                + (cycle_basis / mid_now - 1) * 0.5) * 1e4
    # Buggy: gross from drifted last_marked_mid (long 110, short 95).
    buggy = ((mid_now / long_marked - 1) * 0.5
             + (short_marked / mid_now - 1) * 0.5) * 1e4
    got = res["gross_pnl_bps"]
    print(f"Expected gross_pnl_bps (from last_cycle_mid=100): {expected:+.4f}")
    print(f"Got from execute_cycle_turnover_aware:            {got:+.4f}")
    print(f"What the buggy path would have returned:          {buggy:+.4f}")

    if abs(got - expected) > 0.01:
        print("✗  FAIL: gross_pnl_bps does not match last_cycle_mid basis.")
        return 1
    if abs(got - buggy) < 50:  # asymmetric drift => buggy off by hundreds of bps
        print(f"✗  FAIL: gross_pnl_bps too close to buggy basis — possible regression.")
        return 1

    # Sanity: surviving positions should have last_cycle_mid advanced to mid_now.
    for p in res["new_positions"]:
        if abs(p.last_cycle_mid - mid_now) > 1e-6:
            print(f"✗  FAIL: {p.symbol}.last_cycle_mid={p.last_cycle_mid} "
                  f"not advanced to mid_now={mid_now}")
            return 1

    print("✓  PASS: cycle gross PnL is independent of hourly mark drift; "
          "last_cycle_mid advances each cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
