"""iter10: SHOW that "illiquid drifts up" (broad) and "strategy shorts the crash" (froth) are different things.

2-D cross-sectional sort, daily: illiquidity (kyle_lambda) tercile × recent run-up (trailing 5d return,
the froth proxy) tercile. Cell = mean forward 3d return (bps), both eras. Marginals isolate each effect.

Expect:
  - illiquidity MARGINAL (avg across run-up) = POSITIVE, bigger for illiquid  -> the broad long-illiquid drift.
  - run-up MARGINAL = NEGATIVE for high run-up (froth crashes), positive for beaten-down (bounce) -> reversal.
  - the HIGH-illiq × HIGH-run-up CELL = the frothy-illiquid CRASH = what the strategy SHORTS.
So the strategy shorts one CELL (crash); the drift is the ROW AVERAGE (positive). Different bets.
Run:  python3 -m live.emergent_iter10_reconcile
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT
from live.emergent_iter5_richatoms import load_ext

MIN_BARS = 100


def terciles_by_day(df, col):
    r = df.groupby("day")[col].rank(pct=True)
    return np.select([r <= 1/3, r <= 2/3], [0, 1], default=2)


def grid(sub, tag):
    g = sub.groupby(["il", "ru"])["f3d"].mean().unstack() * 1e4
    print(f"\n  [{tag}] MARKET-NEUTRAL (daily-demeaned) forward-3d return (bps), rows illiquidity × cols run-up:",
          flush=True)
    print(f"    {'':<12}{'run-up LOW':>12}{'run-up MID':>12}{'run-up HIGH':>12}   {'ROW AVG':>9}", flush=True)
    for il, name in [(0, "illiq LOW"), (1, "illiq MID"), (2, "illiq HIGH")]:
        row = g.loc[il]
        avg = row.mean()
        print(f"    {name:<12}{row[0]:>12.1f}{row[1]:>12.1f}{row[2]:>12.1f}   {avg:>9.1f}", flush=True)
    colavg = g.mean(axis=0)
    print(f"    {'COL AVG':<12}{colavg[0]:>12.1f}{colavg[1]:>12.1f}{colavg[2]:>12.1f}", flush=True)


def main():
    D = load_ext(["symbol", "bar_time", "kyle_lambda", "return_5min"])
    D["day"] = D["bar_time"].dt.floor("1D")
    D["lr"] = np.log1p(D["return_5min"].clip(lower=-0.99))
    daily = (D.groupby(["symbol", "day"])
             .agg(kyle=("kyle_lambda", "mean"), lr=("lr", "sum"), n=("lr", "size"))
             .reset_index())
    daily = daily[daily["n"] >= MIN_BARS].sort_values(["symbol", "day"])
    gl = daily.groupby("symbol")["lr"]
    daily["ru5"] = gl.rolling(5).sum().reset_index(level=0, drop=True)     # trailing 5d run-up (known at t)
    daily["f3"] = sum(gl.shift(-i) for i in (1, 2, 3))                     # forward 3d
    daily = daily.dropna(subset=["kyle", "ru5", "f3"])
    daily["f3d"] = daily["f3"] - daily.groupby("day")["f3"].transform("mean")  # market-neutral (demean/day)
    daily["il"] = terciles_by_day(daily, "kyle")
    daily["ru"] = terciles_by_day(daily, "ru5")
    daily["bar_time"] = pd.to_datetime(daily["day"], utc=True)
    print(f"daily rows {len(daily):,} | {daily['symbol'].nunique()} syms", flush=True)
    grid(daily[daily["bar_time"] < CUT], "OOS")
    grid(daily[daily["bar_time"] >= CUT], "REC")
    print("\n  READ: ROW AVG (illiq HIGH) = broad long-illiquid drift; the illiq-HIGH × run-up-HIGH CELL = "
          "frothy-illiquid CRASH = what the strategy shorts. Different cells = different bets.", flush=True)


if __name__ == "__main__":
    main()
