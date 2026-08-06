"""iter9: does the illiquidity signal PUMP or DUMP at the daily horizon (the strategy's horizon)?

Intraday (5m->4h) kyle_lambda/vpin XS-IC is POSITIVE and still climbing (long illiquid earns more).
The froth-crash story predicts it REVERSES by daily (illiquid froth dumps). Test directly: aggregate to
daily, cross-sectional rank-IC of daily kyle_lambda / vpin vs forward 1d/3d/5d returns, both eras,
horizon-sized block-bootstrap CI.

  IC stays POSITIVE at daily -> genuine slow long-illiquid premium (headwind to a short-froth strategy).
  IC flips NEGATIVE at daily -> pump(intraday)->dump(daily): illiquid froth reverses; a 1d short catches it.
Run:  python3 -m live.emergent_iter9_daily_illiq
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.flow_harness import CUT, xsic
from live.emergent_harness import block_ci
from live.emergent_iter5_richatoms import load_ext

MIN_BARS = 100


def main():
    D = load_ext(["symbol", "bar_time", "kyle_lambda", "vpin", "return_5min"])
    D["day"] = D["bar_time"].dt.floor("1D")
    D["lr"] = np.log1p(D["return_5min"].clip(lower=-0.99))
    g = D.groupby(["symbol", "day"])
    daily = g.agg(kyle=("kyle_lambda", "mean"), vpin=("vpin", "mean"),
                  lr=("lr", "sum"), n=("lr", "size")).reset_index()
    daily = daily[daily["n"] >= MIN_BARS].sort_values(["symbol", "day"])
    gl = daily.groupby("symbol")["lr"]
    daily["f1"] = gl.shift(-1)
    daily["f3"] = sum(gl.shift(-i) for i in (1, 2, 3))
    daily["f5"] = sum(gl.shift(-i) for i in (1, 2, 3, 4, 5))
    daily = daily.rename(columns={"day": "bar_time"})
    daily["bar_time"] = pd.to_datetime(daily["bar_time"], utc=True)
    print(f"daily rows {len(daily):,} | {daily['symbol'].nunique()} syms | "
          f"{daily['bar_time'].min().date()}..{daily['bar_time'].max().date()}\n", flush=True)
    m = {"OOS": (daily["bar_time"] < CUT).to_numpy(), "REC": (daily["bar_time"] >= CUT).to_numpy()}
    blk = {"f1": 3, "f3": 7, "f5": 10}
    print(f"{'signal':<8}{'horizon':<9}{'OOS IC [block-CI]':<30}{'REC IC [block-CI]':<30}", flush=True)
    for feat in ("kyle", "vpin"):
        for h in ("f1", "f3", "f5"):
            ic_o = xsic(daily, feat, h, row_mask=m["OOS"])
            ic_r = xsic(daily, feat, h, row_mask=m["REC"])
            ao, lo, uo = block_ci(ic_o, block_days=blk[h])
            ar, lr, ur = block_ci(ic_r, block_days=blk[h])
            so = "*" if (lo > 0 or uo < 0) else " "
            sr = "*" if (lr > 0 or ur < 0) else " "
            print(f"  {feat:<6}{h:<9}{f'{ao:+.4f}[{lo:+.4f},{uo:+.4f}]{so}':<30}"
                  f"{f'{ar:+.4f}[{lr:+.4f},{ur:+.4f}]{sr}':<30}", flush=True)
    print("\n(daily kyle/vpin = mean of 5-min; fwd = next 1/3/5 daily log-returns. + = long illiquid still "
          "earns; − = illiquid reverses/dumps by daily.)", flush=True)


if __name__ == "__main__":
    main()
