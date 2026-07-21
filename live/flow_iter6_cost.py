"""iter6 (gate #4 — cost/capacity): turn 'sub-cost' from an IC-translation into a real net backtest.

Build the actual 5-min cross-sectional dollar-neutral portfolio for the best both-era signals,
apply per-rebalance cost, and report gross vs net annualized Sharpe + break-even cost, both eras.
Signals: imb_change_5min (the cleanest both-era continuation) and signed_pressure_5min.
A 5-min signal rebalances every bar -> turnover is huge -> this quantifies exactly how sub-cost it is.
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd
from live.flow_harness import SLIM, CUT

BARS_PER_YEAR = 288 * 365
SIGNALS = {"imb_change_5min": +1, "signed_pressure_5min": +1}  # sign = long-high (continuation)


def load_long(sig):
    parts = []
    for f in sorted(glob.glob(f"{SLIM}/*.parquet")):
        d = pd.read_parquet(f, columns=["symbol", "bar_time", sig, "fwd_5m"])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    return d


def backtest(d, sig, sign):
    ws = d.pivot_table(index="bar_time", columns="symbol", values=sig)
    wf = d.pivot_table(index="bar_time", columns="symbol", values="fwd_5m")
    wf = wf.reindex_like(ws)
    S = sign * ws.to_numpy(dtype=np.float64)
    F = wf.to_numpy(dtype=np.float64)
    mu = np.nanmean(S, axis=1, keepdims=True)
    Sd = S - mu
    denom = np.nansum(np.abs(Sd), axis=1, keepdims=True)
    denom[denom == 0] = np.nan
    W = np.where(np.isnan(Sd), 0.0, Sd) / denom     # dollar-neutral, gross ~1 each bar
    Fz = np.where(np.isnan(F), 0.0, F)
    gross = np.nansum(W * Fz, axis=1)                # per-bar gross return
    turn = np.empty(len(W)); turn[0] = np.nan
    turn[1:] = np.abs(np.diff(W, axis=0)).sum(axis=1)
    idx = ws.index
    return pd.DataFrame({"gross": gross, "turn": turn}, index=idx)


def sharpe(x):
    x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(BARS_PER_YEAR) if len(x) > 10 and x.std() > 0 else np.nan


def main():
    for sig, sign in SIGNALS.items():
        print(f"\n===== signal = {sig} (long-high) =====")
        d = load_long(sig)
        bt = backtest(d, sig, sign)
        for era, m in [("OOS", bt.index < CUT), ("REC", bt.index >= CUT)]:
            sub = bt[m]
            g = sub["gross"].to_numpy(); t = sub["turn"].to_numpy()
            gs = sharpe(g)
            avg_turn = np.nanmean(t)
            print(f"  [{era}] gross Sharpe {gs:+.2f} | avg turnover/bar {avg_turn:.2f} "
                  f"| mean gross/bar {np.nanmean(g)*1e4:+.3f}bps")
            row = "        net Sharpe @cost: "
            be = None
            for c in [0, 0.5, 1, 2, 5, 10]:
                net = g - c / 1e4 * t
                ns = sharpe(net)
                row += f"{c}bp {ns:+.2f}  "
                if be is None and ns < 0:
                    be = c
            print(row)
            # break-even cost (net mean = 0): cost_be = mean(gross)/mean(turn) in bps
            cbe = np.nanmean(g) / np.nanmean(t) * 1e4
            print(f"        break-even cost = {cbe:.3f} bps/rebalance (net PnL=0 above this)")
    print("\nread: 5-min signals rebalance every bar; break-even cost << realistic 2-10bps => sub-cost. ITER6DONE",
          flush=True)


if __name__ == "__main__":
    main()
