"""Concrete trading-flow walkthrough of the final signal (pos=-clip(z90(agg imb1),+/-1.5), EWMA hl=42=7d) around the
2025-10 crash: daily agg_imb z, TARGET position (how the book tells us to lean), the day's market move, and the
strategy P&L — showing the ENTRY ramp (short builds as the book gets crowded-bid pre-crash), the HOLD through the
crash, and the EXIT (position decays back to flat as the book normalizes). pos<0 = SHORT the market, pos>0 = LONG.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def main():
    D = agg_ob().join(market_ret().rename("mkt"), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    D["z"] = zc(D["agg_imb"])
    D["pos"] = (-D["z"]).clip(-1.5, 1.5).ewm(halflife=42).mean()
    D["strat"] = D["pos"] * D["mkt"] - 0.0010 * D["pos"].diff().abs().fillna(0)
    g = D.groupby(D.index.floor("1D"))
    day = pd.DataFrame({"mkt": g["mkt"].apply(lambda x: (1 + x).prod() - 1), "z": g["z"].last(),
                        "pos": g["pos"].last(), "dpos": g["pos"].last().diff(),
                        "strat": g["strat"].apply(lambda x: (1 + x).prod() - 1)})
    w = day.loc["2025-09-27":"2025-10-19"].copy()
    w["cum"] = (1 + w["strat"]).cumprod() - 1
    print("ENTRY builds short as book crowds bid-heavy (z>0) -> HOLD through crash -> EXIT as book normalizes\n")
    print(f"{'date':11s}{'mkt day':>9s}{'book z':>8s}{'target':>8s}{'action':>9s}{'strat day':>11s}{'cum P&L':>9s}")
    for d, r in w.iterrows():
        act = "add short" if r.dpos < -0.03 else ("cover" if r.dpos > 0.03 else "hold")
        side = "SHORT" if r.pos < -0.05 else ("LONG" if r.pos > 0.05 else "flat")
        print(f"{str(d.date()):11s}{r.mkt*100:>+8.1f}%{r.z:>+8.2f}{r.pos:>+8.2f}{act:>9s}{r.strat*100:>+10.1f}%{r.cum*100:>+8.1f}%   ({side})")
    print(f"\nover the window: market {((1+w['mkt']).prod()-1)*100:+.1f}%  |  strategy {w['cum'].iloc[-1]*100:+.1f}%")
    print("WALKDONE")

if __name__ == "__main__":
    main()
