"""The synthesized signal (strength + price + OB): contrarian on the aggregate +/-1% book crowding, SCALED by strength
(raw = -clip(z,+/-1.5), act big on strong imbalance), PRICE-GATED (suppress the SHORT when price is trending up = don't
fight a confirmed uptrend; short-side only, since bid-heavy+price-up fails both eras but ask-heavy+price-down works),
EWMA-smoothed to the 2d horizon. Backtest vs the OB-only base, net 10bps, both eras + effect on the short-into-rally
losses. Gate = 1 param (tapers the short to 0 as trailing-1d market return z rises 0->1 sigma). Honest: single series.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret, daily, stats
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def bt(pos, mkt, cost=0.0010):
    strat = (pos * mkt - cost * pos.diff().abs().fillna(0)).dropna()
    sd = daily(strat)
    return sd

def line(tag, sd):
    o, r = stats(sd[sd.index < CUT]), stats(sd[sd.index >= CUT])
    print(f"  {tag:22s} | OOS Sharpe {o['sharpe']:+.2f} maxDD {o['maxdd']:+.0f}% | RECENT Sharpe {r['sharpe']:+.2f} maxDD {r['maxdd']:+.0f}%")

def main():
    D = agg_ob().join(market_ret().rename("mkt"), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    D["z"] = zc(D["agg_imb"])
    lr = np.log1p(D["mkt"]); D["ztrail"] = zc(lr.rolling(6, min_periods=6).sum().shift(1))
    D = D.dropna(subset=["z"])
    raw = (-D["z"]).clip(-1.5, 1.5)
    # price gate: shorts (raw<0) tapered as trailing-1d market return rises; longs untouched
    g = np.where(raw < 0, np.clip(1 - np.clip(D["ztrail"].values, 0, None), 0, 1), 1.0)
    posA = raw.ewm(halflife=12).mean()                    # base: OB + strength
    posB = (raw * g).ewm(halflife=12).mean()              # + price gate
    print("=== SYNTHESIZED SIGNAL vs OB-only base (net 10bps) ===")
    line("A) OB + strength", bt(posA, D["mkt"]))
    line("B) + price gate (short)", bt(posB, D["mkt"]))

    # effect on short-into-rally days: days where market rose but base was short
    day = pd.DataFrame({"mkt": daily(D["mkt"]), "posA": posA.groupby(posA.index.floor("1D")).mean(),
                        "sA": daily(posA * D["mkt"] - 0.001 * posA.diff().abs().fillna(0)),
                        "sB": daily((raw * g).ewm(halflife=12).mean() * D["mkt"] - 0.001 * (raw * g).ewm(halflife=12).mean().diff().abs().fillna(0))}).dropna()
    rally_short = day[(day["mkt"] > 0.03) & (day["posA"] < -0.1)]   # market up >3% while base was short
    print(f"\n  short-into-rally days (mkt>+3% while base SHORT, n={len(rally_short)}):")
    print(f"    base strat avg {rally_short['sA'].mean()*100:+.2f}%/day  ->  gated strat avg {rally_short['sB'].mean()*100:+.2f}%/day  (gate should lift this)")
    # worst market days: does the gate hurt the crash catches?
    worst = day.sort_values("mkt").head(8)
    print(f"  worst 8 market days: base avg {worst['sA'].mean()*100:+.2f}% | gated avg {worst['sB'].mean()*100:+.2f}%  (should be ~unchanged — gate only touches shorts-into-UP)")
    print("\nread: gate EARNS its place only if it lifts Sharpe (esp OOS) or cuts short-into-rally losses WITHOUT")
    print("hurting the crash catches. Else the base OB+strength signal is already the whole thing. COMBINEDONE")

if __name__ == "__main__":
    main()
