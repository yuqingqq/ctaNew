"""Direction question: the signal is symmetric (short bid-crowding / long ask-crowding), but the LONG side is the
weaker, noisier half (every caught-long falling-knife loss is long-side). Test SYMMETRIC vs SHORT-ONLY (short or flat,
never long = pure defensive hedge) vs LONG-ONLY, net 10bps, both eras. Does the long side EARN its place, or is a
short-only hedge cleaner (and safer — never caught long into a crash)?
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret, daily, stats
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def bt(pos, mkt):
    return daily((pos * mkt - 0.0010 * pos.diff().abs().fillna(0)).dropna())

def main():
    D = agg_ob().join(market_ret().rename("mkt"), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    raw = (-zc(D["agg_imb"])).clip(-1.5, 1.5)
    variants = {"SYMMETRIC (short+long)": raw.ewm(halflife=42).mean(),
                "SHORT-ONLY (short/flat)": raw.clip(upper=0).ewm(halflife=42).mean(),
                "LONG-ONLY (long/flat)": raw.clip(lower=0).ewm(halflife=42).mean()}
    print("=== SYMMETRIC vs SHORT-ONLY vs LONG-ONLY (net 10bps) ===")
    md = daily(D["mkt"])
    for name, pos in variants.items():
        sd = bt(pos, D["mkt"]); o, r = stats(sd[sd.index < CUT]), stats(sd[sd.index >= CUT])
        exp = pos.abs().mean()
        print(f"  {name:24s} | OOS Sh {o['sharpe']:+.2f} DD {o['maxdd']:+.0f}% | REC Sh {r['sharpe']:+.2f} DD {r['maxdd']:+.0f}% | avg|pos| {exp:.2f}")

    print("\n=== defensive check: return on UP vs DOWN market days (both eras pooled) ===")
    day = pd.DataFrame({"mkt": md})
    for name, pos in variants.items():
        day[name] = bt(pos, D["mkt"])
    up, dn = day["mkt"] > 0, day["mkt"] <= 0
    for name in variants:
        s = day[name].dropna()
        print(f"  {name:24s} | UP-day {s[up.reindex(s.index)].mean()*1e4:+6.1f}bps | DOWN-day {s[dn.reindex(s.index)].mean()*1e4:+6.1f}bps")

    print("\n=== the caught-long days: does SHORT-ONLY avoid them? (long-side falling knives) ===")
    for d in ["2025-04-06", "2024-06-24", "2025-02-24"]:
        row = {n: day[n].get(pd.Timestamp(d, tz="UTC"), np.nan) for n in variants}
        m = md.get(pd.Timestamp(d, tz="UTC"), np.nan)
        print(f"  {d}: mkt {m*100:+.1f}% | sym {row['SYMMETRIC (short+long)']*100:+.1f}% | short-only {row['SHORT-ONLY (short/flat)']*100:+.1f}%")
    print("\nread: if SHORT-ONLY ~= or > symmetric on Sharpe AND cleaner on down-days/caught-long -> drop the long side. SIDESDONE")

if __name__ == "__main__":
    main()
