"""Event study: does the agg_imb defensive market-timing strategy actually WORK in the extreme cases — the worst
market crashes — or just on average? For the worst market DAYS and MONTHS, report the strategy's return and its
POSITION (was it short/defensive GOING IN, or caught long?). NOTE: this is a MARKET-AGGREGATE strategy — it captures
BROAD froth->crash episodes, NOT single-name pump-dumps (those wash out in the 175-name average). Same signal as the
backtest (W=90 z, cap 1.5, EWMA-smooth 2d, net 10bps).
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret, position

def main():
    D = agg_ob().join(market_ret(), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    D["pos"] = position(D["agg_imb"], 90, 1.5, 12)
    D["strat"] = D["pos"] * D["mkt"] - 0.0010 * D["pos"].diff().abs().fillna(0)
    D = D.dropna(subset=["pos", "strat"])
    g = D.groupby(D.index.floor("1D"))
    day = pd.DataFrame({"mkt": g["mkt"].apply(lambda x: (1 + x).prod() - 1),
                        "strat": g["strat"].apply(lambda x: (1 + x).prod() - 1),
                        "pos": g["pos"].mean(), "pos_in": g["pos"].first()})   # pos_in = position at start of day (going in)
    mg = day.groupby(pd.Grouper(freq="MS"))
    mon = pd.DataFrame({"mkt": mg["mkt"].apply(lambda x: (1 + x).prod() - 1),
                        "strat": mg["strat"].apply(lambda x: (1 + x).prod() - 1), "pos": mg["pos"].mean()})

    print("=== WORST MARKET MONTHS — did the strategy protect? (pos<0 = short/defensive) ===")
    print(f"{'month':8s} | {'market':8s} | {'strategy':8s} | avg pos")
    for d, r in mon.sort_values("mkt").head(8).iterrows():
        print(f"{str(d.date())[:7]:8s} | {r.mkt*100:+7.1f}% | {r.strat*100:+7.1f}% | {r.pos:+.2f}")

    print("\n=== WORST MARKET DAYS — was it positioned short GOING IN? (pos_in<0 = yes) ===")
    print(f"{'day':10s} | {'market':8s} | {'strategy':8s} | pos going-in")
    for d, r in day.sort_values("mkt").head(12).iterrows():
        print(f"{str(d.date()):10s} | {r.mkt*100:+7.1f}% | {r.strat*100:+7.1f}% | {r.pos_in:+.2f}")

    thr = day["mkt"].quantile(0.10)
    ext = day[day["mkt"] <= thr]
    print(f"\n=== TAIL SUMMARY: worst-decile market days (mkt <= {thr*100:.1f}%, n={len(ext)}) ===")
    print(f"  strategy avg {ext['strat'].mean()*100:+.2f}%/day  (market avg {ext['mkt'].mean()*100:+.2f}%) | strat POSITIVE on {(ext['strat']>0).mean()*100:.0f}% of them | avg pos-going-in {ext['pos_in'].mean():+.2f}")
    print(f"  -> defensive in the tail if strat avg > 0 AND positioned short (pos_in<0) going into the worst days. EVENTDONE")

if __name__ == "__main__":
    main()
