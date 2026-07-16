"""User's refinement: EVENT-DRIVEN (informative) imbalances have EXTREME relative strength -> they CONTINUE, while
moderate (crowding) imbalances REVERT. So the linear contrarian rule makes its biggest fades exactly where it's most
likely wrong (the caught-long losses are at extreme |z|). Test: (A) shape of fwd market return across SIGNED z septiles
(pure contrarian = monotonic; if extremes flatten/reverse, strength separates informative from crowding); (B) the FADE
edge per |z| quintile (fade = -sign(z)*fwd; if it's +ve at moderate |z| but 0/-ve at extreme |z|, the user is right);
(C) backtest a STRENGTH-FILTERED strategy (stand aside when |z|>THR) vs the original, net 10bps, both eras.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret, daily, stats
from live.bookdepth_crowding_horizon import market_series
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def main():
    A = agg_ob()
    z = (A["agg_imb"] - A["agg_imb"].rolling(90, min_periods=45).mean()) / A["agg_imb"].rolling(90, min_periods=45).std()
    D = pd.DataFrame({"z": z, "az": z.abs()}).join(market_series()[["mkt_1d", "mkt_2d"]]).join(market_ret().rename("mbar")).dropna(subset=["z", "mkt_2d", "mbar"])
    eras = {"OOS": D[D.index < CUT], "RECENT": D[D.index >= CUT]}

    print("=== (A) fwd 2d market return by SIGNED z septile (contrarian=monotonic DOWN; extremes flatten/reverse?) ===")
    print(f"{'z septile':22s} | {'OOS mean fwd2d':14s} | RECENT mean fwd2d")
    for e in ["OOS", "RECENT"]:
        eras[e]["zb"] = pd.qcut(eras[e]["z"], 7, labels=False, duplicates="drop")
    for b in range(7):
        o = eras["OOS"].loc[eras["OOS"]["zb"] == b]; r = eras["RECENT"].loc[eras["RECENT"]["zb"] == b]
        lab = {0: "0 extreme-ASK(z<<0)", 3: "3 middle(z~0)", 6: "6 extreme-BID(z>>0)"}.get(b, f"{b}")
        print(f"{lab:22s} | {o['mkt_2d'].mean()*100:+13.2f}% | {r['mkt_2d'].mean()*100:+.2f}%  (z~{o['z'].mean():+.1f})")

    print("\n=== (B) FADE edge by |z| quintile  (fade = -sign(z)*fwd2d; +ve=fade works, <=0=continuation) ===")
    print(f"{'|z| quintile':16s} | {'OOS fade':10s} | {'RECENT fade':11s} | median|z|")
    for e in ["OOS", "RECENT"]:
        eras[e]["azb"] = pd.qcut(eras[e]["az"], 5, labels=False, duplicates="drop")
        eras[e]["fade"] = -np.sign(eras[e]["z"]) * eras[e]["mkt_2d"]
    for b in range(5):
        o = eras["OOS"].loc[eras["OOS"]["azb"] == b]; r = eras["RECENT"].loc[eras["RECENT"]["azb"] == b]
        tag = {0: "Q1 (weakest)", 4: "Q5 (strongest)"}.get(b, f"Q{b+1}")
        print(f"{tag:16s} | {o['fade'].mean()*100:+9.2f}% | {r['fade'].mean()*100:+10.2f}% | {o['az'].median():.2f}")

    print("\n=== (C) BACKTEST: STRENGTH-FILTERED (stand aside when |z|>THR) vs original, net 10bps ===")
    print(f"{'variant':18s} | {'OOS Sharpe':10s} {'maxDD':7s} | {'RECENT Sharpe':13s} {'maxDD':7s} | avg|pos| worstday")
    for THR in [99, 3.0, 2.5, 2.0, 1.5]:
        raw = (-D["z"]).where(D["az"] < THR, 0.0).clip(-1.5, 1.5)
        pos = raw.ewm(halflife=12).mean()
        strat = (pos * D["mbar"] - 0.0010 * pos.diff().abs().fillna(0)).dropna()
        sd = daily(strat); day_pos = pos.groupby(pos.index.floor("1D")).mean()
        so = stats(sd[sd.index < CUT]); sr = stats(sd[sd.index >= CUT])
        worst = daily(D["mbar"]).min()  # worst market day (context)
        wd = sd.loc[daily(D["mbar"]).idxmin()] if daily(D["mbar"]).idxmin() in sd.index else np.nan
        lab = "original (no filter)" if THR == 99 else f"stand-aside |z|>{THR}"
        print(f"{lab:18s} | {so['sharpe']:+9.2f} {so['maxdd']:+6.0f}% | {sr['sharpe']:+12.2f} {sr['maxdd']:+6.0f}% | {pos.abs().mean():.2f}   {wd*100 if not np.isnan(wd) else np.nan:+.1f}%")
    print("\nread: if fade edge (B) dies at Q5 AND standing-aside on extremes lifts Sharpe/cuts DD -> strength filter WORKS. STRDONE")

if __name__ == "__main__":
    main()
