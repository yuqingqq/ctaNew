"""User's symmetric price x OB REACTION idea. Four cells -> two mirror pairs:
  AGREE    (book confirms price): price UP + bid-heavy  <-> price DOWN + ask-heavy   (momentum)
  DISAGREE (book fights price):   price UP + ask-heavy  <-> price DOWN + bid-heavy    (absorption/reversal)
Test: (A) forward 2d market return in the 2x2 (does the mirror symmetry hold, both eras?); (B) backtest a SYMMETRIC
reaction-modulated signal vs OB-only base: pos = -z(imb) * (1 - rho * sign(z)*sign(trail)) -> dampens the fade when
price AGREES with the book (momentum, don't fight), amplifies when they DISAGREE (crowding tested -> fade harder).
Symmetric on both sides (unlike the asymmetric price-gate, which was rejected).
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_timing_backtest import market_ret, daily, stats
from live.bookdepth_crowding_horizon import market_series
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def bt(pos, mkt):
    return daily((pos * mkt - 0.0010 * pos.diff().abs().fillna(0)).dropna())

def main():
    D = agg_ob().join(market_ret().rename("mkt"), how="inner").sort_index().dropna(subset=["agg_imb", "mkt"])
    D["z"] = zc(D["agg_imb"])
    lr = np.log1p(D["mkt"]); D["trail"] = lr.rolling(6, min_periods=6).sum().shift(1)   # trailing 1d price move
    D = D.join(market_series()[["mkt_2d"]]).dropna(subset=["z", "trail", "mkt_2d"])
    eras = {"OOS": D[D.index < CUT], "RECENT": D[D.index >= CUT]}

    print("=== (A) fwd 2d market return in the price x book 2x2 (does the mirror symmetry hold?) ===")
    for e, s in eras.items():
        bid = s["z"] > 0; up = s["trail"] > 0
        cell = lambda a, b: s[a & b]["mkt_2d"].mean() * 100
        print(f"  [{e}]  AGREE: (up,bid) {cell(up, bid):+.2f}%  ~ (down,ask) {cell(~up, ~bid):+.2f}%  |  "
              f"DISAGREE: (up,ask) {cell(up, ~bid):+.2f}%  ~ (down,bid) {cell(~up, bid):+.2f}%")

    print("\n=== (B) SYMMETRIC reaction-modulated vs OB-only base (net 10bps) ===")
    agree = np.sign(D["z"]) * np.sign(D["trail"])      # +1 book & price agree (momentum), -1 disagree
    base = (-D["z"]).clip(-1.5, 1.5).ewm(halflife=42).mean()
    for rho in [0.0, 0.4, 0.7, 1.0]:
        pos = ((-D["z"]) * (1 - rho * agree)).clip(-1.5, 1.5).ewm(halflife=42).mean()
        sd = bt(pos, D["mkt"]); o, r = stats(sd[sd.index < CUT]), stats(sd[sd.index >= CUT])
        tag = "OB-only (base)" if rho == 0 else f"reaction rho={rho}"
        print(f"  {tag:18s} | OOS Sh {o['sharpe']:+.2f} DD {o['maxdd']:+.0f}% | RECENT Sh {r['sharpe']:+.2f} DD {r['maxdd']:+.0f}%")
    print("\nread: (A) symmetric if the two AGREE cells match & the two DISAGREE cells match, BOTH eras. (B) reaction")
    print("EARNS place only if rho>0 lifts BOTH eras vs base. Else price-reaction doesn't robustly add. REACTIONDONE")

if __name__ == "__main__":
    main()
