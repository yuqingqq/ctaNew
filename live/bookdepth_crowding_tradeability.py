"""Two decisive diagnostics for the agg_imb contrarian market-timing lead (agg_imb -> market fwd, OOS 1-2d IC
-0.09/-0.12, survives overlap+trend control). Before it's worth anything it must be (1) BROAD not crash-driven +
temporally consistent, and (2) TRADEABLE net of cost with a REAL-TIME signal (no look-ahead).

(1) EPISODE-STABILITY: 6-month sub-period ICs (consistency); IC after EXCLUDING the worst 5%/10% down-move bars
    (if it collapses -> it only predicts crashes = fragile). OOS era (the powered one).
(2) TRADEABILITY: contrarian position = -clip(z(agg_imb), +/-1.5) using REAL-TIME normalization (expanding-z and
    trailing-90 z; NOT the full-sample level = no look-ahead), non-overlapping H-bar holds, net of flip cost (5bps
    x turnover). Net Sharpe vs gross vs BUY-HOLD, both eras. NOTE the tension: the LEVEL carries the IC but the
    DEVIATION was null, so a real-time (deviation-like) signal may NOT harvest it — that's the honest question.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_market_timing import agg_ob
from live.bookdepth_crowding_horizon import market_series
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def z_expanding(s):
    return (s - s.expanding(60).mean().shift(1)) / s.expanding(60).std().shift(1)

def z_trailing(s, W):
    return (s - s.rolling(W, min_periods=W // 2).mean().shift(1)) / s.rolling(W, min_periods=W // 2).std().shift(1)

def episode(sub, feat, tgt):
    print(f"  6-month sub-period IC(agg_imb -> {tgt}):")
    ics = []
    for name, g in sub.groupby(pd.Grouper(freq="2QS")):
        gg = g[[feat, tgt]].dropna()
        if len(gg) >= 60:
            ic = spearmanr(gg[feat], gg[tgt]).correlation; ics.append(ic)
            print(f"    {str(name.date())}: IC {ic:+.4f}  (n={len(gg)})")
    ics = np.array(ics)
    print(f"    -> {int((ics<0).sum())}/{len(ics)} sub-periods negative; mean {ics.mean():+.4f}, range [{ics.min():+.3f},{ics.max():+.3f}]")
    w = sub[[feat, tgt]].dropna()
    full = spearmanr(w[feat], w[tgt]).correlation
    for q in [0.05, 0.10]:
        ex = w[w[tgt] > w[tgt].quantile(q)]
        print(f"    IC excl worst {int(q*100)}% down bars: {spearmanr(ex[feat], ex[tgt]).correlation:+.4f}  (full {full:+.4f})")

def backtest(D, zcol, Hlab, H, cost=0.0005, cap=1.5):
    d = D[[zcol, f"mkt_{Hlab}"]].dropna().iloc[::H]           # non-overlapping H-bar holds
    if len(d) < 20: return None
    pos = (-d[zcol].clip(-cap, cap)).values; ret = d[f"mkt_{Hlab}"].values
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl = pos * ret - cost * turn; gross = pos * ret
    ppy = 2191.5 / H
    sh = lambda x: x.mean() / x.std() * np.sqrt(ppy) if x.std() > 0 else np.nan
    return dict(net=sh(pnl), grs=sh(gross), bh=sh(ret), meanbps=pnl.mean() * 1e4,
                pos=pos.mean(), turn=turn.mean(), corr=np.corrcoef(pos, ret)[0, 1], n=len(d))

def main():
    D = agg_ob().join(market_series(), how="inner").sort_index()
    D["z_exp"] = z_expanding(D["agg_imb"]); D["z_tr90"] = z_trailing(D["agg_imb"], 90)   # real-time, before slicing
    eras = {"OOS": D[D.index < CUT], "RECENT": D[D.index >= CUT]}
    print("=== (1) EPISODE-STABILITY (OOS, agg_imb -> market 2d fwd) ===")
    episode(eras["OOS"], "agg_imb", "mkt_2d")
    print("\n=== (2) TRADEABILITY: contrarian -z(agg_imb) market-timing, net 5bps/turnover, vs BUY-HOLD ===")
    print(f"{'era':7s} {'signal':7s} {'H':3s} | {'netSharpe':9s} {'gross':7s} {'buyhold':7s} | meanbps avgpos turn corr(pos,mkt) N")
    for era, sub in eras.items():
        for zcol in ["z_exp", "z_tr90"]:
            for Hlab, H in [("1d", 6), ("2d", 12)]:
                r = backtest(sub, zcol, Hlab, H)
                if r:
                    print(f"{era:7s} {zcol:7s} {Hlab:3s} | {r['net']:+9.2f} {r['grs']:+7.2f} {r['bh']:+7.2f} | {r['meanbps']:+7.1f} {r['pos']:+.2f} {r['turn']:.2f} {r['corr']:+.2f} {r['n']:4d}")
    print("\nread: TRADEABLE if net Sharpe > 0 AND > buy-hold in BOTH eras with sane turnover; broad if sub-periods")
    print("mostly-negative + IC survives excluding crash bars. Else = statistically real but fragile/unharvestable. TRADEDONE")

if __name__ == "__main__":
    main()
