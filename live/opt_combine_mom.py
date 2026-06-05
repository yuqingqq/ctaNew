"""Iteration 3: does a TREND book diversify the mean-rev production? Combine momentum single-book
PnL with the production two-book (ab_wfund175) over a momentum-weight grid. Report standalone Sharpes,
book correlation (the whole point), and combined Sharpe/maxDD per weight. Low/neg corr + combined
Sharpe > +2.13 = the user's trend-book hypothesis CONFIRMED.
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); ANN = np.sqrt(6*365)

def sharpe(x): return x.mean()/x.std()*ANN if x.std()>0 else 0.0
def maxdd(x): eq=x.cumsum(); return float((eq-eq.cummax()).min())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mom"); ap.add_argument("--label", default="trend_book")
    a = ap.parse_args()
    print(f"\n################  {a.label}  ################")
    mom = pd.read_csv(Path(a.mom)/"cycles.csv")[["open_time","pnl_bps"]].rename(columns={"pnl_bps":"mom"})
    mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    for pol in ("monthly","daily"):
        mr = pd.read_csv(REPO/f"live/state/ab_wfund175/{pol}/combine/twobook_equity.csv")
        mr = mr[["open_time","pnl_bps_combined_fill0"]].rename(columns={"pnl_bps_combined_fill0":"mr"})
        mr["open_time"] = pd.to_datetime(mr["open_time"], utc=True)
        m = mr.merge(mom, on="open_time", how="inner").fillna(0)
        corr = m["mr"].corr(m["mom"])
        s_mr, s_mom = sharpe(m["mr"]), sharpe(m["mom"])
        print(f"[{pol:7s}] mean-rev Sharpe {s_mr:+.3f} | momentum Sharpe {s_mom:+.3f} | BOOK CORR {corr:+.3f}  (n={len(m)})")
        best=None
        for w in (0.0,0.15,0.25,0.35,0.50):
            c = (1-w)*m["mr"] + w*m["mom"]
            sh, dd = sharpe(c), maxdd(c)
            tag = "  <-- baseline" if w==0 else ""
            print(f"    w_mom={w:.2f}: combined Sharpe {sh:+.3f}  maxDD {dd:+.0f}{tag}")
            if w>0 and (best is None or sh>best[1]): best=(w,sh)
        if pol=="monthly" and best:
            print(f"  >>> best momentum weight {best[0]:.2f} -> Sharpe {best[1]:+.3f} vs baseline {sharpe(m['mr']):+.3f} "
                  f"(Δ{best[1]-sharpe(m['mr']):+.3f}) | {'CONFIRMED — validate nested' if best[1]-sharpe(m['mr'])>0.15 else 'no diversification edge'}")

if __name__ == "__main__": main()
