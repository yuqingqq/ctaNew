"""Long vs short leg RETURN DISTRIBUTION (vanilla book, clean) — does long farm the tail, short grind?

Tests the DDI-2-style pattern (from vBTC): "long farms the long tail (few jackpots, right-skew), short
grinds (steady, many small wins)." Measures per-cycle leg PnL distribution on v4 clean books: mean,
median, skew, top-decile (jackpot) share of gross-positive, worst-decile CVaR (tail loss). Long =
top-1 long-pred alpha (+); short = -bottom-2 base-pred alpha (short PnL). Residual, bps, GROSS (pre-cost
so the distribution shape is clean).
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load
import warnings; warnings.filterwarnings("ignore")

def skew(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return float(((x-x.mean())**3).mean()/x.std()**3) if len(x)>2 and x.std()>0 else np.nan
def jackpot(x):  # top-decile share of total (how tail-driven is the mean)
    x=np.sort(np.asarray(x,float)); n=len(x); top=x[-max(1,n//10):].sum(); tot=x.sum()
    return top/tot*100 if tot!=0 else np.nan
def cvar(x,q=10):
    x=np.sort(np.asarray(x,float)); return float(x[:max(1,len(x)*q//100)].mean())

def legs(base, long, reg):
    lg=long.groupby("open_time"); rows=[]
    for t,g in base.groupby("open_time"):
        if len(g)<5 or t not in reg: continue
        try: gl=lg.get_group(t)
        except KeyError: continue
        Lp=gl.nlargest(1,"pred"); S=g.nsmallest(2,"pred")
        if len(Lp)<1 or len(S)<2: continue
        la=Lp.iloc[0]["alpha_A"]*1e4; sp=-S["alpha_A"].mean()*1e4
        if np.isfinite(la) and np.isfinite(sp): rows.append((la,sp))
    return pd.DataFrame(rows,columns=["L","S"])

def describe(x, name):
    x=x[np.isfinite(x)]
    print(f"  {name:<7} mean {x.mean():+6.1f}  median {np.median(x):+6.1f}  skew {skew(x):+5.2f}  "
          f"win% {(x>0).mean()*100:4.0f}  jackpot(top10%) {jackpot(x):4.0f}%  worst10%CVaR {cvar(x):+7.0f}")

def main():
    reg=btc_reg()
    for era,bp,lp in (("OOS 2023-25","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix"),
                     ("RECENT 2025-10+","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix")):
        base,long=load(bp,lp); d=legs(base,long,reg)
        print(f"\n===== {era}: leg PnL distribution (vanilla, residual bps, gross) n={len(d)} =====")
        describe(d["L"].values,"LONG")
        describe(d["S"].values,"SHORT")
    print("\n  jackpot = top-decile share of summed PnL (high = mean is tail-driven = 'farming the tail').")
    print("  A grinder has jackpot≈10-20% (even), win%>50, near-zero/positive skew; a lottery has jackpot>>,")
    print("  low win%, high +skew. Left-skew + bad worst-decile = squeeze tail.\nLSDISTDONE")

if __name__=="__main__":
    main()
