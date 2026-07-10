"""Deep re-analysis: SHORT-ONLY (drop the dragging long; beta-neutralize with zero-residual BTC hedge)
vs the 1L/2S book. Distinct from LH1 (which added an alt-BASKET hedge whose OWN residual is era-dependent
-3.9 OOS = the real cause of LH1's OOS loss). In the RESIDUAL frame a BTC hedge adds ~0 (BTC is the
benchmark), so residual short-only = short-only + BTC-hedge. Non-deep-bull, both eras, net pinned cost.
Gate: short-only ≥ 1L/2S in BOTH eras (net Sharpe) + jackpot-preservation. Skeptical, book-level.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
WL = WS = 0.5

def build(base, long, reg):
    lg=long.groupby("open_time"); rows=[]; prevL=None; prevS=set()
    for t,g in base.groupby("open_time"):
        if len(g)<5 or reg.get(t) in (None,"deepbull"): continue
        try: gl=lg.get_group(t)
        except KeyError: continue
        Lp=gl.nlargest(1,"pred"); S=g.nsmallest(2,"pred")
        if len(Lp)<1 or len(S)<2: continue
        la=Lp.iloc[0]["alpha_A"]*1e4; sa=S["alpha_A"].mean()*1e4
        Ln,Ss=Lp.iloc[0]["symbol"],set(S["symbol"])
        lt=1.0 if (prevL is None or Ln!=prevL) else 0.0
        st=(len(Ss-prevS)/2.0) if prevS else 1.0
        lcost=lt*0.5*WL*COST/0.5; scost=st*0.5*WS*COST/0.5
        book_1l2s = WL*la - WS*sa - lcost - scost       # current 1L/2S
        short_only = -WS*sa - scost                     # drop the long (BTC-hedge adds 0 residual)
        rows.append((t, book_1l2s, short_only))
        prevL,prevS=Ln,Ss
    return pd.DataFrame(rows,columns=["t","b","s"])

def sh(x,c):
    d=pd.to_datetime(x["t"]).dt.date; dr=x[[c]].groupby(d).sum()[c]
    return dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan
def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())
def jkp(x): x=np.sort(np.asarray(x,float)); return x[-max(1,len(x)//10):].sum()/x.sum()*100 if x.sum()!=0 else np.nan

def main():
    reg=btc_reg(); res={}
    for era,bp,lp in (("RECENT","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix"),
                     ("OOS","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix")):
        base,long=load(bp,lp); d=build(base,long,reg)
        bS,sS=sh(d,"b"),sh(d,"s"); res[era]=(bS,sS)
        print(f"\n===== {era}: 1L/2S vs SHORT-ONLY (non-deep-bull, residual net, n={len(d)}) =====")
        print(f"  1L/2S (current)  : Sharpe {bS:+.2f}  mean {d.b.mean():+.1f}  maxDD {mdd(d.b.values):+.0f}  jackpot {jkp(d.b.values):.0f}%")
        print(f"  SHORT-ONLY+BTChdg: Sharpe {sS:+.2f}  mean {d.s.mean():+.1f}  maxDD {mdd(d.s.values):+.0f}  jackpot {jkp(d.s.values):.0f}%")
        print(f"  Δ Sharpe {sS-bS:+.2f}  Δ maxDD {mdd(d.s.values)-mdd(d.b.values):+.0f}")
    both=all(s>=b for b,s in res.values())
    print(f"\n  >> GATE (short-only ≥ 1L/2S BOTH eras): {'PASS -> candidate solution (source b: long era-conditionality)' if both else 'FAIL'}")
    print(f"     recent Δ {res['RECENT'][1]-res['RECENT'][0]:+.2f} | OOS Δ {res['OOS'][1]-res['OOS'][0]:+.2f}")
    print("SHORTONLYDONE")

if __name__=="__main__":
    main()
