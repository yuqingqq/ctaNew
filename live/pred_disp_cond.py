"""NEW IDEA: is the edge ERA-CONSISTENT under a PIT conviction state (pred_disp)?

Not "does sizing by pred_disp improve Sharpe" (tested, failed) — the sharper question: is there a
pred_disp REGIME where the 1L/2S edge is positive in BOTH eras? If high-conviction (high cross-sectional
pred dispersion) cycles are era-stable, trading only them = a conditional era-robustness lever. pred_disp
is PIT (computed from the cycle's own preds → deployable). Book-level residual net, both eras, terciled.
Discipline: if a tercile looks era-stable, the paired-CI screen must confirm (short-only lesson).
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0,"live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
WL=WS=0.5

def build(base, long, reg):
    lg=long.groupby("open_time"); rows=[]; prevL=None; prevS=set()
    for t,g in base.groupby("open_time"):
        if len(g)<5 or reg.get(t) in (None,"deepbull"): continue
        try: gl=lg.get_group(t)
        except KeyError: continue
        Lp=gl.nlargest(1,"pred"); S=g.nsmallest(2,"pred")
        if len(Lp)<1 or len(S)<2: continue
        la=Lp.iloc[0]["alpha_A"]*1e4; sa=S["alpha_A"].mean()*1e4
        pd_disp=float(g["pred"].std())        # PIT cross-sectional conviction
        Ln,Ss=Lp.iloc[0]["symbol"],set(S["symbol"])
        lt=1.0 if (prevL is None or Ln!=prevL) else 0.0; st=(len(Ss-prevS)/2.0) if prevS else 1.0
        net=WL*la - WS*sa - lt*0.5*WL*COST/0.5 - st*0.5*WS*COST/0.5
        rows.append((t, net, pd_disp)); prevL,prevS=Ln,Ss
    return pd.DataFrame(rows,columns=["t","net","disp"])

def sh(x,c):
    d=pd.to_datetime(x["t"]).dt.date; dr=x[[c]].groupby(d).sum()[c]
    return dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan

def main():
    reg=btc_reg(); D={}
    for era,bp,lp in (("RECENT","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix"),
                     ("OOS","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix")):
        base,long=load(bp,lp); D[era]=build(base,long,reg)
    # use RECENT terciles as fixed PIT-ish thresholds applied to both eras (era-stable thresholds)
    q1,q2=D["RECENT"]["disp"].quantile([1/3,2/3])
    print(f"pred_disp tercile thresholds (from recent): low<{q1:.3f} mid<{q2:.3f} high>=")
    for era in ("RECENT","OOS"):
        d=D[era].copy(); d["tier"]=np.where(d.disp>=q2,"HIGH",np.where(d.disp>=q1,"mid","low"))
        print(f"\n===== {era}: 1L/2S book Sharpe by pred_disp tercile =====")
        for t in ["low","mid","HIGH"]:
            g=d[d.tier==t]
            if len(g)<20: continue
            print(f"  {t:>4} conviction (n={len(g):4d}): Sharpe {sh(g,'net'):+.2f}  mean {g.net.mean():+.1f}  net-sum {g.net.sum():+.0f}")
        print(f"  ALL   (n={len(d):4d}): Sharpe {sh(d,'net'):+.2f}")
    print("\n  >> is any tercile ERA-STABLE (Sharpe>0 in BOTH eras)? if HIGH -> conditional lever (then CI-screen)")
    print("PREDDISPDONE")

if __name__=="__main__":
    main()
