"""CI screen on the pred_disp finding (short-only lesson): is MID-conviction era-stable SIGNIFICANTLY,
and is HIGH-conviction era-fragility real? Block-bootstrap Sharpe CI per tercile per era + per-quarter.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0,"live")
from pred_disp_cond import build
from attribution_v4_regime import btc_reg, load
import warnings; warnings.filterwarnings("ignore")
rng=np.random.default_rng(48)

def daily(x):
    d=pd.to_datetime(x["t"]).dt.date; return x[["net"]].groupby(d).sum()["net"]
def boot_sharpe_ci(s,L=10,n=2000):
    v=s.values; m=len(v); nb=int(np.ceil(m/L)); out=[]
    for _ in range(n):
        st=rng.integers(0,max(1,m-L+1),nb); ii=np.concatenate([np.arange(x,x+L) for x in st])[:m]%m
        w=v[ii]
        if w.std()>0: out.append(w.mean()/w.std()*np.sqrt(365))
    return np.percentile(out,[2.5,50,97.5]) if out else (np.nan,)*3

def main():
    reg=btc_reg(); D={}
    for era,bp,lp in (("RECENT","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix"),
                     ("OOS","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix")):
        base,long=load(bp,lp); D[era]=build(base,long,reg)
    q1,q2=D["RECENT"]["disp"].quantile([1/3,2/3])
    for era in ("RECENT","OOS"):
        d=D[era].copy(); d["tier"]=np.where(d.disp>=q2,"HIGH",np.where(d.disp>=q1,"mid","low"))
        print(f"\n===== {era}: Sharpe CI by tercile =====")
        for t in ["mid","HIGH"]:
            g=d[d.tier==t]; s=daily(g); lo,md,hi=boot_sharpe_ci(s)
            sig="SIG>0" if lo>0 else ("SIG<0" if hi<0 else "NOT SIG (crosses 0)")
            print(f"  {t:>4}: Sharpe CI [{lo:+.2f}, {hi:+.2f}] {sig}")
            if t=="mid":
                g2=g.copy(); g2["q"]=pd.to_datetime(g2["t"]).dt.to_period("Q").astype(str)
                qs=[f"{q}:{daily(gg).sum():+.0f}" for q,gg in g2.groupby("q") if len(gg)>=15]
                print(f"        mid per-quarter net-sum: {qs}")
    print("PREDDISPCIDONE")

if __name__=="__main__":
    main()
