"""WHY pooled (IC +0.047) loses to per-symbol (IC +0.029) at the portfolio. Strategy trades K=3 extremes.
Diagnose: (1) decile response of pred vs realized xs_z, (2) the actual top-3/bottom-3 leg returns, (3) concentration
(unique symbols traded), (4) is the pooled pred just ranking by a persistent per-symbol characteristic?
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
CONV=REPO/"live/state/convexity"

def load(sub):
    d=pd.read_parquet(CONV/sub/"v0full_hl60.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    g=d.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
    d["xs_z"]=((d["return_pct"]-g["return_pct"].transform("mean"))/sd)   # realized XS z (the target space)
    return d

for name, sub in [("PROD per-symbol","hl_residrev"), ("POOLED","hl_pooled_residrev")]:
    d=load(sub)
    # (1) decile response: per cycle, bucket pred into deciles, mean realized xs_z per decile (avg over cycles)
    d["dec"]=d.groupby("open_time")["pred"].transform(lambda x: pd.qcut(x.rank(method="first"),10,labels=False) if x.notna().sum()>=10 else np.nan)
    dec=d.groupby("dec")["xs_z"].mean()
    # (2) traded legs: top-3 long, bottom-3 short per cycle
    def legs(x):
        x=x.dropna(subset=["pred"]);
        if len(x)<6: return pd.Series({"long3":np.nan,"short3":np.nan})
        L=x.nlargest(3,"pred")["xs_z"].mean(); S=x.nsmallest(3,"pred")["xs_z"].mean()
        return pd.Series({"long3":L,"short3":S})
    lg=d.groupby("open_time").apply(legs)
    ls_spread=lg["long3"].mean()-lg["short3"].mean()   # long should be >0, short <0 -> spread>0
    # (3) concentration: unique symbols ever in top-3 long / bottom-3 short
    topL=d.groupby("open_time").apply(lambda x: x.nlargest(3,"pred")["symbol"].tolist() if x["pred"].notna().sum()>=6 else [])
    botS=d.groupby("open_time").apply(lambda x: x.nsmallest(3,"pred")["symbol"].tolist() if x["pred"].notna().sum()>=6 else [])
    uL=len(set(sum(topL.tolist(),[]))); uS=len(set(sum(botS.tolist(),[])))
    # turnover: avg fraction of long basket changing cycle-to-cycle
    print(f"\n===== {name} ({sub}) =====")
    print("decile->mean realized xs_z (0=lowest pred ... 9=highest pred; strategy LONGS dec9, SHORTS dec0):")
    print("   "+"  ".join(f"d{i}:{dec.get(i,np.nan):+.3f}" for i in range(10)))
    print(f"traded legs: long3 realized xs_z {lg['long3'].mean():+.4f}  short3 {lg['short3'].mean():+.4f}  L-S spread {ls_spread:+.4f}")
    print(f"concentration: unique syms in top-3-long {uL}, bottom-3-short {uS} (more = less concentrated)")
