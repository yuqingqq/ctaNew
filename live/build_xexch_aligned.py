"""#185 FIX — boundary-aligned cross-exchange premium. ALL venues priced at the SAME instant (hh:00 open):
  Binance open@hh:00 (local 5m 'open'), OKX/Coinbase open@hh:00 (re-pulled). Fixes the prior bug where
  Binance used close@hh:05 but OKX/CB used close@(hh+4):00 (4h misalignment).
premium=log(venue_open/binance_open), XS-demean per bar, PIT shift(1). Saves features parquet + prints univariate IC.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.convexity_paper_bot as bot
XEO=REPO/"data/ml/cache/xexch_open"; OOS=pd.Timestamp("2025-10-04",tz="UTC")

def binance_open_4h(sym):
    sd=bot.KLINES/sym/"5m"
    if not sd.exists(): return None
    df=pd.concat([pd.read_parquet(f,columns=["open_time","open"]) for f in sorted(sd.glob("*.parquet"))],ignore_index=True)
    df=df.drop_duplicates("open_time").sort_values("open_time"); df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    s=df.set_index("open_time")["open"].astype(float)
    return s[(s.index.hour%4==0)&(s.index.minute==0)]

def venue_open(venue,sym):
    f=XEO/venue/f"{sym}.parquet"
    if not f.exists(): return None
    d=pd.read_parquet(f); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    return d.set_index("open_time")["open"].astype(float)

# target
pan=pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","return_pct"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
g=pan.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
pan["xs_z"]=((pan["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10); tgt=pan[["symbol","open_time","xs_z"]]

allfeat=[]; rows=[]
for venue in ("okx","coinbase"):
    prem={}
    for f in (XEO/venue).glob("*.parquet"):
        s=f.stem; vo=venue_open(venue,s); bo=binance_open_4h(s)
        if vo is None or bo is None or not len(bo): continue
        j=pd.concat([np.log(vo).rename("v"),np.log(bo).rename("b")],axis=1).dropna()
        if len(j)<100: continue
        prem[s]=(j["v"]-j["b"]).rename(s)
    if not prem: print(f"{venue}: none"); continue
    P=pd.concat(prem.values(),axis=1); Pxs=P.sub(P.median(axis=1),axis=0)
    feat=Pxs.shift(1).reset_index().melt(id_vars="open_time",var_name="symbol",value_name=f"{venue}_level").dropna()
    feat["open_time"]=pd.to_datetime(feat["open_time"],utc=True); allfeat.append(feat)
    m=feat.merge(tgt,on=["symbol","open_time"],how="inner").dropna()
    for lbl,sub in [("all",m),("oos",m[m.open_time>=OOS])]:
        ics=sub.groupby("open_time").apply(lambda d: spearmanr(d[f"{venue}_level"],d["xs_z"]).correlation if len(d)>=5 else np.nan).dropna()
        rows.append({"feature":f"{venue}_level","window":lbl,"IC":round(ics.mean(),4),
                     "t":round(ics.mean()/ics.std()*np.sqrt(len(ics)),1),"n_syms":sub.symbol.nunique()})
# merge all venue features -> one parquet for the gen
out=allfeat[0]
for f in allfeat[1:]: out=out.merge(f,on=["symbol","open_time"],how="outer")
out.to_parquet(XEO/"aligned_premium.parquet")
print("\n=== ALIGNED cross-exchange premium — univariate XS IC (vs misaligned: okx -0.040, cb -0.035) ===")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nsaved features -> {XEO/'aligned_premium.parquet'}")
