import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0_LEAN); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
PAN["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
g=PAN.groupby("open_time"); sd=g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
PAN["z_res"]=((PAN["alpha_vs_btc_realized"]-g["alpha_vs_btc_realized"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
FEATS=[c for c in V0+RR if c not in {"ret_3d","autocorr_pctile_7d"}]
print(f"feats({len(FEATS)}): {FEATS}",flush=True)
rec=[]
for i in range(len(CUTS)-1):
    c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
    tr=PAN[(PAN.exit_time<fc)&PAN["z_res"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
    t_end=tr["open_time"].max()
    for sym,gg in tr.groupby("symbol"):
        if len(gg)<300: continue
        try:
            s,h=x6.fit_preproc(gg,FEATS); X=x6.apply_preproc(gg,FEATS,s,h)
            w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["z_res"].to_numpy(),sample_weight=w)
            gte=te[te.symbol==sym]
            if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,FEATS,s,h)),"fold":i}))
        except Exception: pass
o=pd.concat(rec,ignore_index=True)
for c in ("open_time","exit_time"): o[c]=pd.to_datetime(o[c],utc=True)
p=REPO/"live/state/convexity/hl_pv_final/v0full_hl60.parquet"; p.parent.mkdir(parents=True,exist_ok=True); o.to_parquet(p)
print("PFDONE")
