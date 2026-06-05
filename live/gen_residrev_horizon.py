"""iter4 — resid_rev HORIZON sweep (the long-ranker's only untested hyperparameter). Production sums past 8h+12h
residual alpha (resid_rev_2,3). Test other window sets. Generates long-ranker preds (V0 + chosen resid_rev_k) ->
hl_rrh_<tag>/v0full_hl60.parquet ; short ranker stays the production base (hl). Env: RR_KS="2,3" (comma list of k bars).
"""
import os, sys, shutil; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); EMB=pd.Timedelta(days=1); HL=60.0
KS=[int(x) for x in os.environ.get("RR_KS","2,3").split(",")]
TAG=os.environ.get("RR_TAG","_".join(map(str,KS)))
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
      "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
CUTS=CUTS+[PAN["open_time"].max().normalize()+pd.Timedelta(days=1)]
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
RR=[]
for k in KS:
    col=f"resid_rev_{k}"; PAN[col]=-a.transform(lambda s,k=k: s.shift(1).rolling(k).sum()).fillna(0.0); RR.append(col)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
feats=V0+RR
rec=[]
for i in range(len(CUTS)-1):
    c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
    tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
    t_end=tr["open_time"].max()
    for sym,gg in tr.groupby("symbol"):
        if len(gg)<300: continue
        gte=te[te.symbol==sym]
        if not len(gte): continue
        try:
            s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
            w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
            rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
        except Exception: pass
out=pd.concat(rec,ignore_index=True)
for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
od=REPO/f"live/state/convexity/hl_rrh_{TAG}"; od.mkdir(parents=True,exist_ok=True)
out.to_parquet(od/"v0full_hl60.parquet"); shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet",od/"fullflow_hl60.parquet")
print(f"DONE RR={RR} -> {od} rows {len(out)}")
