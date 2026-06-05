"""iter6 — isolate WHICH part of per-symbol fitting matters: standardization vs coefficients.
HYBRID: per-symbol STANDARDIZATION (x6.fit_preproc per symbol -> regime-stable normalization) + ONE COMMON
coefficient vector fit on the pooled per-symbol-standardized data (borrow coefficient strength). If this matches/beats
production, the per-symbol benefit is the STANDARDIZATION and we can share coefficients. If it fails, per-symbol
COEFFICIENTS matter too. Outputs hl_shrink/ + hl_shrink_residrev/.
"""
import sys, shutil; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
      "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
CUTS=CUTS+[PAN["open_time"].max().normalize()+pd.Timedelta(days=1)]
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()).fillna(0.0)
PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum()).fillna(0.0)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def gen(feats,outsub):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if not len(te): continue
        t_end=tr["open_time"].max()
        Xs,ys,ws,preproc=[],[],[],{}
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)   # PER-SYMBOL standardization
                preproc[sym]=(s,h); Xs.append(X); ys.append(gg["xs_z"].to_numpy())
                ws.append(np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL))
            except Exception: pass
        if not Xs: continue
        Xall=np.vstack(Xs); yall=np.concatenate(ys); wall=np.concatenate(ws)
        m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xall,yall,sample_weight=wall)   # ONE common coefficient vector
        for sym,gte in te.groupby("symbol"):
            if sym not in preproc: continue
            s,h=preproc[sym]
            rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    od=REPO/"live/state/convexity"/outsub; od.mkdir(parents=True,exist_ok=True)
    out.to_parquet(od/"v0full_hl60.parquet"); shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet",od/"fullflow_hl60.parquet")
    return len(out)
print("base",gen(V0,"hl_shrink"),"long",gen(V0+RR,"hl_shrink_residrev"),"DONE")
