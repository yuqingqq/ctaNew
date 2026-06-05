"""iter7 — PARTIAL per-symbol coefficient shrinkage (denoise noisy per-symbol fits toward the common vector, while
KEEPING per-symbol heterogeneity). All on ONE pooled standardization (std is irrelevant per iter6b). For each fold:
  beta_common = RidgeCV on pooled stacked data ; beta_sym = RidgeCV per symbol
  beta = (1-a)*beta_sym + a*beta_common      (a=0 => per-sym = +3.455 ; a=1 => common = -0.71)
Env COEF_ALPHA. Outputs hl_cs<tag>/ + hl_cs<tag>_residrev/.
"""
import os, sys, shutil; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
A=float(os.environ.get("COEF_ALPHA","0.2")); TAG=os.environ.get("COEF_TAG",str(A).replace(".","p"))
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
        s,h=x6.fit_preproc(tr,feats)                      # one pooled standardization
        Xall=x6.apply_preproc(tr,feats,s,h); wall=np.exp(-((t_end-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        mc=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xall,tr["xs_z"].to_numpy(),sample_weight=wall)  # common
        bc=mc.coef_; ic=mc.intercept_
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            gte=te[te.symbol==sym]
            if not len(gte): continue
            X=x6.apply_preproc(gg,feats,s,h); w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            try:
                ms=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                beta=(1-A)*ms.coef_+A*bc; inter=(1-A)*ms.intercept_+A*ic
                Xte=x6.apply_preproc(gte,feats,s,h); pred=Xte@beta+inter
                rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":pred,"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    od=REPO/"live/state/convexity"/outsub; od.mkdir(parents=True,exist_ok=True)
    out.to_parquet(od/"v0full_hl60.parquet"); shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet",od/"fullflow_hl60.parquet")
    return len(out)
print(f"alpha={A}: base",gen(V0,f"hl_cs{TAG}"),"long",gen(V0+RR,f"hl_cs{TAG}_residrev"),"DONE")
