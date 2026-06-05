import sys, shutil; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
import live.gen_shrinkage_bookB as g0   # reuse PAN + CUTS
PAN=g0.PAN; CUTS=g0.CUTS
def gen(feats,outsub):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if not len(te): continue
        t_end=tr["open_time"].max()
        s,h=x6.fit_preproc(tr,feats)   # POOLED standardization (one scaler on all)
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            gte=te[te.symbol==sym]
            if not len(gte): continue
            X=x6.apply_preproc(gg,feats,s,h)   # pooled scale, PER-SYMBOL fit
            w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            try:
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    od=REPO/"live/state/convexity"/outsub; od.mkdir(parents=True,exist_ok=True)
    out.to_parquet(od/"v0full_hl60.parquet"); shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet",od/"fullflow_hl60.parquet")
    return len(out)
print("base",gen(V0,"hl_psc"),"long",gen(V0+RR,"hl_psc_residrev"),"DONE")
