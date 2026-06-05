"""iter5 — POOLED cross-sectional model vs production PER-SYMBOL Ridge. v1 fits a separate RidgeCV per symbol;
this fits ONE pooled RidgeCV across all stacked symbols (pooled standardization) for both legs. Same walk-forward,
same features (base V0 short ranker, V0+resid_rev long ranker), same recency weight. Outputs:
  hl_pooled/v0full_hl60.parquet           (pooled base  -> short)
  hl_pooled_residrev/v0full_hl60.parquet  (pooled V0+RR -> long)
Also dumps per-cycle IC + per-symbol n for diagnosis -> live/state/opt_loop/pooled_diag.csv
"""
import sys, shutil; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
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

def gen(feats, outsub):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()].dropna(subset=feats)
        te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)].dropna(subset=feats)
        if len(tr)<2000 or not len(te): continue
        t_end=tr["open_time"].max()
        w=np.exp(-((t_end-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        s,h=x6.fit_preproc(tr,feats); X=x6.apply_preproc(tr,feats,s,h)   # POOLED standardization + one fit
        m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,tr["xs_z"].to_numpy(),sample_weight=w)
        pred=m.predict(x6.apply_preproc(te,feats,s,h))
        rec.append(pd.DataFrame({"symbol":te["symbol"].values,"open_time":te["open_time"].values,
            "alpha_A":te["alpha_vs_btc_realized"].values,"return_pct":te["return_pct"].values,
            "exit_time":te["exit_time"].values,"pred":pred,"fold":i}))
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    od=REPO/"live/state/convexity"/outsub; od.mkdir(parents=True,exist_ok=True)
    out.to_parquet(od/"v0full_hl60.parquet"); shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet",od/"fullflow_hl60.parquet")
    return out

b=gen(V0,"hl_pooled"); l=gen(V0+RR,"hl_pooled_residrev")
# diagnosis: pooled long-ranker per-cycle IC vs production per-symbol long-ranker
prod=pd.read_parquet(REPO/"live/state/convexity/hl_residrev/v0full_hl60.parquet")
prod["open_time"]=pd.to_datetime(prod["open_time"],utc=True)
def cyc_ic(d):
    ics=d.groupby("open_time").apply(lambda x: spearmanr(x["pred"], (x["return_pct"])).correlation if len(x)>=5 else np.nan).dropna()
    return ics.mean()
print(f"POOLED long-ranker mean per-cycle IC(pred,ret) = {cyc_ic(l):+.4f}")
print(f"PROD   long-ranker mean per-cycle IC(pred,ret) = {cyc_ic(prod):+.4f}")
print(f"DONE pooled base rows {len(b)} long rows {len(l)}")
