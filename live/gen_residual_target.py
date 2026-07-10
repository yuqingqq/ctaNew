"""Retrain the two books on RESIDUAL target vs RETURN target (row-matched, identical machinery) to test whether
training on the residual (what we farm) beats training on raw return. Only the training LABEL differs.
  return target   : xs_z(return_pct)
  residual target : xs_z(alpha_vs_btc_realized)
Emits base+long for each -> hl_tgt_{ret,res}_{base,long}/v0full_hl60.parquet
"""
import sys, os; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
SUF=os.environ.get("V4_BOOK_SUFFIX","")   # audit remediation: clean-panel recent books
x6=tt.x6; V0_LEAN=list(tt.V0_LEAN); EMB=pd.Timedelta(days=1); HL=60.0; RR=["resid_rev_2","resid_rev_3"]
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0_LEAN)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g=PAN.groupby("open_time")
def xsz(col): sd=g[col].transform("std").replace(0,np.nan); return ((PAN[col]-g[col].transform("mean"))/sd).clip(-10,10)
PAN["z_ret"]=xsz("return_pct"); PAN["z_res"]=xsz("alpha_vs_btc_realized")
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"rows={len(PAN)} syms={PAN.symbol.nunique()}",flush=True)
def gen(feats,tgt,outpath):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN[tgt].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg[tgt].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h))}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    Path(outpath).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(outpath); return out["symbol"].nunique(),len(out)
D=REPO/"live/state/convexity"
if not SUF:  # v3 return-target reference books only needed for the leaked/original build
    print("ret base",gen(V0_LEAN,"z_ret",D/"hl_tgt_ret_base/v0full_hl60.parquet"),flush=True)
    print("ret long",gen(V0_LEAN+RR,"z_ret",D/"hl_tgt_ret_long/v0full_hl60.parquet"),flush=True)
print("res base",gen(V0_LEAN,"z_res",D/f"hl_tgt_res_base{SUF}/v0full_hl60.parquet"),flush=True)
print("res long",gen(V0_LEAN+RR,"z_res",D/f"hl_tgt_res_long{SUF}/v0full_hl60.parquet"),flush=True)
print("GENDONE")
