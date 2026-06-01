"""LOOP iter-15 — pooled NO-sym_id LGBM on xs_z target (non-linear, extensible), monthly-WF recency60.
Attacks the selection-skill frontier: LGBM captures cross-sectional interactions per-sym Ridge can't.
Generate system-compatible preds for replay at K=3 model-L/S vs per-sym Ridge +2.95.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location("x6",REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6=importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
BASELINE=REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT=REPO/"live/state/convexity/lgbm_xsz60_preds.parquet"
V0=x6.BASE+x6.COHORT_EXTRAS; EMB=pd.Timedelta(days=1); HL=60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

def main():
    t0=time.time(); print("=== LOOP iter-15: pooled no-sym_id LGBM on xs_z ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    gc=pan.groupby("open_time"); mu=gc["return_pct"].transform("mean"); sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
    p=dict(objective="regression",metric="rmse",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,
           feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,verbose=-1,num_threads=8)
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fit_cut=c0-EMB
        tr=pan[(pan.exit_time<fit_cut)&pan["xs_z"].notna()]; te=pan[(pan.open_time>=c0)&(pan.open_time<c1)]
        t_end=tr["open_time"].max(); w=np.exp(-((t_end-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        ds=lgb.Dataset(tr[V0].fillna(0).values,label=tr["xs_z"].values,weight=w)
        mdl=lgb.train(p,ds,num_boost_round=400,callbacks=[lgb.log_evaluation(0)])
        pr=mdl.predict(te[V0].fillna(0).values)
        rec.append(pd.DataFrame({"symbol":te["symbol"].values,"open_time":te["open_time"].values,"pred_l":pr}))
        print(f"  {c0.date()}→{c1.date()}: train {len(tr):,} [{time.time()-t0:.0f}s]",flush=True)
    rec=pd.concat(rec,ignore_index=True); rec["open_time"]=pd.to_datetime(rec["open_time"],utc=True)
    bl=pd.read_parquet(BASELINE); bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True); bl["exit_time"]=pd.to_datetime(bl["exit_time"],utc=True)
    oos=bl[bl.open_time>=CUTS[0]].copy().merge(rec,on=["symbol","open_time"],how="left")
    n=oos["pred_l"].notna().sum(); oos["pred"]=oos["pred_l"].where(oos["pred_l"].notna(),oos["pred"]); oos.drop(columns=["pred_l"]).to_parquet(OUT)
    print(f"\n  wrote {OUT.name}: swapped {n:,} ({100*n/len(oos):.0f}%)\n  DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
