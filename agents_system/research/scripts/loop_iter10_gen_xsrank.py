"""LOOP iter-10 — production xs_z + cross-sectional-rank features, monthly-WF recency60.
Generate system-compatible preds file for replay at K=3.
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location("x6",REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6=importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
BASELINE=REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT=REPO/"live/state/convexity/xsz_xsrank60_preds.parquet"
V0=x6.BASE+x6.COHORT_EXTRAS; EMB=pd.Timedelta(days=1); HL=60.0
XSR_SRC=["corr_to_btc_1d","rvol_7d","atr_pct","return_1d","ret_3d","idio_vol_to_btc_1d"]
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

def main():
    t0=time.time(); print("=== LOOP iter-10: gen xs_z+xsrank preds ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    gc=pan.groupby("open_time"); mu=gc["return_pct"].transform("mean"); sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
    xsr=[]
    for f in XSR_SRC:
        c="xsr_"+f; pan[c]=gc[f].rank(pct=True); xsr.append(c)
    FEAT=V0+xsr
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fit_cut=c0-EMB
        tr=pan[(pan.exit_time<fit_cut)&pan["xs_z"].notna()]; te=pan[(pan.open_time>=c0)&(pan.open_time<c1)]
        t_end=tr["open_time"].max(); models,ss,hh={},{},{}
        for sym,g in tr.groupby("symbol"):
            if len(g)<300: continue
            try:
                s,h=x6.fit_preproc(g,FEAT); X=x6.apply_preproc(g,FEAT,s,h)
                w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                models[sym]=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,g["xs_z"].to_numpy(),sample_weight=w); ss[sym]=s; hh[sym]=h
            except: pass
        for sym,g in te.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(g,FEAT,ss[sym],hh[sym]); rec.append(pd.DataFrame({"symbol":sym,"open_time":g["open_time"].values,"pred_x":models[sym].predict(X)}))
        print(f"  {c0.date()}→{c1.date()}: {len(models)} syms [{time.time()-t0:.0f}s]",flush=True)
    rec=pd.concat(rec,ignore_index=True); rec["open_time"]=pd.to_datetime(rec["open_time"],utc=True)
    bl=pd.read_parquet(BASELINE); bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True); bl["exit_time"]=pd.to_datetime(bl["exit_time"],utc=True)
    oos=bl[bl.open_time>=CUTS[0]].copy().merge(rec,on=["symbol","open_time"],how="left")
    n=oos["pred_x"].notna().sum(); oos["pred"]=oos["pred_x"].where(oos["pred_x"].notna(),oos["pred"]); oos.drop(columns=["pred_x"]).to_parquet(OUT)
    print(f"\n  wrote {OUT.name}: swapped {n:,} ({100*n/len(oos):.0f}%)\n  DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
