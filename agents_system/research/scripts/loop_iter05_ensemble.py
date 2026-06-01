"""LOOP iter-05 — gen xs_z per-sym preds (recency60 WF) + build two system files:
(a) xs_z-alone, (b) 50/50 ensemble rank(resid_z=recency60) + rank(xs_z).
Both written as system-compatible preds files for replay.
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
RECENCY=REPO/"live/state/convexity/recency60_preds.parquet"   # resid_z + recency60 (iter-041)
OUT_XSZ=REPO/"live/state/convexity/xsz60_preds.parquet"
OUT_ENS=REPO/"live/state/convexity/ensemble60_preds.parquet"
FEAT=x6.BASE+x6.COHORT_EXTRAS; EMB=pd.Timedelta(days=1); HL=60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

def main():
    t0=time.time(); print("=== LOOP iter-05: xs_z preds + ensemble ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+FEAT)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    gc=pan.groupby("open_time"); mu=gc["return_pct"].transform("mean"); sd=gc["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"]=((pan["return_pct"]-mu)/sd).clip(-10,10)
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
            X=x6.apply_preproc(g,FEAT,ss[sym],hh[sym]); rec.append(pd.DataFrame({"symbol":sym,"open_time":g["open_time"].values,"pred_xsz":models[sym].predict(X)}))
        print(f"  {c0.date()}→{c1.date()}: {len(models)} syms [{time.time()-t0:.0f}s]",flush=True)
    rec=pd.concat(rec,ignore_index=True); rec["open_time"]=pd.to_datetime(rec["open_time"],utc=True)

    bl=pd.read_parquet(BASELINE); bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True); bl["exit_time"]=pd.to_datetime(bl["exit_time"],utc=True)
    oos=bl[bl.open_time>=CUTS[0]].copy()
    # (a) xs_z-alone
    a=oos.merge(rec,on=["symbol","open_time"],how="left"); a["pred"]=a["pred_xsz"].where(a["pred_xsz"].notna(),a["pred"]); a.drop(columns=["pred_xsz"]).to_parquet(OUT_XSZ)
    # (b) ensemble: 0.5*rank(resid_z recency) + 0.5*rank(xs_z), per cycle
    rc=pd.read_parquet(RECENCY)[["symbol","open_time","pred"]].rename(columns={"pred":"pred_resid"}); rc["open_time"]=pd.to_datetime(rc["open_time"],utc=True)
    e=oos.merge(rec,on=["symbol","open_time"],how="left").merge(rc,on=["symbol","open_time"],how="left")
    g2=e.groupby("open_time")
    rr=g2["pred_resid"].rank(pct=True); rx=g2["pred_xsz"].rank(pct=True)
    ens=0.5*rr+0.5*rx
    e["pred"]=ens.where(ens.notna(), e["pred"])   # fallback to baseline pred where missing
    e.drop(columns=["pred_xsz","pred_resid"]).to_parquet(OUT_ENS)
    print(f"\n  wrote {OUT_XSZ.name} & {OUT_ENS.name}\n  DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
