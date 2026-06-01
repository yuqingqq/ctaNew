"""LOOP iter-03 (P2→system) — generate per-sym raw_rank-target preds (monthly WF, recency60)
into a system-compatible file, for replay validation of the target breakthrough.

Per-sym Ridge on raw_rank target (cross-sectional rank of fwd return), recency 60d, monthly
expanding-window walk-forward over the OOS window. Swap pred into the baseline preds structure
(keep alpha_A/return_pct/exit_time/fold) so the bot's load_preds reads it.
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
OUT=REPO/"live/state/convexity/rank60_preds.parquet"
FEAT=x6.BASE+x6.COHORT_EXTRAS; EMB=pd.Timedelta(days=1); HL=60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in
      ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

def main():
    t0=time.time(); print("=== LOOP iter-03: generate per-sym raw_rank preds ===\n",flush=True)
    pan=pd.read_parquet(PANEL,columns=["symbol","open_time","exit_time","return_pct"]+FEAT)
    pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
    pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"]).reset_index(drop=True)
    pan["raw_rank"]=pan.groupby("open_time")["return_pct"].rank(pct=True)
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fit_cut=c0-EMB
        tr=pan[(pan.exit_time<fit_cut)&pan["raw_rank"].notna()]; te=pan[(pan.open_time>=c0)&(pan.open_time<c1)]
        t_end=tr["open_time"].max(); models,ss,hh={},{},{}
        for sym,g in tr.groupby("symbol"):
            if len(g)<300: continue
            try:
                s,h=x6.fit_preproc(g,FEAT); X=x6.apply_preproc(g,FEAT,s,h)
                w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                models[sym]=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,g["raw_rank"].to_numpy(),sample_weight=w); ss[sym]=s; hh[sym]=h
            except: pass
        for sym,g in te.groupby("symbol"):
            if sym not in models: continue
            X=x6.apply_preproc(g,FEAT,ss[sym],hh[sym]); rec.append(pd.DataFrame({"symbol":sym,"open_time":g["open_time"].values,"pred_rk":models[sym].predict(X)}))
        print(f"  {c0.date()}→{c1.date()}: {len(models)} syms [{time.time()-t0:.0f}s]",flush=True)
    rec=pd.concat(rec,ignore_index=True); rec["open_time"]=pd.to_datetime(rec["open_time"],utc=True)
    bl=pd.read_parquet(BASELINE); bl["open_time"]=pd.to_datetime(bl["open_time"],utc=True); bl["exit_time"]=pd.to_datetime(bl["exit_time"],utc=True)
    oos=bl[bl.open_time>=CUTS[0]].copy().merge(rec,on=["symbol","open_time"],how="left")
    n=oos["pred_rk"].notna().sum(); oos["pred"]=oos["pred_rk"].where(oos["pred_rk"].notna(),oos["pred"]); oos=oos.drop(columns=["pred_rk"])
    oos.to_parquet(OUT)
    print(f"\n  wrote {OUT}: {len(oos):,} rows, swapped {n:,} ({100*n/len(oos):.0f}%)\n  DONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
