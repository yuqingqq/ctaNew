"""Loop iter1: WEEKLY-retrain walk-forward vs monthly. Frozen-forward found fresh IC +0.039 > stale +0.026 —
does retraining weekly (vs monthly) recover IC and lift the strategy? Honest: per-cycle IC + strategy Sharpe.
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); RR=["resid_rev_2","resid_rev_3"]; EMB=pd.Timedelta(days=1); HL=60.0
OOS0=pd.Timestamp("2025-10-04",tz="UTC"); END=pd.Timestamp("2026-06-05",tz="UTC")
WEEKLY=pd.date_range(OOS0, END, freq="7D", tz="UTC")    # weekly refit boundaries
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
feats=V0+RR
def gen(cuts):
    rec=[]
    for i in range(len(cuts)-1):
        c0,c1=cuts[i],cuts[i+1]; tr=PAN[(PAN.exit_time<c0-EMB)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if not len(te): continue
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"xs_z":gte["xs_z"].values}))
            except Exception: pass
    return pd.concat(rec,ignore_index=True)
t0=time.time(); W=gen(list(WEEKLY)); W["open_time"]=pd.to_datetime(W["open_time"],utc=True)
W=W[W.open_time>="2026-02-01"]   # compare on the same window as monthly frozen-forward analysis
ic=W.dropna().groupby("open_time").apply(lambda x:x["pred"].corr(x["xs_z"],method="spearman")).dropna()
print(f"WEEKLY-refit WF ({len(WEEKLY)} cuts, {time.time()-t0:.0f}s): per-cycle IC 2/01-6/04 mean {ic.mean():+.4f}  %>0 {(ic>0).mean()*100:.0f}%")
print(f"  REF: monthly-refit +0.0392 | frozen +0.0257")
print(f"  weekly vs monthly IC lift: {ic.mean()-0.0392:+.4f}  ({'WEEKLY HELPS' if ic.mean()>0.0392+0.003 else 'no gain over monthly — cadence not the lever' if ic.mean()<0.0392+0.003 else 'marginal'})")
W.to_parquet(REPO/"live/state/v3loop/weekly_long_preds.parquet")
print("DONE weekly_wf")
