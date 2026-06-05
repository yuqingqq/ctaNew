"""Confirm the #165 IC gain was POOLING, not nonlinearity. Same 4 folds, apples-to-apples:
  P0  per-symbol linear (production architecture: one RidgeCV per symbol)
  P1  pooled linear (one cross-sectional RidgeCV, sym-demeaned features + sym dummies optional)
  P2  pooled GBM
Mean OOS per-cycle Spearman IC.  P1 >> P0 with P2<=P1  => pooling is the lever.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0=tt.V0; EMB=pd.Timedelta(days=1); HL=60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-12-01","2026-02-01","2026-04-01","2026-05-27"]]
F,flowcols=tt.build_flow()
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F,on=["symbol","open_time"],how="left")
FEATS=V0+flowcols
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
P=PAN[PAN["xs_z"].notna()].copy()

def cyc_ic(te): return te.groupby("open_time").apply(lambda x:x["pred_"].corr(x["xs_z"],method="spearman")).mean()

def per_symbol(folds):
    ics=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[P.exit_time<fc]; te=P[(P.open_time>=c0)&(P.open_time<c1)].copy()
        if len(tr)<5000 or len(te)==0: continue
        te["pred_"]=np.nan; t_end=tr["open_time"].max()
        for sym,gtr in tr.groupby("symbol"):
            if len(gtr)<300: continue
            gte=te[te.symbol==sym]
            if len(gte)==0: continue
            X=gtr[FEATS].fillna(0).to_numpy(); mu,sg=X.mean(0),X.std(0); sg[sg==0]=1
            w=np.exp(-((t_end-gtr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            m=RidgeCV(alphas=(0.1,1,10,100)).fit((X-mu)/sg,gtr["xs_z"],sample_weight=w)
            te.loc[gte.index,"pred_"]=m.predict((gte[FEATS].fillna(0).to_numpy()-mu)/sg)
        ics.append(cyc_ic(te.dropna(subset=["pred_"])))
    return float(np.mean(ics))

def pooled(kind):
    ics=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[P.exit_time<fc]; te=P[(P.open_time>=c0)&(P.open_time<c1)].copy()
        if len(tr)<5000 or len(te)==0: continue
        w=np.exp(-((tr["open_time"].max()-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        Xtr=tr[FEATS].fillna(0).to_numpy(); Xte=te[FEATS].fillna(0).to_numpy()
        if kind=="lin":
            mu,sg=Xtr.mean(0),Xtr.std(0); sg[sg==0]=1
            m=RidgeCV(alphas=(0.1,1,10,100)).fit((Xtr-mu)/sg,tr["xs_z"],sample_weight=w)
            te["pred_"]=m.predict((Xte-mu)/sg)
        else:
            m=HistGradientBoostingRegressor(max_depth=4,max_iter=200,learning_rate=0.05,
                min_samples_leaf=200,l2_regularization=1.0).fit(Xtr,tr["xs_z"],sample_weight=w)
            te["pred_"]=m.predict(Xte)
        ics.append(cyc_ic(te))
    return float(np.mean(ics))

p0=per_symbol(None); p1=pooled("lin"); p2=pooled("gbm")
print("============== POOLING vs NONLINEARITY (same 4 folds, OOS per-cycle IC) ==============")
print(f"  P0  per-symbol linear (PRODUCTION arch)   IC = {p0:+.4f}")
print(f"  P1  pooled linear (one cross-sec model)   IC = {p1:+.4f}   (pooling lever: {p1-p0:+.4f})")
print(f"  P2  pooled GBM                            IC = {p2:+.4f}   (nonlinearity: {p2-p1:+.4f})")
print(f"\n  VERDICT: {'POOLING is the lever' if (p1-p0)>abs(p2-p1) and p1>p0 else 'mixed'} | "
      f"pooling {p1-p0:+.4f} vs nonlinearity {p2-p1:+.4f}")
