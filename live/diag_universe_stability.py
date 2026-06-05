"""Is the POOLED model as universe-dependent as per-symbol? IC-level stress (model-quality axis).
For each drop-level, draw random symbol subsets, refit BOTH architectures on the subset, measure mean
OOS per-cycle IC. Compare the spread (std / min) across draws. Tighter spread = more universe-robust.
NOTE: this isolates the MODEL-quality dependence. The SELECTION/concentration dependence (K=3 tail in
high-vol names) is architecture-independent and tested separately via the backtest.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0=tt.V0; EMB=pd.Timedelta(days=1); HL=60.0
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-12-01","2026-02-01","2026-04-01","2026-05-27"]]
DROPS=[0,10,20,30]; NDRAW=20; SEED0=12345
F,flowcols=tt.build_flow()
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F,on=["symbol","open_time"],how="left")
FEATS=V0+flowcols; PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
ALLSYMS=sorted(PAN.symbol.unique()); print(f"universe = {len(ALLSYMS)} syms; drops={DROPS} draws={NDRAW}")

def prep(df):
    g=df.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
    df=df.assign(xs_z=((df["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10))
    return df[df["xs_z"].notna()]

def cyc_ic(te): return te.groupby("open_time").apply(lambda x:x["pred_"].corr(x["xs_z"],method="spearman")).mean()

def run(sub, arch):
    P=prep(PAN[PAN.symbol.isin(sub)]); ics=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[P.exit_time<fc]; te=P[(P.open_time>=c0)&(P.open_time<c1)].copy()
        if len(tr)<3000 or len(te)==0: continue
        if arch=="pooled":
            Xtr=tr[FEATS].fillna(0).to_numpy(); mu,sg=Xtr.mean(0),Xtr.std(0); sg[sg==0]=1
            w=np.exp(-((tr["open_time"].max()-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            m=RidgeCV(alphas=(0.1,1,10,100)).fit((Xtr-mu)/sg,tr["xs_z"],sample_weight=w)
            te["pred_"]=m.predict((te[FEATS].fillna(0).to_numpy()-mu)/sg)
        else:
            te["pred_"]=np.nan; t_end=tr["open_time"].max()
            for sym,gtr in tr.groupby("symbol"):
                if len(gtr)<300: continue
                gte=te[te.symbol==sym]
                if len(gte)==0: continue
                X=gtr[FEATS].fillna(0).to_numpy(); mu,sg=X.mean(0),X.std(0); sg[sg==0]=1
                w=np.exp(-((t_end-gtr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                mm=RidgeCV(alphas=(0.1,1,10,100)).fit((X-mu)/sg,gtr["xs_z"],sample_weight=w)
                te.loc[gte.index,"pred_"]=mm.predict((gte[FEATS].fillna(0).to_numpy()-mu)/sg)
            te=te.dropna(subset=["pred_"])
        if len(te): ics.append(cyc_ic(te))
    return float(np.mean(ics)) if ics else np.nan

rows=[]
for arch in ("pooled","persym"):
    for d in DROPS:
        if d==0:
            ics=[run(ALLSYMS, arch)]
        else:
            ics=[]
            for j in range(NDRAW):
                rng=np.random.RandomState(SEED0+j+1000*d)
                sub=sorted(rng.choice(ALLSYMS, size=len(ALLSYMS)-d, replace=False))
                ics.append(run(sub, arch))
        ics=np.array([x for x in ics if np.isfinite(x)])
        rows.append({"arch":arch,"drop":d,"n_sub":len(ALLSYMS)-d,"mean_IC":ics.mean(),
                     "std_IC":ics.std() if len(ics)>1 else 0.0,"min_IC":ics.min(),"max_IC":ics.max()})
        print(f"[{arch} drop={d:2d}] meanIC={ics.mean():+.4f} std={ics.std() if len(ics)>1 else 0:.4f} "
              f"min={ics.min():+.4f} max={ics.max():+.4f}")
R=pd.DataFrame(rows)
print("\n============== UNIVERSE STABILITY (IC, both architectures) ==============")
print(R.round(4).to_string(index=False))
piv=R.pivot(index="drop",columns="arch",values="std_IC")
print("\nIC std across draws (lower = more universe-robust):"); print(piv.round(4).to_string())
print(f"\nVERDICT: pooled is {'MORE universe-robust (tighter IC spread)' if piv['pooled'].mean()<piv['persym'].mean() else 'AS/MORE fragile'} "
      f"| avg std pooled={piv['pooled'].mean():.4f} persym={piv['persym'].mean():.4f}")
R.to_csv(REPO/"live/state/universe_stability.csv", index=False)
