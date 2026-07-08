"""Generalized honest validation for shortlisted factor reps (parametrized version of
live/alpha191_validate.py — identical checks, env-driven paths/reps).

Checks: (1) coverage, (2) honest t on NON-OVERLAPPING cycles (stride H=6), (3) per-cycle orthogonal IC
(factor & target residualized on V0_LEAN WITHIN each cycle), (4) 4-fold sign stability, (5) within-cycle
placebo shuffle (look-ahead sanity).
Env: FACTORS_PATH, REPS (comma-separated factor names).
"""
import os
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
import sys; sys.path.insert(0,R)
import live.train_twobook_models as tt
V0_LEAN=list(tt.V0_LEAN)
FACTORS_PATH=os.environ["FACTORS_PATH"]; REPS=os.environ["REPS"].split(",")

fac=pd.read_parquet(FACTORS_PATH,columns=["symbol","open_time"]+REPS)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","alpha_vs_btc_realized"]+V0_LEAN)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))
d=fac.merge(pan[["symbol","open_time","fwd"]+V0_LEAN],on=["symbol","open_time"],how="inner")
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"]).reset_index(drop=True)
cyc=d["open_time"].to_numpy(); ucyc=np.array(sorted(d["open_time"].unique()))
nonoverlap=set(ucyc[::H])
Xcols=d[V0_LEAN].fillna(0.0).to_numpy(dtype=float)

def within_resid(y):
    y=pd.Series(y,index=d.index).fillna(0.0).to_numpy(dtype=float); out=np.full(len(y),np.nan)
    for t,idx in pd.Series(range(len(d))).groupby(cyc).groups.items():
        ii=np.asarray(idx)
        if len(ii)<20: continue
        Xg=np.c_[np.ones(len(ii)),Xcols[ii]]; b,_,_,_=lstsq(Xg,y[ii],rcond=None); out[ii]=y[ii]-Xg@b
    return out
def rank_in_cycle(a): return pd.Series(a,index=d.index).groupby(cyc).rank(pct=True).to_numpy()
def per_cycle_ic(fr,tr,mask_nonoverlap=False):
    df=pd.DataFrame({"c":cyc,"f":fr,"t":tr}).dropna()
    if mask_nonoverlap: df=df[df["c"].isin(nonoverlap)]
    ics=df.groupby("c").apply(lambda g: g["f"].corr(g["t"],method="spearman") if len(g)>=20 else np.nan).dropna()
    return float(ics.mean()), (float(ics.mean()/ics.std()*np.sqrt(len(ics))) if ics.std()>0 else np.nan), len(ics), ics

tgt_resid=within_resid(d["fwd"].to_numpy())
rng=np.random.RandomState(0)
print(f"rows {len(d)}, cycles {len(ucyc)}, non-overlap cycles {len(nonoverlap)}\n")
print(f"{'factor':14s} {'cover%':>6s} {'rawIC':>8s} {'raw_t/no':>8s} {'margIC':>8s} {'marg_t/no':>9s} {'folds+':>7s} {'placebo_t':>9s}")
for f in REPS:
    fv=d[f].to_numpy(); cover=100*np.isfinite(fv).mean()
    ic_r,t_r,_,_=per_cycle_ic(rank_in_cycle(fv), rank_in_cycle(d["fwd"].to_numpy()), mask_nonoverlap=True)
    fr=rank_in_cycle(within_resid(fv)); tr=rank_in_cycle(tgt_resid)
    ic_m,t_m,_,ics_m=per_cycle_ic(fr,tr,mask_nonoverlap=True)
    folds=np.array_split(np.array(sorted(ics_m.index)),4)
    fsign=sum(1 for fo in folds if ics_m.reindex(fo).mean()* (1 if ic_m>=0 else -1) > 0)
    tsh=pd.Series(tgt_resid,index=d.index).groupby(cyc).transform(lambda s: s.sample(frac=1,random_state=rng).values)
    _,tp,_,_=per_cycle_ic(fr, rank_in_cycle(tsh.to_numpy()), mask_nonoverlap=True)
    print(f"{f:14s} {cover:6.0f} {ic_r:+8.4f} {t_r:+8.2f} {ic_m:+8.4f} {t_m:+9.2f} {fsign:>5d}/4 {tp:+9.2f}")
print("\nGuide: honest bar |raw_t/no|&|marg_t/no|>~2.5 ; folds+ >=3/4 ; |placebo_t|<~2 (else look-ahead).")
print("VALDONE")
