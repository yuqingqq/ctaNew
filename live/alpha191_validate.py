"""Validate the Alpha191 screen for the 7 de-duplicated representatives BEFORE any backtest.
Checks:
 (1) coverage  - non-NaN fraction (a factor that's mostly NaN can't be trusted).
 (2) honest t  - IC t-stat on NON-OVERLAPPING cycles (stride H=6) — removes the 6-bar forward-overlap
                 inflation that made the screen t-stats ~2.4x too big.
 (3) per-cycle orthogonal IC - residualize BOTH factor and target on V0_LEAN WITHIN each cycle (the frame a
                 cross-sectional model actually uses), vs the screen's pooled OLS. Real orthogonal signal must survive.
 (4) fold stability - marg IC sign across 4 time folds (reject fold-concentrated).
 (5) placebo   - shuffle target across symbols within each cycle -> IC must collapse to ~0 (look-ahead sanity).
"""
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
REPS=["alpha082","alpha095","alpha023","alpha122","alpha071","alpha040","alpha010","alpha052","alpha133","alpha159"]
import live.train_twobook_models as tt
V0_LEAN=list(tt.V0_LEAN)

import os as _os
fac=pd.read_parquet(_os.environ.get("FACTORS_PATH",f"{R}/data/ml/cache/alpha191_factors.parquet"),columns=["symbol","open_time"]+REPS)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","alpha_vs_btc_realized"]+V0_LEAN)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))
d=fac.merge(pan[["symbol","open_time","fwd"]+V0_LEAN],on=["symbol","open_time"],how="inner")
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"]).reset_index(drop=True)
cyc=d["open_time"].to_numpy(); ucyc=np.array(sorted(d["open_time"].unique()))
nonoverlap=set(ucyc[::H])                                   # every 6th cycle = independent 24h windows
Xcols=d[V0_LEAN].fillna(0.0).to_numpy(dtype=float)

def within_resid(y):                                       # residualize y on V0_LEAN WITHIN each cycle
    y=pd.Series(y,index=d.index).fillna(0.0).to_numpy(dtype=float); out=np.full(len(y),np.nan)
    for t,idx in pd.Series(range(len(d))).groupby(cyc).groups.items():
        ii=np.asarray(idx)
        if len(ii)<20: continue
        Xg=np.c_[np.ones(len(ii)),Xcols[ii]]; b,_,_,_=lstsq(Xg,y[ii],rcond=None); out[ii]=y[ii]-Xg@b
    return out
def rank_in_cycle(a): return pd.Series(a,index=d.index).groupby(cyc).rank(pct=True).to_numpy()
def per_cycle_ic(fr,tr,mask_nonoverlap=False):             # mean & t of per-cycle Spearman
    df=pd.DataFrame({"c":cyc,"f":fr,"t":tr}).dropna()
    if mask_nonoverlap: df=df[df["c"].isin(nonoverlap)]
    ics=df.groupby("c").apply(lambda g: g["f"].corr(g["t"],method="spearman") if len(g)>=20 else np.nan).dropna()
    return float(ics.mean()), (float(ics.mean()/ics.std()*np.sqrt(len(ics))) if ics.std()>0 else np.nan), len(ics), ics

tgt_resid=within_resid(d["fwd"].to_numpy())                # target residual on V0_LEAN, per cycle
rng=np.random.RandomState(0)
print(f"rows {len(d)}, cycles {len(ucyc)}, non-overlap cycles {len(nonoverlap)}\n")
print(f"{'factor':9s} {'cover%':>6s} {'rawIC':>8s} {'raw_t/no':>8s} {'margIC':>8s} {'marg_t/no':>9s} {'folds+':>7s} {'placebo_t':>9s}")
for f in REPS:
    fv=d[f].to_numpy(); cover=100*np.isfinite(fv).mean()
    # raw IC on non-overlapping cycles
    ic_r,t_r,_,_=per_cycle_ic(rank_in_cycle(fv), rank_in_cycle(d["fwd"].to_numpy()), mask_nonoverlap=True)
    # per-cycle orthogonal IC (residual factor vs residual target), non-overlapping t
    fr=rank_in_cycle(within_resid(fv)); tr=rank_in_cycle(tgt_resid)
    ic_m,t_m,_,ics_m=per_cycle_ic(fr,tr,mask_nonoverlap=True)
    # fold stability: sign of marg IC across 4 folds (all cycles)
    folds=np.array_split(np.array(sorted(ics_m.index)),4)
    fsign=sum(1 for fo in folds if ics_m.reindex(fo).mean()* (1 if ic_m>=0 else -1) > 0)
    # placebo: shuffle target across symbols within cycle
    tsh=pd.Series(tgt_resid,index=d.index).groupby(cyc).transform(lambda s: s.sample(frac=1,random_state=rng).values)
    _,tp,_,_=per_cycle_ic(fr, rank_in_cycle(tsh.to_numpy()), mask_nonoverlap=True)
    print(f"{f:9s} {cover:6.0f} {ic_r:+8.4f} {t_r:+8.2f} {ic_m:+8.4f} {t_m:+9.2f} {fsign:>5d}/4 {tp:+9.2f}")
print("\nGuide: honest bar |raw_t/no|&|marg_t/no|>~2.5 ; folds+ >=3/4 ; |placebo_t|<~2 (else look-ahead).")
print("VALDONE")
