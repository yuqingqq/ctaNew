"""Screen the Alpha191 factors against the beta-neutral residual target on the 175-sym 4h panel.
Two metrics per factor, both cross-sectional (per-cycle) Spearman, aggregated over cycles:
  raw_IC  : factor vs forward 24h residual alpha (alpha_vs_btc_realized, 6-bar fwd sum)
  marg_IC : resid(factor ~ V0_LEAN) vs resid(target ~ V0_LEAN)  [pooled OLS] — the ORTHOGONAL signal,
            i.e. what the factor adds beyond the 14 features the model already has. (This killed funding_z.)
Vectorized mean-IC: within-cycle z-score of ranks, product, groupby-mean -> per-cycle IC -> mean & t-stat.
Memory-safe: per-factor loop, float32. Output: ranked shortlist by |marg_IC| t-stat.
"""
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
import live.train_twobook_models as tt
V0_LEAN=list(tt.V0_LEAN)

import os as _os
fac=pd.read_parquet(_os.environ.get("FACTORS_PATH",f"{R}/data/ml/cache/alpha191_factors.parquet"))
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
FCOLS=[c for c in fac.columns if c.startswith("alpha")]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","alpha_vs_btc_realized"]+V0_LEAN)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(H).sum().shift(-(H-1)))
d=fac.merge(pan[["symbol","open_time","fwd"]+V0_LEAN],on=["symbol","open_time"],how="inner")
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")]     # focus on the in-sample trading window
d=d.dropna(subset=["fwd"]).reset_index(drop=True)
print(f"screen rows {len(d)}, cycles {d.open_time.nunique()}, factors {len(FCOLS)}",flush=True)

cyc=d["open_time"].to_numpy()
def cs_rank(a):                                   # cross-sectional pct rank per cycle
    return pd.Series(a,index=d.index).groupby(cyc).rank(pct=True).to_numpy()
def cs_z(a):                                      # within-cycle z of a (for vectorized IC)
    s=pd.Series(a,index=d.index); g=s.groupby(cyc)
    return ((s-g.transform("mean"))/g.transform("std").replace(0,np.nan)).to_numpy()
def mean_ic(fr,tr):                               # fr,tr already within-cycle-standardized ranks
    p=pd.Series(fr*tr,index=d.index).groupby(cyc).mean()
    p=p.replace([np.inf,-np.inf],np.nan).dropna()
    return float(p.mean()), float(p.mean()/p.std()*np.sqrt(len(p))) if p.std()>0 else np.nan, len(p)

# target ranks (raw + orthogonalized to V0_LEAN, pooled OLS)
X=np.c_[np.ones(len(d)), d[V0_LEAN].fillna(0).to_numpy(dtype=float)]
def resid(y):
    yv=pd.Series(y).fillna(0).to_numpy(dtype=float); b,_,_,_=lstsq(X,yv,rcond=None); return yv-X@b
tr_raw = cs_z(cs_rank(d["fwd"].to_numpy()))
tr_orth= cs_z(cs_rank(resid(d["fwd"].to_numpy())))

rows=[]
for i,f in enumerate(FCOLS,1):
    fv=d[f].to_numpy()
    if not np.isfinite(fv).any(): continue
    fr_raw = cs_z(cs_rank(fv))
    ic_raw,t_raw,_ = mean_ic(fr_raw, tr_raw)
    fr_orth= cs_z(cs_rank(resid(fv)))
    ic_m,t_m,n = mean_ic(fr_orth, tr_orth)
    rows.append((f,ic_raw,t_raw,ic_m,t_m,n))
    if i%20==0: print(f"  {i}/{len(FCOLS)}",flush=True)
res=pd.DataFrame(rows,columns=["factor","raw_IC","raw_t","marg_IC","marg_t","ncyc"]).sort_values("marg_t",key=lambda s:s.abs(),ascending=False)
res.to_csv(f"{R}/live/state/longtail/alpha191_screen.csv",index=False)
pd.set_option("display.width",140)
print("\n=== TOP 25 by |marginal-IC t-stat| (orthogonal to V0_LEAN) ===")
print(res.head(25).to_string(index=False,float_format=lambda v:f"{v:+.4f}"))
print(f"\nfactors with |marg_t|>3: {(res.marg_t.abs()>3).sum()} | >4: {(res.marg_t.abs()>4).sum()} | raw |t|>4 but marg |t|<2 (redundant): {((res.raw_t.abs()>4)&(res.marg_t.abs()<2)).sum()}")
print("SCREENDONE")
