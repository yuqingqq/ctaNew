"""Regime-conditional factor screen: is there a PIT-classifiable regime where a factor is reliable?

Regimes (all PIT, from data <= t-1):
  trend  : production classification from convexity_paper_bot — btc_ret_30d (180 bars):
           bear < -0.10 | side [-0.10,0.10] | mild_bull [0.10,0.15) | hot_bull >= 0.15 (BULL_DEEP_THR)
  disp   : idiosyncratic-dispersion regime (2024 root-cause axis) — trailing 42-bar mean of the
           per-cycle cross-sectional std of alpha_vs_btc_realized, above/below trailing-1y median.

Metric per factor x regime: per-cycle cross-sectional Spearman of V0_LEAN-orthogonal factor vs
V0_LEAN-orthogonal 24h fwd residual target (pooled OLS resid, full history), aggregated within
regime bucket. t-stats computed on NON-OVERLAPPING cycles only (stride H=6) — honest from the start.

Multiple-testing note: ~321 factors x 6 buckets ~ 1900 tests -> chance gives ~2 at |t|>3.3.
Only |t|>=4 with episode consistency (stage 2) counts.

Env: FACTORS_PATH, PREFIX, OUT_CSV.
"""
import os
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
import sys; sys.path.insert(0,R)
import live.train_twobook_models as tt
V0_LEAN=list(tt.V0_LEAN)
FACTORS_PATH=os.environ["FACTORS_PATH"]; PREFIX=os.environ.get("PREFIX","alpha")
OUT_CSV=os.environ["OUT_CSV"]

fac=pd.read_parquet(FACTORS_PATH)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
FCOLS=[c for c in fac.columns if c.startswith(PREFIX)]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","alpha_vs_btc_realized"]+V0_LEAN)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(H).sum().shift(-(H-1)))
d=fac.merge(pan[["symbol","open_time","fwd","alpha_vs_btc_realized"]+V0_LEAN],on=["symbol","open_time"],how="inner")
d=d.dropna(subset=["fwd"]).reset_index(drop=True)          # FULL history
cyc=d["open_time"].to_numpy(); ucyc=pd.DatetimeIndex(sorted(d["open_time"].unique()))
nonoverlap=set(ucyc[::H])
print(f"regime screen rows {len(d)}, cycles {len(ucyc)}, factors {len(FCOLS)}",flush=True)

# ---- PIT regime flags on the cycle grid ----
btc=pd.read_parquet(f"{R}/data/ml/cache/btc4h_close_cache.parquet")
btc["open_time"]=pd.to_datetime(btc["open_time"],utc=True)
btc=btc.set_index("open_time")["close"].reindex(ucyc).ffill()
btc30=(btc/btc.shift(180)-1).shift(1)                       # trailing 30d ret, PIT
trend=pd.Series(np.select([btc30<-0.10,(btc30>=-0.10)&(btc30<=0.10),(btc30>0.10)&(btc30<0.15),btc30>=0.15],
                          ["bear","side","mild_bull","hot_bull"],default=None),index=ucyc)
csd=pd.Series(d["alpha_vs_btc_realized"].to_numpy(),index=d.index).groupby(cyc).std()  # per-cycle xs dispersion
csd=csd.reindex(ucyc)
disp_tr=csd.rolling(42,min_periods=21).mean().shift(1)      # trailing 7d mean, PIT
disp_med=disp_tr.rolling(2160,min_periods=360).median()     # trailing ~1y median
disp=pd.Series(np.where(disp_tr.isna()|disp_med.isna(),None,
               np.where(disp_tr>disp_med,"disp_hi","disp_lo")),index=ucyc)
for name,s in [("trend",trend),("disp",disp)]:
    print(f"  {name}: "+", ".join(f"{k}={v}" for k,v in s.value_counts().items()),flush=True)
BUCKETS=[("bear",trend=="bear"),("side",trend=="side"),("mild_bull",trend=="mild_bull"),
         ("hot_bull",trend=="hot_bull"),("disp_hi",disp=="disp_hi"),("disp_lo",disp=="disp_lo")]

# ---- orthogonal per-cycle IC machinery (pooled resid on V0_LEAN, full history) ----
def cs_rank(a): return pd.Series(a,index=d.index).groupby(cyc).rank(pct=True).to_numpy()
def cs_z(a):
    s=pd.Series(a,index=d.index); g=s.groupby(cyc)
    return ((s-g.transform("mean"))/g.transform("std").replace(0,np.nan)).to_numpy()
X=np.c_[np.ones(len(d)), d[V0_LEAN].fillna(0).to_numpy(dtype=float)]
def resid(y):
    yv=pd.Series(y).fillna(0).to_numpy(dtype=float); b,_,_,_=lstsq(X,yv,rcond=None); return yv-X@b
tr_orth=cs_z(cs_rank(resid(d["fwd"].to_numpy())))

rows=[]
for i,f in enumerate(FCOLS,1):
    fv=d[f].to_numpy()
    if not np.isfinite(fv).any(): continue
    fr=cs_z(cs_rank(resid(fv)))
    p=pd.Series(fr*tr_orth,index=d.index).groupby(cyc).mean()  # per-cycle IC (vectorized)
    p=p.replace([np.inf,-np.inf],np.nan).reindex(ucyc)
    rec={"factor":f}
    for bname,bmask in BUCKETS:
        s=p[bmask.reindex(ucyc).fillna(False).to_numpy()]
        sn=s[s.index.isin(nonoverlap)].dropna()               # honest non-overlap subset
        rec[f"{bname}_ic"]=float(s.mean()) if len(s) else np.nan
        rec[f"{bname}_t"]=float(sn.mean()/sn.std()*np.sqrt(len(sn))) if len(sn)>10 and sn.std()>0 else np.nan
        rec[f"{bname}_n"]=int(len(sn))
    sn_all=p[p.index.isin(nonoverlap)].dropna()
    rec["full_t"]=float(sn_all.mean()/sn_all.std()*np.sqrt(len(sn_all))) if sn_all.std()>0 else np.nan
    rows.append(rec)
    if i%25==0: print(f"  {i}/{len(FCOLS)}",flush=True)
res=pd.DataFrame(rows)
res["best_abs_t"]=res[[f"{b}_t" for b,_ in BUCKETS]].abs().max(axis=1)
res=res.sort_values("best_abs_t",ascending=False)
res.to_csv(OUT_CSV,index=False)
pd.set_option("display.width",200)
cols=["factor"]+[f"{b}_t" for b,_ in BUCKETS]+["full_t"]
print("\n=== TOP 15 by best per-regime honest |t| ===")
print(res[cols].head(15).to_string(index=False,float_format=lambda v:f"{v:+.2f}"))
print("REGIMESCREENDONE")
