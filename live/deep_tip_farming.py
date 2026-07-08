"""We farm the TIP, not the average. Deep analysis, 2023-2026:
 (A) Tip GRADIENT + reliability by depth: long-leg realized alpha at top-{1,2,3,5,10,20,dec}, short-leg at bottom-,
     with monthly %pos + monthly-Sharpe. Is the very tip the strongest AND stable, or noisy/anti-calibrated at the extreme?
 (B) Breadth vs STABILITY: L/S at K={1,2,3,5,10,20,30}, monthly mean/std, %pos months, longest neg streak. Which K
     converts the (stable) skill into a STABLE harvest?
 (C) Which leg drives tip instability (long vs short, monthly)."""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def cat(paths,col):
    ps=[]
    for p in paths:
        d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); ps.append(d)
    return pd.concat(ps).drop_duplicates(["symbol","open_time"]).rename(columns={"pred":col})
base=cat(["hl_lean175_oos","hl_lean175"],"base"); lng=cat(["hl_residrev_oos","hl_residrev_lean"],"pl")
d=base.merge(lng,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]).dropna(subset=["fwd"])
ANN=np.sqrt(365)
# (A) tip gradient by depth
print("=== (A) TIP GRADIENT + reliability by depth (2023-2026) — is the VERY tip strongest AND stable? ===\n")
print(f"  {'depth':>6s} | {'LONG realized':>14s} {'mo%pos':>6s} {'moSh':>5s} | {'SHORT PnL':>10s} {'mo%pos':>6s} {'moSh':>5s}")
def legseries(col,depth,side):
    v={}
    for ot,g in d.groupby("open_time"):
        n=len(g)
        if n<2*max(depth if isinstance(depth,int) else 30,5): continue
        k=depth if isinstance(depth,int) else max(1,n//10)
        pick=g.nlargest(k,col) if side=="long" else g.nsmallest(k,col)
        v[ot]=pick["fwd"].mean()
    return pd.Series(v)
for depth in [1,2,3,5,10,20,"dec"]:
    lo=legseries("pl",depth,"long"); sh=legseries("base",depth,"short")
    lom=lo.resample("1ME").mean(); shm=sh.resample("1ME").mean()
    def mosh(s): s=s.dropna(); return s.mean()/s.std()*np.sqrt(len(s)) if len(s)>3 and s.std()>0 else np.nan
    print(f"  {str(depth):>6s} | {lo.mean():+14.1f} {100*(lom>0).mean():5.0f}% {mosh(lom):+5.2f} | {-sh.mean():+10.1f} {100*(shm<0).mean():5.0f}% {mosh(-shm):+5.2f}")
print("  (LONG wants top realized HIGH & stable; SHORT wants bottom realized LOW so -mean HIGH; mo%pos = months leg paid)")
# (B) breadth vs stability
print("\n=== (B) BREADTH vs STABILITY — L/S at each K, monthly (which K best converts stable skill?) ===")
print(f"  {'K':>3s} {'L/S mean':>9s} {'moSh':>6s} {'mo%pos':>7s} {'longest_neg':>12s}")
for K in [1,2,3,5,10,20,30]:
    v={}
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K: continue
        v[ot]=g.nlargest(K,"pl")["fwd"].mean()-g.nsmallest(K,"base")["fwd"].mean()
    m=pd.Series(v).resample("1ME").mean().dropna()
    neg=(m<0).astype(int); st=mx=0
    for x in neg: st=st+1 if x else 0; mx=max(mx,st)
    mosh=m.mean()/m.std()*np.sqrt(len(m)) if m.std()>0 else np.nan
    print(f"  {K:>3d} {m.mean():+9.1f} {mosh:+6.2f} {100*(m>0).mean():6.0f}% {mx:>10d}mo")
print("DTFDONE")
