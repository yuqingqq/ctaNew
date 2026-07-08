"""Deep analysis: WHY do the factors hurt, and how does K interact?
Compare v4 baseline (V0_LEAN+resid_rev) vs +factors, both residual target, gross forward 24h residual alpha.
(A) K-sweep: gross L/S factor vs baseline at K=1..50. If factor helps ONLY at high K -> its IC gain is broad
    (middle of the book), wasted at the sparse traded extremes. If it hurts at ALL K -> genuinely bad selection.
(B) Leg decomposition: which leg (long/short) the factor breaks, per K.
(C) Turnover: does the factor churn the picks (more cost)?
(D) Tail: worst-cycle / kurtosis / %pos at low K -> does the factor add tail risk (explains 69% worse maxDD)?
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from scipy.stats import kurtosis
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
b=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"base"})
f=pd.read_parquet(f"{R}/live/state/convexity/hl_v4fac_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"fac"})
for x in (b,f): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
d=b.merge(f,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
print(f"rows {len(d)} cycles {d.open_time.nunique()}  corr(base,fac) {d['base'].corr(d['fac'],method='spearman'):.3f}\n")

# (A) K-sweep gross L/S
print("=== (A) K-sweep: gross L/S (bps) factor vs baseline — does the factor help at HIGH K? ===")
print(f"  {'K':>3s} {'base L/S':>9s} {'fac L/S':>9s} {'Δ':>7s} | {'base long':>9s} {'fac long':>9s} | {'base short':>10s} {'fac short':>9s}")
for K in [1,2,3,5,10,20,40]:
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K: continue
        bl=g.nlargest(K,"base")["fwd"].mean(); bs=g.nsmallest(K,"base")["fwd"].mean()
        fl=g.nlargest(K,"fac")["fwd"].mean(); fs=g.nsmallest(K,"fac")["fwd"].mean()
        rows.append((bl,bs,fl,fs))
    A=np.array(rows); bl,bs,fl,fs=A.mean(0)
    print(f"  {K:>3d} {bl-bs:+9.1f} {fl-fs:+9.1f} {(fl-fs)-(bl-bs):+7.1f} | {bl:+9.1f} {fl:+9.1f} | {-bs:+10.1f} {-fs:+9.1f}")

# (C) turnover + (D) tail at K=2
print("\n=== (C/D) at K=2: turnover & tail (why net maxDD blew up) ===")
def book_series(col,K):
    prev=set(); ls=[]; ch=[]
    for ot,g in sorted(d.groupby("open_time"),key=lambda kv:kv[0]):
        if len(g)<2*K: continue
        L=set(g.nlargest(K,col)["symbol"]); S=set(g.nsmallest(K,col)["symbol"]); book=L|S
        ls.append(g[g.symbol.isin(L)]["fwd"].mean()-g[g.symbol.isin(S)]["fwd"].mean())
        ch.append(len(book^prev)); prev=book
    return np.array(ls),np.mean(ch)
for col,lbl in [("base","baseline"),("fac","+factors")]:
    ls,tn=book_series(col,2)
    print(f"  {lbl:10s}: mean{ls.mean():+.1f} %pos{100*(ls>0).mean():.0f} worst{ls.min():+.0f} kurtosis{kurtosis(ls):+.1f} turnover(legs/cyc){tn:.2f}")
print("VDFDONE")
