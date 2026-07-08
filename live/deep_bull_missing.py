"""Identify the 'missing property' in BULL. Hypothesis: the bull LONG alpha is in the RAW/beta/momentum dimension
that v4's residual target strips out, so a beta-neutral residual model under-captures bull. For bull (mild/deep) x
window, measure the LONG picks (top-K) by 3 rankers — v4 residual (v4l), v3 return (v3l), momentum (trail3=recent
winners) — reporting BOTH realized RESIDUAL fwd (alpha_vs_btc) AND realized RAW fwd (return_pct). If RAW >> RESIDUAL
in bull, the long edge is beta/momentum the residual removes. K=2. bps.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized","return_pct"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
g=pan.groupby("symbol")
pan["trail3"]=g["alpha_vs_btc_realized"].transform(lambda s:s.shift(1).rolling(3).sum())
pan["fwd_res"]=g["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
pan["fwd_raw"]=g["return_pct"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
pan=pan.dropna(subset=["fwd_res","fwd_raw","trail3"])
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
def build(b,l):
    return pan.merge(lp(b,"v3l" if "ret" in l or "residrev" in l else "tmp"),on=["symbol","open_time"]) if False else \
           pan.merge(lp(l,"long"),on=["symbol","open_time"])
# need v4-long and v3-long both; build per window
WIN={"RECENT":("hl_tgt_res_long","hl_tgt_ret_long"),"OOS":("hl_v4long_oos","hl_residrev_oos")}
frames={}
for win,(v4l,v3l) in WIN.items():
    d=pan.merge(lp(v4l,"v4l"),on=["symbol","open_time"]).merge(lp(v3l,"v3l"),on=["symbol","open_time"])
    frames[win]=d
allg=pd.DatetimeIndex(sorted(set().union(*[set(d["open_time"].unique()) for d in frames.values()])))
def fm(per):
    try:
        r=requests.get(f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode(); hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr); x.columns=["open_time","o","h","l","close","v","ct","qv","n","tb","tbq","ig"][:x.shape[1]]
        vv=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if vv.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(vv,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[q for q in ex.map(fm,pd.period_range("2022-06",allg.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(allg)))).ffill(); r30=(btc/btc.shift(180)-1)
rr={t:v for t,v in r30.items()}
def longedge(sub,col,asc=False):
    res=[];raw=[]
    for ot,gg in sub.groupby("open_time"):
        if len(gg)<K: continue
        pk=gg.nsmallest(K,col) if asc else gg.nlargest(K,col)
        res.append(pk["fwd_res"].mean()); raw.append(pk["fwd_raw"].mean())
    return np.nanmean(res),np.nanmean(raw)
for win,d in frames.items():
    d=d.copy(); d["r30"]=d["open_time"].map(rr)
    print(f"\n=== {win} — BULL LONG edge: residual vs RAW fwd, by ranker (K={K}) ===")
    for lab,mask in [("bull-mild",(d.r30>0.10)&(d.r30<=0.20)),("bull-deep",d.r30>0.20),("bear(ref)",d.r30<-0.10),("side(ref)",(d.r30>=-0.10)&(d.r30<=0.10))]:
        sub=d[mask]; nc=sub.open_time.nunique()
        if nc<20: print(f"  {lab:<11s} n={nc} (too few)"); continue
        v4r,v4raw=longedge(sub,"v4l"); v3r,v3raw=longedge(sub,"v3l"); mr,mraw=longedge(sub,"trail3")
        print(f"  {lab:<11s} n={nc:>4d} | v4-resid-long: res{v4r:+6.0f} raw{v4raw:+6.0f} | v3-return-long: res{v3r:+6.0f} raw{v3raw:+6.0f} | momentum-long(trail3): res{mr:+6.0f} raw{mraw:+6.0f}")
print("\n(res=residual fwd the beta-neutral strategy farms; raw=raw return fwd. raw>>res => long alpha is beta/momentum residual removes.)")
print("BULLMISSDONE")
