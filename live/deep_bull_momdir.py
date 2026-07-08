"""Validate whether the model's predictions are MOMENTUM-following in bull. Per regime x window:
(1) rank-corr(long-book pred v4l, trail3) and rank-corr(base-book pred v4b, trail3): sign says if the model's ranking
    aligns with recent momentum (+ = momentum: high pred = recent winner) or against it (- = reversion).
(2) NET momentum tilt of the book = mean trail3 of the K longs - mean trail3 of the K shorts. >0 = book net-LONG
    momentum (momentum-following); <0 = net-SHORT momentum (reversion book).
(3) do the picks work: long fwd, short PnL. trail3 = recent residual momentum. Both windows. bps.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["trail3"]=a.transform(lambda s:s.shift(1).rolling(3).sum())*1e4
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
pan=pan.dropna(subset=["fwd","trail3"])
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
WIN={"RECENT":("hl_tgt_res_base","hl_tgt_res_long"),"OOS":("hl_v4base_oos","hl_v4long_oos")}
frames={k:pan.merge(lp(v[0],"v4b"),on=["symbol","open_time"]).merge(lp(v[1],"v4l"),on=["symbol","open_time"]) for k,v in WIN.items()}
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
def analyze(sub):
    cl=[];cb=[];nettilt=[];lf=[];sf=[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<2*K: continue
        cl.append(g["v4l"].rank().corr(g["trail3"].rank())); cb.append(g["v4b"].rank().corr(g["trail3"].rank()))
        L=g.nlargest(K,"v4l"); S=g.nsmallest(K,"v4b")
        nettilt.append(L["trail3"].mean()-S["trail3"].mean()); lf.append(L["fwd"].mean()); sf.append(-S["fwd"].mean())
    return np.nanmean(cl),np.nanmean(cb),np.nanmean(nettilt),np.nanmean(lf),np.nanmean(sf)
for win,d in frames.items():
    d=d.copy(); d["r30"]=d["open_time"].map(rr)
    print(f"\n=== {win} — is the model MOMENTUM-following? (corr(pred,trail3): + = momentum, - = reversion) ===")
    print(f"  {'regime':<11s} {'n':>4s} | {'corr(Lpred,mom)':>15s} {'corr(Bpred,mom)':>15s} | {'net-mom tilt':>12s} | {'LONG':>6s} {'SHORT':>6s} | model is")
    for lab,mask in [("side",(d.r30>=-0.10)&(d.r30<=0.10)),("bear",d.r30<-0.10),("bull-mild",(d.r30>0.10)&(d.r30<=0.20)),("bull-deep",d.r30>0.20)]:
        sub=d[mask]; nc=sub.open_time.nunique()
        if nc<20: print(f"  {lab:<11s} {nc:>4d} | (too few)"); continue
        cl,cb,tilt,lf,sf=analyze(sub)
        # long-book momentum? high corr(Lpred,mom)>0 => longs winners. book net tilt>0 => momentum book.
        mo = "MOMENTUM-book" if tilt>0 else "reversion-book"
        print(f"  {lab:<11s} {nc:>4d} | {cl:+15.3f} {cb:+15.3f} | {tilt:+12.0f} | {lf:+6.0f} {sf:+6.0f} | {mo}")
print("\n(corr(Lpred,mom)>0: long-book longs recent winners=momentum. net-mom tilt>0: book net-long momentum=momentum-following;")
print(" <0: shorts higher-momentum than it longs = reversion book. LONG/SHORT = realized picks (bps).)")
print("BULLMOMDIRDONE")
