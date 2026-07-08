"""Characterize the TIP distribution per regime: is the tip NOISIER, or DIFFERENTLY SHAPED (fat tail/skew), in the
failure regimes (bull)? For the model's SHORT picks (bottom-K base) and LONG picks (top-K long), per regime x window:
 - individual-pick fwd: mean, STD (noise), SKEW (tail shape). Short-side squeeze = positive skew on shorted names.
 - per-cycle edge: mean, per-cycle Sharpe (SNR). Lower SNR = the edge is drowned in noise.
 - full-universe fwd dispersion (std) + skew per regime for context.
bps. RECENT v4=hl_tgt_res_*, OOS v4=hl_v4*_oos.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from scipy import stats as sps
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
pan=pan.dropna(subset=["fwd"])
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
def dist(sub,col,asc):
    ind=[]; cyc=[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<K: continue
        pk=g.nsmallest(K,col) if asc else g.nlargest(K,col)
        ind.extend(pk["fwd"].tolist()); cyc.append(pk["fwd"].mean())
    ind=np.array(ind); cyc=np.array(cyc)
    csh=cyc.mean()/cyc.std()*np.sqrt(len(cyc)) if cyc.std()>0 else np.nan
    return ind.mean(), ind.std(), sps.skew(ind), csh
for win,d in frames.items():
    d=d.copy(); d["r30"]=d["open_time"].map(rr)
    print(f"\n=== {win} — TIP distribution by regime (individual-pick fwd: mean/STD/SKEW; per-cycle SNR) ===")
    print(f"  {'regime':<11s} {'n':>4s} | SHORT-picks {'mean':>5s} {'std':>5s} {'skew':>5s} {'SNR':>5s} | LONG-picks {'mean':>5s} {'std':>5s} {'skew':>5s} {'SNR':>5s} | xs-disp")
    for lab,mask in [("side",(d.r30>=-0.10)&(d.r30<=0.10)),("bear",d.r30<-0.10),("bull-mild",(d.r30>0.10)&(d.r30<=0.20)),("bull-deep",d.r30>0.20)]:
        sub=d[mask]; nc=sub.open_time.nunique()
        if nc<20: print(f"  {lab:<11s} {nc:>4d} | (too few)"); continue
        sm,ss,sk,ssn=dist(sub,"v4b",True); lm,ls,lk,lsn=dist(sub,"v4l",False)
        xsd=sub.groupby("open_time")["fwd"].std().median()
        print(f"  {lab:<11s} {nc:>4d} | {'':11s} {sm:+5.0f} {ss:5.0f} {sk:+5.1f} {ssn:+5.2f} | {'':10s} {lm:+5.0f} {ls:5.0f} {lk:+5.1f} {lsn:+5.2f} | {xsd:5.0f}")
print("\n(std=noise; skew: SHORT +skew = fat right tail on shorted names=squeeze risk; SNR=per-cycle Sharpe(~2.4x infl); xs-disp=median cross-sec std.)")
print("TIPDISTDONE")
