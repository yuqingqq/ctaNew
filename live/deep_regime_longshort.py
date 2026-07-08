"""Consolidated v4 LONG vs SHORT leg edge across ALL regimes (side / bear / bull-mild / bull-deep), both windows,
to show WHY sometimes the long is the problem and sometimes the short. For each: long-leg (top-K realized) and
short-leg (-bottom-K PnL) at K=2, per-cycle Sharpe, + a CONTINUATION proxy: mean trailing 3-bar residual of the
picked names (long picks' recent momentum, short picks' recent momentum) — tests mean-reversion vs momentum.
Residual alpha, bps. RECENT v4=hl_tgt_res_*, OOS v4=hl_v4base_oos/hl_v4long_oos.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["trail3"]=a.transform(lambda s:s.shift(1).rolling(3).sum())*1e4   # PIT trailing 3-bar residual (recent momentum of the name)
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
def build(b,l):
    return (lp(b,"pb").merge(lp(l,"pl"),on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd","trail3"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
WIN={"RECENT":("hl_tgt_res_base","hl_tgt_res_long"),"OOS":("hl_v4base_oos","hl_v4long_oos")}
data={k:build(*v) for k,v in WIN.items()}
allg=pd.DatetimeIndex(sorted(set().union(*[set(d["open_time"].unique()) for d in data.values()])))
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
def stats(sub):
    L=[];S=[];Lmom=[];Smom=[]
    for ot in sub.open_time.unique():
        g=sub[sub.open_time==ot]
        if len(g)<2*K: continue
        lg=g.nlargest(K,"pl"); sg=g.nsmallest(K,"pb")
        L.append(lg["fwd"].mean()); S.append(-sg["fwd"].mean())
        Lmom.append(lg["trail3"].mean()); Smom.append(sg["trail3"].mean())
    def sh(x): x=pd.Series(x); return x.mean(), (x.mean()/x.std()*np.sqrt(len(x)) if x.std()>0 else np.nan)
    lm,ls=sh(L); sm,ss=sh(S)
    return lm,ls,sm,ss,np.nanmean(Lmom),np.nanmean(Smom),len(L)
print(f"=== v4 LONG vs SHORT leg by regime (K={K}). LONG=top-K realized; SHORT=-bottom-K PnL. Want +. ===")
print("   trail3 = mean trailing-3bar residual of the picks (LONG picks should be washed-out=NEG; SHORT picks over-extended=POS).\n")
for win,d in data.items():
    d=d.copy(); d["r30"]=d["open_time"].map(rr)
    def sel(name,mask): return (name,d[mask])
    buckets=[sel("side",(d.r30>=-0.10)&(d.r30<=0.10)),sel("bear",d.r30<-0.10),
             sel("bull-mild",(d.r30>0.10)&(d.r30<=0.20)),sel("bull-deep",d.r30>0.20)]
    print(f"--- {win} ---")
    print(f"  {'regime':<11s} {'n':>4s} | {'LONG':>7s} {'Lsh':>5s} | {'SHORT':>7s} {'Ssh':>5s} | {'L-picks mom':>11s} {'S-picks mom':>11s} | issue")
    for name,sub in buckets:
        if sub.open_time.nunique()<10: print(f"  {name:<11s} {sub.open_time.nunique():>4d} | (too few)"); continue
        lm,ls,sm,ss,lmom,smom,n=stats(sub)
        iss = "LONG" if (lm<0 and sm>0) else ("SHORT" if (sm<0 and lm>0) else ("BOTH-" if (lm<0 and sm<0) else "both+"))
        print(f"  {name:<11s} {n:>4d} | {lm:+7.1f} {ls:+5.2f} | {sm:+7.1f} {ss:+5.2f} | {lmom:+11.0f} {smom:+11.0f} | {iss}")
    print()
print("BULLLSDONE")
