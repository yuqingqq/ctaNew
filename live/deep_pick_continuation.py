"""Validate: the model predicts reversion, but do the SELECTED symbols actually CHASE MOMENTUM (continue) or REVERT?
For the model's SHORT picks (over-extended, bottom-K base pred) and LONG picks (top-K long pred), per regime x window:
 - %continue = share of picks whose forward residual has the SAME sign as their trailing residual (winner keeps
   winning / loser keeps losing = chasing momentum). >50% => picks chase momentum; <50% => picks revert.
 - mean trail3 (recent momentum of the picks) and mean fwd (what they did next).
SHORT picks are shorted expecting DOWN: fwd>0 => they kept rising = chased momentum = short LOSES.
LONG picks are longed expecting UP-bounce: fwd<0 => they kept falling = chased momentum = long LOSES.
bps. RECENT v4=hl_tgt_res_*, OOS v4=hl_v4*_oos.
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
def cont(sub,col,asc):
    picks=[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<K: continue
        pk=g.nsmallest(K,col) if asc else g.nlargest(K,col)
        picks.append(pk[["trail3","fwd"]])
    if not picks: return np.nan,np.nan,np.nan
    p=pd.concat(picks); contpct=(np.sign(p["trail3"])==np.sign(p["fwd"])).mean()*100
    return contpct, p["trail3"].mean(), p["fwd"].mean()
for win,d in frames.items():
    d=d.copy(); d["r30"]=d["open_time"].map(rr)
    print(f"\n=== {win} — do the model's SELECTED picks chase momentum (continue) or revert? ===")
    print(f"  {'regime':<11s} {'n':>4s} | SHORT picks: {'%cont':>5s} {'mom':>5s} {'fwd':>5s} | LONG picks: {'%cont':>5s} {'mom':>5s} {'fwd':>5s} | picks")
    for lab,mask in [("side",(d.r30>=-0.10)&(d.r30<=0.10)),("bear",d.r30<-0.10),("bull-mild",(d.r30>0.10)&(d.r30<=0.20)),("bull-deep",d.r30>0.20)]:
        sub=d[mask]; nc=sub.open_time.nunique()
        if nc<20: print(f"  {lab:<11s} {nc:>4d} | (too few)"); continue
        scp,sm,sf=cont(sub,"v4b",True); lcp,lm,lf=cont(sub,"v4l",False)
        tag="CHASE-momentum" if (scp>50 and lcp>50) else ("short-chases" if scp>50 else ("long-chases" if lcp>50 else "revert"))
        print(f"  {lab:<11s} {nc:>4d} | {'':11s} {scp:5.0f}% {sm:+5.0f} {sf:+5.0f} | {'':10s} {lcp:5.0f}% {lm:+5.0f} {lf:+5.0f} | {tag}")
print("\n(%cont>50 = picks CONTINUE their momentum=chase (winner keeps winning); SHORT fwd>0=short squeezed; LONG fwd<0=long fails.)")
print("PICKCONTDONE")
