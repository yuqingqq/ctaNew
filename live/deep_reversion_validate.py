"""Validate whether BOTH legs mean-revert, per regime — bypass the model, sort names by RECENT residual momentum
(trailing 3-bar residual, PIT), and measure forward residual alpha at each end:
  WASHED-OUT end (bottom-K trail3): fwd > 0 => it BOUNCES (long-side reversion works).
  OVER-EXTENDED end (top-K trail3): fwd < 0 => it REVERTS (short-side reversion works; short PnL = -fwd > 0).
  reversion-IC = -rank_corr(trail3, fwd) per cycle (>0 = mean-reversion regime; <0 = momentum regime).
Split bull by depth. Both windows. This is model-independent: pure market mean-reversion character. bps.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=3
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["trail3"]=a.transform(lambda s:s.shift(1).rolling(3).sum())          # recent residual momentum (PIT)
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
pan=pan.dropna(subset=["fwd","trail3"])
grid=pd.DatetimeIndex(sorted(pan["open_time"].unique()))
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
    parts=[q for q in ex.map(fm,pd.period_range("2022-06",grid.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
pan["r30"]=pan["open_time"].map({t:v for t,v in r30.items()})
mid=pd.Timestamp("2025-10-01",tz="UTC")
def analyze(d):
    wash=[];over=[];ic=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K: continue
        wash.append(g.nsmallest(K,"trail3")["fwd"].mean())   # washed-out -> want + (bounce)
        over.append(g.nlargest(K,"trail3")["fwd"].mean())    # over-extended -> want - (revert)
        ic.append(-g["trail3"].rank().corr(g["fwd"].rank())) # reversion IC (>0 = MR)
    def st(x): x=pd.Series(x); return x.mean(), (x.mean()/x.std()*np.sqrt(len(x)) if x.std()>0 else np.nan)
    wm,wsh=st(wash); om,osh=st(over)
    return wm,wsh,om,osh,np.nanmean(ic),len(wash)
for win,dd in [("RECENT 2025-10+",pan[pan.open_time>=mid]),("OOS 2023-25",pan[(pan.open_time<mid)&(pan.open_time>=pd.Timestamp('2023-01-01',tz='UTC'))])]:
    print(f"\n=== {win} — model-free mean-reversion by regime (sort by recent residual momentum, K={K}) ===")
    print(f"  {'regime':<11s} {'n':>5s} | {'WASHED fwd':>10s} {'sh':>5s} | {'OVER fwd':>9s} {'-sh':>5s} | {'revIC':>6s} | reverts?")
    for name,mask in [("side",(dd.r30>=-0.10)&(dd.r30<=0.10)),("bear",dd.r30<-0.10),
                      ("bull-mild",(dd.r30>0.10)&(dd.r30<=0.20)),("bull-deep",dd.r30>0.20)]:
        sub=dd[mask]
        if sub.open_time.nunique()<10: print(f"  {name:<11s} {sub.open_time.nunique():>5d} | (too few)"); continue
        wm,wsh,om,osh,ic,n=analyze(sub)
        # washed bounces if wm>0 ; over reverts if om<0
        lab=("BOTH revert" if (wm>0 and om<0) else ("only-SHORT(over reverts)" if om<0 else ("only-LONG(washed bounces)" if wm>0 else "NEITHER (momentum)")))
        print(f"  {name:<11s} {n:>5d} | {wm:+10.1f} {wsh:+5.2f} | {om:+9.1f} {-osh:+5.2f} | {ic:+6.3f} | {lab}")
print("\n(WASHED fwd>0 => losers bounce=long reverts; OVER fwd<0 => winners revert=short reverts (so -sh>0). revIC>0=mean-reversion regime.)")
print("REVVALIDATEDONE")
