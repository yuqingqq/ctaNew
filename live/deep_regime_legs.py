"""Leg x REGIME x depth: does the long leg pick real alpha in SOME regimes (stock-pick) vs broken in others
(BTC-hedge)? For each regime, long-leg realized alpha at top-{1,2,3,5} and short PnL at bottom-{1,2,3,5}, with
monthly %pos + monthly Sharpe. Informs regime-conditional construction (stock-pick long where it works, hedge where not)."""
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
grid=pd.DatetimeIndex(sorted(d["open_time"].unique()))
def fm(per):
    try:
        r=requests.get(f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode(); hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr); x.columns=["open_time","o","h","l","close","v","ct","qv","n","tb","tbq","ig"][:x.shape[1]]
        vv=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if vv.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(vv,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=16) as ex:
    parts=[q for q in ex.map(fm,pd.period_range("2022-06",grid.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
regd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
d["reg"]=d["open_time"].map(regd)
def leg(sub,col,K,side):
    v={}
    for ot,g in sub.groupby("open_time"):
        if len(g)<2*K: continue
        v[ot]=(g.nlargest(K,col) if side=="long" else g.nsmallest(K,col))["fwd"].mean()
    s=pd.Series(v); m=s.resample("1ME").mean().dropna()
    mo=m.mean()/m.std()*np.sqrt(len(m)) if len(m)>3 and m.std()>0 else np.nan
    return s.mean(), (100*(m>0).mean() if side=="long" else 100*(m<0).mean()), mo
for rg in ["bear","side","bull"]:
    sub=d[d.reg==rg]; print(f"\n=== {rg.upper()} (n_cyc {sub.open_time.nunique()}) — leg realized alpha by depth ===")
    print(f"  {'K':>3s} | {'LONG realized':>13s} {'mo%pos':>6s} {'moSh':>5s} | {'SHORT PnL(-fwd)':>15s} {'mo%pos':>6s} {'moSh':>5s}")
    for K in [1,2,3,5]:
        lm,lp,ls=leg(sub,"pl",K,"long"); sm,sp,ss=leg(sub,"base",K,"short")
        print(f"  {K:>3d} | {lm:+13.1f} {lp:5.0f}% {ls:+5.2f} | {-sm:+15.1f} {sp:5.0f}% {ss:+5.2f}")
print("\n(LONG wants realized HIGH+stable=stock-pickable; if weak/negative -> use BTC hedge there. SHORT PnL=-fwd, wants HIGH)")
print("DRLDONE")
