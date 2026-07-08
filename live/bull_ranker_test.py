"""I2 capture-layer test: is alpha070 a better BULL short ranker than the incumbent return_1d?
In BULL cycles only, short kS=2 names chosen by each ranker; short PnL = -(mean fwd 24h residual alpha of the
shorted names). Higher = better. Report distribution (mean/median/%pos/top3-share/per-cycle Sharpe + non-overlap).
Rankers: return_1d (incumbent, short highest), alpha070 (short highest, neg-IC), pred (short lowest, side-style).
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; kS=2
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","count","tb","tbq","ig"]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized","return_1d"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time","alpha070"])
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
bp=pd.read_parquet(f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
bp["open_time"]=pd.to_datetime(bp["open_time"],utc=True)
d=pan.merge(fac,on=["symbol","open_time"]).merge(bp,on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd","return_1d","alpha070","pred"])
grid=pd.DatetimeIndex(sorted(d["open_time"].unique()))
def fm(per):
    try:
        r=requests.get(f"{BASE}/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode()
        hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr,names=None if hdr==0 else COLS); x.columns=COLS[:x.shape[1]]
        v=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if v.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(v,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[p for p in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
d["reg"]=d["open_time"].map(reg); bull=d[d.reg=="bull"]
print(f"bull cycles: {bull['open_time'].nunique()}\n")
def short_pnl(colsel):   # colsel(g)->list of shorted symbols; short PnL = -mean fwd
    rows=[]
    for ot,g in bull.groupby("open_time"):
        if len(g)<kS: continue
        S=colsel(g); rows.append((ot, -g[g.symbol.isin(S)]["fwd"].mean()))
    return pd.Series([r[1] for r in rows]).dropna()
def stats(s,lbl):
    top3=s.nlargest(3).sum()/s.sum()*100 if s.sum()!=0 else np.nan
    sh=s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
    no=s.iloc[::H]; shno=no.mean()/no.std()*np.sqrt(len(no)) if no.std()>0 else np.nan
    print(f"  {lbl:16s} n{len(s):3d} mean{s.mean():+7.1f} med{s.median():+7.1f} %pos{100*(s>0).mean():3.0f} top3{top3:4.0f}% cycSharpe{sh:+.2f} | nonoverlap Sh{shno:+.2f}")
print("BULL short-leg PnL (bps, higher=better short):")
stats(short_pnl(lambda g:g.nlargest(kS,"return_1d")["symbol"]), "return_1d (INCUMB)")
stats(short_pnl(lambda g:g.nlargest(kS,"alpha070")["symbol"]),  "alpha070")
stats(short_pnl(lambda g:g.nsmallest(kS,"pred")["symbol"]),     "pred (side-style)")
print("BULLRDONE")
