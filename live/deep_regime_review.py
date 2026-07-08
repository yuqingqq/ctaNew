"""Alpha-edge regime review over 2023-2026 (OOS 2023-09/2025 + in-sample 2025-10+). Per regime & per year:
long-leg edge, short-leg edge (PnL=-fwd), L/S, per-cycle Sharpe, hit-rate, cross-sectional dispersion.
Reveals where the edge is strong/weak and the structural weaknesses (bull squeeze, 2024 break, long-leg weakness)."""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def cat(paths,col):
    parts=[]
    for p in paths:
        d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
        d["open_time"]=pd.to_datetime(d["open_time"],utc=True); parts.append(d)
    return pd.concat(parts).drop_duplicates(["symbol","open_time"]).rename(columns={"pred":col})
base=cat(["hl_lean175_oos","hl_lean175"],"base"); lng=cat(["hl_residrev_oos","hl_residrev_lean"],"pl")
d=base.merge(lng,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]).dropna(subset=["fwd"])
# BTC regime
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
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
# per-cycle legs
rows=[]
for ot,g in d.groupby("open_time"):
    if len(g)<2*K: continue
    lo=g.nlargest(K,"pl")["fwd"].mean(); sh=g.nsmallest(K,"base")["fwd"].mean(); disp=g["fwd"].std()
    rows.append((ot,reg.get(ot,"side"),ot.year,lo,sh,disp))
T=pd.DataFrame(rows,columns=["ot","reg","yr","long","short","disp"]).set_index("ot")
T["ls"]=T["long"]-T["short"]
def agg(df):
    dd=(df["ls"]/1e4).resample("1D").sum(); shp=dd.mean()/dd.std()*ANN if len(dd)>5 and dd.std()>0 else np.nan
    return pd.Series(dict(n=len(df),long=df["long"].mean(),short_pnl=-df["short"].mean(),ls=df["ls"].mean(),
                          cycSh=shp,hit=100*(df["ls"]>0).mean(),disp=df["disp"].mean()))
print("=== ALPHA EDGE BY REGIME (2023-2026, K=2 L/S, forward 24h residual alpha bps) ===\n")
print(f"  {'regime':7s} {'n':>6s} {'long':>7s} {'short_pnl':>9s} {'L/S':>7s} {'cycSh':>6s} {'hit%':>5s} {'disp':>6s}")
for rg in ["side","bear","bull","ALL"]:
    s=agg(T if rg=="ALL" else T[T.reg==rg])
    print(f"  {rg:7s} {int(s['n']):6d} {s['long']:+7.1f} {s['short_pnl']:+9.1f} {s['ls']:+7.1f} {s['cycSh']:+6.2f} {s['hit']:4.0f}% {s['disp']:6.0f}")
print("\n=== BY YEAR x REGIME (L/S mean bps; where it breaks) ===")
print(f"  {'year':6s} {'side':>18s} {'bear':>18s} {'bull':>18s}")
for y in sorted(T.yr.unique()):
    cells=[]
    for rg in ["side","bear","bull"]:
        sub=T[(T.yr==y)&(T.reg==rg)]
        cells.append(f"{sub['ls'].mean():+.0f}(n{len(sub)})" if len(sub)>5 else "--")
    print(f"  {y:6d} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")
print("\n=== DISPERSION (idiosyncratic opportunity) by year — collapse = edge dies ===")
for y in sorted(T.yr.unique()): print(f"  {y}: mean XS dispersion {T[T.yr==y]['disp'].mean():.0f} bps  | L/S {T[T.yr==y]['ls'].mean():+.0f}")
print("DRRDONE")
