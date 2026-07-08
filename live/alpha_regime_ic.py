"""Per-REGIME raw alpha quality: measure each beta-neutral factor's orthogonal IC (vs V0_LEAN) SEPARATELY in
bull / side / bear cycles. v3 runs different logic per regime, so aggregate IC hides regime-specific alpha —
a factor may only pay in one regime. Regime = btc_ret_30d (BTC 180-bar return): >+0.10 bull, <-0.10 bear, else side.
IC = per-cycle Spearman of (factor resid on V0_LEAN) vs (fwd 24h residual alpha resid on V0_LEAN), non-overlapping.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
from numpy.linalg import lstsq
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
REPS=["alpha082","alpha095","alpha023","alpha052","alpha159","alpha072","alpha047","alpha010","alpha070","alpha088"]
import live.train_twobook_models as tt
V0_LEAN=list(tt.V0_LEAN)
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","count","tb","tbq","ig"]

fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time"]+REPS)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","alpha_vs_btc_realized"]+V0_LEAN)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))

# BTC 4h -> btc_ret_30d -> regime per cycle
def fm(per):
    try:
        r=requests.get(f"{BASE}/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode()
        hdr=0 if raw.split(",",1)[0]=="open_time" else None
        d=pd.read_csv(io.StringIO(raw),header=hdr,names=None if hdr==0 else COLS); d.columns=COLS[:d.shape[1]]
        v=pd.to_numeric(d["open_time"],errors="coerce"); unit="us" if v.dropna().median()>1e15 else "ms"
        d["open_time"]=pd.to_datetime(v,unit=unit,utc=True); d["close"]=pd.to_numeric(d["close"],errors="coerce")
        return d[["open_time","close"]]
    except Exception: return None
grid=pd.DatetimeIndex(sorted(fac["open_time"].unique()))
with ThreadPoolExecutor(max_workers=16) as ex:
    parts=[p for p in ex.map(fm,pd.period_range("2020-08",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"].reindex(grid).ffill()
r30=(btc/btc.shift(180)-1)
reg=pd.Series(np.where(r30>0.10,"bull",np.where(r30<-0.10,"bear","side")),index=grid)
regmap=reg.to_dict()

d=fac.merge(pan[["symbol","open_time","fwd"]+V0_LEAN],on=["symbol","open_time"],how="inner")
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"]).reset_index(drop=True)
d["regime"]=d["open_time"].map(regmap)
cyc=d["open_time"].to_numpy(); Xc=d[V0_LEAN].fillna(0.0).to_numpy(float)
print("cycles by regime:", d.groupby("regime")["open_time"].nunique().to_dict(),"\n")

def wresid(y):
    y=pd.Series(y,index=d.index).fillna(0.0).to_numpy(float); out=np.full(len(y),np.nan)
    for t,idx in pd.Series(range(len(d))).groupby(cyc).groups.items():
        ii=np.asarray(idx)
        if len(ii)<20: continue
        Xg=np.c_[np.ones(len(ii)),Xc[ii]]; b,_,_,_=lstsq(Xg,y[ii],rcond=None); out[ii]=y[ii]-Xg@b
    return out
def rk(a): return pd.Series(a,index=d.index).groupby(cyc).rank(pct=True).to_numpy()
tr=rk(wresid(d["fwd"].to_numpy()))
def ic_by_regime(fr):
    df=pd.DataFrame({"c":cyc,"reg":d["regime"].to_numpy(),"f":fr,"t":tr}).dropna()
    out={}
    for rg,g in df.groupby("reg"):
        ics=g.groupby("c").apply(lambda x:x["f"].corr(x["t"],method="spearman") if len(x)>=20 else np.nan).dropna()
        out[rg]=(ics.mean(), ics.mean()/ics.std()*np.sqrt(len(ics)) if ics.std()>0 else np.nan, len(ics))
    return out

print(f"{'factor':9s} | {'BULL ic(t,n)':>20s} | {'SIDE ic(t,n)':>20s} | {'BEAR ic(t,n)':>20s}")
for f in REPS:
    fr=rk(wresid(d[f].to_numpy())); o=ic_by_regime(fr)
    def fmt(rg):
        if rg not in o: return f"{'--':>20s}"
        ic,t,n=o[rg]; return f"{ic:+.4f} (t{t:+.1f} n{n})"
    print(f"{f:9s} | {fmt('bull'):>20s} | {fmt('side'):>20s} | {fmt('bear'):>20s}")
print("RDONE")
