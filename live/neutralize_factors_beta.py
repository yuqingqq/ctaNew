"""Frame-fix: the convexity book farms BTC-beta-neutral cross-sectional alpha, but the Alpha191 factors were
built from RAW prices (dominated by BTC beta). Neutralize every factor against BTC beta so it measures the
IDIOSYNCRATIC signal, matching the strategy's residual frame.

Steps (all PIT):
 1. Fetch BTCUSDT 4h closes (Vision monthly).
 2. Trailing 180-bar BTC beta per (symbol,bar) = cov(r_sym,r_btc)/var(r_btc), shift(1)  [matches bot's beta window].
 3. Cross-sectional (per-cycle) residualize each factor on beta: resid = f - fbar - slope*(beta-betabar),
    slope = sum(fc*bc)/sum(bc^2). Removes the linear beta-loading; keeps idiosyncratic factor.
Output: data/ml/cache/alpha191_factors_betaneut.parquet
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; BASE="https://data.binance.vision/data/futures/um/monthly/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","count",
      "taker_buy_volume","taker_buy_quote_volume","ignore"]

fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors.parquet")
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
FCOLS=[c for c in fac.columns if c.startswith("alpha")]
grid=pd.DatetimeIndex(sorted(fac["open_time"].unique()))
MONTHS=pd.period_range("2020-08", grid.max().to_period("M"), freq="M")

def _ts(col):
    v=pd.to_numeric(col,errors="coerce"); unit="us" if v.dropna().median()>1e15 else "ms"
    return pd.to_datetime(v,unit=unit,utc=True)
def fetch_month(per):
    url=f"{BASE}/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip"
    try:
        r=requests.get(url,timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode()
        hdr=0 if raw.split(",",1)[0]=="open_time" else None
        d=pd.read_csv(io.StringIO(raw),header=hdr,names=None if hdr==0 else COLS); d.columns=COLS[:d.shape[1]]
        d["open_time"]=_ts(d["open_time"]); d["close"]=pd.to_numeric(d["close"],errors="coerce")
        return d[["open_time","close"]]
    except Exception: return None
print("fetching BTCUSDT 4h...",flush=True)
with ThreadPoolExecutor(max_workers=16) as ex:
    parts=[p for p in ex.map(fetch_month,MONTHS) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()
btc=btc.reindex(grid).ffill(); btc_r=np.log(btc["close"]).diff()
print(f"BTC bars {btc['close'].notna().sum()}",flush=True)

# per-symbol trailing 180-bar beta vs BTC (PIT via shift 1)
close=fac.merge(pd.read_parquet(f"{R}/data/ml/cache/alpha191_ohlc4h.parquet",columns=["symbol","open_time","close"])
                .assign(open_time=lambda d:pd.to_datetime(d["open_time"],utc=True)),on=["symbol","open_time"],how="left")
C=close.pivot(index="open_time",columns="symbol",values="close").reindex(grid)
r=np.log(C).diff(); br=btc_r.reindex(C.index)
bvar=br.rolling(180,min_periods=42).var()
beta=pd.DataFrame({s: r[s].rolling(180,min_periods=42).cov(br)/bvar for s in C.columns}).shift(1)
betaL=beta.stack().rename("beta").reset_index()
betaL.columns=["open_time","symbol","beta"]
d=fac.merge(betaL,on=["open_time","symbol"],how="left")
print(f"beta merged, non-nan {d['beta'].notna().mean():.2f}",flush=True)

# cross-sectional per-cycle residualization of each factor on beta
cyc=d["open_time"].to_numpy(); b=d["beta"].to_numpy()
gb=pd.Series(b,index=d.index).groupby(cyc)
bbar=gb.transform("mean"); bc=(pd.Series(b,index=d.index)-bbar); bc=bc.fillna(0.0)
bc2sum=pd.Series(bc.values**2,index=d.index).groupby(cyc).transform("sum")
out=d[["symbol","open_time"]].copy()
for i,f in enumerate(FCOLS,1):
    fv=pd.Series(d[f].to_numpy(),index=d.index)
    fc=fv-fv.groupby(cyc).transform("mean")
    num=pd.Series((fc.values*bc.values),index=d.index).groupby(cyc).transform("sum")
    slope=(num/bc2sum.replace(0,np.nan)).fillna(0.0)
    out[f]=(fc-slope*bc).astype("float32").values
    if i%25==0: print(f"  neutralized {i}/{len(FCOLS)}",flush=True)
out.to_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet")
print(f"SAVED betaneut factors: {len(out)} rows, {len(FCOLS)} factors")
print("NEUTDONE")
