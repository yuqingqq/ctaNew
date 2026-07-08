"""Does measuring/training in FULLY beta-removed (residual) space tell us if cross-sectional performance is good?
Two targets, same pooled WF ridge, features beta-neutral:
  RAW target      = xs_z(return_pct)            [what the model currently trains on — has beta]
  RESIDUAL target = xs_z(alpha_vs_btc_realized) [fully beta-removed — what the strategy actually farms]
IC always measured vs forward residual alpha. Compare baseline IC and per-feature ΔIC across the two targets, per
regime. Then line ΔIC up against the known strategy ΔSharpe to see if residual-space IC is a better predictor.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
import live.train_twobook_models as tt
V0=list(tt.V0_LEAN)
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-07-01"]]
NEW=["alpha082","alpha070","alpha010","alpha023"]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
g=pan.groupby("open_time")
def xsz(col):
    sd=g[col].transform("std").replace(0,np.nan); return ((pan[col]-g[col].transform("mean"))/sd).clip(-10,10)
pan["z_raw"]=xsz("return_pct"); pan["z_res"]=xsz("alpha_vs_btc_realized")
fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time"]+NEW)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True); pan=pan.merge(fac,on=["symbol","open_time"],how="left")
for c in NEW: pan[c]=pan[c].fillna(0.0)
grid=pd.DatetimeIndex(sorted(pan[pan.open_time>=CUTS[0]]["open_time"].unique()))
def fm(per):
    try:
        r=requests.get(f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode(); hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr); x.columns=["open_time","open","high","low","close","volume","ct","qv","n","tb","tbq","ig"][:x.shape[1]]
        v=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if v.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(v,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[p for p in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
def wf_ic(feats,tgt):
    recs=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-pd.Timedelta(days=1)
        tr=pan[(pan.exit_time<fc)&pan[tgt].notna()].dropna(subset=feats)
        te=pan[(pan.open_time>=c0)&(pan.open_time<c1)].dropna(subset=feats+["fwd"])
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[feats]); m=Ridge(alpha=10.0).fit(sc.transform(tr[feats]),tr[tgt])
        recs.append(te.assign(pred=m.predict(sc.transform(te[feats])))[["open_time","pred","fwd"]])
    d=pd.concat(recs); d["reg"]=d["open_time"].map(reg)
    def ic(df): s=df.groupby("open_time").apply(lambda x:x["pred"].corr(x["fwd"],method="spearman") if len(x)>=20 else np.nan).dropna(); return s.mean()
    return {"ALL":ic(d),"bull":ic(d[d.reg=="bull"]),"side":ic(d[d.reg=="side"]),"bear":ic(d[d.reg=="bear"])}
for tgt,lbl in [("z_raw","RAW target xs_z(return)"),("z_res","RESIDUAL target xs_z(alpha_vs_btc)")]:
    print(f"\n=== {lbl} — IC vs forward residual alpha ===")
    b=wf_ic(V0,tgt)
    print(f"  V0_LEAN base:  ALL {b['ALL']:+.4f}  bull {b['bull']:+.4f}  side {b['side']:+.4f}  bear {b['bear']:+.4f}")
    for f in NEW:
        o=wf_ic(V0+[f],tgt)
        print(f"  +{f:9s} ΔIC: ALL {o['ALL']-b['ALL']:+.4f}  bull {o['bull']-b['bull']:+.4f}  side {o['side']-b['side']:+.4f}  bear {o['bear']-b['bear']:+.4f}")
print("QRDONE")
