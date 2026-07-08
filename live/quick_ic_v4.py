"""Do Alpha191 factors improve prediction ON TOP OF the v4 base (V0_LEAN + resid_rev), residual target?
Fast pooled WF ridge ΔIC per regime. Gate (protocol): keep only if ΔIC > ~+0.003 in a pred-using regime (side/bear).
Factors are beta-neutral. This is the fair re-test in the v4 frame (stronger base + residual target).
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
import live.train_twobook_models as tt
V0=list(tt.V0_LEAN); RR=["resid_rev_2","resid_rev_3"]; BASE=V0+RR
CUTS=[pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-07-01"]]
NEW=["alpha082","alpha095","alpha070","alpha023","alpha052","alpha010","alpha159","alpha088","alpha072","alpha047"]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); pan["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: pan[c]=pan[c].fillna(0.0)
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
g=pan.groupby("open_time"); sd=g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
pan["tgt"]=((pan["alpha_vs_btc_realized"]-g["alpha_vs_btc_realized"].transform("mean"))/sd).clip(-10,10)  # residual target
fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time"]+NEW)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True); pan=pan.merge(fac,on=["symbol","open_time"],how="left")
for c in NEW: pan[c]=pan[c].fillna(0.0)
grid=pd.DatetimeIndex(sorted(pan[pan.open_time>=CUTS[0]]["open_time"].unique()))
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
    parts=[q for q in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
def wf_ic(feats):
    recs=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-pd.Timedelta(days=1)
        tr=pan[(pan.exit_time<fc)&pan["tgt"].notna()].dropna(subset=feats); te=pan[(pan.open_time>=c0)&(pan.open_time<c1)].dropna(subset=feats+["fwd"])
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[feats]); m=Ridge(alpha=10.0).fit(sc.transform(tr[feats]),tr["tgt"])
        recs.append(te.assign(pred=m.predict(sc.transform(te[feats])))[["open_time","pred","fwd"]])
    d=pd.concat(recs); d["reg"]=d["open_time"].map(reg)
    def ic(x): s=x.groupby("open_time").apply(lambda z:z["pred"].corr(z["fwd"],method="spearman") if len(z)>=20 else np.nan).dropna(); return s.mean()
    return {"ALL":ic(d),"bull":ic(d[d.reg=="bull"]),"side":ic(d[d.reg=="side"]),"bear":ic(d[d.reg=="bear"])}
b=wf_ic(BASE)
print(f"v4 base (V0_LEAN+resid_rev, residual target) IC:  ALL {b['ALL']:+.4f}  bull {b['bull']:+.4f}  side {b['side']:+.4f}  bear {b['bear']:+.4f}\n")
print(f"{'+factor':11s} {'ΔALL':>8s} {'Δbull':>8s} {'Δside':>8s} {'Δbear':>8s}  verdict")
for f in NEW:
    o=wf_ic(BASE+[f]); dA,ds,dbr=o['ALL']-b['ALL'],o['side']-b['side'],o['bear']-b['bear']
    v="KEEP" if (ds>0.003 or dbr>0.003) else "drop"
    print(f"+{f:10s} {o['ALL']-b['ALL']:+8.4f} {o['bull']-b['bull']:+8.4f} {ds:+8.4f} {dbr:+8.4f}  {v}")
print("QV4DONE")
