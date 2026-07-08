"""FAST check: does adding a new signal help the model predict cross-sectional residual alpha?
Pooled walk-forward Ridge (fast proxy for the per-symbol model): train on xs_z target per fold, predict test,
measure per-cycle Spearman IC of pred vs forward 24h residual alpha (alpha_vs_btc_realized). Compare feature sets
overall + per regime. Positive ΔIC = the signal improves cross-sectional prediction.
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
NEW=["alpha070","alpha082","alpha010","alpha095","alpha023","alpha052"]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan["exit_time"]=pd.to_datetime(pan["exit_time"],utc=True)
pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
g=pan.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
pan["xs_z"]=((pan["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time"]+NEW)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
pan=pan.merge(fac,on=["symbol","open_time"],how="left")
for c in NEW: pan[c]=pan[c].fillna(0.0)
# regime
grid=pd.DatetimeIndex(sorted(pan[pan.open_time>=CUTS[0]]["open_time"].unique()))
def fm(per):
    try:
        r=requests.get(f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode(); hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr); 
        x.columns=["open_time","open","high","low","close","volume","ct","qv","n","tb","tbq","ig"][:x.shape[1]]
        v=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if v.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(v,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[p for p in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}

def wf_ic(feats):
    recs=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-pd.Timedelta(days=1)
        tr=pan[(pan.exit_time<fc)&pan["xs_z"].notna()].dropna(subset=feats)
        te=pan[(pan.open_time>=c0)&(pan.open_time<c1)].dropna(subset=feats+["fwd"])
        if len(tr)<500 or len(te)<50: continue
        sc=StandardScaler().fit(tr[feats]); m=Ridge(alpha=10.0).fit(sc.transform(tr[feats]),tr["xs_z"])
        te=te.assign(pred=m.predict(sc.transform(te[feats])))
        recs.append(te[["open_time","symbol","pred","fwd"]])
    d=pd.concat(recs); d["reg"]=d["open_time"].map(reg)
    def ic(df): 
        s=df.groupby("open_time").apply(lambda x:x["pred"].corr(x["fwd"],method="spearman") if len(x)>=20 else np.nan).dropna()
        return s.mean(), s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
    out={"ALL":ic(d)}
    for rg in ["bull","side","bear"]: out[rg]=ic(d[d.reg==rg])
    return out

print("Pooled WF Ridge — per-cycle IC (pred vs fwd 24h residual alpha), overall + per regime\n")
base=wf_ic(V0)
def line(lbl,o,b=None):
    def cell(k):
        ic,t=o[k]; s=f"{ic:+.4f}(t{t:+.1f})"
        if b is not None: s+=f" Δ{ic-b[k][0]:+.4f}"
        return s
    print(f"  {lbl:22s} ALL {cell('ALL'):28s} bull {cell('bull'):28s} side {cell('side'):28s} bear {cell('bear')}")
line("V0_LEAN (baseline)",base)
for f in NEW: line(f"+{f}",wf_ic(V0+[f]),base)
line("+ALL6",wf_ic(V0+NEW),base)
print("QICDONE")
