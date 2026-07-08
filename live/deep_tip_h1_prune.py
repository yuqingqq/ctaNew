"""MEASURE per-feature regime tip on H1 -> define per-regime prune sets (reversers = H1 tip <= 0). No retraining.
Legs: bear-LONG (top-1 in bear), bear-SHORT (bottom-4 in bear), side-SHORT (bottom-4 in side).
Feature oriented by H1 pooled rank-IC (high=>predicts high alpha). Dumps keep-sets to JSON for the retrain step.
"""
import io, zipfile, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/home/yuqing/ctaNew")
import live.train_twobook_models as tt
R="/home/yuqing/ctaNew"; H=6
V0L=[f for f in tt.V0 if not f.startswith("funding")]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"]+V0L)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
pan["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
LONG_UNIV=V0L+["resid_rev_2","resid_rev_3"]; SHORT_UNIV=V0L
d=pan.dropna(subset=["fwd"]).copy()
# restrict to the OOS pred grid so regime/halves line up with the models
grid=pd.DatetimeIndex(sorted(pd.read_parquet(f"{R}/live/state/convexity/hl_v4base_oos/v0full_hl60.parquet",columns=["open_time"])["open_time"].pipe(pd.to_datetime,utc=True).unique()))
d=d[d.open_time.isin(grid)]
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
regd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
d["reg"]=d["open_time"].map(regd); mid=grid[len(grid)//2]; d["half"]=np.where(d["open_time"]<mid,"h1","h2")
h1=d[d.half=="h1"]
def rankic(col):
    return h1.groupby("open_time").apply(lambda g:g[col].rank().corr(g["fwd"].rank())).mean()
sign={f:(1.0 if rankic(f)>=0 else -1.0) for f in set(LONG_UNIV)|set(SHORT_UNIV)}
groups_h1=[(g,g["reg"].iloc[0]) for _,g in h1.groupby("open_time")]
def tip(col,leg,reg):
    v=[]
    for g,rg in groups_h1:
        if rg!=reg: continue
        o=g[col].to_numpy()*sign[col]; gg=g.assign(_o=o)
        if leg=="long": v.append(gg.nlargest(1,"_o")["fwd"].mean())
        elif len(gg)>=4: v.append(-gg.nsmallest(4,"_o")["fwd"].mean())
    return np.mean(v) if v else np.nan
def report(leg,reg,univ):
    rows=sorted([(f,tip(f,leg,reg)) for f in univ],key=lambda x:-x[1])
    keep=[f for f,t in rows if t>0]; prune=[f for f,t in rows if t<=0]
    print(f"\n--- {reg.upper()} {leg.upper()} (H1 tip) ---")
    for f,t in rows: print(f"   {f:<26s} {t:+7.0f}  {'keep' if t>0 else 'PRUNE'}")
    print(f"   => keep({len(keep)}): {keep}")
    print(f"   => prune({len(prune)}): {prune}")
    return keep
print("=== H1 per-feature TIP -> prune sets (reversers pruned). Validate the retrained pruned books on H2 next. ===")
ks={}
ks["bear_long"]=report("long","bear",LONG_UNIV)
ks["bear_short"]=report("short","bear",SHORT_UNIV)
ks["side_short"]=report("short","side",SHORT_UNIV)
json.dump(ks,open(f"{R}/live/state/longtail/prune_keepsets.json","w"),indent=1)
print("\nsaved keep-sets -> live/state/longtail/prune_keepsets.json")
print("TIPH1DONE")
