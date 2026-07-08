"""Isolate the regime-tailoring: test bear-LONG-only, bear-SHORT-only, and BOTH, vs pooled. Restrict per-fold
win-rate to BEAR-CONTAINING folds (tailoring is a no-op elsewhere, so all-fold win-rate is diluted). H1/H2 split.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred","fold"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
d=(lp("hl_v4base_oos","p_b")[["symbol","open_time","p_b","fold"]]
   .merge(lp("hl_v4long_oos","p_l")[["symbol","open_time","p_l"]],on=["symbol","open_time"])
   .merge(lp("hl_v4base_bearcal","t_b")[["symbol","open_time","t_b"]],on=["symbol","open_time"])
   .merge(lp("hl_v4long_bearcal","t_l")[["symbol","open_time","t_l"]],on=["symbol","open_time"])
   .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
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
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[q for q in ex.map(fm,pd.period_range("2022-06",grid.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
regd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
d["reg"]=d["open_time"].map(regd); mid=grid[len(grid)//2]
groups=[(ot,g,g["reg"].iloc[0],("h1" if ot<mid else "h2"),g["fold"].iloc[0]) for ot,g in d.groupby("open_time")]
bear_folds=set(f for _,_,rg,_,f in groups if rg=="bear")
print(f"bear-containing folds: {len(bear_folds)} of 33\n")
def series(KL,KS,tl,ts):  # tl=tailor long in bear, ts=tailor short in bear
    rows=[]; hv=[]; fl=[]; rg_=[]
    for ot,g,rg,hf,f in groups:
        dl=(rg=="bear"); ds=(rg in ("bear","side"))
        lcol=("t_l" if (tl and rg=="bear") else "p_l"); bcol=("t_b" if (ts and rg=="bear") else "p_b")
        L=g.nlargest(KL,lcol)["fwd"].mean() if (dl and len(g)>=KL) else 0.0
        S=g.nsmallest(KS,bcol)["fwd"].mean() if (ds and len(g)>=KS) else 0.0
        rows.append((L if dl else 0.0)-(S if ds else 0.0)); hv.append(hf); fl.append(f); rg_.append(rg)
    return pd.DataFrame({"pnl":rows,"hf":hv,"fold":fl,"reg":rg_})
def sh(s): return s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
base=series(1,3,False,False)
print("=== variants vs pooled, K=1/3 (gross Sh ~2.4x inflated). Δ = variant − pooled ===")
for tl,ts,name in [(True,False,"LONG-only tailor"),(False,True,"SHORT-only tailor"),(True,True,"BOTH tailor")]:
    v=series(1,3,tl,ts)
    dall=v["pnl"].mean()-base["pnl"].mean()
    h2p=base[base.hf=="h2"]["pnl"]; h2v=v[v.hf=="h2"]["pnl"]
    h1p=base[base.hf=="h1"]["pnl"]; h1v=v[v.hf=="h1"]["pnl"]
    # per-fold restricted to bear folds
    dv=v.copy(); dv["dp"]=v["pnl"]-base["pnl"]
    bf=dv[dv.fold.isin(bear_folds)].groupby("fold")["dp"].mean()
    print(f"\n  [{name}]")
    print(f"    ALL Δmean{dall:+5.1f} | H1 pooled{h1p.mean():+6.1f}/Sh{sh(h1p):+4.2f} tail{h1v.mean():+6.1f}/Sh{sh(h1v):+4.2f} | H2 pooled{h2p.mean():+6.1f}/Sh{sh(h2p):+4.2f} tail{h2v.mean():+6.1f}/Sh{sh(h2v):+4.2f}")
    print(f"    bear-fold wins {int((bf>0).sum())}/{len(bf)}  mean Δ/bear-fold {bf.mean():+.1f}")
print("\nDecision: adopt a variant only if H2 Sh up AND bear-fold wins are a clear majority.")
print("V4CAL2DONE")
