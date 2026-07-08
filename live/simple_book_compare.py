"""Simple GATE-FREE book: fixed K, equal weight, no regime/hedge/conc-cap/sizing. Run the SAME simple mechanism on
both target-variants (return-trained vs residual-trained preds) — the only difference is the training label.
Per cycle: L = top-K_L by long pred, S = bottom-K_S by base pred; L/S = mean(L fwd)-mean(S fwd) 24h residual alpha.
Report gross L/S mean/cycle, total, daily Sharpe, per regime. Fair isolation of the training-target effect.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def loadv(base,long):
    b=pd.read_parquet(f"{R}/live/state/convexity/{base}/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
    l=pd.read_parquet(f"{R}/live/state/convexity/{long}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"pl"})
    for x in (b,l): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
    m=b.merge(l,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
    return m[m.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
grid=pd.DatetimeIndex(sorted(loadv("hl_tgt_ret_base","hl_tgt_ret_long")["open_time"].unique()))
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
    parts=[p for p in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
def book(m,KL,KS):
    rows=[]
    for ot,gc in m.groupby("open_time"):
        if len(gc)<KL+KS: continue
        L=gc.nlargest(KL,"pl")["fwd"].mean(); S=gc.nsmallest(KS,"pred")["fwd"].mean()
        rows.append((ot,reg.get(ot,"side"),L-S))
    d=pd.DataFrame(rows,columns=["ot","reg","ls"]).set_index("ot")
    dd=(d["ls"]/1e4).resample("1D").sum(); sh=dd.mean()/dd.std()*ANN if dd.std()>0 else np.nan
    return d,sh
def report(lbl,m,KL,KS):
    d,sh=book(m,KL,KS)
    byreg={rg:d[d.reg==rg]["ls"].mean() for rg in ["bull","side","bear"]}
    print(f"  {lbl:26s} dailySharpe {sh:+.2f}  L/S mean/cyc {d['ls'].mean():+6.1f}  tot {d['ls'].sum():+8.0f} | "+" ".join(f"{k}:{v:+.0f}" for k,v in byreg.items()))
for KL,KS in [(1,2),(2,2),(3,3)]:
    print(f"\n=== SIMPLE gate-free book, K_long={KL} K_short={KS}, gross 24h L/S residual alpha ===")
    report("RETURN-trained", loadv("hl_tgt_ret_base","hl_tgt_ret_long"), KL,KS)
    report("RESIDUAL-trained", loadv("hl_tgt_res_base","hl_tgt_res_long"), KL,KS)
print("SBDONE")

# --- robustness of the K_long=1/K_short=2 residual advantage (paired per-cycle) ---
print("\n=== robustness: RESIDUAL - RETURN paired L/S per cycle, K_long=1 K_short=2 ===")
mr=loadv("hl_tgt_ret_base","hl_tgt_ret_long"); ms=loadv("hl_tgt_res_base","hl_tgt_res_long")
def ls_cyc(m):
    out={}
    for ot,gc in m.groupby("open_time"):
        if len(gc)<3: continue
        out[ot]=gc.nlargest(1,"pl")["fwd"].mean()-gc.nsmallest(2,"pred")["fwd"].mean()
    return pd.Series(out)
a=ls_cyc(mr); b=ls_cyc(ms); idx=a.index.intersection(b.index); diff=(b[idx]-a[idx]).sort_index()
h1,h2=np.array_split(diff.values,2)
mo=pd.Series(diff.values,index=pd.DatetimeIndex(diff.index)).resample("1ME").mean()
print(f"  n={len(diff)} mean{diff.mean():+.1f} median{diff.median():+.1f} %pos{100*(diff>0).mean():.0f} top3share{diff.nlargest(3).sum()/diff.sum()*100:.0f}%")
print(f"  half1 mean{h1.mean():+.1f}  half2 mean{h2.mean():+.1f}  | monthly: "+" ".join(f"{str(m)[:7]}:{v:+.0f}" for m,v in mo.items()))
print(f"  months residual>return: {int((mo>0).sum())}/{len(mo)}")
print("RBDONE")
