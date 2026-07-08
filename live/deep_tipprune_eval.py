"""Validate the H1-decided tip-prune on H2. Pooled(full-14) vs pruned regime-conditional:
  bear-LONG -> hl_tipprune_bearlong ; bear-SHORT -> hl_tipprune_bearshort ; side-SHORT -> pooled(full).
Gate = bear:L/S, side:short-only, bull:flat. Isolate LONG-only / SHORT-only / BOTH prune. Report H2 (validation)
gated Sharpe + bear-leg tip + bear-fold breadth. Residual alpha, bps.
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
   .merge(lp("hl_tipprune_bearshort","t_b")[["symbol","open_time","t_b"]],on=["symbol","open_time"])
   .merge(lp("hl_tipprune_bearlong","t_l")[["symbol","open_time","t_l"]],on=["symbol","open_time"])
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
# bear-leg tip H1/H2
def btip():
    o={k:{"h1":[],"h2":[]} for k in ["Lp","Lt","Sp","St"]}
    for ot,g,rg,hf,f in groups:
        if rg!="bear": continue
        o["Lp"][hf].append(g.nlargest(1,"p_l")["fwd"].mean()); o["Lt"][hf].append(g.nlargest(1,"t_l")["fwd"].mean())
        if len(g)>=4: o["Sp"][hf].append(-g.nsmallest(4,"p_b")["fwd"].mean()); o["St"][hf].append(-g.nsmallest(4,"t_b")["fwd"].mean())
    m=lambda x:np.mean(x) if x else np.nan
    print("=== BEAR-leg tip: pooled vs pruned (H1 build / H2 validate) ===")
    print(f"  bear LONG  : pooled H1{m(o['Lp']['h1']):+6.0f} H2{m(o['Lp']['h2']):+6.0f} | pruned H1{m(o['Lt']['h1']):+6.0f} H2{m(o['Lt']['h2']):+6.0f}")
    print(f"  bear SHORT : pooled H1{m(o['Sp']['h1']):+6.0f} H2{m(o['Sp']['h2']):+6.0f} | pruned H1{m(o['St']['h1']):+6.0f} H2{m(o['St']['h2']):+6.0f}")
btip()
def series(KL,KS,pl,ps):
    rows=[];hv=[];fl=[]
    for ot,g,rg,hf,f in groups:
        dl=(rg=="bear"); ds=(rg in ("bear","side"))
        lcol=("t_l" if (pl and rg=="bear") else "p_l"); bcol=("t_b" if (ps and rg=="bear") else "p_b")
        L=g.nlargest(KL,lcol)["fwd"].mean() if (dl and len(g)>=KL) else 0.0
        S=g.nsmallest(KS,bcol)["fwd"].mean() if (ds and len(g)>=KS) else 0.0
        rows.append((L if dl else 0.0)-(S if ds else 0.0)); hv.append(hf); fl.append(f)
    return pd.DataFrame({"pnl":rows,"hf":hv,"fold":fl})
def sh(s): return s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
base=series(1,3,False,False)
print("\n=== gated per-cycle L/S, K=1/3. Δ = pruned − pooled. H2 is the validation half. ===")
for pl,ps,name in [(True,False,"bear-LONG prune"),(False,True,"bear-SHORT prune"),(True,True,"BOTH prune")]:
    v=series(1,3,pl,ps); dv=v.copy(); dv["dp"]=v["pnl"]-base["pnl"]
    for lab in ["h1","h2"]:
        p=base[base.hf==lab]["pnl"]; t=v[v.hf==lab]["pnl"]
        print(f"  [{name}] {lab.upper()}: pooled{p.mean():+6.1f}/Sh{sh(p):+5.2f}  pruned{t.mean():+6.1f}/Sh{sh(t):+5.2f}  Δmean{t.mean()-p.mean():+5.1f}")
    bf=dv[dv.fold.isin(bear_folds)].groupby("fold")["dp"].mean()
    print(f"          bear-fold wins {int((bf>0).sum())}/{len(bf)}  mean Δ/bear-fold {bf.mean():+.1f}\n")
print("ADOPT only if H2 Sh up AND bear-fold majority.  TIPPRUNEEVALDONE")
