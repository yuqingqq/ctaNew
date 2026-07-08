"""Stability of the ALPHA-RESIDUAL edge (the model's SKILL, not the raw L/S PnL). Monthly rank-IC of pred vs
forward 24h residual alpha (alpha_vs_btc_realized), 2023-2026, overall + per regime. Plus the traded tip L/S monthly.
Stability metrics: mean, std, %pos months, IC-'Sharpe' (mean/std), longest neg streak, autocorr(IC), regime split.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def cat(paths,col):
    ps=[]
    for p in paths:
        d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); ps.append(d)
    return pd.concat(ps).drop_duplicates(["symbol","open_time"]).rename(columns={"pred":col})
base=cat(["hl_lean175_oos","hl_lean175"],"base"); lng=cat(["hl_residrev_oos","hl_residrev_lean"],"pl")
d=base.merge(lng,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]).dropna(subset=["fwd"])
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
regd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
# per-cycle IC (residual skill) + tip L/S, then monthly
d["reg"]=d["open_time"].map(regd)
cyc=[]
for ot,g in d.groupby("open_time"):
    if len(g)<max(2*K,20): continue
    ic=spearmanr(g["pl"],g["fwd"]).correlation
    ls=g.nlargest(K,"pl")["fwd"].mean()-g.nsmallest(K,"base")["fwd"].mean()
    cyc.append((ot,g["reg"].iloc[0],ic,ls))
C=pd.DataFrame(cyc,columns=["ot","reg","ic","ls"]).set_index("ot").sort_index()
M=C["ic"].resample("1ME").mean(); Mls=C["ls"].resample("1ME").mean()
def stab(s,lbl):
    s=s.dropna(); pos=100*(s>0).mean()
    icsh=s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
    # longest consecutive negative-month streak
    neg=(s<0).astype(int); streak=mx=0
    for v in neg: streak=streak+1 if v else 0; mx=max(mx,streak)
    ac=s.autocorr(1)
    print(f"  {lbl:18s} mean{s.mean():+.4f} std{s.std():.4f} %pos{pos:3.0f} 'IC-Sh'{icsh:+.2f} longest_neg_streak {mx}mo autocorr{ac:+.2f}")
print("=== ALPHA-RESIDUAL SKILL (monthly rank-IC of pred vs fwd residual alpha), 2023-2026 ===\n")
print("monthly IC series:")
for m,v in M.items(): print(f"  {str(m)[:7]}: IC {v:+.4f}  tipL/S {Mls.get(m,float('nan')):+.0f}")
print("\nSTABILITY (monthly IC):")
stab(M,"ALL")
for rg in ["side","bear","bull"]:
    stab(C[C.reg==rg]["ic"].resample("1ME").mean(),rg)
print("\nTIP-L/S monthly stability:")
stab(Mls,"tipL/S(bps)")
print("DSDONE")
