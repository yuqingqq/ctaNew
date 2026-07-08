"""v4 (residual target) vs v3 (return target) — the HONEST OOS comparison (2023-01 .. 2025-09, genuinely held-out).
v3: base=hl_lean175_oos, long=hl_residrev_oos.  v4: base=hl_v4base_oos, long=hl_v4long_oos.
Three lenses: (A) per-regime leg edge, (B) tip accuracy at traded K {mean, per-cycle Sharpe, hit-rate},
(C) monthly stability -> WHERE is v4 weak. All in beta-adjusted residual alpha (alpha_vs_btc_realized), bps.
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
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
d=(lp("hl_lean175_oos","v3b").merge(lp("hl_residrev_oos","v3l"),on=["symbol","open_time"])
   .merge(lp("hl_v4base_oos","v4b"),on=["symbol","open_time"]).merge(lp("hl_v4long_oos","v4l"),on=["symbol","open_time"])
   .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
grid=pd.DatetimeIndex(sorted(d["open_time"].unique()))
print(f"OOS window {grid.min().date()} .. {grid.max().date()}  cycles={len(grid)}  rows={len(d)}\n",flush=True)
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
d["reg"]=d["open_time"].map(regd); d["mo"]=d["open_time"].dt.to_period("M")

# ---- (A) per-regime leg edge ----
def legmean(sub,col,K,side):
    v=[sub[sub.open_time==ot].nlargest(K,col)["fwd"].mean() if side=="long" else sub[sub.open_time==ot].nsmallest(K,col)["fwd"].mean() for ot in sub.open_time.unique() if (sub.open_time==ot).sum()>=2*K]
    return np.nanmean(v)
print("=== (A) per-regime leg realized alpha, OOS (bps).  LONG=top-K want HIGH; SHORT=-bottom-K want HIGH ===\n")
for rg in ["side","bear","bull"]:
    sub=d[d.reg==rg]; print(f"--- {rg.upper()} (n_cyc {sub.open_time.nunique()}) ---")
    print(f"  {'K':>2s} | {'LONG v3':>8s} {'LONG v4':>8s} {'Δ':>6s} | {'SHORT v3':>9s} {'SHORT v4':>9s} {'Δ':>6s}")
    for K in [1,2,3]:
        l3=legmean(sub,"v3l",K,"long"); l4=legmean(sub,"v4l",K,"long")
        s3=-legmean(sub,"v3b",K,"short"); s4=-legmean(sub,"v4b",K,"short")
        print(f"  {K:>2d} | {l3:+8.1f} {l4:+8.1f} {l4-l3:+6.1f} | {s3:+9.1f} {s4:+9.1f} {s4-s3:+6.1f}")

# ---- (B) tip accuracy at traded K: per-cycle L/S {mean, Sharpe, hit} ----
def tip_ls(KL,KS,lcol,bcol):
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<KL+KS: continue
        L=g.nlargest(KL,lcol)["fwd"].mean(); S=g.nsmallest(KS,bcol)["fwd"].mean()
        rows.append(L-S)
    s=pd.Series(rows); return s.mean(), s.mean()/s.std()*np.sqrt(len(s)), (s>0).mean()*100, s
print("\n=== (B) TRADED-TIP L/S at each K (long by long-book, short by base-book), OOS ===")
print(f"  {'KL/KS':>6s} | {'v3 mean':>8s} {'v3 Sh':>6s} {'v3 hit%':>7s} | {'v4 mean':>8s} {'v4 Sh':>6s} {'v4 hit%':>7s}")
series={}
for KL,KS in [(1,2),(1,3),(2,2),(1,1)]:
    m3,sh3,h3,_=tip_ls(KL,KS,"v3l","v3b"); m4,sh4,h4,s4s=tip_ls(KL,KS,"v4l","v4b")
    series[(KL,KS)]=s4s
    print(f"  {KL}/{KS:<4d} | {m3:+8.1f} {sh3:+6.2f} {h3:6.0f}% | {m4:+8.1f} {sh4:+6.2f} {h4:6.0f}%")

# ---- (C) monthly stability of the production tip (v4 vs v3), locate weakness ----
def monthly(KL,KS,lcol,bcol):
    out={}
    for mo,gm in d.groupby("mo"):
        vals=[gm[gm.open_time==ot].nlargest(KL,lcol)["fwd"].mean()-gm[gm.open_time==ot].nsmallest(KS,bcol)["fwd"].mean()
              for ot in gm.open_time.unique() if (gm.open_time==ot).sum()>=KL+KS]
        out[mo]=np.nanmean(vals)
    return pd.Series(out)
m3=monthly(1,2,"v3l","v3b"); m4=monthly(1,2,"v4l","v4b")
def streak(s):
    mx=c=0
    for v in s:
        c=c+1 if v<0 else 0; mx=max(mx,c)
    return mx
print("\n=== (C) production-tip (KL=1/KS=2) MONTHLY stability, OOS ===")
print(f"  v3: %pos_mo {100*(m3>0).mean():.0f}%  mean_mo {m3.mean():+.1f}  longest_neg_streak {streak(m3)}mo")
print(f"  v4: %pos_mo {100*(m4>0).mean():.0f}%  mean_mo {m4.mean():+.1f}  longest_neg_streak {streak(m4)}mo")
worst=m4.nsmallest(5)
print(f"  v4 worst 5 months: "+", ".join(f'{k}:{v:+.0f}' for k,v in worst.items()))
print("V4OOSDONE")
