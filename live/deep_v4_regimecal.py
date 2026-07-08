"""Regime-TAILORED vs POOLED v4 — honest test. Tailored uses the bear-calibrated books ONLY in bear:
  bear LONG  = hl_v4long_bearcal (dropped bear-long reversers)  vs pooled hl_v4long_oos
  bear SHORT = hl_v4base_bearcal (dropped bear-short reversers) vs pooled hl_v4base_oos
  side SHORT = pooled hl_v4base_oos (unchanged); bull = flat.
Gate = bear:L/S, side:short-only, bull:flat. Decision: adopt tailored ONLY if it beats pooled on H2 (out-of-time),
not just H1. Report bear-leg tip (H1/H2), full gated per-cycle Sharpe (H1/H2), per-fold wins. Residual alpha, bps.
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
groups=[(ot,g,g["reg"].iloc[0],("h1" if ot<mid else "h2")) for ot,g in d.groupby("open_time")]

# --- bear-leg tip: pooled vs tailored, H1/H2 ---
def bear_tip():
    o={"L_p":{"h1":[],"h2":[]},"L_t":{"h1":[],"h2":[]},"S_p":{"h1":[],"h2":[]},"S_t":{"h1":[],"h2":[]}}
    for ot,g,rg,hf in groups:
        if rg!="bear": continue
        o["L_p"][hf].append(g.nlargest(1,"p_l")["fwd"].mean()); o["L_t"][hf].append(g.nlargest(1,"t_l")["fwd"].mean())
        if len(g)>=4:
            o["S_p"][hf].append(-g.nsmallest(4,"p_b")["fwd"].mean()); o["S_t"][hf].append(-g.nsmallest(4,"t_b")["fwd"].mean())
    m=lambda x:np.mean(x) if x else np.nan
    print("=== BEAR-leg tip: pooled vs tailored (bps), H1 build / H2 validate ===")
    print(f"  bear LONG  top-1 : pooled H1{m(o['L_p']['h1']):+6.0f} H2{m(o['L_p']['h2']):+6.0f} | tailored H1{m(o['L_t']['h1']):+6.0f} H2{m(o['L_t']['h2']):+6.0f}")
    print(f"  bear SHORT bot-4 : pooled H1{m(o['S_p']['h1']):+6.0f} H2{m(o['S_p']['h2']):+6.0f} | tailored H1{m(o['S_t']['h1']):+6.0f} H2{m(o['S_t']['h2']):+6.0f}")
bear_tip()

# --- full gated per-cycle L/S: pooled vs tailored ---
def series(KL,KS,tailored):
    rows=[]; halves=[]; folds=[]
    for ot,g,rg,hf in groups:
        dl=(rg=="bear"); ds=(rg in ("bear","side"))
        lcol = ("t_l" if (tailored and rg=="bear") else "p_l")
        bcol = ("t_b" if (tailored and rg=="bear") else "p_b")
        L=g.nlargest(KL,lcol)["fwd"].mean() if (dl and len(g)>=KL) else 0.0
        S=g.nsmallest(KS,bcol)["fwd"].mean() if (ds and len(g)>=KS) else 0.0
        rows.append((L if dl else 0.0)-(S if ds else 0.0)); halves.append(hf); folds.append(g["fold"].iloc[0])
    return pd.Series(rows),pd.Series(halves),pd.Series(folds)
def sh(s): return s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
print("\n=== FULL GATED per-cycle L/S: pooled vs tailored (gross Sh ~2.4x inflated) ===")
for KL,KS in [(1,2),(1,3)]:
    sp,hp,fp=series(KL,KS,False); st,ht,ft=series(KL,KS,True)
    print(f"\n  K={KL}/{KS}")
    for lab,sub in [("ALL",slice(None)),("H1",hp=="h1"),("H2",hp=="h2")]:
        mask = (hp==lab.lower()) if lab in ("H1","H2") else pd.Series(True,index=sp.index)
        pp=sp[mask]; tt_=st[mask]
        print(f"    {lab:>3s}: pooled mean{pp.mean():+6.1f} Sh{sh(pp):+5.2f} | tailored mean{tt_.mean():+6.1f} Sh{sh(tt_):+5.2f} | Δmean{tt_.mean()-pp.mean():+5.1f}")
    # per-fold wins (tailored beats pooled)
    df=pd.DataFrame({"p":sp,"t":st,"f":fp}); w=df.groupby("f").apply(lambda x:x["t"].mean()-x["p"].mean())
    print(f"    per-fold: tailored beats pooled in {int((w>0).sum())}/{w.notna().sum()} folds; mean Δ/fold {w.mean():+.1f}")
print("\nADOPT tailored only if it beats pooled on H2 AND majority folds.")
print("V4CALDONE")
