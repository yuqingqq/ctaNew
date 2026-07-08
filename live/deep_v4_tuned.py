"""Fair v4 vs v3 with a BASIC regime gate applied IDENTICALLY to both (no per-model tuning).
Motivation: raw untuned tip penalizes v4 on the bull-long leg, which any regime-gated strategy suppresses.
Gate rule (basic, same for both models), driven only by the leg table (long works only in bear; short works
in side+bear; bull both legs negative):
  BEAR -> long top-KL (long book) + short bottom-KS (base book)
  SIDE -> short-only bottom-KS
  BULL -> flat (sit out)
Report per-cycle net L/S {mean, Sharpe, %pos} over ALL cycles (flat cycle = 0), v3 vs v4, at a few (KL,KS).
Also G0 = raw no-gate for reference. All in residual alpha (bps).
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
d["reg"]=d["open_time"].map(regd)
groups={ot:g for ot,g in d.groupby("open_time")}
def series(lcol,bcol,KL,KS,gated):
    out=[]
    for ot,g in groups.items():
        reg=g["reg"].iloc[0]
        if gated:
            do_long = (reg=="bear"); do_short = (reg in ("bear","side"))
        else:
            do_long=True; do_short=True
        if len(g) < (KL if do_long else 0)+(KS if do_short else 0) or (not do_long and not do_short):
            out.append(0.0); continue
        L = g.nlargest(KL,lcol)["fwd"].mean() if do_long else 0.0
        S = g.nsmallest(KS,bcol)["fwd"].mean() if do_short else 0.0
        # long PnL = +L ; short PnL = -S ; if a leg is off it contributes 0
        pnl = (L if do_long else 0.0) - (S if do_short else 0.0)
        out.append(pnl)
    return pd.Series(out)
def stat(s):
    a=s[s!=0] if (s!=0).any() else s
    return s.mean(), (s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan), (s>0).mean()*100, (s!=0).mean()*100
print("=== v3 vs v4 under IDENTICAL basic regime gate (bear=L/S, side=short-only, bull=flat), OOS ===")
print("    per-cycle net L/S over ALL cycles (flat=0). G0=raw no-gate.\n")
for KL,KS in [(1,2),(1,3),(2,2)]:
    print(f"  KL/KS = {KL}/{KS}")
    for gated,lab in [(False,"G0 raw   "),(True,"G1 gated ")]:
        m3,sh3,h3,act3=stat(series("v3l","v3b",KL,KS,gated))
        m4,sh4,h4,act4=stat(series("v4l","v4b",KL,KS,gated))
        print(f"    {lab} | v3 mean{m3:+6.1f} Sh{sh3:+5.2f} pos{h3:4.0f}% act{act3:3.0f}% | v4 mean{m4:+6.1f} Sh{sh4:+5.2f} pos{h4:4.0f}% act{act4:3.0f}%")
    print()
print("V4TUNEDDONE")
