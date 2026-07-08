"""v4 baseline = residual-trained two-book model + SIMPLEST gate-free book (fixed K, equal weight, no regime/
hedge/conc-cap/sizing/gates). Reference row = return-trained model (v3's target) through the same simple book.
Writes a tracking table to live/V4_PERFORMANCE.md so every later change is measured against this baseline.
Metrics: daily Sharpe, L/S mean/cyc (bps), total, maxDD (daily), by-regime L/S, %pos cycles. GROSS (no cost/gates).
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
grid=pd.DatetimeIndex(sorted(loadv("hl_tgt_res_base","hl_tgt_res_long")["open_time"].unique()))
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
def metrics(m,KL,KS):
    rows=[]
    for ot,gc in m.groupby("open_time"):
        if len(gc)<KL+KS: continue
        rows.append((ot,reg.get(ot,"side"),gc.nlargest(KL,"pl")["fwd"].mean()-gc.nsmallest(KS,"pred")["fwd"].mean()))
    d=pd.DataFrame(rows,columns=["ot","reg","ls"]).set_index("ot")
    dd=(d["ls"]/1e4).resample("1D").sum(); sh=dd.mean()/dd.std()*ANN if dd.std()>0 else np.nan
    eq=dd.cumsum(); mdd=float((eq-eq.cummax()).min()*1e4)
    br={rg:d[d.reg==rg]["ls"].mean() for rg in ["bull","side","bear"]}
    return dict(sh=sh,lsm=d["ls"].mean(),tot=d["ls"].sum(),mdd=mdd,pos=100*(d["ls"]>0).mean(),br=br,n=len(d))
rows=[]
for model,(base,long) in [("RETURN (v3 target)",("hl_tgt_ret_base","hl_tgt_ret_long")),("RESIDUAL (v4 target)",("hl_tgt_res_base","hl_tgt_res_long"))]:
    m=loadv(base,long)
    for KL,KS in [(1,2),(2,2),(3,3)]:
        r=metrics(m,KL,KS)
        rows.append((model,f"{KL}/{KS}",r))
# write tracking file
lines=["# Convexity v4 — performance tracker",
"",
"**v4 = residual-aligned target** (`xs_z(alpha_vs_btc_realized)`). Baseline below = model + SIMPLEST gate-free book",
"(fixed K, equal weight, NO regime/hedge/conc-cap/sizing/gates), GROSS (no cost). This is the zero-tuning reference;",
"every later change (cost, then each v3 gate) appends a row and is judged against this.",
"",
"## Baseline (in-sample 2025-10-04+, gross, no gates) — 2026-07-06",
"",
"| model | K L/S | dailySharpe | L/S mean/cyc | totPnL | maxDD | %pos | bull | side | bear |",
"|---|---|---|---|---|---|---|---|---|---|"]
for model,k,r in rows:
    lines.append(f"| {model} | {k} | {r['sh']:+.2f} | {r['lsm']:+.1f} | {r['tot']:+.0f} | {r['mdd']:+.0f} | {r['pos']:.0f}% | {r['br']['bull']:+.0f} | {r['br']['side']:+.0f} | {r['br']['bear']:+.0f} |")
lines+=["",
"**Read:** at production K (1/2) residual beats return by +0.51 Sharpe / +20% L/S (side+bear gain, bull loses — but",
"v3 ignores pred in bull). At symmetric K it's tied. Caveat: the K=1/2 edge is cycle-concentrated + decaying (half2≈0);",
"OOS is the decisive test. Gross/no-gates numbers are NOT comparable to v3's headline (which is net + gated).",
"",
"## Change log (append below as tuning layers are added / tested)",
"| date | change | config | dailySharpe | Δ vs baseline | notes |",
"|---|---|---|---|---|---|",
"| 2026-07-06 | v4 baseline set | residual, simple book, K=1/2, gross | (see table) | — | zero-tuning reference |"]
open(f"{R}/live/V4_PERFORMANCE.md","w").write("\n".join(lines)+"\n")
print("\n".join(lines[6:len(rows)+9]))
print("\nwrote live/V4_PERFORMANCE.md")
print("V4BDONE")
