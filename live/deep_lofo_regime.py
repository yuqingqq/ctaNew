"""Per-regime LOFO pruning analysis for the base/short book. For each dropped feature, marginal impact on the
SHORT tip (bottom-4) in each traded short-regime (side, bear) = tip(drop_f) - tip(FULL14). Decide on H1, validate
on H2. A feature is a robust per-regime prune if dropping it helps on BOTH H1 and H2 (same sign, positive).
"""
import io, zipfile, glob, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; D=f"{R}/live/state/convexity/hl_lofo_base"
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(tag):
    d=pd.read_parquet(f"{D}/{tag}/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d.rename(columns={"pred":tag})
tags=[os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{D}/*/v0full_hl60.parquet")]
drops=[t for t in tags if t.startswith("drop_")]
base=lp("FULL14").merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
for t in drops: base=base.merge(lp(t),on=["symbol","open_time"],how="left")
base=base.dropna(subset=["fwd"])
grid=pd.DatetimeIndex(sorted(base["open_time"].unique()))
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
base["reg"]=base["open_time"].map(regd); mid=grid[len(grid)//2]
base["half"]=np.where(base["open_time"]<mid,"h1","h2")
def short_tip(col,reg,half):
    sub=base[(base.reg==reg)&(base.half==half)]; v=[]
    for ot,g in sub.groupby("open_time"):
        if g[col].notna().sum()>=4: v.append(-g.nsmallest(4,col)["fwd"].mean())
    return np.mean(v) if v else np.nan
print("=== Per-regime LOFO on BASE/short book: marginal Δ short-tip from DROPPING each feature (bps). ===")
print("    +Δ = dropping the feature HELPS that regime's short. Decide on H1, validate on H2.\n")
for reg in ["side","bear"]:
    full_h1=short_tip("FULL14",reg,"h1"); full_h2=short_tip("FULL14",reg,"h2")
    print(f"--- {reg.upper()} short (FULL14: H1 {full_h1:+.0f}, H2 {full_h2:+.0f}) ---")
    rows=[]
    for t in drops:
        f=t[5:]; dh1=short_tip(t,reg,"h1")-full_h1; dh2=short_tip(t,reg,"h2")-full_h2
        rows.append((f,dh1,dh2))
    rows.sort(key=lambda x:-x[1])  # by H1 marginal (the decision axis)
    print(f"  {'drop feature':<26s} {'ΔH1(decide)':>11s} {'ΔH2(valid)':>11s}  robust-prune?")
    for f,dh1,dh2 in rows:
        rob="PRUNE✓" if (dh1>0 and dh2>0) else ("h1-only" if dh1>0 else "")
        print(f"  {f:<26s} {dh1:+11.1f} {dh2:+11.1f}  {rob}")
    print()
print("Robust per-regime prune = drop helps on BOTH H1 and H2. (Next: combined-prune retrain + gated eval.)")
print("LOFOREGDONE")
