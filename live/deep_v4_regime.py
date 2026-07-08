"""v4 (residual target) vs v3 (return target) — per-regime long/short leg edge on the recent/in-sample window.
v4: base=hl_tgt_res_base (short ranker), long=hl_tgt_res_long. v3: base=hl_lean175, long=hl_residrev_lean.
Per regime x K: long-leg realized alpha & short-leg PnL, v3 vs v4. Where is v4's edge, is the long leg still broken?"""
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
d=(lp("hl_lean175","v3b").merge(lp("hl_residrev_lean","v3l"),on=["symbol","open_time"])
   .merge(lp("hl_tgt_res_base","v4b"),on=["symbol","open_time"]).merge(lp("hl_tgt_res_long","v4l"),on=["symbol","open_time"])
   .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")]
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
    parts=[q for q in ex.map(fm,pd.period_range("2025-03",grid.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
regd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
d["reg"]=d["open_time"].map(regd)
def leg(sub,col,K,side):
    v=[sub[sub.open_time==ot].nlargest(K,col)["fwd"].mean() if side=="long" else sub[sub.open_time==ot].nsmallest(K,col)["fwd"].mean() for ot in sub.open_time.unique() if (sub.open_time==ot).sum()>=2*K]
    return np.nanmean(v)
print("=== v4 (residual) vs v3 (return) — per-regime leg realized alpha, recent/in-sample 2025-10+ (bps) ===\n")
for rg in ["side","bear","bull"]:
    sub=d[d.reg==rg]; print(f"--- {rg.upper()} (n_cyc {sub.open_time.nunique()}) ---")
    print(f"  {'K':>2s} | {'LONG v3':>8s} {'LONG v4':>8s} {'Δ':>6s} | {'SHORT v3':>9s} {'SHORT v4':>9s} {'Δ':>6s}")
    for K in [1,2,3]:
        l3=leg(sub,"v3l",K,"long"); l4=leg(sub,"v4l",K,"long")
        s3=-leg(sub,"v3b",K,"short"); s4=-leg(sub,"v4b",K,"short")
        print(f"  {K:>2d} | {l3:+8.1f} {l4:+8.1f} {l4-l3:+6.1f} | {s3:+9.1f} {s4:+9.1f} {s4-s3:+6.1f}")
print("\n(LONG=realized of top-K longs, want HIGH. SHORT=PnL=-realized of bottom-K, want HIGH. Δ=v4-v3)")
print("V4RDONE")
