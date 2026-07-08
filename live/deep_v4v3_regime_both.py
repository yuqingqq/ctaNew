"""v4 (residual target) vs v3 (return target) — long/short leg realized-alpha edge BY REGIME, on BOTH windows
(RECENT 2025-10+ primary, OOS 2023-25 reference). LONG = mean fwd of top-K by long book (want HIGH). SHORT = -mean
fwd of bottom-K by base book (PnL, want HIGH). Residual alpha (alpha_vs_btc_realized), bps.
Books: RECENT v3=hl_tgt_ret_*, v4=hl_tgt_res_*; OOS v3=hl_lean175_oos/hl_residrev_oos, v4=hl_v4base_oos/hl_v4long_oos.
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
def build(v3b,v3l,v4b,v4l):
    d=(lp(v3b,"v3b").merge(lp(v3l,"v3l"),on=["symbol","open_time"])
       .merge(lp(v4b,"v4b"),on=["symbol","open_time"]).merge(lp(v4l,"v4l"),on=["symbol","open_time"])
       .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
    return d
WIN={"RECENT 2025-10+":("hl_tgt_ret_base","hl_tgt_ret_long","hl_tgt_res_base","hl_tgt_res_long"),
     "OOS 2023-25":("hl_lean175_oos","hl_residrev_oos","hl_v4base_oos","hl_v4long_oos")}
data={k:build(*v) for k,v in WIN.items()}
allg=pd.DatetimeIndex(sorted(set().union(*[set(d["open_time"].unique()) for d in data.values()])))
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
    parts=[q for q in ex.map(fm,pd.period_range("2022-06",allg.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(allg)))).ffill(); r30=(btc/btc.shift(180)-1)
regd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
def leg(sub,col,K,side):
    v=[sub[sub.open_time==ot].nlargest(K,col)["fwd"].mean() if side=="long" else sub[sub.open_time==ot].nsmallest(K,col)["fwd"].mean() for ot in sub.open_time.unique() if (sub.open_time==ot).sum()>=2*K]
    return np.nanmean(v)
for win,d in data.items():
    d=d.copy(); d["reg"]=d["open_time"].map(regd)
    print(f"\n{'='*78}\n=== {win}  (v4 residual vs v3 return) — per-regime leg realized alpha, bps ===")
    for rg in ["side","bear","bull"]:
        sub=d[d.reg==rg]
        print(f"--- {rg.upper()} (n_cyc {sub.open_time.nunique()}) ---")
        print(f"  {'K':>2s} | {'LONG v3':>8s} {'LONG v4':>8s} {'Δ':>6s} | {'SHORT v3':>9s} {'SHORT v4':>9s} {'Δ':>6s}")
        for K in [1,2,3]:
            l3=leg(sub,"v3l",K,"long"); l4=leg(sub,"v4l",K,"long")
            s3=-leg(sub,"v3b",K,"short"); s4=-leg(sub,"v4b",K,"short")
            print(f"  {K:>2d} | {l3:+8.1f} {l4:+8.1f} {l4-l3:+6.1f} | {s3:+9.1f} {s4:+9.1f} {s4-s3:+6.1f}")
print("\n(LONG=top-K realized want+; SHORT=-bottom-K PnL want+. Δ=v4-v3.)")
print("V4V3REGDONE")
