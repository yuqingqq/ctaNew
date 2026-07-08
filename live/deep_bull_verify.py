"""Independent verification of the v4 BULL alpha edge. Per-leg (long/short) per-K realized alpha WITH significance
(per-cycle Sharpe + n), both windows (RECENT 2025-10+, OOS 2023-25). Plus a DEPTH split (mild bull 0.10-0.20 vs deep
bull >0.20) — the BULL_DEEP_THR logic assumes edge exists only in mild bull. v4 vs v3. Residual alpha, bps.
Books: RECENT v4=hl_tgt_res_*, v3=hl_tgt_ret_*; OOS v4=hl_v4base_oos/hl_v4long_oos, v3=hl_lean175_oos/hl_residrev_oos.
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
    return (lp(v3b,"v3b").merge(lp(v3l,"v3l"),on=["symbol","open_time"]).merge(lp(v4b,"v4b"),on=["symbol","open_time"])
            .merge(lp(v4l,"v4l"),on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
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
regs={t:v for t,v in r30.items()}
def leg_stats(sub,col,K,side):
    vals=[sub[sub.open_time==ot].nlargest(K,col)["fwd"].mean() if side=="long" else -sub[sub.open_time==ot].nsmallest(K,col)["fwd"].mean()
          for ot in sub.open_time.unique() if (sub.open_time==ot).sum()>=2*K]
    s=pd.Series(vals); return (s.mean(), s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan, len(s), (s>0).mean()*100)
for win,d in data.items():
    d=d.copy(); d["r30"]=d["open_time"].map(regs); d=d[d.r30>0.10]  # BULL only
    print(f"\n{'='*80}\n=== {win} — v4 BULL edge verification (n_bull_cyc {d.open_time.nunique()}) ===")
    for lab,sub in [("ALL BULL",d),("mild 0.10-0.20",d[(d.r30>0.10)&(d.r30<=0.20)]),("deep >0.20",d[d.r30>0.20])]:
        nc=sub.open_time.nunique()
        print(f"\n  [{lab}]  n_cyc={nc}")
        if nc<10: print("    (too few cycles)"); continue
        print(f"    {'leg/K':>8s} | {'v4 mean':>7s} {'v4 Sh':>6s} {'v4 %+':>5s} {'n':>4s} | {'v3 mean':>7s} {'v3 Sh':>6s}")
        for side,lc,bc in [("LONG","v4l","v3l"),("SHORT","v4b","v3b")]:
            for K in [1,2,3]:
                m4,sh4,n4,p4=leg_stats(sub,lc if side=="LONG" else "v4b",K,side.lower())
                m3,sh3,_,_=leg_stats(sub,bc if side=="LONG" else "v3b",K,side.lower())
                print(f"    {side[:1]}{K:>6d} | {m4:+7.1f} {sh4:+6.2f} {p4:4.0f}% {n4:>4d} | {m3:+7.1f} {sh3:+6.2f}")
print("\n(LONG=top-K realized; SHORT=-bottom-K PnL. Want +. Sh=per-cycle ~2.4x overlap-inflated.)")
print("BULLVERIFYDONE")
