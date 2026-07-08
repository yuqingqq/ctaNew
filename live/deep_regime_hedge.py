"""VALIDATE regime-conditional long: keep stock-picked long in BEAR (real alpha), replace with BTC hedge (=0 residual)
in SIDE+BULL (broken long). Short K=3, long K=1. Compare vs baseline (stock-pick long all regimes) on residual L/S +
STABILITY (monthly %pos, longest neg streak, moSh), overall + OOS(2023-2025) vs in-sample(2025-10+)."""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; KL=1; KS=3
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
rows=[]
for ot,g in d.groupby("open_time"):
    if len(g)<KL+KS: continue
    rg=regd.get(ot,"side")
    lo=g.nlargest(KL,"pl")["fwd"].mean(); sh=g.nsmallest(KS,"base")["fwd"].mean()
    base_ls=lo-sh                                  # stock-pick long all regimes
    hedge_ls=(lo if rg=="bear" else 0.0)-sh        # BTC (0 residual) long in side/bull, stock-pick in bear
    rows.append((ot,rg,base_ls,hedge_ls))
T=pd.DataFrame(rows,columns=["ot","reg","base","hedge"]).set_index("ot").sort_index()
def stats(s,lbl):
    m=s.resample("1ME").mean().dropna(); neg=(m<0).astype(int); st=mx=0
    for v in neg: st=st+1 if v else 0; mx=max(mx,st)
    mosh=m.mean()/m.std()*np.sqrt(len(m)) if m.std()>0 else np.nan
    print(f"  {lbl:28s} L/S {s.mean():+6.1f}  moSh {mosh:+.2f}  mo%pos {100*(m>0).mean():3.0f}  longest_neg {mx}mo")
print(f"=== VALIDATE regime-conditional long (K_long=1 bear stock-pick / side+bull BTC-hedge, K_short=3) ===\n")
print("FULL 2023-2026:")
stats(T["base"],"baseline (stock-pick all)"); stats(T["hedge"],"regime-hedge long")
print("\nOOS 2023-2025 (held-out):")
oos=T[T.index<pd.Timestamp('2025-10-01',tz='UTC')]
stats(oos["base"],"baseline"); stats(oos["hedge"],"regime-hedge")
print("\nIN-SAMPLE 2025-10+:")
ins=T[T.index>=pd.Timestamp('2025-10-01',tz='UTC')]
stats(ins["base"],"baseline"); stats(ins["hedge"],"regime-hedge")
print("\nby regime (L/S mean): base vs hedge")
for rg in ["side","bear","bull"]:
    s=T[T.reg==rg]; print(f"  {rg}: base {s['base'].mean():+.1f} -> hedge {s['hedge'].mean():+.1f}")
print("DHDONE")
