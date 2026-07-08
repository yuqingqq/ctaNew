"""Current v4 performance — GATES OFF (raw two-book model, always long K_L + short K_S every cycle), net of fees.
No regime gate, no logic. Books: base=hl_v4base_oos (short ranker), long=hl_v4long_oos. Two windows:
OOS 2023-01..2025-09 (honest) and in-sample 2025-10+ (recent). Net of turnover cost (FEE 4.5/leg). Metrics:
daily-resampled Sharpe (x sqrt365, the honest headline), per-cycle Sharpe (overlap-inflated ref), total net PnL,
maxDD, %pos days, and per-regime net mean. Residual alpha, bps.
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
def load(base,long,tag):
    d=(lp(base,"b").merge(lp(long,"l"),on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
    return d
oos=load("hl_v4base_oos","hl_v4long_oos","oos")
ins=load("hl_tgt_res_base","hl_tgt_res_long","ins")  # in-sample 2025-10+ books
allg=pd.DatetimeIndex(sorted(set(oos["open_time"].unique())|set(ins["open_time"].unique())))
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
def perf(d,KL,KS,cost=4.5):
    d=d.copy(); d["reg"]=d["open_time"].map(regd)
    prevL=set(); prevS=set(); rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<KL+KS: continue
        L=g.nlargest(KL,"l"); S=g.nsmallest(KS,"b")
        gross=L["fwd"].mean()-S["fwd"].mean()
        Ls,Ss=set(L["symbol"]),set(S["symbol"])
        cst=(len(Ls^prevL)/KL+len(Ss^prevS)/KS)*cost; prevL,prevS=Ls,Ss
        rows.append((ot,gross-cst,g["reg"].iloc[0]))
    s=pd.DataFrame(rows,columns=["t","net","reg"]).set_index("t")
    daily=s["net"].resample("1D").mean().dropna()
    dsh=daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else np.nan
    csh=s["net"].mean()/s["net"].std()*np.sqrt(len(s)) if s["net"].std()>0 else np.nan
    eq=daily.cumsum(); dd=(eq-eq.cummax()).min()
    byreg={rg:s[s.reg==rg]["net"].mean() for rg in ["side","bear","bull"]}
    return dsh,csh,s["net"].sum(),dd,(daily>0).mean()*100,byreg,len(s)
print("=== v4 GATES-OFF performance (raw two-book, always L/S, net FEE 4.5/leg) ===\n")
for name,d in [("OOS 2023-01..2025-09 (honest)",oos),("IN-SAMPLE 2025-10+ (recent)",ins)]:
    print(f"--- {name} ---")
    print(f"  {'K':>5s} | {'dailySh':>7s} {'cycSh*':>6s} | {'netPnL':>8s} {'maxDD':>7s} {'%posD':>5s} | side/bear/bull net-bps/cyc")
    for KL,KS in [(1,2),(1,3)]:
        dsh,csh,tot,dd,pos,br,n=perf(d,KL,KS)
        print(f"  {KL}/{KS:<3d} | {dsh:+7.2f} {csh:+6.2f} | {tot:+8.0f} {dd:+7.0f} {pos:4.0f}% | {br['side']:+.0f} / {br['bear']:+.0f} / {br['bull']:+.0f}   (n={n})")
    print()
print("* cycSh = per-cycle Sharpe (~2.4x overlap-inflated); dailySh = daily-resampled x sqrt365 = honest headline.")
print("V4PERFDONE")
