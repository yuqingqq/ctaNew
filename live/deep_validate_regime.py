"""VALIDATE the regime-confinement issue is real in the DATA, not a cost/gate/look-ahead artifact.
Strip ALL cost + gates. Per calendar year, measure the RAW GROSS alpha the strategy farms + its mechanism:
 (1) gross tip L/S (K=1/2, top-1 long / bottom-2 short realized fwd residual) — mean + per-cycle Sharpe.
 (2) xs DISPERSION = median per-cycle cross-sectional std of fwd residual (the farmable spread; git log: '2024
     collapsed idiosyncratic dispersion').
 (3) mean-reversion IC = per-cycle rank-IC of resid_rev_2 (-trailing residual) vs fwd residual (git log: '2024
     mean-rev sign flip'). +IC = losers bounce (MR regime); -IC = momentum regime.
 (4) per-regime gross tip.
If gross alpha flips/collapses in 2024 => the regime-break is REAL in the data (not our engine). PIT: fwd uses only
future (target), preds are WF, resid_rev/regime strictly trailing. Books: OOS 2023..2025-09 + in-sample 2025-10+.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["rr2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)   # PIT trailing
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
oos=lp("hl_v4base_oos","b").merge(lp("hl_v4long_oos","l"),on=["symbol","open_time"])
ins=lp("hl_tgt_res_base","b").merge(lp("hl_tgt_res_long","l"),on=["symbol","open_time"])
d=pd.concat([oos,ins],ignore_index=True).drop_duplicates(["symbol","open_time"])
d=d.merge(pan[["symbol","open_time","fwd","rr2"]],on=["symbol","open_time"]).dropna(subset=["fwd"])
d["yr"]=d["open_time"].dt.year; d["ym"]=d["open_time"].dt.to_period("M")
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
def tip(sub,KL=1,KS=2):
    v=[g.nlargest(KL,"l")["fwd"].mean()-g.nsmallest(KS,"b")["fwd"].mean() for _,g in sub.groupby("open_time") if len(g)>=KL+KS]
    s=pd.Series(v); return (s.mean(), s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan, len(s))
def disp(sub): return sub.groupby("open_time")["fwd"].std().median()
def mric(sub): return sub.groupby("open_time").apply(lambda g:g["rr2"].rank().corr(g["fwd"].rank())).mean()
print("=== VALIDATION: raw GROSS alpha per year (NO cost, NO gates) — is the regime-break real in the data? ===\n")
print(f"  {'year':>6s} {'ncyc':>5s} | {'grossTip L/S':>12s} {'cycSh':>6s} | {'xs disp':>7s} | {'MR-IC':>6s} | side/bear/bull grossTip")
for yr in [2023,2024,2025]:
    for half in ([None] if yr!=2025 else ["H1_Jan-Sep","H2_Oct+"]):
        if yr==2025 and half=="H1_Jan-Sep": sub=d[(d.yr==2025)&(d.open_time<pd.Timestamp('2025-10-01',tz='UTC'))]
        elif yr==2025: sub=d[(d.yr==2025)&(d.open_time>=pd.Timestamp('2025-10-01',tz='UTC'))]
        elif yr==2026: sub=d[d.yr==2026]
        else: sub=d[d.yr==yr]
        if not len(sub): continue
        m,sh,n=tip(sub); dp=disp(sub); mr=mric(sub)
        brs={rg:tip(sub[sub.reg==rg])[0] for rg in ["side","bear","bull"]}
        lab=f"{yr}" if half is None else f"{yr}{half[:2]}"
        print(f"  {lab:>6s} {n:>5d} | {m:+12.1f} {sh:+6.2f} | {dp:7.0f} | {mr:+6.3f} | {brs['side']:+.0f}/{brs['bear']:+.0f}/{brs['bull']:+.0f}")
# 2026 recent
sub=d[d.yr==2026]
if len(sub):
    m,sh,n=tip(sub); print(f"  {'2026':>6s} {n:>5d} | {m:+12.1f} {sh:+6.2f} | {disp(sub):7.0f} | {mric(sub):+6.3f} | (recent)")
print("\nMR-IC>0 = mean-reversion works (short winners); <0 = momentum regime (strategy's core signal INVERTS).")
print("VALIDATEDONE")
