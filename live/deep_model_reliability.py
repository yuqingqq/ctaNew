"""Per-regime MODEL RELIABILITY router. For each regime class, measure whether the model's tip edge is RELIABLE —
present (SNR) AND STABLE across independent samples (OOS first-half, OOS second-half, RECENT). A regime is
USE-PRED if the tip SNR is positive & consistent across all 3 samples; UNRELIABLE otherwise (→ switch signal or sit out).
Tip = top-K_long long / bottom-K_short short by v4 pred. K_long=1,K_short=2. bps.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; KL=1; KS=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
def build(b,l,tag):
    d=lp(b,"pb").merge(lp(l,"pl"),on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]).dropna(subset=["fwd"])
    d["src"]=tag; return d
oos=build("hl_v4base_oos","hl_v4long_oos","oos")
rec=build("hl_tgt_res_base","hl_tgt_res_long","rec")
mid=oos["open_time"].quantile(0.5)
oos["samp"]=np.where(oos["open_time"]<mid,"OOS-H1","OOS-H2"); rec["samp"]="RECENT"
d=pd.concat([oos,rec],ignore_index=True)
allg=pd.DatetimeIndex(sorted(d["open_time"].unique()))
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
d["r30"]=d["open_time"].map(r30.to_dict())
def regime(v):
    if v<-0.10: return "bear"
    if v>0.20: return "bull-deep"
    if v>0.10: return "bull-mild"
    return "side"
d["reg"]=d["r30"].apply(lambda v: regime(v) if pd.notna(v) else None)
def tipstats(sub):
    tips=[g.nlargest(KL,"pl")["fwd"].mean()-g.nsmallest(KS,"pb")["fwd"].mean() for _,g in sub.groupby("open_time") if len(g)>=KL+KS]
    s=pd.Series(tips)
    if len(s)<10: return None
    return s.mean(), s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan, (s>0).mean()*100, len(s)
print("=== PER-REGIME MODEL RELIABILITY (tip SNR across 3 independent samples) ===")
print("   USE-PRED iff tip SNR positive & consistent across OOS-H1, OOS-H2, RECENT.\n")
print(f"  {'regime':<11s} | {'OOS-H1 (mean/SNR)':>18s} | {'OOS-H2 (mean/SNR)':>18s} | {'RECENT (mean/SNR)':>18s} | verdict")
for reg in ["bear","side","bull-mild","bull-deep"]:
    cells={}
    for samp in ["OOS-H1","OOS-H2","RECENT"]:
        st=tipstats(d[(d.reg==reg)&(d.samp==samp)]); cells[samp]=st
    def fmt(st): return f"{st[0]:+6.0f}/{st[1]:+5.2f}" if st else "  n/a  "
    snrs=[cells[s][1] for s in cells if cells[s]]
    npos=sum(1 for s in cells if cells[s] and cells[s][1]>0.3)
    verdict = "USE-PRED (reliable)" if (npos>=2 and all(cells[s][1]>-0.3 for s in cells if cells[s])) else ("UNRELIABLE -> switch/sit-out" if npos==0 else "PARTIAL (regime-dependent)")
    print(f"  {reg:<11s} | {fmt(cells['OOS-H1']):>18s} | {fmt(cells['OOS-H2']):>18s} | {fmt(cells['RECENT']):>18s} | {verdict}")
print("\n(mean bps / per-cycle SNR ~2.4x infl. RELIABLE = SNR>+0.3 in >=2 of 3 samples & never <-0.3.)")
print("RELIABDONE")
