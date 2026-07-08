"""NAKED capture per regime: strip all v3 regime machinery. Each cycle: rank by model pred, take top-K_L longs
(long book) / bottom-K_S shorts (base book), equal weight, measure forward 24h residual-alpha L/S spread. NO regime
gate, NO bull/bear overlays, NO conc_cap. Compare baseline (V0_LEAN) vs +factor, split by regime -> does the factor
improve RAW capture in its favored regime? (Fair test: full-stack backtest masks regime-specific alpha.)
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; KS=2; KL=1; D=f"{R}/live/state/convexity"
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","count","tb","tbq","ig"]

pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4

def loadpair(base,long):
    b=pd.read_parquet(f"{D}/{base}/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
    l=pd.read_parquet(f"{D}/{long}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"pl"})
    for x in (b,l): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
    m=b.merge(l,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
    return m[m.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])

# regime
grid=pd.DatetimeIndex(sorted(pan[pan.open_time>=pd.Timestamp("2025-01-01",tz="UTC")]["open_time"].unique()))
def fm(per):
    try:
        r=requests.get(f"{BASE}/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode()
        hdr=0 if raw.split(",",1)[0]=="open_time" else None
        d=pd.read_csv(io.StringIO(raw),header=hdr,names=None if hdr==0 else COLS); d.columns=COLS[:d.shape[1]]
        v=pd.to_numeric(d["open_time"],errors="coerce"); u="us" if v.dropna().median()>1e15 else "ms"
        d["open_time"]=pd.to_datetime(v,unit=u,utc=True); d["close"]=pd.to_numeric(d["close"],errors="coerce")
        return d[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[p for p in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill()
r30=(btc/btc.shift(180)-1); reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}

def capture(m):
    rows=[]
    for ot,g in m.groupby("open_time"):
        if len(g)<KS+KL: continue
        lo=g.nlargest(KL,"pl")["fwd"].mean(); sh=g.nsmallest(KS,"pred")["fwd"].mean()
        rows.append((reg.get(ot,"side"), lo-sh))       # L/S gross spread (bps)
    df=pd.DataFrame(rows,columns=["reg","ls"])
    return df.groupby("reg")["ls"].agg(["mean","count"]), df["ls"].mean()

base=loadpair("hl_cand_base_lean","hl_cand_long_lean")
print(f"NAKED L/S capture (K_short={KS}, K_long={KL}, gate-free), gross 24h L/S bps by regime\n")
bR,bAll=capture(base); print("baseline V0_LEAN:", {k:f'{v:+.1f}(n{int(n)})' for k,(v,n) in bR.iterrows()}, f"| ALL {bAll:+.1f}")
for fn in ["alpha010","alpha095"]:
    fR,fAll=capture(loadpair(f"hl_cand_{fn}_base",f"hl_cand_{fn}_long"))
    print(f"\n+{fn}:", {k:f'{v:+.1f}(n{int(n)})' for k,(v,n) in fR.iterrows()}, f"| ALL {fAll:+.1f}")
    print(f"  Δ vs baseline:", {k:f"{(fR.loc[k,'mean']-bR.loc[k,'mean']):+.1f}" for k in fR.index if k in bR.index}, f"| ALL {fAll-bAll:+.1f}")
print("NDONE")

# --- robustness: is alpha095's bull lift broad or squeeze-concentrated? ---
def paired_bull(base_m, fac_m):
    b=base_m.set_index("open_time") if "open_time" in base_m else base_m
    rows=[]
    bg=dict(tuple(base_m.groupby("open_time"))); fg=dict(tuple(fac_m.groupby("open_time")))
    for ot in bg:
        if reg.get(ot)!="bull": continue
        gb=bg[ot]; gf=fg.get(ot)
        if gf is None or len(gb)<KS+KL or len(gf)<KS+KL: continue
        lb=gb.nlargest(KL,"pl")["fwd"].mean()-gb.nsmallest(KS,"pred")["fwd"].mean()
        lf=gf.nlargest(KL,"pl")["fwd"].mean()-gf.nsmallest(KS,"pred")["fwd"].mean()
        rows.append(lf-lb)
    d=pd.Series(rows)
    print(f"\nBULL per-cycle L/S delta (alpha095-baseline): n={len(d)} mean{d.mean():+.1f} median{d.median():+.1f} "
          f"| >0: {100*(d>0).mean():.0f}% | top-3 cycles share {100*d.nlargest(3).sum()/d.sum():.0f}% of total")
paired_bull(base, loadpair("hl_cand_alpha095_base","hl_cand_alpha095_long"))
print("BDONE")
