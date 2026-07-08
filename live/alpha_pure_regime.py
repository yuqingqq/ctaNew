"""PURE standalone alpha per regime + distribution shape. For each beta-neutral factor, trade it ALONE (rank the
eligible names by the factor each cycle, long/short the tails), unconfounded by the V0_LEAN model or v3's regime
mechanism. Report the per-cycle L/S DISTRIBUTION per regime to answer: are these alphas broad or long-tailed?
  mean, median, %pos, top-3 cycle share (concentration), per-cycle Sharpe (mean/std — the robustness metric).
Compare to the BASE convexity model pred (reference: is the base broad while the factors are tail-driven?).
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; K=2
REPS=["alpha082","alpha095","alpha023","alpha052","alpha159","alpha010","alpha070","alpha088"]
BASE="https://data.binance.vision/data/futures/um/monthly/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","count","tb","tbq","ig"]

pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time"]+REPS)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True)
basep=pd.read_parquet(f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
basep["open_time"]=pd.to_datetime(basep["open_time"],utc=True)
d=fac.merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]).merge(basep,on=["symbol","open_time"],how="left")
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])

# regime
grid=pd.DatetimeIndex(sorted(d["open_time"].unique()))
def fm(per):
    try:
        r=requests.get(f"{BASE}/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode()
        hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr,names=None if hdr==0 else COLS); x.columns=COLS[:x.shape[1]]
        v=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if v.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(v,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce")
        return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[p for p in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if p is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
d["reg"]=d["open_time"].map(reg)

def ls_series(col, sign):    # standalone L/S per cycle: sign>0 -> long top, short bottom
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K or g[col].isna().all(): continue
        top=g.nlargest(K,col)["fwd"].mean(); bot=g.nsmallest(K,col)["fwd"].mean()
        rows.append((ot,reg.get(ot,"side"),sign*(top-bot)))
    return pd.DataFrame(rows,columns=["ot","reg","ls"])

def describe(df,label):
    print(f"\n{label}")
    for rg in ["bull","side","bear","ALL"]:
        s=df["ls"] if rg=="ALL" else df[df.reg==rg]["ls"]
        if len(s)<5: continue
        top3=s.nlargest(3).sum()/s.sum()*100 if s.sum()!=0 else np.nan
        shp=s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan
        print(f"  {rg:5s} n{len(s):4d} mean{s.mean():+7.1f} med{s.median():+7.1f} %pos{100*(s>0).mean():3.0f} "
              f"top3share{top3:4.0f}% cycSharpe{shp:+.2f}")

# base model reference (sign +: long high pred, short low pred)
describe(ls_series("pred",+1.0), "BASE convexity model (reference):")
# each factor standalone, oriented by full-sample IC sign
for f in REPS:
    ic=np.sign(pd.Series(d[f].values).corr(pd.Series(d["fwd"].values),method="spearman"))
    describe(ls_series(f, ic if ic!=0 else 1.0), f"{f} standalone (IC-oriented):")
print("PDONE")

# --- robustness of the BULL standouts: non-overlapping cycles + sub-period split ---
print("\n=== BULL robustness of alpha070 / alpha010 / (base ref) ===")
for f,sign in [("pred",+1.0),("alpha070",None),("alpha010",None)]:
    s = np.sign(pd.Series(d[f].values).corr(pd.Series(d["fwd"].values),method="spearman")) if sign is None else sign
    df=ls_series(f, s if s!=0 else 1.0); bull=df[df.reg=="bull"].sort_values("ot").reset_index(drop=True)
    no=bull.iloc[::H]                                      # non-overlapping bull cycles
    def sh(x): return x.mean()/x.std()*np.sqrt(len(x)) if len(x)>2 and x.std()>0 else np.nan
    h1,h2=np.array_split(bull["ls"].values,2)
    print(f"  {f:8s}: bull cycSharpe(all) {sh(bull['ls']):+.2f} | non-overlap(n{len(no)}) mean{no['ls'].mean():+.1f} med{no['ls'].median():+.1f} Sh{sh(no['ls']):+.2f} "
          f"| half1 mean{h1.mean():+.1f} half2 mean{h2.mean():+.1f}")
print("BRDONE")
