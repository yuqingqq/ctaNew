"""Per-feature TIP performance BY REGIME — for fine calibration. Fixed H1-pooled sign per feature (honest);
report full-OOS tip in each regime. A feature positive in one regime and NEGATIVE in another (under the fixed
sign) = it REVERSES by regime -> calibration opportunity (regime-conditional use / sign flip).
  LONG  tip = realized alpha of top-1 by oriented feature (want + where we go long: BEAR).
  SHORT tip = PnL (-alpha) of bottom-4 by oriented feature (want + where we short: SIDE, BEAR).
BULL columns shown too: LONG bull is gated off (info only); SHORT bull identifies bull-short-ranker candidates
(the I2 lever vs return_1d). Model rows v4l/v4b = the benchmark the features feed. All residual alpha (bps).
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/home/yuqing/ctaNew")
import live.train_twobook_models as tt
R="/home/yuqing/ctaNew"; H=6
V0L=[f for f in tt.V0 if not f.startswith("funding")]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
    columns=["symbol","open_time","alpha_vs_btc_realized"]+V0L)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan=pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
pan["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
FEATS=V0L+["resid_rev_2","resid_rev_3"]
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
prd=lp("hl_v4base_oos","v4b").merge(lp("hl_v4long_oos","v4l"),on=["symbol","open_time"])
d=pan.merge(prd,on=["symbol","open_time"]).dropna(subset=["fwd"])
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
mid=grid[len(grid)//2]
h1=d[d.open_time<mid]
def rankic(col):
    x=h1.groupby("open_time").apply(lambda g: g[col].rank().corr(g["fwd"].rank())); return x.mean()
sign={f:(1.0 if rankic(f)>=0 else -1.0) for f in FEATS+["v4b","v4l"]}
groups=[(ot,g,g["reg"].iloc[0]) for ot,g in d.groupby("open_time")]
nreg={rg:sum(1 for _,_,r in groups if r==rg) for rg in ["side","bear","bull"]}
print(f"=== Per-feature TIP by regime, OOS (cyc: side {nreg['side']}, bear {nreg['bear']}, bull {nreg['bull']}). Fixed H1 sign. ===")
print("    LONG=top-1 realized alpha (trade long in BEAR). SHORT=bottom-4 PnL (trade short in SIDE+BEAR).")
print("    Negative under fixed sign => feature REVERSES in that regime.\n")
def by_reg(col):
    s=sign[col]; L={"side":[],"bear":[],"bull":[]}; S={"side":[],"bear":[],"bull":[]}
    for ot,g,rg in groups:
        v=g[col].to_numpy()*s; gg=g.assign(_o=v)
        L[rg].append(gg.nlargest(1,"_o")["fwd"].mean())
        if len(gg)>=4: S[rg].append(-gg.nsmallest(4,"_o")["fwd"].mean())
    return {k:(np.mean(L[k]) if L[k] else np.nan) for k in L},{k:(np.mean(S[k]) if S[k] else np.nan) for k in S}
print(f"  {'feature':<24s} {'sg':>2s} |{'LONG side':>9s}{'bear':>7s}{'bull':>7s} |{'SHRT side':>9s}{'bear':>7s}{'bull':>7s}")
for f in FEATS+["v4l","v4b"]:
    L,S=by_reg(f); tag="  <-MODEL" if f.startswith("v4") else ""
    print(f"  {f:<24s} {sign[f]:+2.0f} |{L['side']:+9.0f}{L['bear']:+7.0f}{L['bull']:+7.0f} |{S['side']:+9.0f}{S['bear']:+7.0f}{S['bull']:+7.0f}{tag}")
print("\n(Long traded only in bear; short in side+bear; bull-short col = candidate bull ranker vs return_1d.)")
print("V4FREGDONE")
