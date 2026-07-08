"""Validate the model's stable tip capture FROM THE FEATURES. If the top-1-long / short-band capture is real
signal (not overfit), the individual features should show tip performance at the same ranks. For each feature:
  sign it on H1 (rank-IC vs fwd residual alpha, so oriented-high => predicts high alpha), then measure the TIP on
  H2 (honest, sign chosen out-of-time) and H1 (in-sample ref):
    LONG  tip = realized alpha of top-1 by oriented feature, in BEAR.
    SHORT tip = PnL (-alpha) of bottom-4 by oriented feature, in SIDE+BEAR.
Compare each feature to the v4 model pred (v4l long / v4b short). Feature with stable H1&H2 tip = load-bearing for
the capture. All residual alpha (bps).
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
mid=grid[len(grid)//2]; d["half"]=np.where(d["open_time"]<mid,"h1","h2")
groups=[(ot,g) for ot,g in d.groupby("open_time")]

# H1 sign per feature: pooled rank-IC of feature vs fwd on h1
h1=d[d.half=="h1"]
def rankic(col):
    x=h1.groupby("open_time").apply(lambda g: g[col].rank().corr(g["fwd"].rank())); return x.mean()
sign={f:(1.0 if rankic(f)>=0 else -1.0) for f in FEATS+["v4b","v4l"]}
print("=== Per-feature TIP performance (validate the model's stable capture). Sign fixed on H1, tip on H2 (honest). ===")
print("   oriented so high=>predicts-high-alpha. LONG=top1 realized alpha in bear (want +). SHORT=bottom4 PnL in side+bear (want +).\n")
def tips(col):
    s=sign[col]; Lh={"h1":[],"h2":[]}; Sh={"h1":[],"h2":[]}
    for ot,g in groups:
        reg=g["reg"].iloc[0]; hf=g["half"].iloc[0]; v=g[col].to_numpy()*s
        gg=g.assign(_o=v)
        if reg=="bear" and len(gg)>=1:
            Lh[hf].append(gg.nlargest(1,"_o")["fwd"].mean())
        if reg in ("side","bear") and len(gg)>=4:
            Sh[hf].append(-gg.nsmallest(4,"_o")["fwd"].mean())
    return (np.mean(Lh["h1"]) if Lh["h1"] else np.nan, np.mean(Lh["h2"]) if Lh["h2"] else np.nan,
            np.mean(Sh["h1"]) if Sh["h1"] else np.nan, np.mean(Sh["h2"]) if Sh["h2"] else np.nan)
rows=[]
for f in FEATS+["v4l","v4b"]:
    lh1,lh2,sh1,sh2=tips(f); rows.append((f,sign[f],lh1,lh2,sh1,sh2))
print(f"  {'feature':<26s} {'sgn':>3s} | {'LONG-tip H1':>11s} {'H2':>6s} | {'SHORT-tip H1':>12s} {'H2':>6s}  stable?")
for f,sg,lh1,lh2,sh1,sh2 in rows:
    lst="L✓" if (np.sign(lh1)==np.sign(lh2) and lh2>0) else "L·"
    sst="S✓" if (np.sign(sh1)==np.sign(sh2) and sh2>0) else "S·"
    tag="MODEL" if f.startswith("v4") else ""
    print(f"  {f:<26s} {sg:+3.0f} | {lh1:+11.0f} {lh2:+6.0f} | {sh1:+12.0f} {sh2:+6.0f}  {lst} {sst} {tag}")
print("\n(L✓ = long top-1 tip positive AND same sign both halves; S✓ = short bottom-4 tip positive & stable.)")
print("V4FEATDONE")
