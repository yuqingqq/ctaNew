"""Does the model capture the TOP-K stably? Three checks, OOS, gated regimes, residual alpha (bps).
(1) PER-RANK marginal edge: short rank i (i-th most-extreme by base book, measured in side+bear where short is active),
    long rank i (i-th by long book, in bear). Monotone decay = model orders the tip correctly. Split H1/H2 for stability.
(2) K SWEEP (gated: bear=L/S, side=short-only, bull=flat): KL x KS grid -> per-cycle Sharpe + monthly %pos + neg-streak.
    Flat across K = robust; peaky = fragile.
(3) reported for v4 (focus) and v3 (reference).
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
d=(lp("hl_lean175_oos","v3b").merge(lp("hl_residrev_oos","v3l"),on=["symbol","open_time"])
   .merge(lp("hl_v4base_oos","v4b"),on=["symbol","open_time"]).merge(lp("hl_v4long_oos","v4l"),on=["symbol","open_time"])
   .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
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
groups=[(ot,g) for ot,g in d.groupby("open_time")]

# ---- (1) per-rank marginal edge, with H1/H2 stability ----
def per_rank(bcol,lcol):
    # short: i-th smallest base pred, in side+bear; long: i-th largest long pred, in bear
    S={i:{"h1":[],"h2":[]} for i in range(1,7)}; L={i:{"h1":[],"h2":[]} for i in range(1,5)}
    for ot,g in groups:
        reg=g["reg"].iloc[0]; half="h1" if ot<mid else "h2"
        if reg in ("side","bear"):
            ss=g.nsmallest(6,bcol)["fwd"].to_numpy()
            for i in range(1,7):
                if len(ss)>=i: S[i][half].append(-ss[i-1])   # short PnL = -fwd
        if reg=="bear":
            ll=g.nlargest(4,lcol)["fwd"].to_numpy()
            for i in range(1,5):
                if len(ll)>=i: L[i][half].append(ll[i-1])
    return S,L
print("=== (1) PER-RANK marginal edge (bps) — is the tip ordered & stable? ===")
for name,(bcol,lcol) in [("v4",("v4b","v4l")),("v3",("v3b","v3l"))]:
    S,L=per_rank(bcol,lcol)
    print(f"\n  [{name}] SHORT rank (side+bear), long PnL=-fwd:")
    print("    rank |  all   H1    H2   (monotone decay + same sign across halves = stable capture)")
    for i in range(1,7):
        al=np.mean(S[i]["h1"]+S[i]["h2"]); h1=np.mean(S[i]["h1"]); h2=np.mean(S[i]["h2"])
        print(f"    {i:>4d} | {al:+5.0f} {h1:+5.0f} {h2:+5.0f}")
    print(f"  [{name}] LONG rank (bear):")
    print("    rank |  all   H1    H2")
    for i in range(1,5):
        al=np.mean(L[i]["h1"]+L[i]["h2"]); h1=np.mean(L[i]["h1"]); h2=np.mean(L[i]["h2"])
        print(f"    {i:>4d} | {al:+5.0f} {h1:+5.0f} {h2:+5.0f}")

# ---- (2) K sweep, gated, per-cycle Sharpe + monthly stability ----
def gated_series(lcol,bcol,KL,KS):
    rows=[]; mos=[]
    for ot,g in groups:
        reg=g["reg"].iloc[0]; dl=(reg=="bear"); ds=(reg in ("bear","side"))
        L=g.nlargest(KL,lcol)["fwd"].mean() if (dl and len(g)>=KL) else 0.0
        S=g.nsmallest(KS,bcol)["fwd"].mean() if (ds and len(g)>=KS) else 0.0
        rows.append((L if dl else 0.0)-(S if ds else 0.0)); mos.append(ot.to_period("M"))
    s=pd.Series(rows); mo=pd.Series(rows,index=mos).groupby(level=0).mean()
    def streak(x):
        mx=c=0
        for v in x:
            c=c+1 if v<0 else 0; mx=max(mx,c)
        return mx
    return s.mean(), (s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan), 100*(mo>0).mean(), streak(mo)
print("\n=== (2) K SWEEP (gated) — flat across K = robust capture; peaky = fragile ===")
for name,(bcol,lcol) in [("v4",("v4b","v4l")),("v3",("v3b","v3l"))]:
    print(f"\n  [{name}]  KL\\KS   (cell = per-cycle Sh / %pos-mo)")
    hdr="   KL  |"+"".join(f"  KS={ks:<9d}" for ks in [1,2,3,4,5]); print(hdr)
    for KL in [1,2,3]:
        cells=[]
        for KS in [1,2,3,4,5]:
            m,sh,pm,st=gated_series(lcol,bcol,KL,KS); cells.append(f"{sh:+4.2f}/{pm:2.0f}%")
        print(f"   {KL:>2d}  |"+"".join(f"  {c:<10s}" for c in cells))
print("\n(per-cycle Sh gross, ~2.4x overlap-inflated. %pos-mo = fraction of months net-positive.)")
print("V4KDONE")
