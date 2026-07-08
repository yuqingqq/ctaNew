"""Pick K net of TURNOVER cost. Gated (bear:L/S, side:short-only, bull:flat), KL=1 (established), sweep KS.
Equal-weight 1/K per side. Each cycle: gross L/S (mean fwd) minus turnover cost = (|ΔL|/KL + |ΔS|/KS)*cost_bps,
where |Δ| = symmetric-difference of this cycle's leg book vs previous (entries+exits; inactive leg = empty set).
This mirrors the bot's cost_of (fee on notional traded); funding is ~K-independent so omitted from K SELECTION.
Report net mean, net per-cycle Sharpe (gross ~2.4x overlap-inflated but consistent across K), avg turnover. v4 & v3.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365*6)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
d=(lp("hl_v4base_oos","v4b").merge(lp("hl_v4long_oos","v4l"),on=["symbol","open_time"])
   .merge(lp("hl_lean175_oos","v3b"),on=["symbol","open_time"]).merge(lp("hl_residrev_oos","v3l"),on=["symbol","open_time"])
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
groups=[(ot,g,g["reg"].iloc[0]) for ot,g in d.groupby("open_time")]
def run(lcol,bcol,KL,KS,cost):
    prevL=set(); prevS=set(); g_=[]; n_=[]; turn=[]
    for ot,g,rg in groups:
        dl=(rg=="bear"); ds=(rg in ("bear","side"))
        L=g.nlargest(KL,lcol) if (dl and len(g)>=KL) else g.iloc[:0]
        S=g.nsmallest(KS,bcol) if (ds and len(g)>=KS) else g.iloc[:0]
        Ls=set(L["symbol"]); Ss=set(S["symbol"])
        gross=(L["fwd"].mean() if len(L) else 0.0)-(S["fwd"].mean() if len(S) else 0.0)
        dL=len(Ls^prevL); dS=len(Ss^prevS)
        cst=(dL/max(KL,1)+dS/max(KS,1))*cost
        g_.append(gross); n_.append(gross-cst); turn.append(dL+dS); prevL,prevS=Ls,Ss
    g_=pd.Series(g_); n_=pd.Series(n_)
    return g_.mean(), n_.mean(), (n_.mean()/n_.std()*np.sqrt(len(n_)) if n_.std()>0 else np.nan), np.mean(turn)
for name,(bcol,lcol) in [("v4",("v4b","v4l")),("v3",("v3b","v3l"))]:
    print(f"\n=== [{name}] KL=1, sweep KS — net of turnover cost (per-cycle Sh ~2.4x overlap-inflated) ===")
    print(f"  {'KS':>2s} | {'gross':>6s} | {'net@4.5':>8s} {'Sh@4.5':>7s} | {'net@9':>7s} {'Sh@9':>7s} | avg legs turned/cyc")
    for KS in [1,2,3,4,5]:
        gm,nm45,sh45,tn=run(lcol,bcol,1,KS,4.5); _,nm9,sh9,_=run(lcol,bcol,1,KS,9.0)
        print(f"  {KS:>2d} | {gm:+6.1f} | {nm45:+8.1f} {sh45:+7.2f} | {nm9:+7.1f} {sh9:+7.2f} | {tn:.2f}")
print("\n(net = gross − turnover×cost/leg. Pick KS by net Sharpe at the realistic cost.)")
print("KNETDONE")
