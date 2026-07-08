"""Verify whether the model selects MOMENTUM-following symbols in bull. Two checks per bull-depth × window:
(A) the model's actual picks: mean trailing-3bar residual momentum of top-K longs and bottom-K shorts (are longs
    recent WINNERS? are shorts recent LOSERS? = momentum-following) + their realized fwd.
(B) raw per-feature rank-IC vs fwd. Momentum features (return_1d, ret_3d): +IC = momentum regime (recent-up ->
    forward-up). Reversion features (resid_rev_2/3 = -trailing residual, high=recent loser): +IC = losers bounce
    (reversion); -IC = losers keep falling (momentum). All residual alpha. RECENT v4=hl_tgt_res_*, OOS=hl_v4*_oos.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/home/yuqing/ctaNew")
import live.train_twobook_models as tt
R="/home/yuqing/ctaNew"; H=6; K=2
V0L=[f for f in tt.V0 if not f.startswith("funding")]
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"]+V0L)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
pan["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
pan["trail3"]=a.transform(lambda s:s.shift(1).rolling(3).sum())*1e4     # recent residual momentum
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
pan=pan.dropna(subset=["fwd"])
FEATS=["return_1d","ret_3d","resid_rev_2","resid_rev_3","atr_pct","rvol_7d","vwap_slope_96","bars_since_high","idio_vol_to_btc_1d"]
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
def build(b,l):
    return pan.merge(lp(b,"pb"),on=["symbol","open_time"]).merge(lp(l,"pl"),on=["symbol","open_time"])
WIN={"RECENT":("hl_tgt_res_base","hl_tgt_res_long"),"OOS":("hl_v4base_oos","hl_v4long_oos")}
data={k:build(*v) for k,v in WIN.items()}
allg=pd.DatetimeIndex(sorted(set().union(*[set(d["open_time"].unique()) for d in data.values()])))
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
rr={t:v for t,v in r30.items()}
def picks_mom(sub):
    Lm=[];Sm=[];Lf=[];Sf=[]
    for ot,g in sub.groupby("open_time"):
        if len(g)<2*K: continue
        lg=g.nlargest(K,"pl"); sg=g.nsmallest(K,"pb")
        Lm.append(lg["trail3"].mean()); Sm.append(sg["trail3"].mean()); Lf.append(lg["fwd"].mean()); Sf.append(sg["fwd"].mean())
    return np.nanmean(Lm),np.nanmean(Lf),np.nanmean(Sm),np.nanmean(Sf)
def featic(sub,f):
    return sub.groupby("open_time").apply(lambda g:g[f].rank().corr(g["fwd"].rank())).mean()
for win,d in data.items():
    d=d.copy(); d["r30"]=d["open_time"].map(rr)
    for lab,mask in [("bull-mild",(d.r30>0.10)&(d.r30<=0.20)),("bull-deep",d.r30>0.20)]:
        sub=d[mask]; nc=sub.open_time.nunique()
        if nc<20: print(f"\n[{win} {lab}] n={nc} (too few)"); continue
        Lm,Lf,Sm,Sf=picks_mom(sub)
        print(f"\n[{win} {lab}] n_cyc={nc}")
        print(f"  (A) MODEL picks: LONG trail3={Lm:+.0f} (want NEG=losers) fwd={Lf:+.0f} | SHORT trail3={Sm:+.0f} (want POS=winners) fwd={Sf:+.0f}")
        print(f"      -> LONG picks are recent {'WINNERS(momentum-long)' if Lm>0 else 'losers(reversion-long)'}; SHORT picks are recent {'WINNERS(reversion-short)' if Sm>0 else 'LOSERS(momentum-short)'}")
        ics={f:featic(sub,f) for f in FEATS}
        print(f"  (B) raw feature IC vs fwd (momentum feats +IC=momentum; resid_rev +IC=reversion works):")
        print(f"      return_1d {ics['return_1d']:+.3f} | ret_3d {ics['ret_3d']:+.3f} | resid_rev_2 {ics['resid_rev_2']:+.3f} | resid_rev_3 {ics['resid_rev_3']:+.3f}")
        print(f"      atr {ics['atr_pct']:+.3f} | rvol {ics['rvol_7d']:+.3f} | vwap_slope {ics['vwap_slope_96']:+.3f} | bars_since_high {ics['bars_since_high']:+.3f} | idio_vol_1d {ics['idio_vol_to_btc_1d']:+.3f}")
print("\nBULLFEATDONE")
