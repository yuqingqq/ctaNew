"""feat-iter3 (FINAL feature batch): microstructure / liquidity — a different mechanism (illiquidity-driven overshoot,
taker imbalance, intraday range/close-location). Built by resampling 5m klines -> 4h OHLCV+volume+taker. Same nested-OOS
marginal IC test. If empty, the feature layer is exhausted on free data (3 distinct batches: recombination, new-axes, microstructure).
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
KLINES = REPO/"data/ml/test/parquet/klines"
x6 = tt.x6; V0 = list(tt.V0); RR = ["resid_rev_2","resid_rev_3"]; EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-06-30"]]

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)

def load_ohlcv_4h(sym, last_n=520):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    files = sorted(sd.glob("*.parquet"))[-last_n:]
    if not files: return None
    want=["open_time","high","low","close","volume","quote_volume","taker_buy_quote_volume"]
    dfs=[]
    for f in files:
        try:
            avail=set(pd.read_parquet(f, columns=None).columns) if False else None
            dfs.append(pd.read_parquet(f, columns=want))
        except Exception:
            d=pd.read_parquet(f)                                   # schema lacks taker col -> read all, fill
            for c in want:
                if c not in d.columns: d[c]=np.nan
            dfs.append(d[want])
    if not dfs: return None
    df=pd.concat(dfs, ignore_index=True).drop_duplicates("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    df=df.set_index("open_time").sort_index()
    r=df.resample("4h")
    o=pd.DataFrame({"high":r["high"].max(),"low":r["low"].min(),"close":r["close"].last(),
                    "vol":r["volume"].sum(),"qvol":r["quote_volume"].sum(),"tbq":r["taker_buy_quote_volume"].sum()})
    return o.dropna()

print("resampling 5m->4h OHLCV per symbol (microstructure)...", flush=True)
syms = PAN["symbol"].unique().tolist()
rows=[]; nbad=0; t0=time.time()
for sym in syms:
    o = load_ohlcv_4h(sym)
    if o is None or len(o)<60: nbad+=1; continue
    ret = o["close"].pct_change().abs()
    amihud = (ret/(o["qvol"]+1.0)).rolling(42).mean()                       # trailing illiquidity
    volspike = o["vol"]/(o["vol"].rolling(42).mean()+1e-9)
    rng = (o["high"]-o["low"])/(o["close"]+1e-9)
    closeloc = (o["close"]-o["low"])/((o["high"]-o["low"]).replace(0,np.nan))
    takerimb = (2*o["tbq"]-o["qvol"])/(o["qvol"]+1.0)
    df=pd.DataFrame({"open_time":o.index,"amihud":amihud.values,"vol_spike":volspike.values,
                     "intraday_range":rng.values,"close_loc":closeloc.values,"taker_imb":takerimb.values})
    df["symbol"]=sym; rows.append(df)
KF=pd.concat(rows,ignore_index=True); KF["open_time"]=pd.to_datetime(KF["open_time"],utc=True)
PAN=PAN.merge(KF,on=["symbol","open_time"],how="left")
PAN["amihud_xsrank"]=PAN.groupby("open_time")["amihud"].rank(pct=True)
CANDS=["amihud","amihud_xsrank","vol_spike","intraday_range","close_loc","taker_imb"]
for c in CANDS: PAN[c]=PAN[c].replace([np.inf,-np.inf],np.nan).fillna(0.0)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"built {len(CANDS)} microstructure features in {time.time()-t0:.0f}s ({nbad}/{len(syms)} syms no/short klines)", flush=True)

def gen_ic(feats):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; tr=PAN[(PAN.exit_time<c0-EMB)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"open_time":gte["open_time"].values,
                    "pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"xs_z":gte["xs_z"].values}))
            except Exception: pass
    W=pd.concat(rec,ignore_index=True); W["open_time"]=pd.to_datetime(W["open_time"],utc=True)
    ic=W.dropna().groupby("open_time").apply(lambda x:x["pred"].corr(x["xs_z"],method="spearman")).dropna()
    cp=pd.to_datetime(CUTS,utc=True)
    pf=[ic[(ic.index>=cp[i])&(ic.index<cp[i+1])].mean() for i in range(len(cp)-1)]
    return ic.mean(), pf

t0=time.time(); base_ic, base_pf = gen_ic(V0+RR); print(f"\nBASELINE (V0+RR) IC {base_ic:+.4f} ({time.time()-t0:.0f}s)")
print("MARGINAL IC LIFT (microstructure; bar lift>=+0.004 & >=6/9 folds):")
res=[]
for name in CANDS:
    t0=time.time(); ic,pf=gen_ic(V0+RR+[name]); fu=sum(1 for b,n in zip(base_pf,pf) if n>b)
    res.append((name, ic-base_ic, fu)); print(f"  {name:18s} lift {ic-base_ic:+.4f}  IC {ic:+.4f}  folds_up {fu}/9 ({time.time()-t0:.0f}s)", flush=True)
res.sort(key=lambda r:-r[1]); win=[r for r in res if r[1]>=0.004 and r[2]>=6]
print(f"\nTOP: {[(n,round(l,4),f) for n,l,f in res[:4]]}")
print(f"WINNERS: {[(n,round(l,4),f) for n,l,f in win] or 'NONE'}")
print("DONE feat_ic3")
