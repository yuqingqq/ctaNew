"""FIXED honest harness for regime-feature-config tests. Agents propose a config; this harness retrains + evaluates
it under a locked PIT protocol so no look-ahead can be introduced. Usage:
    python3 live/regime_feat_harness.py <config.json> <out_metrics.json>
config.json = {"name":str, "bear_long":[feats], "bear_short":[feats], "side_short":[feats]}
  - bear_long : feature set for the LONG book used in BEAR (universe = V0_LEAN + resid_rev_2/3). default full.
  - bear_short: feature set for the SHORT/base book used in BEAR (universe = V0_LEAN). default full-14.
  - side_short: feature set for the SHORT/base book used in SIDE (universe = V0_LEAN). default full-14.
Non-bear long is never traded (gate). Books retrained WF (residual target, HL60, embargo 1d, min300) over
2023-01..2025-09, cached by feature-set hash. Eval: gated KL=1/KS=3, regime-conditional vs POOLED v4
(hl_v4base_oos/hl_v4long_oos). Decide on H1, VALIDATE on H2. Metrics: H1/H2 paired diff, H2 block-bootstrap CI,
per-fold breadth (bear folds + all folds), H2 Sharpe. Verdict: PASS iff H2 CI low>0 AND bear-fold majority;
FAIL iff H2 CI high<0; else NEUTRAL. Prints one-line summary + writes metrics json.
"""
import sys, json, hashlib, io, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); V0_LEAN=[f for f in V0 if not f.startswith("funding")]
RR=["resid_rev_2","resid_rev_3"]; LONG_UNIV=V0_LEAN+RR; SHORT_UNIV=V0_LEAN
EMB=pd.Timedelta(days=1); HL=60.0; KL=1; KS=3
CACHE=REPO/"live/state/convexity/hl_rfcache"; CACHE.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp("2023-01-01",tz="UTC"); END=pd.Timestamp("2025-10-01",tz="UTC")
CUTS=list(pd.date_range(START,END,freq="MS",tz="UTC"))
cfg=json.load(open(sys.argv[1])); OUT=sys.argv[2]
def norm(feats,univ):
    fs=[f for f in feats if f in univ]; return fs if fs else list(univ)
bl=norm(cfg.get("bear_long",LONG_UNIV),LONG_UNIV)
bs=norm(cfg.get("bear_short",SHORT_UNIV),SHORT_UNIV)
ss=norm(cfg.get("side_short",SHORT_UNIV),SHORT_UNIV)

_PAN=[None]
def panel():
    if _PAN[0] is None:
        P=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","alpha_vs_btc_realized"]+V0)
        P["open_time"]=pd.to_datetime(P["open_time"],utc=True); P["exit_time"]=pd.to_datetime(P["exit_time"],utc=True)
        P=P[(P.open_time.dt.hour%4==0)&(P.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
        a=P.groupby("symbol")["alpha_vs_btc_realized"]
        P["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
        P["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)
        g=P.groupby("open_time"); sd=g["alpha_vs_btc_realized"].transform("std").replace(0,np.nan)
        P["xs_z"]=((P["alpha_vs_btc_realized"]-g["alpha_vs_btc_realized"].transform("mean"))/sd).clip(-10,10)
        _PAN[0]=P.sort_values(["symbol","open_time"]).reset_index(drop=True)
    return _PAN[0]
def book(feats):
    h=hashlib.md5((",".join(sorted(feats))).encode()).hexdigest()[:12]
    out=CACHE/f"{h}.parquet"
    if out.exists(): d=pd.read_parquet(out); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
    P=panel(); rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[(P.exit_time<fc)&P["xs_z"].notna()]; te=P[(P.open_time>=c0)&(P.open_time<c1)]
        if not len(tr) or not len(te): continue
        t_end=tr["open_time"].max()
        for sym,g in tr.groupby("symbol"):
            if len(g)<300: continue
            try:
                s,hh=x6.fit_preproc(g,feats); X=x6.apply_preproc(g,feats,s,hh)
                w=np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,g["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "pred":m.predict(x6.apply_preproc(gte,feats,s,hh)),"fold":i}))
            except Exception: pass
    d=pd.concat(rec,ignore_index=True); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d.to_parquet(out); return d
def regime():
    rc=REPO/"live/state/longtail/btc_reg.parquet"
    if rc.exists(): r=pd.read_parquet(rc); return dict(zip(pd.to_datetime(r["open_time"],utc=True),r["reg"]))
    grid=pd.DatetimeIndex(sorted(pd.read_parquet(f"{REPO}/live/state/convexity/hl_v4base_oos/v0full_hl60.parquet",columns=["open_time"])["open_time"].pipe(pd.to_datetime,utc=True).unique()))
    def fm(per):
        try:
            rr=requests.get(f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
            if rr.status_code!=200: return None
            z=zipfile.ZipFile(io.BytesIO(rr.content)); raw=z.read(z.namelist()[0]).decode(); hdr=0 if raw.split(",",1)[0]=="open_time" else None
            x=pd.read_csv(io.StringIO(raw),header=hdr); x.columns=["open_time","o","h","l","close","v","ct","qv","n","tb","tbq","ig"][:x.shape[1]]
            vv=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if vv.dropna().median()>1e15 else "ms"
            x["open_time"]=pd.to_datetime(vv,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
        except Exception: return None
    with ThreadPoolExecutor(max_workers=12) as ex:
        parts=[q for q in ex.map(fm,pd.period_range("2022-06",grid.max().to_period("M"),freq="M")) if q is not None]
    btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
    btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
    rd={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
    pd.DataFrame({"open_time":list(rd.keys()),"reg":list(rd.values())}).to_parquet(rc); return rd

pan=pd.read_parquet(f"{REPO}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(6).sum().shift(-5))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{REPO}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred","fold"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
base=(lp("hl_v4base_oos","p_b").merge(lp("hl_v4long_oos","p_l")[["symbol","open_time","p_l"]],on=["symbol","open_time"])
      .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])).dropna(subset=["fwd"])
# attach config books
base=base.merge(book(bl).rename(columns={"pred":"c_l"})[["symbol","open_time","c_l"]],on=["symbol","open_time"],how="left")
base=base.merge(book(bs).rename(columns={"pred":"c_bs"})[["symbol","open_time","c_bs"]],on=["symbol","open_time"],how="left")
base=base.merge(book(ss).rename(columns={"pred":"c_ss"})[["symbol","open_time","c_ss"]],on=["symbol","open_time"],how="left")
regd=regime(); base["reg"]=base["open_time"].map(regd)
grid=pd.DatetimeIndex(sorted(base["open_time"].unique())); mid=grid[len(grid)//2]
groups=[(ot,g,g["reg"].iloc[0],("h1" if ot<mid else "h2"),g["fold"].iloc[0]) for ot,g in base.groupby("open_time")]
def cyc(g,rg,cfg_on):
    dl=(rg=="bear"); ds=(rg in ("bear","side"))
    lcol=("c_l" if cfg_on else "p_l")
    bcol=("c_bs" if (cfg_on and rg=="bear") else ("c_ss" if (cfg_on and rg=="side") else "p_b"))
    L=g.nlargest(KL,lcol)["fwd"].mean() if (dl and len(g)>=KL and g[lcol].notna().all()) else 0.0
    S=g.nsmallest(KS,bcol)["fwd"].mean() if (ds and len(g)>=KS and g[bcol].notna().any()) else 0.0
    return (L if dl else 0.0)-(S if ds else 0.0)
rows=[]
for ot,g,rg,hf,f in groups:
    rows.append((cyc(g,rg,False),cyc(g,rg,True),hf,f,rg))
df=pd.DataFrame(rows,columns=["pool","cfg","hf","fold","reg"]); df["d"]=df["cfg"]-df["pool"]
def sh(s): return float(s.mean()/s.std()*np.sqrt(len(s))) if s.std()>0 else float("nan")
h1=df[df.hf=="h1"]; h2=df[df.hf=="h2"]
# block bootstrap CI on H2 paired diff (5-day ~ 30-cycle blocks)
h2d=h2["d"].to_numpy(); B=30; nb=max(1,len(h2d)//B); rng=np.random.RandomState(0)
boot=[np.mean(np.concatenate([h2d[i*B:(i+1)*B] for i in rng.randint(0,nb,nb)])) for _ in range(2000)] if nb>0 else [0]
ci_lo,ci_hi=float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))
bf=df[df.fold.isin(set(df[df.reg=='bear']['fold']))].groupby("fold")["d"].mean()
af=df.groupby("fold")["d"].mean()
verdict="PASS" if (ci_lo>0 and (bf>0).sum()>len(bf)/2) else ("FAIL" if ci_hi<0 else "NEUTRAL")
m={"name":cfg.get("name","?"),"bear_long":bl,"bear_short":bs,"side_short":ss,
   "h1_diff":float(h1["d"].mean()),"h2_diff":float(h2["d"].mean()),"h2_ci":[ci_lo,ci_hi],
   "h2_sh_pool":sh(h2["pool"]),"h2_sh_cfg":sh(h2["cfg"]),
   "bearfold_wins":[int((bf>0).sum()),int(len(bf))],"allfold_wins":[int((af>0).sum()),int(len(af))],"verdict":verdict}
json.dump(m,open(OUT,"w"),indent=1)
print(f"[{m['name']}] {verdict} | H1Δ{m['h1_diff']:+.1f} H2Δ{m['h2_diff']:+.1f} CI[{ci_lo:+.1f},{ci_hi:+.1f}] "
      f"H2Sh {m['h2_sh_pool']:+.2f}->{m['h2_sh_cfg']:+.2f} bearfold {m['bearfold_wins'][0]}/{m['bearfold_wins'][1]}")
