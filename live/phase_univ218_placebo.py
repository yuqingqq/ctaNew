"""MATCHED PLACEBO for the 175->217 'improvement': isolate the xs_z renormalization channel.

The 175->217 run mixed THREE channels: (1) xs_z label renormalization (the per-open_time mean/std
that defines the per-symbol Ridge label moves when symbols are added); (2) bars_since_high_xs_rank
silently dropped (panel218 lacked it, apply_preproc zero-fills); (3) mpit high-vol exclusion cutoff
shifted (52% of 217 vs 52% of 175). The 42 new symbols are NEVER traded (locked out of univ_meta=175).

This harness ISOLATES channel 1: it holds bars_since_high_xs_rank (recomputed over the 175) and the
mpit exclusion set (top-52% of the 175) FIXED across every variant, and varies ONLY the returns of
the added symbols that feed the xs_z normalization. Then it asks: does adding the REAL 42 beat adding
information-free placebos (time-shuffled / gaussian-noise / random-subset)?
  - If real ~= placebos -> the gain is a normalization-rescaling artifact, NOT 'more information'.
  - If real >> placebos -> the broader cross-section genuinely sharpens the labels.

Dense-window replay (test folds 2024-11..2026-06; Sharpe measured 2025-01..2026-06). Trains full-history
expanding. Trades the 175 only. Writes results incrementally to placebo_results.csv.
"""
import sys, os, time, subprocess, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")

REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); RR=["resid_rev_2","resid_rev_3"]; EMB=pd.Timedelta(days=1); HL=60.0; ANN=np.sqrt(365)
OUT=REPO/"live/state/v3loop/univ218_placebo"; OUT.mkdir(parents=True,exist_ok=True)
RESCSV=OUT/"placebo_results.csv"
t0=time.time()

# ---- load the 217 panel (has return_pct + all features + bars_since_high for all syms) ----
P=pd.read_parquet(REPO/"live/state/v3loop/univ218/panel218.parquet")
P["open_time"]=pd.to_datetime(P["open_time"],utc=True); P["exit_time"]=pd.to_datetime(P["exit_time"],utc=True)
OLD=set(pd.read_parquet(tt.PANEL,columns=["symbol"]).symbol.unique())
NEW=sorted(set(P.symbol.unique())-OLD)
print(f"175 old + {len(NEW)} new = {P.symbol.nunique()} syms in panel218",flush=True)

P175=P[P.symbol.isin(OLD)].copy().sort_values(["symbol","open_time"]).reset_index(drop=True)
PNEW=P[P.symbol.isin(NEW)].copy()
# bars_since_high_xs_rank: HELD FIXED over the 175 (channel-2 controlled)
P175["bars_since_high_xs_rank"]=P175.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
# resid_rev per symbol (channel-independent)
a=P175.groupby("symbol")["alpha_vs_btc_realized"]
P175["resid_rev_2"]=(-a.transform(lambda s:s.shift(1).rolling(2).sum())).fillna(0.0)
P175["resid_rev_3"]=(-a.transform(lambda s:s.shift(1).rolling(3).sum())).fillna(0.0)

# normalization base = the 175 returns (always real); added rows extend the cross-section
BASE_RET=P175[["open_time","return_pct"]].copy()
NEW_RET=PNEW[["symbol","open_time","return_pct"]].copy()
NOISE_SIGMA=float(NEW_RET["return_pct"].std())

# dense test folds only (train is full-history expanding regardless)
CUTS=[pd.Timestamp(t,tz="UTC") for t in pd.date_range("2024-11-01","2026-06-01",freq="MS")]+[pd.Timestamp("2026-06-05",tz="UTC")]

# ---- mpit exclusion FIXED over the 175 (channel-3 controlled) ----
panv=pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","rvol_7d"])
panv["open_time"]=pd.to_datetime(panv["open_time"],utc=True); panv=panv[panv.symbol.isin(OLD)]
FRAC=0.52
def excl_for(c0):
    lo=c0-pd.Timedelta(days=30); r=panv[(panv.open_time>=lo)&(panv.open_time<c0)].groupby("symbol")["rvol_7d"].mean().dropna()
    return set(r.sort_values(ascending=False).index[:int(round(FRAC*len(r)))])
EXCL={i:excl_for(CUTS[i]) for i in range(len(CUTS)-1)}

def _utc(df):  # keep open_time tz-aware (UTC) — .values/.to_numpy() strips tz and breaks the groupby/map
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True); return df

def added_returns(variant,seed):
    """Return the (open_time, return_pct) rows of the *added* symbols for this variant."""
    if variant=="B0":      return None
    if variant=="real":    return _utc(NEW_RET[["open_time","return_pct"]].copy())
    rng=np.random.RandomState(seed)
    if variant=="shuffle":
        out=[]
        for s,g in NEW_RET.groupby("symbol"):
            v=g["return_pct"].to_numpy().copy(); rng.shuffle(v)
            out.append(pd.DataFrame({"open_time":g["open_time"].to_numpy(),"return_pct":v}))
        return _utc(pd.concat(out,ignore_index=True))
    if variant=="noise":   # synthetic gaussian on the new-grid -> count+vol, zero real info
        return _utc(pd.DataFrame({"open_time":NEW_RET["open_time"].to_numpy(),
                             "return_pct":rng.normal(0.0,NOISE_SIGMA,len(NEW_RET))}))
    if variant=="subset21":
        keep=set(rng.choice(NEW,size=21,replace=False)); g=NEW_RET[NEW_RET.symbol.isin(keep)]
        return _utc(g[["open_time","return_pct"]].copy())
    raise ValueError(variant)

def build_xsz(add):
    allr=BASE_RET if add is None else pd.concat([BASE_RET,add[["open_time","return_pct"]]],ignore_index=True)
    st=allr.groupby("open_time")["return_pct"].agg(["mean","std"])
    m=P175["open_time"].map(st["mean"]); sd=P175["open_time"].map(st["std"]).replace(0,np.nan)
    return ((P175["return_pct"]-m)/sd).clip(-10,10).to_numpy()

def gen(PAN,feats,outp):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if len(tr)==0 or len(te)==0: continue
        tend=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((tend-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    o=pd.concat(rec,ignore_index=True)
    for cc in ("open_time","exit_time"): o[cc]=pd.to_datetime(o[cc],utc=True)
    o.to_parquet(outp); return o

def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else np.nan
def maxdd(s):
    eq=(s.fillna(0)).cumsum(); return float((eq-eq.cummax()).min())

PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")

def run_variant(variant,seed,tag):
    vd=OUT/tag; vd.mkdir(exist_ok=True)
    PAN=P175.copy(); PAN["xs_z"]=build_xsz(added_returns(variant,seed))
    PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
    gen(PAN,V0+RR,vd/"long_full.parquet"); gen(PAN,V0,vd/"short_full.parquet")
    for book,full in [("base","short_full"),("long","long_full")]:
        d=pd.read_parquet(vd/f"{full}.parquet"); keep=[]
        for i in range(len(CUTS)-1):
            w=d[(d.open_time>=CUTS[i])&(d.open_time<CUTS[i+1])]; keep.append(w[~w.symbol.isin(EXCL[i])])
        pd.concat(keep,ignore_index=True).to_parquet(vd/f"{book}.parquet")
    env=dict(os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(vd),
        CONVEXITY_PREDS_PATH=str(vd/"base.parquet"),CONVEXITY_PREDS_LONG=str(vd/"long.parquet"),
        CONVEXITY_UNIVERSE_META=str(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"))
    subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
        stdout=open(vd/"run.log","w"),stderr=subprocess.STDOUT)
    c=pd.read_csv(vd/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
    c=c.sort_values("open_time").set_index("open_time")
    dense=c.loc['2025-01-01':'2026-06-04','pnl_bps']; y25=c.loc['2025-01-01':'2025-12-31','pnl_bps']
    return dict(tag=tag,variant=variant,seed=seed,dense_sharpe=round(dsh(dense),3),
                dense_maxDD=round(maxdd(dense),0),dense_pnl=round(dense.sum(),0),sharpe_2025=round(dsh(y25),3))

PLAN=[("B0",0,"B0"),("real",0,"real")]
PLAN+=[("shuffle",s,f"shuffle_s{s}") for s in range(1,6)]
PLAN+=[("noise",s,f"noise_s{s}") for s in range(1,4)]
PLAN+=[("subset21",s,f"subset21_s{s}") for s in range(1,4)]

rows=[]; done=set()
if RESCSV.exists():
    prev=pd.read_csv(RESCSV); rows=prev.to_dict("records"); done=set(prev.tag)
    print(f"resuming: {len(done)} variants already done: {sorted(done)}",flush=True)
for variant,seed,tag in PLAN:
    if tag in done: continue
    r=run_variant(variant,seed,tag); rows.append(r)
    pd.DataFrame(rows).to_csv(RESCSV,index=False)
    print(f"RESULT {tag:14s} dense_sharpe {r['dense_sharpe']:+.3f}  maxDD {r['dense_maxDD']:+.0f}  pnl {r['dense_pnl']:+.0f}  2025 {r['sharpe_2025']:+.3f}  [{time.time()-t0:.0f}s]",flush=True)

# summary
df=pd.DataFrame(rows)
b0=df[df.variant=="B0"].dense_sharpe.iloc[0]; real=df[df.variant=="real"].dense_sharpe.iloc[0]
print(f"\n=== ISOLATED RENORMALIZATION CHANNEL ===",flush=True)
print(f"  B0 (175, no additions):   {b0:+.3f}",flush=True)
print(f"  real 42 additions:        {real:+.3f}   (channel-1 lift {real-b0:+.3f})",flush=True)
for v in ("shuffle","noise","subset21"):
    sub=df[df.variant==v]
    if len(sub):
        print(f"  placebo {v:9s} n={len(sub)}: mean {sub.dense_sharpe.mean():+.3f}  "
              f"[{sub.dense_sharpe.min():+.3f},{sub.dense_sharpe.max():+.3f}]  lift {sub.dense_sharpe.mean()-b0:+.3f}",flush=True)
# percentile of real vs the pooled placebo distribution
allp=df[df.variant.isin(["shuffle","noise","subset21"])].dense_sharpe.to_numpy()
if len(allp): print(f"  real percentile vs pooled placebo: {(allp<real).mean()*100:.0f}%  (n_placebo={len(allp)})",flush=True)
print("DONE placebo",flush=True)
