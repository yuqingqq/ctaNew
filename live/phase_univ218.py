"""Option A: +43-symbol (218 = 175 panel + 43 already-cached) universe test on 2025-2026.
Extends the panel with the 43 extra (xs_feats cached), recomputes XS-rank on the union, regenerates WF preds
(same RidgeCV/HL60/embargo as fullhist), re-curates the full-universe low-vol cohort (mpit rule), replays, and runs
the IDIO (low-corr) tilt. Compares to the 175-universe dense-window baseline (+1.33). Reuses X70/x6 helpers verbatim.
"""
import sys, json, time, importlib.util; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); RR=["resid_rev_2","resid_rev_3"]; EMB=pd.Timedelta(days=1); HL=60.0; ANN=np.sqrt(365)
spec=importlib.util.spec_from_file_location("x70m",REPO/"research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
X70=importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
CACHE=REPO/"data/ml/cache"; OUT=REPO/"live/state/v3loop/univ218"; OUT.mkdir(parents=True,exist_ok=True)
t0=time.time()
# --- 1. existing 175 panel + the 43 extra (klines+xs_feats cached, not in panel) ---
P175=pd.read_parquet(tt.PANEL); P175["open_time"]=pd.to_datetime(P175["open_time"],utc=True)
import os
have=set(s for s in os.listdir(REPO/"data/ml/test/parquet/klines") if s.endswith("USDT"))
extra=sorted(have-set(P175.symbol.unique()))
extra=[s for s in extra if (CACHE/f"xs_feats_{s}.parquet").exists() and (CACHE/f"funding_{s}.parquet").exists()]
print(f"extending panel: 175 + {len(extra)} extra -> {175+len(extra)} syms",flush=True)
btc=X70.load_closes("BTCUSDT")
def s4(df): ot=pd.to_datetime(df["open_time"],utc=True); return df[(ot.dt.hour%4==0)&(ot.dt.minute==0)]
new=[]
for i,s in enumerate(extra,1):
    try:
        d=X70.build_sym(s,btc)
        if d is None or len(d)==0: continue
        d["open_time"]=pd.to_datetime(d["open_time"],utc=True); d["exit_time"]=pd.to_datetime(d["exit_time"],utc=True)
        d=d.dropna(subset=["alpha_vs_btc_realized"]); d=x6.build_target_z(d); d=s4(d)
        new.append(d)
    except Exception as e: print(f"  {s} ERR {e}",flush=True)
newp=pd.concat(new,ignore_index=True)
# add cohort features (rvol_7d, ret_3d, btc_rvol_7d) — loads 5m closes for the new syms + BTC, merges by (sym,open_time)
newp=X70.x6b.build_cohort_fixed(newp)
print(f"  cohort features added; newp cols now {len(newp.columns)}",flush=True)
# align columns to P175
common=[c for c in P175.columns if c in newp.columns]
_missing=[c for c in P175.columns if c not in newp.columns]
if _missing: print(f"  WARN newp missing P175 cols: {_missing}",flush=True)
panel=pd.concat([P175[common],newp[common]],ignore_index=True).sort_values(["open_time","symbol"])
# recompute the cross-sectional rank feature on the UNION (it's the only XS feature in V0)
if "bars_since_high_xs_rank" in panel.columns and "bars_since_high" in panel.columns:
    panel["bars_since_high_xs_rank"]=panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
panel.to_parquet(OUT/"panel218.parquet")
print(f"panel218 built: {panel.symbol.nunique()} syms, {len(panel)} rows [{time.time()-t0:.0f}s]",flush=True)
# --- 2. regenerate preds (same machinery as fullhist) ---
PAN=panel.copy()
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
CUTS=[pd.Timestamp(t,tz="UTC") for t in pd.date_range("2022-01-01","2026-06-01",freq="MS")]+[pd.Timestamp("2026-06-05",tz="UTC")]
def gen(feats,outp):
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
L=gen(V0+RR,OUT/"long_full.parquet"); S=gen(V0,OUT/"short_full.parquet")
print(f"preds generated: {S.symbol.nunique()} syms [{time.time()-t0:.0f}s]",flush=True)
# --- 3. mpit universe curation (exclude top-52% high-vol by trailing-30d rvol_7d) ---
panv=pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","rvol_7d"])
# rvol for the 43 extra from their xs_feats
rv_extra=[]
for s in extra:
    try:
        xf=pd.read_parquet(CACHE/f"xs_feats_{s}.parquet")
        if "rvol_7d" in xf.columns: rv_extra.append(pd.DataFrame({"symbol":s,"open_time":pd.to_datetime(xf.index if xf.index.name else xf["open_time"],utc=True),"rvol_7d":xf["rvol_7d"].values}))
    except Exception: pass
panv=pd.concat([panv]+rv_extra,ignore_index=True) if rv_extra else panv
panv["open_time"]=pd.to_datetime(panv["open_time"],utc=True)
FRAC=0.52
def excl_for(c0):
    lo=c0-pd.Timedelta(days=30); r=panv[(panv.open_time>=lo)&(panv.open_time<c0)].groupby("symbol")["rvol_7d"].mean().dropna()
    return set(r.sort_values(ascending=False).index[:int(round(FRAC*len(r)))])
for book,full in [("base","short_full"),("long","long_full")]:
    d=pd.read_parquet(OUT/f"{full}.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); keep=[]
    for i in range(len(CUTS)-1):
        ex=excl_for(CUTS[i]); w=d[(d.open_time>=CUTS[i])&(d.open_time<CUTS[i+1])]; keep.append(w[~w.symbol.isin(ex)])
    pd.concat(keep,ignore_index=True).to_parquet(OUT/f"{book}.parquet")
print(f"mpit-curated [{time.time()-t0:.0f}s]",flush=True)
# --- 4. replay ---
import os as _os, subprocess
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
env=dict(_os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(OUT),
    CONVEXITY_PREDS_PATH=str(OUT/"base.parquet"),CONVEXITY_PREDS_LONG=str(OUT/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),stdout=open(OUT/"run.log","w"),stderr=subprocess.STDOUT)
c=pd.read_csv(OUT/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else np.nan
print(f"\n=== 218-UNIVERSE RESULT (replay rc {r.returncode}) ===")
print(f"  dense 2025-01..2026-06: dSharpe {dsh(c.loc['2025-01-01':'2026-06-04','pnl_bps']):+.3f}  (175-univ baseline +1.33)")
for yr,gg in c.groupby(c.index.year): print(f"  {yr}: {dsh(gg['pnl_bps']):+.2f}  avgUniv {gg['n_predicted'].mean():.0f}")
print("DONE univ218",flush=True)
