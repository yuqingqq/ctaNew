"""FULL-HISTORY backtest — the biggest untested free-data direction. The production backtest covers only 243d
(Oct2025-Jun2026, net-bearish); the panel has 2021-2026. Regenerate base/long WF preds with monthly CUTS spanning
2022->2026 (same RidgeCV/HL=60/embargo machinery as exp_xs94_genpreds), then replay the v2 stack and report HONEST
daily Sharpe per YEAR + overall. Tests whether the edge holds across bull (2023/24) + bear (2022/25/26) regimes.

CAVEAT (honest): the exclude_high_vol universe filter is computed on the full sample (mild look-ahead over 5y); this
is a ROBUSTNESS screen, not a deployable number. If the edge is present every year -> strong validation; if it only
appears in the bearish window -> the +3.3 is regime-luck.
"""
import sys, json, time, os, subprocess; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
x6=tt.x6; V0=list(tt.V0); RR=["resid_rev_2","resid_rev_3"]; EMB=pd.Timedelta(days=1); HL=60.0
EXCL=set(json.load(open(REPO/"live/models/convexity_v1_universe.json"))["exclude_high_vol"])
# monthly cuts 2022-01 -> 2026-06 (2021 = warmup/train only)
CUTS=[pd.Timestamp(t,tz="UTC") for t in pd.date_range("2022-01-01","2026-06-01",freq="MS")]
_last=pd.read_parquet(tt.PANEL,columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS=CUTS+[_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
OUT=REPO/"live/state/v3loop/fullhist"; OUT.mkdir(parents=True,exist_ok=True)

def build_pan():
    PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
    PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
    PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
    a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
    for c in RR: PAN[c]=PAN[c].fillna(0.0)
    g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
    PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
    return PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def gen(PAN,feats,outpath):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        if len(tr)==0 or len(te)==0: continue
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    out.to_parquet(outpath); return out

t0=time.time(); PAN=build_pan(); print(f"panel built {time.time()-t0:.0f}s, {PAN.open_time.min().date()}->{PAN.open_time.max().date()}",flush=True)
if not (OUT/"long.parquet").exists():
    L=gen(PAN,V0+RR,OUT/"long_full.parquet"); S=gen(PAN,V0,OUT/"short_full.parquet")
    pd.read_parquet(OUT/"long_full.parquet").query("symbol not in @EXCL").to_parquet(OUT/"long.parquet")
    pd.read_parquet(OUT/"short_full.parquet").query("symbol not in @EXCL").to_parquet(OUT/"base.parquet")
    print(f"preds generated {time.time()-t0:.0f}s: long {L.open_time.min().date()}->{L.open_time.max().date()} {L.symbol.nunique()}syms {len(L)}rows",flush=True)

# replay through v2 stack
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
env=dict(os.environ); env.update(PROD); env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(OUT),
    CONVEXITY_PREDS_PATH=str(OUT/"base.parquet"),CONVEXITY_PREDS_LONG=str(OUT/"long.parquet")); env.pop("CONVEXITY_UNIVERSE_META",None)
r=subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                 stdout=open(OUT/"run.log","w"),stderr=subprocess.STDOUT)
print(f"replay rc {r.returncode} ({time.time()-t0:.0f}s)",flush=True)
c=pd.read_csv(OUT/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time").set_index("open_time")
ANN=np.sqrt(365)
def dsh(s): d=(s.fillna(0)/1e4).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else np.nan
print("\n=== FULL-HISTORY honest daily Sharpe (per year) ===",flush=True)
print(f"  OVERALL {c.index.min().date()}->{c.index.max().date()}: dSharpe {dsh(c['pnl_bps']):+.2f}  totPnL {c['pnl_bps'].sum():+.0f}  cycles {len(c)}",flush=True)
for yr,g in c.groupby(c.index.year):
    print(f"  {yr}: dSharpe {dsh(g['pnl_bps']):+.2f}  totPnL {g['pnl_bps'].sum():+7.0f}  cycles {len(g)}  regime {g['regime'].value_counts().to_dict()}",flush=True)
print("DONE fullhist",flush=True)
