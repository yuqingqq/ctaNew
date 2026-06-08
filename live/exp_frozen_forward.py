"""Frozen-model forward test: fit the deploy models ONCE at an early cutoff, run them FROZEN forward over the
following months (how a deployed model behaves between retrains). Then measure per-cycle alpha capture (IC) and
its DECAY with model age — the weakness the monthly-refit backtest hides.
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0); RR = ["resid_rev_2","resid_rev_3"]; EMB = pd.Timedelta(days=1); HL=60.0
CUT = pd.Timestamp("2026-02-01", tz="UTC")   # freeze here; forward = CUT -> data end (~4 months)

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
a=PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); PAN["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def gen_frozen(feats):
    tr=PAN[(PAN.exit_time<CUT-EMB)&PAN["xs_z"].notna()]; te=PAN[PAN.open_time>=CUT]
    t_end=tr["open_time"].max(); rec=[]
    for sym,gg in tr.groupby("symbol"):
        if len(gg)<300: continue
        try:
            s,h=x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
            w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
            gte=te[te.symbol==sym]
            if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                "pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"xs_z":gte["xs_z"].values,"ret":gte["return_pct"].values}))
        except Exception: pass
    return pd.concat(rec,ignore_index=True)

t0=time.time(); P=gen_frozen(V0+RR)   # long-ranker (also serves IC analysis); short uses same model fwd-ret rank
P["open_time"]=pd.to_datetime(P["open_time"],utc=True)
print(f"frozen-forward (fit@{CUT.date()}, fwd {P.open_time.min().date()}->{P.open_time.max().date()}, {P.open_time.nunique()} cycles, {time.time()-t0:.0f}s)")
# per-cycle IC (Spearman pred vs forward xs_z) and its decay with model age
def ic(gp):
    return gp["pred"].corr(gp["xs_z"], method="spearman")
per=P.groupby("open_time").apply(ic).rename("ic").reset_index().dropna()
per["age_d"]=(per["open_time"]-CUT).dt.days
print(f"\nPER-CYCLE IC (pred vs forward xs_z), {len(per)} cycles: mean {per.ic.mean():+.4f}  median {per.ic.median():+.4f}  %>0 {(per.ic>0).mean()*100:.0f}%")
print("\nIC DECAY by model age (month since freeze) — is the frozen model degrading?:")
per["month"]=(per["age_d"]//30)
for mo,gm in per.groupby("month"):
    print(f"  age {mo*30:3d}-{mo*30+30:3d}d (n={len(gm):3d}): mean IC {gm.ic.mean():+.4f}  %>0 {(gm.ic>0).mean()*100:.0f}%")
# alpha capture: top-3 vs bottom-3 forward spread per cycle (what the L/S basket would catch), by age
def spread(gp):
    gp=gp.sort_values("pred"); k=3
    if len(gp)<2*k: return np.nan
    return (gp.tail(k)["ret"].mean()-gp.head(k)["ret"].mean())*1e4
sp=P.groupby("open_time").apply(spread).rename("sp").reset_index().dropna(); sp["age_d"]=(sp["open_time"]-CUT).dt.days
print("\nALPHA CAPTURE (top3-bot3 fwd return spread, bps) by model age:")
for mo,gm in sp.groupby(sp["age_d"]//30):
    print(f"  age {mo*30:3d}-{mo*30+30:3d}d (n={len(gm):3d}): mean spread {gm.sp.mean():+6.1f}bp")
print(f"\n  overall capture spread {sp.sp.mean():+.1f}bp.  Decay = frozen-model weakness; if IC/spread drop with age,")
print(f"  the strategy needs more frequent retraining (or features are non-stationary).")
print("DONE frozen_forward")
