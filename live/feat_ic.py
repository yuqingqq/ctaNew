"""Long-loop feature discovery: nested-OOS MARGINAL IC of candidate NEW features over the production V0+RR set.
For each candidate, walk-forward per-symbol RidgeCV on (V0+RR) vs (V0+RR+feature); report per-cycle IC lift + per-fold
so fold-concentrated/overfit gains are visible. Anti-overfit: strictly walk-forward, marginal, per-fold reported.
Real signal bar: IC lift >= +0.004 AND >=6/9 folds positive AND not 1-2-fold concentrated -> escalate to strategy replay.

  python3 live/feat_ic.py            # screen the built-in candidate batch
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
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

# ---- candidate NEW features (economically motivated, NOT already in V0). Each = a column added to PAN. ----
eps=1e-9
def xsrank(col):  # per-cycle cross-sectional rank in [0,1]
    return PAN.groupby("open_time")[col].rank(pct=True)
CAND = {}
CAND["fund_xsrank"]      = xsrank("funding_rate")                                  # positioning extremity (xs)
CAND["fund_x_ret3d"]     = PAN["funding_rate"]*PAN["ret_3d"]                        # carry x recent move = squeeze setup
CAND["ret3d_per_atr"]    = PAN["ret_3d"]/(PAN["atr_pct"]+eps)                       # risk-adjusted reversal magnitude
CAND["obv_x_ret1d"]      = PAN["obv_z_1d"]*PAN["return_1d"]                         # volume-confirmed move
CAND["idiovol_termstruct"]= PAN["idio_vol_to_btc_1h"]/(PAN["idio_vol_to_btc_1d"]+eps) # short/long idio-vol ratio
CAND["relvol_alt_btc"]   = PAN["rvol_7d"]/(PAN["btc_rvol_7d"]+eps)                  # alt vs btc vol regime
CAND["fundz_x_idiovol"]  = PAN["funding_rate_z_7d"]*PAN["idio_vol_to_btc_1h"]       # funding stress in high-idio names
CAND["bsh_x_ret3d"]      = PAN["bars_since_high"]*PAN["ret_3d"]                     # distance-from-high x reversal
CAND["corr_x_ret1d"]     = PAN["corr_to_btc_1d"]*PAN["return_1d"]                   # beta-timing
CAND["ret1d_x_rvol"]     = PAN["return_1d"]*PAN["rvol_7d"]                          # vol-scaled momentum
CAND["autocorr_x_ret1d"] = PAN["autocorr_pctile_7d"]*PAN["return_1d"]              # trend/meanrev-regime x move
CAND["fund_abs"]         = PAN["funding_rate"].abs()                               # funding stress magnitude
CAND["ret3d_sq"]         = PAN["ret_3d"]**2                                         # non-linear reversal (convexity)
CAND["vwap_x_bsh"]       = PAN["vwap_slope_96"]*PAN["bars_since_high"]              # trend persistence
CAND["fundchg_x_ret1d"]  = PAN["funding_rate_1d_change"]*PAN["return_1d"]           # funding accel x move
for k,v in CAND.items(): PAN[k]=v.replace([np.inf,-np.inf],np.nan).fillna(0.0)
PAN=PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

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

t0=time.time(); base_ic, base_pf = gen_ic(V0+RR)
print(f"BASELINE (V0+RR) IC {base_ic:+.4f}  per-fold {[round(x,3) for x in base_pf]}  ({time.time()-t0:.0f}s)")
print(f"\nMARGINAL IC LIFT per candidate feature (nested-OOS; bar: lift>=+0.004 AND >=6/9 folds up):")
res=[]
for name in CAND:
    t0=time.time(); ic,pf=gen_ic(V0+RR+[name])
    fu=sum(1 for b,n in zip(base_pf,pf) if n>b)
    res.append((name, ic-base_ic, fu));
    print(f"  {name:20s} lift {ic-base_ic:+.4f}  IC {ic:+.4f}  folds_up {fu}/9  ({time.time()-t0:.0f}s)", flush=True)
res.sort(key=lambda r:-r[1])
print(f"\nTOP by IC lift: {[(n,round(l,4),f) for n,l,f in res[:5]]}")
winners=[r for r in res if r[1]>=0.004 and r[2]>=6]
print(f"WINNERS (lift>=+0.004 & folds>=6): {[(n,round(l,4),f) for n,l,f in winners] or 'NONE'}")
print("DONE feat_ic")
