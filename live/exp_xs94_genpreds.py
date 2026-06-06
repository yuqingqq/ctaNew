"""Experiment: bars_since_high_xs_rank ranked over the 94 traded low-vol cohort vs the full 175 panel.
Regenerates BOTH arms' WF preds with identical code (baseline-repro validates the replication), then the v2
replay runs separately per arm. Treatment recomputes ONLY bars_since_high_xs_rank over the low-vol cohort
(exclude_high_vol complement, the SAME set used for selection — apples-to-apples). 18 per-symbol feats unchanged.
"""
import sys, json, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0); RR = ["resid_rev_2","resid_rev_3"]; EMB = pd.Timedelta(days=1); HL = 60.0
EXCL = set(json.load(open(REPO/"live/models/convexity_v1_universe.json"))["exclude_high_vol"])
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]

def build_pan(treatment):
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
    PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
    PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
    if treatment:
        # recompute bars_since_high_xs_rank over the low-vol cohort ONLY (per open_time, pct-rank among non-excluded)
        PAN["bars_since_high_xs_rank"] = PAN["bars_since_high_xs_rank"].astype("float64")  # allow float64 ranks
        lv = ~PAN["symbol"].isin(EXCL)
        xr94 = PAN[lv].groupby("open_time")["bars_since_high"].rank(pct=True)   # index-aligned
        PAN.loc[lv, "bars_since_high_xs_rank"] = xr94
        # excluded names keep their original value (never traded → irrelevant)
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
    PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
    return PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def gen(PAN, feats, outpath):
    rec = []
    for i in range(len(CUTS)-1):
        c0,c1 = CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr = PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            try:
                s,h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_z"].to_numpy(), sample_weight=w)
                gte = te[te.symbol==sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    out.to_parquet(outpath); return out["symbol"].nunique(), len(out)

for arm in ["baseline","treatment"]:
    t0=time.time(); od = REPO/f"live/state/exp_xs94/{arm}"; od.mkdir(parents=True, exist_ok=True)
    if (od/"long.parquet").exists() and (od/"base.parquet").exists():
        print(f"[{arm}] already done, skipping", flush=True); continue
    PAN = build_pan(arm=="treatment")
    nl = gen(PAN, V0+RR, od/"long_full.parquet")     # long ranker (resid_rev)
    ns = gen(PAN, V0,    od/"short_full.parquet")     # short ranker (base)
    # filter to low-vol book (exclude high-vol), like run_convexity_v1.sh
    for src,dst in [("long_full.parquet","long.parquet"),("short_full.parquet","base.parquet")]:
        d = pd.read_parquet(od/src); d[~d["symbol"].isin(EXCL)].to_parquet(od/dst)
    print(f"[{arm}] long {nl} short {ns} | {time.time()-t0:.0f}s -> {od}", flush=True)
print("DONE genpreds")
