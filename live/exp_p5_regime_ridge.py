"""v3 loop iter4 — P5: regime-aware per-symbol Ridge (the one ALPHA lever left).
Low-DOF form: fit ONE Ridge per symbol on regime-INTERACTED features [X, X*1(bull), X*1(bear)] (side=base).
Ridge shrinks the bull/bear deviation coefs toward 0, so a regime only gets its own coefs if the data supports
it (handles the bear ~1/3-data noise via shrinkage, not 3 separate noisy fits). Predict each test bar with its
regime's interaction active. WF (8 monthly folds), V0+RR, xs_z target — identical machinery to baseline gen.
Output: live/state/exp_p5/{long,short}.parquet (then monthly-PIT filter + replay, same as baseline).
"""
import sys, time; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
import live.convexity_paper_bot as bot
x6 = tt.x6; V0 = list(tt.V0); RR = ["resid_rev_2","resid_rev_3"]; EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]

# market-wide regime per open_time (BTC 30d + N=3 hysteresis) — same as the bot
c4 = bot.load_close_4h("BTCUSDT").sort_index(); btc30 = (c4/c4.shift(180)-1)
times = btc30.dropna().index
raw = [bot.regime_for_cycle(x) for x in btc30.dropna().values]
eff = bot.apply_hysteresis(raw, n=3)
REG = pd.Series(eff, index=times)   # open_time -> effective regime

def build_pan():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
    PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
    PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c]=PAN[c].fillna(0.0)
    g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
    PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
    PAN["reg"] = PAN["open_time"].map(REG)
    return PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def interact(X, reg):   # [X, X*1(bull), X*1(bear)] — side is the base
    bull = (reg=="bull").to_numpy().astype(np.float32)[:,None]
    bear = (reg=="bear").to_numpy().astype(np.float32)[:,None]
    return np.hstack([X, X*bull, X*bear])

def gen(PAN, feats, outpath):
    rec=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te=PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end=tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h=x6.fit_preproc(gg,feats); Xb=x6.apply_preproc(gg,feats,s,h)
                X=interact(Xb, gg["reg"]); w=np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m=RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_z"].to_numpy(), sample_weight=w)
                gte=te[te.symbol==sym]
                if len(gte):
                    Xt=interact(x6.apply_preproc(gte,feats,s,h), gte["reg"])
                    rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values,"pred":m.predict(Xt),"fold":i}))
            except Exception: pass
    out=pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    out.to_parquet(outpath); return out["symbol"].nunique(), len(out)

t0=time.time(); od=REPO/"live/state/exp_p5"; od.mkdir(parents=True,exist_ok=True)
PAN=build_pan()
print(f"regime dist in panel: {PAN['reg'].value_counts().to_dict()}", flush=True)
nl=gen(PAN, V0+RR, od/"long_full.parquet"); ns=gen(PAN, V0, od/"short_full.parquet")
import json
EXCL=set(json.load(open(REPO/"live/models/convexity_v1_universe.json"))["exclude_high_vol"])
for src,dst in [("long_full.parquet","long.parquet"),("short_full.parquet","base.parquet")]:
    d=pd.read_parquet(od/src); d[~d["symbol"].isin(EXCL)].to_parquet(od/dst)
print(f"P5 regime-Ridge: long {nl} short {ns} | {time.time()-t0:.0f}s -> {od}")
print("DONE p5gen")
