"""POOLED cross-sectional linear walk-forward preds (#167, the real IC lever: pooling +0.022 IC vs per-symbol).
ONE RidgeCV across all symbols per fold on xs_z, standardized features, recency-weighted. Production 9-fold cuts.
Output for ALL syms -> used for both books in the two-book backtest (pred is cross-sectionally comparable).
-> live/state/convexity/hl_pooled175/{fullflow_hl60,v0full_hl60}.parquet  (identical preds in both files)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0 = tt.V0; EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
OUT = REPO/"live/state/convexity/hl_pooled175"; OUT.mkdir(parents=True, exist_ok=True)

F, flowcols = tt.build_flow()
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F,on=["symbol","open_time"],how="left")
FEATS = V0 + flowcols
g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

rec = []
for i in range(len(CUTS)-1):
    c0, c1 = CUTS[i], CUTS[i+1]; fc = c0-EMB
    tr = PAN[(PAN.exit_time < fc) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
    if len(tr) < 5000 or len(te) == 0: continue
    Xtr = tr[FEATS].fillna(0).to_numpy(); mu, sg = Xtr.mean(0), Xtr.std(0); sg[sg==0]=1
    w = np.exp(-((tr["open_time"].max()-tr["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
    m = RidgeCV(alphas=(0.1,1,10,100)).fit((Xtr-mu)/sg, tr["xs_z"].to_numpy(), sample_weight=w)
    pr = m.predict((te[FEATS].fillna(0).to_numpy()-mu)/sg)
    rec.append(pd.DataFrame({"symbol":te["symbol"].values, "open_time":te["open_time"].values,
        "alpha_A":te["alpha_vs_btc_realized"].values, "return_pct":te["return_pct"].values,
        "exit_time":te["exit_time"].values, "pred":pr, "fold":i}))
out = pd.concat(rec, ignore_index=True)
for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
out.to_parquet(OUT/"fullflow_hl60.parquet"); out.to_parquet(OUT/"v0full_hl60.parquet")
print(f"pooled-linear preds: {out['symbol'].nunique()} syms, {len(out)} rows -> {OUT}")
