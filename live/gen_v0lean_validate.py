"""Re-validate v1 on the DEPLOYED feature set (V0_LEAN, funding dropped) — the backtest used full-V0 but the
frozen deploy models use V0_LEAN (corr 0.47 mismatch). Walk-forward (no look-ahead), per-symbol RidgeCV recency-60,
SAME CUTS/machinery as gen_residrev. Produces:
  hl_lean/         : V0_LEAN base preds (shorts ranker + baseline book B)
  hl_residrev_lean/: V0_LEAN + resid_rev preds (longs ranker)
Both write v0full_hl60 AND fullflow_hl60 (copy; book A ignored in book-B-only) so the harness runs.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0_LEAN = tt.V0_LEAN; RR = tt.RR; EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+list(tt.V0))
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
_a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"] = -_a.transform(lambda s: s.shift(1).rolling(2).sum())
PAN["resid_rev_3"] = -_a.transform(lambda s: s.shift(1).rolling(3).sum())
for c in RR: PAN[c] = PAN[c].fillna(0.0)
g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
CUTS = CUTS + [PAN["open_time"].max().normalize()+pd.Timedelta(days=1)]

def gen(feats, outdir):
    outdir.mkdir(parents=True, exist_ok=True); rec=[]
    for i in range(len(CUTS)-1):
        c0,c1 = CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr = PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg)<300: continue
            try:
                s,h = x6.fit_preproc(gg,feats); X=x6.apply_preproc(gg,feats,s,h)
                w = np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X,gg["xs_z"].to_numpy(),sample_weight=w)
                gte = te[te.symbol==sym]
                if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                    "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                    "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out = pd.concat(rec,ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    out.to_parquet(outdir/"v0full_hl60.parquet"); out.to_parquet(outdir/"fullflow_hl60.parquet")  # book A copy (ignored)
    return out["symbol"].nunique(), len(out)

a1=gen(V0_LEAN, REPO/"live/state/convexity/hl_lean")
a2=gen(V0_LEAN+RR, REPO/"live/state/convexity/hl_residrev_lean")
print(f"V0_LEAN base: {a1}; V0_LEAN+resid_rev: {a2}")
