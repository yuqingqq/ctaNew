"""Add BTC-RESIDUAL short-horizon REVERSAL features to V0 and regenerate per-symbol two-book preds (iter11).
Root cause (diag_along_signals): model's reversal feats (return_1d/ret_3d) are RAW returns = mostly BTC beta for
high-vol names; it MISSES the BTC-residual short-horizon reversal (orthIC -0.04..-0.05, ~100% orthogonal to pred).
Features (PIT — trailing sum of PAST realized residual alpha, strictly before t):
  resid_rev_2 : -(sum of prev 8h BTC-residual alpha)   resid_rev_3 : -(sum of prev 12h)
  (negated so HIGHER = more washed-out = more bounce-prone, aligning sign with a long signal)
-> live/state/convexity/hl_residrev/{fullflow_hl60,v0full_hl60}.parquet
"""
import os, sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0); EMB = pd.Timedelta(days=1); HL = float(os.environ.get("RESIDREV_HL","60"))
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
OUT = REPO/("live/state/convexity/hl_residrev_HL"+os.environ.get("RESIDREV_HL","60")); OUT.mkdir(parents=True, exist_ok=True)
RR = ["resid_rev_2","resid_rev_3"]

F, flowcols = tt.build_flow(); FLOWSYMS = set(F.symbol.unique())
fk = [c for c in ["fl_vpin","fl_vpin_1d","fl_tfi","fl_tfi_1d"] if c in flowcols]
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
# PIT residual-reversal features: trailing sum of PAST per-bar realized residual alpha (shift(1) => strictly before t)
a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())   # 8h
PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())   # 12h
for c in RR: PAN[c] = PAN[c].fillna(0.0)
PAN = PAN.merge(F, on=["symbol","open_time"], how="left").reset_index(drop=True)
g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"resid-rev feats added; rows={len(PAN)}")

def gen(use_flow, outpath):
    rec = []
    for i in range(len(CUTS)-1):
        c0,c1 = CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr = PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            uf = use_flow and (sym in FLOWSYMS) and gg[fk].notna().any().all()
            feats = V0 + RR + (fk if uf else [])
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

a_=gen(True, OUT/"fullflow_hl60.parquet"); b_=gen(False, OUT/"v0full_hl60.parquet")
print(f"V0+resid_rev preds: flow {a_}, price {b_} -> {OUT}")
