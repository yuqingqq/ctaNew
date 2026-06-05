"""Hinge-augmented preds: add asymmetric basis-expansion features so the LINEAR model can learn asymmetric
reversion (short gainers, don't long fallers) — the nonlinearity it structurally can't fit with one slope.
Adds ret_3d_pos/neg, return_1d_pos/neg (+ |ret_3d| magnitude). Flow book = V0+funding+VPIN/TFI+hinges; price = V0+funding+hinges.
-> live/state/convexity/hl_hinge175/{fullflow_hl60,v0full_hl60}.parquet
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = tt.V0
FLOW_KEEP = ["fl_vpin", "fl_vpin_1d", "fl_tfi", "fl_tfi_1d"]
HINGE = ["ret_3d_pos", "ret_3d_neg", "return_1d_pos", "return_1d_neg", "ret_3d_absmag"]
EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
OUT = REPO/"live/state/convexity/hl_hinge175"; OUT.mkdir(parents=True, exist_ok=True)

F, flowcols = tt.build_flow(); FLOWSYMS = set(F.symbol.unique())
fk = [c for c in FLOW_KEEP if c in flowcols]
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"] = pd.to_datetime(_last["open_time"], utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize() + pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F, on=["symbol","open_time"], how="left")
# hinge / basis-expansion features (asymmetric reversion)
PAN["ret_3d_pos"]   = PAN["ret_3d"].clip(lower=0)
PAN["ret_3d_neg"]   = PAN["ret_3d"].clip(upper=0)
PAN["return_1d_pos"]= PAN["return_1d"].clip(lower=0)
PAN["return_1d_neg"]= PAN["return_1d"].clip(upper=0)
PAN["ret_3d_absmag"]= PAN["ret_3d"].abs()
_g = PAN.groupby("open_time"); _mu = _g["return_pct"].transform("mean"); _sd = _g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-_mu)/_sd).clip(-10,10); PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
V0H = list(V0) + HINGE   # V0 (with funding) + hinge features


def gen(use_flow, outpath):
    rec = []
    for i in range(len(CUTS)-1):
        c0, c1 = CUTS[i], CUTS[i+1]; fit_cut = c0-EMB
        tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym, g in tr.groupby("symbol"):
            if len(g) < 300: continue
            uf = use_flow and (sym in FLOWSYMS) and g[fk].notna().any().all()
            feats = V0H + fk if uf else V0H
            try:
                s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
                w = np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, g["xs_z"].to_numpy(), sample_weight=w)
                gte = te[te.symbol==sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol":sym, "open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values, "return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values, "pred":m.predict(x6.apply_preproc(gte,feats,s,h)), "fold":i}))
            except Exception: pass
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time","exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    out.to_parquet(outpath); return out["symbol"].nunique(), len(out)


nfs, nfr = gen(True, OUT/"fullflow_hl60.parquet")
nps, npr = gen(False, OUT/"v0full_hl60.parquet")
print(f"hinge-augmented 175 preds: flow {nfs} syms/{nfr} rows, price {nps} syms/{npr} rows -> {OUT}")
