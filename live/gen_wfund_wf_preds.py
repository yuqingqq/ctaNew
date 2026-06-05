"""WITH-funding variant of gen_lean_wf_preds: flow book = V0(full, incl funding) + VPIN/TFI; price = V0(full).
Panel-derived (175-covering) so the funding re-test runs on the HONEST universe (not the x132-capped 160).
Pairs with hl_lean175 (no-funding) for an apples-to-apples funding ablation on corrected maturity.
-> live/state/convexity/hl_wfund175/{fullflow_hl60,v0full_hl60}.parquet
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = tt.V0
V0_USE = list(V0)                          # WITH funding
FLOW_KEEP = ["fl_vpin", "fl_vpin_1d", "fl_tfi", "fl_tfi_1d"]
EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
OUT = REPO/"live/state/convexity/hl_wfund175"; OUT.mkdir(parents=True, exist_ok=True)

F, flowcols = tt.build_flow(); FLOWSYMS = set(F.symbol.unique())
fk = [c for c in FLOW_KEEP if c in flowcols]
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"] = pd.to_datetime(_last["open_time"], utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize() + pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F, on=["symbol","open_time"], how="left")
_g = PAN.groupby("open_time"); _mu = _g["return_pct"].transform("mean"); _sd = _g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-_mu)/_sd).clip(-10,10); PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)


def gen(use_flow, outpath):
    rec = []
    for i in range(len(CUTS)-1):
        c0, c1 = CUTS[i], CUTS[i+1]; fit_cut = c0-EMB
        tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym, g in tr.groupby("symbol"):
            if len(g) < 300: continue
            uf = use_flow and (sym in FLOWSYMS) and g[fk].notna().any().all()
            feats = V0_USE + fk if uf else V0_USE
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
print(f"WITH-funding 175 preds: flow {nfs} syms/{nfr} rows, price {nps} syms/{npr} rows -> {OUT}")
