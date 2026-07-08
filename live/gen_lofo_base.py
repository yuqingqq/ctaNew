"""Leave-one-feature-out (LOFO) base/short books for per-regime pruning. Universe = the 14 V0_LEAN features.
Drop each feature once, retrain the base book (residual target, same WF pipeline as gen_oos_v4), full OOS window.
Output: hl_lofo_base/drop_<feat>/v0full_hl60.parquet for each feature (+ 'FULL14' = no drop, = hl_v4base_oos).
Downstream (deep_lofo_regime.py) computes each drop's per-regime marginal impact on H1 (decide) and H2 (validate).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0)
V0_LEAN = [f for f in V0 if not f.startswith("funding")]
EMB = pd.Timedelta(days=1); HL = 60.0
START = pd.Timestamp("2023-01-01", tz="UTC"); END = pd.Timestamp("2025-10-01", tz="UTC")
CUTS = list(pd.date_range(START, END, freq="MS", tz="UTC"))
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","alpha_vs_btc_realized"]+V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol","open_time"])
_g = PAN.groupby("open_time"); _sd = _g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
PAN["xs_z"] = ((PAN["alpha_vs_btc_realized"] - _g["alpha_vs_btc_realized"].transform("mean")) / _sd).clip(-10, 10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
def gen(feats, outpath):
    if outpath.exists(): print(f"  skip (exists) {outpath.parent.name}", flush=True); return
    rec = []
    for i in range(len(CUTS)-1):
        c0, c1 = CUTS[i], CUTS[i+1]; fit_cut = c0 - EMB
        tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if not len(tr) or not len(te): continue
        t_end = tr["open_time"].max()
        for sym, g in tr.groupby("symbol"):
            if len(g) < 300: continue
            try:
                s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
                w = np.exp(-((t_end - g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, g["xs_z"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte): rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                    "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "fold": i}))
            except Exception: pass
    out = pd.concat(rec, ignore_index=True); out["open_time"]=pd.to_datetime(out["open_time"],utc=True)
    outpath.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(outpath)
D = REPO/"live/state/convexity/hl_lofo_base"
for fi, drop in enumerate(["__none__"]+V0_LEAN):
    feats = V0_LEAN if drop=="__none__" else [f for f in V0_LEAN if f!=drop]
    tag = "FULL14" if drop=="__none__" else f"drop_{drop}"
    gen(feats, D/tag/"v0full_hl60.parquet")
    print(f"[{fi}/14] {tag} ({len(feats)} feats) done", flush=True)
print("LOFOBASEDONE")
