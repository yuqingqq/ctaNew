"""Base-book (V0_LEAN) walk-forward pred generator with an OPTIONAL decouple_z feature, for the
feature-test A/B. Same RidgeCV per-symbol pipeline as gen_lean_wf_preds. Merges precomputed decouple_z.
Usage: python3 live/gen_dz_base.py <START> <END> <USE_DZ 0|1> <outdir>
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

START = pd.Timestamp(sys.argv[1], tz="UTC"); END = pd.Timestamp(sys.argv[2], tz="UTC")
USE_DZ = sys.argv[3] == "1"; OUT = REPO / sys.argv[4]; OUT.mkdir(parents=True, exist_ok=True)
CUTS = list(pd.date_range(START, END, freq="MS", tz="UTC"))
FEATS = V0_LEAN + (["decouple_z"] if USE_DZ else [])
print(f"CUTS {CUTS[0].date()}..{CUTS[-1].date()} ({len(CUTS)-1} folds), USE_DZ={USE_DZ}, nfeats={len(FEATS)}", flush=True)

PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"] + V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
if USE_DZ:
    dz = pd.read_parquet(REPO / "live/state/convexity/decouple_z.parquet")
    dz["open_time"] = pd.to_datetime(dz["open_time"], utc=True)
    PAN = PAN.merge(dz, on=["symbol", "open_time"], how="left")
    PAN["decouple_z"] = PAN["decouple_z"].fillna(0.0)
_g = PAN.groupby("open_time"); _sd = _g["return_pct"].transform("std").replace(0, np.nan)
PAN["xs_z"] = ((PAN["return_pct"] - _g["return_pct"].transform("mean")) / _sd).clip(-10, 10)
PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

rec = []
for i in range(len(CUTS) - 1):
    c0, c1 = CUTS[i], CUTS[i + 1]; fit_cut = c0 - EMB
    tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
    if not len(tr) or not len(te): continue
    t_end = tr["open_time"].max()
    for sym, g in tr.groupby("symbol"):
        if len(g) < 300: continue
        try:
            s, h = x6.fit_preproc(g, FEATS); X = x6.apply_preproc(g, FEATS, s, h)
            w = np.exp(-((t_end - g["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, g["xs_z"].to_numpy(), sample_weight=w)
            gte = te[te.symbol == sym]
            if len(gte):
                rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                    "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                    "exit_time": gte["exit_time"].values, "pred": m.predict(x6.apply_preproc(gte, FEATS, s, h)), "fold": i}))
        except Exception: pass
out = pd.concat(rec, ignore_index=True)
for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
out.to_parquet(OUT / "v0full_hl60.parquet")
print(f"DONE {out['symbol'].nunique()} syms {len(out)} rows -> {OUT}", flush=True)
